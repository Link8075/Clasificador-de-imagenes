import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os
from data_loader import get_data_loaders
from model import Cifar10CNN

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda'):
    """
    Función principal de entrenamiento con monitorización.
    """
    print(f"Iniciando entrenamiento en: {device}")
    
    # Para guardar el historial y graficar después (Requisito: Gráficas de curvas)
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- FASE DE ENTRENAMIENTO ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 1. Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 2. Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Estadísticas
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train
        
        # --- FASE DE VALIDACIÓN ---
        # (Requisito: Uso de validation set para ajustes)
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad(): # No calculamos gradientes en validación (ahorra VRAM)
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / total_val
        val_acc = correct_val / total_val
        
        end_time = time.time()
        
        # Guardar historial
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Tiempo: {end_time - start_time:.1f}s")
        
        # --- GUARDAR EL MEJOR MODELO (Checkpointing) ---
        if val_acc > best_acc:
            best_acc = val_acc
            # Crear carpeta models si no existe
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), 'models/best_model_cifar10.pth')
            print(f"  -> Nuevo mejor modelo guardado! (Acc: {val_acc:.4f})")

    # Guardar historial en un json para graficar luego
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f)
        
    return history

if __name__ == "__main__":
    # Configuración
    BATCH_SIZE = 128 # Aumentado a 128 para aprovechar tu 4060 y 32GB RAM
    EPOCHS = 30      # Suficiente para ver convergencia
    LEARNING_RATE = 0.001
    
    # Detectar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar datos
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)
    
    # Inicializar modelo
    model = Cifar10CNN(num_classes=10).to(device)
    
    # Definir Loss y Optimizador (Requisito: Selección y justificación del optimizador)
    # CrossEntropyLoss es estándar para clasificación multiclase.
    criterion = nn.CrossEntropyLoss()
    # Adam suele converger más rápido que SGD puro.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Entrenar
    print("Comenzando entrenamiento...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, device=device)
    print("Entrenamiento finalizado.")