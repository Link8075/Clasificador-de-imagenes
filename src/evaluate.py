import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import get_data_loaders
from model import Cifar10CNN

# Clases de CIFAR-10
CLASSES = ['avión', 'auto', 'pájaro', 'gato', 'ciervo', 
           'perro', 'rana', 'caballo', 'barco', 'camión']

def plot_training_curves(history_path='models/training_history.json', save_dir='reporte'):
    """Genera las gráficas de Loss y Accuracy requeridas."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Gráfica de Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Entrenamiento')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validación')
    plt.title('Precisión (Accuracy) durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Gráfica de Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Entrenamiento')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validación')
    plt.title('Pérdida (Loss) durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/curvas_entrenamiento.png')
    print("Gráficas guardadas en reporte/curvas_entrenamiento.png")

def evaluate_model(model, test_loader, device):
    """Evalúa el modelo en el Test Set y genera matriz de confusión."""
    model.eval()
    all_preds = []
    all_labels = []
    
    # Recolectar todas las predicciones
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Reporte de clasificación (Precision, Recall, F1)
    print("\nReporte de Clasificación:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))
    
    # Matriz de Confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig('reporte/matriz_confusion.png')
    print("Matriz guardada en reporte/matriz_confusion.png")
    
    return all_preds, all_labels

def show_errors(model, test_loader, device, num_errors=5):
    """Muestra ejemplos mal clasificados para el análisis de errores."""
    model.eval()
    errors_found = 0
    plt.figure(figsize=(15, 3))
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_gpu = images.to(device)
            outputs = model(images_gpu)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    # Des-normalizar imagen para mostrarla bien
                    img = images[i].cpu() * std + mean
                    img = np.clip(img.permute(1, 2, 0).numpy(), 0, 1)
                    
                    ax = plt.subplot(1, num_errors, errors_found + 1)
                    ax.imshow(img)
                    ax.set_title(f"Real: {CLASSES[labels[i]]}\nPred: {CLASSES[preds[i]]}", color='red')
                    ax.axis('off')
                    
                    errors_found += 1
                    if errors_found >= num_errors:
                        plt.tight_layout()
                        plt.savefig('reporte/errores_analisis.png')
                        print("Errores guardados en reporte/errores_analisis.png")
                        return

if __name__ == "__main__":
    # Crear carpeta de reporte
    if not os.path.exists('reporte'):
        os.makedirs('reporte')

    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_data_loaders(batch_size=128)
    
    # Cargar el MEJOR modelo guardado
    model = Cifar10CNN(num_classes=10).to(device)
    model.load_state_dict(torch.load('models/best_model_cifar10.pth', map_location=device)) # type: ignore
    print("Modelo cargado exitosamente.")
    
    # 1. Graficar curvas
    plot_training_curves()
    
    # 2. Evaluar métricas y matriz
    evaluate_model(model, test_loader, device)
    
    # 3. Mostrar errores
    show_errors(model, test_loader, device)