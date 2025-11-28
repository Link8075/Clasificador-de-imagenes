import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir='./data', batch_size=64, val_split=0.1):
    """
    Descarga CIFAR-10 y prepara los loaders con Data Augmentation.
    
    Args:
        data_dir: Directorio para guardar el dataset.
        batch_size: Tamaño del lote (64 o 128 es bueno para tu GPU).
        val_split: Porcentaje del set de entrenamiento a usar para validación (ej. 10%).
    """
    
    # -----------------------------------------------------------
    # 1. Definición de Transformaciones (Data Augmentation)
    # Requisito: "Pipeline completo de preprocesamiento optimizado" [cite: 37]
    # -----------------------------------------------------------
    
    # Estadísticas conocidas de CIFAR-10 para normalización exacta
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # Transformaciones para ENTRENAMIENTO (Augmentation fuerte para evitar overfitting)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # Recorte aleatorio con relleno
        transforms.RandomHorizontalFlip(),          # Volteo horizontal (espejo)
        transforms.RandomRotation(15),              # Rotación leve (+- 15 grados)
        transforms.ToTensor(),                      # Convertir a tensor
        transforms.Normalize(mean, std)             # Estandarización 
    ])

    # Transformaciones para VALIDACIÓN y TEST (Solo normalización, sin aleatoriedad)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # -----------------------------------------------------------
    # 2. Descarga y Split del Dataset
    # Requisito: "Manejo de splits de entrenamiento/validación" 
    # -----------------------------------------------------------
    
    # Descargamos el set completo de entrenamiento
    train_dataset_full = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    # Calculamos tamaños para dividir train y validation
    val_size = int(len(train_dataset_full) * val_split)
    train_size = len(train_dataset_full) - val_size

    # Dividimos el set
    train_subset, val_subset = random_split(train_dataset_full, [train_size, val_size])
    
    # IMPORTANTE: Al set de validación le quitamos el Data Augmentation aleatorio
    # (Sobrescribimos su transformación para que sea determinista como el test)
    val_subset.dataset.transform = test_transform 

    # Descargamos el set de test independiente
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # -----------------------------------------------------------
    # 3. Creación de DataLoaders
    # -----------------------------------------------------------
    
    # num_workers=2 usa la CPU para cargar datos mientras la GPU entrena
    # pin_memory=True acelera la transferencia RAM -> VRAM (GPU)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

# Bloque de prueba: Si ejecutas este archivo solo, verificará que todo funcione
if __name__ == "__main__":
    print("Probando carga de datos...")
    t, v, te = get_data_loaders()
    print(f"Batches de entrenamiento: {len(t)}")
    print(f"Batches de validación: {len(v)}")
    print(f"Batches de test: {len(te)}")
    
    # Verificar dimensiones de un lote
    images, labels = next(iter(t))
    print(f"Dimensiones de imágenes: {images.shape}") # Debería ser [64, 3, 32, 32]
    print("¡Carga exitosa!")