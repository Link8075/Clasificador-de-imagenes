import torch
import torch.nn as nn

class Cifar10CNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(Cifar10CNN, self).__init__()
        
        # JUSTIFICACIÓN DE ARQUITECTURA:
        # Usamos bloques convolucionales progresivos (32 -> 64 -> 128 filtros).
        # Esto permite capturar características simples (bordes) al inicio y complejas (formas) al final.
        #  Selección y justificación de capas convolucionales.
        
        # Bloque 1: Detectores de características básicas
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),     # Normalización para acelerar convergencia [cite: 71]
            nn.ReLU(),              # Activación no lineal [cite: 58]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)      # Reduce dimensión a 16x16 [cite: 56]
        )

        # Bloque 2: Características intermedias
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)      # Reduce dimensión a 8x8
        )

        # Bloque 3: Características complejas
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)      # Reduce dimensión a 4x4
        )

        # Clasificador (Fully Connected)
        # Flatten: 256 canales * 4 * 4 píxeles = 4096 neuronas de entrada
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Regularización para evitar overfitting 
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x

# Bloque de prueba para verificar dimensiones
if __name__ == "__main__":
    # Simular una imagen de 32x32 (Batch size 1, 3 canales, 32 alto, 32 ancho)
    dummy_input = torch.randn(1, 3, 32, 32)
    model = Cifar10CNN()
    output = model(dummy_input)
    print(f"Arquitectura creada exitosamente.")
    print(f"Dimensiones de salida: {output.shape}") # Debe ser [1, 10] (10 clases)