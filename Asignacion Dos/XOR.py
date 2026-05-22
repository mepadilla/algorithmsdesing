#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 08:44:54 2026

@author: meppadilla

"""
import numpy as np

# Datos XOR (4 muestras, 2 características)
# X: forma (características, muestras) -> (2, 4)
X_train = np.array([
    [0, 0, 1, 1], # Característica 1
    [0, 1, 0, 1]  # Característica 2
])

# Etiquetas XOR (1 salida por muestra)
# Y: forma (salidas, muestras) -> (1, 4)
Y_train = np.array([[0, 1, 1, 0]])

# 1. Instanciar la red: [2 entradas, 3 ocultas, 1 salida]
mlp = NeuralNetworkMLP(capas_arquitectura=[2, 3, 1], tasa_aprendizaje=0.5)

# 2. Entrenar durante 2000 épocas
print("Iniciando entrenamiento...")
mlp.entrenar(X_train, Y_train, epochs=2000)
print("Entrenamiento finalizado.")

# 3. Probar predicciones
print("\nPredicciones finales (redondeadas):")
predicciones = mlp.predecir(X_train)
for i in range(X_train.shape[1]):
    entrada = X_train[:, i]
    pred = predicciones[0, i]
    print(f"Entrada: {entrada} -> Predicción: {pred:.4f} (Redondeado: {int(round(pred))})")