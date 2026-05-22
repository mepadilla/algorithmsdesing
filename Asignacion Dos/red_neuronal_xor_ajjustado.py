#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:08:19 2026

@author: meppadilla
"""

import numpy as np

# =====================================================================
# 1. FUNCIONES AUXILIARES
# =====================================================================
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def sigmoide_derivada(salida_sigmoide):
    return salida_sigmoide * (1 - salida_sigmoide)

# =====================================================================
# 2. CLASE: PERCEPTRÓN MULTICAPA OPTIMIZADO (XAVIER + MOMENTUM)
# =====================================================================
class NeuralNetworkMLPOptimizada:
    def __init__(self, capas_arquitectura, tasa_aprendizaje=0.1, momentum=0.9):
        self.arquitectura = capas_arquitectura
        self.lr = tasa_aprendizaje
        self.momentum = momentum
        self.parametros = {} 
        self.velocidad = {} # Diccionario nuevo para almacenar la inercia
        self.memoria = {}     

        for l in range(1, len(capas_arquitectura)):
            # MEJORA 1: Inicialización de Xavier
            # La varianza se escala inversamente al número de conexiones de entrada
            n_in = capas_arquitectura[l-1]
            n_out = capas_arquitectura[l]
            
            self.parametros['W' + str(l)] = np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)
            self.parametros['b' + str(l)] = np.zeros((n_out, 1))
            
            # Inicializamos las velocidades en cero
            self.velocidad['dW' + str(l)] = np.zeros((n_out, n_in))
            self.velocidad['db' + str(l)] = np.zeros((n_out, 1))

    def forward(self, X):
        activacion_anterior = X
        self.memoria['A0'] = X 
        num_capas = len(self.arquitectura) - 1

        for l in range(1, num_capas + 1):
            W = self.parametros['W' + str(l)]
            b = self.parametros['b' + str(l)]
            
            entrada_neta = np.dot(W, activacion_anterior) + b
            activacion_actual = sigmoide(entrada_neta)
            
            self.memoria['Z' + str(l)] = entrada_neta
            self.memoria['A' + str(l)] = activacion_actual
            activacion_anterior = activacion_actual

        return activacion_actual 

    def backward(self, Y_real):
        num_muestras = Y_real.shape[1]
        num_capas = len(self.arquitectura) - 1
        A_final = self.memoria['A' + str(num_capas)]
        gradientes = {}

        # Error en capa de salida
        dZ_actual = (A_final - Y_real) * sigmoide_derivada(A_final)
        activacion_anterior = self.memoria['A' + str(num_capas - 1)]
        
        gradientes['dW' + str(num_capas)] = (1 / num_muestras) * np.dot(dZ_actual, activacion_anterior.T)
        gradientes['db' + str(num_capas)] = (1 / num_muestras) * np.sum(dZ_actual, axis=1, keepdims=True)
        dZ_propagar = dZ_actual

        # Propagación hacia atrás
        for l in range(num_capas - 1, 0, -1):
            W_siguiente = self.parametros['W' + str(l+1)]
            A_actual = self.memoria['A' + str(l)]
            
            error_propagado = np.dot(W_siguiente.T, dZ_propagar)
            dZ_actual = error_propagado * sigmoide_derivada(A_actual)

            activacion_capa_anterior = self.memoria['A' + str(l-1)]
            gradientes['dW' + str(l)] = (1 / num_muestras) * np.dot(dZ_actual, activacion_capa_anterior.T)
            gradientes['db' + str(l)] = (1 / num_muestras) * np.sum(dZ_actual, axis=1, keepdims=True)
            dZ_propagar = dZ_actual

        # MEJORA 2: Actualización con Momento (Inercia)
        for l in range(1, num_capas + 1):
            # v = momentum * v + lr * gradiente
            self.velocidad['dW' + str(l)] = (self.momentum * self.velocidad['dW' + str(l)]) + (self.lr * gradientes['dW' + str(l)])
            self.velocidad['db' + str(l)] = (self.momentum * self.velocidad['db' + str(l)]) + (self.lr * gradientes['db' + str(l)])
            
            # W = W - v
            self.parametros['W' + str(l)] -= self.velocidad['dW' + str(l)]
            self.parametros['b' + str(l)] -= self.velocidad['db' + str(l)]

    def entrenar(self, X, Y, epochs):
        for epoch in range(epochs):
            Y_pred = self.forward(X)
            self.backward(Y)
            if epoch % 1000 == 0:
                costo = np.mean(np.square(Y_pred - Y))
                print(f"Época {epoch:4d} | Costo MSE: {costo:.6f}")

    def predecir(self, X):
        return self.forward(X)

# =====================================================================
# 3. EJECUCIÓN
# =====================================================================
if __name__ == "__main__":
    X_train = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    Y_train = np.array([[0, 1, 1, 0]])

    # Añadimos el hiperparámetro de momentum (típicamente 0.9)
    mlp = NeuralNetworkMLPOptimizada(capas_arquitectura=[2, 4, 1], tasa_aprendizaje=0.5, momentum=0.9)

    print("Iniciando entrenamiento OPTIMIZADO (Xavier + Momentum)...")
    # Reducimos las épocas porque convergerá mucho más rápido
    mlp.entrenar(X_train, Y_train, epochs=4000) 
    print("Entrenamiento finalizado.\n")

    print("RESULTADOS DE LA PREDICCIÓN:")
    predicciones = mlp.predecir(X_train)
    for i in range(X_train.shape[1]):
        entrada = X_train[:, i]
        valor_real = Y_train[0, i]
        pred_cruda = predicciones[0, i]
        pred_redondeada = int(np.round(pred_cruda))
        resultado = "CORRECTO" if pred_redondeada == valor_real else "INCORRECTO"
        print(f"Entrada: {entrada} | Esperado: {valor_real} | Red: {pred_cruda:.4f} -> {pred_redondeada} [{resultado}]")