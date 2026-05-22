#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 08:44:11 2026

@author: meppadilla
"""

import numpy as np

# 0. Funciones Matemáticas Auxiliares
def sigmoide(z):
    """Función de activación Sigmoide."""
    return 1 / (1 + np.exp(-z))

def sigmoide_derivada(salida_sigmoide):
    """
    Derivada de la Sigmoide.
    Ojo: Recibe la SALIDA de la sigmoide (ya activada), no la entrada z.
    La fórmula matemática es sigmoide(z) * (1 - sigmoide(z)).
    """
    return salida_sigmoide * (1 - salida_sigmoide)

class NeuralNetworkMLP:
    def __init__(self, capas_arquitectura, tasa_aprendizaje=0.1):
        """
        Argumentos:
            capas_arquitectura: Lista con el número de neuronas por capa.
                               Ej: [2, 4, 1] -> 2 entradas, 4 ocultas, 1 salida.
            tasa_aprendizaje: Coeficiente alpha para el gradient descent.
        """
        self.arquitectura = capas_arquitectura
        self.lr = tasa_aprendizaje
        self.parametros = {} # Aquí guardaremos W y b
        self.memoria = {}     # Aquí guardaremos activaciones (A) y entradas netas (Z)

        # 1. Inicialización de Parámetros
        for l in range(1, len(capas_arquitectura)):
            # Pesos W^[l]: matriz (neuronas_capa_actual, neuronas_capa_anterior)
            self.parametros['W' + str(l)] = np.random.randn(
                capas_arquitectura[l], capas_arquitectura[l-1]
            ) * 0.1 # Escala pequeña
            
            # Sesgos b^[l]: vector columna (neuronas_capa_actual, 1)
            self.parametros['b' + str(l)] = np.zeros((capas_arquitectura[l], 1))

    def forward(self, X):
        """
        Propagación hacia adelante.
        Recibe X: datos de entrada de forma (características, muestras).
        Retorna la activación final (predicción).
        """
        # La 'activación' de la capa 0 es la entrada misma.
        activacion_anterior = X
        self.memoria['A0'] = X # Guardar en memoria para backprop

        num_capas = len(self.arquitectura) - 1

        for l in range(1, num_capas + 1):
            # Obtener W y b de la capa l
            W = self.parametros['W' + str(l)]
            b = self.parametros['b' + str(l)]

            # Combinación lineal: Z^[l] = W^[l] * A^[l-1] + b^[l]
            entrada_neta = np.dot(W, activacion_anterior) + b
            
            # Función de activación: A^[l] = sigma(Z^[l])
            activacion_actual = sigmoide(entrada_neta)

            # Guardar en memoria para usar en backprop
            self.memoria['Z' + str(l)] = entrada_neta
            self.memoria['A' + str(l)] = activacion_actual
            
            # Actualizar para la siguiente iteración
            activacion_anterior = activacion_actual

        return activacion_actual # La última activación es la predicción Y_hat

    def backward(self, Y_real):
        """
        Retropropagación y actualización de parámetros.
        Recibe Y_real: etiquetas reales de forma (salidas, muestras).
        """
        num_muestras = Y_real.shape[1]
        num_capas = len(self.arquitectura) - 1
        A_final = self.memoria['A' + str(num_capas)]

        # --- A. Capa de Salida (L) ---
        # 1. Error de la capa de salida (delta L).
        # Proviene de derivar MSE respecto a Z^[L]: (Y_hat - Y) * sigmoide'(Z^[L])
        # Pero sigmoide'(Z) = A * (1 - A), así que usamos A_final.
        
        # Ojo: dA (derivada pérdida vs activación final) es (A_final - Y_real)
        dZ_actual = (A_final - Y_real) * sigmoide_derivada(A_final)

        # 2. Gradientes de W y b para la capa de salida.
        # dW^[L] = (1/m) * dZ^[L] * A^[L-1]^T
        activacion_anterior = self.memoria['A' + str(num_capas - 1)]
        dW_actual = (1 / num_muestras) * np.dot(dZ_actual, activacion_anterior.T)
        db_actual = (1 / num_muestras) * np.sum(dZ_actual, axis=1, keepdims=True)

        # 3. Guardar gradientes y actualizar dZ para la capa anterior (Regla de la cadena)
        gradientes = {}
        gradientes['dW' + str(num_capas)] = dW_actual
        gradientes['db' + str(num_capas)] = db_actual
        
        # dZ para propagar hacia atrás: dZ^[l-1] = (W^[l]^T * dZ^[l]) * sigmoide'(Z^[l-1])
        dZ_propagar = dZ_actual

        # --- B. Capas Ocultas (L-1 hasta 1) ---
        for l in range(num_capas - 1, 0, -1):
            # dZ_propagar es dZ^[l+1] (capa siguiente)
            # W_siguiente es W^[l+1]
            W_siguiente = self.parametros['W' + str(l+1)]
            
            # Z de la capa actual
            Z_actual = self.memoria['Z' + str(l)]
            # A de la capa actual
            A_actual = self.memoria['A' + str(l)]
            
            # Calcular dZ^[l] (Error de la capa actual) usando REGLA DE LA CADENA
            # (Derivada del error respecto a la entrada neta Z de esta capa)
            
            # dZ^[l] = (W^[l+1]^T * dZ^[l+1]) * sigmoide'(Z^[l])
            # sigmoide'(Z) = A * (1-A)
            
            error_propagado = np.dot(W_siguiente.T, dZ_propagar)
            derivada_activacion = sigmoide_derivada(A_actual)
            dZ_actual = error_propagado * derivada_activacion

            # Gradientes de W y b para esta capa oculta
            # dW^[l] = (1/m) * dZ^[l] * A^[l-1]^T
            activacion_capa_anterior = self.memoria['A' + str(l-1)]
            dW_actual = (1 / num_muestras) * np.dot(dZ_actual, activacion_capa_anterior.T)
            db_actual = (1 / num_muestras) * np.sum(dZ_actual, axis=1, keepdims=True)

            # Guardar gradientes
            gradientes['dW' + str(l)] = dW_actual
            gradientes['db' + str(l)] = db_actual
            
            # dZ_propagar se convierte en dZ^[l] para la siguiente iteración de bucle (capa l-1)
            dZ_propagar = dZ_actual

        # --- C. Actualización de Parámetros (Gradient Descent) ---
        for l in range(1, num_capas + 1):
            # W^[l] = W^[l] - lr * dW^[l]
            self.parametros['W' + str(l)] -= self.lr * gradientes['dW' + str(l)]
            # b^[l] = b^[l] - lr * db^[l]
            self.parametros['b' + str(l)] -= self.lr * gradientes['db' + str(l)]

    def entrenar(self, X, Y, epochs):
        """Bucle principal de entrenamiento."""
        for epoch in range(epochs):
            # 1. Forward
            Y_pred = self.forward(X)
            
            # 2. Backward & Update
            self.backward(Y)
            
            # 3. Mostrar progreso opcionalmente (costo MSE)
            if epoch % 100 == 0:
                costo = np.mean(np.square(Y_pred - Y)) # MSE
                print(f"Época {epoch}: Costo MSE = {costo:.6f}")

    def predecir(self, X):
        """Hace inferencia usando forward."""
        return self.forward(X)