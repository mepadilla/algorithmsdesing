#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 08:55:43 2026

@author: meppadilla
"""

import numpy as np

# =====================================================================
# 1. FUNCIONES AUXILIARES (ACTIVACIÓN)
# =====================================================================
def sigmoide(z):
    """Función de activación Sigmoide. Comprime los valores entre 0 y 1."""
    return 1 / (1 + np.exp(-z))

def sigmoide_derivada(salida_sigmoide):
    """
    Derivada de la Sigmoide.
    Matemáticamente es: sigmoide(z) * (1 - sigmoide(z)).
    """
    return salida_sigmoide * (1 - salida_sigmoide)


# =====================================================================
# 2. CLASE: PERCEPTRÓN MULTICAPA (DEEP LEARNING)
# =====================================================================
class NeuralNetworkMLP:
    def __init__(self, capas_arquitectura, tasa_aprendizaje=0.1):
        self.arquitectura = capas_arquitectura
        self.lr = tasa_aprendizaje
        self.parametros = {} 
        self.memoria = {}     

        # Inicialización de pesos (aleatorios pequeños) y sesgos (ceros)
        for l in range(1, len(capas_arquitectura)):
            self.parametros['W' + str(l)] = np.random.randn(
                capas_arquitectura[l], capas_arquitectura[l-1]
            ) * 0.1 
            self.parametros['b' + str(l)] = np.zeros((capas_arquitectura[l], 1))

    def forward(self, X):
        """Propagación hacia adelante: de la entrada a la predicción."""
        activacion_anterior = X
        self.memoria['A0'] = X 

        num_capas = len(self.arquitectura) - 1

        for l in range(1, num_capas + 1):
            W = self.parametros['W' + str(l)]
            b = self.parametros['b' + str(l)]

            # Z = W*A + b
            entrada_neta = np.dot(W, activacion_anterior) + b
            # A = sigmoide(Z)
            activacion_actual = sigmoide(entrada_neta)

            self.memoria['Z' + str(l)] = entrada_neta
            self.memoria['A' + str(l)] = activacion_actual
            
            activacion_anterior = activacion_actual

        return activacion_actual 

    def backward(self, Y_real):
        """Retropropagación: cálculo de gradientes aplicando Regla de la Cadena."""
        num_muestras = Y_real.shape[1]
        num_capas = len(self.arquitectura) - 1
        A_final = self.memoria['A' + str(num_capas)]

        gradientes = {}

        # 1. Error en la capa de salida
        dZ_actual = (A_final - Y_real) * sigmoide_derivada(A_final)
        activacion_anterior = self.memoria['A' + str(num_capas - 1)]
        
        gradientes['dW' + str(num_capas)] = (1 / num_muestras) * np.dot(dZ_actual, activacion_anterior.T)
        gradientes['db' + str(num_capas)] = (1 / num_muestras) * np.sum(dZ_actual, axis=1, keepdims=True)
        
        dZ_propagar = dZ_actual

        # 2. Propagar el error hacia las capas ocultas
        for l in range(num_capas - 1, 0, -1):
            W_siguiente = self.parametros['W' + str(l+1)]
            A_actual = self.memoria['A' + str(l)]
            
            # Regla de la cadena
            error_propagado = np.dot(W_siguiente.T, dZ_propagar)
            dZ_actual = error_propagado * sigmoide_derivada(A_actual)

            activacion_capa_anterior = self.memoria['A' + str(l-1)]
            gradientes['dW' + str(l)] = (1 / num_muestras) * np.dot(dZ_actual, activacion_capa_anterior.T)
            gradientes['db' + str(l)] = (1 / num_muestras) * np.sum(dZ_actual, axis=1, keepdims=True)
            
            dZ_propagar = dZ_actual

        # 3. Actualización de pesos y sesgos (Descenso del Gradiente)
        for l in range(1, num_capas + 1):
            self.parametros['W' + str(l)] -= self.lr * gradientes['dW' + str(l)]
            self.parametros['b' + str(l)] -= self.lr * gradientes['db' + str(l)]

    def entrenar(self, X, Y, epochs):
        """Bucle principal para entrenar la red."""
        for epoch in range(epochs):
            Y_pred = self.forward(X)
            self.backward(Y)
            
            # Imprimir el costo cada 1000 iteraciones para evaluar el progreso
            if epoch % 1000 == 0:
                costo = np.mean(np.square(Y_pred - Y)) # Error Cuadrático Medio (MSE)
                print(f"Época {epoch:4d} | Costo MSE: {costo:.6f}")

    def predecir(self, X):
        return self.forward(X)


# =====================================================================
# 3. EJECUCIÓN, LÓGICA DE USO Y EVALUACIÓN DE RESULTADOS
# =====================================================================
if __name__ == "__main__":
    
    print("=========================================")
    print("  PRUEBA DE RED NEURONAL: PROBLEMA XOR")
    print("=========================================")
    print("LÓGICA: La compuerta XOR (O exclusivo) devuelve 1 solo si")
    print("las entradas son diferentes (0,1 o 1,0). Si son iguales")
    print("(0,0 o 1,1) devuelve 0. Es un problema no linealmente")
    print("separable, por lo que requiere capas ocultas para resolverse.\n")

    # Datos de entrenamiento XOR
    # X_train: Las columnas son las muestras, las filas las características
    # Muestra 1: [0,0]^T, Muestra 2: [0,1]^T, Muestra 3: [1,0]^T, Muestra 4: [1,1]^T
    X_train = np.array([
        [0, 0, 1, 1],  # Entrada 1
        [0, 1, 0, 1]   # Entrada 2
    ])

    # Y_train: Las etiquetas esperadas
    Y_train = np.array([[0, 1, 1, 0]])

    # Instanciamos la red: 
    # - 2 neuronas de entrada (las dos características)
    # - 4 neuronas en la capa oculta (suficientes para trazar las fronteras no lineales)
    # - 1 neurona de salida (la predicción final 0 o 1)
    # Usamos una tasa de aprendizaje relativamente alta (0.5) para acelerar la convergencia
    mlp = NeuralNetworkMLP(capas_arquitectura=[2, 4, 1], tasa_aprendizaje=0.5)

    print("Iniciando entrenamiento (10,000 épocas)...")
    mlp.entrenar(X_train, Y_train, epochs=10000)
    print("Entrenamiento finalizado.\n")

    print("=========================================")
    print("        RESULTADOS DE LA PREDICCIÓN")
    print("=========================================")
    
    # Hacemos la predicción con los mismos datos para ver si los aprendió
    predicciones = mlp.predecir(X_train)
    
    # Evaluamos mostrando la probabilidad cruda y el valor redondeado
    for i in range(X_train.shape[1]):
        entrada = X_train[:, i]
        valor_real = Y_train[0, i]
        pred_cruda = predicciones[0, i]
        pred_redondeada = int(np.round(pred_cruda))
        
        # Validar si acertó
        resultado = "CORRECTO" if pred_redondeada == valor_real else "INCORRECTO"
        
        print(f"Entrada: {entrada} | Esperado: {valor_real} | Predicción red: {pred_cruda:.4f} -> {pred_redondeada} [{resultado}]")