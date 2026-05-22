#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 08:40:14 2026

@author: meppadilla
"""

import numpy as np

class RegresionLineal:
    def __init__(self, tasa_aprendizaje=0.01, iteraciones=1000):
        # Hiperparámetros del modelo
        self.lr = tasa_aprendizaje
        self.iteraciones = iteraciones
        self.pesos = None
        self.sesgo = None

    def entrenar(self, X, y):
        muestras, caracteristicas = X.shape
        
        # Inicializamos los parámetros matemáticos (vector w y escalar b)
        self.pesos = np.zeros(caracteristicas)
        self.sesgo = 0

        # Proceso de Optimización (Gradient Descent)
        for _ in range(self.iteraciones):
            # 1. Función de hipótesis lineal: y = Xw + b
            y_pred = np.dot(X, self.pesos) + self.sesgo

            # 2. Cálculo de gradientes (Derivadas parciales del Error Cuadrático Medio)
            # dw = (1/N) * X^T * (y_pred - y)
            dw = (1 / muestras) * np.dot(X.T, (y_pred - y))
            # db = (1/N) * sum(y_pred - y)
            db = (1 / muestras) * np.sum(y_pred - y)

            # 3. Actualización de parámetros en dirección opuesta al gradiente
            self.pesos -= self.lr * dw
            self.sesgo -= self.lr * db

    def predecir(self, X):
        return np.dot(X, self.pesos) + self.sesgo