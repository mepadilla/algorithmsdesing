#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 08:42:36 2026

@author: meppadilla
"""

import numpy as np
from collections import Counter

def distancia_euclidiana(x1, x2):
    # Norma L2: D(x, y) = sqrt(sum(|x_i - y_i|^2))
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def entrenar(self, X, y):
        # Memorización pura del espacio espacial de datos
        self.X_train = X
        self.y_train = y

    def predecir(self, X):
        predicciones = [self._predecir_uno(x) for x in X]
        return np.array(predicciones)

    def _predecir_uno(self, x):
        # 1. Calcular las distancias entre el punto 'x' y todo el dataset
        distancias = [distancia_euclidiana(x, x_train) for x_train in self.X_train]
        
        # 2. Ordenar por distancia y extraer los índices de los 'k' más cercanos
        k_indices = np.argsort(distancias)[:self.k]
        
        # 3. Consultar las etiquetas (clases) de esos vecinos
        k_etiquetas = [self.y_train[i] for i in k_indices]
        
        # 4. Inferencia por votación mayoritaria (moda)
        mas_comun = Counter(k_etiquetas).most_common(1)
        return mas_comun[0][0]