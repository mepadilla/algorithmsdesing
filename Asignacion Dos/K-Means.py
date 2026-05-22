#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 08:43:24 2026

@author: meppadilla
"""

import numpy as np

class KMeans:
    def __init__(self, K=3, max_iter=100):
        self.K = K
        self.max_iter = max_iter
        self.centroides = []

    def entrenar(self, X):
        muestras, caracteristicas = X.shape
        
        # 1. Inicialización: Elegir K puntos aleatorios como centroides iniciales
        indices_aleatorios = np.random.choice(muestras, self.K, replace=False)
        self.centroides = X[indices_aleatorios]

        for _ in range(self.max_iter):
            # 2. Asignación: Crear agrupaciones asociando cada punto a su centroide más cercano
            clusters = self._crear_clusters(X)
            
            # Guardamos el estado anterior para verificar la convergencia
            centroides_anteriores = self.centroides.copy()
            
            # 3. Actualización: Mover el centroide a la media matemática de su agrupación
            self.centroides = self._obtener_nuevos_centroides(clusters, X)
            
            # 4. Condición de convergencia: Si los centroides ya no se mueven, terminamos
            if np.all(centroides_anteriores == self.centroides):
                break
                
        return self._obtener_etiquetas_cluster(clusters, X)

    def _crear_clusters(self, X):
        # Inicializar listas vacías para cada cluster
        clusters = [[] for _ in range(self.K)]
        for idx, muestra in enumerate(X):
            # Distancia euclidiana (norma vectorial) a todos los centroides
            distancias = [np.linalg.norm(muestra - c) for c in self.centroides]
            centroide_mas_cercano = np.argmin(distancias)
            clusters[centroide_mas_cercano].append(idx)
        return clusters

    def _obtener_nuevos_centroides(self, clusters, X):
        nuevos_centroides = np.zeros((self.K, X.shape[1]))
        for idx_cluster, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue # Proteccion contra clusters vacíos
            # La nueva ubicación es el vector medio de los puntos asignados
            media_cluster = np.mean(X[cluster], axis=0)
            nuevos_centroides[idx_cluster] = media_cluster
        return nuevos_centroides

    def _obtener_etiquetas_cluster(self, clusters, X):
        # Aplanar la lista de agrupaciones a un vector unidimensional de etiquetas
        etiquetas = np.empty(X.shape[0])
        for idx_cluster, cluster in enumerate(clusters):
            for idx_muestra in cluster:
                etiquetas[idx_muestra] = idx_cluster
        return etiquetas