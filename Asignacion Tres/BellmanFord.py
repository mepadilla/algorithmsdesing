#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:21:57 2026

@author: meppadilla
"""

class GrafoBellmanFord:
    def __init__(self, vertices):
        self.V = vertices
        self.aristas = []
        # Diccionario para mapear nombres de nodos a índices numéricos
        self.nodos = {}
        self.nombres = []

    def agregar_nodo(self, nombre):
        if nombre not in self.nodos:
            self.nodos[nombre] = len(self.nombres)
            self.nombres.append(nombre)

    def agregar_arista(self, origen, destino, peso):
        # Aseguramos que los nodos existan
        self.agregar_nodo(origen)
        self.agregar_nodo(destino)
        
        u = self.nodos[origen]
        v = self.nodos[destino]
        self.aristas.append([u, v, peso])

    def calcular_rutas(self, origen):
        # 1. Inicialización
        idx_origen = self.nodos[origen]
        distancias = {i: float("Inf") for i in range(self.V)}
        predecesores = {i: None for i in range(self.V)}
        distancias[idx_origen] = 0

        # 2. Fase de Relajación (Iterar V - 1 veces)
        for _ in range(self.V - 1):
            for u, v, w in self.aristas:
                if distancias[u] != float("Inf") and distancias[u] + w < distancias[v]:
                    distancias[v] = distancias[u] + w
                    predecesores[v] = u

        # 3. Detección de Ciclos de Costo Negativo
        # Una iteración extra para ver si podemos seguir optimizando
        for u, v, w in self.aristas:
            if distancias[u] != float("Inf") and distancias[u] + w < distancias[v]:
                print(f"⚠️ ALERTA: Se ha detectado un ciclo de costo negativo estructural.")
                print(f"La ruta hacia el nodo {self.nombres[v]} tiende a -Infinito.")
                return False, None, None

        return True, distancias, predecesores

    def imprimir_solucion(self, origen, distancias, predecesores):
        print(f"--- Análisis de Rutas Óptimas (Bellman-Ford) desde: {origen} ---")
        for i in range(self.V):
            nombre_destino = self.nombres[i]
            costo = distancias[i]
            
            # Reconstruir la ruta
            ruta = []
            nodo_actual = i
            while nodo_actual is not None:
                ruta.insert(0, self.nombres[nodo_actual])
                nodo_actual = predecesores[nodo_actual]
                
            print(f"Hacia {nombre_destino:15} | Costo de Energía: {costo:4} | Ruta: {' -> '.join(ruta)}")

# --- Ejecución del Modelo ---

# Tenemos 5 puntos de conexión en la planta
grafo = GrafoBellmanFord(5)

# Agregamos las conexiones (Origen, Destino, Costo/Peso)
# Costos positivos = caída de tensión / pérdida
grafo.agregar_arista("Generador", "Subestacion 1", 10)
grafo.agregar_arista("Generador", "Planta Baja", 5)
grafo.agregar_arista("Subestacion 1", "Planta Alta", 1)
grafo.agregar_arista("Planta Baja", "Subestacion 1", 3)
grafo.agregar_arista("Planta Baja", "Planta Alta", 9)
grafo.agregar_arista("Planta Baja", "Motor Retorno", 2)

# ESTA ES LA CLAVE: El Motor Retorno tiene un sistema regenerativo que inyecta
# energía hacia la Planta Alta, modelado como un peso negativo (-6).
# Dijkstra fallaría estrepitosamente aquí.
grafo.agregar_arista("Motor Retorno", "Planta Alta", -6)

# Calculamos desde el nodo de generación principal
nodo_inicio = "Generador"
exito, distancias_optimas, arbol_predecesores = grafo.calcular_rutas(nodo_inicio)

if exito:
    grafo.imprimir_solucion(nodo_inicio, distancias_optimas, arbol_predecesores)