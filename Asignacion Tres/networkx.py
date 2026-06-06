#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:23:31 2026

@author: meppadilla
"""

import networkx as nx

# 1. Instanciamos un grafo dirigido
G = nx.DiGraph()

# 2. Añadimos las conexiones (Origen, Destino, Peso)
# Usaremos un escenario de distribución de potencia
G.add_weighted_edges_from([
    ("Tablero Principal", "Subestacion A", 10),
    ("Tablero Principal", "Panel de Control", 5),
    ("Subestacion A", "Motor B (460V)", 1),
    ("Panel de Control", "Subestacion A", 3),
    ("Panel de Control", "Motor A (460V)", 2),
    
    # Tramo regenerativo: inyecta energía a la red (peso negativo)
    ("Motor A (460V)", "Motor B (460V)", -6),
    
    # Falla estructural: El Motor B retroalimenta al Panel de Control, 
    # creando un ciclo cerrado con ganancia neta infinita.
    ("Motor B (460V)", "Panel de Control", -2) 
])

print("--- Análisis de Red con Bellman-Ford (NetworkX) ---\n")

try:
    # 3. Intentamos detectar si existe un ciclo negativo usando la implementación nativa
    # NetworkX ejecutará la complejidad O(V * E) internamente
    ciclo_negativo = nx.find_negative_cycle(G, source="Tablero Principal", weight="weight")
    
    print("⚠️ ALERTA ESTRUCTURAL: Ciclo de costo negativo detectado.")
    print("Esto indica una anomalía topológica en el diseño de la red.")
    print(f"Componentes atrapados en el bucle: {' -> '.join(ciclo_negativo)}")
    
except nx.NetworkXError:
    print("Red estable (sin ciclos negativos). Calculando rutas óptimas...\n")
    
    # Si la red es estable, calculamos las rutas más cortas de forma segura
    longitudes, rutas = nx.single_source_bellman_ford(G, source="Tablero Principal", weight="weight")
    
    destino = "Motor B (460V)"
    print(f"Ruta óptima hacia {destino}: {' -> '.join(rutas[destino])}")
    print(f"Costo total acumulado: {longitudes[destino]}")