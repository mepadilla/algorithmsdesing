#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:35:25 2026

@author: meppadilla
"""

import networkx as nx

# 1. Instanciar un Grafo Dirigido
G = nx.DiGraph()

# 2. Definir la red con sus CAPACIDADES (no distancias)
# Modelamos un sistema con válvulas y equipos de bombeo. Unidades en L/s.
conexiones = [
    ("Tanque Principal", "Manifold A", {"capacity": 100}),
    ("Tanque Principal", "Manifold B", {"capacity": 50}),
    ("Manifold A", "Bomba US Motors 1", {"capacity": 40}),
    ("Manifold A", "Manifold B", {"capacity": 30}),  # Válvula de interconexión (bypass)
    ("Manifold B", "Bomba US Motors 2", {"capacity": 60}),
    ("Bomba US Motors 1", "Planta Destino", {"capacity": 80}),
    ("Bomba US Motors 2", "Planta Destino", {"capacity": 70})
]

G.add_edges_from(conexiones)

# Nodos fuente y sumidero
origen = "Tanque Principal"
destino = "Planta Destino"

# --- Cálculo de Flujo Máximo (Edmonds-Karp) ---
# Calcula el volumen máximo que soporta el sistema
caudal_maximo, distribucion_flujo = nx.maximum_flow(
    G, 
    _s=origen, 
    _t=destino, 
    flow_func=nx.algorithms.flow.edmonds_karp
)

print(f"--- Análisis de Capacidad del Sistema de Bombeo ---")
print(f"Caudal Máximo Soportado: {caudal_maximo} L/s\n")

print("Distribución del flujo por componente (Flujo / Capacidad Límite):")
for u, flujos_salientes in distribucion_flujo.items():
    for v, flujo in flujos_salientes.items():
        capacidad_max = G[u][v]['capacity']
        estado = "⚠️ SATURADO" if flujo == capacidad_max else "OK"
        print(f"  {u:20} -> {v:20} : {flujo:3} / {capacidad_max:3} L/s [{estado}]")

# --- Identificación del Cuello de Botella (Corte Mínimo) ---
# Particiona la red para encontrar el punto de falla de capacidad
valor_corte, particiones = nx.minimum_cut(
    G, 
    _s=origen, 
    _t=destino, 
    flow_func=nx.algorithms.flow.edmonds_karp
)

nodos_antes_del_corte = particiones[0]
nodos_despues_del_corte = particiones[1]

print(f"\n--- Diagnóstico de Cuello de Botella (Corte Mínimo) ---")
print(f"El límite del sistema está dictado por los enlaces que cruzan esta división:")
for u in nodos_antes_del_corte:
    for v in G.neighbors(u):
        if v in nodos_despues_del_corte:
            print(f" -> Tubería crítica a redimensionar: {u} hacia {v} (Capacidad actual: {G[u][v]['capacity']} L/s)")