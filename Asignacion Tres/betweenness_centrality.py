#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:43:17 2026

@author: meppadilla
"""

import networkx as nx

# 1. Creamos un Grafo Dirigido (el flujo de información y control tiene dirección)
G = nx.DiGraph()

# 2. Definimos la topología de la red de control industrial
enlaces = [
    # El servidor central envía datos a los paneles de control de área
    ("Servidor ERP (Odoo)", "HMI Planta Alta"),
    ("Servidor ERP (Odoo)", "HMI Planta Baja"),
    
    # Los paneles de control comandan a los PLC
    ("HMI Planta Alta", "PLC Norte"),
    ("HMI Planta Baja", "PLC Sur"),
    ("HMI Planta Baja", "PLC Centro"),
    
    # Los PLC controlan directamente los motores
    ("PLC Norte", "US Motors - Eje X"),
    ("PLC Norte", "US Motors - Eje Y"),
    ("PLC Sur", "US Motors - Bomba Principal"),
    ("PLC Centro", "US Motors - Compresor"),
    
    # Un enlace crítico de retroalimentación de seguridad
    # El compresor envía telemetría directa al HMI Planta Alta
    ("US Motors - Compresor", "HMI Planta Alta"),
    
    # El PLC Centro actúa como relé de emergencia para el PLC Norte
    ("PLC Centro", "PLC Norte")
]

G.add_edges_from(enlaces)

# --- 3. Cálculo de Métricas de Centralidad ---

# Betweenness Centrality (¿Quién es el puente más crítico?)
# endpoints=False ignora al origen y destino como puentes de sí mismos
betweenness = nx.betweenness_centrality(G, endpoints=False)

# PageRank (¿Quién concentra la mayor dominancia/influencia estructural?)
# alpha=0.85 es el factor de amortiguación estándar de Google
pagerank = nx.pagerank(G, alpha=0.85)

# --- 4. Análisis y Resultados ---

print("--- DIAGNÓSTICO DE CENTRALIDAD ESTRUCTURAL ---\n")

print("[1] Top 3 Nodos - Centralidad de Intermediación (Puentes Críticos):")
# Ordenamos de mayor a menor
top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
for nodo, valor in top_betweenness:
    print(f" -> {nodo:25} | Score: {valor:.4f}")

print("\n[2] Top 3 Nodos - PageRank (Dominancia de Red):")
top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]
for nodo, valor in top_pagerank:
    print(f" -> {nodo:25} | Score: {valor:.4f}")