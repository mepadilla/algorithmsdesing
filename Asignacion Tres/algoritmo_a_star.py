#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:34:13 2026

@author: meppadilla
"""

import heapq

# 1. Definición de la función Heurística h(n)
# Usaremos la Distancia Manhattan ya que la tubería solo puede girar a 90 grados
def heuristica_manhattan(nodo_actual, destino):
    x1, y1 = nodo_actual
    x2, y2 = destino
    return abs(x1 - x2) + abs(y1 - y2)

def algoritmo_a_star(cuadricula, inicio, fin):
    filas = len(cuadricula)
    columnas = len(cuadricula[0])
    
    # Cola de prioridad que almacena tuplas: (f_score, g_score, (x, y))
    cola_prioridad = [(0, 0, inicio)]
    
    # Diccionarios para rastrear los costos g(n) y reconstruir la ruta
    g_scores = {inicio: 0}
    predecesores = {inicio: None}
    
    # Movimientos ortogonales permitidos: Arriba, Abajo, Izquierda, Derecha
    movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while cola_prioridad:
        # Extraemos el nodo con el f(n) más bajo
        f_actual, g_actual, nodo_actual = heapq.heappop(cola_prioridad)
        
        # Si llegamos al motor de 460V, terminamos prematuramente (ventaja de A*)
        if nodo_actual == fin:
            break
            
        x, y = nodo_actual
        
        # 2. Fase de Exploración de Vecinos
        for dx, dy in movimientos:
            vecino_x, vecino_y = x + dx, y + dy
            vecino = (vecino_x, vecino_y)
            
            # Verificamos que el vecino esté dentro del plano y no sea un obstáculo (1)
            if 0 <= vecino_x < filas and 0 <= vecino_y < columnas:
                if cuadricula[vecino_x][vecino_y] == 1:
                    continue # Es un obstáculo físico, lo ignoramos
                
                # Asumimos un costo de 1 metro por cada celda avanzada
                g_calculado = g_actual + 1
                
                # Si encontramos un camino más corto hacia este vecino, relajamos
                if vecino not in g_scores or g_calculado < g_scores[vecino]:
                    g_scores[vecino] = g_calculado
                    
                    # f(n) = g(n) + h(n)
                    f_calculado = g_calculado + heuristica_manhattan(vecino, fin)
                    
                    predecesores[vecino] = nodo_actual
                    heapq.heappush(cola_prioridad, (f_calculado, g_calculado, vecino))
                    
    # 3. Reconstrucción de la ruta
    ruta = []
    nodo_paso = fin
    while nodo_paso is not None:
        ruta.append(nodo_paso)
        nodo_paso = predecesores.get(nodo_paso)
        
    return ruta[::-1] if ruta[-1] == inicio else []

# --- Ejemplo de Uso Práctico ---

# Plano de la planta: 0 = Espacio libre, 1 = Obstáculo (Maquinaria/Estructuras)
plano_planta = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0], # Muro bloqueando el paso directo
    [0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]

origen = (0, 0)  # Tablero
destino = (4, 5) # Motor 460V

ruta_optima = algoritmo_a_star(plano_planta, origen, destino)

print(f"Ruta óptima encontrada (Coordenadas):")
for paso in ruta_optima:
    print(f" -> {paso}")
print(f"\nDistancia total: {len(ruta_optima) - 1} metros (unidades)")