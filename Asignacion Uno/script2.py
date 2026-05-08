#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:44:22 2026

@author: meppadilla
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# Aumentar el límite de recursión para QuickSort en datasets grandes
sys.setrecursionlimit(200000)

# --- Algoritmos de Búsqueda ---
def busqueda_lineal(lista, objetivo):
    for i in range(len(lista)):
        if lista[i] == objetivo: return i
    return -1

def busqueda_binaria(lista, objetivo):
    bajo, alto = 0, len(lista) - 1
    while bajo <= alto:
        medio = (bajo + alto) // 2
        if lista[medio] == objetivo: return medio
        elif lista[medio] < objetivo: bajo = medio + 1
        else: alto = medio - 1
    return -1

# --- Algoritmos de Ordenamiento Cuadráticos ---
def ordenamiento_burbuja(lista):
    n = len(lista)
    for i in range(n):
        for j in range(0, n - i - 1):
            if lista[j] > lista[j + 1]:
                lista[j], lista[j + 1] = lista[j + 1], lista[j]

def ordenamiento_seleccion(lista):
    n = len(lista)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if lista[min_idx] > lista[j]: min_idx = j
        lista[i], lista[min_idx] = lista[min_idx], lista[i]

# --- Algoritmo de Ordenamiento Eficiente: QuickSort ---
def quicksort(lista):
    if len(lista) <= 1:
        return lista
    pivote = lista[len(lista) // 2]
    izq = [x for x in lista if x < pivote]
    centro = [x for x in lista if x == pivote]
    der = [x for x in lista if x > pivote]
    return quicksort(izq) + centro + quicksort(der)

# --- Configuración de Pruebas ---
n_busqueda = [100, 1000, 10000, 100000, 500000, 1000000]
n_ordenamiento_lento = [10, 100, 500, 1000, 2500, 5000] 
n_ordenamiento_rapido = [10, 100, 1000, 5000, 10000, 50000, 100000]

resultados = {
    "Lineal": [], "Binaria": [],
    "Burbuja": [], "Seleccion": [], "QuickSort": []
}

print("Iniciando Benchmarking comparativo...")

# 1. Pruebas de Búsqueda
for n in n_busqueda:
    data = sorted([random.randint(0, n*10) for _ in range(n)])
    target = -1
    
    t0 = time.perf_counter()
    busqueda_lineal(data, target)
    resultados["Lineal"].append(time.perf_counter() - t0)
    
    t0 = time.perf_counter()
    busqueda_binaria(data, target)
    resultados["Binaria"].append(time.perf_counter() - t0)

# 2. Pruebas de Ordenamiento Lento (O(n^2))
for n in n_ordenamiento_lento:
    data = [random.randint(0, n*10) for _ in range(n)]
    
    # Burbuja
    c1 = data.copy()
    t0 = time.perf_counter()
    ordenamiento_burbuja(c1)
    resultados["Burbuja"].append(time.perf_counter() - t0)
    
    # Selección
    c2 = data.copy()
    t0 = time.perf_counter()
    ordenamiento_seleccion(c2)
    resultados["Seleccion"].append(time.perf_counter() - t0)

# 3. Pruebas de Ordenamiento Rápido (O(n log n))
for n in n_ordenamiento_rapido:
    data = [random.randint(0, n*10) for _ in range(n)]
    t0 = time.perf_counter()
    quicksort(data)
    resultados["QuickSort"].append(time.perf_counter() - t0)

# --- Gráficas ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfica de Búsqueda
ax1.plot(n_busqueda, resultados["Lineal"], 'o-', label="Lineal O(n)")
ax1.plot(n_busqueda, resultados["Binaria"], 's-', label="Binaria O(log n)")
ax1.set_title("Comparativa de Búsqueda")
ax1.set_xlabel("n")
ax1.set_ylabel("Tiempo (s)")
ax1.legend()
ax1.grid(True)

# Gráfica de Ordenamiento (Aquí es donde QuickSort brilla)
ax2.plot(n_ordenamiento_lento, resultados["Burbuja"], 'r-o', label="Burbuja O(n^2)")
ax2.plot(n_ordenamiento_lento, resultados["Seleccion"], 'g-s', label="Selección O(n^2)")
ax2.plot(n_ordenamiento_rapido, resultados["QuickSort"], 'b-d', label="QuickSort O(n log n)")
ax2.set_title("Comparativa de Ordenamiento")
ax2.set_xlabel("n")
ax2.set_ylabel("Tiempo (s)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

print("\n¡Prueba completada! Observa cómo QuickSort (línea azul) se mantiene cerca de cero")
print("incluso cuando n es 10 veces mayor que el límite de los otros métodos.")