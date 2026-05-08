#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:16:19 2026

@author: meppadilla
"""

import time
import random
import matplotlib.pyplot as plt
import sys

# Aumentar el límite de recursión por si acaso, aunque usaremos métodos iterativos
sys.setrecursionlimit(2000)

# --- Algoritmos de Búsqueda ---
def busqueda_lineal(lista, objetivo):
    for i in range(len(lista)):
        if lista[i] == objetivo:
            return i
    return -1

def busqueda_binaria(lista, objetivo):
    bajo = 0
    alto = len(lista) - 1
    while bajo <= alto:
        medio = (bajo + alto) // 2
        if lista[medio] == objetivo:
            return medio
        elif lista[medio] < objetivo:
            bajo = medio + 1
        else:
            alto = medio - 1
    return -1

# --- Algoritmos de Ordenamiento ---
def ordenamiento_burbuja(lista):
    n = len(lista)
    for i in range(n):
        for j in range(0, n - i - 1):
            if lista[j] > lista[j + 1]:
                lista[j], lista[j + 1] = lista[j + 1], lista[j]
    return lista

def ordenamiento_seleccion(lista):
    n = len(lista)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if lista[min_idx] > lista[j]:
                min_idx = j
        lista[i], lista[min_idx] = lista[min_idx], lista[i]
    return lista

# --- Configuración de la Prueba ---
ensayos = [10, 50, 100, 1000, 5000] # Reducido para ordenamiento (Burbuja es O(n^2))
ensayos_busqueda = [10, 50, 100, 1000, 10000, 100000, 1000000]

resultados = {
    "Lineal (Desordenado)": [],
    "Lineal (Ordenado)": [],
    "Binaria (Ordenado)": [],
    "Burbuja": [],
    "Selección": []
}

# Ejecución de Benchmarking para Búsqueda
for n in ensayos_busqueda:
    dataset = [random.randint(0, n*10) for _ in range(n)]
    objetivo = -1 # Forzamos el peor escenario (que no lo encuentre)
    
    # Búsqueda Lineal Desordenada
    t0 = time.time()
    busqueda_lineal(dataset, objetivo)
    resultados["Lineal (Desordenado)"].append(time.time() - t0)
    
    # Pre-ordenamos para las siguientes pruebas
    dataset.sort()
    
    # Búsqueda Lineal Ordenada
    t0 = time.time()
    busqueda_lineal(dataset, objetivo)
    resultados["Lineal (Ordenado)"].append(time.time() - t0)
    
    # Búsqueda Binaria
    t0 = time.time()
    busqueda_binaria(dataset, objetivo)
    resultados["Binaria (Ordenado)"].append(time.time() - t0)

# Ejecución para Ordenamiento (Solo hasta 5000 por el tiempo de ejecución)
for n in ensayos:
    dataset = [random.randint(0, n*10) for _ in range(n)]
    
    # Burbuja
    copia_b = dataset.copy()
    t0 = time.time()
    ordenamiento_burbuja(copia_b)
    resultados["Burbuja"].append(time.time() - t0)
    
    # Selección
    copia_s = dataset.copy()
    t0 = time.time()
    ordenamiento_seleccion(copia_s)
    resultados["Selección"].append(time.time() - t0)

# --- Gráficas ---
plt.figure(figsize=(12, 5))

# Gráfica de Búsqueda
plt.subplot(1, 2, 1)
plt.plot(ensayos_busqueda, resultados["Lineal (Desordenado)"], label="Lineal (Desordenado)")
plt.plot(ensayos_busqueda, resultados["Lineal (Ordenado)"], label="Lineal (Ordenado)")
plt.plot(ensayos_busqueda, resultados["Binaria (Ordenado)"], label="Binaria")
plt.title("Tiempos de Búsqueda")
plt.xlabel("n (Elementos)")
plt.ylabel("Tiempo (segundos)")
plt.legend()
plt.grid(True)

# Gráfica de Ordenamiento
plt.subplot(1, 2, 2)
plt.plot(ensayos, resultados["Burbuja"], label="Burbuja")
plt.plot(ensayos, resultados["Selección"], label="Selección")
plt.title("Tiempos de Ordenamiento")
plt.xlabel("n (Elementos)")
plt.ylabel("Tiempo (segundos)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()