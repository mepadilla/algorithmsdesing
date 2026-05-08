#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:46:34 2026

@author: meppadilla
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# Optimización para manejar la recursividad de QuickSort en datasets grandes
sys.setrecursionlimit(200000)

# ==========================================
# 1. ALGORITMOS DE BÚSQUEDA
# ==========================================
def busqueda_lineal(lista, objetivo):
    for i in range(len(lista)):
        if lista[i] == objetivo:
            return i
    return -1

def busqueda_binaria(lista, objetivo):
    bajo, alto = 0, len(lista) - 1
    while bajo <= alto:
        medio = (bajo + alto) // 2
        if lista[medio] == objetivo:
            return medio
        elif lista[medio] < objetivo:
            bajo = medio + 1
        else:
            alto = medio - 1
    return -1

# ==========================================
# 2. ALGORITMOS DE ORDENAMIENTO
# ==========================================
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
            if lista[min_idx] > lista[j]:
                min_idx = j
        lista[i], lista[min_idx] = lista[min_idx], lista[i]

def quicksort(lista):
    if len(lista) <= 1:
        return lista
    pivote = lista[len(lista) // 2]
    izq = [x for x in lista if x < pivote]
    centro = [x for x in lista if x == pivote]
    der = [x for x in lista if x > pivote]
    return quicksort(izq) + centro + quicksort(der)

# ==========================================
# 3. LÓGICA DE ESTIMACIÓN BIG O
# ==========================================
def func_lineal(n, a, b): return a * n + b
def func_cuadratica(n, a, b): return a * (n**2) + b
def func_logaritmica(n, a, b): return a * np.log2(n + 1) + b
def func_nlogn(n, a, b): return a * n * np.log2(n + 1) + b

def estimar_big_o(n_vals, t_vals, tipo="busqueda"):
    n_pts = np.array(n_vals)
    t_pts = np.array(t_vals)
    try:
        popt_lin, _ = curve_fit(func_lineal, n_pts, t_pts)
        err_lin = np.sum((func_lineal(n_pts, *popt_lin) - t_pts)**2)
        
        if tipo == "busqueda":
            popt_log, _ = curve_fit(func_logaritmica, n_pts, t_pts)
            err_log = np.sum((func_logaritmica(n_pts, *popt_log) - t_pts)**2)
            return "O(log n)" if err_log < err_lin else "O(n)"
        elif tipo == "lento":
            popt_cua, _ = curve_fit(func_cuadratica, n_pts, t_pts)
            err_cua = np.sum((func_cuadratica(n_pts, *popt_cua) - t_pts)**2)
            return "O(n^2)" if err_cua < err_lin else "O(n)"
        else: # rapido
            popt_nlogn, _ = curve_fit(func_nlogn, n_pts, t_pts)
            err_nlogn = np.sum((func_nlogn(n_pts, *popt_nlogn) - t_pts)**2)
            return "O(n log n)" if err_nlogn < err_lin else "O(n)"
    except:
        return "Indeterminado"

# ==========================================
# 4. CONFIGURACIÓN Y EJECUCIÓN
# ==========================================
n_busqueda = [100, 1000, 10000, 100000, 500000, 1000000, 2000000]
n_ord_lento = [10, 100, 500, 1000, 2500, 5000]
n_ord_rapido = [10, 100, 1000, 5000, 10000, 50000, 100000]

resultados = {
    "Lin_Des": [], "Lin_Ord": [], "Binaria": [],
    "Burbuja": [], "Seleccion": [], "QuickSort": []
}

print("Ejecutando Benchmarking Integral...")

# Pruebas de Búsqueda
for n in n_busqueda:
    data = [random.randint(0, n*10) for _ in range(n)]
    target = -1
    
    t0 = time.perf_counter()
    busqueda_lineal(data, target)
    resultados["Lin_Des"].append(time.perf_counter() - t0)
    
    data.sort()
    
    t0 = time.perf_counter()
    busqueda_lineal(data, target)
    resultados["Lin_Ord"].append(time.perf_counter() - t0)
    
    t0 = time.perf_counter()
    busqueda_binaria(data, target)
    resultados["Binaria"].append(time.perf_counter() - t0)

# Pruebas de Ordenamiento
for n in n_ord_lento:
    data = [random.randint(0, n*10) for _ in range(n)]
    # Burbuja
    c1 = data.copy(); t0 = time.perf_counter()
    ordenamiento_burbuja(c1)
    resultados["Burbuja"].append(time.perf_counter() - t0)
    # Selección
    c2 = data.copy(); t0 = time.perf_counter()
    ordenamiento_seleccion(c2)
    resultados["Seleccion"].append(time.perf_counter() - t0)

for n in n_ord_rapido:
    data = [random.randint(0, n*10) for _ in range(n)]
    t0 = time.perf_counter()
    quicksort(data)
    resultados["QuickSort"].append(time.perf_counter() - t0)

# ==========================================
# 5. RESULTADOS Y GRÁFICAS
# ==========================================
print("\n" + "="*45)
print(f"{'Algoritmo':<25} | {'Big O Estimado':<15}")
print("-" * 45)
print(f"{'Búsqueda Lineal (Des)':<25} | {estimar_big_o(n_busqueda, resultados['Lin_Des'], 'busqueda')}")
print(f"{'Búsqueda Binaria':<25} | {estimar_big_o(n_busqueda, resultados['Binaria'], 'busqueda')}")
print(f"{'Ordenamiento Burbuja':<25} | {estimar_big_o(n_ord_lento, resultados['Burbuja'], 'lento')}")
print(f"{'Ordenamiento Selección':<25} | {estimar_big_o(n_ord_lento, resultados['Seleccion'], 'lento')}")
print(f"{'QuickSort':<25} | {estimar_big_o(n_ord_rapido, resultados['QuickSort'], 'rapido')}")
print("="*45)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(n_busqueda, resultados["Lin_Des"], 'o-', label="Lineal (Desordenado)")
ax1.plot(n_busqueda, resultados["Lin_Ord"], 's-', label="Lineal (Ordenado)")
ax1.plot(n_busqueda, resultados["Binaria"], 'd-', label="Binaria")
ax1.set_title("Comparativa de Búsqueda")
ax1.set_xlabel("n")
ax1.set_ylabel("Tiempo (s)")
ax1.legend(); ax1.grid(True)

ax2.plot(n_ord_lento, resultados["Burbuja"], 'r-o', label="Burbuja O(n²)")
ax2.plot(n_ord_lento, resultados["Seleccion"], 'g-s', label="Selección O(n²)")
ax2.plot(n_ord_rapido, resultados["QuickSort"], 'b-d', label="QuickSort O(n log n)")
ax2.set_title("Comparativa de Ordenamiento")
ax2.set_xlabel("n")
ax2.set_ylabel("Tiempo (s)")
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.show()