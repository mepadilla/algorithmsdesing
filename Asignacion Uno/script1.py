#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:25:41 2026

@author: meppadilla
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# Optimización de entorno
sys.setrecursionlimit(5000)

# --- Algoritmos de Búsqueda ---
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

# --- Funciones de Ajuste para Notación Big O ---
def func_lineal(n, a, b): return a * n + b
def func_cuadratica(n, a, b): return a * (n**2) + b
def func_logaritmica(n, a, b): return a * np.log2(n + 1) + b

def estimar_big_o(n_vals, t_vals, dominio="busqueda"):
    n_pts = np.array(n_vals)
    t_pts = np.array(t_vals)
    try:
        popt_lin, _ = curve_fit(func_lineal, n_pts, t_pts)
        err_lin = np.sum((func_lineal(n_pts, *popt_lin) - t_pts)**2)
        
        if dominio == "busqueda":
            popt_log, _ = curve_fit(func_logaritmica, n_pts, t_pts)
            err_log = np.sum((func_logaritmica(n_pts, *popt_log) - t_pts)**2)
            return "O(log n)" if err_log < err_lin else "O(n)"
        else:
            popt_cua, _ = curve_fit(func_cuadratica, n_pts, t_pts)
            err_cua = np.sum((func_cuadratica(n_pts, *popt_cua) - t_pts)**2)
            return "O(n^2)" if err_cua < err_lin else "O(n)"
    except:
        return "Indeterminado (Ruido en datos)"

# --- Configuración Incrementada ---
# Valores para búsqueda (Hasta 2 millones)
n_busqueda = [100, 1000, 10000, 100000, 500000, 1000000, 2000000]
# Valores para ordenamiento (Subimos a 10,000)
n_ordenamiento = [10, 100, 500, 1000, 2500, 5000, 7500, 10000]

resultados = {
    "Lineal_Des": [], "Lineal_Ord": [], "Binaria": [],
    "Burbuja": [], "Seleccion": []
}

print("Iniciando pruebas intensivas...")

# Benchmarking de Búsqueda
for n in n_busqueda:
    print(f"Probando búsqueda para n = {n}...")
    data = [random.randint(0, n*10) for _ in range(n)]
    target = -7 # Un valor que probablemente no exista para forzar el peor caso
    
    t0 = time.perf_counter() # Mayor precisión que time.time()
    busqueda_lineal(data, target)
    resultados["Lineal_Des"].append(time.perf_counter() - t0)
    
    data.sort()
    
    t0 = time.perf_counter()
    busqueda_lineal(data, target)
    resultados["Lineal_Ord"].append(time.perf_counter() - t0)
    
    t0 = time.perf_counter()
    busqueda_binaria(data, target)
    resultados["Binaria"].append(time.perf_counter() - t0)

# Benchmarking de Ordenamiento
for n in n_ordenamiento:
    print(f"Probando ordenamiento para n = {n}...")
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

# --- Reporte Final ---
print("\n" + "="*40)
print(" RESULTADOS DE ANÁLISIS ASINTÓTICO")
print("="*40)
print(f"Búsqueda Lineal:   {estimar_big_o(n_busqueda, resultados['Lineal_Des'], 'busqueda')}")
print(f"Búsqueda Binaria:  {estimar_big_o(n_busqueda, resultados['Binaria'], 'busqueda')}")
print(f"Método Burbuja:    {estimar_big_o(n_ordenamiento, resultados['Burbuja'], 'orden')}")
print(f"Método Selección:  {estimar_big_o(n_ordenamiento, resultados['Seleccion'], 'orden')}")

# --- Gráficos ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(n_busqueda, resultados["Lineal_Des"], 'o-', label="Lineal (Des)")
ax1.plot(n_busqueda, resultados["Lineal_Ord"], 's-', label="Lineal (Ord)")
ax1.plot(n_busqueda, resultados["Binaria"], 'd-', label="Binaria")
ax1.set_title("Búsqueda (Peor Caso)")
ax1.set_xlabel("n")
ax1.set_ylabel("Tiempo (s)")
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.legend()
ax1.grid(True)

ax2.plot(n_ordenamiento, resultados["Burbuja"], 'r-o', label="Burbuja")
ax2.plot(n_ordenamiento, resultados["Seleccion"], 'g-s', label="Selección")
ax2.set_title("Ordenamiento (Cuadrático)")
ax2.set_xlabel("n")
ax2.set_ylabel("Tiempo (s)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()