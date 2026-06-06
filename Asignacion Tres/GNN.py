#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:54:36 2026

@author: meppadilla
"""

import torch
import torch.nn as nn
import numpy as np

# --- 1. Definición de la Topología ---
# Grafo de 3 nodos (0: Tablero, 1: Motor A, 2: Motor B)
# Conexiones: Tablero <-> Motor A, Motor A <-> Motor B
A = torch.tensor([[0., 1., 0.],
                  [1., 0., 1.],
                  [0., 1., 0.]])

# Añadimos auto-enlaces (Matriz Identidad) para que cada nodo recuerde su propio estado
I = torch.eye(3)
A_tilde = A + I

# --- 2. Normalización Simétrica ---
# Calculamos el grado de cada nodo (cuántas conexiones tiene, incluyendo el autoenlace)
grados = A_tilde.sum(dim=1)
# Matriz diagonal D_tilde^(-1/2)
D_tilde_inv_sqrt = torch.diag(torch.pow(grados, -0.5))

# Calculamos el operador de paso de mensajes: D^(-1/2) * A_tilde * D^(-1/2)
operador_convolucional = torch.matmul(torch.matmul(D_tilde_inv_sqrt, A_tilde), D_tilde_inv_sqrt)

print("Operador Convolucional Normalizado:")
print(np.round(operador_convolucional.numpy(), 3))

# --- 3. Datos de los Nodos (Matriz H) ---
# Supongamos 2 características por equipo: [Temperatura, Nivel de Vibración]
H_l = torch.tensor([[45.0, 0.1],  # Tablero (frío, sin vibración)
                    [80.0, 2.5],  # Motor A (caliente, vibrando mucho)
                    [50.0, 0.5]]) # Motor B (normal)

# --- 4. Parámetros Entrenables (Matriz W) ---
# Proyectamos de 2 características a 3 características latentes (ocultas)
W_l = nn.Parameter(torch.rand(2, 3))

# --- 5. APLICACIÓN DE LA ECUACIÓN (7) ---
# H^(l+1) = sigma( D^(-1/2) A_tilde D^(-1/2) H^(l) W^(l) )

# a. Paso de mensajes (Los nodos mezclan su estado con sus vecinos)
mezcla_vecinal = torch.matmul(operador_convolucional, H_l)

# b. Transformación lineal (Aprendizaje)
transformacion = torch.matmul(mezcla_vecinal, W_l)

# c. Activación no lineal (sigma)
H_l_mas_1 = torch.relu(transformacion)

print("\nMatriz de Características de Entrada H^(l):")
print(H_l)
print("\nNuevo Estado Oculto de los Nodos H^(l+1):")
print(H_l_mas_1.detach().numpy())