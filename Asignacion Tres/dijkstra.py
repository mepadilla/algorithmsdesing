import heapq

def dijkstra_optimizado(grafo, origen):
    # 1. Inicialización
    # Asignamos infinito a todos los nodos, excepto al origen que es 0
    distancias = {nodo: float('inf') for nodo in grafo}
    distancias[origen] = 0
    
    # Diccionario para reconstruir la ruta final
    predecesores = {nodo: None for nodo in grafo}
    
    # Cola de prioridad (Min-Heap). Almacena tuplas: (distancia_acumulada, nodo)
    cola_prioridad = [(0, origen)]
    
    while cola_prioridad:
        # 2. Extracción del Mínimo (O(log V))
        distancia_actual, nodo_actual = heapq.heappop(cola_prioridad)
        
        # Si extraemos una ruta obsoleta que ya fue mejorada, la ignoramos
        if distancia_actual > distancias[nodo_actual]:
            continue
            
        # 3. Fase de Relajación (Ecuación 2 del documento)
        for vecino, peso_arista in grafo[nodo_actual].items():
            distancia_calculada = distancia_actual + peso_arista
            
            # Condición: si d(v) > d(u) + w(u, v) => d(v) = d(u) + w(u, v)
            if distancia_calculada < distancias[vecino]:
                distancias[vecino] = distancia_calculada
                predecesores[vecino] = nodo_actual
                
                # Actualizamos la cola de prioridad con la nueva mejor distancia
                heapq.heappush(cola_prioridad, (distancia_calculada, vecino))
                
    return distancias, predecesores

def reconstruir_ruta(predecesores, origen, destino):
    ruta = []
    nodo_actual = destino
    while nodo_actual is not None:
        ruta.insert(0, nodo_actual)
        nodo_actual = predecesores[nodo_actual]
    return ruta if ruta[0] == origen else []

# --- Ejemplo de Uso Práctico ---

# Modelado de una red de canalizaciones (los pesos son metros de cable/tubería)
red_planta = {
    'Tablero Principal': {'Subestacion A': 12, 'Panel de Control': 18},
    'Subestacion A': {'Tablero Principal': 12, 'Motor (460V)': 45, 'Generador': 20},
    'Panel de Control': {'Tablero Principal': 18, 'Generador': 15, 'Motor (460V)': 22},
    'Generador': {'Subestacion A': 20, 'Panel de Control': 15, 'Motor (460V)': 10},
    'Motor (460V)': {'Subestacion A': 45, 'Panel de Control': 22, 'Generador': 10}
}

nodo_inicio = 'Tablero Principal'
nodo_fin = 'Motor (460V)'

distancias_minimas, arbol_predecesores = dijkstra_optimizado(red_planta, nodo_inicio)
ruta_optima = reconstruir_ruta(arbol_predecesores, nodo_inicio, nodo_fin)

print(f"Ruta óptima trazada: {' -> '.join(ruta_optima)}")
print(f"Longitud total de cableado requerido: {distancias_minimas[nodo_fin]} metros")