import numpy as np
from collections import Counter

# =====================================================================
# 1. CLASE: REGRESIÓN LINEAL (APRENDIZAJE SUPERVISADO - CONTINUO)
# =====================================================================
class RegresionLineal:
    def __init__(self, tasa_aprendizaje=0.01, iteraciones=1000):
        self.lr = tasa_aprendizaje
        self.iteraciones = iteraciones
        self.pesos = None
        self.sesgo = None

    def entrenar(self, X, y):
        muestras, caracteristicas = X.shape
        self.pesos = np.zeros(caracteristicas)
        self.sesgo = 0

        # Optimización por Descenso del Gradiente
        for _ in range(self.iteraciones):
            y_pred = np.dot(X, self.pesos) + self.sesgo
            
            # Derivadas parciales
            dw = (1 / muestras) * np.dot(X.T, (y_pred - y))
            db = (1 / muestras) * np.sum(y_pred - y)

            # Actualización de parámetros
            self.pesos -= self.lr * dw
            self.sesgo -= self.lr * db

    def predecir(self, X):
        return np.dot(X, self.pesos) + self.sesgo


# =====================================================================
# 2. CLASE: K-VECINOS MÁS CERCANOS (APRENDIZAJE SUPERVISADO - CLASIFICACIÓN)
# =====================================================================
def distancia_euclidiana(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def entrenar(self, X, y):
        # El algoritmo KNN es "lazy", solo memoriza los datos en esta fase
        self.X_train = X
        self.y_train = y

    def predecir(self, X):
        predicciones = [self._predecir_uno(x) for x in X]
        return np.array(predicciones)

    def _predecir_uno(self, x):
        distancias = [distancia_euclidiana(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distancias)[:self.k]
        k_etiquetas = [self.y_train[i] for i in k_indices]
        mas_comun = Counter(k_etiquetas).most_common(1)
        return mas_comun[0][0]


# =====================================================================
# 3. CLASE: K-MEANS (APRENDIZAJE NO SUPERVISADO - AGRUPAMIENTO)
# =====================================================================
class KMeans:
    def __init__(self, K=3, max_iter=100):
        self.K = K
        self.max_iter = max_iter
        self.centroides = []

    def entrenar(self, X):
        muestras, caracteristicas = X.shape
        # Inicialización aleatoria de centroides
        indices_aleatorios = np.random.choice(muestras, self.K, replace=False)
        self.centroides = X[indices_aleatorios]

        for _ in range(self.max_iter):
            clusters = self._crear_clusters(X)
            centroides_anteriores = self.centroides.copy()
            self.centroides = self._obtener_nuevos_centroides(clusters, X)
            
            # Condición de convergencia: los centroides ya no se mueven
            if np.all(centroides_anteriores == self.centroides):
                break
                
        return self._obtener_etiquetas_cluster(clusters, X)

    def _crear_clusters(self, X):
        clusters = [[] for _ in range(self.K)]
        for idx, muestra in enumerate(X):
            distancias = [np.linalg.norm(muestra - c) for c in self.centroides]
            centroide_mas_cercano = np.argmin(distancias)
            clusters[centroide_mas_cercano].append(idx)
        return clusters

    def _obtener_nuevos_centroides(self, clusters, X):
        nuevos_centroides = np.zeros((self.K, X.shape[1]))
        for idx_cluster, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
            media_cluster = np.mean(X[cluster], axis=0)
            nuevos_centroides[idx_cluster] = media_cluster
        return nuevos_centroides

    def _obtener_etiquetas_cluster(self, clusters, X):
        etiquetas = np.empty(X.shape[0])
        for idx_cluster, cluster in enumerate(clusters):
            for idx_muestra in cluster:
                etiquetas[idx_muestra] = idx_cluster
        return etiquetas


# =====================================================================
# 4. EJECUCIÓN, LÓGICA DE USO Y EVALUACIÓN DE RESULTADOS
# =====================================================================
if __name__ == "__main__":
    
    print("=========================================")
    print("1. PRUEBA DE REGRESIÓN LINEAL")
    print("=========================================")
    # LÓGICA: Queremos predecir un valor continuo basado en una variable.
    # Simulamos una función lineal con un poco de ruido: y = ~3x + 2
    X_reg = np.array([[1], [2], [3], [4], [5]])
    y_reg = np.array([5.1, 7.9, 11.2, 13.8, 17.1])
    
    # Instanciamos y entrenamos
    modelo_reg = RegresionLineal(tasa_aprendizaje=0.01, iteraciones=1000)
    modelo_reg.entrenar(X_reg, y_reg)
    
    # Evaluamos con un dato no visto (Extrapolación)
    X_nuevo = np.array([[6], [7]])
    prediccion_reg = modelo_reg.predecir(X_nuevo)
    
    print(f"Ecuación aprendida: y = {modelo_reg.pesos[0]:.2f}x + {modelo_reg.sesgo:.2f}")
    print(f"Predicción para X=6: {prediccion_reg[0]:.2f} (Esperado: ~20)")
    print(f"Predicción para X=7: {prediccion_reg[1]:.2f} (Esperado: ~23)\n")


    print("=========================================")
    print("2. PRUEBA DE K-VECINOS MÁS CERCANOS (KNN)")
    print("=========================================")
    # LÓGICA: Queremos clasificar nuevos elementos comparando su similitud con datos conocidos.
    # Formato: [Característica 1, Característica 2]
    # Supongamos que es [Grosor del tronco, Altura del árbol]
    # Clase 0: Arbustos pequeños | Clase 1: Pinos altos
    X_knn_train = np.array([
        [2, 10], [3, 12], [2.5, 11],  # Clase 0
        [15, 60], [18, 65], [16, 62]  # Clase 1
    ])
    y_knn_train = np.array([0, 0, 0, 1, 1, 1])
    
    # Instanciamos buscando los 3 vecinos más cercanos
    modelo_knn = KNN(k=3)
    modelo_knn.entrenar(X_knn_train, y_knn_train)
    
    # Evaluamos con datos mixtos para ver si el modelo los clasifica correctamente
    X_knn_test = np.array([
        [3, 15],   # Claramente un arbusto (Clase 0)
        [17, 63],  # Claramente un pino (Clase 1)
        [10, 35]   # Caso dudoso (El algoritmo calculará la distancia)
    ])
    predicciones_knn = modelo_knn.predecir(X_knn_test)
    
    for i, muestra in enumerate(X_knn_test):
        print(f"Datos de entrada {muestra} -> Clasificado como: Clase {predicciones_knn[i]}")
    print("\n")


    print("=========================================")
    print("3. PRUEBA DE K-MEANS (CLUSTERING)")
    print("=========================================")
    # LÓGICA: Tenemos un mar de datos sin etiquetas y queremos encontrar grupos naturales.
    # Generamos 6 puntos que visualmente caerían en dos esquinas distintas de un gráfico.
    X_kmeans = np.array([
        [1.0, 1.5], [1.2, 1.1], [0.8, 1.0],  # Grupo "A" (esquina inferior izquierda)
        [10.0, 10.5], [10.2, 9.8], [9.8, 10.1] # Grupo "B" (esquina superior derecha)
    ])
    
    # Instanciamos pidiéndole que encuentre 2 clusters
    modelo_kmeans = KMeans(K=2, max_iter=100)
    etiquetas_kmeans = modelo_kmeans.entrenar(X_kmeans)
    
    print("Coordenadas finales de los 2 centroides matemáticos:")
    for i, centroide in enumerate(modelo_kmeans.centroides):
        print(f"Centroide {i}: {centroide}")
        
    print("\nAsignación final de cada punto a un cluster:")
    for i, punto in enumerate(X_kmeans):
        # Nota: K-Means asigna etiquetas (0 o 1) aleatoriamente a los grupos, 
        # pero los puntos similares tendrán la misma etiqueta.
        print(f"Punto {punto} -> Pertenece al Cluster {int(etiquetas_kmeans[i])}")