import pandas as pd
import numpy as np
import os

# Construir la ruta de forma más segura
ruta_archivo = os.path.join('funciones_y_modulos_comunes_en_python_para_C_D', 'tienda_1 .csv') # Asegúrate que 'tienda_1 .csv' sea el nombre real

# Leer el archivo CSV
try:
    df = pd.read_csv(ruta_archivo)
    # El resto de tu código aquí...
    precios = df['Precio'].to_numpy() # Convierte la columna 'Precio' del DataFrame a un array de NumPy para operaciones numéricas eficientes.

    # Calcular el promedio de los precios usando la función mean() de NumPy.
    promedio_precios = np.mean(precios)
    print(f"Promedio de precios: {promedio_precios}")

    # Encontrar el precio máximo y mínimo en el array de precios usando las funciones max() y min() de NumPy.
    precio_maximo = np.max(precios)
    precio_minimo = np.min(precios)
    print(f"Precio máximo: {precio_maximo}, Precio mínimo: {precio_minimo}")

    # Calcular la desviación estándar de los precios usando la función std() de NumPy, que mide la dispersión de los valores.
    desviacion_estandar = np.std(precios)
    print(f"Desviación estándar de los precios: {desviacion_estandar}")

    # Crear arrays de NumPy para 'Precio' y 'Cantidad de cuotas' y multiplicarlos elemento a elemento para obtener el precio total por fila.
    precio_array = df['Precio'].to_numpy()
    cantidad_cuotas_array = df['Cantidad de cuotas'].to_numpy()
    precio_total_array = precio_array * cantidad_cuotas_array
    print("\nArray de precio total por fila (Precio * Cantidad de cuotas):")
    print(precio_total_array)

    # Ejemplo de creación de un array de booleanos basado en una condición del array de precios.
    precios_altos = precios > np.mean(precios)
    print("\nArray booleano indicando precios mayores que el promedio:")
    print(precios_altos)

    # Ejemplo de filtrado del array de precios usando el array booleano.
    precios_solo_altos = precios[precios_altos]
    print("\nSolo los precios mayores que el promedio:")
    print(precios_solo_altos)

    # Ejemplo de cálculo del producto punto (dot product) - aunque no directamente relevante para este dataset, es una función común de NumPy.
    vector_a = np.array([1, 2, 3])
    vector_b = np.array([4, 5, 6])
    producto_punto = np.dot(vector_a, vector_b)
    print(f"\nProducto punto de [1, 2, 3] y [4, 5, 6]: {producto_punto}")

    # Ejemplo de creación de una matriz (array bidimensional) de precios y cantidades.
    matriz_precios_cantidades = np.array([df['Precio'].to_numpy(), df['Cantidad de cuotas'].to_numpy()]).T # .T para transponer y tener filas como [precio, cantidad]
    print("\nMatriz de Precios y Cantidades (transpuesta):")
    print(matriz_precios_cantidades[:5]) # Mostrar solo las primeras 5 filas

    # Ejemplo de cálculo de la suma de todos los elementos de la matriz.
    suma_total_matriz = np.sum(matriz_precios_cantidades)
    print(f"\nSuma total de todos los precios y cantidades: {suma_total_matriz}")

except FileNotFoundError:
    print(f"Error: No se pudo encontrar el archivo en la ruta: {ruta_archivo}")
except KeyError as e:
    print(f"Error de clave: La columna '{e}' no se encontró en el DataFrame. Verifica los nombres de las columnas.")