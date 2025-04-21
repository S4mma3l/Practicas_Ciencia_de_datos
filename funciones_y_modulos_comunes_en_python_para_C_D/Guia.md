# Funciones y Módulos Comunes en Python para Ciencia de Datos

Este documento presenta una visión general de las funciones y módulos más utilizados en Python para tareas de ciencia de datos. Están organizados por su área principal de aplicación, con ejemplos concisos para ilustrar su uso.

## 1. Manipulación y Análisis de Datos (Pandas)

Pandas es la biblioteca fundamental para la manipulación y el análisis de datos en Python. Proporciona estructuras de datos de alto rendimiento y fáciles de usar.

* **`pd.read_csv(filepath)`**: Lee datos desde un archivo de valores separados por comas (CSV) y los carga en un objeto `DataFrame`.

    ```python
    import pandas as pd

    # Leer un archivo CSV llamado 'datos.csv' ubicado en la misma carpeta
    df = pd.read_csv('datos.csv')
    print(df.head()) # Muestra las primeras 5 filas del DataFrame por defecto
    ```

* **`pd.DataFrame(data)`**: Crea un DataFrame a partir de diferentes tipos de datos como diccionarios de listas, listas de diccionarios, o incluso arrays de NumPy.

    ```python
    import pandas as pd

    data = {'Nombre': ['Alice', 'Bob', 'Charlie'],
            'Edad': [25, 30, 28],
            'Ciudad': ['San José', 'Heredia', 'Alajuela']}
    df = pd.DataFrame(data)
    print(df)
    ```

* **`df.head(n)` y `df.tail(n)`**: Muestran las primeras `n` filas y las últimas `n` filas del DataFrame respectivamente. Útil para inspeccionar los datos.

    ```python
    print(df.head(2)) # Muestra las primeras 2 filas
    print(df.tail())  # Muestra las últimas 5 filas por defecto
    ```

* **`df.info()`**: Proporciona un resumen informativo del DataFrame, incluyendo el número de filas, columnas, tipos de datos de cada columna y la cantidad de valores no nulos.

    ```python
    df.info()
    ```

* **`df.describe()`**: Genera estadísticas descriptivas de las columnas numéricas del DataFrame, como la media, desviación estándar, mínimo, máximo y los cuartiles.

    ```python
    print(df.describe())
    ```

* **`df['columna']` o `df.columna`**: Selecciona una única columna del DataFrame, devolviendo un objeto `Series` de Pandas.

    ```python
    edades = df['Edad']
    print(edades)
    ```

* **`df[['columna1', 'columna2']]`**: Selecciona múltiples columnas del DataFrame, devolviendo un nuevo `DataFrame` que contiene solo esas columnas.

    ```python
    subset = df[['Nombre', 'Ciudad']]
    print(subset)
    ```

* **`df.loc[filas, columnas]`**: Permite la selección de datos basada en etiquetas (nombres) de filas y columnas.

    ```python
    # Seleccionar la fila con la etiqueta de índice 0 y todas las columnas
    print(df.loc[0])
    # Seleccionar las filas con etiquetas de índice 0 y 1, y las columnas 'Nombre' y 'Edad'
    print(df.loc[[0, 1], ['Nombre', 'Edad']])
    ```

* **`df.iloc[indices_filas, indices_columnas]`**: Permite la selección de datos basada en índices enteros (posición) de filas y columnas.

    ```python
    # Seleccionar la primera fila (índice 0) y todas las columnas
    print(df.iloc[0])
    # Seleccionar las dos primeras filas (índices 0 y 1) y las dos primeras columnas (índices 0 y 1)
    print(df.iloc[0:2, 0:2])
    ```

* **`df.groupby('columna')`**: Agrupa las filas del DataFrame basándose en los valores únicos de una columna especificada. Esto es fundamental para realizar análisis agregados por categorías.

    ```python
    import pandas as pd

    data_ventas = {'Producto': ['A', 'B', 'A', 'C', 'B', 'C'],
                   'Cantidad': [10, 5, 12, 8, 6, 15]}
    df_ventas = pd.DataFrame(data_ventas)
    ventas_por_producto = df_ventas.groupby('Producto')['Cantidad'].sum()
    print(ventas_por_producto)
    ```

* **`df.sort_values(by='columna', ascending=True)`**: Ordena el DataFrame según los valores de una o más columnas. El argumento `ascending=False` ordena de forma descendente.

    ```python
    df_ordenado = df.sort_values(by='Edad', ascending=False)
    print(df_ordenado)
    ```

* **`df.dropna()`**: Elimina las filas que contienen al menos un valor faltante (NaN).

    ```python
    import pandas as pd

    data_nulos = {'Nombre': ['Alice', 'Bob', None], 'Edad': [25, None, 28]}
    df_nulos = pd.DataFrame(data_nulos)
    df_sin_nulos = df_nulos.dropna()
    print(df_sin_nulos)
    ```

* **`df.fillna(value)`**: Reemplaza todos los valores faltantes (NaN) con un valor especificado.

    ```python
    df_llenado = df_nulos.fillna(0)
    print(df_llenado)
    ```

* **`df.apply(función, axis=0)`**: Aplica una función a lo largo de un eje del DataFrame. `axis=0` aplica la función a cada columna, y `axis=1` la aplica a cada fila.

    ```python
    def aumentar_edad(edad):
        return edad + 1

    df['Edad_Nueva'] = df['Edad'].apply(aumentar_edad)
    print(df)
    ```

* **`df.merge(right, on='columna', how='inner')`**: Combina dos DataFrames basados en una o más columnas en común. El argumento `how` especifica el tipo de unión ('inner', 'outer', 'left', 'right').

    ```python
    import pandas as pd

    df1 = pd.DataFrame({'ID': [1, 2, 3], 'Nombre': ['Alice', 'Bob', 'Charlie']})
    df2 = pd.DataFrame({'ID': [2, 3, 4], 'Ciudad': ['Heredia', 'Alajuela', 'Cartago']})
    df_combinado = pd.merge(df1, df2, on='ID', how='inner')
    print(df_combinado)
    ```

## 2. Computación Numérica (NumPy)

NumPy es la biblioteca fundamental para la computación numérica en Python. Proporciona soporte para arrays multidimensionales y una gran colección de funciones matemáticas de alto nivel.

* **`np.array(object)`**: Crea un array de NumPy a partir de un objeto iterable como una lista o tupla.

    ```python
    import numpy as np

    lista = [1, 2, 3, 4, 5]
    array_np = np.array(lista)
    print(array_np)
    ```

* **`np.arange(start, stop, step)`**: Crea un array de valores espaciados uniformemente dentro de un intervalo definido por `start` (inicio), `stop` (fin, no incluido) y `step` (incremento).

    ```python
    array_rango = np.arange(0, 10, 2) # Resultado: [0 2 4 6 8]
    print(array_rango)
    ```

* **`np.linspace(start, stop, num)`**: Crea un array con `num` números espaciados uniformemente dentro del intervalo [`start`, `stop`].

    ```python
    array_lineal = np.linspace(0, 1, 5) # Resultado: [0.   0.25 0.5  0.75 1.  ]
    print(array_lineal)
    ```

* **`np.zeros(shape)` y `np.ones(shape)`**: Crean arrays de una forma especificada (`shape`) llenos de ceros o unos respectivamente. `shape` es una tupla que define las dimensiones del array (ej., `(2, 3)` para una matriz de 2x3).

    ```python
    zeros_array = np.zeros((2, 3))
    ones_array = np.ones((3,))
    print(zeros_array)
    print(ones_array)
    ```

* **`np.random.rand(d0, d1, ..., dn)`**: Crea un array de la forma dada con números aleatorios distribuidos uniformemente entre 0 y 1.

    ```python
    random_array = np.random.rand(2, 2)
    print(random_array)
    ```

* **Operaciones aritméticas elementales (`+`, `-`, `*`, `/`, `**`)**: Cuando se aplican a arrays de NumPy, estas operaciones se realizan elemento a elemento.

    ```python
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(a + b)   # Resultado: [5 7 9]
    print(a * 2)   # Resultado: [2 4 6]
    ```

* **Funciones universales (`np.sin()`, `np.cos()`, `np.exp()`, `np.log()`):** Estas funciones matemáticas se aplican también elemento a elemento a los arrays de NumPy.

    ```python
    angulos = np.array([0, np.pi/2, np.pi])
    senos = np.sin(angulos)
    print(senos)
    ```

* **`array.shape`**: Atributo que devuelve una tupla indicando las dimensiones del array.
* **`array.reshape(new_shape)`**: Método que devuelve un nuevo array con la misma data pero con una forma diferente. Es importante que el número total de elementos permanezca constante.
* **`np.mean(array)`, `np.median(array)`, `np.std(array)`, `np.sum(array)`**: Funciones para calcular la media, mediana, desviación estándar y suma de los elementos de un array, respectivamente.

## 3. Visualización de Datos (Matplotlib y Seaborn)

Estas bibliotecas son esenciales para crear visualizaciones informativas y atractivas de los datos. Matplotlib proporciona la base, mientras que Seaborn se construye sobre Matplotlib y ofrece una interfaz de alto nivel con estilos predefinidos y gráficos estadísticos más avanzados.

* **`plt.plot(x, y)`**: Crea un gráfico de líneas, donde `x` son los valores del eje horizontal e `y` los valores del eje vertical.

    ```python
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 1, 3, 5])
    plt.plot(x, y)
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.title('Gráfico de Líneas')
    plt.show()
    ```

* **`plt.scatter(x, y)`**: Crea un diagrama de dispersión, útil para visualizar la relación entre dos variables.

    ```python
    plt.scatter(x, y)
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.title('Diagrama de Dispersión')
    plt.show()
    ```

* **`plt.hist(x, bins)`**: Crea un histograma, que muestra la distribución de frecuencia de una variable. `bins` especifica el número de contenedores a utilizar.

    ```python
    datos = np.random.randn(1000)
    plt.hist(datos, bins=30)
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.title('Histograma')
    plt.show()
    ```

* **`sns.histplot(data, x='columna', kde=False)`**: Crea un histograma con Seaborn. El argumento `kde=True` añade una estimación de densidad del kernel.

    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    data = {'Edad': np.random.randint(18, 65, 100)}
    df = pd.DataFrame(data)
    sns.histplot(df, x='Edad', kde=True)
    plt.title('Distribución de Edades')
    plt.show()
    ```

* **`sns.scatterplot(data, x='columna1', y='columna2')`**: Crea un diagrama de dispersión utilizando Seaborn, a menudo con opciones de estilo y coloración más avanzadas.
* **`sns.boxplot(data, x='categoria', y='valor')`**: Crea un diagrama de caja (boxplot), útil para comparar la distribución de una variable numérica entre diferentes categorías.
* **`plt.xlabel()`, `plt.ylabel()`, `plt.title()`, `plt.legend()`, `plt.show()`**: Funciones de Matplotlib para añadir etiquetas a los ejes, un título al gráfico, una leyenda y para mostrar el gráfico, respectivamente.

## 4. Aprendizaje Automático (Scikit-learn)

Scikit-learn es una biblioteca integral para tareas de aprendizaje automático, proporcionando herramientas para clasificación, regresión, clustering, reducción de dimensionalidad, selección de modelos y preprocesamiento.

* **Modelos (ejemplos):** Scikit-learn incluye una amplia variedad de algoritmos de aprendizaje automático.

    ```python
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    ```

* **Preprocesamiento:** Módulos para escalar datos, manejar valores faltantes y dividir conjuntos de datos.

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.impute import SimpleImputer
    ```

* **Evaluación:** Métricas y funciones para evaluar el rendimiento de los modelos.

    ```python
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
    ```

**Ejemplo sencillo de Scikit-learn (Regresión Lineal):**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Datos de ejemplo
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Dividir los datos en conjuntos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo utilizando los datos de entrenamiento
modelo.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el rendimiento del modelo utilizando el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
print(f"Error Cuadrático Medio: {mse}")
print(f"Predicciones en el conjunto de prueba: {y_pred}")