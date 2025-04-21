import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def analyze_lottery_numbers(csv_file):
    """
    Analiza un archivo CSV que contiene números de lotería para encontrar los números más frecuentes,
    las fechas en que aparecieron y los sorteos con la misma combinación.

    Args:
        csv_file (str): La ruta al archivo CSV.

    Returns:
        tuple: Una tupla que contiene:
            - pandas.DataFrame: DataFrame con los números más frecuentes, su frecuencia y las fechas.
            - pandas.DataFrame: DataFrame con los sorteos que tienen la misma combinación.
            - pandas.DataFrame: El DataFrame original con los datos de los sorteos.
    """

    # 1. Leer el archivo CSV en un DataFrame de Pandas.
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {csv_file}")
        return None, None, None  # Retornar None, None, None para indicar un error

    # 2. Imprimir los nombres de las columnas y sus tipos de datos para inspección.
    print("Información inicial del DataFrame:")
    print(df.info())
    print("\nPrimeras 5 filas del DataFrame:")
    print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

    # 3. Limpieza de datos: Manejar posibles datos no numéricos en las columnas 'No'.
    number_cols = ['No 1', 'No 2', 'No 3', 'No 4', 'No 5']
    for col in number_cols:
        non_numeric_values = df[pd.to_numeric(df[col], errors='coerce').isna()][col].unique()
        if (len(non_numeric_values) > 0):
            print(f"\nValores no numéricos encontrados en la columna '{col}': {non_numeric_values}")
        df = df[pd.to_numeric(df[col], errors='coerce').notna()]
        df[col] = pd.to_numeric(df[col])

    print("\nDataFrame después de limpiar las columnas de números:")
    print(df.info())
    print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

    # 4. Función para encontrar los números más frecuentes y sus fechas.
    def find_most_frequent_with_dates(df, col_name):
        number_counts = df[col_name].value_counts().reset_index()
        number_counts.columns = ['Number', 'Frequency']
        max_frequency = number_counts['Frequency'].max()
        most_frequent_numbers = number_counts[number_counts['Frequency'] == max_frequency]
        result_data = []
        for number in most_frequent_numbers['Number']:
            dates = ', '.join(df[df[col_name] == number]['Fecha'].unique())
            result_data.append({'Column': col_name, 'Number': number, 'Frequency': max_frequency, 'Dates': dates})
        return pd.DataFrame(result_data)

    # 5. Función para encontrar sorteos con la misma combinación de números.
    def find_same_number_combinations(df):
        """
        Encuentra sorteos que tienen la misma combinación de números.

        Args:
            df (pd.DataFrame): DataFrame con los datos de los sorteos.

        Returns:
            pandas.DataFrame: DataFrame con los sorteos que tienen la misma combinación.
        """

        # 5.1. Ordenar los números de cada sorteo para que el orden no afecte la comparación.
        df['Sorted Combination'] = df[number_cols].apply(lambda row: tuple(sorted(row)), axis=1)

        # 5.2. Contar la frecuencia de cada combinación ordenada.
        combination_counts = df['Sorted Combination'].value_counts()

        # 5.3. Filtrar las combinaciones que aparecen más de una vez (duplicados).
        duplicate_combinations = combination_counts[combination_counts > 1].index

        # 5.4. Crear un DataFrame con los sorteos que tienen combinaciones duplicadas.
        same_combination_data = []
        for comb in duplicate_combinations:
            same_sorteos = ', '.join(df[df['Sorted Combination'] == comb]['Sorteo'].unique())
            same_combination_data.append({'Combination': comb, 'Sorteos': same_sorteos, 'Frequency': combination_counts[comb]})

        return pd.DataFrame(same_combination_data)

    # 6. Aplicar las funciones de análisis.
    results = pd.concat([
        find_most_frequent_with_dates(df, 'No 1'),
        find_most_frequent_with_dates(df, 'No 2'),
        find_most_frequent_with_dates(df, 'No 3'),
        find_most_frequent_with_dates(df, 'No 4'),
        find_most_frequent_with_dates(df, 'No 5')
    ], ignore_index=True)

    same_combinations = find_same_number_combinations(df)

    return results, same_combinations, df  # Return the original DataFrame as well


def build_and_train_neural_network(df):  # Now takes df as an argument
    """
    Construye y entrena una red neuronal para "predecir" la próxima combinación de números.

    Args:
        df (pd.DataFrame): DataFrame con los datos de los sorteos.

    Returns:
        tuple: Una tupla que contiene:
            - sklearn.neural_network.MLPRegressor: El modelo entrenado.
            - sklearn.preprocessing.StandardScaler: El escalador utilizado para las características.
            - sklearn.preprocessing.StandardScaler: El escalador utilizado para los números de lotería (y).
    """

    # 1. Preparar los datos para el entrenamiento.
    number_cols = ['No 1', 'No 2', 'No 3', 'No 4', 'No 5']
    X = df.index.values.reshape(-1, 1)  # Usar el índice del sorteo como característica (muy limitado)
    y = df[number_cols].values  # Números de lotería como objetivo

    # 2. Escalar los datos (importante para las redes neuronales).
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    # 3. Dividir los datos en conjuntos de entrenamiento y prueba.
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # 4. Construir y entrenar la red neuronal.
    model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluar el modelo (esto es muy importante, pero las métricas pueden ser engañosas aquí).
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nMean Squared Error del modelo: {mse}")

    return model, scaler_X, scaler_y  # Devolver scaler_X también


def predict_next_combination(model, scaler_X, scaler_y, last_sorteo_index):
    """
    "Predice" la próxima combinación de números usando el modelo entrenado.

    Args:
        model (sklearn.neural_network.MLPRegressor): El modelo entrenado.
        scaler_X (sklearn.preprocessing.StandardScaler): El escalador para las características.
        scaler_y (sklearn.preprocessing.StandardScaler): El escalador para los números de lotería.
        last_sorteo_index (int): El índice del último sorteo.

    Returns:
        numpy.ndarray: La "predicción" de la próxima combinación de números.
    """

    # 1. Crear la característica para el próximo sorteo.
    next_sorteo = np.array([[last_sorteo_index + 1]])
    next_sorteo_scaled = scaler_X.transform(next_sorteo)  # Usar el scaler_X entrenado

    # 2. Realizar la predicción.
    next_combination_scaled = model.predict(next_sorteo_scaled)

    # 3. Desescalar la predicción.
    next_combination = scaler_y.inverse_transform(next_combination_scaled)

    return next_combination


# 7. Ejemplo de uso.
if __name__ == '__main__':
    csv_file_path = 'conversor\\archivo2.csv'
    analysis_results, same_combinations, df = analyze_lottery_numbers(csv_file_path)  # Get df

    if analysis_results is not None:
        print("\nResultados del análisis de los números más frecuentes:")
        print(analysis_results.to_markdown(index=False, numalign="left", stralign="left"))

    if same_combinations is not None and not same_combinations.empty:
        print("\nSorteos con la misma combinación:")
        print(same_combinations.to_markdown(index=False, numalign="left", stralign="left"))
    else:
        print("\nNo se encontraron sorteos con la misma combinación.")

    if analysis_results is not None:
        model, scaler_X, scaler_y = build_and_train_neural_network(df)  # Pass df
        last_sorteo_index = df.index[-1]
        next_combination = predict_next_combination(model, scaler_X, scaler_y, last_sorteo_index)
        print("\nPredicción de la próxima combinación (¡con precaución!):")
        print(next_combination)