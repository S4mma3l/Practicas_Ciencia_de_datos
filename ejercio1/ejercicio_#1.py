import pandas as pd
import numpy as np

def analyze_lottery_numbers(csv_file):
    """
    Analiza un archivo CSV que contiene números de lotería para encontrar los números más frecuentes
    en cada columna 'No' y las fechas en que aparecieron.

    Args:
        csv_file (str): La ruta al archivo CSV.

    Returns:
        pandas.DataFrame: DataFrame que contiene los números más frecuentes, su frecuencia
                        y las fechas en que ocurrieron.
    """

    # 1. Leer el archivo CSV en un DataFrame de Pandas.
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {csv_file}")
        return None  # Retornar None para indicar un error

    # 2. Imprimir los nombres de las columnas y sus tipos de datos para inspección.
    print("Información inicial del DataFrame:")
    print(df.info())  # Útil para entender la estructura de los datos
    print("\nPrimeras 5 filas del DataFrame:")
    print(df.head().to_markdown(index=False, numalign="left", stralign="left"))  # Mostrar los datos

    # 3. Limpieza de datos: Manejar posibles datos no numéricos en las columnas 'No'.
    number_cols = ['No 1', 'No 2', 'No 3', 'No 4', 'No 5']
    for col in number_cols:
        # 3.1. Identificar e informar sobre los valores no numéricos (para depuración/información).
        non_numeric_values = df[pd.to_numeric(df[col], errors='coerce').isna()][col].unique()
        if (len(non_numeric_values) > 0):
            print(f"\nValores no numéricos encontrados en la columna '{col}': {non_numeric_values}")

        # 3.2. Limpieza robusta: Eliminar las filas con valores no numéricos en las columnas 'No'.
        df = df[pd.to_numeric(df[col], errors='coerce').notna()]

        # 3.3. Convertir la columna al tipo de dato numérico.
        df[col] = pd.to_numeric(df[col])

    print("\nDataFrame después de limpiar las columnas de números:")
    print(df.info())  # Mostrar los tipos de datos actualizados
    print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

    # 4. Función para encontrar los números más frecuentes y sus fechas.
    def find_most_frequent_with_dates(df, col_name):
        """
        Encuentra los números más frecuentes en una columna especificada y las fechas en que aparecieron.

        Args:
            df (pd.DataFrame): DataFrame que contiene los datos.
            col_name (str): Nombre de la columna a analizar (p. ej., 'No 1').

        Returns:
            pd.DataFrame: DataFrame con los números más frecuentes, su frecuencia y las fechas.
        """

        # 4.1. Calcular la frecuencia de cada número en la columna.
        number_counts = df[col_name].value_counts().reset_index()
        number_counts.columns = ['Number', 'Frequency']

        # 4.2. Encontrar la frecuencia máxima.
        max_frequency = number_counts['Frequency'].max()

        # 4.3. Filtrar los números con la frecuencia máxima.
        most_frequent_numbers = number_counts[number_counts['Frequency'] == max_frequency]

        # 4.4. Recopilar todas las fechas únicas en que apareció cada número más frecuente.
        result_data = []
        for number in most_frequent_numbers['Number']:
            dates = ', '.join(df[df[col_name] == number]['Fecha'].unique())  # Asume que existe la columna 'Fecha'
            result_data.append({'Column': col_name, 'Number': number, 'Frequency': max_frequency, 'Dates': dates})

        return pd.DataFrame(result_data)

    # 5. Aplicar la función de análisis a cada columna 'No' y concatenar los resultados.
    results = pd.concat([
        find_most_frequent_with_dates(df, 'No 1'),
        find_most_frequent_with_dates(df, 'No 2'),
        find_most_frequent_with_dates(df, 'No 3'),
        find_most_frequent_with_dates(df, 'No 4'),
        find_most_frequent_with_dates(df, 'No 5')
    ], ignore_index=True)

    return results


# 6. Ejemplo de uso (reemplazar 'your_file.csv' con el nombre real del archivo).
if __name__ == '__main__':
    csv_file_path = 'conversor\\archivo2.csv'
    analysis_results = analyze_lottery_numbers(csv_file_path)

    if analysis_results is not None:  # Solo imprimir si el análisis fue exitoso
        print("\nResultados del análisis de los números más frecuentes:")
        print(analysis_results.to_markdown(index=False, numalign="left", stralign="left"))