import pandas as pd

# Leer el archivo CSV en un DataFrame
df = pd.read_csv('funciones_y_modulos_comunes_en_python_para_C_D\\tienda_1 .csv')

# Mostrar las primeras 5 filas
print("Primeras 5 filas:")
print(df.head())

# Obtener información general del DataFrame
print("\nInformación del DataFrame:")
print(df.info())

# Estadísticas descriptivas de las columnas numéricas
print("\nEstadísticas Descriptivas:")
print(df.describe())

# Convertir la columna 'fecha' al tipo datetime
df['Fecha de Compra'] = pd.to_datetime(df['Fecha de Compra'])

# Calcular el total de ventas
total_ventas = df['Precio'].sum()
print(f"\nTotal de ventas: {total_ventas}")

# Agrupar por fecha y calcular el total de ventas por día
ventas_por_dia = df.groupby('Fecha de Compra')['Precio'].sum()
print("\nVentas por día:")
print(ventas_por_dia)

# Encontrar el producto más vendido
producto_mas_vendido = df['Producto'].value_counts().idxmax()
cantidad_mas_vendido = df['Producto'].value_counts().max()
print(f"\nProducto más vendido: {producto_mas_vendido} (Cantidad: {cantidad_mas_vendido})")

# Agregar una columna con el precio total por fila (precio * cantidad)
df['precio_total'] = df['Precio'] * df['Cantidad de cuotas']  # ¡Aquí está la corrección!

# Filtrar las ventas donde la cantidad es mayor a 1
ventas_mayores_a_uno = df[df['Cantidad de cuotas'] > 1]  # También corregí aquí, asumiendo que 'Cantidad de cuotas' representa la cantidad
print("\nVentas con cantidad mayor a 1:")
print(ventas_mayores_a_uno)

# FILTRAR PRODUCTOS DE UNA CATEGORIA ESPECIFICA
categoria_deseada = 'Electrónicos'
electronicos_df = df[df['Categoría del Producto'] == categoria_deseada]
print(f"Productos de la categoría '{categoria_deseada}':")
print(electronicos_df.head())

# FILTRAR VENTAS EN UN RANGO DE PRECIOS
precio_minimo = 100000
precio_maximo = 500000
rango_precios_df = df[(df['Precio'] >= precio_minimo) & (df['Precio'] <= precio_maximo)]
print(f"\nVentas con precios entre {precio_minimo} y {precio_maximo}:")
print(rango_precios_df.head())

# FILTRAR VENTAS REALIZADAS POR UN VENDEDOR ESPECIFICO
vendedor_deseado = 'Maria Alfonso'
ventas_vendedor_df = df[df['Vendedor'] == vendedor_deseado]
print(f"\nVentas realizadas por '{vendedor_deseado}':")
print(ventas_vendedor_df.head())

# FILTRAR VENTAS REALIZADAS EN CIERTAS CIUDADES
ciudades_deseadas = ['Bogotá', 'Medellín']
ventas_ciudades_df = df[df['Lugar de Compra'].isin(ciudades_deseadas)]
print(f"\nVentas realizadas en {ciudades_deseadas}:")
print(ventas_ciudades_df.head())

# FILTRAR VENTAS REALIZADAS EN UN RANGO DE FECHAS
fecha_inicio = '2023-01-01'
fecha_fin = '2023-12-31'
ventas_fechas_df = df[(df['Fecha de Compra'] >= fecha_inicio) & (df['Fecha de Compra'] <= fecha_fin)]
print(f"\nVentas realizadas entre {fecha_inicio} y {fecha_fin}:")
print(ventas_fechas_df.head())

# AGREGACIONES Y CALCULOS
    # CALCULAR EL PRECIO PROMEDIO POR CATEGORIA DE PRODUCTO
promedio_precio_categoria = df.groupby('Categoría del Producto')['Precio'].mean()
print("\nPrecio promedio por categoría de producto:")
print(promedio_precio_categoria)

    # CALCULAR EL TOTAL DE VENTAS POR VENDEDOR
ventas_por_vendedor = df.groupby('Vendedor')['Precio'].sum()
print("\nTotal de ventas por vendedor:")
print(ventas_por_vendedor)

    # ENCONTRAR EL PRODUCTO MAS CARO POR CATEGORIA
producto_mas_caro_categoria = df.loc[df.groupby('Categoría del Producto')['Precio'].idxmax()]
print("\nProducto más caro por categoría:")
print(producto_mas_caro_categoria[['Categoría del Producto', 'Producto', 'Precio']])

    # CALCULAR EL COSTO DE ENVIO PROMEDIO
costo_envio_promedio = df['Costo de envío'].mean()
print(f"\nCosto de envío promedio: {costo_envio_promedio}")

# MANIPULACION DE COLUMNAS
    # CREAR UNA COLUMNA CON EL MES DE COMPRA
df['Mes de Compra'] = df['Fecha de Compra'].dt.month
print("\nDataFrame con la columna 'Mes de Compra':")
print(df[['Fecha de Compra', 'Mes de Compra']].head())

    # CREAR UNA COLUMNA CON EL DIA DE LA SEMANA DE LA COMPRA
nombres_dias = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
df['Día de la Semana'] = df['Fecha de Compra'].dt.dayofweek.map(nombres_dias)
print("\nDataFrame con la columna 'Día de la Semana':")
print(df[['Fecha de Compra', 'Día de la Semana']].head())  

    # APLICAR UNA FUNCION A UNA COLUMNA(POR EJEPLO, FORMATEAR EL PRECIO):
def formatear_precio(precio):
    return f"${precio:,.2f}"

df['Precio Formateado'] = df['Precio'].apply(formatear_precio)
print("\nDataFrame con la columna 'Precio Formateado':")
print(df[['Precio', 'Precio Formateado']].head())

# ORDENAMINETO DE DATOS
    # ORDENAR EL DATAFRAME POR PRECIO DE FORMA DESCENDENTE
df_ordenado_precio = df.sort_values(by='Precio', ascending=False)
print("\nDataFrame ordenado por precio (descendente):")
print(df_ordenado_precio.head())

    # ORDENAR EL DATAFRAME POR FECHA DE COMPRA:
df_ordenado_fecha = df.sort_values(by='Fecha de Compra')
print("\nDataFrame ordenado por fecha de compra:")
print(df_ordenado_fecha.head(10))

# VALORES UNICOS Y CONTEO
    # OBTENER LOS VALORES UNICOS DE LA COLUMNA 'LUGAR DE COMPRA'
lugares_unicos = df['Lugar de Compra'].unique()
print("\nLugares de compra únicos:")
print(lugares_unicos)

    # CONTAR LA FRECUENCIA DE CADA CATEGORIA DE PRODUCTO
conteo_categorias = df['Categoría del Producto'].value_counts()
print("\nConteo de cada categoría de producto:")
print(conteo_categorias)