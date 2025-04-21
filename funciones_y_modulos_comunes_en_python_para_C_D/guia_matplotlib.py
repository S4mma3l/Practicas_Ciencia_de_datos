import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo CSV
df = pd.read_csv('funciones_y_modulos_comunes_en_python_para_C_D\\tienda_1 .csv')

# Imprimir los nombres de las columnas para verificar
print("Nombres de las columnas en el DataFrame:")
print(df.columns)

# Convertir la columna 'Fecha de Compra' al tipo datetime
df['Fecha de Compra'] = pd.to_datetime(df['Fecha de Compra'])

# 1. Gráfico de Líneas: Ventas Diarias (Ya existente)
# Muestra la tendencia de las ventas a lo largo del tiempo.
ventas_por_dia = df.groupby('Fecha de Compra')['Precio'].sum()
plt.figure(figsize=(10, 6))
plt.plot(ventas_por_dia.index, ventas_por_dia.values, marker='o', linestyle='-')
plt.title('Ventas Diarias')
plt.xlabel('Fecha de Compra')
plt.ylabel('Total Ventas')
plt.grid(True)
plt.show()

# 2. Gráfico de Barras: Cantidad de Productos Vendidos (Ya existente)
# Compara la popularidad de diferentes productos según la cantidad vendida.
conteo_productos = df['Producto'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(conteo_productos.index, conteo_productos.values)
plt.title('Cantidad de Productos Vendidos')
plt.xlabel('Producto')
plt.ylabel('Cantidad')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. Gráfico de Dispersión: Precio vs. Calificación
# Examina si existe alguna relación entre el precio de un producto y la calificación que le dan los compradores.
plt.figure(figsize=(8, 6))
plt.scatter(df['Precio'], df['Calificación'], alpha=0.5) # 'alpha' controla la transparencia de los puntos
plt.title('Precio vs. Calificación del Producto')
plt.xlabel('Precio')
plt.ylabel('Calificación')
plt.grid(True)
plt.show()

# 4. Histograma: Distribución de Precios
# Muestra la distribución de frecuencia de los precios de los productos.
plt.figure(figsize=(8, 6))
plt.hist(df['Precio'], bins=20, edgecolor='black') # 'bins' define el número de barras
plt.title('Distribución de Precios de los Productos')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.show()

# 5. Gráfico de Pastel: Distribución de Métodos de Pago
# Muestra la proporción de cada método de pago utilizado por los compradores.
conteo_metodos_pago = df['Método de pago'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(conteo_metodos_pago, labels=conteo_metodos_pago.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribución de Métodos de Pago')
plt.axis('equal') # Asegura que el pastel sea un círculo.
plt.show()

# 6. Gráfico de Barras Apiladas: Ventas por Categoría y Año
# Compara las ventas de diferentes categorías de productos a lo largo de los años.
df['Año de Compra'] = df['Fecha de Compra'].dt.year
ventas_categoria_año = df.groupby(['Año de Compra', 'Categoría del Producto'])['Precio'].sum().unstack()
ventas_categoria_año.plot(kind='bar', stacked=True, figsize=(12, 7))
plt.title('Ventas por Categoría de Producto por Año')
plt.xlabel('Año de Compra')
plt.ylabel('Total Ventas')
plt.xticks(rotation=0)
plt.legend(title='Categoría')
plt.tight_layout()
plt.show()

# 7. Gráfico de Caja (Boxplot): Precio por Categoría de Producto
# Compara la distribución de precios entre diferentes categorías de productos, mostrando la mediana, cuartiles y valores atípicos.
plt.figure(figsize=(10, 7))
df.boxplot(column='Precio', by='Categoría del Producto')
plt.title('Distribución de Precios por Categoría de Producto')
plt.ylabel('Precio')
plt.xlabel('Categoría del Producto')
plt.suptitle('') # Elimina el título automático generado por pandas
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()