import tabula

# especificar la ruta del archivo pdf
pdf_path = 'conversor\\archivo2.pdf'

# especificar la ruta donde se guardara el archivo csv
csv_path = 'conversor\\archivo2.csv'

try:
    # se usa tablula para leer los PDF.
    # 'paginas' puedes ser 'all' o especificar el numero de paginas (e.g., '1', '1,2,3').
    # 'formato de salida' puede ser 'csv', 'json', 'dataframe'. nosotros queremos 'csv'.
    tabula.convert_into(pdf_path, csv_path, output_format='csv', pages='all')
    print(f"Conversion exitosa para el archivo '{pdf_path}' a '{csv_path}'")

except FileNotFoundError:
    print(f"Error: el archivo PDF no funciona '{pdf_path}'")
except Exception as e:
    print(f"a ocurrido un error: {e}")
