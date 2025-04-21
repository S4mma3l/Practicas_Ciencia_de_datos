import tabula

# Specify the path to your PDF file
pdf_path = 'conversor\\archivo2.pdf'

# Specify the path where you want to save the CSV file
csv_path = 'conversor\\archivo2.csv'

try:
    # Use tabula to read tables from the PDF.
    # 'pages' can be 'all' or a specific page number (e.g., '1', '1,2,3').
    # 'output_format' can be 'csv', 'json', 'dataframe'. We want 'csv'.
    tabula.convert_into(pdf_path, csv_path, output_format='csv', pages='all')
    print(f"Successfully converted tables from '{pdf_path}' to '{csv_path}'")

except FileNotFoundError:
    print(f"Error: PDF file not found at '{pdf_path}'")
except Exception as e:
    print(f"An error occurred: {e}")