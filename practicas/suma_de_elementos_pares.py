def sumar_pares(lista_numeros):
  """
  Esta función toma una lista de números enteros y devuelve la suma de todos los números pares.

  Args:
    lista_numeros (list): Una lista de números enteros.

  Returns:
    int: La suma de los números pares en la lista.
  """
  suma = 0  # Inicializamos una variable para almacenar la suma de los números pares.
  for numero in lista_numeros:  # Iteramos sobre cada número en la lista.
    if numero % 2 == 0:      # Verificamos si el número actual es par (el resto de la división por 2 es 0).
      suma = suma + numero   # Si el número es par, lo añadimos a la variable 'suma'.
  return suma  # Devolvemos la suma total de los números pares encontrados.

# Solicitamos al usuario que introduzca una lista de números separados por comas y la guardamos como una cadena.
entrada_usuario = input("Introduce una lista de números separados por comas: ")

# Dividimos la cadena de entrada en una lista de subcadenas utilizando la coma como delimitador.
string_numeros = entrada_usuario.split(',')

# Convertimos cada subcadena en un entero y creamos una nueva lista de números enteros.
numeros = [int(num.strip()) for num in string_numeros]
# El método .strip() se usa para eliminar espacios en blanco alrededor de cada número.

# Llamamos a la función para sumar los números pares de la lista.
resultado = sumar_pares(numeros)

# Imprimimos el resultado.
print(f"La suma de los números pares en {numeros} es: {resultado}")