def es_palindromo(cadena):
  """
  Esta función toma una cadena y devuelve True si es un palíndromo, False en caso contrario.
  Un palíndromo es una cadena que se lee igual de izquierda a derecha que de derecha a izquierda,
  ignorando mayúsculas y espacios.

  Args:
    cadena (str): La cadena que se va a verificar.

  Returns:
    bool: True si la cadena es un palíndromo, False en caso contrario.
  """
  cadena = cadena.lower()        # Convertimos la cadena a minúsculas para ignorar la diferencia entre mayúsculas y minúsculas.
  cadena = ''.join(filter(str.isalnum, cadena))  # Eliminamos caracteres no alfanuméricos (espacios, signos de puntuación, etc.).

  izquierda = 0                 # Inicializamos un puntero al inicio de la cadena.
  derecha = len(cadena) - 1     # Inicializamos un puntero al final de la cadena.

  while izquierda < derecha:     # Iteramos mientras el puntero izquierdo sea menor que el derecho.
    if cadena[izquierda] != cadena[derecha]:  # Comparamos los caracteres en los punteros.
      return False              # Si los caracteres no son iguales, no es un palíndromo, devolvemos False.
    izquierda = izquierda + 1   # Movemos el puntero izquierdo una posición hacia la derecha.
    derecha = derecha - 1     # Movemos el puntero derecho una posición hacia la izquierda.

  return True                   # Si el bucle termina sin encontrar diferencias, la cadena es un palíndromo, devolvemos True.

# Ejemplo de uso:
texto1 = "radar"
texto2 = "Hola"
texto3 = "A man, a plan, a canal: Panama"
print(f"'{texto1}' es palíndromo: {es_palindromo(texto1)}")
print(f"'{texto2}' es palíndromo: {es_palindromo(texto2)}")
print(f"'{texto3}' es palíndromo: {es_palindromo(texto3)}")