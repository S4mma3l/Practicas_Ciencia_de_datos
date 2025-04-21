def invertir_cadena(s):
  """
  Esta función toma una cadena 's' como entrada y devuelve su versión invertida.

  Args:
    s (str): La cadena que se va a invertir.

  Returns:
    str: La cadena invertida.
  """
  cadena_invertida = ""  # Inicializamos una cadena vacía para almacenar el resultado.
  indice = len(s) - 1   # Obtenemos el índice del último carácter de la cadena.

  while indice >= 0:     # Iteramos mientras el índice sea mayor o igual a 0 (el primer carácter).
    cadena_invertida = cadena_invertida + s[indice]  # Añadimos el carácter actual (desde el final) a la cadena invertida.
    indice = indice - 1   # Decrementamos el índice para movernos al carácter anterior.

  return cadena_invertida  # Devolvemos la cadena resultante invertida.

# Ejemplo de uso:
texto = input("Introduce el texto a invertir: ")  # Solicitamos al usuario que introduzca una cadena.
resultado = invertir_cadena(texto)
print(f"La cadena '{texto}' invertida es: '{resultado}'")