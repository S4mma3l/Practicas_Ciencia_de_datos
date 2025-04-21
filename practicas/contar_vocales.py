def contar_vocales(cadena):
  """
  Esta función toma una cadena y devuelve el número de vocales (a, e, i, o, u)
  y el número de consonantes que contiene, sin importar si son mayúsculas o minúsculas.

  Args:
    cadena (str): La cadena en la que se van a contar las vocales y consonantes.

  Returns:
    tuple: Una tupla que contiene el número total de vocales y el número total de consonantes.
  """
  vocales = "aeiou"       # Definimos una cadena con todas las vocales en minúscula.
  contador_vocales = 0    # Inicializamos un contador para el número de vocales.
  contador_consonantes = 0 # Inicializamos un contador para el número de consonantes.

  for caracter in cadena.lower():  # Iteramos sobre cada carácter de la cadena, convertido a minúscula.
    if caracter.isalpha():  # Esto asegura que solo contemos letras y evitemos contar espacios, números o signos de puntuación como consonantes.
      if caracter in vocales:      # Si es una letra, verificamos si es una vocal.
        contador_vocales += 1   # Si es una vocal, incrementamos el contador de vocales.
      else:
        contador_consonantes += 1 # Si es una letra y no es vocal, es una consonante, incrementamos su contador.

  return contador_vocales, contador_consonantes  # Devolvemos el número total de vocales y consonantes.

# Ejemplo de uso:
texto = input("Introduce el texto para contar las vocales y consonantes: ")  # Solicitamos al usuario que introduzca una cadena.
resultado_vocales, resultado_consonantes = contar_vocales(texto)
print(f"La cadena '{texto}' tiene {resultado_vocales} vocales y {resultado_consonantes} consonantes.")