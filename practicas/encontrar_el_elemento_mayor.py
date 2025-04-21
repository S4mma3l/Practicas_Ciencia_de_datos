def encontrar_mayor(lista_numeros):
  """
  Esta función toma una lista de números y devuelve el elemento más grande.

  Args:
    lista_numeros (list): Una lista de números.

  Returns:
    int or float: El número más grande en la lista. Devuelve None si la lista está vacía.
  """
  if not lista_numeros:  # Verificamos si la lista está vacía.
    return None         # Si la lista está vacía, no hay elemento mayor, devolvemos None.

  mayor = lista_numeros[0]  # Inicializamos la variable 'mayor' con el primer elemento de la lista.

  for numero in lista_numeros:  # Iteramos sobre cada número en la lista.
    if numero > mayor:      # Comparamos el número actual con el valor actual de 'mayor'.
      mayor = numero       # Si el número actual es mayor, actualizamos el valor de 'mayor'.

  return mayor  # Devolvemos el valor más grande encontrado en la lista.

# Ejemplo de uso:
numeros = [10, 5, 20, 8]
resultado = encontrar_mayor(numeros)
print(f"El elemento mayor en {numeros} es: {resultado}")