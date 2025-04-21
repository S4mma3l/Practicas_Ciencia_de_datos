def eliminar_duplicados(lista):
  """
  Esta función toma una lista y devuelve una nueva lista con los mismos elementos pero sin duplicados.
  El orden de los elementos en la lista resultante no está garantizado.

  Args:
    lista (list): La lista de la que se van a eliminar los duplicados.

  Returns:
    list: Una nueva lista con los elementos únicos de la lista original.
  """
  elementos_unicos = []  # Inicializamos una lista vacía para almacenar los elementos únicos.

  for elemento in lista:  # Iteramos sobre cada elemento de la lista original.
    if elemento not in elementos_unicos:  # Verificamos si el elemento actual ya está en la lista de elementos únicos.
      elementos_unicos.append(elemento)   # Si el elemento no está en la lista de únicos, lo añadimos.

  return elementos_unicos  # Devolvemos la lista con solo los elementos únicos.

# Ejemplo de uso:
numeros_con_duplicados = [1, 2, 2, 3, 4, 4, 5]
resultado = eliminar_duplicados(numeros_con_duplicados)
print(f"La lista sin duplicados de {numeros_con_duplicados} es: {resultado}")