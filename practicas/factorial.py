def factorial(n):
  """
  Esta función toma un número entero no negativo 'n' y devuelve su factorial.
  El factorial de n (n!) es el producto de todos los enteros positivos menores o iguales a n.

  Args:
    n (int): Un número entero no negativo.

  Returns:
    int: El factorial de n. Devuelve 1 si n es 0. Devuelve None si n es negativo.
  """
  if n < 0:          # Verificamos si el número es negativo.
    return None      # El factorial no está definido para números negativos, devolvemos None.
  elif n == 0:      # Caso base: el factorial de 0 es 1.
    return 1
  else:
    resultado = 1  # Inicializamos el resultado en 1 (el elemento neutro de la multiplicación).
    for i in range(1, n + 1):  # Iteramos desde 1 hasta n (inclusive).
      resultado = resultado * i  # En cada iteración, multiplicamos el resultado actual por el número actual.
    return resultado  # Devolvemos el factorial calculado.

# Ejemplo de uso:
numero = 5
resultado = factorial(numero)
print(f"El factorial de {numero} es: {resultado}")