#!/usr/bin/env python3
"""
Problema de las 8 Reinas
========================

Colocar 8 reinas en un tablero de ajedrez de 8x8 de forma que ninguna ataque
a otra: no puede haber dos reinas en la misma fila, columna ni diagonal.

Se resuelve de dos formas:

1. Backtracking (búsqueda en profundidad con retroceso): encuentra las 92
   soluciones exactas.
2. Algoritmo genético: cada individuo es una permutación (la posición i es la
   fila de la reina en la columna i), así filas y columnas nunca se repiten y
   solo hay que evitar choques en diagonal.

Uso:
    python ocho_reinas.py            # tablero de 8x8
    python ocho_reinas.py 10         # cualquier N
"""

import random
import sys
import time


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def imprimir_tablero(solucion):
    """solucion[c] = fila de la reina en la columna c."""
    n = len(solucion)
    print("   " + " ".join(chr(ord("a") + c) for c in range(n)))
    for f in range(n):
        fila = []
        for c in range(n):
            if solucion[c] == f:
                fila.append("Q")
            else:
                fila.append("." if (f + c) % 2 == 0 else ":")
        print(f"{n - f:2d} " + " ".join(fila))
    print()


def es_valida(solucion):
    """Comprueba que ninguna pareja de reinas se ataque."""
    n = len(solucion)
    for i in range(n):
        for j in range(i + 1, n):
            if solucion[i] == solucion[j]:
                return False
            if abs(solucion[i] - solucion[j]) == abs(i - j):
                return False
    return True


# ----------------------------------------------------------------------------
# 1. Backtracking
# ----------------------------------------------------------------------------
def backtracking(n):
    """Devuelve todas las soluciones del problema de las N reinas."""
    soluciones = []
    solucion = [0] * n
    filas_usadas = [False] * n
    diag1 = [False] * (2 * n - 1)  # fila + columna
    diag2 = [False] * (2 * n - 1)  # fila - columna + n - 1

    def colocar(col):
        if col == n:
            soluciones.append(solucion.copy())
            return
        for fila in range(n):
            d1, d2 = fila + col, fila - col + n - 1
            if filas_usadas[fila] or diag1[d1] or diag2[d2]:
                continue  # poda: esta casilla está atacada
            solucion[col] = fila
            filas_usadas[fila] = diag1[d1] = diag2[d2] = True
            colocar(col + 1)
            filas_usadas[fila] = diag1[d1] = diag2[d2] = False  # retroceso

    colocar(0)
    return soluciones


# ----------------------------------------------------------------------------
# 2. Algoritmo genético
# ----------------------------------------------------------------------------
def conflictos(individuo):
    """Número de parejas de reinas que se atacan en diagonal."""
    n = len(individuo)
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(individuo[i] - individuo[j]) == j - i:
                total += 1
    return total


def seleccion_torneo(poblacion, aptitudes, k=3):
    elegidos = random.sample(range(len(poblacion)), k)
    mejor = min(elegidos, key=lambda i: aptitudes[i])
    return poblacion[mejor]


def cruce_ox(padre1, padre2):
    """Cruce ordenado (OX): conserva un tramo del padre 1 y completa con el
    orden del padre 2, de modo que el hijo sigue siendo una permutación."""
    n = len(padre1)
    a, b = sorted(random.sample(range(n), 2))
    hijo = [None] * n
    hijo[a:b + 1] = padre1[a:b + 1]
    resto = [g for g in padre2 if g not in hijo]
    for i in range(n):
        if hijo[i] is None:
            hijo[i] = resto.pop(0)
    return hijo


def mutacion_intercambio(individuo, prob):
    if random.random() < prob:
        i, j = random.sample(range(len(individuo)), 2)
        individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo


def genetico(n, tam_poblacion=100, generaciones=1000, prob_cruce=0.9,
             prob_mutacion=0.3, elitismo=2, semilla=None):
    """Busca una solución con un algoritmo genético. Devuelve
    (mejor_individuo, generacion_en_que_termino, historial_de_conflictos)."""
    if semilla is not None:
        random.seed(semilla)

    poblacion = [random.sample(range(n), n) for _ in range(tam_poblacion)]
    historial = []

    for gen in range(generaciones):
        aptitudes = [conflictos(ind) for ind in poblacion]
        orden = sorted(range(tam_poblacion), key=lambda i: aptitudes[i])
        mejor = poblacion[orden[0]]
        historial.append(aptitudes[orden[0]])

        if aptitudes[orden[0]] == 0:
            return mejor, gen, historial

        # Elitismo: los mejores pasan sin cambios
        nueva = [poblacion[i].copy() for i in orden[:elitismo]]
        while len(nueva) < tam_poblacion:
            p1 = seleccion_torneo(poblacion, aptitudes)
            p2 = seleccion_torneo(poblacion, aptitudes)
            hijo = cruce_ox(p1, p2) if random.random() < prob_cruce else p1.copy()
            nueva.append(mutacion_intercambio(hijo, prob_mutacion))
        poblacion = nueva

    aptitudes = [conflictos(ind) for ind in poblacion]
    mejor = min(poblacion, key=conflictos)
    return mejor, generaciones, historial


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    if n < 4:
        print("No existe solución para N < 4 (excepto N = 1).")
        return

    print("=" * 50)
    print(f"  PROBLEMA DE LAS {n} REINAS")
    print("=" * 50)

    # --- Backtracking ---
    print("\n[1] BACKTRACKING")
    t0 = time.perf_counter()
    soluciones = backtracking(n)
    t = time.perf_counter() - t0
    print(f"Soluciones encontradas: {len(soluciones)}  ({t * 1000:.2f} ms)")
    print(f"Primera solución: {soluciones[0]}")
    imprimir_tablero(soluciones[0])

    # --- Algoritmo genético ---
    print("[2] ALGORITMO GENÉTICO")
    t0 = time.perf_counter()
    mejor, gen, historial = genetico(n)
    t = time.perf_counter() - t0
    c = conflictos(mejor)
    if c == 0:
        print(f"Solución encontrada en la generación {gen}  ({t * 1000:.2f} ms)")
    else:
        print(f"No se llegó a 0 conflictos; mejor individuo con {c} conflictos")
    print(f"Individuo: {mejor}")
    imprimir_tablero(mejor)
    print(f"¿Es válida? {'Sí' if es_valida(mejor) else 'No'}")
    print(f"¿Está entre las soluciones del backtracking? "
          f"{'Sí' if mejor in soluciones else 'No'}")


if __name__ == "__main__":
    main()
