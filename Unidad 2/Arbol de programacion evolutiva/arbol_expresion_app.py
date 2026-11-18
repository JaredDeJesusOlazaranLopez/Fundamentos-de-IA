#!/usr/bin/env python3
"""
ÁRBOL DE EXPRESIÓN  ·  Aplicación de escritorio (Python + Tkinter + Matplotlib)
==============================================================================

Escribe una ecuación dentro de la aplicación y obtén:

  * El árbol de expresión dibujado, con el valor que calcula cada nodo.
  * El recorrido PREORDEN animado paso a paso (nodos numerados y camino resaltado).
  * Los tres recorridos: preorden (prefija), inorden (infija) y postorden (postfija).
  * La evaluación usando la notación prefija (preorden) con una pila, paso a paso.
  * La gráfica de f(x), su derivada, la recta tangente, raíces y dominio.
  * La derivada simbólica (y su propio árbol), y una tabla de valores.
  * Puedes escribir la ecuación en notación INFIJA  o directamente en PREFIJA (preorden).
  * Altura máxima configurable para el árbol (valida lo que escribes y limita el generador).
  * Botón «Generar aleatoria»: crea ecuaciones al azar sin pasar de la altura máxima.
  * Pestaña «Datos (CSV)»: abre un CSV, grafica los puntos y, comparando MSE, RMSE y NMSE de
    16 familias de funciones, identifica cuál es (y la puedes usar como f(x) para ver su árbol).
  * Genera el CSV desde tu f(x) (rango, paso y ruido a elegir), lo guarda, lo grafica y lo compara;
    también puedes abrir cualquier otro CSV para comparar contra tu función.
  * ALGORITMO GENÉTICO (regresión simbólica) generacional: selección por RULETA, cruce de subárboles
    y mutación; evoluciona árboles de expresión que se aproximan a los puntos sin pasar de la altura máxima.

Ejecutar:   python arbol_expresion_app.py
Requisitos: pip install numpy matplotlib      (tkinter viene con Python;
            en Ubuntu/Debian:  sudo apt install python3-tk)
"""

import bisect
import csv
import io
import itertools
import math
import random
import re
import warnings

import numpy as np
import matplotlib

try:                                    # la interfaz necesita tkinter; el motor de ajuste no
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter import font as tkfont

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    HAY_TK = True
except ImportError:                     # p. ej. servidor sin tkinter: se puede importar igual
    tk = None
    HAY_TK = False
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════════════
#  1. NODOS DEL ÁRBOL
# ══════════════════════════════════════════════════════════════════════════
FUNCIONES = {
    "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "ln": np.log, "log": np.log, "log10": np.log10, "abs": np.abs,
}
CONSTANTES = {"pi": math.pi, "e": math.e}
OPS = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "^": np.power}
SIMBOLO = {"+": "+", "-": "−", "*": "×", "/": "÷", "^": "^"}


def fmt(v, dig=6):
    """Formatea un número para mostrarlo."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(v):
        return "indefinido"
    if math.isinf(v):
        return "∞" if v > 0 else "−∞"
    if abs(v) < 1e-12:
        v = 0.0
    return f"{v:.{dig}g}"


def fmt_m(v, dig=5):
    """Formato para métricas de error: no redondea a 0 los valores diminutos (p. ej. 8e-22)."""
    if not math.isfinite(v):
        return "—"
    return f"{v:.{dig}g}"


class Nodo:
    """tipo: 'num' | 'const' | 'var' | 'op' | 'func'
       valor: número, nombre, 'x', operador ('+','-','*','/','^','neg') o nombre de función"""

    def __init__(self, tipo, valor, hijos=None):
        self.tipo, self.valor, self.hijos = tipo, valor, list(hijos or [])

    @property
    def etiqueta(self):
        if self.tipo == "num":
            return f"{self.valor:.6g}"
        if self.tipo == "const":
            return "π" if self.valor == "pi" else self.valor
        if self.tipo in ("var", "func"):
            return self.valor
        return "neg" if self.valor == "neg" else SIMBOLO[self.valor]

    @property
    def token(self):
        """Texto ASCII del nodo (el que se escribe en notación prefija)."""
        return f"{self.valor:.12g}" if self.tipo == "num" else str(self.valor)

    def evaluar(self, x):
        """Evalúa el subárbol. Acepta escalares o arrays de numpy."""
        with np.errstate(all="ignore"):
            if self.tipo == "num":
                return np.float64(self.valor) + 0 * x
            if self.tipo == "const":
                return np.float64(CONSTANTES[self.valor]) + 0 * x
            if self.tipo == "var":
                return x
            if self.tipo == "func":
                return FUNCIONES[self.valor](self.hijos[0].evaluar(x))
            if self.valor == "neg":
                return -self.hijos[0].evaluar(x)
            a, b = (h.evaluar(x) for h in self.hijos)
            return OPS[self.valor](a, b)

    def copiar(self):
        return Nodo(self.tipo, self.valor, [h.copiar() for h in self.hijos])

    def depende_de_x(self):
        return self.tipo == "var" or any(h.depende_de_x() for h in self.hijos)


# ══════════════════════════════════════════════════════════════════════════
#  2. ANÁLISIS DEL TEXTO  (infija → árbol,  prefija → árbol)
# ══════════════════════════════════════════════════════════════════════════
TOKEN_RE = re.compile(
    r"\s*(?:(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+)|([A-Za-z_][A-Za-z_0-9]*)|(\*\*|[-+*/^(),]))")


def tokenizar(texto):
    tokens, pos, texto = [], 0, texto.strip()
    while pos < len(texto):
        m = TOKEN_RE.match(texto, pos)
        if not m:
            malo = texto[pos:].lstrip()[:1]
            if not malo:
                break
            raise SyntaxError(f"Carácter no válido: «{malo}»")
        num, nombre, op = m.groups()
        pos = m.end()
        if num:
            tokens.append(("num", float(num)))
        elif nombre:
            mx = re.fullmatch(r"x(\d+)", nombre)      # x2  →  x ^ 2
            if mx:
                tokens += [("name", "x"), ("op", "^"), ("num", float(mx.group(1)))]
            else:
                tokens.append(("name", nombre))
        else:
            tokens.append(("op", "^" if op == "**" else op))
    return tokens


def multiplicacion_implicita(tokens):
    """2x → 2*x ,  2(x+1) → 2*(x+1) ,  (a)(b) → (a)*(b)"""
    salida = []
    for tok in tokens:
        if salida:
            ant = salida[-1]
            fin_valor = ant[0] in ("num", "name") or ant == ("op", ")")
            ini_valor = tok[0] in ("num", "name") or tok == ("op", "(")
            if fin_valor and ini_valor and not (ant[0] == "name" and ant[1] in FUNCIONES):
                salida.append(("op", "*"))
        salida.append(tok)
    return salida


class ParserInfijo:
    """expr := term (('+'|'-') term)* ;  term := unary (('*'|'/') unary)* ;
       unary := '-' unary | power ;  power := atom ('^' unary)? ;
       atom := num | x | const | func '(' expr ')' | '(' expr ')'"""

    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def ver(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def tomar(self):
        tok = self.ver()
        self.i += 1
        return tok

    def esperar(self, s):
        if self.tomar() != ("op", s):
            raise SyntaxError(f"Se esperaba «{s}»")

    def parsear(self):
        if not self.t:
            raise SyntaxError("La expresión está vacía")
        nodo = self.expr()
        if self.i != len(self.t):
            raise SyntaxError(f"Símbolo inesperado: «{self.ver()[1]}»")
        return nodo

    def expr(self):
        n = self.term()
        while self.ver() in (("op", "+"), ("op", "-")):
            op = self.tomar()[1]
            n = Nodo("op", op, [n, self.term()])
        return n

    def term(self):
        n = self.unary()
        while self.ver() in (("op", "*"), ("op", "/")):
            op = self.tomar()[1]
            n = Nodo("op", op, [n, self.unary()])
        return n

    def unary(self):
        if self.ver() == ("op", "-"):
            self.tomar()
            return Nodo("op", "neg", [self.unary()])
        if self.ver() == ("op", "+"):
            self.tomar()
            return self.unary()
        return self.power()

    def power(self):
        base = self.atom()
        if self.ver() == ("op", "^"):
            self.tomar()
            return Nodo("op", "^", [base, self.unary()])
        return base

    def atom(self):
        tipo, val = self.tomar()
        if tipo == "num":
            return Nodo("num", val)
        if tipo == "name":
            if val in FUNCIONES:
                self.esperar("(")
                arg = self.expr()
                self.esperar(")")
                return Nodo("func", val, [arg])
            if val == "x":
                return Nodo("var", "x")
            if val in CONSTANTES:
                return Nodo("const", val)
            raise SyntaxError(f"Nombre desconocido: «{val}»")
        if (tipo, val) == ("op", "("):
            n = self.expr()
            self.esperar(")")
            return n
        raise SyntaxError("Expresión incompleta")


def construir_arbol(texto):
    return ParserInfijo(multiplicacion_implicita(tokenizar(texto))).parsear()


def parsear_prefija(texto):
    """Construye el árbol leyendo una expresión en PREORDEN (notación prefija).
    Ej:  sqrt - + * 2 + ^ x 2 3 * 5 x 3     (usar 'neg' para el signo menos unario)"""
    toks = texto.replace(",", " ").split()
    if not toks:
        raise SyntaxError("La expresión está vacía")
    pos = [0]

    def rec():
        if pos[0] >= len(toks):
            raise SyntaxError("Faltan operandos al final de la expresión prefija")
        t = toks[pos[0]]
        pos[0] += 1
        t = "^" if t == "**" else t
        if t in OPS:
            izq = rec()
            return Nodo("op", t, [izq, rec()])
        if t == "neg":
            return Nodo("op", "neg", [rec()])
        if t in FUNCIONES:
            return Nodo("func", t, [rec()])
        if t == "x":
            return Nodo("var", "x")
        if t in CONSTANTES:
            return Nodo("const", t)
        try:
            return Nodo("num", float(t))
        except ValueError:
            raise SyntaxError(f"Token desconocido: «{t}»")

    raiz = rec()
    if pos[0] != len(toks):
        raise SyntaxError(f"Sobran tokens a partir de «{toks[pos[0]]}»")
    return raiz


def leer_numero(texto):
    """Lee un número (admite cosas como pi/2 o 3e-2)."""
    n = construir_arbol(texto)
    if n.depende_de_x():
        raise SyntaxError("Aquí debe ir un número, no una expresión con x")
    v = float(n.evaluar(np.float64(0.0)))
    if not math.isfinite(v):
        raise ValueError("Número no válido")
    return v


# ══════════════════════════════════════════════════════════════════════════
#  3. RECORRIDOS DEL ÁRBOL
# ══════════════════════════════════════════════════════════════════════════
def preorden(n, nivel=0):
    """Raíz → hijo izquierdo → hijo derecho.  Devuelve [(nodo, nivel), ...]"""
    res = [(n, nivel)]
    for h in n.hijos:
        res += preorden(h, nivel + 1)
    return res


def postorden(n):
    """Hijos → raíz."""
    res = []
    for h in n.hijos:
        res += postorden(h)
    return res + [n]


PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "neg": 3, "^": 4}


def _inf(n):
    if n.tipo == "num":
        return n.token, (3 if n.valor < 0 else 9)
    if n.tipo in ("var", "const"):
        return n.token, 9
    if n.tipo == "func":
        return f"{n.valor}({_inf(n.hijos[0])[0]})", 9
    if n.valor == "neg":
        s, p = _inf(n.hijos[0])
        return "-" + (f"({s})" if p < 3 else s), 3
    (l, pl), (r, pr) = _inf(n.hijos[0]), _inf(n.hijos[1])
    p = PREC[n.valor]
    if n.valor == "^":
        if pl <= p:
            l = f"({l})"
        if pr < p:
            r = f"({r})"
        return f"{l}^{r}", p
    if pl < p:
        l = f"({l})"
    if pr <= p:
        r = f"({r})"
    return f"{l} {n.valor} {r}", p


def a_infija(n):
    """Recorrido INORDEN con los paréntesis mínimos necesarios."""
    return _inf(n)[0]


def a_prefija(n):
    return " ".join(x.token for x, _ in preorden(n))


def a_postfija(n):
    return " ".join(x.token for x in postorden(n))


def altura(n):
    return 1 + max((altura(h) for h in n.hijos), default=0)


def mapa_padres(raiz):
    """id(nodo) → (padre, 'izquierdo'/'derecho'/'único')"""
    m = {}

    def rec(n):
        for i, h in enumerate(n.hijos):
            lado = "único" if len(n.hijos) == 1 else ("izquierdo" if i == 0 else "derecho")
            m[id(h)] = (n, lado)
            rec(h)

    rec(raiz)
    return m


def aplicar(tok, a, b=None):
    with np.errstate(all="ignore"):
        if tok == "neg":
            return float(-a)
        if tok in OPS:
            return float(OPS[tok](np.float64(a), np.float64(b)))
        return float(FUNCIONES[tok](np.float64(a)))


def evaluar_prefija_pasos(raiz, x0):
    """Evalúa la expresión LEYENDO LA NOTACIÓN PREFIJA (preorden) de derecha a izquierda
    con una pila.  Devuelve (lista_de_pasos, resultado)."""
    orden = [n for n, _ in preorden(raiz)]
    pila, pasos = [], []
    for idx in range(len(orden) - 1, -1, -1):
        n = orden[idx]
        tok = n.token
        if not n.hijos:
            val = float(x0) if n.tipo == "var" else (
                CONSTANTES[n.valor] if n.tipo == "const" else n.valor)
            pila.append(val)
            accion = f"Operando → apilar {fmt(val)}"
        elif len(n.hijos) == 1:
            a = pila.pop()
            val = aplicar(tok, a)
            pila.append(val)
            accion = (f"desapilar {fmt(a)}  →  −({fmt(a)}) = {fmt(val)}" if tok == "neg"
                      else f"desapilar {fmt(a)}  →  {tok}({fmt(a)}) = {fmt(val)}")
        else:
            a = pila.pop()          # operando izquierdo (queda arriba al leer de derecha a izquierda)
            b = pila.pop()          # operando derecho
            val = aplicar(tok, a, b)
            pila.append(val)
            accion = f"desapilar {fmt(a)} y {fmt(b)}  →  {fmt(a)} {tok} {fmt(b)} = {fmt(val)}"
        pasos.append((len(pasos) + 1, idx + 1, tok, accion,
                      "[ " + ",  ".join(fmt(v, 5) for v in pila) + " ]"))
    return pasos, pila[0]


# ══════════════════════════════════════════════════════════════════════════
#  3b. GENERADOR DE ECUACIONES ALEATORIAS (respeta una altura máxima)
# ══════════════════════════════════════════════════════════════════════════
def generar_aleatoria(max_alt, rango=(-10, 10)):
    """Crea un árbol aleatorio con altura <= max_alt, que contenga x y que esté
    definido en la mayor parte del rango (para que la gráfica sea interesante)."""
    max_alt = max(2, int(max_alt))
    funcs = ["sin", "cos", "sin", "cos", "sqrt", "sqrt", "abs", "ln", "exp"]
    ops = ["+", "+", "+", "-", "-", "-", "*", "*", "*", "/", "^"]

    def hoja():
        if random.random() < 0.6:
            return Nodo("var", "x")
        return Nodo("num", float(random.randint(1, 9)))

    def gen(h, es_raiz=False):
        if h <= 1 or (not es_raiz and random.random() < 0.2):
            return hoja()
        if random.random() < 0.22:
            return Nodo("func", random.choice(funcs), [gen(h - 1)])
        op = random.choice(ops)
        if op == "^":     # exponente pequeño para que la gráfica sea razonable
            return Nodo("op", "^", [gen(h - 1), Nodo("num", float(random.choice([2, 2, 3])))])
        return Nodo("op", op, [gen(h - 1), gen(h - 1)])

    def asegurar_x(t):
        if t.depende_de_x():
            return
        exps = {id(n.hijos[1]) for n, _ in preorden(t) if n.valor == "^" and n.tipo == "op"}
        hojas = [n for n, _ in preorden(t) if n.tipo == "num" and id(n) not in exps]
        n = random.choice(hojas)
        n.tipo, n.valor = "var", "x"

    def natural(t):
        """Descarta trivialidades: subárboles constantes (7-7), x-x, x/x, exp(ln(..))."""
        for n, _ in preorden(t):
            if not n.hijos:
                continue
            if not n.depende_de_x():
                return False
            if n.tipo == "func" and n.hijos[0].tipo == "func":
                return False
            if n.tipo == "op" and n.valor in ("-", "/") and \
                    a_prefija(n.hijos[0]) == a_prefija(n.hijos[1]):
                return False
        return True

    xs = np.linspace(rango[0], rango[1], 400)
    t = None
    for intento in range(300):
        t = gen(max_alt, True)
        asegurar_x(t)
        if intento < 250 and not natural(t):
            continue
        if intento < 250 and altura(t) < max(2, max_alt - 1):
            continue                      # intenta acercarse a la altura máxima
        with np.errstate(all="ignore"):
            ys = t.evaluar(xs).astype(float)
        fin = np.isfinite(ys)
        if fin.mean() < 0.7 or not fin.any():
            continue
        v = ys[fin]
        if np.ptp(v) < 1e-9 or np.max(np.abs(v)) > 1e6:
            continue
        return t
    return t


# ══════════════════════════════════════════════════════════════════════════
#  4. DERIVADA SIMBÓLICA (con simplificación básica)
# ══════════════════════════════════════════════════════════════════════════
def _num(v):
    return Nodo("num", float(v))


def _es(n, v=None):
    return n.tipo == "num" and (v is None or n.valor == v)


def s_neg(a):
    if _es(a):
        return _num(-a.valor)
    if a.tipo == "op" and a.valor == "neg":
        return a.hijos[0]
    return Nodo("op", "neg", [a])


def s_suma(a, b):
    if _es(a) and _es(b):
        return _num(a.valor + b.valor)
    if _es(a, 0):
        return b
    if _es(b, 0):
        return a
    return Nodo("op", "+", [a, b])


def s_resta(a, b):
    if _es(a) and _es(b):
        return _num(a.valor - b.valor)
    if _es(b, 0):
        return a
    if _es(a, 0):
        return s_neg(b)
    return Nodo("op", "-", [a, b])


def s_mul(a, b):
    if _es(a) and _es(b):
        return _num(a.valor * b.valor)
    if _es(a, 0) or _es(b, 0):
        return _num(0)
    if _es(a, 1):
        return b
    if _es(b, 1):
        return a
    if _es(a, -1):
        return s_neg(b)
    if _es(b, -1):
        return s_neg(a)
    return Nodo("op", "*", [a, b])


def s_div(a, b):
    if _es(a) and _es(b) and b.valor != 0:
        return _num(a.valor / b.valor)
    if _es(a, 0):
        return _num(0)
    if _es(b, 1):
        return a
    return Nodo("op", "/", [a, b])


def s_pot(a, b):
    if _es(b, 0):
        return _num(1)
    if _es(b, 1):
        return a
    if _es(a) and _es(b):
        try:
            r = a.valor ** b.valor
            if isinstance(r, float) and math.isfinite(r):
                return _num(r)
        except (OverflowError, ZeroDivisionError):
            pass
    return Nodo("op", "^", [a, b])


def s_func(nombre, a):
    return Nodo("func", nombre, [a])


def derivar(n):
    t = n.tipo
    if t in ("num", "const"):
        return _num(0)
    if t == "var":
        return _num(1)
    if t == "func":
        a = n.hijos[0]
        da, f = derivar(a), n.valor
        if f == "sqrt":
            return s_div(da, s_mul(_num(2), n))
        if f == "sin":
            return s_mul(s_func("cos", a), da)
        if f == "cos":
            return s_neg(s_mul(s_func("sin", a), da))
        if f == "tan":
            return s_div(da, s_pot(s_func("cos", a), _num(2)))
        if f == "exp":
            return s_mul(n, da)
        if f in ("ln", "log"):
            return s_div(da, a)
        if f == "log10":
            return s_div(da, s_mul(a, s_func("ln", _num(10))))
        if f == "abs":
            return s_mul(s_div(a, n), da)
    if n.valor == "neg":
        return s_neg(derivar(n.hijos[0]))
    a, b = n.hijos
    da, db = derivar(a), derivar(b)
    op = n.valor
    if op == "+":
        return s_suma(da, db)
    if op == "-":
        return s_resta(da, db)
    if op == "*":
        return s_suma(s_mul(da, b), s_mul(a, db))
    if op == "/":
        return s_div(s_resta(s_mul(da, b), s_mul(a, db)), s_pot(b, _num(2)))
    # potencia
    if not b.depende_de_x():
        return s_mul(s_mul(b, s_pot(a, s_resta(b, _num(1)))), da)
    if not a.depende_de_x():
        return s_mul(s_mul(n, s_func("ln", a)), db)
    return s_mul(n, s_suma(s_mul(db, s_func("ln", a)), s_div(s_mul(b, da), a)))


# ══════════════════════════════════════════════════════════════════════════
#  5. ANÁLISIS NUMÉRICO: raíces y dominio
# ══════════════════════════════════════════════════════════════════════════
def _f_escalar(raiz, x):
    return float(raiz.evaluar(np.float64(x)))


def encontrar_raices(raiz, a, b, n=4000):
    xs = np.linspace(a, b, n)
    ys = raiz.evaluar(xs).astype(float)
    raices = []

    def agregar(r):
        if all(abs(r - q) > 1e-5 * max(1.0, abs(r)) for q in raices):
            raices.append(r)

    # 1) cambios de signo → bisección
    for i in range(n - 1):
        if len(raices) >= 40:                      # tope: funciones oscilantes tienen miles
            break
        y1, y2 = ys[i], ys[i + 1]
        if not (np.isfinite(y1) and np.isfinite(y2)):
            continue
        if y1 == 0:
            agregar(xs[i])
        elif y1 * y2 < 0:
            lo, hi, flo = xs[i], xs[i + 1], y1
            for _ in range(70):
                mid = (lo + hi) / 2
                fm = _f_escalar(raiz, mid)
                if not np.isfinite(fm):
                    break
                if flo * fm <= 0:
                    hi = mid
                else:
                    lo, flo = mid, fm
            r = (lo + hi) / 2
            fr = _f_escalar(raiz, r)
            if np.isfinite(fr) and abs(fr) < 1e-6:
                agregar(r)
    if np.isfinite(ys[-1]) and ys[-1] == 0:
        agregar(xs[-1])

    # 2) mínimos de |f| cercanos a 0 (raíces que "tocan" el eje, como en x² o sqrt)
    fin = np.abs(ys[np.isfinite(ys)])
    if fin.size:
        umbral = 0.02 * np.percentile(fin, 90) + 1e-9
        absy = np.where(np.isfinite(ys), np.abs(ys), np.inf)
        for i in range(1, n - 1):
            if absy[i] < umbral and absy[i] <= absy[i - 1] and absy[i] < absy[i + 1]:
                lo, hi = xs[i - 1], xs[i + 1]

                def g(x):
                    v = _f_escalar(raiz, x)
                    return abs(v) if np.isfinite(v) else np.inf

                for _ in range(100):
                    m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
                    if g(m1) < g(m2):
                        hi = m2
                    else:
                        lo = m1
                r = (lo + hi) / 2
                if g(r) < 1e-4:
                    agregar(r)
            if len(raices) >= 40:
                break
    return sorted(raices)


def intervalos_indefinidos(raiz, a, b, n=4000):
    xs = np.linspace(a, b, n)
    ys = raiz.evaluar(xs).astype(float)
    mask = ~np.isfinite(ys)
    res, i = [], 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            res.append((xs[max(i - 1, 0)] if i > 0 else xs[i], xs[min(j + 1, n - 1)] if j < n - 1 else xs[j]))
            i = j + 1
        else:
            i += 1
    return res


# ══════════════════════════════════════════════════════════════════════════
#  5b. DATOS CSV: lectura, ajuste de modelos y métricas (MSE, RMSE, NMSE)
# ══════════════════════════════════════════════════════════════════════════
def leer_csv(ruta):
    """Lee un CSV (separador y cabecera detectados solos; admite coma decimal).
    Devuelve (nombres, columnas) con solo las columnas numéricas."""
    texto = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(ruta, encoding=enc, newline="") as f:
                texto = f.read()
            break
        except UnicodeDecodeError:
            continue
    if not texto or not texto.strip():
        raise ValueError("El archivo está vacío")
    try:
        dial = csv.Sniffer().sniff(texto[:5000], delimiters=",;\t| ")
    except csv.Error:
        dial = csv.excel
    sep = dial.delimiter
    filas = [r for r in csv.reader(io.StringIO(texto), dial) if any(c.strip() for c in r)]

    def num(c):
        c = c.strip().strip('"')
        if sep != ",":
            c = c.replace(",", ".")
        try:
            return float(c)
        except ValueError:
            return None

    ncol = max(len(r) for r in filas)
    cabecera = any(num(c) is None for c in filas[0])
    if cabecera:
        nombres = [(filas[0][i].strip() if i < len(filas[0]) and filas[0][i].strip() else f"col{i + 1}")
                   for i in range(ncol)]
        filas = filas[1:]
    else:
        nombres = [f"col{i + 1}" for i in range(ncol)]
    out_n, out_c = [], []
    for i in range(ncol):
        vals = [num(r[i]) if i < len(r) else None for r in filas]
        arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
        if np.isfinite(arr).sum() >= 3 and np.isfinite(arr).mean() >= 0.8:
            out_n.append(nombres[i])
            out_c.append(arr)
    if not out_c:
        raise ValueError("No encontré columnas numéricas en el archivo")
    return out_n, out_c


TIPOS_EJEMPLO = ["Lineal", "Cuadrática", "Exponencial", "Seno", "Logarítmica", "Gaussiana", "Logística"]


def datos_ejemplo(tipo, n=60, semilla=None, ruido=0.03):
    """Datos sintéticos (por defecto con ~3 % de ruido) para probar la identificación."""
    rng = np.random.default_rng(semilla)
    if tipo == "Lineal":
        x = np.linspace(0, 10, n); y = 2.5 * x + 4
    elif tipo == "Cuadrática":
        x = np.linspace(-5, 5, n); y = 0.8 * x ** 2 - 1.5 * x + 2
    elif tipo == "Exponencial":
        x = np.linspace(0, 5, n); y = 1.5 * np.exp(0.6 * x)
    elif tipo == "Seno":
        x = np.linspace(0, 10, n + 20); y = 3 * np.sin(1.2 * x + 0.5) + 1
    elif tipo == "Logarítmica":
        x = np.linspace(0.5, 20, n); y = 4 * np.log(x) + 2
    elif tipo == "Gaussiana":
        x = np.linspace(-6, 6, n); y = 5 * np.exp(-(x - 1) ** 2 / (2 * 1.5 ** 2))
    else:  # Logística
        x = np.linspace(-6, 10, n); y = 8 / (1 + np.exp(-1.1 * (x - 2)))
    return x, y + ruido * np.ptp(y) * rng.normal(size=len(x))


def metricas(y, yhat):
    """Fórmulas (pizarra):
         MSE  = (1/n) Σ (yᵢ − ŷᵢ)²                 RMSE = √MSE
         NMSE = Σ |y − ŷ|² / Σ |y|²
         R²   = 1 − Σ (yᵢ − ŷᵢ)² / Σ (yᵢ − ȳ)²      con  ȳ = (1/n) Σ yᵢ
       (fvu = SSE/SST = 1 − R² se usa internamente para el criterio de parsimonia)"""
    malo = dict(mse=math.inf, rmse=math.inf, nmse=math.inf, r2=-math.inf, fvu=math.inf)
    with np.errstate(all="ignore"):
        if yhat.shape != y.shape or not np.all(np.isfinite(yhat)):
            return malo
        sse = float(np.sum((y - yhat) ** 2))
        mse = sse / len(y)
        sy2 = float(np.sum(y ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        nmse = sse / sy2 if sy2 > 0 else (0.0 if sse == 0 else math.inf)
        fvu = sse / sst if sst > 0 else (0.0 if sse < 1e-20 else math.inf)
        return dict(mse=mse, rmse=math.sqrt(mse), nmse=nmse, r2=1 - fvu, fvu=fvu)


def limpiar(n):
    """Convierte neg(número) en número negativo (árbol más limpio)."""
    n.hijos = [limpiar(h) for h in n.hijos]
    if n.tipo == "op" and n.valor == "neg" and n.hijos[0].tipo == "num":
        return Nodo("num", -n.hijos[0].valor)
    return n


# ---- mínimos cuadrados no lineales (Levenberg–Marquardt, solo numpy) ----
def _lm(f, p0, x, y, max_iter=150):
    p = np.array(p0, dtype=float)
    with np.errstate(all="ignore"):
        r = f(x, p) - y
        if not np.all(np.isfinite(r)):
            return None
        c, lam = float(r @ r), 1e-3
        for _ in range(max_iter):
            J = np.empty((len(x), len(p)))
            for j in range(len(p)):
                dp = np.zeros_like(p)
                dp[j] = 1e-7 * max(1.0, abs(p[j]))
                J[:, j] = ((f(x, p + dp) - y) - r) / dp[j]
            if not np.all(np.isfinite(J)):
                break
            A, g = J.T @ J, J.T @ r
            mejora = False
            for _ in range(14):
                try:
                    paso = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                rn = f(x, p + paso) - y
                cn = float(rn @ rn) if np.all(np.isfinite(rn)) else math.inf
                if cn < c:
                    p, r, c, lam, mejora = p + paso, rn, cn, max(lam / 3, 1e-12), True
                    break
                lam *= 5
            if not mejora or np.linalg.norm(paso) < 1e-12 * (1 + np.linalg.norm(p)):
                break
    return p


def _mejor_lm(f, starts, x, y):
    mejor, mejor_c = None, math.inf
    for s in starts:
        try:
            p = _lm(f, s, x, y)
        except Exception:
            p = None
        if p is None:
            continue
        with np.errstate(all="ignore"):
            c = float(np.sum((f(x, p) - y) ** 2))
        if np.isfinite(c) and c < mejor_c:
            mejor, mejor_c = p, c
    return mejor


def _lstsq(bases):
    def aj(x, y):
        A = np.column_stack([b(x) for b in bases])
        return np.linalg.lstsq(A, y, rcond=None)[0]
    return aj


def _aj_exp(x, y):
    f = lambda x, p: p[0] * np.exp(p[1] * x)
    span = (x.max() - x.min()) or 1.0
    starts, s = [], (np.sign(np.median(y)) or 1.0)
    if np.all(s * y > 0):
        b, lna = np.polyfit(x, np.log(s * y), 1)
        starts.append([s * math.exp(lna), b])
    with np.errstate(all="ignore"):
        for b0 in (1 / span, -1 / span, 3 / span, -3 / span):
            a0 = float(np.mean(y) / np.mean(np.exp(b0 * x)))
            if np.isfinite(a0):
                starts.append([a0, b0])
    return _mejor_lm(f, starts, x, y)


def _aj_pot(x, y):
    f = lambda x, p: p[0] * np.power(x, p[1])
    starts, s = [[1, 1], [1, 2], [1, 0.5], [1, -1]], (np.sign(np.median(y)) or 1.0)
    if np.all(s * y > 0):
        b, lna = np.polyfit(np.log(x), np.log(s * y), 1)
        starts.insert(0, [s * math.exp(lna), b])
    return _mejor_lm(f, starts, x, y)


def _aj_seno(x, y):
    n, span = len(x), x.max() - x.min()
    if span <= 0:
        return None
    cand = []
    if n > 4000:                                                      # el barrido usa una submuestra
        idx = np.linspace(0, n - 1, 4000).astype(int)
        xg, yg = x[idx], y[idx]
    else:
        xg, yg = x, y
    for k in np.arange(0.5, max(2.0, min(len(xg) / 2, 200.0)), 0.25):      # ciclos dentro del rango
        b = 2 * math.pi * k / span
        A = np.column_stack([np.sin(b * xg), np.cos(b * xg), np.ones(len(xg))])
        c = np.linalg.lstsq(A, yg, rcond=None)[0]
        cand.append((float(np.sum((A @ c - yg) ** 2)), b, c))
    cand.sort(key=lambda t: t[0])
    starts = [[math.hypot(c[0], c[1]), b, math.atan2(c[1], c[0]), c[2]] for _, b, c in cand[:4]]
    return _mejor_lm(lambda x, p: p[0] * np.sin(p[1] * x + p[2]) + p[3], starts, x, y)


def _aj_gauss(x, y):
    f = lambda x, p: p[0] * np.exp(-((x - p[1]) ** 2) / (2 * p[2] ** 2))
    span = (x.max() - x.min()) or 1.0
    i = int(np.argmax(np.abs(y - np.median(y))))
    return _mejor_lm(f, [[y[i], x[i], span / d] for d in (6, 12, 3, 20)], x, y)


def _aj_logistica(x, y):
    f = lambda x, p: p[0] / (1 + np.exp(-p[1] * (x - p[2])))
    span = (x.max() - x.min()) or 1.0
    starts = []
    for L0 in (y.max() * 1.05, y.min() * 1.05, y.max() - y.min()):
        if L0 == 0:
            continue
        x0 = x[int(np.argmin(np.abs(y - 0.5 * L0)))]
        for k0 in (4 / span, -4 / span, 10 / span, -10 / span):
            starts.append([L0, k0, x0])
    return _mejor_lm(f, starts, x, y)


def _barrido_b(u, y, bs):
    """Para cada b de la rejilla ajusta y ≈ A·cos(b·u) + B·sin(b·u) + C (mínimos cuadrados cerrados,
    vectorizado). Devuelve el error cuadrático (SSE) de cada b."""
    m, sse = len(u), np.empty(len(bs))
    ys, sy, yy = y, float(np.sum(y)), float(np.sum(y * y))
    paso = max(1, int(2_000_000 // max(m, 1)))
    for i in range(0, len(bs), paso):
        ang = np.outer(bs[i:i + paso], u)
        c, s_ = np.cos(ang), np.sin(ang)
        G = np.empty((len(c), 3, 3))
        G[:, 0, 0], G[:, 1, 1] = np.sum(c * c, 1), np.sum(s_ * s_, 1)
        G[:, 0, 1] = G[:, 1, 0] = np.sum(c * s_, 1)
        G[:, 0, 2] = G[:, 2, 0] = np.sum(c, 1)
        G[:, 1, 2] = G[:, 2, 1] = np.sum(s_, 1)
        G[:, 2, 2] = m
        r = np.stack([c @ ys, s_ @ ys, np.full(len(c), sy)], axis=1)
        try:
            w = np.linalg.solve(G + 1e-12 * np.eye(3), r[..., None])[..., 0]
        except np.linalg.LinAlgError:
            sse[i:i + paso] = np.inf
            continue
        sse[i:i + paso] = yy - np.sum(w * r, axis=1)
    return sse


def _aj_coseno_pot(k):
    """Ajusta y ≈ a·cos(b·x^k + c) + d.  Una frecuencia b mal estimada desfasa muchísimo cuando x^k
    es grande, así que se ajusta por etapas: primero cerca del origen (barrido de b + LM) y luego se
    va ampliando el radio |x| ≤ r, partiendo siempre de la solución anterior."""
    f = lambda x, p: p[0] * np.cos(p[1] * np.power(x, k) + p[2]) + p[3]

    def aj(x, y):
        n, ax = len(x), np.abs(x)
        xmax = float(ax.max())
        if n < 40 or xmax <= 0:
            return None
        xs = np.sort(x)
        dx = max(float(np.median(np.diff(xs))), 1e-12)
        r = min(xmax, 12000 * 0.1 * k * dx / math.pi)       # ≈12 000 valores de b en la rejilla
        sub = ax <= r
        if sub.sum() < 30:
            r = float(np.sort(ax)[29])
            sub = ax <= r
        u = np.power(x[sub], k)
        bmax = math.pi / (k * r ** (k - 1) * dx)             # límite de Nyquist de esta etapa
        bs = np.arange(0.2 / r ** k, bmax, 0.2 / r ** k)
        if len(bs) < 3:
            return None
        sse = _barrido_b(u, y[sub], bs)
        elegidos = []
        for i in np.argsort(sse):
            if np.isfinite(sse[i]) and all(abs(bs[i] - bs[j]) > 3 * (bs[1] - bs[0]) for j in elegidos):
                elegidos.append(i)
            if len(elegidos) == 3:
                break
        cands = []
        for i in elegidos:
            b = bs[i]
            A = np.column_stack([np.cos(b * u), np.sin(b * u), np.ones(len(u))])
            w = np.linalg.lstsq(A, y[sub], rcond=None)[0]
            cands.append(np.array([math.hypot(w[0], w[1]), b, math.atan2(-w[1], w[0]), w[2]]))
        while True:
            sub = ax <= r
            cands = [q for q in (_lm(f, c, x[sub], y[sub]) for c in cands) if q is not None]
            if not cands or r >= xmax:
                break
            r = min(xmax, r * 1.6)
        if not cands:
            return None
        with np.errstate(all="ignore"):
            p = min(cands, key=lambda q: float(np.sum((f(x, q) - y) ** 2)))
        if p[0] < 0:                                         # forma canónica: a > 0
            p[0], p[2] = -p[0], p[2] + math.pi
        p[2] = (p[2] + math.pi) % (2 * math.pi) - math.pi
        return p
    return aj


def _expr_coseno(p, k):
    a, b, c, d = p
    interior = f"{_fnum(b)}*x^{k}" + ("" if abs(c) < 1e-8 else (f" + {_fnum(c)}" if c > 0 else f" - {_fnum(-c)}"))
    return _poli_str([a, d], [f"cos({interior})", ""])


def _fnum(v):
    return f"{float(v):.10g}"


def _poli_str(coefs, bases):
    """Une términos con el signo correcto:  2*x^2 - 0.5*x + 1"""
    partes = []
    for i, (c, b) in enumerate(zip(coefs, bases)):
        a = abs(float(c))
        if b == "":
            t = _fnum(a)
        elif b.startswith("1/"):
            t = f"{_fnum(a)}/{b[2:]}"
        elif abs(a - 1) < 1e-12:
            t = b
        else:
            t = f"{_fnum(a)}*{b}"
        partes.append((("-" if c < 0 else "") if i == 0 else (" - " if c < 0 else " + ")) + t)
    return "".join(partes)


def _x_menos(b):
    return f"(x - {_fnum(b)})" if b >= 0 else f"(x + {_fnum(-b)})"


def _bases_poli(d):
    return [f"x^{k}" for k in range(d, 1, -1)] + ["x", ""][(0 if d >= 1 else 1):] if d >= 1 else [""]


def _modelo_poli(nombre, d):
    return dict(nombre=nombre, np=d + 1, req=None,
                ajustar=(lambda x, y: np.array([np.mean(y)])) if d == 0 else (lambda x, y: np.polyfit(x, y, d)),
                expr=lambda p: _poli_str(p, _bases_poli(len(p) - 1)))


MODELOS = [
    _modelo_poli("Constante", 0),
    _modelo_poli("Lineal", 1),
    _modelo_poli("Cuadrática", 2),
    _modelo_poli("Cúbica", 3),
    _modelo_poli("Polinomio grado 4", 4),
    _modelo_poli("Polinomio grado 5", 5),
    dict(nombre="Logarítmica", np=2, req="pos", ajustar=_lstsq([np.log, np.ones_like]),
         expr=lambda p: _poli_str(p, ["ln(x)", ""])),
    dict(nombre="Raíz cuadrada", np=2, req="nonneg", ajustar=_lstsq([np.sqrt, np.ones_like]),
         expr=lambda p: _poli_str(p, ["sqrt(x)", ""])),
    dict(nombre="Inversa", np=2, req="nonzero", ajustar=_lstsq([lambda x: 1 / x, np.ones_like]),
         expr=lambda p: _poli_str(p, ["1/x", ""])),
    dict(nombre="Exponencial", np=2, req=None, ajustar=_aj_exp,
         expr=lambda p: f"{_fnum(p[0])}*exp({_fnum(p[1])}*x)"),
    dict(nombre="Potencia", np=2, req="pos", ajustar=_aj_pot,
         expr=lambda p: f"{_fnum(p[0])}*x^{_fnum(p[1])}"),
    dict(nombre="Senoidal", np=4, req=None, ajustar=_aj_seno,
         expr=lambda p: _poli_str([p[0], p[3]], [f"sin({_poli_str([p[1], p[2]], ['x', ''])})", ""])),
    dict(nombre="Coseno de x²", np=4, req=None, ajustar=_aj_coseno_pot(2), expr=lambda p: _expr_coseno(p, 2)),
    dict(nombre="Coseno de x³", np=4, req=None, ajustar=_aj_coseno_pot(3), expr=lambda p: _expr_coseno(p, 3)),
    dict(nombre="Gaussiana", np=3, req=None, ajustar=_aj_gauss,
         expr=lambda p: f"{_fnum(p[0])}*exp(-{_x_menos(p[1])}^2/(2*{_fnum(abs(p[2]))}^2))"),
    dict(nombre="Logística", np=3, req=None, ajustar=_aj_logistica,
         expr=lambda p: f"{_fnum(p[0])}/(1 + exp({_fnum(-p[1])}*{_x_menos(p[2])}))"),
]

_REQ = {"pos": (lambda x: np.all(x > 0), "x > 0"),
        "nonneg": (lambda x: np.all(x >= 0), "x ≥ 0"),
        "nonzero": (lambda x: np.all(x != 0), "x ≠ 0")}


def ajustar_modelos(x, y, max_alt):
    """Ajusta todos los modelos y calcula MSE, RMSE y NMSE de cada uno.
    Los modelos cuya expresión supera la altura máxima quedan excluidos."""
    out = []
    for sp in MODELOS:
        m = dict(nombre=sp["nombre"], nparams=sp["np"], n=len(x), expr="", arbol=None, altura=0, estado="",
                 elegible=False, mse=math.inf, rmse=math.inf, nmse=math.inf, r2=-math.inf, fvu=math.inf)
        out.append(m)
        if sp["req"] and not _REQ[sp["req"]][0](x):
            m["estado"] = f"no aplica (requiere {_REQ[sp['req']][1]})"
            continue
        if len(x) < sp["np"] + 2:
            m["estado"] = "pocos datos"
            continue
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                p = sp["ajustar"](x, y)
        except Exception:
            p = None
        if p is None or not np.all(np.isfinite(p)):
            m["estado"] = "no convergió"
            continue
        try:
            arbol = limpiar(construir_arbol(sp["expr"](p)))
        except Exception:
            m["estado"] = "error al construir la expresión"
            continue
        m.update(expr=a_infija(arbol), arbol=arbol, altura=altura(arbol))
        with np.errstate(all="ignore"):
            yh = np.asarray(arbol.evaluar(x), dtype=float)
        m.update(metricas(y, yh))
        if not np.isfinite(m["nmse"]):
            m["estado"] = "valores no finitos"
        elif m["altura"] > max_alt:
            m["estado"] = f"excluido: altura {m['altura']} > {max_alt}"
        else:
            m["elegible"] = True
    return out


PENALIZACION = 3.0      # más alto = más conservador (prefiere modelos simples)


def elegir_ganador(modelos, parsimonia=True):
    """Sin parsimonia: gana el menor NMSE (equivale al menor MSE / RMSE).
    Con parsimonia: criterio tipo BIC = n·ln(1−R²) + c·k·ln(n). Un modelo con más parámetros k solo
    gana si mejora el error lo suficiente; así un polinomio de grado 5 no «gana» ajustando el ruido."""
    el = [m for m in modelos if m["elegible"]]
    if not el:
        return None
    if not parsimonia:
        return min(el, key=lambda m: m["nmse"])
    bic = lambda m: m["n"] * math.log(max(m["fvu"], 1e-10)) + PENALIZACION * m["nparams"] * math.log(m["n"])
    return min(el, key=lambda m: (bic(m), m["nparams"]))


# ══════════════════════════════════════════════════════════════════════════
#  5c. ALGORITMO GENÉTICO · REGRESIÓN SIMBÓLICA (generacional, selección por RULETA)
# ══════════════════════════════════════════════════════════════════════════
#  Cada llamada a GP.paso() es UNA GENERACIÓN completa:
#    1. Evaluación : error SSE/SST (= 1 − R²; ordena igual que MSE, RMSE y NMSE) + pequeña penalización
#                    por tamaño.  Aptitud (a maximizar) = 1 / (error + 1e-4).
#    2. Selección  : RULETA. Cada árbol ocupa una porción proporcional a su aptitud:
#                    P(i) = aptitud_i / Σ aptitud.  Se gira la ruleta una vez por cada padre.
#    3. Cruce      : con probabilidad p_cruce se intercambian subárboles entre dos padres.
#    4. Mutación   : con probabilidad p_mut (de subárbol, de punto o de constante).
#    5. Reemplazo  : toda la población es sustituida por los hijos (se conservan los `elitismo` mejores).
#  Ningún árbol supera la «altura máxima» (cruce y mutación la respetan).
FUNCS_GP = ["sin", "cos", "exp", "ln", "sqrt", "abs"]
OPS_GP = ["+", "+", "-", "*", "*", "/", "^"]          # + − * se repiten: son los más útiles


def _gp_lista(t):
    """[(nodo, padre, índice_en_padre, profundidad), ...] en preorden (raíz con profundidad 0)."""
    res = []

    def rec(n, padre, i, d):
        res.append((n, padre, i, d))
        for k, h in enumerate(n.hijos):
            rec(h, n, k, d + 1)

    rec(t, None, 0, 0)
    return res


def _gp_reemplazar(raiz, padre, i, sub):
    if padre is None:
        return sub
    padre.hijos[i] = sub
    return raiz


class GP:
    def __init__(self, x, y, max_alt, pob=250, p_cruce=0.8, p_mut=0.3, elitismo=1, afinar=False,
                 semillas=(), semilla=None, max_puntos=400, paciencia=150):
        self.r = random.Random(semilla)
        self.max_alt = max(2, int(max_alt))
        self.pob_n = int(pob)
        self.p_cruce, self.p_mut = p_cruce, p_mut
        self.elit, self.afinar = max(0, int(elitismo)), afinar
        self.paciencia = paciencia
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) > max_puntos:                      # submuestra uniforme → generaciones rápidas
            idx = np.linspace(0, len(x) - 1, max_puntos).astype(int)
            x, y = x[idx], y[idx]
        self.x, self.y = x, y
        self.sst = float(np.sum((y - np.mean(y)) ** 2)) or 1.0
        self.gen = 0
        self.sin_mejora = 0
        self.mejor, self.mejor_fvu = None, math.inf
        self._acum, self._total = [], 0.0
        self.pob = self._poblacion_inicial(semillas)
        self._registrar_mejor()

    # ───────────────────────── aptitud ─────────────────────────
    def _fit(self, t):
        """Devuelve (error_penalizado, fvu). fvu = SSE/SST = 1 − R². Menor es mejor."""
        with np.errstate(all="ignore"):
            try:
                yh = np.asarray(t.evaluar(self.x), dtype=float)
            except Exception:
                return math.inf, math.inf
        if yh.shape != self.y.shape or not np.all(np.isfinite(yh)) or np.max(np.abs(yh)) > 1e12:
            return math.inf, math.inf
        fvu = float(np.sum((self.y - yh) ** 2)) / self.sst
        n = len(preorden(t))
        return fvu * (1 + 0.01 * n) + 1e-7 * n, fvu     # penalización relativa por tamaño

    def _indiv(self, t):
        f, fvu = self._fit(t)
        return [f, fvu, t]

    # ───────────────────────── generación de árboles ─────────────────────────
    def _hoja(self):
        r = self.r
        if r.random() < 0.55:
            return Nodo("var", "x")
        if r.random() < 0.5:
            return Nodo("num", float(r.randint(1, 9)))
        return Nodo("num", round(r.uniform(-5, 5), 3))

    def _gen(self, h, full):
        """Árbol de altura <= h (método 'full' o 'grow')."""
        r = self.r
        if h <= 1 or (not full and r.random() < 0.25):
            return self._hoja()
        if r.random() < 0.25:
            return Nodo("func", r.choice(FUNCS_GP), [self._gen(h - 1, full)])
        op = r.choice(OPS_GP)
        if op == "^":                                   # exponente pequeño y constante
            return Nodo("op", "^", [self._gen(h - 1, full), Nodo("num", float(r.choice([2, 2, 3, 0.5, -1])))])
        return Nodo("op", op, [self._gen(h - 1, full), self._gen(h - 1, full)])

    def _con_x(self, t):
        """Garantiza que el árbol dependa de x."""
        if not t.depende_de_x():
            hojas = [n for n, _ in preorden(t) if not n.hijos]
            n = self.r.choice(hojas)
            n.tipo, n.valor = "var", "x"
        return t

    def _aleatorio(self, i=None):
        i = self.r.randrange(10_000) if i is None else i
        h = 2 + i % (self.max_alt - 1)                  # «ramped half-and-half»: alturas 2..max_alt
        return self._con_x(self._gen(h, i % 2 == 0))

    def _poblacion_inicial(self, semillas):
        pob = []
        for s in semillas:                              # las familias ya ajustadas
            t = limpiar(s.copiar())
            if altura(t) <= self.max_alt:
                if self.afinar:
                    self._afinar(t, 30)
                pob.append(self._indiv(t))
        for s in list(pob):                             # variantes mutadas de las semillas
            for _ in range(min(6, max(1, self.pob_n // (8 * max(1, len(pob)))))):
                pob.append(self._indiv(self._normalizar(self._mutar(s[2]))))
        i = 0
        while len(pob) < self.pob_n:
            pob.append(self._indiv(self._normalizar(self._aleatorio(i))))
            i += 1
        return pob[:max(self.pob_n, len(pob))]

    # ───────────────────────── selección por RULETA ─────────────────────────
    def _preparar_ruleta(self):
        """Porción de ruleta de cada individuo: aptitud = 1/(error+1e-4); los no válidos (inf) valen 0."""
        pesos = [1.0 / (i[0] + 1e-4) if math.isfinite(i[0]) else 0.0 for i in self.pob]
        self._acum = list(itertools.accumulate(pesos))
        self._total = self._acum[-1] if self._acum else 0.0

    def _ruleta(self):
        """Un giro de la ruleta: devuelve el árbol elegido con probabilidad proporcional a su aptitud."""
        if self._total <= 0:
            return self.r.choice(self.pob)[2]
        k = bisect.bisect_left(self._acum, self.r.random() * self._total)
        return self.pob[min(k, len(self.pob) - 1)][2]

    # ───────────────────────── cruce y mutación ─────────────────────────
    def _elegir(self, lista):
        internos = [e for e in lista if e[0].hijos]
        if internos and self.r.random() < 0.9:
            return self.r.choice(internos)
        return self.r.choice(lista)

    def _cruce(self, a, b):
        hijo = a.copiar()
        n, padre, i, d = self._elegir(_gp_lista(hijo))
        lb = _gp_lista(b)
        cand = [e for e in lb if altura(e[0]) <= self.max_alt - d]   # la altura se respeta siempre
        if not cand:
            return hijo
        internos = [e for e in cand if e[0].hijos]
        sub = (self.r.choice(internos) if internos and self.r.random() < 0.9 else self.r.choice(cand))[0].copiar()
        return _gp_reemplazar(hijo, padre, i, sub)

    def _mutar(self, t):
        r = self.r
        t = t.copiar()
        lista = _gp_lista(t)
        n, padre, i, d = self._elegir(lista)
        k = r.random()
        if k < 0.4:                                     # mutación de subárbol
            sub = self._gen(r.randint(1, self.max_alt - d), r.random() < 0.5)
            return _gp_reemplazar(t, padre, i, sub)
        if k < 0.7:                                     # mutación de punto
            n = r.choice(lista)[0]
            if n.tipo == "op" and n.valor in ("+", "-", "*", "/"):
                n.valor = r.choice([o for o in ("+", "-", "*", "/") if o != n.valor])
            elif n.tipo == "func":
                n.valor = r.choice([f for f in FUNCS_GP if f != n.valor])
            elif n.tipo == "var":
                n.tipo, n.valor = "num", float(r.randint(1, 9))
            elif n.tipo == "num":
                if r.random() < 0.3:
                    n.tipo, n.valor = "var", "x"
                else:
                    n.valor = float(r.randint(1, 9)) if r.random() < 0.5 else round(r.uniform(-5, 5), 3)
            return t
        nums = [e[0] for e in lista if e[0].tipo == "num"]     # mutación de constante
        if not nums:
            sub = self._gen(r.randint(1, self.max_alt - d), False)
            return _gp_reemplazar(t, padre, i, sub)
        c = r.choice(nums)
        c.valor = c.valor + r.gauss(0, 1) * (0.1 * abs(c.valor) + 0.1)
        return t

    def _plegar(self, n):
        """Los subárboles sin x se sustituyen por su valor (2*3 → 6): árboles más bajos y legibles."""
        n.hijos = [self._plegar(h) for h in n.hijos]
        if n.hijos and not n.depende_de_x():
            with np.errstate(all="ignore"):
                v = float(n.evaluar(np.float64(0.0)))
            if math.isfinite(v):
                return Nodo("num", v)
        return n

    def _normalizar(self, t):
        t = limpiar(self._plegar(t))
        if altura(t) > self.max_alt:                    # no debería ocurrir; por seguridad
            t = self._aleatorio()
        return t

    # ───────────────────────── constantes (afinado opcional y redondeo) ─────────────────────────
    def _afinar(self, t, iters=40):
        hojas = [n for n, _ in preorden(t) if n.tipo == "num"]
        if not hojas or len(hojas) > 10:
            return
        p0 = np.array([n.valor for n in hojas], float)
        f0 = self._fit(t)[0]

        def f(x, p):
            for n, v in zip(hojas, p):
                n.valor = float(v)
            return np.asarray(t.evaluar(x), dtype=float)

        try:
            p = _lm(f, p0, self.x, self.y, max_iter=iters)
            ok = p is not None and np.all(np.isfinite(p))
            if ok:
                f(self.x, p)
                ok = self._fit(t)[0] < f0
        except Exception:
            ok = False
        if not ok:
            f(self.x, p0)

    def _redondear(self, t):
        """Constantes «bonitas» (2.5 en vez de 2.4999997) si no empeoran el error."""
        for n, _ in preorden(t):
            if n.tipo != "num":
                continue
            base = self._fit(t)[1]
            viejo = n.valor
            for cand in (round(viejo), float(f"{viejo:.3g}"), float(f"{viejo:.5g}")):
                n.valor = float(cand)
                if self._fit(t)[1] <= base * (1 + 1e-6) + 1e-12:
                    break
                n.valor = viejo

    # ───────────────────────── bucle evolutivo ─────────────────────────
    def _registrar_mejor(self):
        self.pob.sort(key=lambda i: i[0])
        f, fvu, t = self.pob[0]
        if fvu < self.mejor_fvu * (1 - 1e-6) - 1e-12 or self.mejor is None:
            self.sin_mejora = 0
        else:
            self.sin_mejora += 1
        if self.mejor is None or fvu <= self.mejor_fvu:
            self.mejor, self.mejor_fvu = t.copiar(), fvu

    def paso(self):
        """Una generación: ruleta → cruce → mutación → reemplazo de toda la población."""
        r = self.r
        if self.afinar and self.gen % 4 == 0:
            for ind in self.pob[:2]:
                self._afinar(ind[2], 40)
                ind[0], ind[1] = self._fit(ind[2])
            self.pob.sort(key=lambda i: i[0])
        self._preparar_ruleta()
        nueva = [list(i) for i in sorted(self.pob, key=lambda i: i[0])[:self.elit]]
        while len(nueva) < self.pob_n:
            if r.random() < self.p_cruce:
                hijo = self._cruce(self._ruleta(), self._ruleta())
            else:
                hijo = self._ruleta().copiar()
            if r.random() < self.p_mut:
                hijo = self._mutar(hijo)
            nueva.append(self._indiv(self._normalizar(hijo)))
        self.pob = nueva
        self.gen += 1
        self._registrar_mejor()

    def terminado(self, max_gen):
        return (self.gen >= max_gen or self.mejor_fvu < 1e-12 or self.sin_mejora >= self.paciencia)

    def pulir(self):
        """Constantes redondeadas (y afinado final si afinar=True). Deja el resultado en self.mejor."""
        t = limpiar(self._plegar(self.mejor.copiar()))
        if self.afinar:
            self._afinar(t, 200)
        self._redondear(t)
        f, fvu = self._fit(t)
        if fvu <= self.mejor_fvu * (1 + 1e-6) + 1e-12 and altura(t) <= self.max_alt:
            self.mejor, self.mejor_fvu = t, fvu
        return self.mejor


# ══════════════════════════════════════════════════════════════════════════
#  6. DIBUJO (matplotlib)
# ══════════════════════════════════════════════════════════════════════════
COLORES = {  # tipo → (relleno, borde)
    "op": ("#FFE0B2", "#EF6C00"),
    "func": ("#E1BEE7", "#7B1FA2"),
    "num": ("#C8E6C9", "#2E7D32"),
    "const": ("#C8E6C9", "#2E7D32"),
    "var": ("#BBDEFB", "#1565C0"),
}


def calcular_posiciones(raiz):
    """Hojas equiespaciadas; cada padre centrado sobre sus hijos."""
    pos, cont = {}, [0]

    def rec(n, prof):
        if not n.hijos:
            x = cont[0]
            cont[0] += 1
        else:
            xs = [rec(h, prof + 1) for h in n.hijos]
            x = sum(xs) / len(xs)
        pos[id(n)] = (x, -prof)
        return x

    rec(raiz, 0)
    return pos


def dibujar_arbol(ax, raiz, x0, valores=True, numerar=False, paso=None):
    """paso=None → vista estática.  paso=k → animación del preorden hasta el nodo k."""
    ax.clear()
    ax.set_axis_off()
    pos = calcular_posiciones(raiz)
    orden = [n for n, _ in preorden(raiz)]
    idx = {id(n): i for i, n in enumerate(orden)}
    hojas = sum(1 for n in orden if not n.hijos)
    fs = max(8.0, min(15.0, 17.0 - 0.55 * hojas))
    anim = paso is not None

    for n in orden:                                             # aristas
        for h in n.hijos:
            (xa, ya), (xb, yb) = pos[id(n)], pos[id(h)]
            vis = anim and idx[id(h)] <= paso
            ax.plot([xa, xb], [ya, yb], color="#EF6C00" if vis else "#B0BEC5",
                    lw=3.6 if vis else 1.8, zorder=1, solid_capstyle="round",
                    alpha=0.45 if (anim and not vis) else 1)

    textos = []
    for n in orden:                                             # nodos
        x, y = pos[id(n)]
        fondo, borde = COLORES[n.tipo]
        i = idx[id(n)]
        vis = (not anim) or i <= paso
        actual = anim and i == paso
        alpha = 1.0 if vis else 0.28
        if actual:
            fondo = "#FFF59D"
        t = ax.text(x, y, n.etiqueta, ha="center", va="center", fontsize=fs,
                    fontweight="bold", color=borde, alpha=alpha, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.38", fc=fondo,
                              ec="#D32F2F" if actual else borde,
                              lw=3.6 if actual else 2, alpha=alpha))
        textos.append((t, n))
        if valores and vis:
            v = _f_escalar(n, x0)
            ax.annotate(f"= {fmt(v, 4)}", (x, y), xytext=(0, -(fs + 9)),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=fs - 3, zorder=2.5,
                        color="#C62828" if not np.isfinite(v) else "#455A64",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        if (numerar and not anim) or (anim and vis):
            dx = 0.32 * fs * len(n.etiqueta) + fs * 0.95
            ax.annotate(str(i + 1), (x, y), xytext=(-dx, fs * 1.15 + 3),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=max(7, fs - 4), color="white", fontweight="bold", zorder=4,
                        bbox=dict(boxstyle="circle,pad=0.25", ec="none",
                                  fc="#D32F2F" if actual else "#37474F"))

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 1.05, 0.75)
    etiquetas = {"op": "Operador", "func": "Función", "num": "Número / constante", "var": "Variable"}
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", ms=10, mfc=COLORES[k][0],
                              mec=COLORES[k][1], mew=2, label=v) for k, v in etiquetas.items()],
              loc="lower center", ncol=4, frameon=False, fontsize=9)
    return textos


def dibujar_grafica(ax, raiz, deriv, x0, rango, ver_deriv, ver_tang, ver_raices, raices, datos=None):
    ax.clear()
    xs = np.linspace(rango[0], rango[1], 2500)
    ys = raiz.evaluar(xs).astype(float)
    ys[~np.isfinite(ys)] = np.nan
    dys = None
    if deriv is not None and ver_deriv:
        dys = deriv.evaluar(xs).astype(float)
        dys[~np.isfinite(dys)] = np.nan

    ax.axhline(0, color="#90A4AE", lw=1)
    ax.axvline(0, color="#90A4AE", lw=1)
    ax.plot(xs, ys, color="#1565C0", lw=2.8, label="f(x)")
    if dys is not None:
        ax.plot(xs, dys, color="#8E24AA", lw=1.9, ls="-.", label="f′(x)")

    nan = np.isnan(ys)
    if nan.any() and not nan.all():
        ax.fill_between(xs, 0, 1, where=nan, transform=ax.get_xaxis_transform(),
                        color="#FFCDD2", alpha=0.5, label="fuera del dominio")

    conj = [ys] + ([dys] if dys is not None else []) + ([np.asarray(datos[1], float)] if datos else [])
    fin = np.concatenate([c[np.isfinite(c)] for c in conj])
    if fin.size:
        lo, hi = np.percentile(fin, [1, 99])
        m = (hi - lo) * 0.2 or 1.0
        ylim = (min(lo, 0) - m, max(hi, 0) + m)
    else:
        ylim = (-1, 1)
    ax.set_xlim(*rango)
    ax.set_ylim(*ylim)

    if datos:
        ax.scatter(datos[0], datos[1], s=24, color="#455A64", alpha=0.75, zorder=4,
                   label=f"datos CSV (n={len(datos[0])})")
    y0 = _f_escalar(raiz, x0)
    if np.isfinite(y0):
        if ver_tang and deriv is not None:
            m = _f_escalar(deriv, x0)
            if np.isfinite(m):
                xt = np.array(rango)
                ax.plot(xt, y0 + m * (xt - x0), color="#2E7D32", ls="--", lw=1.8,
                        label=f"tangente (m = {fmt(m, 4)})")
        ax.vlines(x0, 0, y0, color="#EF6C00", ls="--", lw=1.4)
        ax.scatter([x0], [y0], s=100, color="#EF6C00", zorder=6, ec="white", lw=1.6,
                   label=f"f({fmt(x0, 4)}) = {fmt(y0, 5)}")
    if ver_raices:
        for r in raices[:12]:
            ax.scatter([r], [0], s=55, color="#D32F2F", zorder=5, ec="white", lw=1.2)
            ax.annotate(fmt(r, 4), (r, 0), xytext=(0, -15), textcoords="offset points",
                        ha="center", fontsize=8.5, color="#B71C1C")
        if raices:
            ax.scatter([], [], s=55, color="#D32F2F", label="raíces")

    ax.set_title(f"f(x) = {a_infija(raiz)}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="best", fontsize=9, framealpha=0.92)


# ══════════════════════════════════════════════════════════════════════════
#  7. APLICACIÓN
# ══════════════════════════════════════════════════════════════════════════
BG, HEADER, ACCENT = "#F4F6F8", "#1F2D3A", "#1565C0"

EJEMPLOS = [
    "sqrt(2*(x2+3)+5*x-3)", "x^2 - 4", "sin(x)/x", "(x+1)*(x-2)/(x-3)",
    "exp(-x^2)", "ln(x) + sqrt(x)", "abs(x) - 2", "2*sin(x) + cos(2*x)", "x^3 - 3x",
    "-x^2 + 3x/(x-1)",
]

AYUDA = """SINTAXIS INFIJA
  Operadores:  +  -  *  /  ^   (también ** y x2 = x^2)
  Multiplicación implícita:  2x,  3(x+1),  (x+1)(x-1)
  Funciones:  sqrt  sin  cos  tan  exp  ln  log  log10  abs
  Constantes: pi  e          Variable: x

SINTAXIS PREFIJA (PREORDEN)
  Tokens separados por espacios; el operador va ANTES de sus operandos.
  Signo menos unario:  neg
  Ejemplo:  sqrt - + * 2 + ^ x 2 3 * 5 x 3
  equivale a  sqrt(2*(x^2+3)+5*x-3)

DATOS CSV
  «Generar CSV…» crea los puntos (x, y) de TU f(x) (x de … a …, paso, ruido %), los guarda y los analiza.
  Con datos densos (p. ej. cos(x³) con paso 0.001) usa «Ver x de … a …» para hacer zoom.
  Pestaña «Datos (CSV)» → «Abrir CSV…» (columnas X e Y numéricas; separador y coma decimal se detectan solos).
  La app ajusta 16 familias de funciones y las compara con:
    MSE = media((y − ŷ)²)    RMSE = √MSE    NMSE = Σ|y−ŷ|² / Σ|y|²    R² = 1 − Σ(y−ŷ)²/Σ(y−ȳ)²
  Gana el modelo de menor NMSE; con «parsimonia» se usa un criterio tipo BIC, que penaliza los parámetros
  extra (evita que un polinomio de grado 5 gane solo por ajustar el ruido).
  Los modelos con altura > «Altura máx.» se excluyen.

EVOLUTIVO (ALGORITMO GENÉTICO · REGRESIÓN SIMBÓLICA)
  «Evolucionar» crea una población de árboles de expresión y la renueva GENERACIÓN a generación:
  1) selección por RULETA (probabilidad proporcional a la aptitud 1/(error+1e-4)), 2) cruce de subárboles,
  3) mutación (subárbol / punto / constante), 4) la población completa se reemplaza por los hijos
  (se conserva el mejor: elitismo). El error es SSE/SST = 1 − R² (ordena igual que MSE, RMSE y NMSE).
  Ningún árbol supera «Altura máx.». «Sembrar» arranca con las 16 familias ya ajustadas.
  El mejor aparece en la tabla como «Evolutivo (GP)» y puedes usarlo con «Usar como f(x) → árbol».

ÁRBOL
  Clic en un nodo → muestra su subexpresión y su valor.
  «Reproducir» anima el recorrido preorden: raíz → izquierdo → derecho.
"""


class App:
    def __init__(self, root):
        self.root = root
        root.title("Árbol de Expresión · Recorrido Preorden")
        root.geometry("1340x880")
        root.minsize(1100, 700)
        root.configure(bg=BG)

        self.raiz = self.deriv = None
        self.paso = None
        self.anim_id = None
        self._pend = None
        self._live = None
        self._lock = False
        self._textos = []
        self.raices, self.indef = [], []
        self.rangos_tok = []

        self.var_modo = tk.StringVar(value="infija")
        self.var_expr = tk.StringVar(value=EJEMPLOS[0])
        self.var_x = tk.StringVar(value="2")
        self.var_min = tk.StringVar(value="-10")
        self.var_max = tk.StringVar(value="10")
        self.var_arbol = tk.StringVar(value="f")
        self.var_valores = tk.BooleanVar(value=True)
        self.var_numerar = tk.BooleanVar(value=True)
        self.var_deriv = tk.BooleanVar(value=False)
        self.var_tang = tk.BooleanVar(value=True)
        self.var_raices = tk.BooleanVar(value=True)
        self.var_vel = tk.DoubleVar(value=0.8)
        self.var_alt = tk.StringVar(value="8")
        self.dx = self.dy = None                 # datos del CSV
        self.csv_nombres, self.csv_cols, self.nombre_datos = [], [], ""
        self.modelos, self.ganador, self.sel = [], None, None
        self.var_xcol, self.var_ycol = tk.StringVar(), tk.StringVar()
        self.var_parsi = tk.BooleanVar(value=True)
        self.var_mia = tk.BooleanVar(value=True)
        self.var_datos_graf = tk.BooleanVar(value=True)
        self.var_gmin, self.var_gmax = tk.StringVar(value="-10"), tk.StringVar(value="10")
        self.var_gstep, self.var_gruido = tk.StringVar(value="0.001"), tk.StringVar(value="0")
        self.var_zmin, self.var_zmax = tk.StringVar(), tk.StringVar()
        # --- algoritmo genético ---
        self.var_pob, self.var_gens = tk.StringVar(value="250"), tk.StringVar(value="200")
        self.var_pc, self.var_pm = tk.StringVar(value="0.8"), tk.StringVar(value="0.3")
        self.var_sembrar = tk.BooleanVar(value=True)
        self.gp = self.gp_id = self.gp_modelo = None
        self.gp_max = 200

        self._estilos()
        self._menu()
        self._ui()
        self.calcular()

    # ───────────────────────── estilos y menú ─────────────────────────
    def _estilos(self):
        for nombre, tam in (("TkDefaultFont", 10), ("TkTextFont", 10), ("TkFixedFont", 11)):
            tkfont.nametofont(nombre).configure(size=tam)
        st = ttk.Style()
        st.theme_use("clam")
        st.configure(".", background=BG)
        st.configure("TLabelframe", background=BG, bordercolor="#CFD8DC")
        st.configure("TLabelframe.Label", background=BG, foreground=ACCENT,
                     font=("TkDefaultFont", 10, "bold"))
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", padding=(16, 7), font=("TkDefaultFont", 10, "bold"))
        st.map("TNotebook.Tab", background=[("selected", "white")],
               foreground=[("selected", ACCENT)])
        st.configure("Accent.TButton", background=ACCENT, foreground="white",
                     font=("TkDefaultFont", 11, "bold"), padding=7)
        st.map("Accent.TButton", background=[("active", "#0D47A1")])
        st.configure("Treeview", rowheight=25, font=("TkDefaultFont", 10))
        st.configure("Treeview.Heading", font=("TkDefaultFont", 10, "bold"))
        st.configure("Big.TLabel", font=("TkDefaultFont", 19, "bold"), foreground=ACCENT)
        st.configure("Err.TLabel", foreground="#C62828")
        st.configure("Hint.TLabel", foreground="#607D8B", font=("TkDefaultFont", 9))
        st.configure("TCheckbutton", background=BG)
        st.configure("TRadiobutton", background=BG)

    def _menu(self):
        m = tk.Menu(self.root)
        a = tk.Menu(m, tearoff=0)
        a.add_command(label="Abrir CSV…", command=self.abrir_csv)
        a.add_command(label="Guardar datos como CSV…", command=self.guardar_csv)
        a.add_separator()
        a.add_command(label="Guardar imagen del árbol…", command=lambda: self.guardar(self.fig_arbol))
        a.add_command(label="Guardar imagen de la gráfica…", command=lambda: self.guardar(self.fig_graf))
        a.add_command(label="Guardar imagen de datos/ajuste…", command=lambda: self.guardar(self.fig_datos))
        a.add_separator()
        a.add_command(label="Salir", command=self.root.destroy)
        m.add_cascade(label="Archivo", menu=a)
        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Sintaxis y ayuda", command=lambda: messagebox.showinfo("Ayuda", AYUDA))
        m.add_cascade(label="Ayuda", menu=h)
        self.root.config(menu=m)

    # ───────────────────────────── interfaz ─────────────────────────────
    def _ui(self):
        cab = tk.Frame(self.root, bg=HEADER)
        cab.pack(fill="x")
        tk.Label(cab, text="Árbol de Expresión", bg=HEADER, fg="white",
                 font=("TkDefaultFont", 17, "bold")).pack(side="left", padx=18, pady=10)
        tk.Label(cab, text="parser · recorrido preorden · evaluación · gráfica · derivada · algoritmo genético",
                 bg=HEADER, fg="#90A4AE").pack(side="left", pady=10)

        cuerpo = tk.Frame(self.root, bg=BG)
        cuerpo.pack(fill="both", expand=True, padx=10, pady=10)
        izq = tk.Frame(cuerpo, bg=BG, width=360)
        izq.pack(side="left", fill="y", padx=(0, 10))
        izq.pack_propagate(False)
        self._panel_izq(izq)
        nb = ttk.Notebook(cuerpo)
        nb.pack(side="left", fill="both", expand=True)
        self._tab_arbol(nb)
        self.nb = nb
        self._tab_grafica(nb)
        self._tab_datos(nb)
        self._tab_recorridos(nb)
        self._tab_evaluacion(nb)
        self._tab_derivada(nb)
        self._tab_tabla(nb)

    def _panel_izq(self, p):
        # 1. ecuación
        f1 = ttk.Labelframe(p, text=" 1 · Ecuación f(x) ", padding=7)
        f1.pack(fill="x")
        fila = ttk.Frame(f1)
        fila.pack(fill="x")
        ttk.Radiobutton(fila, text="Infija", value="infija", variable=self.var_modo,
                        command=self._cambiar_modo).pack(side="left")
        ttk.Radiobutton(fila, text="Prefija (preorden)", value="prefija", variable=self.var_modo,
                        command=self._cambiar_modo).pack(side="left", padx=12)
        self.entrada = ttk.Entry(f1, textvariable=self.var_expr, font=("TkFixedFont", 13))
        self.entrada.pack(fill="x", pady=(8, 4), ipady=4)
        self.entrada.bind("<Return>", lambda e: self.calcular())
        self.entrada.bind("<KeyRelease>", self._escritura_viva)
        self.lbl_hint = ttk.Label(f1, style="Hint.TLabel", wraplength=320, justify="left")
        self.lbl_hint.pack(fill="x")
        self.combo = ttk.Combobox(f1, state="readonly")
        self.combo.pack(fill="x", pady=(8, 6))
        self.combo.bind("<<ComboboxSelected>>", self._elegir_ejemplo)
        ttk.Button(f1, text="Calcular  (Enter)", style="Accent.TButton",
                   command=self.calcular).pack(fill="x")
        fila = ttk.Frame(f1)
        fila.pack(fill="x", pady=(8, 0))
        ttk.Label(fila, text="Altura máx.").pack(side="left")
        self.spin_alt = ttk.Spinbox(fila, from_=2, to=12, width=4, textvariable=self.var_alt,
                                    command=self._cambio_alt)
        self.spin_alt.pack(side="left", padx=6)
        self.spin_alt.bind("<Return>", lambda e: self._cambio_alt())
        self.spin_alt.bind("<FocusOut>", lambda e: self._cambio_alt())
        ttk.Button(fila, text="Generar aleatoria", command=self.aleatoria).pack(side="right")
        self.lbl_err = ttk.Label(f1, style="Err.TLabel", wraplength=330, justify="left")
        self.lbl_err.pack(fill="x", pady=(6, 0))

        # 2. valor de x
        f2 = ttk.Labelframe(p, text=" 2 · Evaluar en x ", padding=7)
        f2.pack(fill="x", pady=(8, 0))
        fila = ttk.Frame(f2)
        fila.pack(fill="x")
        ttk.Label(fila, text="x =").pack(side="left")
        e = ttk.Entry(fila, textvariable=self.var_x, width=10, font=("TkFixedFont", 12))
        e.pack(side="left", padx=6)
        e.bind("<Return>", lambda ev: self.refrescar())
        e.bind("<FocusOut>", lambda ev: self.refrescar())
        self.escala = ttk.Scale(fila, from_=-10, to=10, command=self._mover_escala)
        self.escala.pack(side="left", fill="x", expand=True, padx=(6, 0))
        fila = ttk.Frame(f2)
        fila.pack(fill="x", pady=(8, 0))
        ttk.Label(fila, text="Rango gráfica:").pack(side="left")
        for var in (self.var_min, self.var_max):
            en = ttk.Entry(fila, textvariable=var, width=6)
            en.pack(side="left", padx=4)
            en.bind("<Return>", lambda ev: self.cambiar_rango())
            en.bind("<FocusOut>", lambda ev: self.cambiar_rango())
            if var is self.var_min:
                ttk.Label(fila, text="a").pack(side="left")

        # 3. resultado
        f3 = ttk.Labelframe(p, text=" Resultado ", padding=7)
        f3.pack(fill="x", pady=(8, 0))
        self.lbl_res = ttk.Label(f3, style="Big.TLabel")
        self.lbl_res.pack(anchor="w")
        self.lbl_der = ttk.Label(f3, foreground="#8E24AA", font=("TkDefaultFont", 11, "bold"))
        self.lbl_der.pack(anchor="w")
        self.lbl_int = ttk.Label(f3, style="Hint.TLabel", wraplength=330, justify="left")
        self.lbl_int.pack(anchor="w", pady=(4, 0))

        # 4. preorden
        f4 = ttk.Labelframe(p, text=" Preorden (notación prefija) ", padding=7)
        f4.pack(fill="x", pady=(8, 0))
        self.txt_pre = tk.Text(f4, height=3, wrap="word", font=("TkFixedFont", 11), relief="flat",
                               bg="white", highlightthickness=1, highlightbackground="#CFD8DC")
        self.txt_pre.tag_configure("actual", background="#FFEB3B", foreground="#B71C1C")
        self.txt_pre.pack(side="left", fill="x", expand=True)
        ttk.Button(f4, text="Copiar", width=7, command=self._copiar_pre).pack(side="right", anchor="n", padx=(6, 0))

        # 5. info
        f5 = ttk.Labelframe(p, text=" Dominio y raíces ", padding=7)
        f5.pack(fill="x", pady=(8, 0))
        self.lbl_info = ttk.Label(f5, wraplength=330, justify="left")
        self.lbl_info.pack(fill="x")

        self._cambiar_modo(inicial=True)

    # ---------- pestañas ----------
    def _canvas(self, parent, fig, toolbar=False):
        c = FigureCanvasTkAgg(fig, master=parent)
        if toolbar:
            tb = NavigationToolbar2Tk(c, parent, pack_toolbar=False)
            tb.update()
            tb.pack(side="bottom", fill="x")
        c.get_tk_widget().pack(fill="both", expand=True)
        return c

    def _tab_arbol(self, nb):
        t = ttk.Frame(nb, padding=6)
        nb.add(t, text="Árbol")
        b1 = ttk.Frame(t)
        b1.pack(fill="x")
        ttk.Label(b1, text="Mostrar árbol de:").pack(side="left")
        for txt, val in (("f(x)", "f"), ("f′(x)", "df")):
            ttk.Radiobutton(b1, text=txt, value=val, variable=self.var_arbol,
                            command=self._cambio_arbol).pack(side="left", padx=6)
        ttk.Checkbutton(b1, text="valores por nodo", variable=self.var_valores,
                        command=self.dibujar_arbol_actual).pack(side="left", padx=(16, 4))
        ttk.Checkbutton(b1, text="numerar preorden", variable=self.var_numerar,
                        command=self.dibujar_arbol_actual).pack(side="left", padx=4)

        b2 = ttk.Frame(t)
        b2.pack(fill="x", pady=(6, 0))
        ttk.Label(b2, text="Recorrido preorden:", font=("TkDefaultFont", 10, "bold"),
                  foreground=ACCENT).pack(side="left", padx=(0, 8))
        for txt, cmd in (("⏮ Reiniciar", self.anim_reiniciar), ("◀ Anterior", self.anim_ant),
                         ("Siguiente ▶", self.anim_sig), ("▶▶ Reproducir", self.anim_play),
                         ("■ Detener", self.detener_anim)):
            ttk.Button(b2, text=txt, command=cmd).pack(side="left", padx=2)
        ttk.Label(b2, text="  velocidad").pack(side="left")
        ttk.Scale(b2, from_=1.6, to=0.15, variable=self.var_vel, length=110).pack(side="left", padx=4)

        self.fig_arbol = Figure(facecolor="white")
        self.ax_arbol = self.fig_arbol.add_axes([0.01, 0.01, 0.98, 0.98])
        self.canvas_arbol = self._canvas(t, self.fig_arbol)
        self.canvas_arbol.mpl_connect("button_press_event", self._clic_arbol)

        self.lbl_paso = ttk.Label(t, wraplength=900, justify="left", font=("TkDefaultFont", 10, "bold"),
                                  foreground="#37474F", padding=(4, 6, 4, 0))
        self.lbl_paso.pack(fill="x")
        self.lbl_nodo = ttk.Label(t, wraplength=900, justify="left", foreground="#1565C0", padding=(4, 0))
        self.lbl_nodo.pack(fill="x")

    def _tab_grafica(self, nb):
        t = ttk.Frame(nb, padding=6)
        nb.add(t, text="Gráfica")
        b = ttk.Frame(t)
        b.pack(fill="x")
        for txt, var in (("Mostrar f′(x)", self.var_deriv), ("Recta tangente en x", self.var_tang),
                         ("Raíces", self.var_raices), ("Datos CSV", self.var_datos_graf)):
            ttk.Checkbutton(b, text=txt, variable=var, command=self.dibujar_grafica_actual).pack(
                side="left", padx=6)
        self.fig_graf = Figure(facecolor="white")
        self.ax_graf = self.fig_graf.add_subplot(111)
        self.fig_graf.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.09)
        self.canvas_graf = self._canvas(t, self.fig_graf, toolbar=True)

    def _tab_datos(self, nb):
        t = ttk.Frame(nb, padding=6)
        nb.add(t, text="Datos (CSV)")
        b1 = ttk.Frame(t)
        b1.pack(fill="x")
        ttk.Button(b1, text="Abrir CSV…", style="Accent.TButton", command=self.abrir_csv).pack(side="left")
        mb = ttk.Menubutton(b1, text="Datos de ejemplo ▾")
        menu = tk.Menu(mb, tearoff=0)
        for nombre in TIPOS_EJEMPLO:
            menu.add_command(label=nombre, command=lambda n=nombre: self.cargar_ejemplo(n))
        mb["menu"] = menu
        mb.pack(side="left", padx=6)
        ttk.Label(b1, text="  X:").pack(side="left")
        self.cb_x = ttk.Combobox(b1, textvariable=self.var_xcol, state="readonly", width=8)
        self.cb_x.pack(side="left")
        ttk.Label(b1, text="  Y:").pack(side="left")
        self.cb_y = ttk.Combobox(b1, textvariable=self.var_ycol, state="readonly", width=8)
        self.cb_y.pack(side="left")
        for cb in (self.cb_x, self.cb_y):
            cb.bind("<<ComboboxSelected>>", self._aplicar_columnas)
        self.lbl_csv = ttk.Label(b1, foreground="#37474F", font=("TkDefaultFont", 9, "bold"))
        self.lbl_csv.pack(side="left", padx=8)

        b2 = ttk.Frame(t)
        b2.pack(fill="x", pady=(6, 0))
        ttk.Button(b2, text="Identificar función", command=self.ajustar_datos).pack(side="left")
        ttk.Button(b2, text="Usar como f(x) → árbol", command=self.usar_modelo).pack(side="left", padx=6)
        ttk.Button(b2, text="Rango = datos", command=self.rango_a_datos).pack(side="left")
        ttk.Checkbutton(b2, text="Parsimonia (preferir simple)", variable=self.var_parsi,
                        command=self._reelegir).pack(side="left", padx=(10, 2))
        ttk.Checkbutton(b2, text="Mi f(x)", variable=self.var_mia,
                        command=self._dibujar_datos).pack(side="left", padx=4)
        ttk.Label(t, style="Hint.TLabel", text=(
            "MSE = media((y−ŷ)²)    RMSE = √MSE    NMSE = Σ|y−ŷ|² / Σ|y|²    "
            "R² = 1 − Σ(y−ŷ)²/Σ(y−ȳ)²")).pack(anchor="w", pady=(4, 0))

        g = ttk.Labelframe(t, text=" Generar CSV desde mi f(x)  (la ecuación del panel izquierdo) ", padding=5)
        g.pack(fill="x", pady=(6, 0))
        for txt, var, w in (("x de", self.var_gmin, 7), ("a", self.var_gmax, 7),
                            ("paso", self.var_gstep, 8), ("ruido %", self.var_gruido, 5)):
            ttk.Label(g, text=txt).pack(side="left", padx=(6, 2))
            ttk.Entry(g, textvariable=var, width=w).pack(side="left")
        ttk.Button(g, text="Generar CSV…", style="Accent.TButton", command=self.generar_csv).pack(side="left", padx=10)
        ttk.Button(g, text="Guardar CSV…", command=self.guardar_csv).pack(side="left")

        ev = ttk.Labelframe(t, text=" Algoritmo genético · regresión simbólica (ruleta + cruce + mutación) ",
                            padding=5)
        ev.pack(fill="x", pady=(6, 0))
        fila = ttk.Frame(ev)
        fila.pack(fill="x")
        for txt, var, w in (("población", self.var_pob, 6), ("generaciones", self.var_gens, 6),
                            ("P. cruce", self.var_pc, 5), ("P. mutación", self.var_pm, 5)):
            ttk.Label(fila, text=txt).pack(side="left", padx=(6, 2))
            ttk.Entry(fila, textvariable=var, width=w).pack(side="left")
        ttk.Checkbutton(fila, text="Sembrar con los 16 modelos", variable=self.var_sembrar).pack(side="left", padx=8)
        ttk.Button(fila, text="Evolucionar", style="Accent.TButton", command=self.evolucionar).pack(side="left", padx=4)
        ttk.Button(fila, text="■ Detener", command=self.detener_gp).pack(side="left")
        self.lbl_gp = ttk.Label(ev, style="Hint.TLabel", wraplength=1000, justify="left")
        self.lbl_gp.pack(anchor="w")

        z = ttk.Frame(t)
        z.pack(fill="x", pady=(4, 0))
        ttk.Label(z, text="Ver x de").pack(side="left")
        for i, var in enumerate((self.var_zmin, self.var_zmax)):
            en = ttk.Entry(z, textvariable=var, width=8)
            en.pack(side="left", padx=4)
            en.bind("<Return>", lambda ev: self._dibujar_datos())
            if i == 0:
                ttk.Label(z, text="a").pack(side="left")
        ttk.Button(z, text="Aplicar zoom", command=self._dibujar_datos).pack(side="left", padx=4)
        ttk.Button(z, text="Ver todo", command=self._zoom_todo).pack(side="left")

        fr, self.tv_rank = self._tabla(
            t, ("Modelo", "Expresión ajustada", "Par.", "MSE", "RMSE", "NMSE", "R²", "Estado"),
            (125, 240, 34, 78, 78, 78, 60, 170), 5)
        fr.pack(side="bottom", fill="x")
        self.tv_rank.bind("<<TreeviewSelect>>", self._sel_fila)
        self.tv_rank.tag_configure("ganador", background="#C8E6C9")
        self.tv_rank.tag_configure("mia", background="#FFE0B2")
        self.tv_rank.tag_configure("gris", foreground="#90A4AE")
        self.lbl_cmp = ttk.Label(t, foreground="#6A1B9A", font=("TkDefaultFont", 10, "bold"))
        self.lbl_cmp.pack(side="bottom", anchor="w")
        self.lbl_met = ttk.Label(t, font=("TkDefaultFont", 11, "bold"), foreground="#37474F")
        self.lbl_met.pack(side="bottom", anchor="w", pady=(2, 4))
        self.lbl_res_datos = ttk.Label(t, style="Big.TLabel", wraplength=900, justify="left",
                                       font=("TkDefaultFont", 14, "bold"))
        self.lbl_res_datos.pack(side="bottom", anchor="w", pady=(6, 0))

        self.fig_datos = Figure(facecolor="white")
        gs = self.fig_datos.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.12,
                                         left=0.07, right=0.98, top=0.97, bottom=0.14)
        self.ax_dat = self.fig_datos.add_subplot(gs[0])
        self.ax_res = self.fig_datos.add_subplot(gs[1], sharex=self.ax_dat)
        self.canvas_datos = self._canvas(t, self.fig_datos)
        self._dibujar_datos()

    # ───────────────────────── CSV ─────────────────────────
    def abrir_csv(self):
        ruta = filedialog.askopenfilename(
            filetypes=[("CSV / texto", "*.csv *.txt *.tsv *.dat"), ("Todos", "*.*")])
        if not ruta:
            return
        try:
            nombres, cols = leer_csv(ruta)
        except Exception as e:
            messagebox.showerror("No se pudo leer el archivo", str(e))
            return
        self._cargar(nombres, cols, ruta.replace("\\", "/").split("/")[-1])

    def cargar_ejemplo(self, tipo):
        x, y = datos_ejemplo(tipo)
        self._cargar(["x", "y"], [x, y], f"Ejemplo: {tipo}")

    def _cargar(self, nombres, cols, nombre):
        self.detener_gp(False)
        self.csv_nombres, self.csv_cols, self.nombre_datos = nombres, cols, nombre
        self.cb_x.configure(values=["(índice)"] + nombres)
        self.cb_y.configure(values=nombres)
        if len(nombres) >= 2:
            self.var_xcol.set(nombres[0])
            self.var_ycol.set(nombres[1])
        else:
            self.var_xcol.set("(índice)")
            self.var_ycol.set(nombres[0])
        self.nb.select(2)
        if self._aplicar_columnas():
            self.rango_a_datos()

    def _aplicar_columnas(self, _=None):
        if not self.csv_nombres or self.var_ycol.get() not in self.csv_nombres:
            return False
        y = self.csv_cols[self.csv_nombres.index(self.var_ycol.get())]
        if self.var_xcol.get() in self.csv_nombres:
            x = self.csv_cols[self.csv_nombres.index(self.var_xcol.get())]
        else:
            x = np.arange(len(y), dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 4:
            self._error("Se necesitan al menos 4 puntos numéricos (X, Y).")
            return False
        self._error("")
        self.detener_gp(False)
        o = np.argsort(x[m], kind="stable")
        self.dx, self.dy = x[m][o], y[m][o]
        self.lbl_csv.configure(text=f"{self.nombre_datos[:34]} · {len(self.dx)} pts")
        self.lbl_gp.configure(text="")
        self.modelos, self.ganador, self.sel = [], None, None
        self.gp, self.gp_modelo = None, None
        self._auto_zoom()
        self.ajustar_datos()
        self.dibujar_grafica_actual()
        return True

    def guardar_csv(self, ruta=None):
        if self.dx is None:
            messagebox.showinfo("Sin datos", "Primero genera un CSV, ábrelo o elige datos de ejemplo.")
            return
        if ruta is None:
            ruta = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="datos_fx.csv",
                                                filetypes=[("CSV", "*.csv")])
        if ruta:
            with open(ruta, "w", encoding="utf-8", newline="") as f:
                f.write("x,y\n")
                for a, b in zip(self.dx, self.dy):
                    f.write(f"{a:.10g},{b:.12g}\n")
            self.lbl_csv.configure(text=f"{self.nombre_datos} · {len(self.dx)} pts · guardado")

    def generar_csv(self, guardar=True):
        """Crea los puntos (x, y) de la f(x) escrita, los carga para analizarlos y ofrece guardarlos."""
        if self.raiz is None:
            self._error("Primero escribe una ecuación f(x).")
            return
        try:
            a, b = leer_numero(self.var_gmin.get()), leer_numero(self.var_gmax.get())
            h, ruido = leer_numero(self.var_gstep.get()), leer_numero(self.var_gruido.get())
            if a >= b or h <= 0 or ruido < 0:
                raise ValueError("Debe cumplirse: x de < a, paso > 0 y ruido ≥ 0")
        except (SyntaxError, ValueError) as e:
            self._error(f"Datos del generador no válidos: {e}")
            return
        n = int(round((b - a) / h)) + 1
        if n < 4 or n > 1_000_000:
            self._error(f"El generador produciría {n} puntos; elige un paso que dé entre 4 y 1 000 000.")
            return
        self._error("")
        x = np.round(a + np.arange(n) * h, 10)
        with np.errstate(all="ignore"):
            y = np.asarray(self.raiz.evaluar(x), dtype=float)
        ok = np.isfinite(y)
        quitados = int((~ok).sum())
        x, y = x[ok], y[ok]
        if len(x) < 4:
            self._error("f(x) no está definida en ese rango.")
            return
        if ruido > 0:
            y = y + (ruido / 100) * (np.ptp(y) or 1.0) * np.random.default_rng().normal(size=len(y))
        nota = f" ({quitados} sin definir, descartados)" if quitados else ""
        self._cargar(["x", "y"], [x, y], "f(x) generada" + (f" con {ruido:g} % de ruido" if ruido > 0 else "") + nota)
        if guardar:
            self.guardar_csv()

    def _zoom_todo(self):
        self.var_zmin.set("")
        self.var_zmax.set("")
        self._dibujar_datos()

    def _auto_zoom(self):
        """Con datos muy densos propone una ventana central (~2 500 puntos) para ver la forma."""
        x = self.dx
        if x is not None and len(x) > 5000:
            c = len(x) // 2
            self.var_zmin.set(f"{x[max(c - 1250, 0)]:.4g}")
            self.var_zmax.set(f"{x[min(c + 1250, len(x) - 1)]:.4g}")
        else:
            self.var_zmin.set("")
            self.var_zmax.set("")

    def _ventana(self):
        try:
            a, b = leer_numero(self.var_zmin.get()), leer_numero(self.var_zmax.get())
        except (SyntaxError, ValueError):
            return None
        return (a, b) if a < b else None

    def _texto_comparacion(self):
        """Original (tu f) vs función recuperada: diferencia punto a punto en los x de los datos."""
        if self.raiz is None or self.ganador is None or self.dx is None:
            return ""
        with np.errstate(all="ignore"):
            yo = np.asarray(self.raiz.evaluar(self.dx), dtype=float)
            yg = np.asarray(self.ganador["arbol"].evaluar(self.dx), dtype=float)
        mk = np.isfinite(yo) & np.isfinite(yg)
        if not mk.any():
            return "Original vs recuperada: tu f(x) no está definida en los x de los datos."
        d = np.abs(yo[mk] - yg[mk])
        mm = metricas(yo[mk], yg[mk])
        return (f"Original (tu f) vs recuperada ({self.ganador['nombre']}):   máx|Δ| = {fmt_m(float(d.max()))}   "
                f"RMSE entre ambas = {fmt_m(mm['rmse'])}   NMSE = {fmt_m(mm['nmse'])}   R² = {fmt_m(mm['r2'], 6)}")

    def rango_a_datos(self):
        if self.dx is None:
            return
        a, b = float(self.dx.min()), float(self.dx.max())
        pad = 0.05 * (b - a) or 1.0
        self.var_min.set(f"{a - pad:.4g}")
        self.var_max.set(f"{b + pad:.4g}")
        self.cambiar_rango()

    # ───────────────────────── ajuste y ranking ─────────────────────────
    def ajustar_datos(self):
        if self.dx is None:
            self.lbl_res_datos.configure(text="Abre un CSV o elige «Datos de ejemplo».")
            return
        self.lbl_res_datos.configure(text="Calculando… (con muchos puntos puede tardar unos segundos)")
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            self.modelos = ajustar_modelos(self.dx, self.dy, self._alt_max())
            if self.gp_modelo and self.gp_modelo["altura"] <= self._alt_max():
                self.modelos.append(self.gp_modelo)
        finally:
            self.root.configure(cursor="")
        self.ganador = elegir_ganador(self.modelos, self.var_parsi.get())
        self.sel = self.ganador
        self._mostrar_ranking()
        self._dibujar_datos()

    def _reelegir(self):
        if self.modelos:
            self.ganador = elegir_ganador(self.modelos, self.var_parsi.get())
            self.sel = self.ganador
            self._mostrar_ranking()
            self._dibujar_datos()

    # ───────────────────────── algoritmo genético (regresión simbólica) ─────────────────────────
    def _leer_gp(self):
        try:
            pob, gens = int(float(self.var_pob.get())), int(float(self.var_gens.get()))
            pc, pm = float(self.var_pc.get()), float(self.var_pm.get())
            if not (10 <= pob <= 5000 and 1 <= gens <= 100000 and 0 <= pc <= 1 and 0 <= pm <= 1):
                raise ValueError
        except ValueError:
            self._error("Parámetros evolutivos no válidos: población 10–5000, generaciones ≥ 1, "
                        "probabilidades entre 0 y 1.")
            return None
        return pob, gens, pc, pm

    def evolucionar(self):
        """Lanza el algoritmo genético sobre los datos cargados (respeta «Altura máx.»)."""
        if self.dx is None:
            self._error("Primero abre un CSV, genera uno o elige datos de ejemplo.")
            return
        p = self._leer_gp()
        if p is None:
            return
        self.detener_gp(False)
        self._error("")
        pob, gens, pc, pm = p
        if self.var_sembrar.get() and not self.modelos:
            self.ajustar_datos()
        semillas = ([m["arbol"] for m in self.modelos if m["elegible"] and not m.get("gp")]
                    if self.var_sembrar.get() else [])
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            self.gp = GP(self.dx, self.dy, self._alt_max(), pob=pob, p_cruce=pc, p_mut=pm, semillas=semillas)
        finally:
            self.root.configure(cursor="")
        self.gp_max = gens
        self._gp_tick()

    def _gp_tick(self):
        """Una generación por tick (la interfaz sigue respondiendo entre generaciones)."""
        gp = self.gp
        if gp is None:
            return
        gp.paso()
        fin = gp.terminado(self.gp_max)
        if fin:
            gp.pulir()
        self.lbl_gp.configure(text=(
            f"{'Terminado' if fin else 'Evolucionando'} · generación {gp.gen}/{self.gp_max} · "
            f"error relativo (1−R²) = {fmt_m(gp.mejor_fvu)} · altura {altura(gp.mejor)} · f(x) = {a_infija(gp.mejor)}"))
        if fin or gp.gen % 5 == 0:
            self._publicar_gp()
        self.gp_id = None if fin else self.root.after(1, self._gp_tick)

    def detener_gp(self, publicar=True):
        if self.gp_id:
            self.root.after_cancel(self.gp_id)
            self.gp_id = None
            if publicar and self.gp is not None:
                self.gp.pulir()
                self._publicar_gp()
                self.lbl_gp.configure(text="Detenido · " + self.lbl_gp.cget("text").split("·", 1)[-1].strip())

    def _publicar_gp(self):
        """Mete el mejor árbol evolucionado en el ranking (MSE, RMSE, NMSE y R² sobre TODOS los datos)."""
        if self.gp is None or self.dx is None:
            return
        t = self.gp.mejor.copiar()
        nodos = preorden(t)
        with np.errstate(all="ignore"):
            yh = np.asarray(t.evaluar(self.dx), dtype=float)
        m = dict(nombre="Evolutivo (GP)", n=len(self.dx), expr=a_infija(t), arbol=t, altura=altura(t),
                 estado="", gp=True,
                 nparams=sum(1 for n, _ in nodos if n.tipo == "num") + sum(1 for n, _ in nodos if n.hijos))
        m.update(metricas(self.dy, yh))
        m["elegible"] = bool(np.isfinite(m["nmse"]))
        if not m["elegible"]:
            m["estado"] = "valores no finitos"
        self.gp_modelo = m
        self.modelos = [q for q in self.modelos if not q.get("gp")] + [m]
        self.ganador = elegir_ganador(self.modelos, self.var_parsi.get())
        self.sel = m
        self._mostrar_ranking()
        self._dibujar_datos()

    def _metricas_mia(self):
        if self.raiz is None or self.dx is None:
            return None
        with np.errstate(all="ignore"):
            return metricas(self.dy, np.asarray(self.raiz.evaluar(self.dx), dtype=float))

    def _mostrar_ranking(self):
        tv = self.tv_rank
        tv.delete(*tv.get_children())
        if self.dx is None:
            return
        f = fmt_m
        el = [m for m in self.modelos if m["elegible"]]
        menor = min(el, key=lambda m: m["nmse"]) if el else None
        orden = sorted(range(len(self.modelos)), key=lambda i: (
            0 if self.modelos[i]["elegible"] else (1 if self.modelos[i]["arbol"] is not None else 2),
            self.modelos[i]["nmse"]))
        for i in orden:
            m = self.modelos[i]
            if m is self.ganador:
                estado, tag = "★ GANADOR", "ganador"
            elif m["elegible"]:
                estado, tag = ("menor error (más complejo)" if m is menor else ""), ""
            else:
                estado, tag = m["estado"], "gris"
            tv.insert("", "end", iid=str(i), tags=(tag,), values=(
                m["nombre"], m["expr"] or "—", m["nparams"], f(m["mse"]), f(m["rmse"]),
                f(m["nmse"]), f(m["r2"]), estado))
        mm = self._metricas_mia()
        if mm is not None:
            tv.insert("", "end", iid="mia", tags=("mia",), values=(
                "Original: tu f(x)", a_infija(self.raiz), "—", f(mm["mse"]), f(mm["rmse"]),
                f(mm["nmse"]), f(mm["r2"]), "la que escribiste"))
        elegido = "mia" if self.sel == "mia" else (str(self.modelos.index(self.sel)) if self.sel in self.modelos else None)
        if elegido and tv.exists(elegido):
            tv.selection_set(elegido)
            tv.see(elegido)

    def _sel_fila(self, _=None):
        s = self.tv_rank.selection()
        if not s:
            return
        self.sel = "mia" if s[0] == "mia" else self.modelos[int(s[0])]
        self._dibujar_datos()

    def _dibujar_datos(self):
        ax, ar = self.ax_dat, self.ax_res
        ax.clear()
        ar.clear()
        if self.dx is None:
            ax.set_axis_off()
            ar.set_axis_off()
            ax.text(0.5, 0.5, "Escribe tu f(x) y pulsa «Generar CSV…»,\n"
                    "o abre un CSV / elige «Datos de ejemplo»\npara graficar los puntos e identificar la función",
                    ha="center", va="center", fontsize=14, color="#78909C", transform=ax.transAxes)
            self.lbl_res_datos.configure(text="")
            self.lbl_met.configure(text="")
            self.lbl_cmp.configure(text="")
            self.canvas_datos.draw_idle()
            return
        x, y = self.dx, self.dy
        vent = self._ventana()
        m = np.ones(len(x), dtype=bool)
        if vent:
            m = (x >= vent[0]) & (x <= vent[1])
            if m.sum() < 2:
                vent, m = None, np.ones(len(x), dtype=bool)
        xv, yv = x[m], y[m]
        if vent:
            xs = np.linspace(vent[0], vent[1], 6000)
        else:
            pad = 0.04 * (np.ptp(x) or 1.0)
            xs = np.linspace(x.min() - pad, x.max() + pad, 700)

        def curva(arbol):
            with np.errstate(all="ignore"):
                v = np.asarray(arbol.evaluar(xs), dtype=float)
            v[~np.isfinite(v)] = np.nan
            return v

        ax.scatter(xv, yv, s=28 if len(xv) <= 1500 else 5, color="#455A64", alpha=0.8, zorder=3,
                   label=f"datos (n={len(xv)}{' en la ventana' if vent else ''} de {len(x)})")
        if self.ganador is not None:
            ax.plot(xs, curva(self.ganador["arbol"]), color="#D32F2F", lw=2.8, zorder=2,
                    label=f"ganador: {self.ganador['nombre']}")
        if isinstance(self.sel, dict) and self.sel is not self.ganador:
            ax.plot(xs, curva(self.sel["arbol"]), color="#2E7D32", lw=2.2, ls="--", zorder=2,
                    label=f"seleccionado: {self.sel['nombre']}")
        if self.var_mia.get() and self.raiz is not None:
            ax.plot(xs, curva(self.raiz), color="#1565C0", lw=1.8, ls="--", zorder=4, label="original: tu f(x)")
        mg = (np.ptp(yv) or 1.0) * 0.25
        ax.set_ylim(yv.min() - mg, yv.max() + mg)
        ax.set_xlim(xs[0], xs[-1])
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8.5, framealpha=0.92)
        ax.set_ylabel("y")
        ax.tick_params(labelbottom=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # modelo mostrado: el seleccionado (o el ganador)
        if self.sel == "mia":
            arbol, titulo, met = self.raiz, "Original: tu f(x)", self._metricas_mia()
        else:
            mo = self.sel if isinstance(self.sel, dict) else self.ganador
            arbol, titulo, met = (mo["arbol"], mo["nombre"], mo) if mo else (None, "", None)
        if arbol is not None and met is not None:
            with np.errstate(all="ignore"):
                res = yv - np.asarray(arbol.evaluar(xv), dtype=float)
            ar.axhline(0, color="#90A4AE", lw=1)
            if len(xv) <= 3000:
                ar.vlines(xv, 0, res, color="#D32F2F", lw=1.2, alpha=0.7)
                ar.scatter(xv, res, s=14, color="#D32F2F", zorder=3)
            else:
                ar.plot(xv, res, ".", ms=1.5, color="#D32F2F")
            ar.set_ylabel("residuo", fontsize=8)
            ar.tick_params(labelsize=8)
            ar.grid(alpha=0.3)
            ar.set_xlabel("x")
            for sp in ("top", "right"):
                ar.spines[sp].set_visible(False)
            if self.sel == "mia":
                etiqueta = titulo
            elif self.sel is self.ganador or self.sel is None:
                etiqueta = "Función identificada:  " + titulo
            else:
                etiqueta = "Modelo seleccionado:  " + titulo
            self.lbl_res_datos.configure(text=f"{etiqueta}   f(x) = {a_infija(arbol)}")
            g = lambda v: fmt_m(v, 6)
            self.lbl_met.configure(text=(
                f"MSE = {g(met['mse'])}      RMSE = {g(met['rmse'])}      "
                f"NMSE = {g(met['nmse'])}      R² = {g(met['r2'])}"))
        else:
            self.lbl_res_datos.configure(text="Ningún modelo es aplicable con estos datos y esta altura máxima.")
            self.lbl_met.configure(text="")
        self.lbl_cmp.configure(text=self._texto_comparacion())
        self.canvas_datos.draw_idle()

    def usar_modelo(self):
        mo = self.sel if isinstance(self.sel, dict) else self.ganador
        if mo is None or mo.get("arbol") is None:
            self._error("Primero identifica una función (abre un CSV).")
            return
        t = mo["arbol"].copiar()
        self.var_expr.set(a_prefija(t) if self.var_modo.get() == "prefija" else a_infija(t))
        self.calcular()
        if not self.lbl_err.cget("text"):
            self.nb.select(0)

    def _tabla(self, parent, columnas, anchos, altura=None):
        fr = ttk.Frame(parent)
        tv = ttk.Treeview(fr, columns=columnas, show="headings", height=altura or 10)
        for c, w in zip(columnas, anchos):
            tv.heading(c, text=c)
            tv.column(c, width=w, anchor="w" if w > 150 else "center")
        sb = ttk.Scrollbar(fr, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        return fr, tv

    def _caja_texto(self, parent, alto=2):
        t = tk.Text(parent, height=alto, wrap="word", font=("TkFixedFont", 12), relief="flat",
                    bg="white", highlightthickness=1, highlightbackground="#CFD8DC", state="disabled")
        t.pack(fill="x", pady=(2, 0))
        return t

    def _tab_recorridos(self, nb):
        t = ttk.Frame(nb, padding=10)
        nb.add(t, text="Recorridos")
        self.lbl_stats = ttk.Label(t, font=("TkDefaultFont", 10, "bold"), foreground="#37474F")
        self.lbl_stats.pack(anchor="w", pady=(0, 6))
        self.txt_rec = {}
        for clave, titulo, regla in (
                ("pre", "PREORDEN → notación prefija", "raíz → izquierdo → derecho"),
                ("in", "INORDEN → notación infija (con paréntesis)", "izquierdo → raíz → derecho"),
                ("post", "POSTORDEN → notación postfija (RPN)", "izquierdo → derecho → raíz")):
            f = ttk.Labelframe(t, text=f" {titulo}    [{regla}] ", padding=6)
            f.pack(fill="x", pady=3)
            self.txt_rec[clave] = self._caja_texto(f)
        ttk.Label(t, text="Nodos en orden de visita (preorden):", foreground=ACCENT,
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(10, 2))
        fr, self.tv_nodos = self._tabla(
            t, ("#", "Nodo", "Tipo", "Nivel", "Hijos", "Valor con x actual"),
            (50, 110, 120, 70, 70, 200), 9)
        fr.pack(fill="both", expand=True)

    def _tab_evaluacion(self, nb):
        t = ttk.Frame(nb, padding=10)
        nb.add(t, text="Evaluación (preorden)")
        ttk.Label(t, wraplength=900, justify="left", text=(
            "Cómo se resuelve con la notación prefija: se lee la lista de preorden de DERECHA a IZQUIERDA. "
            "Cada operando se apila; cada operador desapila los operandos que necesita, calcula y apila el "
            "resultado. Al terminar, la pila contiene solo la respuesta.")).pack(anchor="w")
        self.lbl_prefija = ttk.Label(t, font=("TkFixedFont", 12, "bold"), foreground=ACCENT)
        self.lbl_prefija.pack(anchor="w", pady=8)
        fr, self.tv_pasos = self._tabla(
            t, ("Paso", "Pos. preorden", "Token", "Acción", "Pila (tope a la derecha)"),
            (60, 110, 80, 430, 260), 14)
        fr.pack(fill="both", expand=True)
        self.lbl_final = ttk.Label(t, style="Big.TLabel", padding=(0, 8, 0, 0))
        self.lbl_final.pack(anchor="w")

    def _tab_derivada(self, nb):
        t = ttk.Frame(nb, padding=14)
        nb.add(t, text="Derivada")
        ttk.Label(t, text="Derivada simbólica (se obtiene recorriendo el árbol de forma recursiva)",
                  foreground=ACCENT, font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Label(t, text="f(x) =").pack(anchor="w", pady=(10, 0))
        self.txt_f = self._caja_texto(t)
        ttk.Label(t, text="f′(x) =").pack(anchor="w", pady=(10, 0))
        self.txt_df = self._caja_texto(t, 3)
        ttk.Label(t, text="Preorden de f′(x):").pack(anchor="w", pady=(10, 0))
        self.txt_dpre = self._caja_texto(t, 3)
        self.lbl_tang = ttk.Label(t, font=("TkDefaultFont", 11, "bold"), foreground="#2E7D32",
                                  wraplength=900, justify="left")
        self.lbl_tang.pack(anchor="w", pady=14)
        ttk.Label(t, style="Hint.TLabel", text=(
            "Tip: en la pestaña «Árbol» elige «f′(x)» para ver y recorrer el árbol de la derivada.")).pack(anchor="w")

    def _tab_tabla(self, nb):
        t = ttk.Frame(nb, padding=10)
        nb.add(t, text="Tabla de valores")
        fr, self.tv_tabla = self._tabla(t, ("x", "f(x)", "f′(x)"), (160, 220, 220), 20)
        fr.pack(fill="both", expand=True)

    # ───────────────────────── utilidades ─────────────────────────
    def _set(self, widget, texto):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", texto)
        widget.configure(state="disabled")

    def _error(self, msg):
        self.lbl_err.configure(text=("⚠ " + msg) if msg else "")

    def _arbol_mostrado(self):
        return self.deriv if (self.var_arbol.get() == "df" and self.deriv is not None) else self.raiz

    # ───────────────────────── entrada de la ecuación ─────────────────────────
    def _cambiar_modo(self, inicial=False):
        prefija = self.var_modo.get() == "prefija"
        self.lbl_hint.configure(text=(
            "Tokens separados por espacio, operador primero. Signo menos unario: neg.  "
            "Ej: + * 2 x neg 3" if prefija else
            "Operadores + − * / ^ · funciones sqrt sin cos tan exp ln abs · 2x, x2 = x^2 · pi, e"))
        ej = []
        for e in EJEMPLOS:
            try:
                ej.append(a_prefija(construir_arbol(e)) if prefija else e)
            except Exception:
                pass
        self.combo.configure(values=ej)
        self.combo.set("Ejemplos…")
        if not inicial and self.raiz is not None:      # convierte lo escrito al otro formato
            self.var_expr.set(a_prefija(self.raiz) if prefija else a_infija(self.raiz))
        elif inicial:
            self.var_expr.set(EJEMPLOS[0])

    def _alt_max(self):
        try:
            v = int(float(self.var_alt.get()))
        except (ValueError, Exception):
            v = 8
        return max(2, min(12, v))

    def _cambio_alt(self):
        self.var_alt.set(str(self._alt_max()))
        if self.raiz is not None:
            self.calcular()
        if self.dx is not None and self.modelos:
            self.ajustar_datos()

    def aleatoria(self):
        """Genera una ecuación al azar con altura <= «Altura máx.»."""
        p = self._leer_parametros()
        t = generar_aleatoria(self._alt_max(), p[1] if p else (-10, 10))
        self.var_expr.set(a_prefija(t) if self.var_modo.get() == "prefija" else a_infija(t))
        self.calcular()

    def _elegir_ejemplo(self, _=None):
        self.var_expr.set(self.combo.get())
        self.calcular()

    def _escritura_viva(self, ev):
        if ev.keysym in ("Return", "Tab", "Shift_L", "Shift_R", "Control_L", "Control_R"):
            return
        if self._live:
            self.root.after_cancel(self._live)
        self._live = self.root.after(650, self.calcular)

    # ───────────────────────── cálculo principal ─────────────────────────
    def calcular(self, _=None):
        if self._live:
            self.root.after_cancel(self._live)
            self._live = None
        texto = self.var_expr.get().strip()
        try:
            raiz = parsear_prefija(texto) if self.var_modo.get() == "prefija" else construir_arbol(texto)
        except (SyntaxError, ValueError) as e:
            self._error(str(e))
            return
        alt, lim = altura(raiz), self._alt_max()
        if alt > lim:
            self._error(f"El árbol de esta ecuación tiene altura {alt}, pero el máximo permitido es "
                        f"{lim}. Simplifica la ecuación o sube «Altura máx.».")
            return
        try:
            deriv = derivar(raiz).copiar()
        except Exception:
            deriv = None
        self._error("")
        self.detener_anim()
        self.paso = None
        self.raiz, self.deriv = raiz, deriv
        if self.var_arbol.get() == "df" and deriv is None:
            self.var_arbol.set("f")
        self._estatico()
        self.refrescar()
        if self.dx is not None:
            self._mostrar_ranking()
            self._dibujar_datos()

    def _leer_parametros(self):
        try:
            x0 = leer_numero(self.var_x.get())
            a, b = leer_numero(self.var_min.get()), leer_numero(self.var_max.get())
            if a >= b:
                raise ValueError("El rango debe cumplir mínimo < máximo")
        except (SyntaxError, ValueError) as e:
            self._error(f"x o rango no válidos: {e}")
            return None
        self._error("")
        return x0, (a, b)

    def _estatico(self):
        """Todo lo que NO depende del valor de x."""
        r = self.raiz
        pre = preorden(r)
        n_hojas = sum(1 for n, _ in pre if not n.hijos)
        self.lbl_stats.configure(text=(
            f"Nodos: {len(pre)}   ·   Hojas: {n_hojas}   ·   Operadores/funciones: {len(pre) - n_hojas}"
            f"   ·   Altura: {altura(r)} (máx. {self._alt_max()})"))
        self._set(self.txt_rec["pre"], a_prefija(r))
        self._set(self.txt_rec["in"], a_infija(r))
        self._set(self.txt_rec["post"], a_postfija(r))
        self._set(self.txt_f, a_infija(r))
        if self.deriv is not None:
            self._set(self.txt_df, a_infija(self.deriv))
            self._set(self.txt_dpre, a_prefija(self.deriv))
        else:
            self._set(self.txt_df, "(no se pudo derivar)")
            self._set(self.txt_dpre, "")
        self.lbl_int.configure(text="Interpretada como:  " + a_infija(r))
        self.lbl_prefija.configure(text="Prefija:  " + a_prefija(r))
        self._poner_preorden()
        self.cambiar_rango(solo_analisis=True)

    def cambiar_rango(self, solo_analisis=False):
        if self.raiz is None:
            return
        p = self._leer_parametros()
        if p is None:
            return
        _, (a, b) = p
        key = (a, b, a_prefija(self.raiz))
        if getattr(self, "_key_rango", None) != key:
            self._key_rango = key
            self.raices = encontrar_raices(self.raiz, a, b)
            self.indef = intervalos_indefinidos(self.raiz, a, b)
            self._tabla_valores(a, b)
            self._texto_info()
        if not solo_analisis:
            self.refrescar()

    def _texto_info(self):
        dom = ("Definida en todo el rango." if not self.indef else
               "No definida (≈) en: " + ",  ".join(f"({fmt(u, 4)}, {fmt(v, 4)})" for u, v in self.indef[:4])
               + ("…" if len(self.indef) > 4 else ""))
        rai = ("Raíces (f = 0): " + ",  ".join(fmt(r, 5) for r in self.raices[:8])
               if self.raices else "Sin raíces en el rango.")
        self.lbl_info.configure(text=dom + "\n" + rai)

    def _tabla_valores(self, a, b):
        self.tv_tabla.delete(*self.tv_tabla.get_children())
        for x in np.linspace(a, b, 41):
            y = _f_escalar(self.raiz, x)
            d = _f_escalar(self.deriv, x) if self.deriv is not None else float("nan")
            self.tv_tabla.insert("", "end", values=(fmt(x, 5), fmt(y, 6), fmt(d, 6)))

    # ───────────────────────── refresco (depende de x) ─────────────────────────
    def refrescar(self, *_):
        if self.raiz is None:
            return
        p = self._leer_parametros()
        if p is None:
            return
        x0, (a, b) = p
        self._lock = True
        self.escala.configure(from_=a, to=b)
        self.escala.set(min(max(x0, a), b))
        self._lock = False

        y0 = _f_escalar(self.raiz, x0)
        self.lbl_res.configure(text=f"f({fmt(x0, 4)}) = {fmt(y0)}")
        if self.deriv is not None:
            self.lbl_der.configure(text=f"f′({fmt(x0, 4)}) = {fmt(_f_escalar(self.deriv, x0))}")
        else:
            self.lbl_der.configure(text="")
        self.dibujar_arbol_actual()
        self.dibujar_grafica_actual()
        self._llenar_nodos(x0)
        self._llenar_pasos(x0)
        self._texto_tangente(x0, y0)

    def _mover_escala(self, v):
        if self._lock:
            return
        self.var_x.set(f"{float(v):.4g}")
        if self._pend:
            self.root.after_cancel(self._pend)
        self._pend = self.root.after(45, self.refrescar)

    def _texto_tangente(self, x0, y0):
        if self.deriv is None:
            self.lbl_tang.configure(text="")
            return
        m = _f_escalar(self.deriv, x0)
        if np.isfinite(m) and np.isfinite(y0):
            n = y0 - m * x0
            self.lbl_tang.configure(text=(
                f"En x = {fmt(x0, 4)}:   f = {fmt(y0)},   f′ = {fmt(m)}\n"
                f"Recta tangente:   y = {fmt(m, 5)}·x {'+' if n >= 0 else '−'} {fmt(abs(n), 5)}"))
        else:
            self.lbl_tang.configure(text="La función o su derivada no están definidas en este x.")

    # ───────────────────────── dibujo ─────────────────────────
    def dibujar_arbol_actual(self):
        raiz = self._arbol_mostrado()
        p = self._leer_parametros()
        if raiz is None or p is None:
            return
        self._textos = dibujar_arbol(self.ax_arbol, raiz, p[0], self.var_valores.get(),
                                     self.var_numerar.get(), self.paso)
        self.canvas_arbol.draw_idle()
        self._texto_paso()

    def dibujar_grafica_actual(self):
        p = self._leer_parametros()
        if self.raiz is None or p is None:
            return
        dibujar_grafica(self.ax_graf, self.raiz, self.deriv, p[0], p[1], self.var_deriv.get(),
                        self.var_tang.get(), self.var_raices.get(), self.raices,
                        datos=(self.dx, self.dy) if (self.dx is not None and self.var_datos_graf.get()) else None)
        self.canvas_graf.draw_idle()

    def _clic_arbol(self, ev):
        if ev.inaxes is None:
            return
        p = self._leer_parametros()
        for t, n in self._textos:
            if t.get_window_extent().contains(ev.x, ev.y):
                v = _f_escalar(n, p[0]) if p else float("nan")
                self.lbl_nodo.configure(
                    text=f"Nodo «{n.etiqueta}»  →  subexpresión:  {a_infija(n)}   =   {fmt(v)}")
                return

    def _cambio_arbol(self):
        self.detener_anim()
        self.paso = None
        self.dibujar_arbol_actual()

    # ───────────────────────── tablas ─────────────────────────
    def _llenar_nodos(self, x0):
        self.tv_nodos.delete(*self.tv_nodos.get_children())
        nombres = {"op": "operador", "func": "función", "num": "número", "const": "constante", "var": "variable"}
        for i, (n, niv) in enumerate(preorden(self.raiz), 1):
            self.tv_nodos.insert("", "end", values=(
                i, n.etiqueta, nombres[n.tipo], niv, len(n.hijos), fmt(_f_escalar(n, x0))))

    def _llenar_pasos(self, x0):
        self.tv_pasos.delete(*self.tv_pasos.get_children())
        pasos, res = evaluar_prefija_pasos(self.raiz, x0)
        for fila in pasos:
            self.tv_pasos.insert("", "end", values=fila)
        hijos = self.tv_pasos.get_children()
        if hijos:
            self.tv_pasos.see(hijos[-1])
        self.lbl_final.configure(text=f"Resultado = {fmt(res)}   (f({fmt(x0, 4)}))")

    # ───────────────────────── preorden: texto + animación ─────────────────────────
    def _poner_preorden(self):
        t = self.txt_pre
        t.configure(state="normal")
        t.delete("1.0", "end")
        self.rangos_tok = []
        toks = [n.token for n, _ in preorden(self.raiz)]
        for i, tok in enumerate(toks):
            ini = t.index("end-1c")
            t.insert("end", tok)
            self.rangos_tok.append((ini, t.index("end-1c")))
            if i < len(toks) - 1:
                t.insert("end", "  ")
        t.configure(state="disabled")

    def _copiar_pre(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(a_prefija(self.raiz))
        self.lbl_err.configure(text="")

    def _resaltar_token(self):
        self.txt_pre.tag_remove("actual", "1.0", "end")
        if self.paso is not None and self.var_arbol.get() == "f" and self.paso < len(self.rangos_tok):
            self.txt_pre.tag_add("actual", *self.rangos_tok[self.paso])

    def _texto_paso(self):
        raiz = self._arbol_mostrado()
        self._resaltar_token()
        if raiz is None:
            return
        orden = [n for n, _ in preorden(raiz)]
        if self.paso is None:
            self.lbl_paso.configure(text=(
                "Pulsa «Siguiente» o «Reproducir» para recorrer el árbol en PREORDEN: "
                "primero la raíz, luego el subárbol izquierdo y después el derecho."))
            return
        n = orden[self.paso]
        padres = mapa_padres(raiz)
        if id(n) in padres:
            p, lado = padres[id(n)]
            rol = f"hijo {lado} de «{p.etiqueta}»" if lado != "único" else f"hijo de «{p.etiqueta}»"
        else:
            rol = "es la RAÍZ (siempre se visita primero)"
        visitados = " ".join(m.token for m in orden[:self.paso + 1])
        fin = "   ✔ recorrido completo" if self.paso == len(orden) - 1 else ""
        self.lbl_paso.configure(text=(
            f"Paso {self.paso + 1} de {len(orden)}:  visito «{n.etiqueta}» — {rol}.{fin}\n"
            f"Preorden hasta ahora:  {visitados}"))

    def anim_sig(self):
        raiz = self._arbol_mostrado()
        if raiz is None:
            return False
        total = len(preorden(raiz))
        if self.paso is None:
            self.paso = 0
        elif self.paso < total - 1:
            self.paso += 1
        else:
            return False
        self.dibujar_arbol_actual()
        return True

    def anim_ant(self):
        self.detener_anim()
        if self.paso is None:
            return
        self.paso = None if self.paso == 0 else self.paso - 1
        self.dibujar_arbol_actual()

    def anim_reiniciar(self):
        self.detener_anim()
        self.paso = None
        self.dibujar_arbol_actual()

    def anim_play(self):
        if self.anim_id:
            return
        raiz = self._arbol_mostrado()
        if raiz is None:
            return
        if self.paso is None or self.paso >= len(preorden(raiz)) - 1:
            self.paso = None
        self._tick()

    def _tick(self):
        if self.anim_sig():
            self.anim_id = self.root.after(int(self.var_vel.get() * 1000), self._tick)
        else:
            self.anim_id = None

    def detener_anim(self):
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None

    # ───────────────────────── guardar ─────────────────────────
    def guardar(self, fig):
        ruta = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if ruta:
            fig.savefig(ruta, dpi=170, bbox_inches="tight", facecolor="white")


def main():
    if not HAY_TK:
        raise SystemExit("Falta tkinter. En Ubuntu/Debian:  sudo apt install python3-tk")
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()