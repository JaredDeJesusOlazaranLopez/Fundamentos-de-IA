#!/usr/bin/env python3
"""
ÁRBOL DE EXPRESIÓN · v2  (nuevo estilo + animaciones + algoritmo genético corregido)
===================================================================================

Este archivo NO reemplaza a tu código: se coloca en la MISMA carpeta que
«arbol_expresion_app.py» y lo reutiliza (parser, derivada, ajuste de modelos, etc.).

    python arbol_expresion_app_v2.py

Qué cambia
----------
 1. ALGORITMO GENÉTICO
    * Ya no se detiene a las 150 generaciones por «paciencia». Ahora, si se estanca,
      da una «SACUDIDA»: reemplaza el 40 % peor de la población por inmigrantes
      aleatorios y mutantes del mejor, y sigue. Solo termina al llegar al máximo de
      generaciones, al lograr un ajuste casi perfecto, o tras 400 generaciones sin mejora.
    * El mensaje de estado dice POR QUÉ terminó. Barra de progreso incluida.
    * Dos modelos nuevos en la identificación: «Raíz de recta» y «Raíz de cuadrática»
      (√(a·x²+b·x+c)). Con ellos tu función se identifica con R² = 1 y además sirven
      de semilla para el genético.
 2. ESTILO: tema índigo/violeta, cabecera con degradado y un mini-árbol animado,
    tablas, botones, pestañas y gráficas con colores nuevos.
 3. ANIMACIONES DEL ÁRBOL
    * PREORDEN: una bolita viaja por las ramas. Baja (rojo) y, al terminar una rama,
      RETROCEDE (gris punteado) hasta el ancestro correcto. Un texto explica cada
      movimiento, el nodo llegado emite una onda, y abajo una tira de «fichas» muestra
      el orden de visita.
    * EVALUAR HACIA ARRIBA (postorden): los valores nacen en las hojas y suben hasta la
      raíz, con flechas verdes; cada nodo «salta» al recibir su valor.
    * Niveles del árbol marcados al fondo, sombras, flechas de dirección.
"""

import math
import time

import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

import arbol_expresion_app as base

tk, ttk, tkfont = base.tk, base.ttk, base.tkfont

preorden, postorden = base.preorden, base.postorden
mapa_padres, altura = base.mapa_padres, base.altura
a_infija, a_prefija = base.a_infija, base.a_prefija
fmt, fmt_m = base.fmt, base.fmt_m

# ══════════════════════════════════════════════════════════════════════════
#  PALETA
# ══════════════════════════════════════════════════════════════════════════
BG = "#F3F5FC"
HEADER = "#1E1B4B"
ACCENT = "#4F46E5"
TXT = "#1E293B"
BORDE = "#C7D2FE"
LIENZO = "#FBFCFF"

TIPO = {  # tipo → (relleno, borde)
    "op": ("#FDE68A", "#D97706"),
    "func": ("#E9D5FF", "#7E22CE"),
    "num": ("#BBF7D0", "#15803D"),
    "const": ("#BBF7D0", "#15803D"),
    "var": ("#BAE6FD", "#0369A1"),
}


# ══════════════════════════════════════════════════════════════════════════
#  MODELOS NUEVOS:  y = sqrt(polinomio)
# ══════════════════════════════════════════════════════════════════════════
def _aj_raiz_poli(d):
    def aj(x, y):
        if np.min(y) < -0.02 * (np.ptp(y) or 1.0):      # una raíz no puede dar valores negativos
            return None
        y0 = np.clip(y, 0, None)
        p0 = np.polyfit(x, y0 ** 2, d)                  # y² ≈ polinomio  (ajuste lineal exacto)
        f = lambda xx, p: np.sqrt(np.abs(np.polyval(p, xx)))
        p = base._lm(f, p0, x, y0)
        if p is None or not np.all(np.isfinite(p)):
            p = p0
        if np.mean(np.polyval(p, x)) < 0:
            p = -p
        return p
    return aj


def _expr_raiz(p):
    return "sqrt(" + base._poli_str(p, base._bases_poli(len(p) - 1)) + ")"


MODELOS_NUEVOS = [
    dict(nombre="Raíz de recta", np=2, req=None, ajustar=_aj_raiz_poli(1), expr=_expr_raiz),
    dict(nombre="Raíz de cuadrática", np=3, req=None, ajustar=_aj_raiz_poli(2), expr=_expr_raiz),
]


# ══════════════════════════════════════════════════════════════════════════
#  ALGORITMO GENÉTICO MEJORADO (mismo motor, pero con «sacudidas»)
# ══════════════════════════════════════════════════════════════════════════
class GPv2(base.GP):
    PERIODO_SACUDIDA = 40       # generaciones sin mejora antes de sacudir
    LIMITE_SIN_MEJORA = 400     # solo entonces se da por terminado

    def __init__(self, *a, **k):
        k["paciencia"] = 10 ** 9
        self.kicks = 0
        super().__init__(*a, **k)

    def paso(self):
        super().paso()
        if self.sin_mejora and self.sin_mejora % self.PERIODO_SACUDIDA == 0:
            self._sacudir()

    def _sacudir(self):
        """Reemplaza el 40 % peor por inmigrantes aleatorios y mutantes del mejor."""
        self.kicks += 1
        n = max(2, int(self.pob_n * 0.4))
        mejor = self.pob[0][2]
        nuevos = []
        for i in range(n):
            if i % 2 == 0:
                t = self._aleatorio(self.r.randrange(10000))
            else:
                t = self._mutar(self._mutar(mejor))
            nuevos.append(self._indiv(self._normalizar(t)))
        self.pob[-n:] = nuevos
        self.pob.sort(key=lambda i: i[0])

    def terminado(self, max_gen):
        return (self.gen >= max_gen or self.mejor_fvu < 1e-9
                or self.sin_mejora >= self.LIMITE_SIN_MEJORA)


# ══════════════════════════════════════════════════════════════════════════
#  DIBUJO DEL ÁRBOL (nuevo)
# ══════════════════════════════════════════════════════════════════════════
def dibujar_arbol_v2(ax, raiz, vals, orden, modo, paso, walk, pulso, caption, valores, numerar,
                     titulo=None, leyenda=True):
    """modo 'pre'  → recorrido preorden (la bolita baja y retrocede)
       modo 'eval' → evaluación de abajo hacia arriba (postorden)
       paso: índice del nodo actual dentro de `orden` (None = vista estática)
       walk: (puntos, fracción) mientras la bolita viaja;  pulso: 0..1 al llegar a un nodo"""
    ax.clear()
    ax.set_axis_off()
    fig = ax.figure
    pos = base.calcular_posiciones(raiz)
    nodos = [n for n, _ in preorden(raiz)]
    idx = {id(n): i for i, n in enumerate(orden)}
    hojas = sum(1 for n in nodos if not n.hijos)
    fs = max(8.0, min(15.0, 17.0 - 0.55 * hojas))
    anim = paso is not None
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x0_, x1_ = min(xs) - 0.8, max(xs) + 0.8
    y0_, y1_ = min(ys) - 2.0, 0.85

    # niveles de fondo
    for d in range(int(-min(ys)) + 1):
        ax.axhline(-d, color="#E0E7FF", lw=1, ls=(0, (2, 4)), zorder=0)
        ax.text(x0_ + 0.05, -d + 0.06, f"nivel {d}", fontsize=8, color="#A5B4FC",
                va="bottom", ha="left", zorder=0)

    # aristas
    for n in nodos:
        for h in n.hijos:
            pa, pb = pos[id(n)], pos[id(h)]
            vis = anim and idx[id(h)] <= paso
            if vis:
                a, b = (pa, pb) if modo == "pre" else (pb, pa)
                ax.annotate("", xy=b, xytext=a, zorder=1, arrowprops=dict(
                    arrowstyle="-|>", color="#6366F1" if modo == "pre" else "#10B981", lw=2.8,
                    shrinkA=fs * 1.5, shrinkB=fs * 1.5, mutation_scale=15))
            else:
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#CBD5E1", lw=2, zorder=1,
                        solid_capstyle="round", alpha=0.55 if anim else 1)

    # nodos
    textos = []
    sombra = offset_copy(ax.transData, fig=fig, x=2.5, y=-3.5, units="points")
    base_s = (fs * 4.0) ** 2
    for n in nodos:
        x, y = pos[id(n)]
        i = idx[id(n)]
        fondo, borde = TIPO[n.tipo]
        hecho = (not anim) or i <= paso
        actual = anim and i == paso and walk is None
        alpha = 1.0 if hecho else 0.30
        if actual:
            fondo = "#FEF08A"
            if pulso is not None:
                for k in range(2):
                    q = min(1.0, max(0.0, pulso - 0.25 * k))
                    ax.scatter([x], [y], s=base_s * (1 + 2.2 * q), facecolors="none",
                               edgecolors="#F43F5E", linewidths=3 * (1 - q) + 0.4,
                               alpha=max(0.0, 1 - q), zorder=2)
            ax.scatter([x], [y], s=base_s * 1.25, color="#FDE047", alpha=0.35, zorder=2)
        ax.text(x, y, n.etiqueta, ha="center", va="center", fontsize=fs, fontweight="bold",
                color=(0, 0, 0, 0), transform=sombra, zorder=2.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="#475569", ec="none", alpha=0.16 * alpha))
        t = ax.text(x, y, n.etiqueta, ha="center", va="center", fontsize=fs, fontweight="bold",
                    color=borde, alpha=alpha, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.4", fc=fondo,
                              ec="#F43F5E" if actual else borde,
                              lw=3.4 if actual else 2, alpha=alpha))
        textos.append((t, n))
        if valores and hecho:
            v = vals.get(id(n), float("nan"))
            sz = fs - 3
            fc, tc = "#EEF2FF", "#4338CA"
            if modo == "eval" and anim:
                fc, tc = "#DCFCE7", "#166534"
                if actual and pulso is not None:
                    sz += 3 * math.sin(math.pi * pulso)       # «salto» al recibir el valor
            if not np.isfinite(v):
                fc, tc = "#FEE2E2", "#B91C1C"
            ax.annotate(f"= {fmt(v, 4)}", (x, y), xytext=(0, -(fs + 9)), textcoords="offset points",
                        ha="center", va="top", fontsize=sz, fontweight="bold", color=tc, zorder=2.6,
                        bbox=dict(boxstyle="round,pad=0.22", fc=fc, ec="none", alpha=0.95))
        if (numerar and not anim) or (anim and hecho):
            dx = 0.32 * fs * len(n.etiqueta) + fs * 0.95
            ax.annotate(str(i + 1), (x, y), xytext=(-dx, fs * 1.15 + 3), textcoords="offset points",
                        ha="center", va="center", fontsize=max(7, fs - 4), color="white",
                        fontweight="bold", zorder=4,
                        bbox=dict(boxstyle="circle,pad=0.25", ec="none",
                                  fc="#F43F5E" if actual else "#4F46E5"))

    # la bolita viajera
    if walk is not None:
        pts, f = walk
        P = np.array(pts, float)
        seg = np.hypot(*np.diff(P, axis=0).T)
        s = f * (seg.sum() or 1.0)
        acc, end = 0.0, P[0]
        for k in range(len(seg)):
            a, b = P[k], P[k + 1]
            if acc + seg[k] <= s:
                end, completo = b, True
            else:
                tt = (s - acc) / seg[k] if seg[k] > 0 else 0.0
                end, completo = a + (b - a) * tt, False
            sube = b[1] > a[1]
            ax.plot([a[0], end[0]], [a[1], end[1]], color="#94A3B8" if sube else "#F43F5E",
                    lw=3.2, ls=(0, (3, 3)) if sube else "-", zorder=5, solid_capstyle="round")
            if not completo:
                break
            acc += seg[k]
        ax.scatter([end[0]], [end[1]], s=1500, color="#F43F5E", alpha=0.16, zorder=6)
        ax.scatter([end[0]], [end[1]], s=420, color="#FB7185", alpha=0.35, zorder=6)
        ax.scatter([end[0]], [end[1]], s=150, color="#E11D48", edgecolors="white",
                   linewidths=2.2, zorder=7)

    # títulos
    if titulo is None:
        titulo = ("PREORDEN  ·  raíz → izquierdo → derecho" if modo == "pre" else
                  "EVALUACIÓN  ·  primero los hijos, luego el nodo (de abajo hacia arriba)")
    ax.text(0.012, 0.985, titulo, transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            fontweight="bold", color="#6366F1", zorder=8)
    if caption:
        ax.text(0.012, 0.93, caption, transform=ax.transAxes, ha="left", va="top", fontsize=11,
                fontweight="bold", color="#9F1239", zorder=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFF1F2", ec="#FDA4AF"))

    # tira de fichas con el orden de visita
    if len(orden) <= 60:
        toks = [n.etiqueta for n in orden]
        w = [len(t) + 2.2 for t in toks]
        u = 0.97 / sum(w)
        fsc = max(6.5, min(11.0, u * 1000))
        ax.text(0.012, 0.085, "orden de visita:" if modo == "pre" else "orden de cálculo:",
                transform=ax.transAxes, fontsize=8.5, color="#64748B", ha="left", va="bottom", zorder=8)
        cum = 0.015
        for i, (n, tk_) in enumerate(zip(orden, toks)):
            xc = cum + u * w[i] / 2
            cum += u * w[i]
            hecho = (not anim) or i <= paso
            act = anim and i == paso
            fondo, borde = TIPO[n.tipo]
            ax.text(xc, 0.04, tk_, transform=ax.transAxes, ha="center", va="center", fontsize=fsc,
                    fontweight="bold", color=borde if hecho else "#94A3B8", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc=fondo if hecho else "#F1F5F9",
                              ec="#F43F5E" if act else (borde if hecho else "#CBD5E1"),
                              lw=2.6 if act else 1.2))

    etiquetas = {"op": "Operador", "func": "Función", "num": "Número", "var": "Variable"}
    if leyenda:
        ax.legend(handles=[Line2D([0], [0], marker="s", ls="", ms=9, mfc=TIPO[k][0], mec=TIPO[k][1],
                                  mew=2, label=v) for k, v in etiquetas.items()],
                  loc="upper right", fontsize=8.5, frameon=True, framealpha=0.92,
                  facecolor="white", edgecolor="#E0E7FF")

    # fondo con degradado suave
    grad = np.linspace(0, 1, 64).reshape(-1, 1)
    cm = LinearSegmentedColormap.from_list("fondo_arbol", ["#E0E7FF", "#FFFFFF"])
    ax.imshow(grad, extent=[x0_, x1_, y0_, y1_], origin="lower", aspect="auto", cmap=cm,
              zorder=-10, interpolation="bilinear")
    ax.set_xlim(x0_, x1_)
    ax.set_ylim(y0_, y1_)
    return textos


def _mezcla(t, stops=("#1E1B4B", "#4338CA", "#A21CAF")):
    """Interpola un degradado de varios colores (t entre 0 y 1)."""
    rgb = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in stops]
    t = min(max(t, 0.0), 1.0) * (len(rgb) - 1)
    k = min(int(t), len(rgb) - 2)
    f = t - k
    return "#%02x%02x%02x" % tuple(int(rgb[k][j] + (rgb[k + 1][j] - rgb[k][j]) * f) for j in range(3))


# ══════════════════════════════════════════════════════════════════════════
#  APLICACIÓN v2
# ══════════════════════════════════════════════════════════════════════════
TOUR_CABECERA = [0, 1, 3, 1, 4, 1, 0, 2, 5, 2, 6, 2, 0]


class AppV2(base.App):
    def __init__(self, root):
        self.modo_anim = tk.StringVar(master=root, value="pre")
        self._tw = self._tw_id = None
        self._play = False
        self._walk = self._pulso = None
        self._caption = ""
        self._vc = None
        self._hdr_t = 0.0
        self.pb = None
        self._evo = self._evo_nuevo()
        self._evo_lock = False
        self._rp_id = None
        self.sl_evo = None
        super().__init__(root)
        self._pulir(root)
        for f in (self.fig_graf, self.fig_datos):
            f.set_facecolor(LIENZO)
        self.canvas_graf.draw_idle()
        self.canvas_datos.draw_idle()

    # ───────────────────────── estilo ─────────────────────────
    def _pulir(self, w):
        for c in w.winfo_children():
            if isinstance(c, tk.Text):
                c.configure(highlightbackground=BORDE, highlightcolor=ACCENT, bg="#FFFFFF",
                            selectbackground=BORDE, padx=6, pady=4)
            self._pulir(c)

    def _estilos(self):
        for nombre, tam in (("TkDefaultFont", 10), ("TkTextFont", 10), ("TkFixedFont", 11)):
            tkfont.nametofont(nombre).configure(size=tam)
        st = ttk.Style()
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=TXT)
        st.configure("TFrame", background=BG)
        st.configure("TLabel", background=BG, foreground=TXT)
        st.configure("TLabelframe", background=BG, bordercolor=BORDE, relief="solid", borderwidth=1)
        st.configure("TLabelframe.Label", background=BG, foreground=ACCENT,
                     font=("TkDefaultFont", 10, "bold"))
        st.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(2, 4, 2, 0))
        st.configure("TNotebook.Tab", padding=(18, 8), font=("TkDefaultFont", 10, "bold"),
                     background="#E0E7FF", foreground="#6366F1", borderwidth=0)
        st.map("TNotebook.Tab", background=[("selected", "white"), ("active", "#EEF2FF")],
               foreground=[("selected", ACCENT)])
        st.configure("TButton", background="#E0E7FF", foreground="#312E81", bordercolor=BORDE,
                     padding=(10, 5), relief="flat", focuscolor="#E0E7FF")
        st.map("TButton", background=[("active", "#C7D2FE"), ("pressed", "#A5B4FC")])
        st.configure("Accent.TButton", background=ACCENT, foreground="white", bordercolor=ACCENT,
                     font=("TkDefaultFont", 11, "bold"), padding=8, focuscolor=ACCENT)
        st.map("Accent.TButton", background=[("active", "#6366F1"), ("pressed", "#3730A3")])
        for w in ("TEntry", "TCombobox", "TSpinbox"):
            st.configure(w, fieldbackground="white", bordercolor=BORDE, lightcolor="white",
                         darkcolor=BORDE, padding=3)
        st.configure("TCombobox", background="#E0E7FF", arrowcolor=ACCENT)
        st.configure("TSpinbox", arrowcolor=ACCENT, background="#E0E7FF")
        st.configure("Treeview", rowheight=26, font=("TkDefaultFont", 10), background="white",
                     fieldbackground="white", bordercolor=BORDE)
        st.configure("Treeview.Heading", font=("TkDefaultFont", 10, "bold"), background="#E0E7FF",
                     foreground="#312E81", relief="flat")
        st.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", "white")])
        st.configure("TScale", background=BG, troughcolor="#C7D2FE")
        st.configure("Horizontal.TProgressbar", troughcolor="#E0E7FF", background="#8B5CF6",
                     bordercolor=BORDE, lightcolor="#A78BFA", darkcolor="#7C3AED")
        st.configure("Big.TLabel", font=("TkDefaultFont", 19, "bold"), foreground=ACCENT)
        st.configure("Err.TLabel", foreground="#BE123C")
        st.configure("Hint.TLabel", foreground="#6B7280", font=("TkDefaultFont", 9))
        st.configure("TCheckbutton", background=BG)
        st.configure("TRadiobutton", background=BG)

    # ───────────────────────── cabecera animada ─────────────────────────
    def _cabecera(self):
        c = tk.Canvas(self.root, height=66, highlightthickness=0, bd=0, bg=HEADER)
        c.pack(fill="x")
        self.cab = c
        c.bind("<Configure>", self._pintar_cabecera)
        self.root.after(100, self._anim_cabecera)

    def _pintar_cabecera(self, ev=None):
        c = self.cab
        w, h = max(c.winfo_width(), 500), 66
        c.delete("all")
        n = 140
        for i in range(n):
            col = _mezcla(i / (n - 1))
            c.create_rectangle(w * i / n, 0, w * (i + 1) / n + 1, h, fill=col, outline=col)
        c.create_text(24, 26, anchor="w", text="Árbol de Expresión", fill="white",
                      font=("TkDefaultFont", 19, "bold"))
        c.create_text(26, 51, anchor="w", fill="#E0E7FF", font=("TkDefaultFont", 9),
                      text="parser · preorden · evaluación · gráfica · derivada · algoritmo genético")
        cx = w - 120
        P = [(cx, 12), (cx - 42, 33), (cx + 42, 33), (cx - 63, 54), (cx - 21, 54),
             (cx + 21, 54), (cx + 63, 54)]
        for a, b in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)):
            c.create_line(*P[a], *P[b], fill="#C7D2FE", width=2)
        cols = ["#FDE68A", "#E9D5FF", "#E9D5FF", "#BBF7D0", "#BAE6FD", "#BBF7D0", "#BAE6FD"]
        for (x, y), col in zip(P, cols):
            c.create_oval(x - 7, y - 7, x + 7, y + 7, fill=col, outline="white", width=2)
        c.create_oval(0, 0, 0, 0, fill="#FB7185", outline="white", width=2, tags="walker")
        self._hdr_P = P

    def _anim_cabecera(self):
        try:
            P = getattr(self, "_hdr_P", None)
            if P:
                self._hdr_t = (self._hdr_t + 0.07) % (len(TOUR_CABECERA) - 1)
                k = int(self._hdr_t)
                f = self._hdr_t - k
                a, b = P[TOUR_CABECERA[k]], P[TOUR_CABECERA[k + 1]]
                x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
                self.cab.coords("walker", x - 5, y - 5, x + 5, y + 5)
            self.root.after(40, self._anim_cabecera)
        except tk.TclError:
            pass

    # ───────────────────────── estructura de la ventana ─────────────────────────
    def _ui(self):
        self._cabecera()
        cuerpo = tk.Frame(self.root, bg=BG)
        cuerpo.pack(fill="both", expand=True, padx=12, pady=12)
        izq = tk.Frame(cuerpo, bg=BG, width=360)
        izq.pack(side="left", fill="y", padx=(0, 12))
        izq.pack_propagate(False)
        self._panel_izq(izq)
        nb = ttk.Notebook(cuerpo)
        nb.pack(side="left", fill="both", expand=True)
        self._tab_arbol(nb)
        self.nb = nb
        self._tab_grafica(nb)
        self._tab_datos(nb)
        self._tab_evo(nb)
        self._tab_recorridos(nb)
        self._tab_evaluacion(nb)
        self._tab_derivada(nb)
        self._tab_tabla(nb)

    def _tab_datos(self, nb):
        super()._tab_datos(nb)
        self.pb = ttk.Progressbar(self.lbl_gp.master, mode="determinate")
        self.pb.pack(fill="x", pady=(4, 2), before=self.lbl_gp)

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
        ttk.Checkbutton(b1, text="numerar nodos", variable=self.var_numerar,
                        command=self.dibujar_arbol_actual).pack(side="left", padx=4)

        b0 = ttk.Frame(t)
        b0.pack(fill="x", pady=(6, 0))
        ttk.Label(b0, text="Animación:", font=("TkDefaultFont", 10, "bold"),
                  foreground=ACCENT).pack(side="left", padx=(0, 8))
        ttk.Radiobutton(b0, text="Recorrido PREORDEN (bajar y retroceder)", value="pre",
                        variable=self.modo_anim, command=self._cambio_modo_anim).pack(side="left")
        ttk.Radiobutton(b0, text="EVALUAR hacia arriba (postorden)", value="eval",
                        variable=self.modo_anim, command=self._cambio_modo_anim).pack(side="left", padx=12)

        b2 = ttk.Frame(t)
        b2.pack(fill="x", pady=(6, 0))
        ttk.Button(b2, text="⏮ Reiniciar", command=self.anim_reiniciar).pack(side="left", padx=2)
        ttk.Button(b2, text="◀ Anterior", command=self.anim_ant).pack(side="left", padx=2)
        ttk.Button(b2, text="Siguiente ▶", command=self.anim_sig).pack(side="left", padx=2)
        ttk.Button(b2, text="▶▶ Reproducir", style="Accent.TButton",
                   command=self.anim_play).pack(side="left", padx=6)
        ttk.Button(b2, text="■ Detener", command=self._btn_detener).pack(side="left", padx=2)
        ttk.Label(b2, text="   lento").pack(side="left")
        ttk.Scale(b2, from_=1.6, to=0.15, variable=self.var_vel, length=120).pack(side="left", padx=4)
        ttk.Label(b2, text="rápido").pack(side="left")

        self.fig_arbol = base.Figure(facecolor=LIENZO)
        self.ax_arbol = self.fig_arbol.add_axes([0.005, 0.005, 0.99, 0.99])
        self.canvas_arbol = self._canvas(t, self.fig_arbol)
        self.canvas_arbol.mpl_connect("button_press_event", self._clic_arbol)

        self.lbl_paso = ttk.Label(t, wraplength=900, justify="left", font=("TkDefaultFont", 10, "bold"),
                                  foreground="#4338CA", padding=(4, 6, 4, 0))
        self.lbl_paso.pack(fill="x")
        self.lbl_nodo = ttk.Label(t, wraplength=900, justify="left", foreground="#0369A1", padding=(4, 0))
        self.lbl_nodo.pack(fill="x")

    # ───────────────────────── GP: mensajes claros ─────────────────────────
    def evolucionar(self):
        if self.pb is not None:
            self.pb["value"] = 0
        base.GP = GPv2
        if self.dx is not None:
            self._evo_stop_replay()
            self._evo = self._evo_nuevo()
        super().evolucionar()

    def _gp_tick(self):
        gp = self.gp
        if gp is None:
            return
        gp.paso()
        fin = gp.terminado(self.gp_max)
        if fin:
            gp.pulir()
        motivo = ""
        if fin:
            if gp.mejor_fvu < 1e-9:
                motivo = " (¡ajuste casi perfecto!)"
            elif gp.gen >= self.gp_max:
                motivo = " (se llegó al máximo de generaciones)"
            else:
                motivo = f" (sin mejora tras {gp.sin_mejora} generaciones y {gp.kicks} sacudidas)"
        self.lbl_gp.configure(text=(
            f"{'Terminado' if fin else 'Evolucionando'}{motivo} · generación {gp.gen}/{self.gp_max} · "
            f"error relativo (1−R²) = {fmt_m(gp.mejor_fvu)} · sacudidas: {gp.kicks} · "
            f"altura {altura(gp.mejor)} · f(x) = {a_infija(gp.mejor)}"))
        if self.pb is not None:
            self.pb["maximum"] = self.gp_max
            self.pb["value"] = self.pb["maximum"] if fin else gp.gen
        self._evo_registrar(gp, fin, motivo)
        if fin or gp.gen % 5 == 0:
            self._publicar_gp()
        self.gp_id = None if fin else self.root.after(1, self._gp_tick)

    # ───────────────────────── animación del árbol ─────────────────────────
    def _orden(self):
        r = self._arbol_mostrado()
        if r is None:
            return []
        return postorden(r) if self.modo_anim.get() == "eval" else [n for n, _ in preorden(r)]

    def _valores(self, raiz, x0):
        c = self._vc
        if c and c[0] is raiz and c[1] == x0:
            return c[2]
        vals = {id(n): base._f_escalar(n, x0) for n, _ in preorden(raiz)}
        self._vc = (raiz, x0, vals)
        return vals

    def _cambio_modo_anim(self):
        self.detener_anim()
        self.paso = None
        self.dibujar_arbol_actual()

    def _btn_detener(self):
        self.detener_anim()
        self.dibujar_arbol_actual()

    def detener_anim(self):
        self._play = False
        if self._tw_id:
            self.root.after_cancel(self._tw_id)
            self._tw_id = None
        if self._tw:
            self.paso = self._tw["b"]
        self._tw = None
        self._walk = self._pulso = None
        self._caption = ""

    def _camino(self, na, nb):
        """Ruta por las ramas de na a nb (sube hasta el ancestro común y luego baja)."""
        raiz = self._arbol_mostrado()
        pos = base.calcular_posiciones(raiz)
        padres = mapa_padres(raiz)

        def cadena(n):
            c = [n]
            while id(n) in padres:
                n = padres[id(n)][0]
                c.append(n)
            return c

        ca, cb = cadena(na), cadena(nb)
        ids_b = {id(n) for n in cb}
        subida = []
        lca = ca[-1]
        for n in ca:
            subida.append(n)
            if id(n) in ids_b:
                lca = n
                break
        bajada = list(reversed(cb[:cb.index(lca)]))
        ruta = subida + bajada
        up, down = len(subida) - 1, len(bajada)
        if self.modo_anim.get() == "pre":
            if up == 0:
                lado = padres[id(nb)][1]
                cap = f"↓ bajo al hijo {lado} de «{na.etiqueta}»"
            else:
                lado = padres[id(nb)][1]
                cap = (f"↑ retrocedo {up} nivel{'es' if up > 1 else ''} hasta «{lca.etiqueta}» "
                       f"y ↓ bajo a su hijo {lado}")
        else:
            if down == 0:
                cap = f"↑ los hijos ya tienen valor: subo a calcular «{nb.etiqueta}»"
            else:
                cap = "→ paso a la siguiente rama pendiente (primero se resuelven los hijos)"
        return [pos[id(n)] for n in ruta], cap

    def anim_sig(self):
        if self._tw:
            if self._tw_id:
                self.root.after_cancel(self._tw_id)
            self._fin_tween(snap=True)
        orden = self._orden()
        if not orden:
            return False
        total = len(orden)
        if self.paso is None:
            b = 0
        elif self.paso < total - 1:
            b = self.paso + 1
        else:
            return False
        a = self.paso
        pts, cap = (None, "")
        if a is not None:
            pts, cap = self._camino(orden[a], orden[b])
        vel = max(0.15, float(self.var_vel.get()))
        self._tw = dict(a=a, b=b, pts=pts, cap=cap, t0=time.monotonic(),
                        dm=0.0 if pts is None else 0.12 + 0.55 * vel, dp=0.35)
        self._frame()
        return True

    def _frame(self):
        tw = self._tw
        if tw is None:
            return
        el = time.monotonic() - tw["t0"]
        if el < tw["dm"]:
            f = el / tw["dm"]
            f = f * f * (3 - 2 * f)
            self.paso, self._walk, self._pulso, self._caption = tw["a"], (tw["pts"], f), None, tw["cap"]
        elif el < tw["dm"] + tw["dp"]:
            self.paso, self._walk = tw["b"], None
            self._pulso, self._caption = (el - tw["dm"]) / tw["dp"], tw["cap"]
        else:
            self._fin_tween()
            return
        self.dibujar_arbol_actual()
        self._tw_id = self.root.after(15, self._frame)

    def _fin_tween(self, snap=False):
        tw = self._tw
        self._tw = self._tw_id = None
        if tw is None:
            return
        self.paso = tw["b"]
        self._walk = self._pulso = None
        self._caption = ""
        self.dibujar_arbol_actual()
        if self._play and not snap:
            if self.paso < len(self._orden()) - 1:
                vel = max(0.15, float(self.var_vel.get()))
                self._tw_id = self.root.after(int(120 + 150 * vel), self._siguiente_play)
            else:
                self._play = False

    def _siguiente_play(self):
        self._tw_id = None
        if self._play:
            self.anim_sig()

    def anim_play(self):
        if self._play or self._tw:
            return
        orden = self._orden()
        if not orden:
            return
        if self.paso is None or self.paso >= len(orden) - 1:
            self.paso = None
            self.dibujar_arbol_actual()
        self._play = True
        self.anim_sig()

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

    def dibujar_arbol_actual(self):
        raiz = self._arbol_mostrado()
        p = self._leer_parametros()
        if raiz is None or p is None:
            return
        self._textos = dibujar_arbol_v2(
            self.ax_arbol, raiz, self._valores(raiz, p[0]), self._orden(), self.modo_anim.get(),
            self.paso, self._walk, self._pulso, self._caption,
            self.var_valores.get(), self.var_numerar.get())
        self.canvas_arbol.draw_idle()
        self._texto_paso()

    def _resaltar_token(self):
        self.txt_pre.tag_remove("actual", "1.0", "end")
        if (self.modo_anim.get() == "pre" and self.paso is not None and self.var_arbol.get() == "f"
                and self.paso < len(self.rangos_tok)):
            self.txt_pre.tag_add("actual", *self.rangos_tok[self.paso])

    def _texto_paso(self):
        self._resaltar_token()
        raiz = self._arbol_mostrado()
        if raiz is None:
            return
        modo = self.modo_anim.get()
        orden = self._orden()
        if self.paso is None:
            self.lbl_paso.configure(text=(
                "Pulsa «Siguiente» o «Reproducir»: una bolita recorrerá el árbol en PREORDEN "
                "(raíz → izquierdo → derecho) y retrocederá cuando termine cada rama." if modo == "pre" else
                "Pulsa «Siguiente» o «Reproducir»: cada nodo se calcula solo cuando sus hijos ya tienen "
                "valor, y los resultados suben hasta la raíz."))
            return
        n = orden[self.paso]
        k, total = self.paso + 1, len(orden)
        if modo == "pre":
            padres = mapa_padres(raiz)
            if id(n) in padres:
                p, lado = padres[id(n)]
                rol = f"hijo {lado} de «{p.etiqueta}»" if lado != "único" else f"hijo de «{p.etiqueta}»"
            else:
                rol = "es la RAÍZ (siempre se visita primero)"
            visitados = " ".join(m.token for m in orden[:k])
            fin = "   ✔ recorrido completo" if k == total else ""
            self.lbl_paso.configure(text=(
                f"Paso {k} de {total}:  visito «{n.etiqueta}» — {rol}.{fin}\n"
                f"Preorden hasta ahora:  {visitados}"))
        else:
            p = self._leer_parametros()
            x0 = p[0] if p else 0.0
            v = base._f_escalar(n, x0)
            if n.hijos:
                hs = ", ".join(fmt(base._f_escalar(h, x0)) for h in n.hijos)
                txt = f"Paso {k} de {total}:  calculo «{n.etiqueta}» con sus hijos ({hs})  →  {fmt(v)}"
            else:
                txt = f"Paso {k} de {total}:  la hoja «{n.etiqueta}» ya conoce su valor: {fmt(v)}"
            if k == total:
                txt += "   ✔ la raíz tiene el resultado final"
            self.lbl_paso.configure(text=txt)

    # ═══════════════════════ PESTAÑA «EVOLUCIÓN» ═══════════════════════
    def _evo_nuevo(self):
        return dict(gen=[], best=[], med=[], snaps=[], x=None, y=None, xs=None, vista=0)

    def detener_gp(self, publicar=True):
        corria = bool(self.gp_id)
        super().detener_gp(publicar)
        if corria and publicar and self.gp is not None and self._evo["x"] is not None:
            self._evo_registrar(self.gp, True, " (detenido por ti)")

    def _tab_evo(self, nb):
        t = ttk.Frame(nb, padding=6)
        nb.add(t, text="Evolución")
        g = ttk.Labelframe(t, text=" Cómo leer esta pantalla ", padding=5)
        g.pack(fill="x")
        ttk.Label(g, wraplength=1050, justify="left", text=(
            "IZQUIERDA: los puntos grises son tus datos (del CSV). La curva azul es lo que dibuja el MEJOR árbol "
            "de ese momento; las curvas violeta pálidas son los mejores anteriores (así ves cómo se va acercando). "
            "Los palitos rojos son el error: cuanto más cortos, mejor.\n"
            "DERECHA: el árbol de expresión (la ecuación en forma de grafo) de esa curva. "
            "ABAJO: el error de la población; si la línea baja, va aprendiendo (las estrellas = nuevo mejor).\n"
            "Tip: primero genera un CSV en «Datos (CSV)» con una f(x) sencilla, por ejemplo  x^2 - 4*x + 3.")
        ).pack(anchor="w")
        b = ttk.Frame(t)
        b.pack(fill="x", pady=(6, 0))
        ttk.Button(b, text="▶ Evolucionar desde cero", style="Accent.TButton",
                   command=lambda: self._evo_lanzar(False)).pack(side="left")
        ttk.Button(b, text="Evolucionar con semillas",
                   command=lambda: self._evo_lanzar(True)).pack(side="left", padx=6)
        ttk.Button(b, text="■ Detener", command=self.detener_gp).pack(side="left")
        ttk.Button(b, text="↻ Repetir animación", command=self._evo_repetir).pack(side="left", padx=6)
        for txt, var in (("población", self.var_pob), ("generaciones", self.var_gens)):
            ttk.Label(b, text="  " + txt).pack(side="left")
            ttk.Entry(b, textvariable=var, width=6).pack(side="left", padx=2)
        ttk.Button(b, text="Usar este árbol → pestaña Árbol", command=self._evo_usar).pack(side="right")
        r = ttk.Frame(t)
        r.pack(fill="x", pady=(4, 0))
        ttk.Label(r, text="Línea de tiempo (arrastra para ver cada mejora):").pack(side="left")
        self.sl_evo = ttk.Scale(r, from_=0, to=1, length=420, command=self._evo_slider)
        self.sl_evo.pack(side="left", padx=8)
        self.sl_evo.set(1.0)

        self.lbl_evo2 = ttk.Label(t, wraplength=1050, justify="left", foreground="#4338CA",
                                  font=("TkDefaultFont", 11, "bold"))
        self.lbl_evo2.pack(side="bottom", anchor="w", pady=(2, 2))
        self.lbl_evo = ttk.Label(t, wraplength=1050, justify="left", foreground="#9F1239",
                                 font=("TkDefaultFont", 11, "bold"))
        self.lbl_evo.pack(side="bottom", anchor="w")

        self.fig_evo = base.Figure(facecolor=LIENZO)
        gs = self.fig_evo.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1.25, 0.6],
                                       left=0.06, right=0.985, top=0.95, bottom=0.10,
                                       hspace=0.40, wspace=0.12)
        self.ax_ec = self.fig_evo.add_subplot(gs[0, 0])
        self.ax_et = self.fig_evo.add_subplot(gs[0, 1])
        self.ax_econv = self.fig_evo.add_subplot(gs[1, :])
        self.canvas_evo = self._canvas(t, self.fig_evo)
        self._evo_vacio()

    def _evo_vacio(self):
        for ax in (self.ax_ec, self.ax_et, self.ax_econv):
            ax.clear()
            ax.set_axis_off()
        self.ax_ec.text(0.5, 0.5, "Carga o genera datos (pestaña «Datos (CSV)»)\ny pulsa «Evolucionar desde cero»",
                        ha="center", va="center", fontsize=13, color="#818CF8", transform=self.ax_ec.transAxes)
        self.canvas_evo.draw_idle()

    def _evo_lanzar(self, sembrar):
        if self.dx is None:
            self.lbl_evo.configure(text="Primero genera un CSV o carga datos (pestaña «Datos (CSV)»).")
            return
        self.var_sembrar.set(sembrar)
        self.lbl_evo.configure(text="")
        self.evolucionar()
        err = self.lbl_err.cget("text")
        if err:
            self.lbl_evo.configure(text=err)

    # ---------- registro de cada generación ----------
    def _evo_registrar(self, gp, fin, motivo=""):
        h = self._evo
        if h["x"] is None:
            h["x"], h["y"] = gp.x.copy(), gp.y.copy()
            pad = 0.03 * (np.ptp(gp.x) or 1.0)
            h["xs"] = np.linspace(gp.x.min() - pad, gp.x.max() + pad, 500)
        fin_ = [i[1] for i in gp.pob if np.isfinite(i[1])]
        h["gen"].append(gp.gen)
        h["best"].append(gp.mejor_fvu if np.isfinite(gp.mejor_fvu) else np.nan)
        h["med"].append(float(np.median(fin_)) if fin_ else np.nan)
        sn = h["snaps"]
        mejoro = (not sn) or gp.mejor_fvu < sn[-1]["fvu"] * (1 - 1e-6) - 1e-12
        if mejoro:
            sn.append(dict(gen=gp.gen, fvu=gp.mejor_fvu, arbol=gp.mejor.copiar()))
        elif fin and sn and gp.mejor_fvu <= sn[-1]["fvu"]:
            sn[-1] = dict(gen=sn[-1]["gen"], fvu=gp.mejor_fvu, arbol=gp.mejor.copiar())
            mejoro = True
        i = len(sn) - 1
        h["vista"] = i
        if mejoro or fin:
            self._evo_curva(i)
            self._evo_arbol(i)
        if mejoro or fin or gp.gen % 2 == 0:
            self._evo_conv()
        self._evo_textos(gp.gen, i, fin=fin, mejoro=mejoro, motivo=motivo, gp=gp)
        self._evo_lock = True
        self.sl_evo.set(1.0)
        self._evo_lock = False
        self.canvas_evo.draw_idle()

    # ---------- paneles ----------
    def _evo_curva(self, i, t=1.0):
        ax, h = self.ax_ec, self._evo
        sn, x, y, xs = h["snaps"], h["x"], h["y"], h["xs"]
        ax.clear()
        ax.set_axis_on()

        def curva(a):
            with np.errstate(all="ignore"):
                v = np.asarray(a.evaluar(xs), dtype=float)
            v = np.where(np.isfinite(v), v, np.nan)
            return v

        cur = curva(sn[i]["arbol"])
        if t < 1 and i > 0:
            prev = curva(sn[i - 1]["arbol"])
            cur = np.where(np.isfinite(prev) & np.isfinite(cur), (1 - t) * prev + t * cur, cur)
        ini = max(0, i - 7)
        for k in range(ini, i):
            ax.plot(xs, curva(sn[k]["arbol"]), color="#A78BFA", lw=1.5, zorder=2,
                    alpha=0.15 + 0.45 * (k - ini + 1) / (i - ini + 1))
        sub = np.linspace(0, len(x) - 1, min(len(x), 70)).astype(int)
        yh = np.interp(x[sub], xs, np.nan_to_num(cur, nan=np.nan))
        ok = np.isfinite(yh)
        ax.vlines(x[sub][ok], y[sub][ok], yh[ok], color="#F43F5E", lw=1.3, alpha=0.75, zorder=3,
                  label="error")
        ax.plot(xs, cur, color="#4F46E5", lw=3.2, zorder=4, label=f"mejor (gen {sn[i]['gen']})")
        ax.scatter(x, y, s=16, color="#475569", alpha=0.8, zorder=5, label="tus datos")
        mg = (np.ptp(y) or 1.0) * 0.3
        ax.set_ylim(y.min() - mg, y.max() + mg)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_title("La curva se acerca a los datos", fontsize=11, fontweight="bold", color="#4338CA")
        ax.grid(alpha=0.35)
        ax.legend(loc="best", fontsize=8.5, framealpha=0.92)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    def _evo_arbol(self, i):
        s = self._evo["snaps"][i]
        t = s["arbol"]
        orden = [n for n, _ in preorden(t)]
        dibujar_arbol_v2(self.ax_et, t, {}, orden, "pre", None, None, None, "", False, False,
                         titulo=f"ÁRBOL DEL MEJOR  (gen {s['gen']}  ·  {len(orden)} nodos  ·  altura {altura(t)})",
                         leyenda=False)

    def _evo_conv(self, marcar=None):
        ax, h = self.ax_econv, self._evo
        ax.clear()
        ax.set_axis_on()
        g = np.array(h["gen"], float)
        b = np.maximum(np.array(h["best"], float), 1e-16)
        m = np.maximum(np.array(h["med"], float), 1e-16)
        ax.semilogy(g, m, color="#C4B5FD", lw=2, label="mediana de la población")
        ax.semilogy(g, b, color="#4F46E5", lw=2.8, label="el mejor")
        for s in h["snaps"]:
            ax.scatter([s["gen"]], [max(s["fvu"], 1e-16)], marker="*", s=110, color="#F59E0B",
                       zorder=5, edgecolors="white", linewidths=0.8)
        if marcar is not None:
            ax.axvline(marcar, color="#F43F5E", ls="--", lw=1.6)
        ax.set_xlim(0, max(getattr(self, "gp_max", 10), 10))
        ax.set_xlabel("generación")
        ax.set_ylabel("error (1−R²)\nmás abajo = mejor", fontsize=9)
        ax.set_title("Qué tan bien lo hace la población", fontsize=11, fontweight="bold", color="#4338CA")
        ax.grid(alpha=0.35, which="both")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    def _evo_textos(self, gen, i, fin=False, mejoro=False, motivo="", gp=None, vista=False):
        h = self._evo
        sn = h["snaps"]
        s = sn[i]
        t = s["arbol"]
        nodos, alt = len(preorden(t)), altura(t)
        ant = sn[i - 1]["fvu"] if i > 0 else None
        if vista:
            msg = (f"Mejora #{i + 1} de {len(sn)} (generación {s['gen']}): " +
                   (f"el error bajó de {fmt_m(ant)} a {fmt_m(s['fvu'])}. " if ant is not None
                    else f"árbol inicial, error {fmt_m(s['fvu'])}. ") +
                   f"Tiene {nodos} nodos y altura {alt}.")
        elif fin:
            msg = f"Terminó en la generación {gen}{motivo}. Mejor error: {fmt_m(s['fvu'])}."
        elif mejoro:
            msg = (f"Generación {gen}: ¡nuevo mejor árbol!  " +
                   (f"error {fmt_m(ant)} → {fmt_m(s['fvu'])}.  " if ant is not None else "") +
                   f"{nodos} nodos, altura {alt}.")
        else:
            igual = gp.sin_mejora if gp is not None else 0
            sac = gp.kicks if gp is not None else 0
            msg = f"Generación {gen}: el mejor sigue igual ({igual} generaciones sin cambio · {sac} sacudidas)."
        self.lbl_evo.configure(text=msg)
        pct = max(0.0, 1 - s["fvu"]) * 100 if np.isfinite(s["fvu"]) else 0.0
        extra = f"\nTu f(x) original:  {a_infija(self.raiz)}" if self.raiz is not None else ""
        self.lbl_evo2.configure(text=(f"Ecuación del mejor:  f(x) = {a_infija(t)}      "
                                      f"→ se parece a los datos en un {pct:.2f} %" + extra))

    # ---------- línea de tiempo y repetición ----------
    def _evo_mostrar(self, i):
        h = self._evo
        sn = h["snaps"]
        if not sn:
            return
        i = max(0, min(i, len(sn) - 1))
        h["vista"] = i
        self._evo_curva(i)
        self._evo_arbol(i)
        self._evo_conv(marcar=sn[i]["gen"])
        self._evo_textos(sn[i]["gen"], i, vista=True)
        self._evo_lock = True
        self.sl_evo.set(i / (len(sn) - 1) if len(sn) > 1 else 1.0)
        self._evo_lock = False
        self.canvas_evo.draw_idle()

    def _evo_slider(self, v):
        if self._evo_lock or self.gp_id or self._rp_id:
            return
        n = len(self._evo["snaps"])
        if n:
            self._evo_mostrar(int(round(float(v) * (n - 1))))

    def _evo_stop_replay(self):
        if self._rp_id:
            self.root.after_cancel(self._rp_id)
            self._rp_id = None

    def _evo_repetir(self):
        n = len(self._evo["snaps"])
        if self.gp_id or n < 2:
            self.lbl_evo.configure(text="Espera a que termine la evolución (o deténla) para repetir la animación.")
            return
        self._evo_stop_replay()
        paso = max(1, int(math.ceil(n / 40)))                 # máx. ~40 cuadros clave
        self._rp_lista = list(range(0, n, paso))
        if self._rp_lista[-1] != n - 1:
            self._rp_lista.append(n - 1)
        self._evo_mostrar(self._rp_lista[0])
        self._rp_k = 0
        self._rp_id = self.root.after(600, self._rp_siguiente)

    def _rp_siguiente(self):
        self._rp_k += 1
        if self._rp_k >= len(self._rp_lista):
            self._rp_id = None
            return
        self._rp_t0 = time.monotonic()
        self._rp_frame()

    def _rp_frame(self):
        i = self._rp_lista[self._rp_k]
        f = (time.monotonic() - self._rp_t0) / 0.9
        if f < 1:
            self._evo["snaps"]  # curva intermedia: se mezcla con la anterior de la lista
            self._evo_curva_mezcla(self._rp_lista[self._rp_k - 1], i, f * f * (3 - 2 * f))
            self._rp_id = self.root.after(30, self._rp_frame)
        else:
            self._evo_mostrar(i)
            self._rp_id = self.root.after(450, self._rp_siguiente)

    def _evo_curva_mezcla(self, i0, i1, t):
        """Curva que se transforma de la mejora i0 a la mejora i1 (solo se redibuja la gráfica izquierda)."""
        h = self._evo
        sn, xs = h["snaps"], h["xs"]
        with np.errstate(all="ignore"):
            a = np.asarray(sn[i0]["arbol"].evaluar(xs), dtype=float)
            b = np.asarray(sn[i1]["arbol"].evaluar(xs), dtype=float)
        a = np.where(np.isfinite(a), a, np.nan)
        b = np.where(np.isfinite(b), b, np.nan)
        mezcla = np.where(np.isfinite(a) & np.isfinite(b), (1 - t) * a + t * b, b)
        ax, x, y = self.ax_ec, h["x"], h["y"]
        ax.clear()
        ax.plot(xs, mezcla, color="#4F46E5", lw=3.2, zorder=4, label=f"mejora → gen {sn[i1]['gen']}")
        ax.scatter(x, y, s=16, color="#475569", alpha=0.8, zorder=5, label="tus datos")
        mg = (np.ptp(y) or 1.0) * 0.3
        ax.set_ylim(y.min() - mg, y.max() + mg)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_title("La curva se acerca a los datos", fontsize=11, fontweight="bold", color="#4338CA")
        ax.grid(alpha=0.35)
        ax.legend(loc="best", fontsize=8.5, framealpha=0.92)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        self.canvas_evo.draw_idle()

    def _evo_usar(self):
        sn = self._evo["snaps"]
        if not sn:
            self.lbl_evo.configure(text="Todavía no hay ningún árbol evolucionado.")
            return
        t = sn[self._evo["vista"]]["arbol"].copiar()
        self.var_expr.set(a_prefija(t) if self.var_modo.get() == "prefija" else a_infija(t))
        self.calcular()
        if not self.lbl_err.cget("text"):
            self.nb.select(0)


def main():
    if not base.HAY_TK:
        raise SystemExit("Falta tkinter. En Ubuntu/Debian:  sudo apt install python3-tk")
    base.BG, base.HEADER, base.ACCENT = BG, HEADER, ACCENT
    base.GP = GPv2
    for m in MODELOS_NUEVOS:
        if all(q["nombre"] != m["nombre"] for q in base.MODELOS):
            base.MODELOS.append(m)
    matplotlib.rcParams.update({
        "axes.facecolor": LIENZO, "axes.edgecolor": "#CBD5E1", "grid.color": "#E2E8F0",
        "axes.labelcolor": "#475569", "xtick.color": "#64748B", "ytick.color": "#64748B",
        "text.color": "#334155", "axes.titleweight": "bold"})
    root = tk.Tk()
    root.title("Árbol de Expresión · v2")
    AppV2(root)
    root.mainloop()


if __name__ == "__main__":
    main()