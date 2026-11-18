import tkinter as tk
from collections import deque

# Configuración del tablero
FILAS, COLS = 10, 10
TAM = 40

class HormigaSolver:
    def __init__(self, matriz, inicio, meta, alimentos):
        self.matriz = matriz
        self.inicio = inicio
        self.meta = meta
        self.alimentos = frozenset(alimentos)

    def obtener_vecinos(self, estado):
        pos, recoleccion = estado
        r, c = pos
        vecinos = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < FILAS and 0 <= nc < COLS and self.matriz[nr][nc] != 1:
                n_pos = (nr, nc)
                n_recoleccion = recoleccion | {n_pos} if n_pos in self.alimentos else recoleccion
                vecinos.append((n_pos, n_recoleccion))
        return vecinos

    def buscar(self, es_bfs=True):
        # Verifica si el inicio ya es alimento
        inicio_recoleccion = frozenset([self.inicio]) if self.inicio in self.alimentos else frozenset()
        inicio_estado = (self.inicio, inicio_recoleccion)
        
        frontera = deque([inicio_estado])
        padres = {inicio_estado: None}
        visitados = {inicio_estado}
        
        while frontera:
            actual = frontera.popleft() if es_bfs else frontera.pop()
            
            if actual[0] == self.meta and len(actual[1]) == len(self.alimentos):
                return self.reconstruir_camino(padres, actual)
            
            for vecino in self.obtener_vecinos(actual):
                if vecino not in visitados:
                    visitados.add(vecino)
                    padres[vecino] = actual
                    frontera.append(vecino)
        return None

    def reconstruir_camino(self, padres, final):
        camino = []
        curr = final
        while curr:
            camino.append(curr[0])
            curr = padres[curr]
        return camino[::-1]

class App:
    def __init__(self, root):
        self.root = root
        self.modo = "inicio"
        self.matriz = [[0 for _ in range(COLS)] for _ in range(FILAS)]
        self.inicio = (0, 0)
        self.meta = (FILAS-1, COLS-1)
        self.alimentos = set()
        
        self.canvas = tk.Canvas(root, width=COLS*TAM, height=FILAS*TAM)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.click)
        
        ctrl = tk.Frame(root)
        ctrl.pack()
        tk.Button(ctrl, text="Inicio", command=lambda: self.set_modo("inicio")).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Meta", command=lambda: self.set_modo("meta")).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Obstáculo", command=lambda: self.set_modo("obstaculo")).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Alimento", command=lambda: self.set_modo("alimento")).pack(side=tk.LEFT)
        tk.Button(ctrl, text="BFS", command=lambda: self.ejecutar("BFS")).pack(side=tk.LEFT)
        tk.Button(ctrl, text="DFS", command=lambda: self.ejecutar("DFS")).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Reiniciar", command=self.reset).pack(side=tk.LEFT)
        
        self.dibujar()

    def set_modo(self, m): self.modo = m

    def click(self, event):
        c, r = event.x // TAM, event.y // TAM
        if self.modo == "inicio": self.inicio = (r, c)
        elif self.modo == "meta": self.meta = (r, c)
        elif self.modo == "obstaculo": self.matriz[r][c] = 1 if self.matriz[r][c] == 0 else 0
        elif self.modo == "alimento":
            if (r, c) in self.alimentos: self.alimentos.remove((r, c))
            else: self.alimentos.add((r, c))
        self.dibujar()

    def dibujar(self):
        self.canvas.delete("all")
        for r in range(FILAS):
            for c in range(COLS):
                color = "white"
                if self.matriz[r][c] == 1: color = "black"
                elif (r, c) == self.inicio: color = "blue"
                elif (r, c) == self.meta: color = "red"
                elif (r, c) in self.alimentos: color = "green"
                self.canvas.create_rectangle(c*TAM, r*TAM, (c+1)*TAM, (r+1)*TAM, fill=color, outline="gray")

    def ejecutar(self, tipo):
        self.dibujar()
        solver = HormigaSolver(self.matriz, self.inicio, self.meta, self.alimentos)
        camino = solver.buscar(es_bfs=(tipo == "BFS"))
        if camino:
            for paso in camino:
                r, c = paso
                self.canvas.create_oval(c*TAM+12, r*TAM+12, (c+1)*TAM-12, (r+1)*TAM-12, fill="yellow")
            print(f"{tipo}: Solución encontrada. Pasos: {len(camino)}")
        else: print(f"{tipo}: No hay solución.")

    def reset(self):
        self.matriz = [[0 for _ in range(COLS)] for _ in range(FILAS)]
        self.alimentos.clear()
        self.dibujar()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Simulador de Hormiga")
    app = App(root)
    root.mainloop()
