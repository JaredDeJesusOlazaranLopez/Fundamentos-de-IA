import heapq
from collections import defaultdict

aristas = [
    (29, 7, 5), (29, 21, 3), (7, 11, 5), (7, 14, 3), (7, 6, 10), (7, 1, 20),
    (21, 11, 2), (21, 24, 2), (11, 24, 7), (11, 14, 9), (11, 10, 7),
    (14, 6, 5), (14, 9, 5), (14, 10, 9), (14, 4, 8),
    (6, 15, 3), (6, 8, 4), (6, 9, 7),
    (15, 0, 8), (15, 8, 2), (15, 1, 8),
    (0, 44, 70), (0, 1, 2), (44, 3, 80), (44, 5, 69),
    (3, 5, 2), (3, 2, 9), (3, 69, 6),
    (5, 20, 2), (5, 69, 3), (20, 69, 3), (20, 25, 7),
    (69, 25, 6), (69, 2, 11), (69, 17, 4), (69, 70, 8), (25, 70, 17),
    (8, 1, 7), (8, 12, 9), (8, 4, 15),
    (9, 10, 3), (10, 1, 5), (10, 4, 6), (10, 27, 9),
    (1, 2, 3), (1, 12, 7), (1, 23, 7),
    (2, 23, 3), (2, 17, 5), (23, 16, 11), (17, 16, 7), (17, 70, 6),
    (12, 16, 16), (12, 16, 3),
    (12, 18, 4), (12, 66, 7), (12, 13, 9), (16, 13, 3),
    (4, 18, 5), (4, 27, 3), (18, 66, 6), (27, 66, 1), (66, 13, 777),
]

grafo = defaultdict(list)
for a, b, w in aristas:
    grafo[a].append((b, w)); grafo[b].append((a, w))

def dijkstra(origen, destino):
    dist = {n: float("inf") for n in grafo}; previo = {}
    dist[origen] = 0; cola = [(0, origen)]
    while cola:
        d, u = heapq.heappop(cola)
        if d > dist[u]: continue
        if u == destino: break
        for v, w in grafo[u]:
            if d + w < dist[v]:
                dist[v] = d + w; previo[v] = u
                heapq.heappush(cola, (dist[v], v))
    camino, n = [], destino
    while n != origen:
        camino.append(n); n = previo[n]
    camino.append(origen)
    return dist[destino], camino[::-1]

def todos_los_caminos(origen, destino, tope):
    res = []
    def dfs(u, vis, cam, costo):
        if costo > tope: return
        if u == destino:
            res.append((costo, cam[:])); return
        for v, w in grafo[u]:
            if v not in vis:
                vis.add(v); cam.append(v)
                dfs(v, vis, cam, costo + w)
                cam.pop(); vis.remove(v)
    dfs(origen, {origen}, [origen], 0)
    return sorted(res)

if __name__ == "__main__":
    c, p = dijkstra(5, 4)
    print(f"DIJKSTRA 5 -> 4: {' -> '.join(map(str,p))}  (costo {c})\n")
    TOPE = 40   # sube/baja este valor segun cuantos caminos quieras
    caminos = todos_los_caminos(5, 4, TOPE)
    print(f"Caminos con costo <= {TOPE}: {len(caminos)}\n")
    for cost, cam in caminos[:25]:
        print(f"{cost:>4}  {' -> '.join(map(str,cam))}")
    with open("/mnt/user-data/outputs/caminos_5_a_4.txt", "w") as f:
        for cost, cam in caminos:
            f.write(f"{cost}\t{' -> '.join(map(str,cam))}\n")