import numpy as np
import queue
from itertools import chain, combinations

class Grafo:
    def __init__(self, arquivo: str) -> None:
        if arquivo == "":
            self.grafo = None
            self.vertices = 0
            self.arestas = 0
            self.graus = np.zeros(self.vertices, dtype=int)
            self.rotulos = np.empty(self.vertices, dtype=object)
            self.vizinhos = np.empty(self.vertices, dtype=object)
            return
        # Leitura do arquivo
        with open(arquivo) as file:
            # *vertices n
            self.vertices = int(file.readline().split()[1])
            self.grafo = self.criar_grafo(self.vertices)
            self.graus = np.zeros(self.vertices, dtype=int)
            self.rotulos = np.empty(self.vertices, dtype=object)
            # n rotulo_n
            for _ in range(self.vertices):
                vertice, *rotulo = file.readline().split()
                rotulo = ' '.join(rotulo)[1:-1]
                self.rotulos[int(vertice)-1] = rotulo.strip('"')
            self._vizinhos = np.empty(self.qtdVertices(), dtype=object)
            # *edges
            file.readline()
            self.arestas = 0
            # a b peso
            while line := file.readline().strip():
                self.arestas += 1
                a, b, peso = line.split()
                a, b, peso = int(a), int(b), peso
                if self[a-1, b-1] == np.inf: 
                    self[a-1, b-1] = peso
                    self.graus[a-1] += 1
                    self.graus[b-1] += 1
                else:
                    print(f"Aresta ({a}, {b}) já existe!")
                self.add_neighbour(int(a), int(b))

    # Vizualizing
    def mostrar_grafo(self) -> None:
        values = [x for x in self.grafo.flatten() if x != np.inf]
        max_value_size = int(np.ceil(np.log10(max(values))))+2
        print(("T"+"-"*(max_value_size+1))*(self.vertices)+"T")
        for i in range(self.vertices):
            for j in range(self.vertices):
                coisa = self[i, j] if self[i, j] != np.inf else "∞"
                print(f"|{coisa:^8}", end="")
            print("|")
            if i == self.vertices-1:
                print(("⊥"+"-"*(max_value_size+1))*(self.vertices)+"⊥")
            else:
                print(("|"+"-"*(max_value_size+1))*(self.vertices)+"|")
    
    def grafo_completo(self) -> np.ndarray:
        distances = np.ones((self.vertices, self.vertices))*np.inf
        for i in range(self.vertices):
            for j in range(self.vertices):
                distances[i, j] = self[i, j]
        return distances

    # Trabalho 1, questão 1
    def qtdVertices(self) -> int:
        return self.vertices
    
    def qtdArestas(self) -> int:
        return self.arestas

    def grau(self, v: int) -> int:
        return self.graus[v]

    def rotulo(self, v: int) -> str: 
        return self.rotulos[v]

    def vizinhos(self, v: int) -> list[int]:
        return self._vizinhos[v]

    def add_neighbour(a, b): raise NotImplementedError("Método abstrato")

    def haAresta(self, u: int, v: int) -> bool:
        return self[u, v] != np.inf
    
    def peso(self, u: int, v: int) -> float:
        return self[u, v]

    # Trabalho 1, questão 2
    def busca_em_largura(self, s: int) -> (np.ndarray, np.ndarray):
        visitados = np.zeros(self.vertices, dtype=bool)
        fila = queue.Queue()
        fila.put(s)
        visitados[s] = 1
        distancias = np.zeros(self.vertices)
        arvore = np.ones(self.vertices, dtype=int)*-1
        while not fila.empty():
            atual = fila.get()
            for i in range(self.vertices):
                if self[atual, i] != np.inf and not visitados[i]:
                    fila.put(i)
                    visitados[i] = 1
                    arvore[i] = atual
                    distancias[i] = distancias[atual] + 1
        return (arvore, distancias)

    # Trabalho 1, questão 3
    def ciclo_euleriano(self):
        grafo_copy = self.grafo_completo()
        ciclo_euleriano = []

        # Escolhe um vértice para começar, de preferência um de grau ímpar ou diferente de 0
        ponto_partida = -1
        for i in range(self.vertices):
            if self.grau(i) % 2 != 0:
                ponto_partida = i
                break
        else:
            for i in range(self.vertices):
                if self.grau(i) != 0:
                    ponto_partida = i
                    break
            else:
                return []  # Não há arestas no grafo

        # Comece a construir o ciclo Euleriano a partir do ponto de partida
        v = ponto_partida
        ciclo_euleriano.append(v)
        while True:
            for i in range(self.vertices):
                if grafo_copy[v, i] != np.inf and i != v:
                    aresta_valida = i
                    break
            else:
                break
            grafo_copy[v, aresta_valida] = np.inf
            grafo_copy[aresta_valida, v] = np.inf
            v = aresta_valida
            ciclo_euleriano.append(v)

        # Verifica se o caminho euleriano encontrado é ciclo
        if ciclo_euleriano[0] != ciclo_euleriano[-1] or (len(ciclo_euleriano) == 1 and self.grau(ciclo_euleriano[0]) != 0):
            return None

        # Verifique se todas as arestas foram percorridas
        for i in range(self.vertices):
            for j in range(self.vertices):
                if grafo_copy[i, j] != np.inf and i != j:
                    return None
        return ciclo_euleriano

    # Trabalho 1, questão 4
    def check_for_negatives(self) -> bool:
        return any(self[i, j] < 0 for i in range(self.vertices) for j in range(i, self.vertices))
    
    def bellman_ford(self, s: int, save_path: bool=True) -> (np.ndarray, np.ndarray) or np.ndarray:
        distancias = np.ones(self.vertices)*np.inf
        distancias[s] = 0
        origem = np.ones(self.vertices)*-1
        for _ in range(self.vertices - 1):
            # Relaxar aresta (u, v)
            for u in range(self.vertices):
                for v in range(self.vertices):
                    if self[u, v] and distancias[v] > distancias[u] + self[u, v]:
                        distancias[v] = distancias[u] + self[u, v]
                        origem[v] = u
        for _ in range(self.vertices - 1):
            # Relaxar aresta (u, v)
            for u in range(self.vertices):
                for v in range(self.vertices):
                    if self[u, v] != np.inf and distancias[u] != -np.inf:
                        if distancias[v] > distancias[u] + self[u, v]:
                            distancias[v] = -np.inf

        return (distancias, origem) if save_path else distancias
    
    def dijkstra(self, s: int, save_path: bool=True) -> (np.ndarray, np.ndarray) or np.ndarray:
        if self.check_for_negatives(): raise Exception("O grafo possui arestas negativas")
        distancias = np.ones(self.vertices)*np.inf
        distancias[s] = 0
        fila = queue.PriorityQueue()
        fila.put((0, s))
        origem = np.ones(self.vertices, dtype=int)*-1
        while not fila.empty():
            dist, u = fila.get()
            if dist > distancias[u]: continue
            for i in range(self.vertices):
                if distancias[i] > distancias[u] + self[u, i]:
                    distancias[i] = distancias[u] + self[u, i]
                    origem[i] = u
                    fila.put((distancias[i], i))
        return (distancias, origem) if save_path else distancias

    # Trabalho 1, questão 5
    def floyd_warshall(self, save_path=True) -> (np.ndarray, np.ndarray):
        distances = self.grafo_completo()
        origem = np.ones((self.vertices, self.vertices), dtype=int)*-1
        for i in range(self.vertices):
            for j in range(self.vertices):
                if distances[i, j] != np.inf:
                    origem[i, j] = i

        for k in range(self.vertices):
            for u in range(self.vertices):
                for v in range(self.vertices):
                    if distances[u, v] > distances[u, k] + distances[k, v]:
                        distances[u, v] = distances[u, k] + distances[k, v]
                        origem[u, v] = k

        for k in range(self.vertices):
            for u in range(self.vertices):
                for v in range(self.vertices):
                    if self[u, v] != np.inf and distances[u, v] != np.inf:
                        if distances[u, v] > distances[u, k] + distances[k, v]:
                            distances[u, v] = -np.inf
        
        return (distances, origem) if save_path else distances

    def transposta(self) -> 'Grafo':
        return self.grafo_completo().T

    def vizinhos_saintes(self, v: int) -> list:
        return [i for i in range(self.vertices) if self[v, i] != np.inf and i != v]
    
    def vizinhos_saintes_transposta(self, v: int) -> list:
        return [i for i in range(self.vertices) if self[i, v] != np.inf and i != v]

    def DFS_CFC(self) -> list:
        raise NotImplementedError("Esse tipo de grafo não suporta essa operação")

    def DFS_CFC_Visit(self, v: int, visited: list, predecessors: list, search_time:list, begin_time: list, time, first: bool) -> list:
        raise NotImplementedError("Esse tipo de grafo não suporta essa operação")

    def DFS_2(self, first: bool, visited: list, predecessors: list, search_time:list, begin_time: list) -> list:
        raise NotImplementedError("Esse tipo de grafo não suporta essa operação")
    
    def DFS_Visit_OT(self, v, visited, begin_time, search_time, tempo, lista):
        raise NotImplementedError("Esse tipo de grafo não suporta essa operação")

    def DFS_OT(self) -> list:
        raise NotImplementedError("Esse tipo de grafo não suporta essa operação")

    def kruskal(self) -> list:
        A = set()
        S = np.empty(self.qtdVertices(), dtype=object)
        for i in range(self.qtdVertices()):
            S[i] = {i}
        arestas = queue.PriorityQueue()
        
        # Percorrendo a matriz e ordenando suas arestas por peso
        for i in range(self.vertices):
            for j in range(i+1, self.vertices):
                if self[i, j] != np.inf:
                    arestas.put((self[i, j], i, j))
        
        for i in range(arestas.qsize()):
            peso, u, v = arestas.get()
            if S[u] != S[v]:
                A.add(frozenset((u, v)))
                x = S[u].union(S[v])
                for y in x:
                    S[y] = x
        return A

    def prim(self) -> list:
        # seleciona um vertice aleatorio
        r = 0
        predecessors = np.full(self.vertices, -1)
        weight = np.ones(self.vertices)*np.inf
        visited = np.zeros(self.vertices, dtype=bool)
        Q = queue.PriorityQueue()
        Q.put((0, r))
        weight[r] = 0
        while not Q.empty():
            u = Q.get()[1]
            for v in range(self.vertices):
                if self[u, v] != np.inf and self[u, v] < weight[v] and not visited[v] and u != v:
                    predecessors[v] = u
                    weight[v] = self[u, v]
                    Q.put((weight[v], v))
            visited[u] = True
        return predecessors

class GrafoDirigido(Grafo):
    def __init__(self, arquivo: str) -> None:
        super().__init__(arquivo)
    
    def __getitem__(self, key: int) -> float:
        return self.grafo[key]

    def __setitem__(self, key: int, value: int) -> None:
        self.grafo[key] = value
    
    def criar_grafo(self, vertices: int) -> np.ndarray:
        return np.ones((vertices, vertices))*np.inf
    
    def grafo_completo(self) -> np.ndarray:
        return self.grafo

    def add_neighbour(self, a, b):
        if self._vizinhos[a-1] is None:
            self._vizinhos[a-1] = [b-1]
        else:
            self._vizinhos[a-1].append(b-1)

    def DFS_CFC(self) -> list:
        visited = np.zeros(self.vertices, dtype=bool)
        begin_time = np.ones(self.vertices)*np.inf
        predecessors = np.full(self.vertices, -1)
        search_time = np.ones(self.vertices)*np.inf
        visited, predecessors, search_time, begin_time, time = self.DFS_2(True, visited, predecessors, search_time, begin_time,)
        visited = np.zeros(self.vertices, dtype=bool)
        begin_time = np.ones(self.vertices)*np.inf
        predecessors = np.full(self.vertices, -1)
        visited, predecessors, search_time, begin_time, time = self.DFS_2(False, visited, predecessors, search_time, begin_time,)
        return predecessors

    def DFS_CFC_Visit(self, v: int, visited: list, predecessors: list, search_time:list, begin_time: list, time, first: bool) -> list:
        v = int(v)
        visited[v] = 1
        time += 1
        begin_time[v] = time
        saintes = self.vizinhos_saintes(v) if first else self.vizinhos_saintes_transposta(v)
        for u in saintes:
            if visited[u] == 0:
                predecessors[u] = v
                visited, predecessors, search_time, begin_time, time = self.DFS_CFC_Visit(u, visited, predecessors, search_time, begin_time, time, first)
        time += 1
        search_time[v] = time
        return visited, predecessors, search_time, begin_time, time

    def DFS_2(self, first: bool, visited: list, predecessors: list, search_time:list, begin_time: list) -> list:
        time = 0
        if not first:
            lista = {}
            for i in range (search_time.size):
                lista[i] = search_time[i]
            aux = dict(sorted(lista.items(), key=lambda item: item[1], reverse=True))
            for u in aux.keys():
                if not visited[u]:
                    self.DFS_CFC_Visit(u, visited, predecessors, search_time, begin_time, time, False)
        else:
            for u in range(self.vertices):
                if not visited[u]:
                    visited, predecessors, search_time, begin_time, time = self.DFS_CFC_Visit(u, visited, predecessors, search_time, begin_time, time, True)
        return visited, predecessors, search_time, begin_time, time

    def DFS_Visit_OT(self, v, visited, begin_time, search_time, tempo, lista):
        visited[v] = 1
        tempo += 1
        begin_time[v] = tempo
        v_saintes = self.vizinhos_saintes(v)
        for u in v_saintes:
            if not visited[u]:
                lista = self.DFS_Visit_OT(u, visited, begin_time, search_time, tempo, lista)
        tempo += 1
        search_time[v] = tempo
        lista.insert(0, v)
        return lista

    def DFS_OT(self) -> list:
        visited = np.zeros(self.vertices, dtype=bool)
        begin_time = np.ones(self.vertices, dtype=int)*np.inf
        search_time = np.ones(self.vertices, dtype=int)*np.inf
        tempo = 0
        lista = []
        for u in range (self.vertices):
            if not visited[u]:
                lista = self.DFS_Visit_OT(u, visited, begin_time, search_time, tempo, lista)
        return lista

    # Trabalho 3
    def edmonds_karp(self, s: int, t: int) -> float:
        if s == t: return 0
        max_flow = 0
        flow = np.zeros((self.vertices, self.vertices))
        while True:
            predecessor = self.edmonds_karp_bfs(s, t, flow)
            print(predecessor)
            if predecessor[t] == -1: break
            path_flow = np.inf
            v = t
            while v != s:
                u = predecessor[v]
                path_flow = min(path_flow, self[u, v] - flow[u, v])
                v = u
            v = t
            while v != s:
                u = predecessor[v]
                flow[u, v] += path_flow
                flow[v, u] -= path_flow
                v = u
            max_flow += path_flow
        return max_flow

    def edmonds_karp_bfs(self, s, t, flow_graph):
        predecessor = np.ones(self.vertices, dtype=int)*-1
        predecessor[s] = -2
        fila = queue.Queue()
        fila.put(s)
        while not fila.empty() and predecessor[t] == -1:
            u = fila.get()
            for v in self.vizinhos(u):
                if predecessor[v] == -1 and self[u, v] - flow_graph[u, v] > 0:
                    predecessor[v] = u
                    fila.put(v)
        return predecessor

class GrafoNaoDirigido(Grafo):
    def __init__(self, arquivo: str) -> None:
        super().__init__(arquivo)

    def __getitem__(self, key: int) -> float:
        i, j = key
        if j >= self.vertices or i >= self.vertices: raise IndexError("Index out of range")
        
        if i > j: i, j = j, i
        if i < j: return self.grafo[int(self.vertices*i + j - i*(i+1)//2 - i - 1)]
        return 0
    
    def __setitem__(self, key: int, value: int) -> None:
        i, j = key
        if j >= self.vertices or i >= self.vertices: raise IndexError("Index out of range")

        if i > j: i, j = j, i        
        if i == j: return
        
        pos = self.vertices*i + j - i*(i+1)//2 - i - 1
        self.grafo[pos] = value

    def criar_grafo(self, vertices: int) -> np.ndarray:
        return np.ones((vertices*(vertices-1)//2,))*np.inf

    def add_neighbour(self, a, b):
        if self._vizinhos[a-1] is None:
            self._vizinhos[a-1] = [b-1]
        else:
            self._vizinhos[a-1].append(b-1)

        if self._vizinhos[b-1] is None:
            self._vizinhos[b-1] = [a-1]
        else:
            self._vizinhos[b-1].append(a-1)

    def add_neighbour(self, a, b):
        if self._vizinhos[a-1] is None:
            self._vizinhos[a-1] = [b-1]
        else:
            self._vizinhos[a-1].append(b-1)

        if self._vizinhos[b-1] is None:
            self._vizinhos[b-1] = [a-1]
        else:
            self._vizinhos[b-1].append(a-1)

    def lawler(self):
        def powerset(iterable):  # Jesus Cristo me liberte desse mal
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        X = {}
        X[0] = 0
        S_powerset = list(powerset(range(self.vertices)))
        for s in range(len(S_powerset)):
            S = S_powerset[s]
            X[s] = np.inf
            G_ = self.subgrafo(S)
        return X[len(X)-1]

    def subgrafo(self, S):
        grafo_novo = GrafoNaoDirigido("")
        grafo_novo.grafo = self.grafo.copy()
        grafo_novo.vertices = len(S)
        grafo_novo.graus = np.zeros(grafo_novo.vertices, dtype=int)
        grafo_novo.rotulos = np.empty(grafo_novo.vertices, dtype=object)
        for i in range(grafo_novo.vertices):
            grafo_novo.rotulos[i] = self.rotulos[i]
            for j in range(grafo_novo.vertices):
                grafo_novo[i, j] = self[S[i], S[j]]
                if grafo_novo[i, j] != np.inf and i != j:
                    grafo_novo.graus[i] += 1
        return grafo_novo

class GrafoBipartido(GrafoNaoDirigido):
    def __init__(self, arquivo: str) -> None:
        super().__init__(arquivo)
        self.X = set()
        self.Y = set()
    
    def bipartir(self, X: list[int]):
        for v in range(self.qtdVertices()):
            if v in X:
                self.X.add(v)
            else:
                self.Y.add(v)
    
    def hopcroft_karp(self):
        D = np.ones(self.qtdVertices())*np.inf
        mate = np.ones(self.qtdVertices())*-1
        m = 0
        while self.hopcroft_karp_bfs(mate, D):
            for x in self.X:
                if mate[x] == -1 and self.hopcroft_karp_dfs(x, mate, D):
                    m += 1
        return m, mate

    def hopcroft_karp_bfs(self, mate, D):
        Q = queue.Queue()
        for x in self.X:
            if mate[x] == -1:
                D[x] = 0
                Q.put(x)
            else:
                D[x] = np.inf
        D[-1] = np.inf
        while not Q.empty():
            x = int(Q.get())
            if D[int(x)] < D[-1]:
                for y in self.vizinhos(x):
                    if D[int(mate[y])] == np.inf:
                        D[int(mate[y])] = D[x] + 1
                        Q.put(mate[y])
        return D[-1] != np.inf
    
    def hopcroft_karp_dfs(self, x, mate, D):
        if x != -1:
            for y in range(self.qtdVertices()):
                if self[x, y] != np.inf and x != y and D[int(mate[y])] == D[int(x)] + 1 and self.hopcroft_karp_dfs(mate[y], mate, D):
                    mate[y] = x
                    mate[int(x)] = y
                    return True
            D[x] = np.inf
            return False
        return True