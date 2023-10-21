import numpy as np
import queue

class Grafo:
    def __init__(self, arquivo: str) -> None:
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
                self.rotulos[int(vertice)-1] = rotulo

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

    # Vizualizing
    def mostrar_grafo(self) -> None:
        values = [x for x in self.grafo if x != np.inf]
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

    def vizinhos(self, v: int) -> list[str]:
        vizinhos = []
        for i in range(self.vertices):
            if self[v, i] != np.inf and i != v:
                vizinhos.append(self.rotulo(i))
        return vizinhos

    def haAresta(self, u: int, v: int) -> bool:
        return self[u, v] != np.inf
    
    def peso(self, u: int, v: int) -> float:
        return self[u, v]

    def resposta_1(self):
        self.mostrar_grafo()
        print(f"Vertices: {self.qtdVertices()}")
        print(f"Arestas: {self.qtdArestas()}")  
        for i in range(self.vertices):
            print(f"Grau({self.rotulo(i)}): {self.grau(i)}")
        for i in range(self.vertices):
            print(f"Vizinhos({self.rotulo(i)}): {self.vizinhos(i)}")

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

    # Não funciona
    def detect_cycle(self) -> bool:
        visited = np.zeros(self.vertices, dtype=bool)
        for v in range(self.vertices):
            if not visited[v]:
                if self.detect_cycle_visit(v, visited, -1):
                    return True
        return False
    
    def detect_cycle_visit(self, v: int, visited: list, parent: int) -> bool:
        visited[v] = True
        for u in self.vizinhos_saintes(v):
            if not visited[u]:
                if self.detect_cycle_visit(u, visited, v):
                    return True
            elif u != parent:
                return True
        return False

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



    def Kruskal(self) -> list:
        A = []
        S = []
        for i in range(self.vertices):
            S.append([i])
        arestas = []
        #percorrendo a matriz
        for i in range(self.vertices):
            for j in range(i+1, self.vertices):
                if self[i, j] != np.inf:
                    arestas.append([i, j, self[i, j]])
        arestas = sorted(arestas)
        for i in range(len(arestas)):
            u = arestas[i][0]
            v = arestas[i][1]
            if S[u] != S[v]:
                #print("S: ", S)
                A.append([u, v])
                S[u].append(S[u] + S[v])
                S[v] = S[u]
        return A

    def lightest(self, arestas: list) -> list:
        menor = np.inf
        for i in range(len(arestas)):
            if arestas[i][2] < menor:
                menor = arestas[i][2]
                aux = i
        return arestas.pop(aux)

    def Prim(self) -> list:
        #seleciona um vertice aleatorio
        r = 0
        predecessors = np.full(self.vertices, -1)
        weight = np.ones(self.vertices)*np.inf
        Q = []
        for i in range(self.vertices):
            Q.append(i)
        aux = self.arestas
        while not Q:
            a = self.lightest(aux)
            Q.remove(a[1])

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
                visited, predecessors, search_time, begin_time, time = self.DFS_Visit(u, visited, predecessors, search_time, begin_time, time, first)
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
                    self.DFS_Visit(u, visited, predecessors, search_time, begin_time, time, False)
        else:
            for u in range(self.vertices):
                if not visited[u]:
                    visited, predecessors, search_time, begin_time, time = self.DFS_Visit(u, visited, predecessors, search_time, begin_time, time, True)
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
        if self.detect_cycle():
            raise Exception("Grafo cíclico!")
        visited = np.zeros(self.vertices, dtype=bool)
        begin_time = np.ones(self.vertices, dtype=int)*np.inf
        search_time = np.ones(self.vertices, dtype=int)*np.inf
        tempo = 0
        lista = []
        for u in range (self.vertices):
            if not visited[u]:
                lista = self.DFS_Visit_OT(u, visited, begin_time, search_time, tempo, lista)

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
