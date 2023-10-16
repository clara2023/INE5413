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

class GrafoDirigido(Grafo):
    def __init__(self, arquivo: str) -> None:
        super().__init__(arquivo)
    
    def __getitem__(self, key: int) -> float:
        return self.grafo[key]

    def __setitem__(self, key: int, value: int) -> None:
        self.grafo[key] = value
    
    def criar_grafo(self, vertices: int) -> np.ndarray:
        return np.ones((vertices, vertices))*np.inf

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
