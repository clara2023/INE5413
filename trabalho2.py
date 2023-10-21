import numpy as np
from funcoes_auxiliares import *

class Grafo:
    def __init__ (self, arquivo: str) -> None:
        with open(arquivo, "r") as file:
            self.vertices = int(file.readline().split()[1])
            total_cells = self.vertices**2
            self.grafo = np.ones((total_cells))*np.inf
            self.graus = np.zeros(self.vertices, dtype=int)
            self.rotulos = np.empty(self.vertices, dtype=object)
            self.file = arquivo

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

    def __getitem__(self, key: int) -> float:
        i, j = key
        if j >= self.vertices or i >= self.vertices: raise IndexError("Index out of range")
        if i == j: return 0
        pos = self.vertices*i + j
        return self.grafo[pos]

    def __setitem__(self, key: int, value: int) -> None:
        i, j = key
        if j >= self.vertices or i >= self.vertices: raise IndexError("Index out of range")
        pos = self.vertices*i + j
        self.grafo[pos] = value

    def get_peso(self, a: int, b: int) -> float:
        return self[a, b]

    def mostra_grafo(self):
        for i in range(self.vertices):
            for j in range(self.vertices):
                print(self[i, j], end=" ")
            print()


    def transposta(self) -> 'Grafo':
        g_t = Grafo(self.file)
        for i in range(self.vertices):
            for j in range(self.vertices):
                g_t[i, j] = self[j, i]
                #print(g_t[i, j], end=" ")
            #print()
        return g_t


    def vizinhos_saintes(self, v: int) -> list:
        return [i for i in range(self.vertices) if self[v, i] != np.inf and i != v]
    
    def vizinhos_saintes_transposta(self, v: int) -> list:
        g_t = self.transposta()
        return [i for i in range(self.vertices) if g_t[v, i] != np.inf and i != v]

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
        visited = np.zeros(self.vertices, dtype=bool)
        begin_time = np.ones(self.vertices, dtype=int)*np.inf
        predecessors = np.full(self.vertices, -1)
        search_time = np.ones(self.vertices, dtype=int)*np.inf
        visited, predecessors, search_time, begin_time, time = self.DFS_2(True, visited, predecessors, search_time, begin_time,)
        print("visited: ", visited)
        print("predecessors: ", predecessors)
        print("search_time: ", search_time)
        print("begin_time: ", begin_time)
        print("time: ", time)
        visited = np.zeros(self.vertices, dtype=bool)
        begin_time = np.ones(self.vertices, dtype=int)*np.inf
        predecessors = np.full(self.vertices, -1)
        visited, predecessors, search_time, begin_time, time = self.DFS_2(False, visited, predecessors, search_time, begin_time,)
        return predecessors

    # def atividade_1(self) -> None:
    #     output = self.DFS()
    #     for i in range(self.vertices):
    #         print()


    def DFS_Visit(self, v: int, visited: list, predecessors: list, search_time:list, begin_time: list, time, first: bool) -> list:
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
        visited = np.zeros(self.vertices, dtype=bool)
        begin_time = np.ones(self.vertices, dtype=int)*np.inf
        search_time = np.ones(self.vertices, dtype=int)*np.inf
        tempo = 0
        lista = []
        # if self.detect_cycle():
        #     print("Tem ciclo!")
        #     return
        for u in range (self.vertices):
            if not visited[u]:
                lista = self.DFS_Visit_OT(u, visited, begin_time, search_time, tempo, lista)

        print("visited: ", visited)
        print("begin_time: ", begin_time)
        print("search_time: ", search_time)
        print("lista: ", lista)

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
        arestas = ordena(arestas)
        #print(arestas)
        for i in range(len(arestas)):
            u = arestas[i][0]
            v = arestas[i][1]
            if S[u] != S[v]:
                #print("S: ", S)
                A.append([u, v])
                S[u].append(S[u] + S[v])
                S[v] = S[u]
        print(A)
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
        



a = Grafo("grafo.txt")
#a.mostra_grafo()
#print(a.DFS_CFC())
#a.DFS_OT()
a.Kruskal()
