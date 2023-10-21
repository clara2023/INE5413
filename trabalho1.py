import numpy as np
from grafo import *

def resposta_5(grafo):
    distancias, origem = grafo.floyd_warshall()
    for i in range(grafo.vertices):
        string = f"{int(i+1)}:"
        for j in range(grafo.vertices):
            if distancias[i, j] == np.inf:
                string += "+∞,"
            elif distancias[i, j] == -np.inf:
                string += "-∞,"
            else:
                string += f"{int(distancias[i, j])},"
        print(string[:-1])

def resposta_4(grafo, s: int, alg: str) -> None:
    if alg[0] in 'Bb': distancias, origem = grafo.bellman_ford(s)
    elif alg[0] in 'Dd': distancias, origem = grafo.dijkstra(s)
    else: raise Exception("Algoritmo inválido")
    
    for vertice in range(grafo.vertices):
        string = f"{grafo.rotulo(vertice)}: "
        caminho = []
        if distancias[vertice] == np.inf:
            string += "Não há caminho; "
        elif distancias[vertice] == -np.inf:
            string += "Há ciclo negativo; "
        else:
            atual = vertice
            while True:
                caminho.insert(0, atual)
                if atual == s:
                    break
                atual = int(origem[atual])
            if caminho == []:
                string += ';'
            else:
                for x in caminho:
                    string += f"{grafo.rotulo(x)},"
                string = string[:-1] + '; '
        string += f"d={distancias[vertice]}"
        print(string)

def resposta_3(grafo):
    ciclo = grafo.ciclo_euleriano()
    if ciclo is None:
        print("0")
    else:
        print("1")
        string = ','.join([str(x+1) for x in ciclo])
        print(string if string else "O grafo não possui arestas")

def resposta_2(grafo, s: int) -> (np.ndarray, np.ndarray):
    arvore, distancias = grafo.busca_em_largura(s)
    for i in range(int(max(distancias)+1)):
        string = f"{i}:"
        for j in range(grafo.vertices):
            if distancias[int(j)] == i:
                string += f" {j}"
        print(string)

def resposta_1(grafo):
    grafo.mostrar_grafo()
    print(f"Vertices: {grafo.qtdVertices()}")
    print(f"Arestas: {grafo.qtdArestas()}")  
    for i in range(grafo.vertices):
        print(f"Grau({grafo.rotulo(i)}): {grafo.grau(i)}")
    for i in range(grafo.vertices):
        print(f"Vizinhos({grafo.rotulo(i)}): {grafo.vizinhos(i)}")

grafo = GrafoNaoDirigido("grafo.txt")
resposta_1(grafo)
