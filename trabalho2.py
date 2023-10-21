import numpy as np
from funcoes_auxiliares import *
from grafo import *

def resposta_1(grafo):
    predecessors = grafo.DFS_CFC()
    S = []
    for i in range(a.vertices):
        S.append([i])
    cfc = grafo.DFS_CFC()
    componentes = np.unique(cfc)
    
    for componente in componentes:
        for i, vertice in enumerate(cfc):
            if vertice == componente: print(grafo.rotulos(i), end=',')
        print()
    # for v in range(a.vertices):
    #     if predecessors[v] != -1:
    #         print('S[v]: ', S[v])
    #         print('S[predecessors[v]]: ', S[predecessors[v]])
    #         a = S[v] + S[predecessors[v]]
    #         for i in range(len(S[predecessors[v]])):
    #             S[predecessors[v]][i] = a
    print(S)



a = GrafoDirigido("grafo.txt")
#a.mostra_grafo()
#print(a.DFS_CFC())
#a.DFS_OT()
#a.Kruskal()
a.resposta_1()
