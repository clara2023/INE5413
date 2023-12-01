from grafo import *

def resposta_1(grafo, s, t):
    valor = grafo.edmonds_karp(s, t)
    print(valor)


grafo = GrafoDirigido("grafo.txt")
resposta_1(grafo, 0, 7)