from grafo import *

def resposta_1(grafo, s, t):
    valor = grafo.edmonds_karp(s, t)
    print(valor)

def resposta_2():
    grafo = GrafoBipartido("grafo.txt")
    X = [0, 1, 2, 3]
    #X = print(input("Lista de vertices no conjunto X: "))
    grafo.bipartir(X)
    m, mate = grafo.hopcroft_karp()
    print("m: ", m)
    print("Mate: ", mate)

def resposta_3():
    grafo = GrafoNaoDirigido("grafo.txt")
    a = grafo.lawler()
    print(a)

grafo = GrafoDirigido("grafo.txt")
#resposta_1(grafo, 0, 5) # 0, n-1 vertices
#resposta_2()
resposta_3()