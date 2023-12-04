from grafo import *
import sys

def resposta_1(grafo, s, t):
    valor = grafo.edmonds_karp(s, t)
    print("Maximum flow: ", valor)

def resposta_2():
    grafo = GrafoBipartido("grafo.txt")
    X = []
    escolha = input("Deseja selecionar os vertices do conjunto X? (s/n): ").upper()
    if escolha == "N":
        n = grafo.qtdVertices() // 2
        X = [i for i in range(n)]
    else:
        X = [int(i)-1 for i in input("Digite os vertices do conjunto X separados por espaco. Não comece do índice 0: ").split()]
    grafo.bipartir(X)
    m, mate = grafo.hopcroft_karp()
    print("m: ", m)
    printed = {}
    for i in range(len(mate)):
        a = grafo.rotulo(i)
        b = grafo.rotulo(mate[i])
        printed[b] = a
        if a not in printed:
            print("Mate:", a, ",", b)
    # for i in range(len(mate)):
    #     print("Mate:", grafo.rotulo(i), ",", grafo.rotulo(mate[i])) 
def resposta_3():
    grafo = GrafoNaoDirigido("grafo.txt")
    valor_cor = grafo.lawler()
    print(valor_cor)


entrada = "grafo.txt"
escolha = '0'

if len(sys.argv) > 1:
    entrada = sys.argv[1]

while escolha not in '123':
    escolha = input("Qual questão? ")

if escolha == '1':
    s = int(input("Vértice de origem: "))
    t = int(input("Vértice de destino: "))
    
if escolha == '2':
    pass

if escolha == '3':
    pass
>>>>>>> Stashed changes
