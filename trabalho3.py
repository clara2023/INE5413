from grafo import *
import sys

# CONFIGURAÇÃO DAS ENTRADAS

# Nome do arquivo utilizado na entrada
entrada = "grafo.txt"
if len(sys.argv) > 1:
    entrada = sys.argv[1]

# Qual questão será executada ['1', '2', '3']
escolha = '0'

# Vértices para o fluxo máximo, se os valores forem negativos, serão pedidos na execução
s = -1
t = -1

# Vértices para a bipartição, se a lista for vazia, será pedida na execução
X = []
# Descomente a linha abaixo para bipartir o grafo meio a meio
# X = [i for i in range(grafo.qtdVertices() // 2)]


# RESPOSTAS, não alterar nada abaixo
def resposta_1(grafo, s, t):
    valor = grafo.edmonds_karp(s, t)
    print("Maximum flow: ", valor)

def resposta_2(grafo):
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

def resposta_3(grafo):
    valor_cor = grafo.lawler()
    print("Minimum coloring value: ", valor_cor)

# EXECUÇÃO
while escolha not in '123':
    escolha = input("Qual questão? ")

if escolha == '1':
    if s < 0: s = int(input("Vértice de origem: "))
    if t < 0: t = int(input("Vértice de destino: "))
    grafo = GrafoDirigido(entrada)
    resposta_1(grafo, s, t)

if escolha == '2':
    grafo = GrafoBipartido(entrada)
    if len(X) == 0:
        X = [int(i)-1 for i in input("Digite os vertices do conjunto X separados por espaço. Não comece do índice 0: ").split()]
    grafo.bipartir(X)
    resposta_2(grafo)

if escolha == '3':
    grafo = GrafoNaoDirigido(entrada)
    resposta_3(grafo)
