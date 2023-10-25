from grafo import *
import sys

def resposta_1(grafo):
    predecessors = grafo.DFS_CFC()
    S = [[i] for i in range(grafo.qtdVertices())]

    for v in range(grafo.qtdVertices()):
        if predecessors[v] != -1:
            a = [y for x in [S[v], S[predecessors[v]]] for y in x] 
            for i in a:
                S[i] = a
    output = []
    for row in S:
        if row not in output:
            output.append(row)

    for i in range(len(output)):
        print(','.join(map(grafo.rotulo, sorted(output[i]))))

def resposta_2(grafo):
    ot = grafo.DFS_OT()
    print('➔'.join(map(grafo.rotulo, ot)))

def resposta_3(grafo):
    alg = input("Prim ou Kruskal? (P/K) ")
    if alg in "Pp":
        resposta_3_prim(grafo)
    elif alg in "Kk":
        resposta_3_kruskal(grafo)
    else:
        raise Exception("Algoritmo inválido")

def resposta_3_prim(grafo):
    agm = grafo.prim()
    arestas = []
    custo = 0
    for v in range(grafo.qtdVertices()):
        if agm[v] != -1:
            rotulo_v = grafo.rotulo(v)
            rotulo_agm_v = grafo.rotulo(agm[v])
            if rotulo_v < rotulo_agm_v:
                arestas.append((rotulo_v, rotulo_agm_v))
            else:
                arestas.append((rotulo_agm_v, rotulo_v))
            custo += grafo.peso(v, agm[v])
    
    print(custo)
    print(', '.join(map(lambda x: f'{x[0]}-{x[1]}', sorted(arestas))))

def resposta_3_kruskal(grafo):
    agm = grafo.kruskal()
    arestas = []
    custo = 0
    for aresta in agm:
        v, w = aresta
        arestas.append((v, w))
        if grafo.peso(v, w) != np.inf:
            custo += grafo.peso(v, w)
        else:
            custo += grafo.peso(w, v)
    
    print(custo)
    print(', '.join(map(lambda x: f'{grafo.rotulo(x[0])}-{grafo.rotulo(x[1])}', sorted(arestas))))

entrada = "grafo.txt"
if len(sys.argv) > 1:
    entrada = sys.argv[1]

escolha = '0'
while escolha not in "123":
    escolha = input("Qual questão? ")

if escolha == '1':
    grafo = GrafoDirigido(entrada)
    resposta_1(grafo)

elif escolha == '2':
    grafo = GrafoDirigido(entrada)
    resposta_2(grafo)

elif escolha == '3':
    grafo = GrafoNaoDirigido(entrada)
    resposta_3(grafo)