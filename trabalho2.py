from grafo import *

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
        print(','.join(map(grafo.rotulo, output[i])))

def resposta_2(grafo):
    ot = grafo.DFS_OT()
    print('➔'.join(map(grafo.rotulo, ot)))

def resposta_3_prim(grafo):
    agm = grafo.Prim()
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
    agm = grafo.Kruskal()
    arestas = []
    custo = 0
    for aresta in agm:
        v, w = aresta
        arestas.append((v, w))
        custo += grafo.peso(v, w)
    
    print(custo)
    print(', '.join(map(lambda x: f'{x[0]}-{x[1]}', sorted(arestas))))
   
grafo = GrafoNaoDirigido("grafo.txt")
resposta_3_kruskal(grafo)
