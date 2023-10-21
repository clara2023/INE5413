def ordena(lista):
    tamanho_da_lista = len(lista)
    lista_temporaria = [0] * tamanho_da_lista
    merge_sort(lista, lista_temporaria, 0, tamanho_da_lista - 1)
    return lista

def merge_sort(lista, lista_temporaria, inicio, fim):
    if inicio < fim:
        meio = (inicio + fim) // 2
        merge_sort(lista, lista_temporaria, inicio, meio)

        merge_sort(lista, lista_temporaria, meio + 1, fim)

        merge(lista, lista_temporaria, inicio, meio + 1, fim)

def merge(lista, lista_temporaria, inicio, meio, fim):
    fim_primeira_parte = meio - 1
    indice_temporario = inicio
    tamanho_da_lista = fim - inicio + 1

    while inicio <= fim_primeira_parte and meio <= fim:
        if lista[inicio][2] <= lista[meio][2]:
            lista_temporaria[indice_temporario] = lista[inicio]
            inicio += 1
        else:
            lista_temporaria[indice_temporario] = lista[meio]
            meio += 1
        indice_temporario += 1

    while inicio <= fim_primeira_parte:
        lista_temporaria[indice_temporario] = lista[inicio]
        indice_temporario += 1
        inicio += 1

    while meio <= fim:
        lista_temporaria[indice_temporario] = lista[meio]
        indice_temporario += 1
        meio += 1

    for _ in range(tamanho_da_lista):
        lista[fim] = lista_temporaria[fim]
        fim -= 1
            
