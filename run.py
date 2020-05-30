from ich import *
from maxcutsdp import *
from junto import *


random.seed(42)
n = 150
p = 0.8
file = "G100"
'''
MAX = 10000
vec = []
edges = []
set = set()
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        if (j, i) in edges:
            continue
        if random.randint(1, MAX)/MAX < p:
            continue
        vec.append((i+1, j+1, random.randint(1, 100)))
        set.add(i)
        set.add(j)
with open(file, "w") as f:
    f.write('{} {}'.format(len(set), len(vec)))
    f.write('\n')
    for i in vec:
        f.write('{} {} {}'.format(*i))
        f.write('\n')
    f.close()'''

prueba="./G"
grafo=cargar_grafos(file)
ich=MaxCutIch(grafo)
par=ich.solve(grafo,5)
valor_par=get_partition_ich(grafo,par)
valor_final=get_cut_value(grafo, valor_par)
valor = 0
sdp=MaxCutSDP(grafo)
sdp.solve(grafo,5)
if valor < sdp.get_results('value'):
    valor = sdp.get_results('value')
print(valor)
bb=branch_and_bound_maxcuts(5)

valor=bb.solve(grafo)
print("ICH {}".format(valor_final))
print("Valor del corte branch and bound: {} \n".format(valor))
