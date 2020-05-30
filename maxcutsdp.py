
# ## Proyecto final de optimización 1

import networkx as nx
import numpy as np
from numpy import linalg as la
from datetime import datetime
from time import time

def cargar_grafos(path):
    graph = nx.Graph()
    with open(path) as file:
        n_nodes = int(file.readline().split(' ', 1)[0])
        graph.add_nodes_from(range(n_nodes))
        for row in file:
            start, end, weight = [int(e) for e in row.strip('\n').split()]
            graph.add_edge(start - 1, end - 1, weight=weight)
    '''graph.add_nodes_from([1, 2, 3, 4])
    graph.add_weighted_edges_from([(1, 2, 1), (2, 3, 4), (1, 4, 3), (3, 4, 2)])'''
    return graph


# ## Parte 2: Creación de la clase abstracta para la solución de máximo corte

#creación de la clase para resolver el problema de la SDP
#Una clase abstracta importada de la librería abc
import cvxpy as cp
from abc import ABCMeta, abstractmethod

class AbstractMaxCut(metaclass=ABCMeta):
    def __init__(self,graph):
        self.graph=graph
        self._solution=None

    def get_results(self, item='cut', verbose=False): #Regresa los lazy evaluated max-cut alcanzados, regresa el corte o su valor en la matriz inicial resolviendo el programa SDP
        if self._results is None:
            self.solve(verbose)
        if item not in self._results:

            valid = ', '.join(["'%s'" % key for key in self._results.keys()])
            raise KeyError(
                "No se encuentra la opción que se solicitó" % valid
            )
        return self._results.get(item)
    @abstractmethod
    def solve(self, verbose=True):
        #Resolver el problema BM formulado del max cut usando RTR
        return NotImplemented


# In[50]:

def best_part(vectors, k):
    np.random.seed(int(time()))
    list_norm=list()
    for i in range(k):
        list_norm.append(np.random.normal(size=vectors.shape[1]))
        list_norm[i]/=la.norm(list_norm[i])
    cut=[]
    for i in range(vectors.shape[0]):
        cantidades=[la.norm(np.dot(vectors[i].T,list_norm[j])) for j in range(len(list_norm))]
        cut.append(cantidades.index(max(cantidades))+1)
        cantidades=[]
    return cut

def get_k_partition(vectors,k):
    vec = []
    for i in range(5):
        vec.append(best_part(vectors, k))
    return max(vec)


def get_cut_value(graph, partition):
    # Regresa el costo de la partición del grafo
    in_cut = sum(attr['weight'] for u, v, attr in graph.edges(data=True) if partition[u] != partition[v])
    total = .5 * nx.adjacency_matrix(graph).sum()
    return in_cut / total

def get_partition_ich(graph,parts):
    cut=np.zeros(len(graph))
    contador=1
    for par in parts:
        for p in par:
            cut[p]=contador
        contador+=1
    return cut

class MaxCutSDP(AbstractMaxCut):
    "solución del problema de los k cortes, para el caso k=3"
    def __init__(self, graph, solver='scs'):
        super().__init__(graph)
        solver = solver.upper()
        if solver not in cp.installed_solvers():
            raise KeyError("Solver '%s' no instalado." % solver)
        self.solver = getattr(cp, solver)

    def solve(self, graph,k,verbose=True):
        matrix = self._solve_sdp(graph,k)
        matrix = nearest_psd(matrix)
        #print("resuelto")
        # Tenemos el corte definido por la matriz
        vectors = np.linalg.cholesky(matrix)
        cut = get_k_partition(vectors,k)
        #print(cut)
        # Tenemos el valor del corte
        value = get_cut_value(graph, cut)
        self._results = {'matrix': matrix, 'cut': cut, 'value': value,'problema':self._problem}
        # Optionally be verbose about the results.
        if verbose:
            print(
                #"Problema SDP-relaxed max-cut resuelto.\n"
                #"Peso total de la solución %f." % value
            )
        #    print(self._results['cut'])
        #    print(self._results['matrix'])


    def _solve_sdp(self,graph,k):
        """Resuelve el problema SDP del maximo corte.
        regresa la matriz que maximiza <C, 1 - X>
        """
        # Propiedades del grafo a cortar
        n_nodes = len(graph)
        adjacent = nx.adjacency_matrix(graph).toarray() #Convertir la matriz de adyacencia a los
        matrix = cp.Variable((n_nodes, n_nodes), PSD=True)
        for i in range(n_nodes):
            adjacent[i][0:(i+1)] = 0
        mult = cp.multiply(adjacent, (1 - matrix).T)
        cut = 0
        for i in range(n_nodes):
            cut += mult[i][i]
        cut *= (k-1)/k
        constraints=[]
        constraints+=[cp.diag(matrix)==1]
        problem = cp.Problem(cp.Maximize(cut), constraints)
        problem.solve(solver = cp.SCS , use_indirect=True)
        #matrix = nearest_psd(matrix.value)
        matrix=nearest_psd(matrix.value)
       	matrix = matrix - min(np.min(matrix) + 1/(k-1), 0)
        self._problem=problem
        return matrix


def nearest_psd(matrix):
	#la matriz regresa la matriz positiva definida más cercana a la matriz original, se checa que sea semipositiva definida si no se crea una matriz semi positiva definida
    if is_psd(matrix):
        return matrix
    spacing = np.spacing(np.linalg.norm(matrix))
    identity = np.identity(len(matrix))
    k = 1
    while not is_psd(matrix):
        min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
        matrix += identity * (- min_eig * (k ** 2) + spacing)
        k += 1
    return matrix


def is_psd(matrix):
    #Checamos si una matriz es semi definida positiva con la factorización de Cholesky
    try:
        _ = np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
         return False
