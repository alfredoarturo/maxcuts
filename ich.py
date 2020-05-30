from maxcutsdp import *
import networkx as nx
import cvxpy as cp
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime

class MaxCutIch(AbstractMaxCut):
    #clase para hacer el corte con la heruristica ICH

    def __init__(self, graph, solver='scs', seed=datetime.now()):
        '''
        Inicialización del la clase para resolver con el algoritmo ICH
        graph -> el grafo que se quiere cortar
        solver -> el solver que se va a usar para resolver la SDP
        '''
        super().__init__(graph)
        random.seed(seed)
        solver = solver.upper()
        if solver not in cp.installed_solvers():
            raise KeyError("Solver '%s' is not installed." % solver)
        self.solver = getattr(cp, solver)


    def solve(self,grafo, k, tol=1.7):
        '''
        Función para resolver el maximo corte del grafo. Se usa una toleracia por defaul de 1.7, que se menciona
        en el paper que funciona bien.
        Input:
            grafo -> el grafo que se quiere cortar
            k -> maximo numero de particiones que se quieren crear
            tol -> tolerancia que se tiene para la heruristica
        Output:
            Las particiones del grafo
        '''
        self.graph = grafo
        #se resulve
        parts = self._solve_aux(self.graph)
        original = parts
        print("Hecho")
        it = 0
        while len(parts) > k:
            # maneja el caso en el que se tienen más particiones de las permitidas
            #plt.figure()
            #nx.draw(grafo, with_labels=True)  # networkx draw()
            #plt.draw()
            #plt.savefig("Cut" + str(it) + '.png')
            #se modifica el grafo y se vuelve a intentar resolver.
            grafo = self.cambiar_grafo(parts, self.graph)
            parts = self._solve_aux(grafo, k, tol, len(parts))
            new = []
            #se actualizan las particiones
            for part in parts:
                aux_set = set()
                for elem in part:
                    aux_set |= original[elem]
                new.append(aux_set)
            original = new
            it+= 1
        grafo = self.cambiar_grafo(parts, self.graph)
        #plt.figure()
        #nx.draw(grafo, with_labels=True)  # networkx draw()
        #plt.draw()
        #plt.savefig("Cut" + str(it) + '.png')
        return original

    def cambiar_grafo(self, part, grafo):
        '''
        Función para cambiar el grafo, se contraen los nodos en las mismas pariciones en un solo nodos
        Input:
            part -> Las particiones que se tienen actualmente
            grafo -> el grafo que se tiene particionado
        Output:
            El grafo modificado
        '''
        adj = nx.adjacency_matrix(grafo)
        n_nodes = len(part)
        nueva_adj = np.zeros((n_nodes, n_nodes))
        edges = grafo.edges()
        for u, v in edges:
            ind1 = self._search_part(part, u)
            ind2 = self._search_part(part, v)
            nueva_adj[ind1][ind2] += edges[u, v]['weight']

        n_aristas = len(np.nonzero(nueva_adj[0]))
        with open('./temp', 'w') as f:
            f.write('{} {}'.format(int(len(part)), int(n_aristas)))
            f.write('\n')
            for i in range(len(part)):
                for j in range(len(part)):
                    if nueva_adj[i][j] == 0:
                        continue
                    f.write('{} {} {}'.format(i+1, j+1, int(nueva_adj[i][j])))
                    f.write('\n')
            f.close()
        return cargar_grafos('./temp')

    def _solve_aux(self,grafo, k=2, tol=1.7,r=0):
        '''
        Se encuentran las particiones del grafo usando la heuristica indicada
        Input:
            grafo -> el grafo que se quiere particionar
            k -> numero maximo de particiones
            tol -> tolerancia
            r -> numero de particiones actuales
        '''
        #r = 0
        ## Resolvemos la SDP
        n_nodes = len(grafo)
        m = n_nodes
        tol = tol
        adjacent = -1 * nx.adjacency_matrix(grafo).toarray().copy()

        matrix = cp.Variable((n_nodes, n_nodes), PSD=True)
        for i in range(n_nodes):
            adjacent[i][0:(i+1)] = 0
        mult = cp.multiply(adjacent, ((1+(k-1)*matrix )/k).T )
        cut = 0
        for i in range(n_nodes):
            cut += mult[i][i]
        constraints=[]
        constraints+=[cp.diag(matrix)==1]
        for i in range(matrix.shape[0]):
            constraints+=[matrix[i][i]==1]
        problem = cp.Problem(cp.Minimize(cut), constraints)
        problem.solve(getattr(cp, self.solver))
        #matrix_1=matrix.value
        matrix = nearest_psd(matrix.value)
        matrix = matrix - min(np.min(matrix) + 1/(k-1), 0)
        print("Problema resuelto ...")
        T = []
        for i in range(n_nodes):
            for j in range(i, n_nodes):
                if i == j:
                    continue
                for h in range(j, n_nodes):
                    if h == i or h ==j:
                        continue
                    val = matrix[i][j]
                    val += matrix[i][h]
                    val += matrix[j][h]
                    T.append((val, i, j, h))
        T = sorted(T, reverse=True) ##checar reverse
        parts = [] #vector de sets
        #Aplicamos la heuristica
        for valor in T:
            if valor[0] > tol:
                ind1 = self._search_part(parts, valor[1])
                ind2 = self._search_part(parts, valor[2])
                ind3 = self._search_part(parts, valor[3])


                if ind1 == -1 and ind2 == -1 and ind3 == -1:
                    parts.append(set([valor[1], valor[2], valor[3]]))
                    continue

                if ind1 == -1:
                    parts.append(set([valor[1]]))
                    ind1 = len(parts)-1

                if ind2 == -1:
                    parts.append(set([valor[2]]))
                    ind2 = len(parts)-1

                if ind3 == -1:
                    parts.append(set([valor[3]]))
                    ind3 = len(parts)-1
                #paso 7
                parts = self._set_handle(parts, ind1, ind2, ind3)


        n = self.get_nodes_in(parts)
        if n < n_nodes:
            parts = self._handle_less(T, tol, parts)

        return parts

    def _handle_less(self, T, tol, parts):
        '''
        Función para tratar los casos en que no todos los nodos se han asignado a una partición
        Input:
            T -> el calculo de los t como se indica en el paper
            tol -> la toleracia a usar
            parts -> las particiones del grafo
        Output:
            Las particiones con todos los nodos asignados a una particion
        '''
        for valor in T:
            if valor[0] > tol:
                continue
            indices = []
            indices.append(self._search_part(parts, valor[1]))
            indices.append(self._search_part(parts, valor[2]))
            indices.append(self._search_part(parts, valor[3]))
            for i in range(1, 4):
                aux = self.get_rand_index(1, len(parts), indices)
                if indices[i-1] != -1:
                    continue
                if aux < 0 :
                    parts.append(set([valor[i]]))
                    indices.append(len(parts)-1)
                    continue
                parts[aux].add(valor[i])
        return parts



    def get_rand_index(self, n, max_val, restricted):
        '''
        Función para generar indices aleatorios de a donde se va a asignar un valor dado.
        Input:
            n -> numero de valores aleatorios
            max_val -> maximo valor
            restricted -> valores prohibidos
        Output:
            vec -> vector de valores aleatorios en el intervalo [-2, max_val-1] que no estan en restricted.
        '''
        vec = None
        if max_val < len(restricted):
            return -1
        while vec == None:
            aux = random.randint(-2, max_val-1)
            if aux not in restricted:
                vec = aux
        return vec

    def get_nodes_in(self, parts):
        n = 0
        for i in parts:
            n+= len(i)
        return n


    def _set_handle(self, parts, ind1, ind2, ind3):
        '''
        Funcion para unir diferentes particiones.
        Input:
            parts -> las particiones actuales
            ind1 -> indice de la primer particion a unir
            ind2 -> indice de la segunda particion a unir
            ind3 -> indice de la tercera particion a unir
        Output:
            regresa las parciciones uniendo las particiones en los indices indicados
        '''
        if ind1 != ind2:
            parts[min(ind1, ind2)] |= parts[max(ind1, ind2)]
            if ind3 == ind1 or ind3 == ind2:
                parts.pop(max(ind1, ind2))
                return parts
            parts[min(ind1, ind2)] |= parts[ind3]
            parts[min(ind1, ind2, ind3)] = parts[min(ind1, ind2)]
            aux = sorted([ind1, ind2, ind3], reverse=True) #checar el orden de mayor a menor
            parts.pop(aux[2])
            parts.pop(aux[1])
            return parts
        if ind1 == ind2:
            if ind1 == ind3:
                return parts
            parts[min(ind1, ind3)] |= parts[max(ind1, ind3)]
            parts.pop(max(ind1, ind3))
            return parts



    def _search_part(self, parts, val):
        '''
        Regresa el indice de la particion en donde se encuentra un valor dado
        Input:
            parts -> las particiones que se tienen actualmente
            val -> el valor a buscar
        Output:
            El indice de la particion a la que pertenece val o -1 si no esta asignado
        '''
        for i in range(len(parts)):
            if val in parts[i]:
                return i
        return -1
