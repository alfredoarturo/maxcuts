from maxcutsdp import *
from ich import *
import random
import numpy as np


lb_best = -float('inf')


class branch_and_bound_maxcuts(object):
    '''
        Clase con la implementación del algoritmo de Branch and Bound.
        Esta clase crea el arbol de soluciones y lo explora con DFS.
        Para cambiar el tipo de exploración del arbolde soluciones, se debe
        modificar la funcion _solve_aux
    '''

    def __init__(self, k, upper_op=float('inf'), lower_op=-float('inf'), problema=[], parent=None, n_nodes=0):
        '''
        Input:
            k -> maximo numero de particiones que se quieren encontrar
            upper_op -> la cota superior encontrada
            lower_op -> la mejor cota superior
            problema -> el subproblema actual
            parent -> el padre en el arbol de soluciones
            n_nodes -> numero de nodos hijos
        '''
        self.k = k
        self.parts = []
        self.upper_op = upper_op
        self.lower_op = lower_op
        if len(problema) == 0:
            self.problema = [set()]*self.k
        self.problema = problema
        self.grafo = None
        self.children = []
        self.max_hijo = -1
        self.parent = parent
        self.n_nodes = n_nodes

    def insert(self, sub_problem):
        '''
        Funcion para insertar un nuevo nodo
        Input:
            sub_problem -> Variable que contine  al subproblema que se quiere explorar y que se va a guardar en este nodo
        '''
        if self.max_hijo >= self.k:
            print("Error al inserta, más de los hijos permitidos")
            return -1
        self.children.append(branch_and_bound_maxcuts(self.k, self.upper_op, self.lower_op, sub_problem, self, self.n_nodes))
        self.max_hijo += 1


    def select_node(self, grafo):
        '''
        Funcion para seleccionar un nodo de manera aleatoria
        Input:
            grafo -> el grafo actual
        Output:
            un nodo elgido de manera aleatoria
        '''
        n_nodes = nx.adjacency_matrix(self.grafo).toarray().shape[0]
        aux = random.randint(0, n_nodes-1)
        return aux

    def select_node_r4(self,grafo):
        '''
        Funcionque selecciona un nodo con la regla R4
        Input:
            grafo -> el grafo actual
        Output:
            un nodo elegirdo con la regla R4
        '''
        adj=nx.adjacency_matrix(grafo).toarray()
        return adj.sum(axis=0).argmax()


    def get_index(self, vertex):
        '''
        Se regresa el indice en donde se encuentra vertex en los subproblemas
        Input:
            vertex -> el vertice a buscar
        Output:
            -1 si vertex no está en el subproblema, sino el indice de la particion en la que se fijo
        '''
        for p in range(len(self.problema)):
            if vertex in self.problema[p]:
                return p
        return -1

    def mod_graph(self):
        '''
        Función para modificar el grafo que se está usando, en funcion a al subproblema que se tienen
        Lo que se hace es colapsar los nodos que se fijaron en una misma particion en un mismo nodo
        '''
        not_in = []
        for i in self.problema:
            for v in i:
                not_in.append(v)
        adj = nx.adjacency_matrix(self.grafo).toarray()
        print(adj.sum())
        n_nodes =adj.shape[0]
        new_n_nodes = n_nodes - len(not_in) + self.k + 1
        new_adj = np.zeros((new_n_nodes, new_n_nodes))
        print(new_n_nodes, n_nodes, not_in, self.k, self.problema)
        it = 0
        order = []
        for i in range(n_nodes):
            if i in not_in:
                continue
            order.append(i)
            it += 1
        edges = self.grafo.edges()
        for u, v in edges:
            if u in order:
                indu = order.index(u)
            else:
                indu = it + self.get_index(u)
            if v in order:
                indv = order.index(v)
            else:
                indv = it + self.get_index(v)
            if u is not v:
                new_adj[indu][indv] += edges[u, v]['weight']
            new_adj[indv][indu] += edges[v, u]['weight']

        print('--------------------', new_adj.sum())
        #print(new_adj)
        G=nx.from_numpy_matrix(new_adj)
        print(nx.adjacency_matrix(G).toarray().sum())
        '''n_aristas = len(np.nonzero(new_adj[0]))
        with open('./temp', 'w') as f:
            f.write('{} {}'.format(int(new_n_nodes), int(n_aristas)))
            f.write('\n')
            for i in range(new_n_nodes):
                for j in range(i, new_n_nodes):
                    if new_adj[i][j] == 0:
                        continue
                    f.write('{} {} {}'.format(i+1, j+1, int(new_adj[i][j])))
                    f.write('\n')
            f.close()'''
        return G

    def get_cut_ich(self, parts,n_nodes):
        '''
        Se pasa ich de un vector de sets a un vector con indices de las particiones
        a las que pertenecen los nodos
        Input:
            parts -> particiones actuales
            n_nodes -> numero de nodos
        Output:
            vector indicando la particion a la que pertenece cada nodo
        '''
        vec=np.zeros(n_nodes).astype(int)
        contador=0
        for v in parts:
            for i in v:
                vec[i]=contador+1
            contador+=1
        return vec

    def solve(self, grafo):
        '''
        Esta funcion es llamada para comenzar a crear el arbol de soluciones.
        Input:
            grafo -> el grafo que se quiere cortar
        Output:
            el mejor porcentaje cortado
        '''
        global lb_best
        #la primera
        #print('Entrando solver general...')
        ICH=MaxCutIch(grafo, seed=42).solve(grafo,self.k)
        #print('Generando corte...')
        corte=get_partition_ich(grafo,ICH)
        self.lower_op=get_cut_value(grafo,corte)
        lb_best = self.lower_op
        #print(self.lower_op)
        self.upper_op=float('inf')
        #print('Calculo de cotas...')
        self.insert([])
        self.children[0]._solve_aux(grafo)
        #print('Saliendo solver general...')
        return lb_best

    def _solve_aux(self, grafo): #sub_problema (grafo,cota)
        '''
        Funcion que busca la mejor solucion explorando el arbol de soluciones con un DFS
        La funcion toma el caso actual, encuenta la cota superor e inferior y
        hace lo indicado en el algorimto de Branch and Bound.
        En cada paso se va actualizando lb_best que es la Variable que contiene la mejor cota inferior
        encontrada hasta el momento.
        Input:
            grafo -> el grafo original que se quiere particionar
        '''
        global lb_best
        print('Entrando sub_problema...')
        self.lower_op = lb_best
        #print(self.upper_op, self.lower_op)
        self.grafo = grafo
        #se obtiene el grafo modificado con forme al subproblema actual
        grafo = self.mod_graph()
        adj = nx.adjacency_matrix(grafo).toarray()
        print('#####################', adj.sum())
        #encontramos las cotas
        ICH=MaxCutIch(grafo).solve(grafo, self.k)
        SDP=MaxCutSDP(grafo)
        SDP.solve(grafo,self.k)
        self.upper_op=SDP.get_results('value')
        corte=self.get_cut_ich(ICH,len(grafo))
        self.lower_op=get_cut_value(grafo,corte)
        #seleccionamos un nodo
        v = self.select_node_r4(grafo)
        if self.parent != None:
            if self.upper_op <= lb_best:
                print('Saliendo sub_problema...')
                return
        if self.get_index(v) != -1:
            print('Saliendo sub_problema...')
            return
        problemas = self.problema.copy()
        for i in range(self.k): #crear los subproblemas
            try:
               problemas[i].add(v)
            except:
               problemas.append(set([v]))
            #se cra un hijo y se explora, esto se debe cambiar si se quiere usar otra fomar de explorar el arbol de soluciones
            self.insert(problemas)
            self.children[i]._solve_aux(self.grafo)
            if lb_best < self.children[i].lower_op:
                self.lower_op = self.children[i].lower_op
                lb_best = self.children[i].lower_op
                print('Actualizando lower_op a {}'.format(lb_best))
            #self.lower_op = self.lower_op if self.lower_op >= self.children[i].lower_op else self.children[i].lower_op
            #print("Cut value {}".format(self.lower_op))
            problemas[i].remove(v)

        print('Saliendo sub_problema...')
        #resolver / explorar el arbol dfs, bfs
