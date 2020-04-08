# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
from scipy.special import comb
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """
    if degree == 0:
        raise ValueError("degree of polynomials should be >= 1")


    if(cell.dim == 1):
        nodes = np.array([i/degree for i in range(degree+1)])
        nodes.shape = [degree+1, 1]
        
    elif(cell.dim == 2):
        nodes = np.array([(j/degree, i/degree) for i in range(degree+1) for j in range(degree+1-i)])
    
    else:
        raise ValueError("not implemented dimension")

    return nodes            



def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    d = cell.dim
    n = degree 
    m = points.shape[0] # number of rows
    k = comb(n+d, n, exact=True) # number of columns
    if grad == False:
        V = np.zeros((m, k)) # Vandermond Matrix
    if grad == True:
        V = np.zeros((m, k, d)) 

    if(d == 1):
        for ni in range(n+1):
                if grad == False:
                    V[:,ni] = points[:,0]**ni
                if grad == True: 
                    #V[:, ni, 0] =  (ni*points[:,0]**(ni-1) if ni-1 >= 0 else 0.0)
                    V[:, ni, 0] =  np.nan_to_num(ni*points[:,0]**(ni-1)) 

    # V = np.array([p[0]**j for p in points for j in range(degree+1)])
    # V.shape = [m,degree+1]                

    elif(d == 2):
        ki = 0 
        for ni in range(n+1): 
            for j in range(ni+1):
                if grad == False :
                    V[:,ki] = points[:,0]**(ni-j)*points[:,1]**j
                if grad == True : 
                    V[:,ki, 0] = np.nan_to_num((ni-j)*points[:,0]**(ni-j-1)*points[:,1]**j)
                    V[:,ki, 1] = np.nan_to_num(points[:,0]**(ni-j)*j*points[:,1]**(j-1))
                ki = ki+1 

    else:
        raise ValueError("not implemented dimension")               

    return V


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0]) for d in range(cell.dim+1)])
            
        # sets self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.
        V = vandermonde_matrix(self.cell, self.degree, self.nodes, grad=False)
        self.basis_coefs = np.linalg.inv(V)

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        V = vandermonde_matrix(self.cell, self.degree, points, grad)
        
        C = self.basis_coefs
        if grad == False:
            T = np.dot(V, C)
        
        if grad == True:
            
            # d = self.cell.dim
            # T = np.zeros(V.shape)
            # T[:,:,0] = np.dot(V[:,:,0], C)
            # if(d == 2):
            #     T[:,:,1] = np.dot(V[:,:,1], C)     

            T = np.einsum("ijk,jl->ilk", V, C)                
        
        return T

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """
        return np.array([fn(Xi) for Xi in lagrange_points(self.cell, self.degree)])

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        # raise NotImplementedError
        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        nodes = lagrange_points(cell, degree)

        if(cell.dim==1):
            entity_nodes = {0:{0:[], 1:[]},1:{0:[]}} 
            for n in range(len(nodes)): 
                if nodes[n] == 0: 
                    entity_nodes[0][0] += [n] 
                elif nodes[n] == 1: 
                    entity_nodes[0][1] += [n] 
                elif(degree > 1):
                    entity_nodes[1][0] += [n]


        if(cell.dim == 2):
            entity_nodes = {0:{0:[], 1:[], 2:[]}, 1:{0:[], 1:[], 2:[]}, 2:{0:[]}} 

            
            for j in range(3):
                for n in range(len(nodes)):
                    if cell.point_in_entity(nodes[n], (0,j)):
                        entity_nodes[0][j] += [n]

            flattned_list = [l for L in entity_nodes[0].values() for l in L]
            
            for j in range(3): 
                for n in range(len(nodes)): 
                    if cell.point_in_entity(nodes[n], (1,j)) and not n in flattned_list : 
                        entity_nodes[1][j] += [n]
                entity_nodes[1][j] = sorted(entity_nodes[1][j])    

            flattned_list += [l for  L in entity_nodes[1].values() for l in L]
            
            for n in range(len(nodes)): 
                if (not n in flattned_list):
                        entity_nodes[2][0] += [n]



        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes=entity_nodes)