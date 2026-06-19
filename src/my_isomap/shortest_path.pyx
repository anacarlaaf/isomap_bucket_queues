"""
Routines for performing shortest-path graph searches

The main interface is in the function :func:`shortest_path`.  This
calls cython routines that compute the shortest path using
the Floyd-Warshall algorithm, Dijkstra's algorithm with priority queue,
the Bellman-Ford algorithm, or Johnson's Algorithm.

"""

# Author: Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# License: BSD, (C) 2011
import warnings

import numpy as np
cimport numpy as np

from scipy.sparse import csr_array, issparse
from scipy.sparse.csgraph._validation import validate_graph
from scipy.sparse._sputils import (convert_pydata_sparse_to_scipy,
                                   safely_cast_index_arrays)

cimport cython

from libc.math cimport INFINITY
from queue cimport bucket_queue
from libcpp.pair cimport pair
from libcpp.vector cimport vector

np.import_array()

# Definições que vinham de parameters.pxi
DTYPE  = np.float64
ITYPE  = np.int32

ctypedef double    DTYPE_t
ctypedef int       ITYPE_t

DEF DTYPE_EPS = 1E-15
DEF NULL_IDX  = -9999


class NegativeCycleError(Exception):
    """
    Negative cycle in graph.

    Parameters
    ----------
    message : str
        Error message.
    """
    def __init__(self, message=''):
        Exception.__init__(self, message)


def shortest_path(csgraph, method='auto',
                  directed=True,
                  return_predecessors=False,
                  unweighted=False,
                  overwrite=False,
                  indices=None):
    """
    shortest_path(csgraph, method='auto', directed=True, return_predecessors=False,
                  unweighted=False, overwrite=False, indices=None)

    Perform a shortest-path graph search on a positive directed or
    undirected graph.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    csgraph : array_like, or sparse array or matrix, 2 dimensions
        The N x N array of distances representing the input graph.
    method : str ['auto'|'FW'|'D'], optional
        Algorithm to use for shortest paths.  Options are:

        'auto' -- (default) select the best among 'FW', 'D', 'BF', or 'J'
                  based on the input data.
        'FW'   -- Floyd-Warshall algorithm.
                  Computational cost is approximately ``O[N^3]``.
                  The input csgraph will be converted to a dense representation.
        'D'    -- Dijkstra's algorithm with priority queue.
                  Computational cost is approximately ``O[I * (E + N) * log(N)]``,
                  where ``E`` is the number of edges in the graph,
                  and ``I = len(indices)`` if ``indices`` is passed. Otherwise,
                  ``I = N``.
                  The input csgraph will be converted to a csr representation.
        'BF'   -- Bellman-Ford algorithm.
                  This algorithm can be used when weights are negative.
                  If a negative cycle is encountered, an error will be raised.
                  Computational cost is approximately ``O[N(N^2 k)]``, where
                  ``k`` is the average number of connected edges per node.
                  The input csgraph will be converted to a csr representation.
        'J'    -- Johnson's algorithm.
                  Like the Bellman-Ford algorithm, Johnson's algorithm is
                  designed for use when the weights are negative. It combines
                  the Bellman-Ford algorithm with Dijkstra's algorithm for
                  faster computation.

    directed : bool, optional
        If True (default), then find the shortest path on a directed graph:
        only move from point i to point j along paths csgraph[i, j].
        If False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i]
    return_predecessors : bool, optional
        If True, return the size (N, N) predecessor matrix.
    unweighted : bool, optional
        If True, then find unweighted distances.  That is, rather than finding
        the path between each point such that the sum of weights is minimized,
        find the path such that the number of edges is minimized.
    overwrite : bool, optional
        If True, overwrite csgraph with the result.  This applies only if
        method == 'FW' and csgraph is a dense, c-ordered array with
        dtype=float64.
    indices : array_like or int, optional
        If specified, only compute the paths from the points at the given
        indices. Incompatible with method == 'FW'.

    Returns
    -------
    dist_matrix : ndarray
        The N x N matrix of distances between graph nodes. dist_matrix[i,j]
        gives the shortest distance from point i to point j along the graph.
    predecessors : ndarray, shape (n_indices, n_nodes,)
        Returned only if return_predecessors == True.
        If `indices` is None then ``n_indices = n_nodes`` and the shape of
        the matrix becomes ``(n_nodes, n_nodes)``.
        The matrix of predecessors, which can be used to reconstruct
        the shortest paths.  Row i of the predecessor matrix contains
        information on the shortest paths from point i: each entry
        predecessors[i, j] gives the index of the previous node in the
        path from point i to point j.  If no path exists between point
        i and j, then predecessors[i, j] = -9999

    Raises
    ------
    NegativeCycleError:
        if there are negative cycles in the graph

    See Also
    --------
    :ref:`word-ladders-example` : An illustratation of the ``shortest_path`` API with a meaninful example.
                                  It also reconstructs the shortest path by using predecessors matrix returned
                                  by this function.

    Notes
    -----
    As currently implemented, Dijkstra's algorithm and Johnson's algorithm
    do not work for graphs with direction-dependent distances when
    directed == False.  i.e., if csgraph[i,j] and csgraph[j,i] are non-equal
    edges, method='D' may yield an incorrect result.

    If multiple valid solutions are possible, output may vary with SciPy and
    Python version.

    Examples
    --------
    >>> from scipy.sparse import csr_array
    >>> from scipy.sparse.csgraph import shortest_path

    >>> graph = [
    ... [0, 0, 7, 0],
    ... [0, 0, 8, 5],
    ... [7, 8, 0, 0],
    ... [0, 5, 0, 0]
    ... ]
    >>> graph = csr_array(graph)
    >>> print(graph)
    <Compressed Sparse Row sparse array of dtype 'int64'
        with 6 stored elements and shape (4, 4)>
        Coords	Values
        (0, 2)	7
        (1, 2)	8
        (1, 3)	5
        (2, 0)	7
        (2, 1)	8
        (3, 1)	5

    >>> sources = [0, 2]
    >>> dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, indices=sources, return_predecessors=True)
    >>> dist_matrix
    array([[ 0., 15.,  7., 20.],
           [ 7.,  8.,  0., 13.]])
    >>> predecessors
    array([[-9999,     2,     0,     1],
           [    2,     2, -9999,     1]], dtype=int32)

    Reconstructing shortest paths from sources to all the nodes of the graph.

    >>> shortest_paths = {}
    >>> for idx in range(len(sources)):
    ...     for node in range(4):
    ...         curr_node = node # start from the destination node
    ...         path = []
    ...         while curr_node != -9999: # no previous node available, exit the loop
    ...             path = [curr_node] + path # prefix the previous node obtained from the last iteration
    ...             curr_node = int(predecessors[idx][curr_node]) # set current node to previous node
    ...         shortest_paths[(sources[idx], node)] = path
    ...

    Computing the length of the shortest path from node 0 to node 3
    of the graph. It can be observed that computed length and the
    ``dist_matrix`` value are exactly same.

    >>> shortest_paths[(0, 3)]
    [0, 2, 1, 3]
    >>> path03 = shortest_paths[(0, 3)]
    >>> sum([graph[path03[0], path03[1]], graph[path03[1], path03[2]], graph[path03[2], path03[3]]])
    np.int64(20)
    >>> dist_matrix[0][3]
    np.float64(20.0)

    Another example of computing shortest path length from node 2 to node 3.
    Here, ``dist_matrix[1][3]`` is used to get the length of the path returned by
    ``shortest_path``. This is because node 2 is the second source, so the
    lengths of the path from it to other nodes in the graph will be at index 1
    in ``dist_matrix``.

    >>> shortest_paths[(2, 3)]
    [2, 1, 3]
    >>> path23 = shortest_paths[(2, 3)]
    >>> sum([graph[path23[0], path23[1]], graph[path23[1], path23[2]]])
    np.int64(13)
    >>> dist_matrix[1][3]
    np.float64(13.0)

    """
    csgraph = convert_pydata_sparse_to_scipy(csgraph, accept_fv=[0, np.inf, np.nan])

    # validate here to catch errors early but don't store the result;
    # we'll validate again later
    validate_graph(csgraph, directed, DTYPE,
                   copy_if_dense=(not overwrite),
                   copy_if_sparse=(not overwrite))

    cdef bint is_sparse
    cdef ssize_t N      # XXX cdef ssize_t Nk fails in Python 3 (?)

    if method == 'auto':
        # guess fastest method based on number of nodes and edges
        N = csgraph.shape[0]
        csgraph = convert_pydata_sparse_to_scipy(csgraph)
        is_sparse = issparse(csgraph)
        if is_sparse:
            Nk = csgraph.nnz
            if csgraph.format in ('csr', 'csc', 'coo'):
                edges = csgraph.data
            else:
                edges = csgraph.tocoo().data
        elif np.ma.isMaskedArray(csgraph):
            Nk = csgraph.count()
            edges = csgraph.compressed()
        else:
            edges = csgraph[np.isfinite(csgraph)]
            edges = edges[edges != 0]
            Nk = edges.size

        if indices is not None or Nk < N * N / 4:
                method = 'D'
        else:
            method = 'FW'

    if method == 'FW':
        if indices is not None:
            raise ValueError("Cannot specify indices with method == 'FW'.")
        return floyd_warshall(csgraph, directed,
                              return_predecessors=return_predecessors,
                              unweighted=unweighted,
                              overwrite=overwrite)

    elif method == 'D':
        return dijkstra(csgraph, directed,
                        return_predecessors=return_predecessors,
                        unweighted=unweighted, indices=indices)

    else:
        raise ValueError("unrecognized method '%s'" % method)


def floyd_warshall(csgraph, directed=True,
                   return_predecessors=False,
                   unweighted=False,
                   overwrite=False):
    """
    floyd_warshall(csgraph, directed=True, return_predecessors=False,
                   unweighted=False, overwrite=False)

    Compute the shortest path lengths using the Floyd-Warshall algorithm.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    csgraph : array_like, or sparse array or matrix, 2 dimensions
        The N x N array of distances representing the input graph.
    directed : bool, optional
        If True (default), then find the shortest path on a directed graph:
        only move from point i to point j along paths csgraph[i, j].
        If False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i]
    return_predecessors : bool, optional
        If True, return the size (N, N) predecessor matrix.
    unweighted : bool, optional
        If True, then find unweighted distances.  That is, rather than finding
        the path between each point such that the sum of weights is minimized,
        find the path such that the number of edges is minimized.
    overwrite : bool, optional
        If True, overwrite csgraph with the result.  This applies only if
        csgraph is a dense, c-ordered array with dtype=float64.

    Returns
    -------
    dist_matrix : ndarray
        The N x N matrix of distances between graph nodes. dist_matrix[i,j]
        gives the shortest distance from point i to point j along the graph.

    predecessors : ndarray
        Returned only if return_predecessors == True.
        The N x N matrix of predecessors, which can be used to reconstruct
        the shortest paths.  Row i of the predecessor matrix contains
        information on the shortest paths from point i: each entry
        predecessors[i, j] gives the index of the previous node in the
        path from point i to point j.  If no path exists between point
        i and j, then predecessors[i, j] = -9999

    Raises
    ------
    NegativeCycleError:
        if there are negative cycles in the graph

    Notes
    -----
    If multiple valid solutions are possible, output may vary with SciPy and
    Python version.

    Examples
    --------
    >>> from scipy.sparse import csr_array
    >>> from scipy.sparse.csgraph import floyd_warshall

    >>> graph = [
    ... [0, 1, 2, 0],
    ... [0, 0, 0, 1],
    ... [2, 0, 0, 3],
    ... [0, 0, 0, 0]
    ... ]
    >>> graph = csr_array(graph)
    >>> print(graph)
    <Compressed Sparse Row sparse array of dtype 'int64'
        with 5 stored elements and shape (4, 4)>
        Coords	Values
        (0, 1)	1
        (0, 2)	2
        (1, 3)	1
        (2, 0)	2
        (2, 3)	3

    >>> dist_matrix, predecessors = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
    >>> dist_matrix
    array([[0., 1., 2., 2.],
           [1., 0., 3., 1.],
           [2., 3., 0., 3.],
           [2., 1., 3., 0.]])
    >>> predecessors
    array([[-9999,     0,     0,     1],
           [    1, -9999,     0,     1],
           [    2,     0, -9999,     2],
           [    1,     3,     3, -9999]], dtype=int32)

    """
    dist_matrix = validate_graph(csgraph, directed, DTYPE,
                                 csr_output=False,
                                 copy_if_dense=not overwrite)
    cdef long long INT64_MAX = np.iinfo(np.int64).max
    if not issparse(csgraph):
        # for dense array input, zero entries represent non-edge
        dist_matrix[dist_matrix == 0] = INT64_MAX

    if unweighted:
        dist_matrix[dist_matrix != INT64_MAX] = 1

    if return_predecessors:
        predecessor_matrix = np.empty(dist_matrix.shape,
                                      dtype=ITYPE, order='C')
    else:
        predecessor_matrix = np.empty((0, 0), dtype=ITYPE)

    _floyd_warshall(dist_matrix,
                    predecessor_matrix,
                    int(directed))

    if np.any(dist_matrix.diagonal() < 0):
        raise NegativeCycleError("Negative cycle in nodes %s"
                                 % np.where(dist_matrix.diagonal() < 0)[0])

    if return_predecessors:
        return dist_matrix, predecessor_matrix
    else:
        return dist_matrix


@cython.boundscheck(False)
cdef void _floyd_warshall(
               np.ndarray[DTYPE_t, ndim=2, mode='c'] dist_matrix,
               np.ndarray[ITYPE_t, ndim=2, mode='c'] predecessor_matrix,
               int directed=0) noexcept:
    # dist_matrix : in/out
    #    on input, the graph
    #    on output, the matrix of shortest paths
    # dist_matrix should be a [N,N] matrix, such that dist_matrix[i, j]
    # is the distance from point i to point j.  Zero-distances imply that
    # the points are not connected.
    cdef unsigned int N = dist_matrix.shape[0]
    assert dist_matrix.shape[1] == N

    cdef unsigned int i, j, k
    cdef DTYPE_t d_ijk
    cdef DTYPE_t INT64_MAX = 9223372036854775807

    # ----------------------------------------------------------------------
    #  Initialize distance matrix
    #   - set diagonal to zero
    #   - symmetrize matrix if non-directed graph is desired
    dist_matrix.flat[::N + 1] = 0
    if not directed:
        for i in range(N):
            for j in range(i + 1, N):
                if dist_matrix[j, i] <= dist_matrix[i, j]:
                    dist_matrix[i, j] = dist_matrix[j, i]
                else:
                    dist_matrix[j, i] = dist_matrix[i, j]

    #----------------------------------------------------------------------
    #  Initialize predecessor matrix
    #   - check matrix size
    #   - initialize diagonal and all non-edges to NULL
    #   - initialize all edges to the row index
    cdef int store_predecessors = False

    if predecessor_matrix.size > 0:
        store_predecessors = True
        assert predecessor_matrix.shape[0] == N
        assert predecessor_matrix.shape[1] == N
        predecessor_matrix.fill(NULL_IDX)
        i_edge = np.where(dist_matrix != INT64_MAX)
        predecessor_matrix[i_edge] = i_edge[0]
        predecessor_matrix.flat[::N + 1] = NULL_IDX

    # Now perform the Floyd-Warshall algorithm.
    # In each loop, this finds the shortest path from point i
    #  to point j using intermediate nodes 0 ... k
    if store_predecessors:
        for k in range(N):
            for i in range(N):
                if dist_matrix[i, k] == INT64_MAX:
                    continue
                for j in range(N):
                    d_ijk = dist_matrix[i, k] + dist_matrix[k, j]
                    if d_ijk < dist_matrix[i, j]:
                        dist_matrix[i, j] = d_ijk
                        predecessor_matrix[i, j] = predecessor_matrix[k, j]
    else:
        for k in range(N):
            for i in range(N):
                if dist_matrix[i, k] == INT64_MAX:
                    continue
                for j in range(N):
                    d_ijk = dist_matrix[i, k] + dist_matrix[k, j]
                    if d_ijk < dist_matrix[i, j]:
                        dist_matrix[i, j] = d_ijk


def dijkstra(csgraph, directed=True, indices=None,
             return_predecessors=False,
             unweighted=False, limit=9223372036854775807,
             bint min_only=False):
    """
    dijkstra(csgraph, directed=True, indices=None, return_predecessors=False,
             unweighted=False, limit=np.inf, min_only=False)

    Dijkstra algorithm using priority queue.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    csgraph : array_like, or sparse array or matrix, 2 dimensions
        The N x N array of non-negative distances representing the input graph.
    directed : bool, optional
        If True (default), then find the shortest path on a directed graph:
        only move from point i to point j along paths csgraph[i, j] and from
        point j to i along paths csgraph[j, i].
        If False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j or j to i along either
        csgraph[i, j] or csgraph[j, i].

        .. warning:: Refer the notes below while using with ``directed=False``.
    indices : array_like or int, optional
        if specified, only compute the paths from the points at the given
        indices.
    return_predecessors : bool, optional
        If True, return the size (N, N) predecessor matrix.
    unweighted : bool, optional
        If True, then find unweighted distances.  That is, rather than finding
        the path between each point such that the sum of weights is minimized,
        find the path such that the number of edges is minimized.
    limit : float, optional
        The maximum distance to calculate, must be >= 0. Using a smaller limit
        will decrease computation time by aborting calculations between pairs
        that are separated by a distance > limit. For such pairs, the distance
        will be equal to np.inf (i.e., not connected).

        .. versionadded:: 0.14.0
    min_only : bool, optional
        If False (default), for every node in the graph, find the shortest path
        from every node in indices.
        If True, for every node in the graph, find the shortest path from any
        of the nodes in indices (which can be substantially faster).

        .. versionadded:: 1.3.0

    Returns
    -------
    dist_matrix : ndarray, shape ([n_indices, ]n_nodes,)
        The matrix of distances between graph nodes. If min_only=False,
        dist_matrix has shape (n_indices, n_nodes) and dist_matrix[i, j]
        gives the shortest distance from point i to point j along the graph.
        If min_only=True, dist_matrix has shape (n_nodes,) and contains for
        a given node the shortest path to that node from any of the nodes
        in indices.
    predecessors : ndarray, shape ([n_indices, ]n_nodes,)
        If ``min_only=False``, this has shape ``(n_indices, n_nodes)``,
        otherwise it has shape ``(n_nodes,)``.
        If `indices` is None and ``min_only=False`` then ``n_indices = n_nodes``
        and the shape of the matrix becomes ``(n_nodes, n_nodes)``.
        Returned only if return_predecessors == True.
        The matrix of predecessors, which can be used to reconstruct
        the shortest paths.  Row i of the predecessor matrix contains
        information on the shortest paths from point i: each entry
        predecessors[i, j] gives the index of the previous node in the
        path from point i to point j.  If no path exists between point
        i and j, then predecessors[i, j] = -9999

    sources : ndarray, shape (n_nodes,)
        Returned only if min_only=True and return_predecessors=True.
        Contains the index of the source which had the shortest path
        to each target.  If no path exists within the limit,
        this will contain -9999.  The value at the indices passed
        will be equal to that index (i.e. the fastest way to reach
        node i, is to start on node i).

    Notes
    -----
    As currently implemented, Dijkstra's algorithm does not work for
    graphs with direction-dependent distances when directed == False.
    i.e., if csgraph[i,j] and csgraph[j,i] are not equal and
    both are nonzero, setting directed=False will not yield the correct
    result.

    Also, this routine does not work for graphs with negative
    distances.  Negative distances can lead to infinite cycles that must
    be handled by specialized algorithms such as Bellman-Ford's algorithm
    or Johnson's algorithm.

    If multiple valid solutions are possible, output may vary with SciPy and
    Python version.

    Examples
    --------
    >>> from scipy.sparse import csr_array
    >>> from scipy.sparse.csgraph import dijkstra

    >>> graph = [
    ... [0, 1, 2, 0],
    ... [0, 0, 0, 1],
    ... [0, 0, 0, 3],
    ... [0, 0, 0, 0]
    ... ]
    >>> graph = csr_array(graph)
    >>> print(graph)
    <Compressed Sparse Row sparse array of dtype 'int64'
        with 4 stored elements and shape (4, 4)>
        Coords	Values
        (0, 1)	1
        (0, 2)	2
        (1, 3)	1
        (2, 3)	3

    >>> dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=0, return_predecessors=True)
    >>> dist_matrix
    array([0., 1., 2., 2.])
    >>> predecessors
    array([-9999,     0,     0,     1], dtype=int32)

    """
    #------------------------------
    # validate csgraph and convert to csr
    csgraph = validate_graph(csgraph, directed, DTYPE, dense_output=False)
    if csgraph.data.dtype != np.float64:
        csgraph = csgraph.astype(np.float64)

    cdef double min_edge_weight = csgraph.data.min()
    cdef double max_edge_weight = csgraph.data.max()
    # C/δ: chave inteira máxima — dimensiona a fila corretamente (método do artigo)
    cdef long long C_ratio = <long long>(max_edge_weight / min_edge_weight)

    if np.any(csgraph.data < 0):
        warnings.warn("Graph has negative weights: dijkstra will give "
                      "inaccurate results if the graph contains negative "
                      "cycles. Consider johnson or bellman_ford.")

    N = csgraph.shape[0]

    #------------------------------
    # initialize/validate indices
    if indices is None:
        indices = np.arange(N, dtype=ITYPE)
        if min_only:
            return_shape = (N,)
        else:
            return_shape = indices.shape + (N,)
    else:
        indices = np.array(indices, order='C', dtype=ITYPE, copy=True)
        if min_only:
            return_shape = (N,)
        else:
            return_shape = indices.shape + (N,)
        indices = np.atleast_1d(indices).reshape(-1)
        indices[indices < 0] += N
        if np.any(indices < 0) or np.any(indices >= N):
            raise ValueError("indices out of range 0...N")

    cdef DTYPE_t limitf = limit
    cdef dijkstra_queue_t* heap = NULL
    if limitf < 0:
        raise ValueError('limit must be >= 0')

    #------------------------------
    # initialize dist_matrix for output
    if min_only:
        dist_matrix = np.full(N, np.inf, dtype=DTYPE)
        dist_matrix[indices] = 0
    else:
        dist_matrix = np.full((len(indices), N), np.inf, dtype=DTYPE)
        dist_matrix[np.arange(len(indices)), indices] = 0

    #------------------------------
    # initialize predecessors for output
    if return_predecessors:
        if min_only:
            predecessor_matrix = np.empty((N), dtype=ITYPE)
            predecessor_matrix.fill(NULL_IDX)
            source_matrix = np.empty((N), dtype=ITYPE)
            source_matrix.fill(NULL_IDX)
        else:
            predecessor_matrix = np.empty((len(indices), N), dtype=ITYPE)
            predecessor_matrix.fill(NULL_IDX)
            source_matrix = np.empty((len(indices), 0), dtype=ITYPE) # unused
    else:
        if min_only:
            predecessor_matrix = np.empty(0, dtype=ITYPE)
            source_matrix = np.empty(0, dtype=ITYPE) # unused
        else:
            predecessor_matrix = np.empty((len(indices), 0), dtype=ITYPE)
            source_matrix = np.empty((len(indices), 0), dtype=ITYPE) # unused

    if unweighted:
        csr_data = np.ones(csgraph.data.shape, dtype=np.float64)
    else:
        csr_data = csgraph.data.astype(np.float64)
    csr_indices, csr_indptr = safely_cast_index_arrays(csgraph, ITYPE, msg="csgraph")

    if directed:
        dummy_double_array = np.empty(0, dtype=DTYPE)
        dummy_int_array = np.empty(0, dtype=ITYPE)
        if min_only:
            heap = new dijkstra_queue_t(C_ratio, N, 3, min_edge_weight)
            _dijkstra(heap,
                      indices,
                      csr_data, csr_indices, csr_indptr,
                      dummy_double_array, dummy_int_array, dummy_int_array,
                      dist_matrix, predecessor_matrix, source_matrix,
                      limitf, min_edge_weight)
            del heap
        else:
            _dijkstra_multi_separate(
                      indices,
                      csr_data, csr_indices, csr_indptr,
                      dummy_double_array, dummy_int_array, dummy_int_array,
                      dist_matrix, predecessor_matrix, source_matrix,
                      limitf, C_ratio, min_edge_weight)

    else:
        csrT = csgraph.T.tocsr()
        csrT_indices, csrT_indptr = safely_cast_index_arrays(csrT, ITYPE, msg="csgraph")
        if unweighted:
            csrT_data = csr_data
        else:
            csrT_data = csrT.data.astype(np.float64)
        if min_only:
            heap = new dijkstra_queue_t(C_ratio, N, 3, min_edge_weight)
            _dijkstra(heap,
                      indices,
                      csr_data, csr_indices, csr_indptr,
                      csrT_data, csrT_indices, csrT_indptr,
                      dist_matrix, predecessor_matrix, source_matrix,
                      limitf, min_edge_weight)
            del heap
        else:
            _dijkstra_multi_separate(
                                 indices,
                                 csr_data, csr_indices, csr_indptr,
                                 csrT_data, csrT_indices, csrT_indptr,
                                 dist_matrix, predecessor_matrix, source_matrix,
                                 limitf, C_ratio, min_edge_weight)

    if return_predecessors:
        if min_only:
            return (dist_matrix.reshape(return_shape),
                    predecessor_matrix.reshape(return_shape),
                    source_matrix.reshape(return_shape))
        else:
            return (dist_matrix.reshape(return_shape),
                    predecessor_matrix.reshape(return_shape))
    else:
        return dist_matrix.reshape(return_shape)


ctypedef unsigned int uint_t
ctypedef pair[double, uint_t] dist_index_pair_t
#ctypedef priority_queue[dist_index_pair_t] dijkstra_queue_t
ctypedef bucket_queue[dist_index_pair_t] dijkstra_queue_t

@cython.boundscheck(False)
cdef void _dijkstra_scan_heap(dijkstra_queue_t &heap,
                         dist_index_pair_t v,
                         const double[:] csr_weights,
                         const int[:] csr_indices,
                         const int[:] csr_indptr,
                         double[:] dist_matrix,
                         int[:] pred,
                         int return_pred,
                         int[:] sources,
                         int return_source,
                         double delta,
                         double limit) noexcept nogil:
    cdef:
        ITYPE_t j
        unsigned int j_current
        double next_val

    for j in range(csr_indptr[v.second], csr_indptr[v.second + 1]):
        j_current = csr_indices[j]
        next_val = v.first + csr_weights[j]
        if next_val <= limit:
            if dist_matrix[j_current] > next_val:
                dist_matrix[j_current] = next_val
                heap.decrease_key(j_current, next_val)
                if return_pred:
                    pred[j_current] = v.second
                if return_source:
                    sources[j_current] = sources[v.second]


@cython.boundscheck(False)
cdef int _dijkstra(
            dijkstra_queue_t* heap,
            const int[:] source_indices,
            const double[:] csr_weights,
            const int[:] csr_indices,
            const int[:] csr_indptr,
            const double[:] csrT_weights,
            const int[:] csrT_indices,
            const int[:] csrT_indptr,
            double[:] dist_matrix,
            int[:] pred,
            int[:] sources,
            double limit,
            double delta) except -1:
    cdef:
        unsigned int Nind = source_indices.shape[0]
        unsigned int N = dist_matrix.shape[0]
        unsigned int i, j_source
        bint return_pred = (pred.shape[0] > 0)
        bint return_sources = (sources.shape[0] > 0)
        bint directed = (csrT_weights.shape[0] == 0)
        dist_index_pair_t v

    if return_pred and pred.shape[0] != N:
        raise RuntimeError(
            f"Invalid predecessors array shape {pred.shape}. Expected {(N,)}."
        )
    if return_sources and sources.shape[0] != N:
        raise RuntimeError(
            f"Invalid sources array shape {sources.shape}. Expected {(N,)}."
        )

    heap[0].reset()

    for i in range(Nind):
        j_source = source_indices[i]
        dist_matrix[j_source] = 0
        heap[0].push(dist_index_pair_t(0.0, j_source))
        if return_sources:
            sources[j_source] = j_source

    while heap[0].size():
        v = heap[0].top()
        heap[0].pop()

        _dijkstra_scan_heap(heap[0], v, csr_weights, csr_indices, csr_indptr,
                            dist_matrix, pred, return_pred,
                            sources, return_sources, delta, limit)
        if not directed:
            _dijkstra_scan_heap(heap[0], v,
                                csrT_weights, csrT_indices, csrT_indptr,
                                dist_matrix, pred, return_pred,
                                sources, return_sources, delta, limit)
    return 0


@cython.boundscheck(False)
cdef int _dijkstra_multi_separate(
            const int[:] source_indices,
            const double[:] csr_weights,
            const int[:] csr_indices,
            const int[:] csr_indptr,
            const double[:] csrT_weights,
            const int[:] csrT_indices,
            const int[:] csrT_indptr,
            double[:, :] dist_matrix,
            int[:, :] pred,
            int[:, :] sources,
            double limit,
            long long C_ratio,
            double delta) except -1:
    cdef:
        unsigned int Nind = source_indices.shape[0]
        unsigned int N = dist_matrix.shape[1]
        unsigned int i
        int source_list[1]
        dijkstra_queue_t* heap = new dijkstra_queue_t(C_ratio, N, 3, delta)

    if dist_matrix.shape[0] != Nind:
        del heap
        raise RuntimeError(
            f"Not enough rows in distances matrix. Got {dist_matrix.shape[0]}, expected {Nind}."
        )
    if pred.shape[0] != Nind:
        del heap
        raise RuntimeError(
            f"Not enough rows in predecessors matrix. Got {pred.shape[0]}, expected {Nind}."
        )
    if sources.shape[0] != Nind:
        del heap
        raise RuntimeError(
            f"Not enough rows in sources matrix. Got {sources.shape[0]}, expected {Nind}."
        )

    for i in range(Nind):
        source_list[0] = source_indices[i]
        _dijkstra(heap,
                  source_list,
                  csr_weights, csr_indices, csr_indptr,
                  csrT_weights, csrT_indices, csrT_indptr,
                  dist_matrix[i], pred[i], sources[i], limit, delta)

    del heap
    return 0