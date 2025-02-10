import numpy as np
class Node(object):
    """Initialize with a set of vectors, then call `split()`.
    """

    def __init__(self, ref: np.ndarray, vecs: List[np.ndarray]):
        self._ref = ref
        self._vecs = vecs
        self._left = None
        self._right = None

    @property
    def ref(self) -> Optional[np.ndarray]:
        """Reference point in n-d hyperspace. Evaluates to `False` if root node.
        """
        return self._ref

    @property
    def vecs(self) -> List[np.ndarray]:
        """Vectors for this leaf node. Evaluates to `False` if not a leaf.
        """
        return self._vecs

    @property
    def left(self) -> Optional[object]:
        """Left node.
        """
        return self._left

    @property
    def right(self) -> Optional[object]:
        """Right node.
        """
        return self._right
    
def _is_query_in_left_half(q, node):
    # returns `True` if query vector resides in left half
    dist_l = np.linalg.norm(q- node.vecs[0])
    dist_r = np.linalg.norm(q- node.vecs[1])
    return dist_l < dist_r


# Building the actual tree

import random

def split_node(node, K: int, imb: float) -> bool:
    
    # stopping condition: maximum # of vectors for a leaf node
    if len(node._vecs) <= K:
        return False
    
    # continue for a maximum of 5 iterations
    for n in range(5):
        left_vecs = []
        right_vecs =[]
        
        # take two random indexes and set as left and right halves
        left_ref = node._vecs.pop(np.random.randint(len(node._vecs)))
        right_ref = node._vecs.pop(np.random.randint(len(node._vecs)))
        
        # Split vectors into halves
        for vec in node._vecs:
            dist_l = np.linalg.norm(vec -left_ref)
            dist_r = np.linalg.norm(vec -right_ref)
            if dist_l < dist_r:
                left_vecs.append(vec)
            else:
                right_vecs.append(vec)
                
        # check to make sure that the tree is mostly balanced
        r = len(left_vecs) / len(node._vecs)
        if r < imb and r > (1 - imb):
            node._left = Node(left_ref, left_vecs)
            node._right = Node(right_ref, right_vecs)
            return True

        # redo tree build process if imbalance is high
        node._vecs.append(left_ref)
        node._vecs.append(right_ref)
    return False

def _build_tree(node, K:int, imb:float):
        """Recurses on left and right halves to build a tree.  """
        node.split(K, imb)
        if(node.left):
            _build_tree(node.left, K, imb)

        if node.right:
            _build_tree(node.right, K=K, imb=imb)

def build_forest(vecs: List[np.ndarray], N: int = 32, K: int = 64, imb: float = 0.95) -> List[Node]:
    """Builds a forest of `N` trees.   """
    forest = []
    for _ in range(N):
        root = Node(None, vecs)
        _build_tree(root, K, imb)
        forest.append(root)
    return forest
# Querying is pretty straightforward; we traverse the tree, continuously moving along the left or right branches until we've arrived at the one we're interested in:

def _query_linear(vecs: List[np.ndarray], q: np.ndarray, k: int) -> List[np.ndarray]:
    return sorted(vecs, key=lambda v: np.linalg.norm(q-v))[:k]


def query_tree(root: Node, q: np.ndarray, k: int) -> List[np.ndarray]:
    """Queries a single tree.
    """

    while root.left and root.right:
        dist_l = np.linalg.norm(q - node.left.ref)
        dist_r = np.linalg.norm(q - node.right.ref)
        root = root.left if dist_l < dist_r else root.right

    # brute-force search the nearest neighbors
    return _query_linear(root.vecs, q, k)

def _select_nearby(node: Node, q: np.ndarray, thresh: int = 0):
    """Functions identically to _is_query_in_left_half, but can return both.
    """
    if not node.left or not node.right:
        return ()
    dist_l = np.linalg.norm(q - node.left.ref)
    dist_r = np.linalg.norm(q - node.right.ref)
    if np.abs(dist_l - dist_r) < thresh:
        return (node.left, node.right)
    if dist_l < dist_r:
        return (node.left,)
    return (node.right,)


def _query_tree(root: Node, q: np.ndarray, k: int) -> List[np.ndarray]:
    """This replaces the `query_tree` function above.
    """

    pq = [root]
    nns = []
    while pq:
        node = pq.pop(0)
        nearby = _select_nearby(node, q, thresh=0.05)

        # if `_select_nearby` does not return either node, then we are at a leaf
        if nearby:
            pq.extend(nearby)
        else:
            nns.extend(node.vecs)

    # brute-force search the nearest neighbors
    return _query_linear(nns, q, k)


def query_forest(forest: List[Node], q, k: int = 10):
    nns = set()
    for root in forest:
        # merge `nns` with query result
        res = _query_tree(root, q, k)
        nns.update(res)
    return _query_linear(nns, q, k)