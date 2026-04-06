class Value:

    def __init__(self, data: int, children=(), label='', _op=''):
        self.data = data
        self.grad = 0.0
        self.prev = children
        self.label = label
        self._op = _op
        self._backward = lambda: None  # default none backward

    def __add__(self, other):
        if isinstance(other, int):
            other = Value(other)
        
        out = Value(self.data + other.data, (self, other,), _op='+')

        def _backward():
            # already have out.grad here?
            self.grad = out.grad * 1
            other.grad = out.grad * 1
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        if isinstance(other, int):
            other = Value(other)
        out = Value(self.data * other.data, children=(self, other), _op='*')

        def _backward():
            self.grad = out.grad * other.data
            other.grad = out.grad * self.data
        out._backward = _backward

        return out
    
    def backward(self):
        from collections import deque
        topo_sort = []
        visited = set()
        visited.add(self)
        # Loss (final node) is the only one with 0 "indegrees", so start from it
        queue = deque([self])
        while queue:
            node = queue.popleft()
            topo_sort.append(node)            
            for prev in node.prev:
                if prev not in visited:
                    visited.add(prev)  # equivalent to "removing node" from graph
                    queue.append(prev)
        
        print("topo sort:", topo_sort)
        self.grad = 1.0
        for node in topo_sort:
            node._backward()

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self.__add__(-other)

    def __repr__(self):
        return f"Value({self.label}: data={self.data}, grad={self.grad})"
    
def draw_graph(root):
    """
    Creates a graphviz visualization of the computation graph.
    Layout is left to right, with the final value (root) on the right.
    Operation nodes are displayed as circles between their inputs and outputs.
    """
    from graphviz import Digraph
    
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    visited = set()
    
    def traverse(node):
        if node in visited:
            return
        visited.add(node)
        
        node_id = str(id(node))
        # Create node label with name, data and grad values
        parts = []
        if node.label:
            parts.append(node.label)
        parts.append(f"data={node.data}")
        parts.append(f"grad={node.grad}")
        label = "\n".join(parts)
        dot.node(node_id, label, shape='box')
        
        # If this node has an operator, create an operation node
        if node._op:
            op_node_id = f"op_{id(node)}"
            dot.node(op_node_id, node._op, shape='circle')
            # Connect operation node to this node
            dot.edge(op_node_id, node_id)
            
            # Connect children to operation node
            for child in node.prev:
                traverse(child)
                child_id = str(id(child))
                dot.edge(child_id, op_node_id)
        else:
            # No operator - just connect children directly
            for child in node.prev:
                traverse(child)
                child_id = str(id(child))
                dot.edge(child_id, node_id)
    
    traverse(root)
    return dot
    
if __name__ == '__main__':
    v1 = Value(2.0)
    v2 = Value(3.0)
    v3 = v1 * v2
    v3.grad = 1.0
    print(v3)
    v3.backward()
    print(v3)
