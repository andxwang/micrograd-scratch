import random
from typing import Any
from value import Value


class Neuron:
    
    def __init__(self, nin: int):
        self.w = [Value(round(random.uniform(-1, 1), 3), label=f'w_{i}') for i in range(nin)]
        self.b = Value(round(random.uniform(-1, 1), 3), label='b')
        
    def __call__(self, x) -> Any:
        activ = sum((wi * xi for wi, xi in zip(self.w, x)), self.b); activ.label = 'activ'
        out = activ.tanh(); out.label = 'out'
        return out
    

class Layer:
    
    def __init__(self, nin, nout) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x) -> Any:
        return [n(x) for n in self.neurons]
    

class MLP:
    
    def __init__(self, nin: int, nouts: list[int]) -> None:
        layer_sizes = [nin] + nouts  # extending list [nin] by items in other list nouts
        # pair up consecutive (nin, nout) in layer_sizes
        self.layers = [Layer(nin=layer_sizes[i], nout=layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        
    def __call__(self, x) -> Any:
        for layer in self.layers:
            x = layer(x)
            
        return x

if __name__ == '__main__':
    random.seed(88)
    x = [2.0, -2.0]
    x = [Value(xi, label=f'x_{i}') for i, xi in enumerate(x)]
    mlp = MLP(len(x), [3, 1])
    print(out := mlp(x))
    from value import draw_graph
    draw_graph(out[0]).view()
