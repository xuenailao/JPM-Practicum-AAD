import numpy as np
from typing import Union, List, Tuple, Optional, Callable
from collections import defaultdict
import weakref


class Dual:
    """Dual number for forward-mode automatic differentiation."""
    
    def __init__(self, value: float, derivative: float = 0.0):
        self.value = value
        self.derivative = derivative
    
    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.value + other.value, self.derivative + other.derivative)
        else:
            return Dual(self.value + other, self.derivative)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.value - other.value, self.derivative - other.derivative)
        else:
            return Dual(self.value - other, self.derivative)
    
    def __rsub__(self, other):
        return Dual(other - self.value, -self.derivative)
    
    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.value * other.value, 
                       self.value * other.derivative + self.derivative * other.value)
        else:
            return Dual(self.value * other, self.derivative * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, Dual):
            return Dual(self.value / other.value,
                       (self.derivative * other.value - self.value * other.derivative) / (other.value ** 2))
        else:
            return Dual(self.value / other, self.derivative / other)
    
    def __rtruediv__(self, other):
        return Dual(other / self.value, -other * self.derivative / (self.value ** 2))
    
    def __neg__(self):
        return Dual(-self.value, -self.derivative)
    
    def __pow__(self, power):
        if isinstance(power, Dual):
            # f(x)^g(x) = exp(g(x) * ln(f(x)))
            return exp(power * log(self))
        else:
            return Dual(self.value ** power, power * (self.value ** (power - 1)) * self.derivative)
    
    def __repr__(self):
        return f"Dual(value={self.value}, derivative={self.derivative})"


def exp(x):
    """Exponential function supporting Dual numbers."""
    if isinstance(x, Dual):
        exp_val = np.exp(x.value)
        return Dual(exp_val, exp_val * x.derivative)
    return np.exp(x)


def log(x):
    """Natural logarithm supporting Dual numbers."""
    if isinstance(x, Dual):
        return Dual(np.log(x.value), x.derivative / x.value)
    return np.log(x)


def sqrt(x):
    """Square root supporting Dual numbers."""
    if isinstance(x, Dual):
        sqrt_val = np.sqrt(x.value)
        return Dual(sqrt_val, x.derivative / (2 * sqrt_val))
    return np.sqrt(x)


def normal_cdf(x):
    """Cumulative distribution function of standard normal distribution."""
    from scipy.stats import norm
    if isinstance(x, Dual):
        return Dual(norm.cdf(x.value), norm.pdf(x.value) * x.derivative)
    return norm.cdf(x)


def normal_pdf(x):
    """Probability density function of standard normal distribution."""
    from scipy.stats import norm
    if isinstance(x, Dual):
        pdf_val = norm.pdf(x.value)
        return Dual(pdf_val, -x.value * pdf_val * x.derivative)
    return norm.pdf(x)


class Variable:
    """Variable for reverse-mode automatic differentiation (backpropagation)."""
    _counter = 0
    
    def __init__(self, value: Union[float, np.ndarray], name: Optional[str] = None):
        self.value = np.array(value, dtype=np.float64) if not isinstance(value, np.ndarray) else value.astype(np.float64)
        self.grad = np.zeros_like(self.value, dtype=np.float64)
        self.name = name or f"var_{Variable._counter}"
        Variable._counter += 1
        self._backward = lambda: None
        self._prev = set()
    
    def backward(self, grad: Optional[np.ndarray] = None):
        """Compute gradients using reverse-mode differentiation."""
        if grad is None:
            grad = np.ones_like(self.value)
        
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        self.grad = grad
        for v in reversed(topo):
            v._backward()
    
    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value + other.value)
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        out._prev = {self, other}
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value * other.value)
        
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        
        out._backward = _backward
        out._prev = {self, other}
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Power must be a scalar"
        out = Variable(self.value ** power)
        
        def _backward():
            self.grad += power * (self.value ** (power - 1)) * out.grad
        
        out._backward = _backward
        out._prev = {self}
        return out
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def exp(self):
        out = Variable(np.exp(self.value))
        
        def _backward():
            self.grad += out.value * out.grad
        
        out._backward = _backward
        out._prev = {self}
        return out
    
    def log(self):
        out = Variable(np.log(self.value))
        
        def _backward():
            self.grad += (1 / self.value) * out.grad
        
        out._backward = _backward
        out._prev = {self}
        return out
    
    def sqrt(self):
        return self ** 0.5
    
    def __repr__(self):
        return f"Variable(value={self.value}, grad={self.grad}, name={self.name})"