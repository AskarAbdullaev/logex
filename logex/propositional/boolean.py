from typing import Union

class _boolean():
    """
    The necessity of this custom class is due to default operations
    precedence which makes it impossible to natively evaluate arbitrary 
    logical formulas in Python. To solve this issue,
    e.g. "^" - bitwise XOR opertaor is reverted to obtain 
    a custom implication, which is performed after conjunction but before
    equivalence, etc.
    """

    def __init__(self, x: Union[bool, int, str]):
        if x == True or x == 1 or x == 'True' or x == '1':
            self.value = 1
        elif x == False or x == 0 or x == 'False' or x == '0':
            self.value = 0
        else:
            raise ValueError(f'impossible to convert to boolean: {x}')

    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return str(self)
    
    def __neg__(self):
        return _boolean((self.value + 1) % 2)

    def __add__(self, y):
        if not isinstance(y, _boolean):
            y = _boolean(y)
        return _boolean(min(self.value + y.value, 1))
    
    def __radd__(self, y):
        if not isinstance(y, _boolean):
            y = _boolean(y)
        return _boolean(min(self.value + y.value, 1))
    
    def __mul__(self, y):
        if not isinstance(y, _boolean):
            y = _boolean(y)
        return _boolean(self.value * y.value)
    
    def __rmul__(self, y):
        if not isinstance(y, _boolean):
            y = _boolean(y)
        return _boolean(self.value * y.value)
    
    def __xor__(self, y):
        if not isinstance(y, _boolean):
            y = _boolean(y)
        return _boolean(int(self.value <= y.value))
    
    def __rxor__(self, y):
        if not isinstance(y, _boolean):
            y = _boolean(y)
        return _boolean(int(self.value >= y.value))
    
    def __eq__(self, y):
        if not isinstance(y, _boolean):
            y = _boolean(y)
        return _boolean(int(self.value == y.value))
    
    def __ne__(self, y):
        if not isinstance(y, _boolean):
            y = _boolean(y)
        return _boolean(int(self.value != y.value))