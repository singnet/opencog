
## Tensor truth value examples

**learn_implications.py** contains example of learning truth values of implications fruit -> color given (color, fruit) pairs.

**learnable_formula.py** contains example of learning function that returns truth value of conclusion, given truth values of premises. Function has form of  sigmoid(wx + b) were w is weights vector. The weight vector may be learned alongside with truth values of fruit -> color implications, like it is done in learn_implications.py.
