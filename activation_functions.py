from abc import ABC, abstractmethod # ABC -> para criar classes abstratas

class ActivationFunction(ABC):
    
    @staticmethod
    @abstractmethod
    def g(u):
        pass


class BinaryStep(ActivationFunction):
    
    def g(u):
        return 1 if u >= 0 else 0 
    
    
class SignFunction(ActivationFunction):
    
    def g(u):
        return 1 if u >= 0 else -1
    