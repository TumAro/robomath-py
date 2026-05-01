from abc import ABC, abstractmethod

class LieGroup(ABC):

    @abstractmethod
    def inv(self):
        pass

    @abstractmethod
    def log(self):
        pass

    @abstractmethod
    def __matmul__(self):
        pass