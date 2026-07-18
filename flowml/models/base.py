"""Defining a base class, so whichever model inherits from this class, must implement these methods. abc stands for abstract base class
    predict_proba() is optional, because not every model supports the probability predictions.
    We don't rely on score() for serious evaluations, it is used only as a quick performance check.
    """
from abc import ABC, abstractmethod 
class BaseModel(ABC): #BaseModel is an abstract class now
    @abstractmethod
    def fit(self,X,y): #This means every child should implement this method, there is no code in here, because every model trains differently
        pass
    @abstractmethod
    def predict(self,X):
        pass
    @abstractmethod
    def score(self,X,y):
        pass
    def predict_proba(self,X):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support probability predictions."
        )
      