from abc import ABCMeta, abstractmethod


class Base(metaclass=ABCMeta):
    def __init__(self):
        self.platform = 'base'

    @abstractmethod
    def show(self, **kwargs):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def summary(self):
        pass
