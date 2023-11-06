from abc import ABCMeta, abstractmethod


class Base(metaclass=ABCMeta):
    def __init__(self):
        self.platform = 'base'

    @abstractmethod
    def show(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass

    @abstractmethod
    def summary(self, **kwargs):
        pass


class Concrete(Base):
    def show(self, **kwargs):
        pass

    def save(self, **kwargs):
        pass

    def summary(self, **kwargs):
        pass
