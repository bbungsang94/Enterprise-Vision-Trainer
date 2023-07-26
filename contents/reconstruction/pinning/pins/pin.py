import json
import os
from typing import List


class Pin:
    def __init__(self, **kwargs):
        self.name = ''
        self.categories = []
        self.landmark_index = 0
        self.basehaed_index = 0
        self.points68_index = 0
        self.set_pin(**kwargs)
        self.__key = 'landmark'

    def switch_to(self, mode: str):
        if "land" in mode:
            self.__key = "landmark"
        elif "base" in mode:
            self.__key = "basehead"
        else:
            self.__key = "68points"
        return self.__key

    def __call__(self) -> int:
        if self.__key == "landmark":
            return self.landmark_index
        elif self.__key == "basehead":
            return self.basehaed_index
        else:
            return self.points68_index

    def set_pin(self, level1: str, level2: str, level3: str,
                word_of_anatomy: str,
                landmark_index: int, basehead_index: int, points68_index: int):

        self.name = word_of_anatomy
        self.categories = [level1, level2, level3]
        self.landmark_index = landmark_index
        self.basehaed_index = basehead_index
        self.points68_index = points68_index


class PinBox:
    def __init__(self, name):
        self.name = name
        self._pins: List[Pin] = []

    def set_pins(self, pins: List[Pin]):
        self._pins.clear()
        self._pins = pins

    def append_pin(self, pin: Pin):
        self._pins.append(pin)

    def append_pins(self, pins: List[Pin]):
        self._pins += pins

    def switch_mode(self, mode: str):
        return [x.switch_to(mode) for x in self._pins]

    def __call__(self) -> List[int]:
        return [x() for x in self._pins]

    def __getitem__(self, indices):
        return self._pins[indices]

    def __len__(self):
        return len(self._pins)


class PinLoader:
    @staticmethod
    def load_pins(path, filename):
        with open(os.path.join(path, filename)) as f:
            pin_info = json.load(f)

        keys = pin_info['Form']['Titles']
        boxes = dict()
        for name, pins in pin_info.items():
            if isinstance(pins, list) is False:
                continue
            box = PinBox(name)
            for values in pins:
                kwargs = dict(zip(keys, values))
                pin = Pin(**kwargs)
                box.append_pin(pin)
            boxes[name] = box
        return boxes

