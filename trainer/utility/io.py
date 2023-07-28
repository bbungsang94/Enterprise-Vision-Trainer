import os
import copy
import json
from datetime import datetime

import pandas as pd

from contents import REGISTRY as MODULES


def read_json(full_path=''):
    if '.json' not in full_path:
        full_path += '.json'
    with open(full_path, "r") as f:
        edge_info = json.load(f)
    return edge_info


def replace_values(source, target):
    for key in source.keys():
        if key in target:
            source[key] = copy.deepcopy(target[key])
    return source


class ModuleLoader:
    def __init__(self, root, params):
        self._root = root
        self.params = params

    def get_module(self, kind, base, **kwargs):
        key = self.params['modules'][kind]

        func, args = self.get_safety_registry(kind, key, **kwargs)
        if func is None:
            key = args['name']
            del args['name']
            func = getattr(base, key)(**args)
        return func

    def get_args(self, kind, key, **options):
        args = read_json(os.path.join(self._root, kind, key))
        hyperparameter = self.params['hyperparameters']
        args = replace_values(args, hyperparameter)
        for key in options.keys():
            args[key] = options[key]

        return args

    def get_safety_registry(self, kind, key, **kwargs):
        args = self.get_args(kind, self.params['modules'][kind], **kwargs)
        if key + kind in MODULES:
            return MODULES[key + kind](**args), args
        else:
            return None, args


class CSVWriter:
    def __init__(self, path):
        self.save_folder = path
        if os.path.exists(path) is False:
            dir_q = []
            sub_path = path
            while True:
                directory, folder = os.path.split(sub_path)
                sub_path = directory
                dir_q.append(folder)
                if os.path.exists(directory):
                    for target in reversed(dir_q):
                        sub_path = os.path.join(sub_path, target)
                        os.mkdir(os.path.join(sub_path))
                    break
        self.datum = dict()

    def add_scalars(self, column, data, **kwargs):
        data = copy.deepcopy(data)
        if column in self.datum:
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        self.datum[column][key] += value
                    else:
                        self.datum[column][key].append(value)
            elif isinstance(data, list):
                self.datum[column] += data
            else:
                self.datum[column].append(data)
        else:
            if isinstance(data, dict):
                self.datum[column] = dict()
                for key, value in data.items():
                    self.datum[column][key] = [value]
            elif isinstance(data, list):
                self.datum[column] = data
            else:
                self.datum[column] = [data]

    def flush(self):
        datum = copy.deepcopy(self.datum)
        del_cols = []
        for key, value in datum.items():
            if isinstance(value, dict):
                del_cols.append(key)
                df = pd.DataFrame(data=value)
                df.to_csv(os.path.join(self.save_folder, key + '.csv'), index=False)
        for col in del_cols:
            datum.pop(col, None)

        if len(datum) > 0:
            df = pd.DataFrame(data=datum)
            df.to_csv(os.path.join(self.save_folder, 'Summary.csv'), index=False)
