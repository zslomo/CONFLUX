# -*- coding: utf-8 -*-
# Copyright (C) 2018, Tencent Inc.
# Author: bintan

import json
import pickle
from tqdm import tqdm

class units(object):
    def __init__(self, logger):
        self.logger = logger

    def get_json_file(self, json_file):
        with open(json_file, 'r') as file:
            ret = json.load(file)
        return ret

    def get_pre_sample(self, last_pre_sample_file):
        ret = []
        with open(last_pre_sample_file, 'r') as file:
            for line in file.readlines():
                ret.append(line.replace("\n", "").split(","))
        return ret


    def load_pickle_samples(self, file_names):
        samples = []
        for file_name in tqdm(file_names):
            with open(file_name, 'rb') as file:
                sub_process_batch_samples = pickle.load(file)
                samples += sub_process_batch_samples
        return samples

    def load_dict(self, load_file):
        ret = []
        with open(load_file, 'rb') as file:
            ret = pickle.load(file)
        return ret
