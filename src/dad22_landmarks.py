#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import cv2
import numpy as np
from images_framework.src.alignment import Alignment
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)


class Dad22Landmarks(Alignment):
    """
    Object alignment using DAD-3DHeads algorithm
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None
        self.gpu = None

    def parse_options(self, params):
        unknown = super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='Dad22Landmarks', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        args, unknown = parser.parse_known_args(unknown)
        print(parser.format_usage())
        self.gpu = args.gpu

    def train(self, anns_train, anns_valid):
        print('Train model')

    def load(self, mode):
        from images_framework.src.constants import Modes
        from .utils import load_yaml
        from .predictor import FaceMeshPredictor
        # Set up a neural network to train
        print('Load model')
        if mode is Modes.TEST:
            config = load_yaml(self.path + 'data/' + self.database + '.yaml')
            self.model = FaceMeshPredictor(self.path + 'data/', config=config)

    def process(self, ann, pred):
        import itertools
        from images_framework.src.datasets import Database
        from images_framework.src.annotations import GenericLandmark
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        idx = [datasets.index(subset) for subset in datasets if self.database in subset]
        parts = Database.__subclasses__()[idx[0]]().get_landmarks()
        indices = list(itertools.chain.from_iterable(parts.values()))
        for img_pred in pred.images:
            # Load image
            image = cv2.imread(img_pred.filename)
            for obj_pred in img_pred.objects:
                # Generate prediction
                predictions = self.model(image)
                print(predictions)
                exit(0)
                # Save prediction
                for idx, pt in enumerate(shape.parts()):
                    label = indices[idx]
                    lp = list(parts.keys())[next((ids for ids, xs in enumerate(parts.values()) for x in xs if x == label), None)]
                    obj_pred.add_landmark(GenericLandmark(label, lp, (pt.x, pt.y), True))
