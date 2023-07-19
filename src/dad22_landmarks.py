#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
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
        self.cfg = None

    def parse_options(self, params):
        unknown = super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='Dad22Landmarks', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        parser.add_argument('--cfg', dest='cfg', type=str,
                            help='Training configuration filename.')
        args, unknown = parser.parse_known_args(unknown)
        print(parser.format_usage())
        self.gpu = args.gpu
        self.cfg = args.cfg

    def train(self, anns_train, anns_valid):
        import yaml
        import hydra
        from omegaconf import OmegaConf
        from .model_training.data import FlameDataset
        from .model_training.model import load_model
        from .model_training.train.trainer import DAD3DTrainer
        from .model_training.train.flame_lightning_model import FlameLightningModel
        # Prepare experiment
        print('Train model')
        experiment_dir = self.path+'logs/'
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        hydra.initialize(config_path='model_training/config')
        hydra_config = hydra.compose(config_name=self.cfg)
        OmegaConf.set_struct(hydra_config, False)
        hydra_config['yaml_path'] = os.path.join(experiment_dir, self.cfg)
        hydra_config['experiment']['folder'] = experiment_dir
        # print(OmegaConf.to_yaml(hydra_config, resolve=True))
        config = yaml.load(OmegaConf.to_yaml(hydra_config, resolve=True), Loader=yaml.FullLoader)
        with open(hydra_config['yaml_path'], 'w') as ofs:
            OmegaConf.save(config=config, f=ofs.name)
        ofs.close()
        print('Experiment dir: %s' % config['experiment']['folder'])
        # Train
        train_dataset = FlameDataset.from_config(config=config['train'])
        val_dataset = FlameDataset.from_config(config=config['val'])
        model = load_model(config['model'], config['constants'])
        dad3d_net = FlameLightningModel(model=model, config=config, train=train_dataset, val=val_dataset)
        dad3d_trainer = DAD3DTrainer(dad3d_net, config)
        dad3d_trainer.fit()

    def load(self, mode):
        from images_framework.src.constants import Modes
        from .utils import load_yaml
        from .predictor import FaceMeshPredictor
        # Set up a neural network to train
        print('Load model')
        if mode is Modes.TEST:
            config = load_yaml(self.path+'data/'+self.database+'.yaml')
            self.model = FaceMeshPredictor(self.path+'data/', config=config)

    def process(self, ann, pred):
        import itertools
        from pytorch_toolbelt.utils import read_rgb_image
        from scipy.spatial.transform import Rotation
        from images_framework.src.datasets import Database
        from images_framework.src.annotations import GenericLandmark
        from .model_training.model.flame import FlameParams, FLAME_CONSTS
        from .model_training.model.utils import rot_mat_from_6dof
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        idx = [datasets.index(subset) for subset in datasets if self.database in subset]
        parts = Database.__subclasses__()[idx[0]]().get_landmarks()
        indices = list(itertools.chain.from_iterable(parts.values()))
        for img_pred in pred.images:
            # Load image
            image = read_rgb_image(img_pred.filename)
            for obj_pred in img_pred.objects:
                # Generate prediction
                predictions = self.model(image)  # 68 2D points, 2.5D projected vertices, 3D vertices, 3DMM params
                # Save prediction
                params_3dmm = predictions['3dmm_params']  # (1, 413)
                flame_params = FlameParams.from_3dmm(params_3dmm, FLAME_CONSTS)
                rot_mat = np.squeeze(rot_mat_from_6dof(flame_params.rotation).cpu().numpy())
                euler = Rotation.from_matrix(rot_mat).as_euler('YXZ', degrees=True)
                obj_pred.headpose = Rotation.from_euler('YXZ', [euler[0], -euler[1], -euler[2]], degrees=True).as_matrix()
                obj_pred.headpose[1:3, :] = -obj_pred.headpose[1:3, :]
                proj_vertices = np.squeeze(predictions['projected_vertices'].cpu().numpy())
                for idx in indices:
                    lp = list(parts.keys())[next((ids for ids, xs in enumerate(parts.values()) for x in xs if x == idx), None)]
                    obj_pred.add_landmark(GenericLandmark(idx, lp, proj_vertices[idx], True))
