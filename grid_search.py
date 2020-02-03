from tqdm import tqdm
import torch
from DPOD.model import DPOD, PoseBlock
import os
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.model_selection import ParameterGrid
import json
from tempfile import TemporaryDirectory
from concurrent.futures import ProcessPoolExecutor


from infer_masks import infer_masks
from apply_ransac import apply_ransac
from DPOD.datasets import make_dataset, PATHS
from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
from DPOD.models_handler import ModelsHandler
from instances2submission import make_submission_from_ransac_directory
from mAP import calculate_loss_of_prediction

class JustImageDataset(Dataset):

    def __init__(self, kaggle_validation):
        super().__init__()
        self.kaggle_validation = kaggle_validation
    
    def __len__(self):
        return len(self.kaggle_validation)

    def __getitem__(self, index):
        image, _, _ = self.kaggle_validation[index]
        return image
    
    def get_IDs(self):
        return self.kaggle_validation.get_IDs()

def main(args):

    param_grid = [
        # {
        #     'iterationsCount': [400],
        #     'reprojectionError': [40.0, 48.0, 32.0, 56.0],
        #     'min_inliers': [50],
        #     'no_class': [False],
        # },
        # {
        #     'iterationsCount': [200],
        #     'reprojectionError': [32.0],
        #     'min_inliers': [50],
        #     'no_class': [False],
        # },
        {
            'iterationsCount': [500],
            'reprojectionError': [20.0, 24.0, 32.0],
            'min_inliers': [50],
            'no_class': [True],
        },

        # {
        #     'iterationsCount': [1000, 1500, 500],
        #     'reprojectionError': [16.0, 12.0, 8.0],
        #     'min_inliers': [30, 50, 100],
        #     'no_class': [True],
        # },
        # {
        #     'iterationsCount': [500, 300, 100],
        #     'reprojectionError': [16.0, 12.0, 8.0],
        #     'min_inliers': [30, 50, 100],
        #     'no_class': [False],
        # }
    ]

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare dataset
    val_data = KaggleImageMaskDataset(PATHS['kaggle'], setup="val")
    val_data = JustImageDataset(val_data)

    # load correspondence block
    model = DPOD(image_size=(2710 // 8, 3384 // 8))
    model.load_state_dict(torch.load(args.path_to_model))
    model.to(device)

    masks_paths = [os.path.join(args.path_to_masks_dir, name + ".npy") for name in val_data.get_IDs()]
    print("Infering masks")
    infer_masks(model, val_data, args.path_to_masks_dir, args.debug, device)
    scores = []

    for settings in tqdm(ParameterGrid(param_grid), smoothing=0.1):
        ransac_block = PoseBlock(
            PATHS['kaggle'], 
            min_inliers=settings['min_inliers'], 
            no_class=settings['no_class'], 
            reprojectionError=settings['reprojectionError'],
            iterationsCount=settings['iterationsCount'],
        )
        with TemporaryDirectory() as tmp:
            print(tmp)
            submission_path = os.path.join(tmp, "submission.csv")
            apply_ransac(masks_paths, ransac_block, tmp, args.debug)
            make_submission_from_ransac_directory(tmp, submission_path)
            loss = calculate_loss_of_prediction(submission_path, os.path.join(PATHS['kaggle'], "train.csv"))
    
            scores.append((loss, settings))

        with open(args.path_to_scores_file, "w") as f:
            json.dump(scores, f)


def grid_search2(args):
    param_grid = [
    #     {
    #         'iterationsCount': [140, 80],
    #         'reprojectionError': [24.0, 16.0, 8.0],
    #         'min_inliers': [50],
    #         'no_class': [False],
    #     },
        # {
        #     'iterationsCount': [200],
        #     'reprojectionError': [40.0, 48.0, 32.0],
        #     'min_inliers': [50],
        #     'no_class': [False],
        # },
        {
            'iterationsCount': [1000, 500],
            'reprojectionError': [16.0, 12.0, 8.0],
            'min_inliers': [30, 50, 100],
            'no_class': [True],
        },
        # {
        #     'iterationsCount': [500, 300, 100],
        #     'reprojectionError': [16.0, 12.0, 8.0],
        #     'min_inliers': [30, 50, 100],
        #     'no_class': [False],
        # }
    ]

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare dataset
    val_data = KaggleImageMaskDataset(PATHS['kaggle'], setup="val")
    val_data = JustImageDataset(val_data)

    # load correspondence block
    model = DPOD(image_size=(2710 // 8, 3384 // 8))
    model.load_state_dict(torch.load(args.path_to_model))
    model.to(device)

    masks_paths = [os.path.join(args.path_to_masks_dir, name + ".npy") for name in val_data.get_IDs()]
    print("Infering masks")
    infer_masks(model, val_data, args.path_to_masks_dir, args.debug, device)

    for settings in tqdm(ParameterGrid(param_grid), smoothing=0.1):
        models_handlers = [
            ModelsHandler(PATHS['kaggle']),
            ModelsHandler(PATHS['kaggle']),
            ModelsHandler(PATHS['kaggle']),
            ModelsHandler(PATHS['kaggle']),
            ModelsHandler(PATHS['kaggle']),
            ModelsHandler(PATHS['kaggle']),
            ModelsHandler(PATHS['kaggle']),
            ModelsHandler(PATHS['kaggle'])
        ]

        with TemporaryDirectory() as tmp:
            submission_path = os.path.join(tmp, "submission.csv")
            n_masks_to_process = 20 if args.debug else 100000

            def ransac_multiprocess(mask_path, idx):
                models_handler = models_handlers[idx % 8]
                image_id = os.path.split(mask_path)[1][:-4]
                output_file_path = os.path.join(
                    tmp,
                    f'{image_id}_instances.pkl'
                )

                tensor = torch.tensor(np.load(mask_path))
                class_, u_channel, v_channel = tensor
                c = class_.cpu().numpy()              # best class pixel wise
                u = u_channel.cpu().numpy()              # best color pixel wise
                v = v_channel.cpu().numpy()              # best color pixel wise
                if settings['no_class']:
                    instances = pnp_ransac_no_class(
                        c, u, v, 8, models_handler, 79,
                        min_inliers=settings["min_inliers"], 
                        k=5,  
                        reprojectionError=settings['reprojectionError'],
                        iterationsCount=settings['iterationsCount']
                    )  # todo optimize min_inliers
                else:
                    instances = pnp_ransac_multiple_instance(
                        c, u, v, 8, models_handler, 79,
                        min_inliers=settings["min_inliers"], 
                        reprojectionError=settings['reprojectionError'],
                        iterationsCount=settings['iterationsCount']
                    )  # todo optimize min_inliers

                output = []  # list of instances
                for success, ransac_rotation_matrix, ransac_translation_vector, pixel_coordinates_of_inliers, model_id in instances:
                    output.append(
                        [
                            model_id,
                            ransac_translation_vector,
                            ransac_rotation_matrix
                        ]
                    )
                ransac_instances = output

                with open(output_file_path, 'wb') as file:
                    pickle.dump(ransac_instances, file)

            bar = tqdm(total=len(masks_paths), smoothing=0.05)
            with torch.no_grad():
                with ProcessPoolExecutor() as executor:
                    for _ in executor.map(ransac_multiprocess, masks_paths, range(len(masks_paths))):
                        bar.update()

            make_submission_from_ransac_directory(tmp, submission_path)
            loss = calculate_loss_of_prediction(submission_path, os.path.join(PATHS['kaggle'], "train.csv"))
    
            scores.append((loss, settings))

        with open(args.path_to_scores_file, "w") as f:
            json.dump(scores, f)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description=
    """Searches parameters of ransac using predefined grid search""")
    arg_parser.add_argument('path_to_model')
    arg_parser.add_argument('path_to_masks_dir')
    arg_parser.add_argument('path_to_scores_file')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='process only 20 images')

    args = arg_parser.parse_args()

    main(args)