"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import argparse
from pathlib import Path

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData
from sklearn.model_selection import train_test_split

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self, root_dir=None, test_results_dir=None, n_epochs=8, learning_rate=0.0002,
                 batch_size=8, patch_size=64, name="Basic_unet"):
        self.name = "Basic_unet"
        project_root = Path(__file__).resolve().parents[1]
        self.root_dir = str(root_dir or os.getenv("HIPPO_DATA_DIR") or (project_root / "data" / "TrainingSet"))
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.test_results_dir = str(test_results_dir or os.getenv("HIPPO_OUTPUT_DIR") or (project_root / "out" / "predictions"))
        self.name = name


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate UNet on hippocampus NIfTI dataset")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path containing images/ and labels/ folders")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory where results and model checkpoint are saved")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--name", type=str, default="Basic_unet")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    c = Config(
        root_dir=args.data_dir,
        test_results_dir=args.output_dir,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        name=args.name,
    )

    required_dirs = [os.path.join(c.root_dir, "images"), os.path.join(c.root_dir, "labels")]
    missing = [p for p in required_dirs if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(
            "Dataset folder is not structured correctly. Expected:\n"
            f"  - {required_dirs[0]}\n"
            f"  - {required_dirs[1]}\n"
            f"Missing: {missing}"
        )

    # Load data
    print("Loading data...")

    # LoadHippocampusData
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)


    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for
    # multi-fold training to improve your model quality

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like
    # a k-fold training and combining the results.

    split = dict()

    # Create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation
    # and testing respectively.
    split['train'], split['test'] = train_test_split(keys, test_size=0.25, random_state=30)
    split['train'], split['val'] = train_test_split(split['train'], test_size=0.25, random_state=30)

    # Set up and run experiment

    # UNetExperiment start
    exp = UNetExperiment(c, split, data)

    # run training
    exp.run()

    # prep and run testing

    # Test method run
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))
