import torch
import logging
import pdb
import os
import datetime
import warnings
warnings.filterwarnings("ignore")

from config import cfg
from data import make_data_loader
from solver import build_optimizer, build_scheduler

from utils.check_point import DetectronCheckpointer
from engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from utils import comm
from utils.backup_files import sync_root

from engine.trainer import do_train
from engine.test_net import run_test

from model.detector import KeypointDetector
from data import build_test_loader
from config.paths_catalog import DatasetCatalog
import resource
from structures.image_list import to_image_list

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.backends.cudnn.enabled = True # enable cudnn and uncertainty imported
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True # enable cudnn to search the best algorithm


def test_dataloader(dataloader, max_iter):
    """
    Test a dataloader by reading all images and labels without inference or training.
    Args:
        dataloader: The dataloader to be tested.
    """
    print("Starting dataloader test...")
    # max_iter =  #cfg.SOLVER.MAX_ITERATION
    start_iter = 0
    for data, iteration in zip(dataloader, range(start_iter, max_iter)):
        print(iteration)
        # try:
        # Assuming your dataloader returns a dictionary or tuple with images and labels
        images = data["images"]
        images = to_image_list(images)
        # print('data', data.keys())
        targets = [target for target in data["targets"]]

        print("===== Target Information =====")
        for target in targets:
            if target.has_field("cls_ids"):
                print(f"Class IDs: {target.get_field('cls_ids')}")
            
            if target.has_field("gt_bboxes"):
                print(f"GT Bounding Boxes: {target.get_field('gt_bboxes')}")
            
            if target.has_field("2d_bboxes"):
                print(f"2D Bounding Boxes: {target.get_field('2d_bboxes')}")
            
            if target.has_field("target_centers"):
                print(f"Target Centers: {target.get_field('target_centers')}")
            
            if target.has_field("calib"):
                print(f"Calib: {target.get_field('calib')}")
            
            if target.has_field("dimensions"):
                print(f"Dimensions: {target.get_field('dimensions')}")
            
            if target.has_field("locations"):
                print(f"Locations: {target.get_field('locations')}")
            
            if target.has_field("orientations"):
                print(f"Orientations: {target.get_field('orientations')}")
            
            if target.has_field("alphas"):
                print(f"Alphas: {target.get_field('alphas')}")
            
            if target.has_field("rotys"):
                print(f"Rotations (rotys): {target.get_field('rotys')}")
            
            if target.has_field("occlusions"):
                print(f"Occlusions: {target.get_field('occlusions')}")
            
            if target.has_field("truncations"):
                print(f"Truncations: {target.get_field('truncations')}")
            
            # Optional: Add more fields as necessary
        print("==============================")

    print("Dataloader test completed.")


def get_dataset_path():
    # Access DATA_DIR and the 'root' value for 'kitti_train'
    data_dir = DatasetCatalog.DATA_DIR
    dataset_info = DatasetCatalog.DATASETS.get('indy_train', {})
    dataset_root = dataset_info.get('root', '')

    # Construct the full path
    full_path = os.path.join(data_dir, dataset_root)
    return full_path

def check_kitti_dataset(dataset_path):
    """
    Checks the integrity of a KITTI-format dataset.
    Verifies that all necessary files exist and that their structure is valid.
    """
    print(f"Checking dataset at: {dataset_path}")
    
    # Define the expected subfolders
    required_folders = ['image_2', 'label_2', 'calib']
    missing_folders = [folder for folder in required_folders if not os.path.isdir(os.path.join(dataset_path, folder))]
    
    if missing_folders:
        print(f"Error: Missing required folders: {missing_folders}")
        return False

    # Verify that each folder contains the same number of files
    image_files = os.listdir(os.path.join(dataset_path, 'image_2'))
    label_files = os.listdir(os.path.join(dataset_path, 'label_2'))
    calib_files = os.listdir(os.path.join(dataset_path, 'calib'))
    
    if not (len(image_files) == len(label_files) == len(calib_files)):
        print("Error: Mismatched number of files between 'image_2', 'label_2', and 'calib' folders.")
        print(f"Images: {len(image_files)}, Labels: {len(label_files)}, Calibrations: {len(calib_files)}")
        return False

    # Verify that all file pairs have matching IDs
    image_ids = {os.path.splitext(f)[0] for f in image_files}
    label_ids = {os.path.splitext(f)[0] for f in label_files}
    calib_ids = {os.path.splitext(f)[0] for f in calib_files}

    if not (image_ids == label_ids == calib_ids):
        print("Error: File IDs do not match across 'image_2', 'label_2', and 'calib'.")
        unmatched_ids = {
            "image_only": image_ids - label_ids - calib_ids,
            "label_only": label_ids - image_ids - calib_ids,
            "calib_only": calib_ids - image_ids - label_ids,
        }
        print(f"Unmatched IDs: {unmatched_ids}")
        return False

    print("Dataset structure looks correct.")
    return True



def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_work
    cfg.TEST.EVAL_DIS_IOUS = args.eval_iou
    cfg.TEST.EVAL_DEPTH = args.eval_depth
    cfg.TEST.VIS = args.vis
    cfg.TEST.VIS_ALL = args.vis_all
    cfg.TEST.VIS_HORIZON = args.vis_horizon
    
    if args.vis_thre > 0:
        cfg.TEST.VISUALIZE_THRESHOLD = args.vis_thre 
    
    if args.output is not None:
        cfg.OUTPUT_DIR = args.output

    if args.test:
        cfg.DATASETS.TEST_SPLIT = 'test'
        cfg.DATASETS.TEST = ("kitti_test",)

    cfg.START_TIME = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H:%M:%S')
    default_setup(cfg, args)

    # Verify dataset if in train or test mode
    dataset_path = get_dataset_path()
    print(dataset_path)
    if not check_kitti_dataset(dataset_path):
        print(dataset_path)
        raise RuntimeError("Dataset integrity check failed! ")


    return cfg

def main(args):
    max_iter = 10
    cfg = setup(args)
    data_loader = make_data_loader(cfg, is_train=True)
    data_loaders_val = build_test_loader(cfg, is_train=False)
    test_dataloader(data_loader, max_iter) #TODO save some images with ground truth bounding box visualization


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    
    print("Command Line Args:", args)
    if not args.eval_only and args.output is not None:
        sync_root('.', os.path.join(args.output, 'backup'))
        import shutil
        shutil.copy2(args.config_file, os.path.join(args.output, 'backup', os.path.basename(args.config_file)))

        print("Finish backup all files")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )