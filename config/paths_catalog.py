import os

class DatasetCatalog():
    # DATA_DIR = "/media/franco/elenagovi/"
    # DATASETS = {
    #     "kitti_train": {
    #         "root": "KITTI/training/",
    #     },
    #     "kitti_test": {
    #         "root": "KITTI/testing/",
    #     },
    # }


    DATA_DIR = '/media/franco/hdd/dataset/dataset_3d'

    DATASETS = {
        'indy_virginia':{
        'root': '20250104_lvms_run02_virginia/camera_fc_cropped',
        },
        'indy_polimove2':{
        'root': '20250105_lvms_run02_multi_polimove/camera_fc_cropped',
        },
        'indy_polimove3':{'root': '20250105_lvms_run03_multi_polimove/camera_fc_cropped',
        }
        }

    @staticmethod
    def get(name):
        if "kitti" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root"]),
            )
            return dict(
                factory="KITTIDataset",
                args=args,
            )
        elif "indy" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root"]),
            )
            return dict(
                factory="INDYDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog():
    IMAGENET_MODELS = {
        "DLA34": "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"
    }

    @staticmethod
    def get(name):
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_imagenet_pretrained(name)

    @staticmethod
    def get_imagenet_pretrained(name):
        name = name[len("ImageNetPretrained/"):]
        url = ModelCatalog.IMAGENET_MODELS[name]
        return url
