import logging
import os
import re

import kagglehub
from torchvision.datasets import ImageFolder

from src.dataset.base import BaseDataModule


def create_dir(base_path, classname):
    path = base_path + classname
    if not os.path.exists(path):
        os.mkdir(path)


def reorg(filename, base_path, wordmap):
    print(len(wordmap))
    with open("val/val_annotations.txt") as vals:
        for line in vals:
            vals = line.split()
            imagename = vals[0]
            print(vals[1])
            classname = wordmap[vals[1]]
            if os.path.exists(base_path + imagename):
                print(base_path + imagename, base_path + classname + "/" + imagename)
                os.rename(
                    base_path + imagename, base_path + classname + "/" + imagename
                )


class TinyImageNetDataModule(BaseDataModule):
    def prepare_data(self):
        """
        Directory reorg code from: https://github.com/pytorch/vision/issues/6127
        """
        dst_path = os.path.join(self.config["data_dir"], "tiny-imagenet-200")
        if os.path.exists(dst_path):
            return

        path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
        src_path = os.path.join(path, "tiny-imagenet-200")
        logging.info(f"Moving {src_path} to {dst_path}")
        os.rename(src_path, dst_path)
        curr_wdir = os.getcwd()
        os.chdir(dst_path)

        logging.info("Reorganizing data to fit ImageFolder expected structure")
        wordmap = {}
        with open("words.txt") as words, open("wnids.txt") as wnids:
            for line in wnids:
                vals = line.split()
                wordmap[vals[0]] = ""
            for line in words:
                vals = line.split()
                if vals[0] in wordmap:
                    single_words = vals[1:]
                    classname = re.sub(",", "", single_words[0])
                    if len(single_words) >= 2:
                        classname += "_" + re.sub(",", "", single_words[1])
                    wordmap[vals[0]] = classname
                    create_dir("./val/images/", classname)
                    if os.path.exists("./train/" + vals[0]):
                        os.rename("./train/" + vals[0], "./train/" + classname)
                    # create_dir('./test/images/', single_words[0])
                    # create_dir('./train/images/', single_words[0])

        reorg("val/val_annotations.txt", "val/images/", wordmap)

        # reorganize val images layout for consistency with train split
        for folder in os.listdir("val/images/"):
            assert os.path.isdir(os.path.join("val/images/", folder))
            os.makedirs(f"val/{folder}/images", exist_ok=False)
            os.rename(f"val/images/{folder}/", f"val/{folder}/images/")
        assert len(os.listdir("val/images")) == 0
        os.rmdir("val/images")

        # kagglehub checks that path exists before downloading; clean up to allow re-download
        # without manually cleaning cache
        assert len(os.listdir(path)) == 0, (
            f"Trying to delete folder at path {path}, but it is not empty!"
        )
        os.rmdir(path)

        os.chdir(curr_wdir)

    def get_dataset(self, split: str, transform):
        if split == "test":
            logging.warning(
                "No labels are available for the test split... Loading val dataset instead!"
            )
            split = "val"
        return ImageFolder(
            os.path.join(self.config["data_dir"], "tiny-imagenet-200", split), transform
        )
