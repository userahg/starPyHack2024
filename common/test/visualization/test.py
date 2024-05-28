import os
import unittest
import shutil
from pathlib import Path
import common.visualization as viz


class TestPrepDirsForAnimationDM(unittest.TestCase):

    def setUp(self) -> None:
        self.src_dir = Path(r"D:\PycharmProjects\siemens\common\test\visualization")
        self.test_src = self.src_dir.joinpath(r"dm_study_dir")
        self.test_wrk = self.src_dir.joinpath(r"rename_img_4_ani_dm")

    def test_all_images(self):
        if self.test_wrk.exists():
            shutil.rmtree(self.test_wrk)
        shutil.copytree(self.test_src, self.test_wrk)
        test_wrk = self.test_wrk
        viz.prepare_dirs_for_animation_dm(test_wrk, clean_first=False)
        images_dir = test_wrk.joinpath("images")
        n_dirs = len([f for f in images_dir.iterdir() if f.is_dir()])
        n_images = []
        img_dir_names = []
        for f in images_dir.iterdir():
            if f.is_dir:
                n_images.append(len([i for i in f.iterdir()]))
                img_dir_names.append(f.name)
        d_true = {"n_images": 4, "GEOM": 88, "Mach": 88, "Mesh": 88, "Press": 88}
        d_test = {"n_images": len(n_images)}
        for i in range(n_dirs):
            d_test[img_dir_names[i]] = n_images[i]
        shutil.rmtree(test_wrk)
        self.assertDictEqual(d_true, d_test)

    def test_specified_images(self):
        if self.test_wrk.exists():
            shutil.rmtree(self.test_wrk)
        shutil.copytree(self.test_src, self.test_wrk)
        test_wrk = self.test_wrk
        viz.prepare_dirs_for_animation_dm(test_wrk, clean_first=False, image_names=["GEOM.png", "Mesh.png"])
        images_dir = test_wrk.joinpath("images")
        n_dirs = len([f for f in images_dir.iterdir() if f.is_dir()])
        n_images = []
        img_dir_names = []
        for f in images_dir.iterdir():
            if f.is_dir:
                n_images.append(len([i for i in f.iterdir()]))
                img_dir_names.append(f.name)
        d_true = {"n_images": 2, "GEOM": 88, "Mesh": 88}
        d_test = {"n_images": len(n_images)}
        for i in range(n_dirs):
            d_test[img_dir_names[i]] = n_images[i]
        shutil.rmtree(test_wrk)
        self.assertDictEqual(d_true, d_test)


class TestPrepDirsForAnimation(unittest.TestCase):

    def setUp(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
