import os
import h5py
import traceback
from pathlib import Path
import sys

sys.path.append("..")
from preprocess import io_
from preprocess.preprocess_utils import *
import numpy as np
# from utils1.visualize import show_graphs


def split_file_path(root, data_list_path, train_lab_num, train_unlab_num, test_num, shuffle=False):
    Path(data_list_path).mkdir(exist_ok=True)

    def save_path(paths, filename):
        with open(filename, 'w') as f:
            for p in paths:
                f.write(p + '\n')
        return filename

    root, data_list_path = Path(root), Path(data_list_path)
    files = []
    for f in root.iterdir():
        files.append(f.name)

    files.sort(key=lambda x: int(x.split('.')[0]))

    if test_num is not None:
        assert len(files) == (train_lab_num + train_unlab_num + test_num), 'Total_files : {}, current files : {}'.format(
            len(files), train_lab_num + train_unlab_num + test_num)
    else:
        test_num = len(files) - (train_lab_num + train_unlab_num)

    if shuffle:
        np.random.shuffle(files)

    train_lab_paths = files[:train_lab_num]
    train_unlab_paths = files[train_lab_num:train_lab_num + train_unlab_num]
    test_paths = files[-test_num:]
    print('Generated labeled train {}, unlabeled train {}, test {}'.format(len(train_lab_paths), len(train_unlab_paths), len(test_paths)))
    return save_path(train_lab_paths, data_list_path / 'train_lab.txt'), \
           save_path(train_unlab_paths, data_list_path / 'train_unlab.txt'), \
           save_path(test_paths, data_list_path / 'test.txt')


def normalize(data):
    # normalized_data = (data - data.mean()) / (data.std() + 1e-10)
    normalized_data = (data - data.min()) / (data.max() - data.min())
    normalized_data = normalized_data  # * 2 - 1
    return normalized_data


def save_to_h5(img, mask, filename):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('image', data=img)
    hf.create_dataset('label', data=mask)
    hf.close()


# root = 'data/'
root = '../data/NIH_raw'
save_to = root
DCM_data = True


def process_case(case_folders):
    try:
        print("yes")
        for case_folder in case_folders:
            print("path: ", case_folder)
            if DCM_data:  # if downloaed DCM data
                if not case_folder.is_dir():
                    return
                img = []
                for inner_folder in case_folder.iterdir():
                    for folder in inner_folder.iterdir():
                        folders = list(folder.iterdir())
                        folders.sort()
                        for slice_path in folders:
                            slice, spacing, affine_pre = io_.read_nii(slice_path)
                            img.append(slice)
                img = np.concatenate(img)  # depth x H x W
                print(case_folder)
                case_idx = str(case_folder)[-4:]
                label_path = root / 'Pancreas-CT-Label' / ('label' + case_idx + '.nii.gz')
                img = img.swapaxes(2, 1).swapaxes(1, 0).swapaxes(1, 2)  # make depth last
                mask = io_.read_img(label_path)
            else:  # if downloaed nii data
                img, spacing, affine_pre = io_.read_nii(case_folder)
                print(case_folder)
                case_idx = str(case_folder).split('.')[0][-4:]
                label_path = root / 'label' / ('label' + case_idx + '.nii.gz')
                mask, _, _ = io_.read_nii(label_path)

            assert mask.shape == img.shape, "{}, {}".format(mask.shape, img.shape)

            # show_graphs(img[:, :, 100:116].clip(-125, 275), figsize=(20, 20)), show_graphs(mask[100:116], figsize=(20, 20))

            # resample to [1, 1, 1]
            target_spacing = (1, 1, 1)
            # change spacing of depth
            spacing = (spacing[1], spacing[1], spacing[1])
            affine_pre = io_.make_affine2(spacing)
            resampled_img, affine = resample_volume_nib(img, affine_pre, spacing, target_spacing, mask=False)
            resampled_mask, affine = resample_volume_nib(mask, affine_pre, spacing, target_spacing, mask=True)
            # resampled_img, resampled_mask = img, mask

            # clip to [-125, 275]
            min_clip, max_clip = -125, 275
            resampled_img = resampled_img.clip(min_clip, max_clip)
            resampled_img = normalize(resampled_img)

            # crop image
            bbox = get_bbox_3d(resampled_mask)
            offset = 25
            bbox = expand_bbox(resampled_img, bbox, expand_size=(offset, offset, offset), min_crop_size=(96, 96, 96))
            cropped_img = crop_img(resampled_img, bbox, min_crop_size=(96, 96, 96))
            cropped_mask = crop_img(resampled_mask, bbox, min_crop_size=(96, 96, 96))

            # show_graphs(cropped_img[100:116], figsize=(10, 10)), show_graphs(cropped_mask[100:116], figsize=(10, 10), filename='mask.png')
            save_to_h5(cropped_img, cropped_mask, save_to + case_idx + '.h5')
            print('saved : {}, resampled shape : {}, cropped shape : {}'.format(case_idx, resampled_img.shape, cropped_img.shape))
    except Exception as e:
        print("No")
        print(e)
        # traceback.print_tb(e)
        traceback.print_exc()


def generate_h5_data(original_pancreas_path, save_path):
    global root, save_to
    root = Path(original_pancreas_path)
    path = Path(root) / 'Pancreas-CT'
    save_to = save_path
    paths = list(path.iterdir())
    paths.sort()
    print(paths)
    Path(save_path).mkdir(exist_ok=True)
    # io_.multiprocess_task(process_case, paths, cpu_num=16)
    process_case(paths)


if __name__ == '__main__':
    path_to_save_generated_data = 'data'
    generate_h5_data('../data/NIH', path_to_save_generated_data)
