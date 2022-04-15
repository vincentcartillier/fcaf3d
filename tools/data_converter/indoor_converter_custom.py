import mmcv
import numpy as np
import os

from tools.data_converter.sunrgbd_data_utils_custom import SUNRGBDData


def create_indoor_info_file(data_path,
                            pkl_prefix='sunrgbd',
                            save_path=None,
                            use_v1=False,
                            workers=4):
    """Create indoor information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str): Prefix of the pkl to be saved. Default: 'sunrgbd'.
        save_path (str): Path of the pkl to be saved. Default: None.
        use_v1 (bool): Whether to use v1. Default: False.
        workers (int): Number of threads to be used. Default: 4.
    """
    assert os.path.exists(data_path)
    assert pkl_prefix in ['sunrgbd'], \
        f'unsupported indoor dataset {pkl_prefix}'
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)

    # generate infos for both detection and segmentation task
    if pkl_prefix in ['sunrgbd']:
        train_filename = os.path.join(save_path,
                                      f'{pkl_prefix}_infos_train_wfeat.pkl')
        val_filename = os.path.join(save_path,
                                    f'{pkl_prefix}_infos_val_wfeat.pkl')
        if pkl_prefix == 'sunrgbd':
            # SUN RGB-D has a train-val split
            train_dataset = SUNRGBDData(
                root_path=data_path,
                split='train',
                use_v1=use_v1
            )
            val_dataset = SUNRGBDData(
                root_path=data_path,
                split='val',
                use_v1=use_v1
            )

        infos_train = train_dataset.get_infos(
            num_workers=workers,
            has_label=True)
        mmcv.dump(infos_train,
                  train_filename,
                  'pkl')
        print(f'{pkl_prefix} info train file is saved to {train_filename}')

        infos_val = val_dataset.get_infos(
            num_workers=workers,
            has_label=True)
        mmcv.dump(infos_val,
                  val_filename,
                  'pkl')
        print(f'{pkl_prefix} info val file is saved to {val_filename}')


