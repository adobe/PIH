import sys
import os
from optparse import OptionParser
import numpy as np
from tqdm import tqdm
import bart
import h5py
import sigpy as sp


def get_args():
    parser = OptionParser()
    parser.add_option(
        "-l",
        "--load",
        dest="load",
        default=False,
        help="Folder directory contains fastMRI dataset (h5 format)",
    )
    parser.add_option(
        "-t",
        "--target",
        dest="target",
        default=False,
        help="Target folder directory for processed data",
    )

    parser.add_option(
        "-c",
        "--calibration",
        dest="calibration",
        default=24,
        type="int",
        help="Size of the calibration region for espirit maps",
    )
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    args = get_args()
    train_folder = args.load
    target_folder = args.target

    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    filenames = os.listdir(train_folder)
    tqdm_bar = tqdm(filenames)
    case = 0
    for name in tqdm_bar:
        f = h5py.File(train_folder + "/" + name, "r")
        kspace = np.array(f["kspace"])
        case += 1
        coil_num = kspace.shape[1]
        tqdm_bar.set_description(
            "the last dimension %d, coil dimensional: %d" % (kspace.shape[-1], coil_num)
        )
        with h5py.File(train_folder + name, "r") as hf:
            acquisition = hf.attrs["acquisition"]
        kspace_low_res = sp.resize(
            sp.resize(kspace, (16, coil_num, 640, 24)), (16, coil_num, 640, 372)
        )
        kspace = sp.resize(kspace, (16, coil_num, 640, 372))

        # Normalize the data with 95% percentile
        im_lowres = sp.rss(sp.ifft(kspace_low_res, axes=(-1, -2)), axes=1)
        magnitude_vals = im_lowres.reshape(-1)
        k = int(round(0.05 * magnitude_vals.shape[0]))
        scale = magnitude_vals[magnitude_vals.argsort()[::-1][k]]
        kspace_norm = kspace / scale

        for sli in range(16):
            kspace_slice = kspace_norm[sli, ...]  # save ksp
            sens = bart.bart(
                1,
                "ecalib -r %d -m1" % (args.calibration),
                kspace_slice.transpose(1, 2, 0)[None, ...],
            )[0, ...].transpose(2, 0, 1)
            im_bart = bart.bart(
                1,
                "pics -i 1 -S",
                kspace_slice.transpose(1, 2, 0)[..., None, :],
                sens.transpose(1, 2, 0)[..., None, :],
            )
            h5f = h5py.File("%s/%d_%d.h5" % (target_folder, case, sli), "w",)
            h5f.create_dataset("acquisition", data=acquisition)
            h5f.create_dataset("filename", data=name)
            h5f.create_dataset("kspace", data=kspace_slice)
            h5f.create_dataset("sensmaps", data=sens)
            h5f.create_dataset("target", data=im_bart)
            h5f.close()
