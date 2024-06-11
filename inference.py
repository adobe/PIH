# Copyright 2023 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import itertools
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.functional_pil as F_pil
from PIL import Image
from tqdm import tqdm

from model import Model_Composite_PL


def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "bg", type=str, help="Directory or Path to the background image(s)."
    )
    parser.add_argument(
        "fg", type=str, help="Directory or Path to the foreground image(s)."
    )
    parser.add_argument(
        "--fg_mask",
        type=str,
        default=None,
        help="Directory or Path to the foreground mask image(s).",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Path to export results.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrained/ckpt_g39.pth",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU -> cuda, CPU -> cpu",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="If specified, will use light model",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="If specified and in GPU inference, can reproduce the results of paper",
    )

    args = vars(parser.parse_args())
    return args


def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def resize_and_pad(
    img: Image.Image, size: Tuple[int, int], value: int = 0
) -> Image.Image:
    target_w, target_h = size
    w, h = img.size

    # resize the longest edge
    target_ratio = target_w / target_h
    ratio = w / h
    if target_ratio >= ratio:
        size_ = (round(ratio * target_h), target_h)
    else:
        size_ = (target_w, round(target_w / ratio))

    img = img.resize(size_, resample=Image.Resampling.LANCZOS)

    # pad value to size (the shortest edge)
    w, h = img.size
    pad_w = (target_w - w) // 2
    pad_h = (target_h - h) // 2
    # [left, top, right, bottom]
    padding_ltrb = [pad_w, pad_h, target_w - w - pad_w, target_h - h - pad_h]

    img = F_pil.pad(img, padding_ltrb, fill=value)
    assert img.size == size
    return img


class Inferencer:
    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda:0",
        light: bool = False,
        deterministic: bool = False,
    ):
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.model = Model_Composite_PL(
            dim=32,
            masking=True,
            brush=True,
            maskoffset=0.6,
            swap=True,
            Vit_bool=False,
            onlyupsample=True,
            aggupsample=True,
            light=light,
            Eff_bool=light,
        ).to(device)

        checkpoint = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.transforms_mask = T.Compose([T.Grayscale(), T.ToTensor()])
        self.transform = T.Compose([T.ToTensor()])
        self.resize = T.Resize((512, 512))

        self.device = device

    def preprocess(
        self,
        bg_paths: List[Union[str, Path]],
        fg_paths: List[Union[str, Path]],
        fg_mask_paths: Optional[List[Union[str, Path]]] = None,
    ):
        if fg_mask_paths is None:
            fg_mask_paths = [None] * len(fg_paths)

        for fg_path, fg_mask_path in zip(fg_paths, fg_mask_paths):
            fg_path = Path(fg_path)
            fg_img = Image.open(fg_path)
            if fg_mask_path is None:
                assert fg_img.mode == "RGBA"
                fg_mask = fg_img.getchannel("A")
            else:
                fg_mask = Image.open(fg_mask_path).convert("1")
            fg_img = fg_img.convert("RGB")

            for bg_path in bg_paths:
                bg_path = Path(bg_path)
                bg_img = Image.open(bg_path).convert("RGB")

                resized_fg_img = resize_and_pad(fg_img, bg_img.size)
                resized_fg_mask = resize_and_pad(fg_mask, bg_img.size)
                composite_img = Image.composite(resized_fg_img, bg_img, resized_fg_mask)

                bg_low = self.resize(bg_img)
                composite_low = self.resize(composite_img)
                mask_low = self.resize(resized_fg_mask)

                torch_bg = self.transform(bg_img)
                torch_composite = self.transform(composite_img)
                torch_mask = self.transforms_mask(resized_fg_mask)

                torch_bg_low = self.transform(bg_low)
                torch_composite_low = self.transform(composite_low)
                torch_mask_low = self.transforms_mask(mask_low)

                yield (
                    bg_path,
                    fg_path,
                    composite_img,
                    resized_fg_mask,
                    torch_bg,
                    torch_composite,
                    torch_mask,
                    torch_bg_low,
                    torch_composite_low,
                    torch_mask_low,
                )

    def to(self, inputs: Sequence, device: str):
        return [
            data.to(device) if isinstance(data, torch.Tensor) else data
            for data in inputs
        ]

    def forward(
        self,
        torch_bg: torch.Tensor,
        torch_composite: torch.Tensor,
        torch_mask: torch.Tensor,
        torch_bg_low: torch.Tensor,
        torch_composite_low: torch.Tensor,
        torch_mask_low: torch.Tensor,
    ):
        with torch.no_grad():
            inter_composite, output_composite, par1, par2 = self.model(
                torch_bg_low.unsqueeze(0),
                torch_composite_low.unsqueeze(0),
                torch_mask_low.unsqueeze(0),
            )

        hr_intermediate = (
            self.model.PL3D(self.model.pl_table, torch_composite.unsqueeze(0))
            * torch_mask
            + (1 - torch_mask) * torch_bg
        )

        gainmap = F.resize(self.model.gainmap, torch_bg.shape[-2:])

        output_results = (
            hr_intermediate * gainmap * torch_mask + (1 - torch_mask) * torch_bg
        )
        output_gainmap = gainmap * torch_mask
        return output_results, output_gainmap, par2

    def visualize(
        self,
        bg_path: Path,
        fg_path: Path,
        composite: Image.Image,
        mask: Image.Image,
        results: torch.Tensor,
        gainmap: torch.Tensor,
        curves: torch.Tensor,
        out_dir: Path = Path("results"),
    ):
        out_dir.mkdir(parents=True, exist_ok=True)

        name_cat = f"{fg_path.stem}_{bg_path.stem}"

        output_lr = F.to_pil_image(results[0])
        output_lr.save(out_dir / f"{name_cat}_final.png")

        output_gm = F.to_pil_image(gainmap[0])
        output_gm.save(out_dir / f"{name_cat}_gainmap.png")

        #### Save Fig

        # curves = curves.cpu().numpy()

        # red_curve = curves[0, 0, 0, 0, :]
        # green_curve = curves[0, 1, 0, :, 0]
        # blue_curve = curves[0, 2, :, 0, 0]

        # plt.figure()
        # plt.plot(np.linspace(0, 1, 32), red_curve, "r")
        # plt.plot(np.linspace(0, 1, 32), green_curve, "g")
        # plt.plot(np.linspace(0, 1, 32), blue_curve, "b")
        # plt.ylim(0, 1)
        # plt.legend(["Reg", "Green", "Blue"])
        # plt.title("Learned Color Curves")

        # plt.savefig(out_dir / f"{name_cat}_color.jpg")

        # plt.close()

        im_final = get_concat_h(composite, get_concat_h(mask, output_lr))

        im_final.save(out_dir / f"{name_cat}_results_summary.png")

    def __call__(
        self,
        bg_paths: List[Union[str, Path]],
        fg_paths: List[Union[str, Path]],
        fg_mask_paths: Optional[List[Union[str, Path]]] = None,
        out_dir: Path = Path("results"),
    ):
        inputs = self.preprocess(bg_paths, fg_paths, fg_mask_paths)
        for data in tqdm(inputs, desc="Processing..."):
            data = self.to(data, self.device)
            ouputs = data[:4] + list(self.forward(*data[4:]))
            self.visualize(*ouputs, out_dir=out_dir)


def collect_images(
    path: Optional[str],
):
    if path is None:
        return None

    root_path = Path(path)
    if root_path.is_dir():
        paths = sorted(
            itertools.chain(root_path.rglob("*.jpg"), root_path.rglob("*.png"))
        )
    else:
        paths = [root_path]

    return paths


if __name__ == "__main__":
    args = get_args()
    bg_paths = collect_images(args.pop("bg"))
    fg_paths = collect_images(args.pop("fg"))
    fg_mask_paths = collect_images(args.pop("fg_mask"))
    out_dir = Path(args.pop("out_dir"))

    evaluater = Inferencer(**args)
    evaluater(bg_paths, fg_paths, fg_mask_paths=fg_mask_paths, out_dir=out_dir)
