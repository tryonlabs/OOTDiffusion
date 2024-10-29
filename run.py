import argparse
import os
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()

os.environ["VIT_PATH"] = f"{PROJECT_ROOT}/checkpoints/clip-vit-large-patch14"
os.environ["VAE_PATH"] = f"{PROJECT_ROOT}/checkpoints/ootd"
os.environ["UNET_PATH"] = f"{PROJECT_ROOT}/checkpoints/ootd/ootd_hd/checkpoint-36000"
os.environ["MODEL_PATH"] = f"{PROJECT_ROOT}/checkpoints/ootd"

from ootd.inference_ootd_dc import OOTDiffusionDC
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.utils import get_mask_location
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose


def run_ootd(model_path, cloth_path, output_path, gpu_id=0, model_type="hd", category=0, image_scale=2.0,
             n_steps=20, n_samples=4, seed=-1):
    openpose_model = OpenPose(gpu_id)
    parsing_model = Parsing(gpu_id)

    category_dict = ['upperbody', 'lowerbody', 'dress']
    category_dict_utils = ['upper_body', 'lower_body', 'dresses']

    if model_type == "hd":
        model = OOTDiffusionHD(gpu_id)
    elif model_type == "dc":
        model = OOTDiffusionDC(gpu_id)
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")

    if model_type == 'hd' and category != 0:
        raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")

    cloth_img = Image.open(cloth_path).resize((768, 1024))
    model_img = Image.open(model_path).resize((768, 1024))
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))

    mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

    os.makedirs(output_path, exist_ok=True)

    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    masked_vton_img.save(os.path.join(output_path, 'mask.jpg'))

    images = model(
        model_type=model_type,
        category=category_dict[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=n_samples,
        num_steps=n_steps,
        image_scale=image_scale,
        seed=seed,
    )

    image_idx = 0
    for image in images:
        image.save(os.path.join(output_path, 'out_' + model_type + '_' + str(image_idx) + '.png'))
        image_idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run ootd')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
    parser.add_argument('--model_path', type=str, default="", required=True)
    parser.add_argument('--cloth_path', type=str, default="", required=True)
    parser.add_argument('--output_path', type=str, default="", required=True)
    parser.add_argument('--model_type', type=str, default="hd", required=False)
    parser.add_argument('--category', '-c', type=int, default=0, required=False)
    parser.add_argument('--image_scale', type=float, default=2.0, required=False)
    parser.add_argument('--n_steps', type=int, default=20, required=False)
    parser.add_argument('--n_samples', type=int, default=4, required=False)
    parser.add_argument('--seed', type=int, default=-1, required=False)
    args = parser.parse_args()

    print(f"args: {args}")

    run_ootd(model_path=args.model_path, cloth_path=args.cloth_path, output_path=args.output_path,
             gpu_id=args.gpu_id, model_type=args.model_type, category=args.category, image_scale=args.image_scale,
             n_steps=args.n_steps, n_samples=args.n_samples, seed=args.seed)
