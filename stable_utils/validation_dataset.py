import gc
import torch
import numpy as np
import wandb
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

from datasets import load_dataset
from stable_utils.flatdata_utils import demosaic_raw, Edata_demosaic_raw2, Edata_demosaic_raw_fullsize
from torchvision import transforms
import random
import os
from pipelines.pipeline_flatcontrolnet import FlatStableDiffusionControlNetPipeline
import json
import yaml

from stable_utils.evaluate_saved import eval_saved

def dict_representor_yaml(dict_in):
    out = {}
    for k,v in dict_in.items():
        data = v.tolist() if isinstance(v, (np.ndarray, torch.Tensor)) else v
        out[k] = data
    return out

import skimage
import skimage.transform
from functools import partial

def make_tag_dataset(args, tokenizer, accelerator, logger, data_tag='train'):
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split=data_tag
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
            )
        raise NotImplementedError # not supported in flat dataset
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset.column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts and is_train:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    totensor = transforms.ToTensor()

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image for image in examples[conditioning_image_column]]
        conditioning_images = [totensor(np.array(meas)) for meas in conditioning_images]
        conditioning_images = [demosaic_raw(meas) for meas in conditioning_images]
        conditioning_images = [torch.from_numpy(meas) for meas in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        if tokenizer is not None:
            examples["input_ids"] = tokenize_captions(examples)
        else:
            examples["input_ids"] = [caption for caption in examples[caption_column]]

        return examples

    class Edata_preprocess_train:
        def __init__(self, config_name='default', real_cap=False):
            self.config_name = config_name
            self.real_cap = real_cap
            if 'fullsize' in self.config_name or 'default' in self.config_name:
                self.tform = skimage.transform.SimilarityTransform(rotation=-0.009)
                i_black = np.load('ib_mean20.npy')
                self.ib = i_black / 4095


        def __call__(self, examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            images = [image_transforms(image) for image in images]

            conditioning_images = [image for image in examples[conditioning_image_column]]
            conditioning_images = [np.array(meas) for meas in conditioning_images]
            if 'small' in self.config_name:
                conditioning_images = [Edata_demosaic_raw2(meas) for meas in conditioning_images]
            else:
                conditioning_images = [Edata_demosaic_raw_fullsize(meas, self.tform, self.ib, real_cap=self.real_cap) for meas in conditioning_images]
            conditioning_images = [torch.from_numpy(meas) for meas in conditioning_images]

            examples["pixel_values"] = images
            examples["conditioning_pixel_values"] = conditioning_images
            #examples["input_ids"] = [caption for caption in examples[caption_column]]
            if tokenizer is not None:
                examples["input_ids"] = tokenize_captions(examples)
            else:
                examples["input_ids"] = [caption for caption in examples[caption_column]]


            return examples

    if args.dataset_name == 'flatcam_Edata_caps':
        preprocess_train = Edata_preprocess_train(config_name=args.dataset_config_name)
    elif args.dataset_name == 'flatcam_data_real':
        preprocess_train = Edata_preprocess_train(config_name=args.dataset_config_name, real_cap=True)


    with accelerator.main_process_first():
            # partial(Edata_preprocess_train, config_name=args.dataset_config_name)

        if args.max_train_samples is not None and data_tag == 'train':
            if args.train_shuffle_seed is not None:
                dataset = dataset.shuffle(seed=args.train_shuffle_seed).select(range(args.max_train_samples))
            else:
                dataset = dataset.select(range(args.max_train_samples))

        if args.max_val_samples is not None and data_tag == 'validation':
            if args.val_shuffle_seed is not None:
                dataset = dataset.shuffle(seed=args.val_shuffle_seed).select(range(args.max_val_samples)) # trim_len
            else:
                dataset = dataset.select(range(args.max_val_samples)) # trim_len

        # Set the training transforms
        tag_dataset = dataset.with_transform(preprocess_train)

    return tag_dataset

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = [example["input_ids"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, global_step, logger,
                   tag_dataset=None):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)
    controlnet.eval()
    pipeline = FlatStableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []
    if tag_dataset is None:
        tag_dataset = make_tag_dataset(args, None, accelerator, logger, data_tag='validation')
    tag_dataloader = torch.utils.data.DataLoader(
        tag_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )
    for step, batch in enumerate(tag_dataloader):
        validation_image = batch["conditioning_pixel_values"]
        validation_prompt = batch["input_ids"]
        validation_target = batch["pixel_values"]
        images = []

        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    validation_prompt, validation_image, num_inference_steps=20, generator=generator, guidance_scale=1
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images,
             "validation_prompt": validation_prompt, "validation_target": validation_target}
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            pass
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            log_dict = {"validation": formatted_images}
            tracker.log(log_dict, step=global_step)
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    controlnet.train()
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs

def log_validation_save_all(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, global_step, logger):
    logger.info("Running validation... ")

    dir_to_save = os.path.join(args.output_dir, 'save_results')
    os.makedirs(dir_to_save, exist_ok=True)
    controlnet = accelerator.unwrap_model(controlnet)
    controlnet.eval()
    pipeline = FlatStableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []
    tag_dataset = make_tag_dataset(args, None, accelerator, logger, data_tag='validation')

    tag_dataloader = torch.utils.data.DataLoader(
        tag_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )
    metrics_dict={}

    for step, batch in enumerate(tag_dataloader):
        validation_image = batch["conditioning_pixel_values"]
        validation_prompt = batch["input_ids"]
        validation_target = batch["pixel_values"]
        images = []
        images_pt = []


        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    validation_prompt, validation_image, num_inference_steps=args.num_inference_steps, generator=generator, output_type='pt',
                    guidance_scale=args.guidance_scale,
                ).images # batch (size 1)
            image_pil = pipeline.image_processor.pt_to_numpy(image) #batch1
            image_pil = pipeline.image_processor.numpy_to_pil(image_pil)[0] #get first image

            image = pipeline.image_processor.normalize(image).cpu().float()
            images_pt.append(image)
            images.append(image_pil)
            metrics_dict = evaluator(image, validation_target, metrics_dict)
        validation_target = validation_target.float().cpu()
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                logger.warn('warning - tensorboard is disabled (print from validation function)')
            elif tracker.name == "wandb":
                formatted_images = []
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

                log_dict = {"validation": formatted_images}
                tracker.log(log_dict, step=step)
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")

        img_path = os.path.join(dir_to_save, f'sample{step:03d}.png')
        for i, image in enumerate(images):
            image.save(img_path.replace('.png', f'_{i}.png'))

        torch.save(torch.cat(images_pt, dim=0), img_path.replace('.png', '.pt'))

        gt_path = os.path.join(dir_to_save, f'gt{step:03d}.png')
        torch.save(validation_target, gt_path.replace('.png', '.pt'))
    logger.info(f'saved_to: {gt_path}')
    controlnet.train()
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    metrics_dict['psnr'] = -10 * torch.log10(metrics_dict['mse'])
    with open(os.path.join(args.output_dir, f'metrics_dict.yaml'), 'w') as outfile:
        yaml.dump(dict_representor_yaml(metrics_dict), outfile, indent=4)

    metrics_dict = eval_saved(args.output_dir)
    metrics_to_wandb = {}
    for k, v in metrics_dict.items():
        if not isinstance(v, dict):
            metrics_to_wandb[k] = v

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(metrics_to_wandb, step=step+1)

    return image_logs

def evaluator(pred, gt, metrics_dict):
    mse_loss = torch.nn.MSELoss()
    mse_prev = metrics_dict.get('mse', 0)
    mse_counter = metrics_dict.get('mse_c', 0)
    mse_item = mse_loss(pred, gt) * pred.shape[0] / 4
    metrics_dict['mse'] = (mse_prev * mse_counter + mse_item) / (mse_counter + 1)
    metrics_dict['mse_c'] = mse_counter + 1

    return metrics_dict

