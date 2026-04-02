import argparse
import copy
from copy import deepcopy
import logging
import os
import shutil

import torch
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import datasets
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
# from image_datasets.control_dataset import loader, image_resize
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import transformers
from PIL import Image
import numpy as np
from optimum.quanto import quantize, qfloat8, freeze
import bitsandbytes as bnb
logger = get_logger(__name__, log_level="INFO")
from diffusers.loaders import AttnProcsLayers
from diffusers import QwenImageEditPipeline
import gc
import math
from typing import List
from dataclasses import dataclass
baseline_momentum = 0.9
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config

import torch
from torch.utils.data import Dataset, DataLoader

# 通过HTTP调用SAM3服务器，避免环境冲突
from sam3_client import process_images

def iou_reward_with_sam(image1, image2, object):
    try:
        iou = process_images(image1, image2, object, server_url="http://127.0.0.1:5000")
        return iou
    except Exception as e:
        print("please start SAM3 server: python sam3_http_server.py")
        raise


class ToyDataset(Dataset):
    def __init__(self, num_samples=100, input_dim=10):
        self.data = torch.randn(num_samples, input_dim)    # random features
        self.labels = torch.randint(0, 2, (num_samples,))  # random labels: 0 or 1

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

def lora_processors(model):
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
        if 'lora' in name:
            processors[name] = module
            print(name)
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

        return processors

    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None
def tensor_to_pil_rgb_uint8(img_chw: torch.Tensor) -> Image.Image:
    with torch.no_grad():
        arr = img_chw.detach().cpu().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(arr, mode="RGB")
def preprocess_for_vae(path: str):
    img = Image.open(path).convert("RGB")
    w, h = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
    img = img.resize((w, h), Image.BICUBIC)
    arr = (np.asarray(img).astype(np.float32) / 127.5) - 1.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    return x, img
def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()
def collate_fn(batch):
    return {
        "target_image_path": [x["target_image_path"] for x in batch],
        "control_image_path": [x["control_image_path"] for x in batch],
        "text": [x["text"] for x in batch],
        "image_stem": [x["image_stem"] for x in batch],
    }
@dataclass
class Sample:
    target_image_path: str
    control_image_path: str
    text_path: str
    image_stem: str
class EditDataset(Dataset):
    def __init__(self, img_dir: str, control_dir: str):
        self.samples: List[Sample] = []
        all_images = [p for p in os.listdir(img_dir) if p.lower().endswith((".png", ".jpg", ".jpeg"))]
        all_images.sort()

        for name in all_images:
            stem = os.path.splitext(name)[0]
            txt = os.path.join(img_dir, f"{stem}.txt")
            tgt = os.path.join(img_dir, name)
            ctl = os.path.join(control_dir, name)
            if os.path.exists(txt) and os.path.exists(ctl):
                self.samples.append(Sample(tgt, ctl, txt, stem))

        if len(self.samples) == 0:
            raise ValueError("No valid samples found. Need paired target/control image and target caption txt.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "target_image_path": s.target_image_path,
            "control_image_path": s.control_image_path,
            "text": read_text(s.text_path),
            "image_stem": s.image_stem,
        }
def main():
    args = OmegaConf.load(parse_args())
    args.save_cache_on_disk = False
    args.precompute_text_embeddings = True
    args.precompute_image_embeddings = True

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    text_encoding_pipeline = QwenImageEditPipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    text_encoding_pipeline.to(accelerator.device)
    cached_text_embeddings = None
    txt_cache_dir = None

    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    cached_image_embeddings = None
    img_cache_dir = None
    cached_image_embeddings_control = None
    # del text_encoding_pipeline
    gc.collect()
    #del vae
    gc.collect()
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",    )
    if args.quantize:
        torch_dtype = weight_dtype
        device = accelerator.device
        all_blocks = list(flux_transformer.transformer_blocks)
        for block in tqdm(all_blocks):
            block.to(device, dtype=torch_dtype)
            quantize(block, weights=qfloat8)
            freeze(block)
            block.to('cpu')
        flux_transformer.to(device, dtype=torch_dtype)
        quantize(flux_transformer, weights=qfloat8)
        freeze(flux_transformer)
        #quantize(flux_transformer, weights=qint8, activations=qint8)
        #freeze(flux_transformer)
        
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    flux_transformer.to(accelerator.device)
    #flux_transformer.add_adapter(lora_config)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    if args.quantize:
        flux_transformer.to(accelerator.device)
    else:
        flux_transformer.to(accelerator.device, dtype=weight_dtype)
    flux_transformer.add_adapter(lora_config)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    flux_transformer.requires_grad_(False)

    flux_transformer.train()
    optimizer_cls = torch.optim.AdamW
    for n, param in flux_transformer.named_parameters():
        if 'lora' not in n:
            param.requires_grad = False
            pass
        else:
            param.requires_grad = True
            print(n)
    print(sum([p.numel() for p in flux_transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')
    lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())
    lora_layers_model = AttnProcsLayers(lora_processors(flux_transformer))
    flux_transformer.enable_gradient_checkpointing()
    if args.adam8bit:
        optimizer = bnb.optim.Adam8bit(lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),)
    else:
        optimizer = optimizer_cls(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    dataset1 = ToyDataset(num_samples=100, input_dim=10)
    dataloader1 = DataLoader(dataset1, batch_size=8, shuffle=True)

    dataset = EditDataset(args.data_config.img_dir, args.data_config.control_dir)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    lora_layers_model, optimizer, _, lr_scheduler = accelerator.prepare(
        lora_layers_model, optimizer, dataloader1, lr_scheduler
    )

    initial_global_step = 0

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    reward_baseline = 0.0
    fst_flag = True
    for epoch in range(1):
        train_loss = 0.0
        for step, batch in enumerate(dataloader):
            prompt_embeds_list = []
            prompt_mask_list = []
            txt_seq_lens = []

            cached_text_embeddings = []
            cached_text_empty_embeddings = []
            cached_image_embeddings = []
            cached_image_embeddings_control = []

            prompts = batch['text']
            control_imgs = batch['control_image_path']
            imgs = batch['target_image_path']
            img_names = batch['image_stem']
            with torch.no_grad():
                # txt processing
                for control_img, prompt in zip(control_imgs, prompts):
                    control_img_pli = Image.open(control_img).convert('RGB')
                    calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, control_img_pli.size[0] / control_img_pli.size[1])
                    prompt_image = text_encoding_pipeline.image_processor.resize(control_img_pli, calculated_height, calculated_width)

                    prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                        image=prompt_image,
                        prompt=[prompt],
                        device=text_encoding_pipeline.device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024,
                    )
                    cached_text_embeddings.append({'prompt_embeds': prompt_embeds[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')})#################
                    prompt_embeds_empty, prompt_embeds_mask_empty = text_encoding_pipeline.encode_prompt(
                        image=prompt_image,
                        prompt=[' '],
                        device=text_encoding_pipeline.device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024,
                    )
                    cached_text_empty_embeddings.append({'prompt_embeds': prompt_embeds_empty[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask_empty[0].to('cpu')})#################

                for img in imgs:
                    img_pli = Image.open(img).convert('RGB')
                    calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, img_pli.size[0] / img_pli.size[1])
                    img_lat = text_encoding_pipeline.image_processor.resize(img_pli, calculated_height, calculated_width)
                    img_lat = torch.from_numpy((np.array(img_lat) / 127.5) - 1)
                    pixel_values = img_lat.permute(2, 0, 1).unsqueeze(2)
                    pixel_values = pixel_values.to(dtype=weight_dtype).to(accelerator.device)
                    pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
                    cached_image_embeddings.append(pixel_latents)#################
                for control_img in control_imgs:
                    control_img_pli = Image.open(control_img).convert('RGB')
                    calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, control_img_pli.size[0] / control_img_pli.size[1])
                    control_lat = text_encoding_pipeline.image_processor.resize(control_img_pli, calculated_height, calculated_width)
                    control_lat = torch.from_numpy((np.array(control_lat) / 127.5) - 1)
                    pixel_values_control = control_lat.permute(2, 0, 1).unsqueeze(2)
                    pixel_values_control = pixel_values_control.to(dtype=weight_dtype).to(accelerator.device)
                    pixel_latents_control = vae.encode(pixel_values_control).latent_dist.sample().to('cpu')[0]
                    cached_image_embeddings_control.append(pixel_latents_control)#################

            with accelerator.accumulate(flux_transformer):
                prompt_embeds = torch.stack([e['prompt_embeds'] for e in cached_text_embeddings]).to(dtype=weight_dtype).to(accelerator.device)
                prompt_embeds_mask = torch.stack([e['prompt_embeds_mask'] for e in cached_text_embeddings]).to(dtype=torch.int32).to(accelerator.device)
                control_img = torch.stack(cached_image_embeddings_control).to(dtype=weight_dtype).to(accelerator.device)
                img = torch.stack(cached_image_embeddings).to(dtype=weight_dtype).to(accelerator.device)
                with torch.no_grad():
                    pixel_latents = img.to(dtype=weight_dtype).to(accelerator.device)

                    pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
                    control_img = control_img.permute(0, 2, 1, 3, 4)
                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, 1, vae.config.z_dim, 1, 1)
                        .to(pixel_latents.device, pixel_latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype
                    )
                    pixel_latents = (pixel_latents - latents_mean) * latents_std
                    control_img = (control_img - latents_mean) * latents_std

                    bsz = pixel_latents.shape[0]
                    noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme="none",
                        batch_size=bsz,
                        logit_mean=0.0,
                        logit_std=1.0,
                        mode_scale=1.29,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                # Concatenate across channels.
                # pack the latents.
                packed_noisy_model_input = QwenImageEditPipeline._pack_latents(
                    noisy_model_input,
                    bsz, 
                    noisy_model_input.shape[2],
                    noisy_model_input.shape[3],
                    noisy_model_input.shape[4],
                )
                packed_control_img = QwenImageEditPipeline._pack_latents(
                    control_img,
                    bsz, 
                    control_img.shape[2],
                    control_img.shape[3],
                    control_img.shape[4],
                )
                # latent image ids for RoPE.
                img_shapes = [[(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                              (1, control_img.shape[3] // 2, control_img.shape[4] // 2)]] * bsz
                packed_noisy_model_input_concated = torch.cat([packed_noisy_model_input, packed_control_img], dim=1)
                with torch.no_grad():
                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input_concated,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed_noisy_model_input.size(1)]

                model_pred = QwenImageEditPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                # flow-matching loss
                target = noise - pixel_latents
                target = target.permute(0, 2, 1, 3, 4)
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                # ===============Reward from IoU==================
                with torch.no_grad():
                    eps = 1e-5
                    pred_x0 = (noisy_model_input - sigmas * model_pred.permute(0, 2, 1, 3, 4)) / torch.clamp(1.0 - sigmas, min=eps)
                    pred_x0 = pred_x0 / latents_std + latents_mean

                    decoded = vae.decode(pred_x0.to(dtype = weight_dtype)).sample
                    decoded = (decoded / 2 + 0.5).clamp(0, 1) * 255.0

                    rs = []
                    for i in range(bsz):
                        pred_pil = tensor_to_pil_rgb_uint8(decoded[i])
                        obj = "robotic arm"
                        tgt_img_pli = Image.open(imgs[i]).convert('RGB')
                        if fst_flag:
                            pred_pil.save("pred_pil.png")
                            tgt_img_pli.save("tgt_img_pli.png")
                            fst_flag = False
                        r = iou_reward_with_sam(pred_pil, tgt_img_pli[i], obj)
                        if r > 0.9:
                            r = 1.0
                        rs.append(r)
                    rs_tensor = torch.tensor(rs, device=loss.device, dtype=loss.dtype)
                    reward_mean = float(rs_tensor.mean().item())
                    adv = reward_mean
                    # reward_baseline = baseline_momentum * reward_baseline + (1.0 - baseline_momentum) * reward_mean
                    # adv = torch.relu(rs_tensor - reward_baseline).detach()
                loss = loss + (1.0 - adv) * 0.05 * loss
                # ===============Reward from IoU==================
                loss = loss.mean()
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    #accelerator.save_state(save_path)
                    try:
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                    except:
                        pass
                    unwrapped_flux_transformer = unwrap_model(flux_transformer)
                    flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_flux_transformer)
                    )

                    QwenImagePipeline.save_lora_weights(
                        save_path,
                        flux_transformer_lora_state_dict,
                        safe_serialization=True,
                    )

                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
