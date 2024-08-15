import torch
# TODO: Update the dataset so that it returns only the specific examples for each task
# TODO: Implement a new training function for the diffuser with the FSDM repository as guidance
# TODO: Create the Relational network based on the Learning to Compare: Relation Network for Few-Shot Learning paper.
# TODO: Create a training function for the Relational network and it also might need a new dataset too.
# TODO: Create a training function for the Refinement model.
import argparse
import sys
import os

module_path = "D:/Praetorian-ARC-AGI/FSDM"
sys.path.append(module_path)
from FSDM.model import select_model
from FSDM.model.set_diffusion import logger
from FSDM.model.set_diffusion.resample import create_named_schedule_sampler
from FSDM.model.set_diffusion.script_util import (add_dict_to_argparser,
                                                args_to_dict,
                                                create_model_and_diffusion,
                                                model_and_diffusion_defaults)
from FSDM.model.set_diffusion.train_util import TrainLoop
from FSDM.utils.util import count_params, set_seed
from FSDM.utils.path import set_folder
from diffusion_dataset import DiffusionDataset
import torch.utils.data as data


import argparse

DIR = "D:/Praetorian-ARC-AGI"
def create_argparser(**kwargs):
    defaults = dict(
        model='ddpm',
        dataset='cifar100',
        image_size=30,
        patch_size=2,
        hdim=256,
        in_channels=1,
        encoder_mode='vit',
        pool='mean',  # mean, mean_patch
        context_channels=256,
        mode_context="deterministic",
        mode_conditioning='film',  # conditions using film, lag conditions using attention, None standard DDPM, film+lag
        augment=False,
        device="cuda",
        data_dir="/home/gigi/ns_data",
        schedule_sampler="uniform",
        num_classes=1,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        batch_size_eval=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        clip_denoised=True,
        use_ddim=False,
        tag=None,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(kwargs)
    
    parser = argparse.ArgumentParser()

    for key, value in defaults.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    
    return parser


def diffusion_train(dataset, sample_size):
    # Simulate passing command-line arguments
    args_list = ["--device", "cpu"]
    
    # Use the argument list to parse the arguments
    args = create_argparser().parse_args(args_list)
    
    print()
    dct = vars(args)
    for k in sorted(dct):
        print(k, dct[k])
    print()

    # dist_util.setup_dist()
    logger.configure(dir=DIR, mode="training", args=args, tag='')

    logger.log("creating model and diffusion...")
    model = select_model(args)(args)
    print(count_params(model))
    model.to(args.device)

    # model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, model.diffusion
    )

    logger.log("creating data loader...")
    train_loader = data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle="True",
        num_workers=1,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        data=train_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        # val_loader=val_loader,
        args=args
    ).run_loop()


def refinement_train(dataset):
    pass


def relational_network_train(dataset):
    pass



if __name__ == '__main__':
    json_dir = "D:/ARC/arc-data/arc-agi_training_challenges.json"
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    task = 1

    dataset = DiffusionDataset(json_dir=json_dir, device="cpu", task=task)
    sample_size = len(dataset)  # This should be returned by the Dataset

    diffusion_train(dataset, sample_size)
