import torch
import argparse
import os
import sys
import io
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from datetime import datetime
from accelerate import Accelerator
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'custom')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'aux')))

from classifier import make_classifier_model
from script_util import (
    add_dict_to_argparser,
    classifier_and_diffusion_defaults,
)
from logger import get_logger

def initialize_accelerator(args):
    accelerator = Accelerator(mixed_precision="fp16" if args.use_fp16 else "no")
    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            reinit=True,
            settings=wandb.Settings(start_method="fork")
        )
    return accelerator

def setup_directories(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.experiment_name = f"{args.attribute_type}_classifier_{timestamp}"
    args.model_save_dir = os.path.join(script_dir, args.output_path, args.wandb_name)
    os.makedirs(args.model_save_dir, exist_ok=True)
    return args

def setup_logger(args):
    logger = get_logger(args.model_save_dir, args.experiment_name)
    return logger

def load_checkpoint(args):
    if args.resume_from_checkpoint:
        args.resume_from_checkpoint = os.path.join(script_dir, args.resume_from_checkpoint)
    return args

def setup_attribute_info(args):
    attribute_info = {
        'gender': {'num_classes': 2, 'attributes': ["male", "female"]},
        'race': {'num_classes': 4, 'attributes': ["white", "black", "asian", "indian"]},
        'age': {'num_classes': 2, 'attributes': ["young", "old"]}
    }
    return attribute_info[args.attribute_type]

def initialize_classifier(args, num_classes):
    classifier = make_classifier_model(
        in_channels=args.in_channels,
        image_size=args.latents_size,
        out_channels=num_classes,
    )
    if args.resume_from_checkpoint:
        state_dict = torch.load(args.resume_from_checkpoint, map_location='cpu')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        classifier.load_state_dict(new_state_dict)
    return classifier

def prepare_training_components(accelerator, classifier, lr):
    opt = torch.optim.Adam(classifier.parameters(), lr=lr)
    classifier, opt = accelerator.prepare(classifier, opt)
    return classifier, opt

def load_dataset(args):
    dataset_path = os.path.join(script_dir, args.dataset_path)
    dataset = torch.load(dataset_path)
    return dataset

def split_dataset(dataset, train_split):
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def create_data_loader(dataset, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def train_classifier_on_epoch(accelerator, classifier, data_loader, opt, attribute_type, use_one_hot, prefix="train"):
    classifier.train() if prefix == "train" else classifier.eval()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    timestep_accs = [0] * 50
    
    for batch in data_loader:
        h_debiased = batch["h_debiased"].to(device=accelerator.device)  # Shape: [batch_size, 50, 1280, 8, 8]
        labels = batch["labels"][attribute_type].to(device=accelerator.device)  # Shape: [batch_size, num_classes]
        
        batch_size = h_debiased.shape[0]
        
        if use_one_hot:
            labels = labels.argmax(dim=-1)  # Convert to class indices
        
        h_debiased = h_debiased.permute(1, 0, 2, 3, 4)  # Shape: [50, batch_size, 1280, 8, 8]
        
        for t in range(50):  # For each timestep
            timestep_data = h_debiased[t]  # Shape: [batch_size, 1280, 8, 8]
            
            with torch.set_grad_enabled(prefix == "train"):
                logits = classifier(timestep_data, [t] * timestep_data.shape[0])
                
                if use_one_hot:
                    loss = F.cross_entropy(logits, labels)
                    predicted_attr = logits.argmax(dim=-1)
                    correct = (predicted_attr == labels).sum().item()
                else:
                    loss = F.cross_entropy(logits, labels)
                    predicted_attr = logits.argmax(dim=-1)
                    true_labels = labels.argmax(dim=-1)
                    correct = (predicted_attr == true_labels).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += timestep_data.shape[0]
                
                if use_one_hot:
                    all_preds.extend(predicted_attr.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())
                else:
                    all_preds.extend(predicted_attr.detach().cpu().numpy())
                    all_labels.extend(true_labels.detach().cpu().numpy())
                
                timestep_accs[t] += correct / timestep_data.shape[0]
                
                if prefix == "train":
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(classifier.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    timestep_accs = [acc / len(data_loader) for acc in timestep_accs]
    
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, output_dict=True)
    
    return avg_loss, avg_acc, cm, cr, timestep_accs

def log_timestep_accuracy(accelerator, timestep_accs, epoch, prefix, attribute_type):
    if accelerator.is_main_process:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, 51), timestep_accs, marker='o')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.set_title(f'{prefix.capitalize()} Accuracy per Timestep - {attribute_type.capitalize()} - Epoch {epoch}')
        ax.grid(True)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        image = Image.open(buf)
        
        timestep_accuracy_plots = wandb.Image(image, caption=f"{prefix.capitalize()} Accuracy per Timestep - {attribute_type.capitalize()} - Epoch {epoch}")
        wandb.log({f"{prefix}_timestep_accuracy_plots": timestep_accuracy_plots})
        
        plt.close(fig)
        image.close()

def log_training_progress(accelerator, logger, epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc):
    logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
    
    if accelerator.is_main_process:
        wandb_log_dict = {
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc,
            "epoch": epoch
        }
        
        wandb.log(wandb_log_dict)

def log_confusion_matrices(accelerator, train_cm, val_cm, attribute_info):
    if accelerator.is_main_process:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        sns.heatmap(train_cm, annot=True, fmt='d', ax=ax1, xticklabels=attribute_info['attributes'], yticklabels=attribute_info['attributes'])
        ax1.set_title("Train Confusion Matrix")
        
        sns.heatmap(val_cm, annot=True, fmt='d', ax=ax2, xticklabels=attribute_info['attributes'], yticklabels=attribute_info['attributes'])
        ax2.set_title("Validation Confusion Matrix")
        
        wandb.log({"confusion_matrices": wandb.Image(fig)})
        plt.close(fig)

def log_classification_reports(accelerator, train_cr, val_cr):
    if accelerator.is_main_process:
        wandb.log({
            "train_classification_report": wandb.Table(dataframe=pd.DataFrame(train_cr).transpose()),
            "val_classification_report": wandb.Table(dataframe=pd.DataFrame(val_cr).transpose())
        })

def save_model_checkpoint(accelerator, classifier, epoch, avg_val_acc, best_acc, save_interval, model_save_dir, logger):
    if (epoch + 1) % save_interval == 0 or avg_val_acc > best_acc:
        unwrapped_model = accelerator.unwrap_model(classifier)
        save_path = os.path.join(model_save_dir, f"model_epoch_{epoch}_acc_{avg_val_acc:.4f}.pt")
        
        if avg_val_acc >= best_acc:
            logger.info(f"New best model checkpoint saved! Epoch {epoch}")
            best_acc = avg_val_acc
            best_model_path = os.path.join(model_save_dir, "best_model.pt")
            accelerator.save(unwrapped_model.state_dict(), best_model_path)
    
    return best_acc

def main():
    args = create_argparser().parse_args()
    
    args = setup_directories(args)
    logger = setup_logger(args)
    args = load_checkpoint(args)
    
    accelerator = initialize_accelerator(args)
    
    logger.info(f"Attribute type: {args.attribute_type}")
    logger.info(f"Training batch size: {args.batch_size}")
    logger.info(f"Training epochs: {args.epochs}")
    
    attribute_info = setup_attribute_info(args)
    
    dataset = load_dataset(args)
    train_dataset, val_dataset = split_dataset(dataset, args.train_split)
    train_loader = create_data_loader(train_dataset, args.batch_size)
    val_loader = create_data_loader(val_dataset, args.batch_size)
    
    classifier = initialize_classifier(args, attribute_info['num_classes'])
    classifier, opt = prepare_training_components(accelerator, classifier, args.lr)
    
    start_epoch = 0
    if args.resume_from_checkpoint:
        start_epoch = int(args.resume_from_checkpoint.split("_")[-1].split(".")[0])
    
    best_acc = 0
    
    for epoch in tqdm(range(start_epoch, args.epochs), desc="Training progress"):
        logger.info(f"Epoch {epoch}")
        
        avg_train_loss, avg_train_acc, train_cm, train_cr, train_timestep_accs = train_classifier_on_epoch(
            accelerator, classifier, train_loader, opt, args.attribute_type, args.use_one_hot
        )
        
        classifier.eval()
        with torch.no_grad():
            avg_val_loss, avg_val_acc, val_cm, val_cr, val_timestep_accs = train_classifier_on_epoch(
                accelerator, classifier, val_loader, None, args.attribute_type, args.use_one_hot, prefix="val"
            )
        
        log_training_progress(accelerator, logger, epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc)
        log_confusion_matrices(accelerator, train_cm, val_cm, attribute_info)
        log_classification_reports(accelerator, train_cr, val_cr)
        log_timestep_accuracy(accelerator, train_timestep_accs, epoch, "train", args.attribute_type)
        log_timestep_accuracy(accelerator, val_timestep_accs, epoch, "val", args.attribute_type)
        
        torch.cuda.empty_cache()
        
        accelerator.wait_for_everyone()
        best_acc = save_model_checkpoint(accelerator, classifier, epoch, avg_val_acc, best_acc, args.save_interval, args.model_save_dir, logger)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.finish()

def create_argparser():
    defaults = classifier_and_diffusion_defaults()
    attribute_type = "race"  # Can be "gender", "race", or "age"
    epochs = 100
    batch_size = 256
    lr = 1e-4
    tv = 0.8
    defaults.update(dict(
        lr=lr,
        latents_size=8,
        in_channels=1280,
        use_fp16=True,
        save_interval=3,
        wandb_project=f"{attribute_type}_qqff",
        wandb_name=f"{attribute_type}_5k_e{epochs}_bs{batch_size}_lr{lr}_tv{tv}_v2",
        resume_from_checkpoint=None,
        epochs=epochs,
        batch_size=batch_size,
        attribute_type=attribute_type,
        output_path=f"classifiers_qqff/5k",
        dataset_path=f"datasets_training_qqff(v2)/5k/dataset.pt",
        train_split=tv,
        use_one_hot=True,
    ))
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()