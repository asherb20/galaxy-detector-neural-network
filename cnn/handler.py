import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from torchvision import datasets
from galaxy_datasets.pytorch import GZ2
from cnn import transforms as T
from cnn.morphology import CLASSES, IGNORE, artifact_mask, morphology_labels
from cnn.network import ConvNeuralNet
from cnn.preresize import preresized_dir, use_preresized
from cnn.wrappers import GALAXY, GalaxyWrapper, NonGalaxyWrapper, ImageFiles

"""Train the galaxy detector + morphology classifier.

usage: python -m cnn.handler [--epochs 30] [--subset 3000]
"""

GZ2_ROOT = 'data/gz2'
CALTECH_ROOT = 'data/caltech101'
LOCAL_NON_GALAXIES = 'data/images/non_galaxies'
LOCAL_GALAXIES = 'data/images/galaxies'
DETECT_NAMES = ['non_galaxy', 'galaxy']


def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument('--epochs', type=int, default=30)
   parser.add_argument('--batch-size', type=int, default=128)
   parser.add_argument('--lr', type=float, default=1e-3)
   parser.add_argument('--weight-decay', type=float, default=1e-4)
   parser.add_argument('--per-class-cap', type=int, default=8000, help='max training galaxies per morphology class')
   parser.add_argument('--subset', type=int, default=None, help='max training galaxies in total, for quick runs')
   parser.add_argument('--noise-fraction', type=float, default=0.05, help='random-noise images, as a fraction of real negatives')
   parser.add_argument('--morph-weight', type=float, default=1.0, help='weight of the morphology loss vs. the detection loss')
   parser.add_argument('--patience', type=int, default=5, help='epochs without val improvement before stopping')
   parser.add_argument('--workers', type=int, default=4)
   parser.add_argument('--seed', type=int, default=42)
   parser.add_argument('--output', default='models/galaxy_morph_cnn.pth')
   parser.add_argument('--wandb-project', default='galaxy-detector-neural-network')
   parser.add_argument('--wandb-mode', default='online', choices=['online', 'offline', 'disabled'])
   return parser.parse_args()


# ---------------------------------------------------------------- data

def split_indices(indices, rng, fractions):
   """Shuffle indices and cut them into consecutive pieces by fraction"""
   indices = rng.permutation(indices)
   bounds = np.cumsum([int(round(f * len(indices))) for f in fractions[:-1]])
   return np.split(indices, bounds)


def stratified_split(indices, strata, rng, val_fraction):
   """Train/val split that keeps each stratum's proportion in both parts"""
   train, val = [], []
   for value in np.unique(strata[indices]):
      group = indices[strata[indices] == value]
      group_train, group_val = split_indices(group, rng, [1 - val_fraction, val_fraction])
      train.append(group_train)
      val.append(group_val)
   return np.concatenate(train), np.concatenate(val)


def cap_per_class(indices, labels, rng, cap):
   """Keep at most `cap` random indices of each label"""
   kept = []
   for value in np.unique(labels[indices]):
      group = indices[labels[indices] == value]
      kept.append(rng.permutation(group)[:cap])
   return np.concatenate(kept)


def load_gz2(train, transform_fn):
   """GZ2 dataset (pre-resized copies when available) and its morphology / artifact labels"""
   image_dir = preresized_dir(GZ2_ROOT)
   crop = None if image_dir else T.GZ2_CROP_FRACTION
   dataset = GZ2(root=GZ2_ROOT, train=train, transform=transform_fn(crop), download=False)
   if image_dir:
      use_preresized(dataset, image_dir)

   labels = morphology_labels(dataset.catalog)
   artifacts = artifact_mask(dataset.catalog)

   # workers pickle the dataset, so drop the 90 catalog columns we don't need
   dataset.catalog = dataset.catalog[['id_str', 'file_loc']].reset_index(drop=True)
   dataset.label_cols = None
   return dataset, labels, artifacts


def photo_folders():
   """Generic (non-astronomy) photo folders used as negatives"""
   caltech = datasets.Caltech101(root=CALTECH_ROOT, download=True)  # only used to download
   return [os.path.join(caltech.root, '101_ObjectCategories'), LOCAL_NON_GALAXIES]


def build_datasets(args, rng):
   """Returns {'train', 'val', 'test'} ConcatDatasets of (image, detect_label, morph_label)"""
   splits = {'train': [], 'val': [], 'test': []}

   # galaxies + artifacts from the GZ2 train catalog -> train / val
   gz2_train, labels, artifacts = load_gz2(True, T.train_transform)
   gz2_val, _, _ = load_gz2(True, T.eval_transform)
   galaxy_idx = np.flatnonzero(~artifacts)
   train_idx, val_idx = stratified_split(galaxy_idx, labels, rng, val_fraction=0.1)
   train_idx = cap_per_class(train_idx, labels, rng, args.per_class_cap)
   val_idx = cap_per_class(val_idx, labels, rng, max(args.per_class_cap // 8, 1))
   if args.subset:
      train_idx = rng.permutation(train_idx)[:args.subset]
      val_idx = rng.permutation(val_idx)[:max(args.subset // 9, 1)]
   splits['train'].append(GalaxyWrapper(gz2_train, train_idx, labels))
   splits['val'].append(GalaxyWrapper(gz2_val, val_idx, labels))

   artifact_train, artifact_val = split_indices(np.flatnonzero(artifacts), rng, [0.9, 0.1])
   splits['train'].append(NonGalaxyWrapper(gz2_train, artifact_train))
   splits['val'].append(NonGalaxyWrapper(gz2_val, artifact_val))

   # GZ2 test catalog -> test only
   gz2_test, test_labels, test_artifacts = load_gz2(False, T.eval_transform)
   test_idx = np.flatnonzero(~test_artifacts)
   if args.subset:
      test_idx = rng.permutation(test_idx)[:max(args.subset // 9, 1)]
   splits['test'].append(GalaxyWrapper(gz2_test, test_idx, test_labels))
   splits['test'].append(NonGalaxyWrapper(gz2_test, np.flatnonzero(test_artifacts)))

   # photos -> train / val / test
   for folder in photo_folders():
      train_files = ImageFiles(folder, T.train_transform())
      eval_files = ImageFiles(folder, T.eval_transform())
      photo_train, photo_val, photo_test = split_indices(np.arange(len(train_files)), rng, [0.8, 0.1, 0.1])
      splits['train'].append(NonGalaxyWrapper(train_files, photo_train))
      splits['val'].append(NonGalaxyWrapper(eval_files, photo_val))
      splits['test'].append(NonGalaxyWrapper(eval_files, photo_test))

   # a little random noise, so static is still rejected
   num_negatives = sum(len(d) for d in splits['train'] if isinstance(d, NonGalaxyWrapper))
   num_noise = int(num_negatives * args.noise_fraction)
   if num_noise:
      noise = datasets.FakeData(size=num_noise, image_size=(T.IN_CHANNELS, T.IMAGE_SIZE, T.IMAGE_SIZE),
                                num_classes=1, transform=T.train_transform(), random_offset=args.seed)
      splits['train'].append(NonGalaxyWrapper(noise))

   return {name: torch.utils.data.ConcatDataset(parts) for name, parts in splits.items()}


def label_arrays(dataset):
   """(detect_labels, morph_labels) of a ConcatDataset of wrappers, without loading images"""
   detect, morph = [], []
   for part in dataset.datasets:
      if isinstance(part, GalaxyWrapper):
         detect.append(np.full(len(part), GALAXY))
         morph.append(part.morph_labels[part.indices])
      else:
         detect.append(np.zeros(len(part), dtype=np.int64))
         morph.append(np.full(len(part), IGNORE))
   return np.concatenate(detect), np.concatenate(morph)


def class_weights(labels, num_classes):
   """Inverse-sqrt frequency weights, normalized to mean 1"""
   counts = np.bincount(labels[labels != IGNORE], minlength=num_classes).astype(np.float64)
   weights = 1 / np.sqrt(np.maximum(counts, 1))
   return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def make_loader(dataset, args, device, shuffle):
   return torch.utils.data.DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=shuffle,
      num_workers=args.workers,
      persistent_workers=args.workers > 0,
      pin_memory=(device.type == 'cuda'),
      drop_last=shuffle  # avoid a tiny last batch upsetting BatchNorm
   )


# ---------------------------------------------------------------- training

def compute_loss(outputs, detect_labels, morph_labels, criteria, morph_weight):
   detect_logits, morph_logits = outputs
   detect_loss = criteria['detect'](detect_logits, detect_labels)

   # morphology only counts for labelled galaxies; an all-ignored batch would give NaN
   has_morph = morph_labels != IGNORE
   if has_morph.any():
      morph_loss = criteria['morph'](morph_logits[has_morph], morph_labels[has_morph])
   else:
      morph_loss = detect_loss.new_zeros(())
   return detect_loss + morph_weight * morph_loss, detect_loss, morph_loss


def warmup_cosine(total_steps, warmup_steps):
   def schedule(step):
      if step < warmup_steps:
         return (step + 1) / warmup_steps
      progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
      return 0.5 * (1 + math.cos(math.pi * progress))
   return schedule


def train_one_epoch(model, loader, criteria, optimizer, scheduler, scaler, device, args):
   model.train()
   totals = {'loss': 0.0, 'detect': 0.0, 'morph': 0.0}
   for images, detect_labels, morph_labels in loader:
      images = images.to(device, non_blocking=True)
      detect_labels = detect_labels.to(device, non_blocking=True)
      morph_labels = morph_labels.to(device, non_blocking=True)

      # forward pass (mixed precision on GPU)
      with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
         outputs = model(images)
         loss, detect_loss, morph_loss = compute_loss(outputs, detect_labels, morph_labels, criteria, args.morph_weight)

      # backward and optimize
      optimizer.zero_grad(set_to_none=True)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      scheduler.step()

      totals['loss'] += loss.item()
      totals['detect'] += detect_loss.item()
      totals['morph'] += morph_loss.item()
   return {name: total / len(loader) for name, total in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device):
   """Confusion matrices for detection (all images) and morphology (labelled galaxies)"""
   model.eval()
   detect_cm = np.zeros((2, 2), dtype=np.int64)
   morph_cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
   for images, detect_labels, morph_labels in loader:
      detect_logits, morph_logits = model(images.to(device, non_blocking=True))
      detect_pred = detect_logits.argmax(1).cpu().numpy()
      morph_pred = morph_logits.argmax(1).cpu().numpy()
      detect_labels = detect_labels.numpy()
      morph_labels = morph_labels.numpy()

      np.add.at(detect_cm, (detect_labels, detect_pred), 1)
      has_morph = morph_labels != IGNORE
      np.add.at(morph_cm, (morph_labels[has_morph], morph_pred[has_morph]), 1)
   return detect_cm, morph_cm


def per_class_scores(cm):
   """(precision, recall, f1) arrays from a confusion matrix (rows = truth)"""
   true_pos = np.diag(cm).astype(np.float64)
   precision = true_pos / np.maximum(cm.sum(0), 1)
   recall = true_pos / np.maximum(cm.sum(1), 1)
   f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
   return precision, recall, f1


def summarize(detect_cm, morph_cm):
   return {
      'detect_acc': np.trace(detect_cm) / max(detect_cm.sum(), 1),
      'morph_acc': np.trace(morph_cm) / max(morph_cm.sum(), 1),
      'morph_f1': per_class_scores(morph_cm)[2].mean()
   }


def print_report(cm, names):
   precision, recall, f1 = per_class_scores(cm)
   width = max(len(n) for n in names)
   print(f"{'':{width}}  precision  recall    f1  support")
   for i, name in enumerate(names):
      print(f'{name:{width}}  {precision[i]:9.3f}  {recall[i]:6.3f}  {f1[i]:.3f}  {cm[i].sum():7d}')
   print('confusion matrix (rows = truth, cols = prediction):')
   for i, name in enumerate(names):
      print(f'{name:{width}}  ' + ' '.join(f'{v:6d}' for v in cm[i]))


@torch.no_grad()
def galaxy_rate(model, folder, device):
   """Fraction of images in a folder the model detects as galaxies"""
   files = ImageFiles(folder, T.eval_transform())
   if len(files) == 0:
      return None
   model.eval()
   images = torch.stack([files[i] for i in range(len(files))]).to(device)
   detect_logits, _ = model(images)
   return (detect_logits.argmax(1) == GALAXY).float().mean().item()


# ---------------------------------------------------------------- main

def report_to_wandb(run, prefix, cm, names):
   """Per-class scores into the run summary, plus the confusion matrix as a table"""
   precision, recall, f1 = per_class_scores(cm)
   for i, name in enumerate(names):
      run.summary[f'{prefix}/{name}/precision'] = precision[i]
      run.summary[f'{prefix}/{name}/recall'] = recall[i]
      run.summary[f'{prefix}/{name}/f1'] = f1[i]
   table = wandb.Table(columns=['truth'] + names, data=[[name] + cm[i].tolist() for i, name in enumerate(names)])
   run.log({f'{prefix}/confusion_matrix': table})


def main():
   args = parse_args()
   load_dotenv()  # WANDB_API_KEY from .env
   rng = np.random.default_rng(args.seed)
   torch.manual_seed(args.seed)

   # device will determine whether to run the training on GPU or CPU
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"using {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

   data = build_datasets(args, rng)
   detect_labels, morph_labels = label_arrays(data['train'])
   for name, dataset in data.items():
      print(f'{name}: {len(dataset)} images')
   print('train galaxies per class:', {c: int((morph_labels == i).sum()) for i, c in enumerate(CLASSES)},
         f'unlabelled: {int(((detect_labels == GALAXY) & (morph_labels == IGNORE)).sum())}',
         f'non-galaxies: {int((detect_labels != GALAXY).sum())}')

   train_loader = make_loader(data['train'], args, device, shuffle=True)
   val_loader = make_loader(data['val'], args, device, shuffle=False)

   # initialize the CNN model
   model = ConvNeuralNet(len(CLASSES), in_channels=T.IN_CHANNELS).to(device)
   print(f'parameters: {sum(p.numel() for p in model.parameters()):,}')

   criteria = {
      'detect': nn.CrossEntropyLoss(weight=class_weights(detect_labels, 2).to(device)),
      'morph': nn.CrossEntropyLoss(weight=class_weights(morph_labels, len(CLASSES)).to(device), label_smoothing=0.1)
   }
   optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer, warmup_cosine(total_steps=args.epochs * len(train_loader), warmup_steps=len(train_loader)))
   scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))

   run = wandb.init(project=args.wandb_project, mode=args.wandb_mode, config={
      **vars(args),
      'classes': CLASSES,
      'image_size': T.IMAGE_SIZE,
      **{f'num_{name}': len(dataset) for name, dataset in data.items()}
   })

   os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
   best_f1, stale_epochs = -1.0, 0
   print(f'\ntraining steps per epoch: {len(train_loader)}')

   for epoch in range(args.epochs):
      epoch_start = time.time()
      losses = train_one_epoch(model, train_loader, criteria, optimizer, scheduler, scaler, device, args)
      val = summarize(*evaluate(model, val_loader, device))
      epoch_time = time.time() - epoch_start

      print(f"epoch [{epoch + 1}/{args.epochs}] loss: {losses['loss']:.4f} "
            f"(detect {losses['detect']:.4f}, morph {losses['morph']:.4f}) | "
            f"val detect acc: {100 * val['detect_acc']:.2f}%, morph acc: {100 * val['morph_acc']:.2f}%, "
            f"morph macro-F1: {val['morph_f1']:.3f} | {epoch_time:.0f}s")
      run.log({
         'epoch': epoch + 1,
         'lr': scheduler.get_last_lr()[0],
         'epoch_time': epoch_time,
         **{f'train/{name}_loss': value for name, value in losses.items()},
         **{f'val/{name}': value for name, value in val.items()}
      })

      # keep only the best checkpoint
      if val['morph_f1'] > best_f1:
         best_f1, stale_epochs = val['morph_f1'], 0
         torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'classes': CLASSES,
            'image_size': T.IMAGE_SIZE,
            'in_channels': T.IN_CHANNELS,
            'val_metrics': {name: float(v) for name, v in val.items()}
         }, args.output)
         print(f'  saved {args.output}')
      else:
         stale_epochs += 1
         if stale_epochs >= args.patience:
            print(f'no improvement for {args.patience} epochs, stopping')
            break

   # final evaluation of the best checkpoint on held-out data
   checkpoint = torch.load(args.output, map_location=device)
   model.load_state_dict(checkpoint['model_state_dict'])
   print(f"\ntest results (best epoch {checkpoint['epoch']}):")
   detect_cm, morph_cm = evaluate(model, make_loader(data['test'], args, device, shuffle=False), device)
   test = summarize(detect_cm, morph_cm)
   print(f"detect acc: {100 * test['detect_acc']:.2f}%, morph acc: {100 * test['morph_acc']:.2f}%, "
         f"morph macro-F1: {test['morph_f1']:.3f}\n")
   print_report(detect_cm, DETECT_NAMES)
   print()
   print_report(morph_cm, CLASSES)

   run.summary['best_epoch'] = checkpoint['epoch']
   for name, value in test.items():
      run.summary[f'test/{name}'] = value
   report_to_wandb(run, 'test/detect', detect_cm, DETECT_NAMES)
   report_to_wandb(run, 'test/morph', morph_cm, CLASSES)

   rate = galaxy_rate(model, LOCAL_GALAXIES, device)
   if rate is not None:
      print(f'\nNASA images in {LOCAL_GALAXIES} detected as galaxies: {100 * rate:.1f}%')
      run.summary['test/nasa_galaxy_rate'] = rate
   run.finish()


# main training loop
if __name__ == "__main__":
   main()
