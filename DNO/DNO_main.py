
import datetime
import os
import random
import re
import time
from pathlib import Path

try:
    from models.FNO import FNO2d, FNO3d
except ImportError as exc:
    FNO2d = FNO3d = None
    FNO_IMPORT_ERROR = exc
else:
    FNO_IMPORT_ERROR = None

try:
    from models.Unet import UNet2d, UNet3d
except ImportError as exc:
    UNet2d = UNet3d = None
    UNET_IMPORT_ERROR = exc
else:
    UNET_IMPORT_ERROR = None

# from models.GFNO_steerable import GFNO2d_steer
# from models.Unet import Unet_Rot, Unet_Rot_M, Unet_Rot_3D
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio
import pandas as pd
from models.DNO import DNO # DNO-3
# from models.DNO_4 import DNO  # DNO-4
# from models.DNO_5 import DNO  # DNO-5


from utils25 import flood_data, LpLoss, nse, corr, critical_success_index

import scipy
import numpy as np
from timeit import default_timer
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import h5py
import xarray as xr
from tqdm import tqdm
from openpyxl import load_workbook
torch.set_num_threads(1)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def natural_sort_key(path_value):
    """Return a deterministic natural-sort key for a PT filename or path."""

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", Path(path_value).name)
    )


def naturally_sorted_pt_files(split_path):
    """Return naturally sorted PT files from one dataset split."""

    pt_files = sorted(
        (
            path
            for path in split_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".pt"
        ),
        key=lambda path: (natural_sort_key(path), path.name.casefold(), path.name),
    )
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in dataset directory: {split_path}")
    return pt_files


def inspect_split_tensor(split_name, split_path, expected_timesteps, expected_channels):
    """Validate the first naturally sorted tensor in a split and return its shape."""

    tensor_path = naturally_sorted_pt_files(split_path)[0]
    try:
        tensor = torch.load(tensor_path, map_location="cpu")
    except Exception as exc:
        raise ValueError(
            f"Could not load first {split_name} tensor {tensor_path}: {exc}"
        ) from exc
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"First {split_name} PT object is not a torch.Tensor: {tensor_path}"
        )
    if tensor.ndim != 4:
        raise ValueError(
            f"First {split_name} tensor must have rank 4 [Sy, Sx, time, channels]: "
            f"{tensor_path} has shape {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.float32:
        raise ValueError(
            f"First {split_name} tensor must use torch.float32: "
            f"{tensor_path} uses {tensor.dtype}"
        )
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError(
            f"First {split_name} tensor has non-positive spatial dimensions: "
            f"{tensor_path} has shape {tuple(tensor.shape)}"
        )
    if tensor.shape[2] != expected_timesteps:
        raise ValueError(
            f"First {split_name} tensor must contain exactly {expected_timesteps} "
            f"timesteps: {tensor_path} has {tensor.shape[2]}"
        )
    if tensor.shape[3] != expected_channels:
        raise ValueError(
            f"First {split_name} tensor must contain exactly {expected_channels} "
            f"channels: {tensor_path} has {tensor.shape[3]}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"First {split_name} tensor contains non-finite values: {tensor_path}")
    return tensor_path, tuple(tensor.shape)


def limit_dataset_events(dataset, requested_count, split_name):
    """Limit a flood_data instance to its first naturally sorted event paths."""

    sorted_paths = sorted(
        dataset.data,
        key=lambda path: (natural_sort_key(path), str(path).casefold(), str(path)),
    )
    available_count = len(sorted_paths)
    selected_count = (
        available_count
        if requested_count <= 0
        else min(requested_count, available_count)
    )
    dataset.data = sorted_paths[:selected_count]
    print(
        f"{split_name} event files: requested={requested_count}, "
        f"available={available_count}, selected={selected_count}"
    )
    return selected_count

def get_eval_pred(model, x, strategy, T, times):

    if strategy == "oneshot":
        pred = model(x)
    else:

        for t in range(T):
            t1 = default_timer()
            im = model(x)
            times.append(default_timer() - t1)
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -2)
            if strategy == "markov":
                x = im
            else:
                x = torch.cat((x[..., 1:, :], im), dim=-2)

    return pred

################################################################
# configs
################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--results_path", type=str, default="Path/Results/", help="path to store results")
parser.add_argument("--suffix", type=str, default="seed1", help="suffix to add to the results path")
parser.add_argument("--txt_suffix", type=str, default="Flood_DNO_Layers_oneset_seed1_t5", help="suffix to add to the results txt")
parser.add_argument("--super", type=str, default='False', help="enable superres testing")
parser.add_argument("--verbose",type=str, default='True')

parser.add_argument("--T", type=int, default=24, help="number of timesteps to predict")
parser.add_argument("--ntrain", type=int, default=100, help="training sample size")
parser.add_argument("--nvalid", type=int, default=13, help="valid sample size")
parser.add_argument("--ntest", type=int, default=12, help="test sample size")
parser.add_argument("--nsuper", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--model_type", type=str, default='DNO')
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--modes", type=int, default=12)
parser.add_argument("--width", type=int, default=20)
parser.add_argument("--Gwidth", type=int, default=10, help="hidden dimension of equivariant layers if model_type=hybrid")
parser.add_argument("--n_equiv", type=int, default=3, help="number of equivariant layers if model_type=hybrid")
parser.add_argument("--reflection", action="store_true", help="symmetry group p4->p4m for data augmentation")
parser.add_argument("--grid", type=str, default=None, help="[symmetric, cartesian, None]")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--early_stopping", type=int, default=50, help="stop if validation error does not improve for successive epochs")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--step", action="store_true", help="use step scheduler")
parser.add_argument("--gamma", type=float, default=0.5, help="gamma for step scheduler")
parser.add_argument("--step_size", type=int, default=None, help="step size for step scheduler")
parser.add_argument("--lmbda", type=float, default=0.0001, help="weight decay for adam")
parser.add_argument("--strategy", type=str, default="oneshot", help="markov, recurrent or oneshot")
parser.add_argument("--time_pad", action="store_true", help="pad the time dimension for strategy=oneshot")
parser.add_argument("--noise_std", type=float, default=0.00, help="amount of noise to inject for strategy=markov")
parser.add_argument(
    "--data_path",
    type=str,
    required=True,
    help="Root directory containing train, valid, and test PT folders",
)

args = parser.parse_args()

assert args.model_type in ["FNO2d", "FNO2d_aug",
                           "FNO3d", "FNO3d_aug",
                           "UNet2d", "UNet3d", "DNO"], f"Invalid model type {args.model_type}"
assert args.strategy in ["teacher_forcing", "markov", "recurrent", "oneshot"], "Invalid training strategy"

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

data_aug = "aug" in args.model_type

data_root = Path(args.data_path).expanduser().resolve()

if not data_root.is_dir():
    raise FileNotFoundError(
        f"Dataset root directory not found: {data_root}"
    )

Path_train = data_root / "train"
Path_valid = data_root / "valid"
Path_test = data_root / "test"

for split_name, split_path in {
    "train": Path_train,
    "valid": Path_valid,
    "test": Path_test,
}.items():
    if not split_path.is_dir():
        raise FileNotFoundError(
            f"Missing '{split_name}' dataset directory: {split_path}"
        )

# FNO data specs
S = 64 # spatial res
S_super = 4 * S # super spatial res
T_in = 1 # number of input times
T = args.T
T_super = 4 * T # prediction temporal super res
d = 2 # spatial res
num_channels = 5
num_channels_y = 3

reference_tensors = {}
for split_name, split_path in {
    "train": Path_train,
    "valid": Path_valid,
    "test": Path_test,
}.items():
    reference_tensors[split_name] = inspect_split_tensor(
        split_name,
        split_path,
        expected_timesteps=T_in + T,
        expected_channels=num_channels,
    )

train_reference_path, inferred_shape = reference_tensors["train"]
for split_name in ("valid", "test"):
    split_path, split_shape = reference_tensors[split_name]
    if split_shape != inferred_shape:
        raise ValueError(
            f"First train, valid, and test tensors must have identical full shapes: "
            f"train {train_reference_path} has {inferred_shape}, "
            f"{split_name} {split_path} has {split_shape}"
        )

Sy, Sx = inferred_shape[:2]
print("Startup data summary")
print(f"  device: {device}")
print(f"  inferred tensor shape [Sy, Sx, time, channels]: {inferred_shape}")
print(f"  Sy: {Sy}")
print(f"  Sx: {Sx}")
print(f"  T_in: {T_in}")
print(f"  T_out: {T}")
print(f"  channels: {num_channels}")

# adjust data specs based on model type and data path
threeD = args.model_type in ["FNO3d",
                             "Unet_Rot_3D", "DNO", "UNet3d"]
swe = False
rdb = False
grid_type = "cartesian"
if args.grid:
    grid_type = args.grid
    assert grid_type in ['symmetric', 'cartesian', 'None']


time_modes = None
time1 = args.strategy == "oneshot" # perform convolutions in space-time
if time1 and not args.time_pad:
    time_modes = 5 if swe else 8 # 6 is based on T=10
elif time1 and swe:
    time_modes = 8

modes = args.modes
width = args.width
n_layer = args.depth
batch_size = args.batch_size

epochs = args.epochs # 500
learning_rate = args.learning_rate
scheduler_step = args.step_size
scheduler_gamma = args.gamma # for step scheduler

initial_step = 1 if args.strategy == "markov" else T_in

root = args.results_path + f"/{'_'.join(str(datetime.datetime.now()).split())}"
if args.suffix:
    root += "_" + args.suffix
path_model = os.path.join(root, 'model.pt')
writer = SummaryWriter(root)

################################################################
# Model init
################################################################
if args.model_type in ["FNO2d", "FNO2d_aug"]:
    if FNO2d is None:
        raise ImportError("FNO2d is unavailable") from FNO_IMPORT_ERROR
    model = FNO2d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes, width=width,
                  grid_type=grid_type).to(device)
elif args.model_type in ["FNO3d", "FNO3d_aug"]:
    if FNO3d is None:
        raise ImportError("FNO3d is unavailable") from FNO_IMPORT_ERROR
    modes3 = time_modes if time_modes else modes
    model = FNO3d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes, modes3=modes3,
                  width=width, time=time1, time_pad=args.time_pad).to(device)
elif args.model_type == "DNO":
    model = DNO(num_channels=num_channels, width=10, initial_step=initial_step, pad=args.time_pad, factor=1).to(device)
elif args.model_type == "UNet3d":
    if UNet3d is None:
        raise ImportError("UNet3d is unavailable") from UNET_IMPORT_ERROR
    model = UNet3d(in_channels=initial_step * num_channels, out_channels=num_channels_y, init_features=32,
                   grid_type=grid_type, time=time1).to(device)
else:
    raise NotImplementedError("Model not recognized")

################################################################
# load data
# Input: DEM/Initial conditions/Rainfall/coords
################################################################
full_data = None # for superres
# SR dataset
dem_tif_path = 'Path/to/moa_bottom.tif'
man_path = 'Path/to/moa_rough.tif'



train_data = flood_data(path_root=Path_train, strategy=args.strategy, T_in=T_in, T_out=T, std=args.noise_std)
train_event_count = limit_dataset_events(train_data, args.ntrain, "train")
ntrain = len(train_data)
print('ntrain dataset samples', ntrain)
valid_data = flood_data(path_root=Path_valid, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
valid_event_count = limit_dataset_events(valid_data, args.nvalid, "valid")
nvalid = len(valid_data)
print('nvalid dataset samples', nvalid)
test_data = flood_data(path_root=Path_test, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
test_event_count = limit_dataset_events(test_data, args.ntest, "test")
ntest = len(test_data)
print('ntest dataset samples', ntest)
print(
    f"Selected event files: train={train_event_count}, "
    f"valid={valid_event_count}, test={test_event_count}"
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
complex_ct = sum(par.numel() * (1 + par.is_complex()) for par in model.parameters())
real_ct = sum(par.numel() for par in model.parameters())
if args.verbose:
    print(f"{args.model_type}; # Params: complex count {complex_ct}, real count: {real_ct}")
writer.add_scalar("Parameters/Complex", complex_ct)
writer.add_scalar("Parameters/Real", real_ct)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.lmbda)
# optimizer = fine_tune_fc1(model=model, learning_rate=learning_rate)
if args.step:
    assert args.step_size is not None, "step_size is None"
    assert scheduler_gamma is not None, "gamma is None"
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=scheduler_gamma)
else:
    num_training_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

lploss = LpLoss(size_average=False)

best_valid = float("inf")

x_train, y_train, _ = next(iter(train_loader))
x = x_train.to(device)
y = y_train.to(device)
x_valid, y_valid, _ = next(iter(valid_loader))
if args.verbose:
    print(f"{args.model_type}; Input shape: {x.shape}, Target shape: {y.shape}")
if args.strategy == "oneshot":
    assert x_train[0].shape == torch.Size([Sy, Sx, T, T_in, num_channels]), x_train[0].shape
    assert y_train[0].shape == torch.Size([Sy, Sx, T, num_channels_y]), y_train[0].shape
    assert x_valid[0].shape == torch.Size([Sy, Sx, T, T_in, num_channels]), x_valid[0].shape
    assert y_valid[0].shape == torch.Size([Sy, Sx, T, num_channels_y]), y_valid[0].shape
elif args.strategy == "markov":
    assert x_train[0].shape == torch.Size([Sy, Sx, 1, num_channels]), x_train[0].shape
    assert y_train[0].shape == torch.Size([Sy, Sx, num_channels_y]), y_train[0].shape
    assert x_valid[0].shape == torch.Size([Sy, Sx, 1, num_channels]), x_valid[0].shape
    assert y_valid[0].shape == torch.Size([Sy, Sx, T, num_channels_y]), y_valid[0].shape
else: # strategy == recurrent or teacher_forcing
    assert x_train[0].shape == torch.Size([Sy, Sx, T_in, num_channels]), x_train[0].shape
    assert x_valid[0].shape == torch.Size([Sy, Sx, T_in, num_channels]), x_valid[0].shape
    assert y_valid[0].shape == torch.Size([Sy, Sx, T, num_channels_y]), y_valid[0].shape
    if args.strategy == "recurrent":
        assert y_train[0].shape == torch.Size([Sy, Sx, T, num_channels_y]), y_train[0].shape
    else: # strategy == teacher_forcing
        assert y_train[0].shape == torch.Size([Sy, Sx, num_channels_y]), y_train[0].shape

model.eval()
start = default_timer()
if args.verbose:
    print("Training...")
step_ct = 0
train_times = []
eval_times = []
for ep in range(epochs):
    model.train()
    t1 = default_timer()

    train_l2 = train_vort_l2 = train_pres_l2 = 0

    for xx, yy, mask in tqdm(train_loader, disable=not args.verbose):
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        mask = mask.to(device)
        yy = yy * mask

        if args.strategy == "recurrent":
            for t in range(yy.shape[-2]):
                y = yy[..., t, :]
                mas = mask[..., t, :]
                im = model(xx)
                im = im * mas
                loss += lploss(im.reshape(len(im), -1, num_channels_y), y.reshape(len(y), -1, num_channels_y))
                xx = torch.cat((xx[..., 1:, :], im), dim=-2)
            loss /= yy.shape[-2]
        else:
            im = model(xx)
            im = im * mask
            if args.strategy == "oneshot":
                im = im.squeeze(-1)
            loss = lploss(im.reshape(len(im), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y))

        train_l2 += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not args.step:
            scheduler.step()
        writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], step_ct)
        step_ct += 1
    if args.step:
        scheduler.step()

    train_times.append(default_timer() - t1)

    # validation
    valid_l2 = valid_vort_l2 = valid_pres_l2 = 0
    valid_loss_by_channel = None
    with torch.no_grad():
        model.eval()
        model(xx)
        for xx, yy, mask in valid_loader:

            xx = xx.to(device)
            yy = yy.to(device)
            mask = mask.to(device)
            yy = yy * mask

            pred = get_eval_pred(model=model, x=xx, strategy=args.strategy, T=T, times=eval_times).view(len(xx), Sy, Sx, T, num_channels_y)
            pred = pred * mask

            valid_l2 += lploss(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()

    t2 = default_timer()
    if args.verbose:
        print(f"Ep: {ep}, time: {t2 - t1}, train: {train_l2 / ntrain}, valid: {valid_l2 / nvalid}")

    writer.add_scalar("Train/Loss", train_l2 / ntrain, ep)
    writer.add_scalar("Valid/Loss", valid_l2 / nvalid, ep)

    if valid_l2 < best_valid:
        best_epoch = ep
        best_valid = valid_l2
        # torch.save(model.state_dict(), path_model)
        state_dict = model.state_dict()
        torch.save({'epoch': best_epoch, 'state_dict': state_dict}, path_model)
    if args.early_stopping:
        if ep - best_epoch > args.early_stopping:
            break

stop = default_timer()
train_time = stop - start
train_times = torch.tensor(train_times).mean().item()
num_eval = len(eval_times)
eval_times = torch.tensor(eval_times).mean().item()
model.eval()
# test
checkpoint = torch.load(path_model)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
model.eval()
test_l2 = test_vort_l2 = test_pres_l2 = test_nse = test_corr = test_csi_1 = test_csi_2 = test_csi_3 = 0
rotations_l2 = 0
reflections_l2 = 0
test_rt_l2 = 0
test_rf_l2 = 0
test_loss_by_channel = None
key = 0
i = 0
total_time = 0
sample_count = 0

with torch.no_grad():
    for xx, yy, mask in test_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        mask = mask.to(device)
        yy = yy * mask
        input_data = xx
        # print('xx', xx.shape)
        # print('yy', yy.shape)
        # Start
        start_time = time.time()
        pred = get_eval_pred(model=model, x=xx, strategy=args.strategy, T=T, times=[]).view(len(xx), Sy, Sx, T, num_channels_y)
        # End
        end_time = time.time()
        batch_time = end_time - start_time
        total_time += batch_time
        sample_count += len(xx)
        # print(f"Average prediction time per sample: {batch_time:.4f} seconds, lens of samples: {len(xx)}")

        pred = pred * mask
        # print('pred', pred.shape)
        test_l2 += lploss(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()
        test_nse += nse(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()
        test_corr += corr(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()
        test_csi_1 += critical_success_index(pred[..., 0:1].reshape(len(pred), -1, 1),
                                             yy[..., 0:1].reshape(len(yy), -1, 1), 0.01).item()
        test_csi_2 += critical_success_index(pred[..., 0:1].reshape(len(pred), -1, 1),
                                             yy[..., 0:1].reshape(len(yy), -1, 1), 0.1).item()
        test_csi_3 += critical_success_index(pred[..., 0:1].reshape(len(pred), -1, 1),
                                             yy[..., 0:1].reshape(len(yy), -1, 1), 0.5).item()

print('sample_count', sample_count)
print('ntest', ntest)
average_time_per_sample = total_time / sample_count if sample_count > 0 else 0
test_time_l2 = test_space_l2 = ntest_super = test_int_space_l2 = test_int_time_l2 = None

print(f"Average prediction time per sample: {average_time_per_sample:.4f} seconds")
print(f"{args.model_type} done training; \nTest: {test_l2 / ntest}, Test_nse: {test_nse / ntest}, Test_corr: {test_corr / ntest}, Test_csi_1: {test_csi_1 / ntest}, Test_csi_2: {test_csi_2 / ntest}, Test_csi_3: {test_csi_3 / ntest}")
summary = f"Args: {str(args)}" \
          f"\nParameters: {complex_ct}" \
          f"\nTrain time: {train_time}" \
          f"\nMean epoch time: {train_times}" \
          f"\nMean inference time: {eval_times}" \
          f"\nNum inferences: {num_eval}" \
          f"\nTrain: {train_l2 / ntrain}" \
          f"\nValid: {valid_l2 / nvalid}" \
          f"\nTest: {test_l2 / ntest}" \
          f"\nTest_nse: {test_nse/ntest}" \
          f"\nTest_corr: {test_corr/ntest}" \
          f"\nTest_csi_1: {test_csi_1/ntest}" \
          f"\nTest_csi_2: {test_csi_2/ntest}" \
          f"\nTest_csi_3: {test_csi_3/ntest}" \
          f"\nSuper Space Test: {test_space_l2}" \
          f"\nSuper Space Interpolation Test: {test_int_space_l2}" \
          f"\nSuper S: {S_super}" \
          f"\nSuper Time Test: {test_time_l2}" \
          f"\nSuper Time Interpolation Test: {test_int_time_l2}" \
          f"\nSuper T: {T_super}" \
          f"\nBest Valid: {best_valid / nvalid}" \
          f"\nEpochs trained: {ep}"
txt = "results"
if args.txt_suffix:
    txt += f"_{args.txt_suffix}"
txt += ".txt"

with open(os.path.join(root, txt), 'w') as f:
    f.write(summary)
writer.flush()

writer.close()





