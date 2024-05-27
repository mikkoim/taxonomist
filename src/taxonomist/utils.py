import bisect
import gc
import importlib.util
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image as Image
import timm
import torch
import torchvision


def load_continuous_transform(name: str):
    """
    Loads function dictionary that transforms values to a specified continuous space.

    Different transformations
    log: Transforms data to the log space. This transformation should only be used for strictly positive data
    """
    if name == "log":
        label_tf = {"fwd": np.log, "inv": np.exp}
    else:
        raise Exception(f"Invalid transform name {name}")
    return label_tf


def load_class_map(fname: str):
    """Loads functions that map classes to indices and the inverse of this"""
    if not Path(fname).exists():
        class_map = load_continuous_transform(fname)
    else:
        with open(fname) as f:
            fwd = {label.strip(): i for i, label in enumerate(f)}
        inv = {v: k for k, v in fwd.items()}
        class_map = {}
        # class_map["fwd"] = lambda x: np.array([fwd[str(int(v))] for v in x])
        # class_map["inv"] = lambda x: np.array([inv[int(v)] for v in x])
        class_map["fwd"] = lambda x: np.array([fwd[v] for v in x])
        class_map["inv"] = lambda x: np.array([inv[v] for v in x])
        class_map["fwd_dict"] = fwd
        class_map["inv_dict"] = inv
    return class_map


def read_image(fpath: str):
    img = Image.open(fpath)
    return img


def show_img(T, name: str = None, to_numpy: bool = False):
    """Displays an arbitary tensor/numpy array"""
    fname = name or "img-" + datetime.now().strftime("%H%M%S") + ".jpg"
    if isinstance(T, Image.Image):
        img = T
        I = np.array(img)
    else:
        if isinstance(T, torch.Tensor):
            if len(T.shape) == 4:
                T = torchvision.utils.make_grid(T)
            T = T.permute(1, 2, 0).numpy()
        elif isinstance(T, np.ndarray):
            T = T.astype(float)
        T -= T.min()
        T = T / (T.max() + 1e-8)
        I = (T * 255).astype(np.uint8)

    if to_numpy:
        return I
    else:
        img = Image.fromarray(I)
    img.save(fname)


def class_batch(ds, target, n=8):
    """Fetches n samples matching the target from the dataset ds"""
    all_inds = np.where(ds.y == target)[0]
    if len(all_inds) == 0:
        raise Exception("No label")
    inds = np.random.choice(all_inds, n)
    x_list = [ds[i]["x"] for i in inds]
    X = torch.stack(x_list)
    return X


def histogram_batch(ds, bins, b, n=8):
    """Fetches n samples from the histogram bin 'bin' for a continuous value ds.y"""

    all_inds = np.where(bins == b)[0]
    inds = np.random.choice(all_inds, n)
    x_list = [ds[i]["x"] for i in inds]
    X = torch.stack(x_list)
    return X


def visualize_dataset(ds, n=8, v=True, name=None, to_numpy=False):
    """Finds unique classes from the dataset ds and fetches n examples of all classes.
    Returns this as a array, or saves to an image
    """
    I_list = []
    fname = name or "img-" + datetime.now().strftime("%H%M%S") + ".jpg"

    # Continuous target
    if len(np.unique(ds.y)) > 50:
        _, bin_edges = np.histogram(ds.y, bins=50)
        bins = [bisect.bisect(bin_edges, x) for x in ds.y]
        for b in np.unique(bins):
            if v:
                print(bin_edges[b - 1])
            T = histogram_batch(ds, bins, b)
            I = show_img(T, to_numpy=True)
            I_list.append(I)
    # Categorical target
    else:
        for target in np.unique(ds.y):
            if v:
                print(target)
            T = class_batch(ds, target, n)
            I = show_img(T, to_numpy=True)
            I_list.append(I)

    I = np.vstack(I_list)

    if to_numpy:
        return I
    else:
        img = Image.fromarray(I)
        img.save(fname)


def load_module_from_path(fpath: str):
    """Loads an python module from an arbitary filepath. Used for importing dataset
    configs
    """
    fpath = Path(fpath)
    spec = importlib.util.spec_from_file_location(str(fpath.stem), str(fpath))
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    return module


def get_pretrained_model_details():
    """
    This method helps to find models with sepcific parameter count. It can be used to find models that fit your memory.
    """

    models = {
        s.split(".")[0] for s in timm.list_models(pretrained=True) if "." in s
    }  # timm.list_models(pretrained=True)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    (
        model_names,
        num_parameters,
        param_memories,
        forward_pass_memories,
        total_memories,
        input_image_sizes,
    ) = [], [], [], [], [], []
    # ijk=0
    for m in models:
        try:
            # Free GPU memory before the check
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)
            # pre_memory = torch.cuda.memory_allocated() / (1024**3)
            # pre_memory_cached = torch.cuda.memory_cached() / (1024**3)

            tf_m = timm.create_model(
                m, num_classes=0, pretrained=False
            ).cuda()  # if pretrained, then the parameters are downloaded, which takes unnecessary time
            param_memory_checkpoint = torch.cuda.memory_allocated() / (1024**3)
            # param_memory_checkpoint_cached = torch.cuda.memory_cached() / (1024**3)
            # Generate some random input data (adjust the size accordingly)
            i_s = tf_m.pretrained_cfg["input_size"]
            input_data = torch.randn(
                (1, i_s[0], i_s[1], i_s[2])
            ).cuda()  # torch.randn((1, 3, 224, 224)).cuda()

            # Check GPU memory before forward pass
            pre_forwardpass_memory = torch.cuda.memory_allocated() / (1024**3)
            pre_forwardpass_memory_cached = torch.cuda.memory_cached() / (1024**3)
            # mem_cached_before = torch.cuda.memory_cached() / (1024 ** 3)

            # Forward pass to allocate memory
            output = tf_m(input_data)

            forward_pass_memory_checkpoint = torch.cuda.memory_allocated() / (1024**3)
            forward_pass_memory_checkpoint_cached = torch.cuda.memory_cached() / (
                1024**3
            )
            # Check GPU memory after forward pass
            # mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
            # mem_cached_after = torch.cuda.memory_cached() / (1024 ** 3)
            # memory_consumption = mem_after - mem_before
            # cached_memory_consumption = mem_cached_after - mem_cached_before

            param_memories += [param_memory_checkpoint]  #  - pre_memory]
            forward_pass_memories += [
                forward_pass_memory_checkpoint
            ]  # - pre_forwardpass_memory]
            total_memories += [forward_pass_memory_checkpoint]  #  -pre_memory]

            model_names += [m]
            num_parameters += [count_parameters(tf_m)]
            # mem_consumptions += [memory_consumption]
            # catched_memory_consumptions += [cached_memory_consumption]
            input_image_sizes += [i_s]
            # import pdb; pdb.set_trace()
            # dict_models[m] = count_parameters(tf_m)
            # print(m, dict_models[m])
            # print(m, memory_consumption, cached_memory_consumption, i_s)

            # Free GPU memory
            # torch.cuda.empty_cache()
            # gc.collect()
            # time.sleep(1)

        except Exception as e:
            print(m, "errored:")
            print(e)
        try:
            del tf_m
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)
        except Exception as e:
            print("memory cleaning failed")
        # ijk +=1
        # if ijk > 10:
        #    break
    return pd.DataFrame(
        {
            "Name": model_names,
            "Parameter count": num_parameters,
            "Forward pass memory": forward_pass_memories,
            "Parameter memory": param_memories,
            "Total memory": total_memories,
            "Input image size": input_image_sizes,
        }
    ).sort_values(
        by="Total memory", ascending=True
    )  # dict(sorted(dict_models.items(), key=lambda item: item[1]))


def write_model_details_to_file(df, filename):
    df.to_csv(filename, sep=";", index=False)


def read_model_details_from_file(filename):
    df = pd.read_csv(filename, sep=";")
    return df
