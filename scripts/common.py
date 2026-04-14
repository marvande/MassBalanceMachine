import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

import massbalancemachine as mbm
from regions.RGI_11_Switzerland.scripts.geodetic.geodetic_processing import (
    get_geodetic_MB,
)
from regions.RGI_11_Switzerland.scripts.config_CH import (
    path_PMB_GLAMOS_csv,
    path_ERA5_raw,
    path_pcsr,
)

from regions.RGI_11_Switzerland.scripts.dataset.data_loader import (
    process_or_load_data,
    get_CV_splits,
    get_stakes_data,
)
from regions.RGI_11_Switzerland.scripts.utils import seed_all

_default_test_glaciers = [
    "tortin",
    "plattalva",
    "sanktanna",
    "schwarzberg",
    "hohlaub",
    "pizol",
    "corvatsch",
    "tsanfleuron",
    "forno",
]
_default_train_glaciers = [
    "clariden",
    "oberaar",
    "otemma",
    "gietro",
    "rhone",
    "silvretta",
    "gries",
    "sexrouge",
    "allalin",
    "corbassiere",
    "aletsch",
    "joeri",
    "basodino",
    "morteratsch",
    "findelen",
    "albigna",
    "gorner",
    "murtel",
    "plainemorte",
    "adler",
    "limmern",
    "schwarzbach",
]

_default_additional_var = [
    "ALTITUDE_CLIMATE",
    "ELEVATION_DIFFERENCE",
    "POINT_ELEVATION",
    "pcsr",
]
_default_vois_climate = [
    "t2m",
    "tp",
    "slhf",
    "sshf",
    "ssrd",
    "fal",
    "str",
    "u10",
    "v10",
]
_default_vois_topographical = [
    "aspect_sgi",
    "slope_sgi",
    "hugonnet_dhdt",
    "consensus_ice_thickness",
    "millan_v",
]
_default_input = (
    _default_additional_var + _default_vois_climate + _default_vois_topographical
)


def parseParams(params):
    lr = float(params["training"].get("lr", 1e-3))
    optim = params["training"].get("optim", "ADAM")
    momentum = float(params["training"].get("momentum", 0.0))
    beta1 = float(params["training"].get("beta1", 0.9))
    beta2 = float(params["training"].get("beta2", 0.999))
    scheduler = params["training"].get("scheduler", None)
    scheduler_gamma = float(params["training"].get("scheduler_gamma", 0.5))
    scheduler_step_size = int(params["training"].get("scheduler_step_size", 200))
    Nepochs = int(params["training"].get("Nepochs", 1000))
    source_data = params["training"].get("source_data", "iceland")
    inputs = params["model"].get("inputs") or mbm.dataloader._default_input(source_data)
    batch_size = int(params["training"].get("batch_size", 128))
    weight_decay = float(params["training"].get("weight_decay", 0.0))
    downscale = params["model"].get("downscale", None)
    scalingStakes = params["training"].get("scalingStakes", "glacier")
    return {
        "model": {
            "type": params["model"]["type"],
            "layers": params["model"]["layers"],
            "inputs": inputs,
            "downscale": downscale,
        },
        "training": {
            "source_data": source_data,
            "lr": lr,
            "momentum": momentum,
            "beta1": beta1,
            "beta2": beta2,
            "optim": optim,
            "scheduler": scheduler,
            "scheduler_gamma": scheduler_gamma,
            "scheduler_step_size": scheduler_step_size,
            "Nepochs": Nepochs,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "scalingStakes": scalingStakes,
            "test_glaciers": params["training"].get("test_glaciers"),
            "train_glaciers": params["training"].get("train_glaciers"),
            "wGeo": params["training"].get("wGeo", 0.0),
            "bestModelCriterion": params["training"].get(
                "bestModelCriterion", "lossVal"
            ),
            "freqVal": params["training"].get("freqVal", 1),
            "log_suffix": params["training"].get("log_suffix", ""),
            "log_dir": params["training"].get("log_dir"),
        },
    }


def loadParams(modelType):
    with open("scripts/netcfg/" + modelType + ".yml") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    parsedParams = parseParams(params)
    return parsedParams
