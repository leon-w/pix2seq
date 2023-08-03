import keras_core as keras
import torch
import torch_optimizer


def get_optimizer(name: str, params, lr: float, **kwargs) -> torch.optim.Optimizer:
    name = name.lower()
    optimizer_cls = getattr(torch.optim, name, None)
    if optimizer_cls is None:
        optimizer_cls = torch_optimizer.get(name)

    return optimizer_cls(params=params, lr=lr, **kwargs)


def build_torch_parameters_to_keras_names_mapping(model: torch.nn.Module) -> dict[int, str]:
    mapping = {}

    def map_params(x):
        if isinstance(x, keras.layers.Layer):
            for w in x.trainable_weights:
                mapping[id(w.value)] = w.name

    model.apply(map_params)

    return mapping


def disable_weight_decay_for(
    parameters,
    exclude_names: list[str],
    name_mapping: dict[int, str],
) -> list[dict]:
    group_default = {"params": []}
    group_no_weight_decay = {"params": [], "weight_decay": 0.0}

    for param in parameters:
        if id(param) in name_mapping and any(exclude_name in name_mapping[id(param)] for exclude_name in exclude_names):
            group_no_weight_decay["params"].append(param)
        else:
            group_default["params"].append(param)

    if len(group_default["params"]) == 0:
        return [group_no_weight_decay]
    elif len(group_no_weight_decay["params"]) == 0:
        return [group_default]
    else:
        return [group_default, group_no_weight_decay]
