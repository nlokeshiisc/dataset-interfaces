import torch
import torch.utils.data as data_utils
import constants as constants
import torch.nn as nn


def get_loader_subset(
    loader: data_utils.DataLoader,
    subset_idxs: list,
    batch_size=None,
    shuffle=False,
    num_workers=4,
):
    """Returns a dataloader with the sunset indices.
    The collate_fn is the same as the original loader.

    Args:
        loader (data_utils.DataLoader): _description_
        subset_idxs (list): _description_
        batch_size (_type_, optional): _description_. Defaults to None.
        shuffle (bool, optional): _description_. Defaults to False.
        num_workers (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """
    subset_ds = data_utils.Subset(
        dataset=loader.dataset,
        indices=subset_idxs,
    )
    if batch_size is None:
        batch_size = loader.batch_size
    sub_loader = data_utils.DataLoader(
        subset_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=loader.collate_fn,
        num_workers=num_workers,
    )
    return sub_loader


def init_loader(
    ds: data_utils.Dataset, batch_size, shuffle=False, num_workers=4, **kwargs
):
    if constants.SAMPLER in kwargs:
        return data_utils.DataLoader(
            ds,
            batch_size=batch_size,
            sampler=kwargs[constants.SAMPLER],
            num_workers=num_workers,
        )
    else:
        return data_utils.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )


def batch_norm_off(model: nn.Module):
    """
    Turns off batch norm for the model. This is needed to handle the batch size of 1
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def batch_norm_on(model: nn.Module):
    """
    Enables the batch norm layers on the model.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.train()
            m.weight.requires_grad = True
            m.bias.requires_grad = True
