import os
import torch

def save(model, model_path, optimizer=None):
    if optimizer is not None:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_path)
    else:
        torch.save({
            'model_state_dict': model.state_dict()
        }, model_path)

def load(model_path, model, optimizer=None):

    saved_checkpoint = torch.load(model_path)
    model.load_state_dict(saved_checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in saved_checkpoint:
        optimizer.load_state_dict(saved_checkpoint['optimizer_state_dict'])

    return model, optimizer


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp