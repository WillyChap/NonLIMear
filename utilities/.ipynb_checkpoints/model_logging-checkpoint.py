"""
Author: wchapman@ucar.edu

Will Chapman
"""


from datetime import datetime
import torch

def update_tqdm(tq, train_loss, val_stats=None, test_stats=None, **kwargs):
    def get_stat_dict(dictio, prefix, all=False):
        dict_two = dict()
        set_if_exists(dictio, dict_two, 'rmse', prefix)
        set_if_exists(dictio, dict_two, 'corrcoef', prefix)
        set_if_exists(dictio, dict_two, 'all_season_cc', prefix)

        if all:
            set_if_exists(dictio, dict_two, 'mae', prefix)
        return dict_two

    if val_stats is None:
        if test_stats is None:
            tq.set_postfix(train_loss=train_loss, **kwargs)
        else:
            test_print = get_stat_dict(test_stats, 'test')
            tq.set_postfix(train_loss=train_loss, **test_print, **kwargs)
    else:
        val_print = get_stat_dict(val_stats, 'val', all=True)
        if test_stats is None:
            tq.set_postfix(train_loss=train_loss, **val_print, **kwargs)
        else:
            test_print = get_stat_dict(test_stats, 'test')
            tq.set_postfix(train_loss=train_loss, **val_print, **test_print, **kwargs)


def set_if_exists(dictio_from, dictio_to, key, prefix):
    if key in dictio_from:
        dictio_to[f'{prefix}_{key}'.lstrip('_')] = dictio_from[key]

def save_model(epochs, model, optimizer, criterion,strpath,arger):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'args':arger
                }, strpath)
