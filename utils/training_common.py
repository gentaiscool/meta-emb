import numpy as np

# Decay functions to be used with lr_scheduler
def lr_decay_noam(params):
    return lambda t: (
        10.0 * params["hidden_size"]**-0.5 * min(
            (t + 1) * params["learning_rate_warmup_steps"]**-1.5, (t + 1)**-0.5))

def lr_decay_exp(params):
    return lambda t: params["learning_rate_falloff"] ** t

def lr_decay_map():
    return {
        'noam': lr_decay_noam,
        'exp': lr_decay_exp}

def compute_num_params(model):
    """
    Computes number of trainable and non-trainable parameters
    """
    sizes = [(np.array(p.data.size()).prod(), int(p.requires_grad)) for p in model.parameters()]
    return sum(map(lambda t: t[0]*t[1], sizes)), sum(map(lambda t: t[0]*(1 - t[1]), sizes))
