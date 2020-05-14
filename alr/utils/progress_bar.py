from tqdm.auto import tqdm, trange


def range_progress_bar(*args, **kwargs):
    return trange(*args, **kwargs)


def progress_bar(*args, **kwargs):
    return tqdm(*args, **kwargs)


