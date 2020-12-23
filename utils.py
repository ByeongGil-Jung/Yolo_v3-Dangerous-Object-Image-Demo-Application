class Utils(object):

    def __init__(self):
        pass

    @classmethod  
    def get_tqdm_env_dict(cls, tqdm_env):
        tqdm_obj, tqdm_disable, is_plot_showed = None, None, None

        if tqdm_env == "script":
            import tqdm
            tqdm_obj = tqdm
        elif tqdm_env == "jupyter":
            import tqdm.notebook as tqdm
            tqdm_obj = tqdm
            is_plot_showed = True
        elif tqdm_env is None:
            import tqdm
            tqdm_obj = tqdm
            tqdm_disable = True

        return dict(
            tqdm=tqdm_obj,
            tqdm_disable=tqdm_disable,
            is_plot_showed=is_plot_showed
        )
