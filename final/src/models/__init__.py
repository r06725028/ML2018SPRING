import importlib


def get_model(model_name, weights=None):
    model = importlib.import_module('.'.join(('src', 'models', model_name)))
    return model.gen_model(weights)
