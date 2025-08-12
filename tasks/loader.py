def load_user_model(model_py: str, device: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("user_task_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    model, transform = mod.load_model(device)
    model.to(device).eval()
    return model, transform
