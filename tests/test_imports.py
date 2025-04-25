import importlib

def test_smoke_imports():
    """
    Smoke-test that core modules can be imported without errors.
    """
    modules = [
        "embeddings",
        "load_data",
        "groundtruth.ground_generator",
        "eval.rag_checker",
        "example_client",
    ]
    for module in modules:
        importlib.import_module(module)