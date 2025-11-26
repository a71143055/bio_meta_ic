import argparse

def get_main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()

def get_task_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--mode", type=str, default="lines")
    return parser.parse_args()
