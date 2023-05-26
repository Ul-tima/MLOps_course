import typer

from ml.ser.load_datasets import load_config
from ml.ser.load_datasets import load_saved_data
from ml.ser.model import save_model_to_registry
from ml.ser.train import train_model

app = typer.Typer()
app.command()(train_model)
app.command()(load_saved_data)
app.command()(load_config)
app.command()(save_model_to_registry)


if __name__ == "__main__":
    app()
