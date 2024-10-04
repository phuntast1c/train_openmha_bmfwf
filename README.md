# Installation Guide
Follow these steps to set up the environment and ensure everything is working correctly:

## 1. Set Up a New Conda Environment
To begin, create a new Conda environment using the provided `environment.yaml` file. This will install all the necessary dependencies except for PyTorch and torchaudio.

1. Download the `environment.yaml` file.

2. Run the following command in your terminal to create the environment:
```bash
conda env create -f environment.yaml
```

3.	After the environment is created, activate it:
```bash
conda activate train_bmfwf
```

## 2. Install PyTorch and Torchaudio
Next, you need to install PyTorch and Torchaudio. You can do this according to the official PyTorch installation instructions based on your system configuration.
Visit the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) to generate the appropriate installation command for your environment. Select the following options:
•	PyTorch Build: Stable
•	Your OS
•	Package: Conda
•	Language: Python
•	Compute Platform: Choose CUDA if you have a compatible GPU, otherwise select CPU.

## 3. Test the Installation
Once all packages are installed and the conda environment is activated, you can verify the setup by running the provided  scripts:

1. Test Training a Model on Dummy Data
Run the model training by using the PyTorch-Lightning command line interface (CLI) (more on PyTorch-Lightning [here](https://lightning.ai/docs/pytorch/stable/starter/introduction.html#) and on its CLI [here](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli)):
```bash
python cli.py fit --trainer=configs/trainer.yaml --trainer.max_epochs=20 --model=configs/bmfwf.
yaml --data=configs/dataset.yaml --data.num_workers=0
```
This should take about a minute depending on your hardware and result in a model not performance well (since it's trained on random dummy data).
The resulting model will be saved in bmfwf/saved/, with one subdirectory per training run.

2. Test Exporting the Model to TorchScript:
Run the export_to_torchscript.py script to export the new model to TorchScript format:
```bash
python export_to_torchscript.py
```
will use the original model; otherwise you can specify the directory of your new model as
```bash
python export_to_torchscript.py --ckpt_dir=XXX
```

3. Test Running the TorchScript Model:
Next, test running the exported TorchScript model on some example .wav file using the run_torchscript_model.py script:
```bash
python run_torchscript_model.py
```
will again use the original model; for more options execute
```bash
python run_torchscript_model.py --help
```

If all scripts run successfully without errors, the installation was successful and your environment is properly set up for development and training.

## 3. Train a Model
To train a model on your own dataset, you can use a similar command as during testing:
```bash
python cli.py fit --trainer=configs/trainer.yaml --trainer.max_epochs=20 --model=configs/bmfwf.
yaml --data=configs/your_dataset_config.yaml --data.num_workers=0
```
where `your_dataset_config.yaml` contains the LightningDataModule class path and initialization arguments of your own dataset (thus replacing the DummyDataModule used in the test step).
As a reference for creating a LightningDataModule for your own data, the LightningDataModule that was used to train the original model (H4a2RLFixedDataModule) is also provided in `bmfwf/datasets.py`.
Please note that the actual original data have not been published at this moment.