This repository provides resources to train the binaural multi-frame Wiener filter (BMFWF) available as an openMHA [plugin](https://github.com/HoerTech-gGmbH/openMHA/tree/b8c8bf613863ed1f866cb1a1c77091770c97fbf1/examples/34-DNN-based-speech-enhancement).
The BMFWF is a deep learning-based speech enhancement algorithm that can be used to improve the speech quality and intelligibility of recorded hearing aid signals.
The model is available as a PyTorch model.
This repository provides the necessary PyTorch-Lightning-based scripts to train the model on your own data and export it to TorchScript format for use as a replacement of the openMHA plugin.

The deep neural network architecture is based on [Wang et al., ICASSP 2023](https://ieeexplore.ieee.org/document/10095700), modified to decrease computational complexity.
The BMFWF implementation is inspired from [Wang et al., T-ASLP 2023](https://ieeexplore.ieee.org/document/10214650), modified for online processing using recursive smoothing.

# Installation Guide
Follow these steps to set up the environment and ensure everything is working correctly:

## 1. Clone the Repository and Change Directory
Begin by cloning the repository to your local machine and changing into the project directory:
```bash
git clone https://github.com/phuntast1c/train_openmha_bmfwf.git
cd train_bmfwf
```

## 2. Set Up a New Conda Environment
To begin, create a new Conda environment using the provided `environment.yaml` file. This will install all the necessary dependencies except for PyTorch and torchaudio.

1. Run the following command in your terminal to create the environment:
```bash
conda env create -f environment.yaml
```

3.	After the environment is created, activate it:
```bash
conda activate train_openmha_bmfwf
```

## 3. Install PyTorch and Torchaudio
Next, you need to install PyTorch and Torchaudio according to the official PyTorch installation instructions based on your system configuration.
Visit [here](https://pytorch.org/get-started/previous-versions/#v201/) to find the appropriate installation command for your environment. Select the following options:
•	Your OS
•	Conda
•	Compute Platform: Choose CUDA if you have a compatible GPU, otherwise select CPU.

## 4. Test the Installation
Once all packages are installed and the conda environment is activated, you can verify the setup by running the provided  scripts:

1. Test Training a Model on Dummy Data
Run the model training by using the PyTorch-Lightning command line interface (CLI) (more on PyTorch-Lightning [here](https://lightning.ai/docs/pytorch/stable/starter/introduction.html#) and on its CLI [here](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli)):
```bash
python cli.py fit --trainer=configs/trainer.yaml --trainer.max_epochs=20 --model=configs/bmfwf.yaml --data=configs/dataset.yaml --data.num_workers=0
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

## 5. Train a Model
To train a model on your own dataset, you can use a similar command as during testing:
```bash
python cli.py fit --trainer=configs/trainer.yaml --trainer.max_epochs=20 --model=configs/bmfwf.yaml --data=configs/your_dataset_config.yaml --data.num_workers=0
```
where `your_dataset_config.yaml` contains the LightningDataModule class path and initialization arguments of your own dataset (thus replacing the DummyDataModule used in the test step).
As a reference for creating a LightningDataModule for your own data, the LightningDataModule that was used to train the original model (H4a2RLFixedDataModule) is also provided in `bmfwf/datasets.py`.
Please note that the actual original data have not been published at this moment.

When satisfied with your model, you can let openMHA use it by modifying the [configuration file](https://github.com/HoerTech-gGmbH/openMHA/blob/b8c8bf613863ed1f866cb1a1c77091770c97fbf1/examples/34-DNN-based-speech-enhancement/index.cfg#L43).