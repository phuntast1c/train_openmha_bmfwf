import argparse
from pathlib import Path

import torch
import torchaudio as ta

torch.set_default_device("cpu")

# example call:
# python run_model.py --checkpoint_path bmfwf/saved/20231018_090629_yellow-dish-5/model_bmfwf_qint8.pt --input test.wav --output test_output.wav


def parse_args():
    """
    Parses command-line arguments for processing a .wav file using a BMFWF model.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments:
        --checkpoint_path (str): Path to the checkpoint of the BMFWF model to be used.
                                 Default is "bmfwf/saved/20231018_090629_yellow-dish-5/model_bmfwf_qint8.pt".
        --input (str): Path to the input .wav file. Default is "test.wav".
        --output (str, optional): Path to save the processed output .wav file. Default is "test_output.wav".
    """
    parser = argparse.ArgumentParser(
        description="Process a .wav file containing 4 input channels using a BMFWF model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to checkpoint of the BMFWF model to be used",
        default="bmfwf/saved/20231018_090629_yellow-dish-5/model_bmfwf_qint8.pt",  # original model, trained to extract speech from the front
    )
    parser.add_argument(
        "--input", type=str, help="Path to the input .wav file", default="test.wav"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to save the processed output .wav file",
        default="test_output.wav",
    )
    return parser.parse_args()


def main():
    """
    Main function to run the model for processing audio input.

    This function performs the following steps:
    1. Parses command-line arguments.
    2. Loads a pre-trained model from a checkpoint.
    3. Loads and processes a 4-channel audio file.
    4. Resamples the audio if the sampling rate does not match the model's expected rate.
    5. Converts the audio to the Short-Time Fourier Transform (STFT) domain.
    6. Runs the model in inference mode to process the STFT audio.
    7. Converts the processed audio back from the STFT domain.
    8. Saves the processed audio to an output file.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()

    model = torch.jit.load(args.checkpoint_path)
    model.eval()  # put into evaluation mode

    wav_file_path = Path(args.input)
    wav_data, fs_wav = ta.load(wav_file_path, channels_first=True)
    assert (
        wav_data.shape[0] == 4
    ), "Only 4-channel input is supported! (2 channels per hearing aid)"
    wav_data = wav_data[:, : 10 * fs_wav]  # only process the first 10 seconds

    if fs_wav != model.fs:
        print(f"Resampling from {fs_wav} to {model.fs}")
        wav_data = ta.functional.resample(wav_data, orig_freq=fs_wav, new_freq=model.fs)

    wav_data = wav_data.unsqueeze(0)
    wav_data_stft = model.stft.get_stft(wav_data)

    print(f"processing {args.input}")
    with torch.no_grad():  # no gradient computation needed for inference
        output = model.forward_utterance_stft(wav_data_stft)["input_proc"]
    output = model.stft.get_istft(output)

    ta.save(args.output, output, fs_wav)
    print(f"saved output to {args.output}")


if __name__ == "__main__":
    main()
