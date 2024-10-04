import glob
import os

import torch
import torch.onnx
from pytorch_lightning import seed_everything

from bmfwf import models
import argparse

DEVICE = "cpu"
torch.set_default_device(DEVICE)
torch.backends.quantized.engine = "x86"

print(f"available quantization engines: {torch.backends.quantized.supported_engines}")
print(f"using: {torch.backends.quantized.engine}")

parser = argparse.ArgumentParser(description="Export models to TorchScript")
parser.add_argument(
    "--ckpt_dir", type=str, help="Path to the checkpoint directory", default="bmfwf/saved/20231018_090629_yellow-dish-5"
)
args = parser.parse_args()

checkpoint_path = glob.glob(os.path.join(args.ckpt_dir, "epoch*.ckpt"))[0]
seed_everything(1337)

model_batch = models.BMWF.load_from_checkpoint(
    checkpoint_path, map_location=DEVICE, batch_size=1, use_mwf=True
)
model_realtime = models.RealTimeBMWF(**model_batch.hparams)
model_realtime.load_state_dict(model_batch.state_dict())
model_batch.eval()
model_realtime.eval()
model_batch.requires_grad_(False)
model_realtime.requires_grad_(False)
config = model_realtime.hparams
del model_realtime.criterion
model_realtime.states = model_realtime.get_initial_states()
script_model = model_realtime.to_torchscript(method="script")
quantized_model = torch.quantization.quantize_dynamic(  # sacrificing some accuracy for speed
    model_realtime,
    {
        torch.nn.GRU,
        torch.nn.Linear,
    },
    dtype=torch.qint8,
).to_torchscript(method="script")
torch.jit.save(script_model, f"{args.ckpt_dir}/model_bmfwf.pt")
torch.jit.save(quantized_model, f"{args.ckpt_dir}/model_bmfwf_qint8.pt")

print(f"finished exporting models to {args.ckpt_dir}")