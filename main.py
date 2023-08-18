from src import main_helper as mh
import constants as constants
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--shift", type=str, default="base", help="pass thye ctr shift")
parser.add_argument("--gpu", type=int, default=0, help="pass the gpu number")

args = parser.parse_args()
shift = args.shift
gpu = args.gpu
device = f"cuda:{gpu}"

shift = constants.FOREST

model = mh.get_model(model_name="resnet50", pretrn=True)

ds, dl = mh.get_ds_dl(dataset_name=shift)

acc_meter: mh.AccuracyMeter = mh.evaluate_model(
    model=model, loader=dl, device=device, cache=True
)

print(f"Accuracy: {acc_meter.accuracy()}")
print(f"Classwise Accuracy: {acc_meter.classwise_accuracy()}")
