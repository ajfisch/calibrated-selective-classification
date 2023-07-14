import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str)
parser.add_arguemnt('--output-dir', type=str)
args = parser.parse_args()

for corruption in os.listdir(args.input_dir):
    for severity in os.listdir(os.path.join(args.input_dir, corruption)):
        input_dir = os.path.join(args.input_dir, corruption, severity)
        for class_name in os.listdir(input_dir):
            output_dir = os.path.join(args.output_dir, corruption, class_name)
            os.makedirs(output_dir, exist_ok=True)
            for filename in os.listdir(os.path.join(input_dir, class_name)):
                src = os.path.join(input_dir, class_name, filename)
                dest = os.path.join(output_dir, f"{severity}_{filename}")
                print(f"{src} --> {dest}")
                shutil.copyfile(src, dest)
