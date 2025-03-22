import numpy as np
from cellpose import models, io
import os, pdb
import tifffile
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_process import STACK_INF0_3D, STACK_INF0_Registration

# Initialize Cellpose logger
io.logger_setup()

def parse_args():
    parser = argparse.ArgumentParser(description="Plant Cell Segmentation using Cellpose")
    
    parser.add_argument("--main_dir", type=str, default="data", help="Main directory containing plant data")
    parser.add_argument("--plant_idx", type=int, default=6, help="Index of the plant in the dataset list")
    parser.add_argument("--segmentation", type=str, default="cellpose", help="Segmentation method name")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    plant_names = ['plant_1', 'plant_2', 'plant_4', 'plant_13', 'plant_15', 'plant_18', 'test_plant']
    
    if args.plant_idx < 0 or args.plant_idx >= len(plant_names):
        raise ValueError(f"Invalid plant index {args.plant_idx}. Choose between 0 and {len(plant_names) - 1}.")
    
    data_dir = os.path.join(args.main_dir, plant_names[args.plant_idx], "microscopic_images")
    segmentation_dir = os.path.join(args.main_dir, plant_names[args.plant_idx], args.segmentation, "seg")

    print(f"Data directory: {data_dir}")
    print(f"Segmentation directory: {segmentation_dir}")

    os.makedirs(segmentation_dir, exist_ok=True)

    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist!")

    files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]

    if not files:
        raise ValueError(f"No TIFF files found in {data_dir}.")

    # Initialize Cellpose model
    model = models.Cellpose(gpu=True, model_type="cyto3")  # Modify model type if needed

    for file_name in tqdm(files, desc="Processing Images"):
        file_path = os.path.join(data_dir, file_name)
        data = tifffile.imread(file_path)

        # Define channels for segmentation
        channels = [0, 0]  # Modify based on your dataset

        # Run segmentation on the 3D dataset
        masks, flows, styles, diams = model.eval(
            data,
            channels=channels,
            diameter=None,  # Automatic size estimation
            do_3D=True      # Enable 3D segmentation
        )

        # Save segmentation masks
        output_mask_path = os.path.join(segmentation_dir, f"{file_name}_mask.tif")
        tifffile.imwrite(output_mask_path, masks)

    # Data Pre-Processing
    STACK_INF0_3D(args.main_dir, plant_names, plant_idx=[args.plant_idx], segmentation=args.segmentation)
    STACK_INF0_Registration(args.main_dir, plant_names, plant_idx=[args.plant_idx], segmentation=args.segmentation)

if __name__ == "__main__":
    main()
