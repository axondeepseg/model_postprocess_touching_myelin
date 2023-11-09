# nn_touching_myelin

This repository is dedicated to the task of converting the Touching Myelin Boundary Detection Dataset from the BIDS format to the format expected by nnUNetv2. The dataset can be found at [this location](https://github.com/axondeepseg/data_touching_myelin).

The repository also includes a training script for training the model with nnUNetv2.

The ultimate goal of this project is to enhance the instance segmentation as discussed in the following issues:
- [Issue #768](https://github.com/axondeepseg/axondeepseg/issues/768)
- [Discussion #766](https://github.com/axondeepseg/axondeepseg/discussions/766)

## Repository Structure

The repository contains the following key components:

- **Conversion Script**: This script is responsible for converting the Touching Myelin Boundary Detection Dataset from the BIDS format to the format expected by nnUNetv2. To run execute the following command: 
```bash
python scripts/convert_from_bids_to_nnunetv2_format.py <PATH/TO/ORIGINAL/DATASET> --TARGETDIR <PATH/TO/NEW/DATASET>
```
## Getting Started

To set up the environment and run the scripts, follow these steps:

1. Set up the environment from the `environment.yaml` file:
```bash
conda env create -f environment.yaml
```
2. Activate the environment:
```bash
conda activate touching_myelin
```
3. Run the conversion script (the default target directory is the current working directory):
```bash
python scripts/convert_from_bids_to_nnunetv2_format.py <PATH/TO/ORIGINAL/DATASET>
```
4. Set up the necessary environment variables:
```bash
export nnUNet_raw="$(pwd)/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/nnUNet_results"
```
5. Run the nnUNet preprocessing command:
```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```


## Contributing

We welcome contributions to this project. If you have a feature request, bug report, or proposal, please open an issue on our GitHub page.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.