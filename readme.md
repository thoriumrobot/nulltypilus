Dependencies:

pip install -r requirements.txt

Extract TDGs from the dataset:

python extract_tdg.py ProjectDir OutputDir

Training command:

python train_typilus.py JsonOutputDir ModelOutputPath

Placement command:

python predict.py ProjectDir ModelPath OutputDir

