## Installation

``` bash
conda create -n artic python=3.10
conda activate artic
# conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```
Please make sure that `xformers` are NOT installed.

## Target Image Generation

``` bash
python scripts/ddim_inversion.py --seed 2222 --text "a tiger standing still" --animal_name tiger_8_views_rest --num_frames 16
python scripts/ddim_inversion.py --seed 2222 --text "a hummingbird resting" --animal_name humming_8_views --num_frames 16 
python scripts/ddim_inversion.py --seed 2222 --text "a giraffe standing still" --animal_name giraffe --num_frames 8

python scripts/articulation.py --seed 2222 --inversion_seed 2222 --text "a tiger running, four legs off ground" --animal_name tiger_8_views_rest --num_frames 16 --run_ddim_depth_ablation
python scripts/articulation.py --seed 2222 --inversion_seed 2222 --text "a hummingbird flapping wings, flying" --animal_name humming_8_views --num_frames 16 --run_ddim_depth_ablation
python scripts/articulation.py --seed 2222 --inversion_seed 2222 --text "a giraffe running, four legs off ground" --animal_name giraffe --num_frames 8 --run_ddim_depth_ablation

python scripts/ddim_depth_ablation_plot.py --seed 2222 --inversion_seed 2222 --text "a tiger running, four legs off ground" --animal_name tiger_8_views_rest --num_frames 16
```
