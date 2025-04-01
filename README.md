## Target Image Generation

``` bash
pip install -e . # just the first time, double check to ensure that the env files are indeed updated  

python scripts/ddim_inversion.py --seed 2222 --text "a tiger standing still" --animal_name tiger_8 --num_frames 16

python scripts/articulation.py --seed 2222 --inversion_seed 2222 --text "a tiger running, four legs off ground" --animal_name tiger_twelve --num_frames 24

```
