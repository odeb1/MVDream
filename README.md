## Target Image Generation

``` bash
pip install -e . # just the first time, double check to ensure that the env files are indeed updated  

python scripts/ddim_inversion.py --seed 2025 --text "tiger streching legs off ground" --animal_name tiger_hunting

python scripts/t2i_customized.py --seed 2025 --inversion_seed 2025 --text "an tiger streching four legs off ground" --animal_name tiger_hunting 

```