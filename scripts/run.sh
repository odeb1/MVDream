# python scripts/ddim_inversion.py --seed 2222 --text "a elephant" --animal_name elephant --num_frames 8
for ((d=5; d<50; d++)); do
   echo $d
   python scripts/articulation.py --seed 2222 --text "an grey elephant running jumping nose down" --animal_name elephant --inversion_seed 2222 --num_frames 8 --camera_azim 315 --ddim_depth $d
   ((d = d + 4))
done


# python scripts/ddim_inversion.py --seed 2222 --text "a tiger" --animal_name tiger --num_frames 8
for ((d=5; d<50; d++)); do
   echo $d
   python scripts/articulation.py --seed 2222 --text "an orange tiger running jumping" --animal_name tiger --inversion_seed 2222 --num_frames 8 --camera_azim 315 --ddim_depth $d
   ((d = d + 4))
done

python scripts/ddim_inversion.py --seed 2222 --text "a light grey kangaroo" --animal_name kangaroo --num_frames 8
for ((d=5; d<50; d++)); do
   echo $d
   python scripts/articulation.py --seed 2222 --text "a light grey kangaroo jumping" --animal_name kangaroo --inversion_seed 2222 --num_frames 8 --camera_azim 315 --ddim_depth $d
   ((d = d + 4))
done
