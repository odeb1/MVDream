#!/bin/bash

# Call the ddim_inversion.py function with the first text input
ddim_inversion_command="python scripts/ddim_inversion.py --seed 2025 --text 'an orange tiger is standing still' --animal_name tiger_8_views_rest --num_frames 16 --camera_azim 135"
echo "Running ddim_inversion command: $ddim_inversion_command"
eval $ddim_inversion_command

# Set the base command with fixed arguments
base_command="python scripts/articulation.py --seed 2025 --inversion_seed 2025 --camera_azim 135 --animal_name tiger_8_views_rest --num_frames 16"

text_inputs=("A tiger is sitting.")

# Loop through the text inputs
for text in "${text_inputs[@]}"; do
  # Construct the full command with the current text input
  command="$base_command --text '$text' "

  # Execute the command
  echo "Running command: $command"
  eval $command
done

echo "All commands executed."