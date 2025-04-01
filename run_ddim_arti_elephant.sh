#!/bin/bash

# Define a list of start_time_step values
# start_time_steps=(32 33 34 35 36 37 38)

start_time_steps=(30)

# Define a list of text inputs
text_inputs=("a grey elephant is standing, rest pose, 3d, blender")

# Set the base command with fixed arguments
base_command="python scripts/articulation.py --seed 2015 --inversion_seed 2015 --camera_azim 135"

# Loop through the start_time_step values
for start_time_step in "${start_time_steps[@]}"; do
  # Call the ddim_inversion.py function with the current start_time_step
  ddim_inversion_command="python scripts/ddim_inversion.py --seed 2015 --text 'a grey elephant standing with trunk raising up.' --animal_name elephant_8_views --num_frames 16 --camera_azim 135 --start_time_step $start_time_step"
  echo "Running ddim_inversion command: $ddim_inversion_command"
  eval $ddim_inversion_command

  # Loop through the text inputs
  for text in "${text_inputs[@]}"; do
    # Construct the full command with the current text input and start_time_step
    command="$base_command --text '$text' --animal_name elephant_8_views --num_frames 16"

    # Execute the command
    echo "Running command: $command"
    eval $command
  done
done

echo "All commands executed."