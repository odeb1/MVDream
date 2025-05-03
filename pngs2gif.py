import os
from PIL import Image

def pngs_to_gif(input_directory, output_gif_name, duration=200, startswith='x_inter'):
    # Get all PNG images from the directory
    png_files = [f for f in os.listdir(input_directory) if (f.endswith('.png') and f.startswith(startswith) and '15' not in f)]
    png_files.sort()  # Sort images to keep the order
    print(png_files)

    # Load images
    images = [Image.open(os.path.join(input_directory, file)) for file in png_files]

    # Save the images as a GIF
    if images:
        images[0].save(
            output_gif_name,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved as {output_gif_name}")
    else:
        print("No PNG files found in the directory.")


if __name__ == '__main__':
    input_directory = 'ddim_inv_artefacts'  # Replace with your directory
    output_gif_name = 'rendered_standing_cow_add_noise.gif'  # Replace with desired output GIF name
    pngs_to_gif(input_directory, output_gif_name)

    input_directory = '/scratch/local/ssd/anjun/consistency/MVDream/forward_50_run/'  # Replace with your directory
    for startswith in ['x_inter', 'pred_x0']:
        output_gif_name = f'forward_50_run_{startswith}.gif'  # Replace with desired output GIF name
        pngs_to_gif(input_directory, output_gif_name, startswith=startswith)
