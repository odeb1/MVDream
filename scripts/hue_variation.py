import os
import shutil
import cv2
import numpy as np
from pathlib import Path

def create_hue_variants(source_dir, base_output_dir, num_variants=5):
    """
    Create hue-shifted variants of images in the source directory.
    
    Args:
        source_dir: Path to the source directory containing images
        base_output_dir: Base path for output directories (without variant suffix)
        num_variants: Number of hue variants to create
    """
    source_path = Path(source_dir)
    
    # Get list of all PNG files matching the pattern
    png_files = list(source_path.glob('tiger_*.png'))
    
    if not png_files:
        print(f"No PNG files found in {source_dir}")
        return
    
    # Calculate hue shifts (equally spaced in 360 degrees)
    hue_shifts = [int(360 * i / num_variants) for i in range(num_variants)]
    
    for variant_idx, hue_shift in enumerate(hue_shifts, 1):
        # Create variant directory
        variant_dir = f"{base_output_dir}_variant_{variant_idx}"
        os.makedirs(variant_dir, exist_ok=True)
        
        print(f"Creating variant {variant_idx} with hue shift: {hue_shift}°")
        
        for png_file in png_files:
            # Read image
            img = cv2.imread(str(png_file))
            
             # Convert to HSV for hue manipulation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Increase brightness slightly (10% increase in Value channel)
            hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.4, 0, 255)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 0.7, 0, 255)
            
            # Shift hue
            hsv[:,:,0] = (hsv[:,:,0] + hue_shift) % 180  # OpenCV uses 0-180 for hue
            
            # Convert back to BGR
            hsv = hsv.astype(np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Save to variant directory
            output_path = Path(variant_dir) / png_file.name
            cv2.imwrite(str(output_path), bgr)
        
        print(f"Variant {variant_idx} completed: {len(png_files)} images processed")
    
    print(f"\nAll {num_variants} variants created successfully!")

# Example usage
if __name__ == "__main__":
    source_directory = "./assets/renderings/tiger_8_views_rest"
    output_base = "./assets/renderings/tiger_8_views_rest"
    
    create_hue_variants(source_directory, output_base, num_variants=5)