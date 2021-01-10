import click
import numpy as np
from PIL import Image
import sys
import time
import os
from shutil import copyfile
import matplotlib.image as mpimg


@click.command()

@click.argument('bin_path', type=click.Path(exists=True))
@click.argument('crack_path', type=click.Path(exists=True))
@click.argument('new_crack_path', type=click.Path(exists=True))
@click.argument('remaining_crack_path', type=click.Path(exists=True))

def copy_non_corner_crack_files(bin_path, crack_path, new_crack_path, remaining_crack_path):
    print("Filtering", os.path.join(crack_path), "into", new_crack_path, "based on", os.path.join(bin_path))
    
    for filename in os.listdir(bin_path):
        crack_folder_filepath = os.path.join(crack_path, filename)
        filtered_crack_folder_filepath = os.path.join(new_crack_path, filename)
        remaining_crack_folder_filepath = os.path.join(remaining_crack_path, filename)
        
        img = mpimg.imread(os.path.join(bin_path, filename))
        if not has_corner_crack(os.path.join(new_crack_path, '2' + filename), img) and not "inverted" in filename:
            copyfile(crack_folder_filepath, filtered_crack_folder_filepath)
        else:
            copyfile(crack_folder_filepath, remaining_crack_folder_filepath)   
    

def has_corner_crack(new_crack_folder_filepath, x: np.array, id: int = 0):
    """ As long as there are no non-black pixels in the center of the patch, continue
        If there are, this patch does not contain a corner crack."""
        
    np.set_printoptions(threshold=sys.maxsize)
    corner_range = range(16, 48)
    pixel_found = False  # is there a non-corner crack pixel?
    normal_cracks = np.array([])
    for i, row in enumerate(x):
        if i in corner_range and not pixel_found:       
            for j, pixel in enumerate(row):
                if j in corner_range and not pixel_found:       
                    if pixel > 0.0:
                        pixel_found = True
                        return not pixel_found
                  
    """if pixel_found:
        img = Image.fromarray(x*255)
        img = img.convert('1')
        img.save(new_crack_folder_filepath, "PNG")"""
    return not pixel_found
    

if __name__ == '__main__':
    copy_non_corner_crack_files()      