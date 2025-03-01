import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from skimage import exposure
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt
import os
from PIL import Image

def read_tfw(tfw_path):
    """Read transformation parameters from a TFW file."""
    with open(tfw_path, 'r') as f:
        lines = f.readlines()
    
    # TFW format: x_scale, row_rotation, col_rotation, y_scale, x_origin, y_origin
    x_scale = float(lines[0])
    row_rotation = float(lines[1])
    col_rotation = float(lines[2])
    y_scale = float(lines[3])
    x_origin = float(lines[4])
    y_origin = float(lines[5])
    
    # Create an Affine transformation
    transform = Affine(x_scale, row_rotation, x_origin, 
                      col_rotation, y_scale, y_origin)
    
    return transform

def read_image_file(file_path):
    """Read an image file (JPEG, PNG) that doesn't have geospatial information."""
    try:
        # Try to open with rasterio first
        with rasterio.open(file_path) as src:
            data = src.read()
            meta = src.meta.copy()
            return data, meta, True
    except Exception as e:
        print(f"Could not open {file_path} with rasterio: {e}")
        print("Trying to open with PIL/Pillow instead...")
        
        # Open with PIL
        img = Image.open(file_path)
        img_array = np.array(img)
        
        # Convert to band-first format if needed
        if len(img_array.shape) == 3 and img_array.shape[2] in [3, 4]:  # RGB or RGBA
            img_array = np.moveaxis(img_array, 2, 0)  # Convert from HWC to CHW format
        elif len(img_array.shape) == 2:  # Grayscale
            img_array = img_array[np.newaxis, :, :]  # Add band dimension
            
        # Create a simple metadata dictionary
        meta = {
            'driver': 'GTiff',
            'dtype': str(img_array.dtype),
            'count': img_array.shape[0],
            'height': img_array.shape[1],
            'width': img_array.shape[2],
            'transform': None,  # Will be set later
            'crs': None  # Will be set later
        }
        
        return img_array, meta, False

def upscale_sentinel_using_lidar(sentinel_path, lidar_path, tfw_path, output_path):
    """
    Upscale Sentinel data (10m) using high-resolution lidar data (1m)
    
    Parameters:
    -----------
    sentinel_path : str
        Path to the Sentinel data (RGB or other bands)
    lidar_path : str
        Path to the lidar DSM data
    tfw_path : str
        Path to the TFW file for the lidar data
    output_path : str
        Path to save the upscaled result
    """
    # Read the lidar data
    with rasterio.open(lidar_path) as lidar_src:
        lidar_data = lidar_src.read(1)
        lidar_meta = lidar_src.meta.copy()
        lidar_crs = lidar_src.crs
        lidar_transform = lidar_src.transform
    
    # Read the Sentinel data
    sentinel_data, sentinel_meta, has_geospatial = read_image_file(sentinel_path)
    
    # Get number of bands
    num_bands = sentinel_data.shape[0]
    
    # Read the TFW transformation
    tfw_transform = read_tfw(tfw_path)
    
    # Update lidar metadata with the correct transformation if needed
    if lidar_meta['transform'] != tfw_transform:
        print("Updating lidar transformation from TFW file")
        lidar_meta['transform'] = tfw_transform
    
    # If sentinel data lacks geospatial information, create a simple one based on lidar
    if not has_geospatial or sentinel_meta['crs'] is None:
        print("Sentinel data lacks geospatial information. Creating a simplified transformation.")
        
        # Create a simplified transform that matches the extent of the lidar data
        sentinel_transform = Affine(
            lidar_transform.a * (lidar_data.shape[1] / sentinel_data.shape[2]),  # Scale x
            0.0,  # No rotation
            lidar_transform.c,  # Same origin x
            0.0,  # No rotation
            lidar_transform.e * (lidar_data.shape[0] / sentinel_data.shape[1]),  # Scale y
            lidar_transform.f   # Same origin y
        )
        
        sentinel_meta['transform'] = sentinel_transform
        sentinel_meta['crs'] = lidar_crs
    
    # Prepare output upscaled array
    upscaled_data = np.zeros((num_bands, lidar_data.shape[0], lidar_data.shape[1]), 
                            dtype=sentinel_data.dtype)
    
    # Reproject each band of the Sentinel data to the lidar resolution
    for i in range(num_bands):
        if has_geospatial:
            # Use reproject if we have proper geospatial data
            reproject(
                sentinel_data[i],
                upscaled_data[i],
                src_transform=sentinel_meta['transform'],
                src_crs=sentinel_meta['crs'],
                dst_transform=lidar_meta['transform'],
                dst_crs=lidar_meta['crs'],
                resampling=Resampling.bilinear
            )
        else:
            # For non-geospatial data, use simple resizing
            from skimage.transform import resize
            upscaled_data[i] = resize(
                sentinel_data[i],
                (lidar_data.shape[0], lidar_data.shape[1]),
                order=1,  # Bilinear
                preserve_range=True
            ).astype(sentinel_data.dtype)
    
    # Now enhance the upscaled data using the lidar information
    # Normalize lidar data to 0-1 range for easier processing
    lidar_min, lidar_max = lidar_data.min(), lidar_data.max()
    if lidar_min != lidar_max:
        lidar_norm = (lidar_data - lidar_min) / (lidar_max - lidar_min)
    else:
        lidar_norm = np.zeros_like(lidar_data)
    
    # Compute edge information from lidar using gradient magnitude
    edge_x = sobel(lidar_norm, axis=0)
    edge_y = sobel(lidar_norm, axis=1)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    
    # Normalize edge magnitude
    edge_min, edge_max = edge_mag.min(), edge_mag.max()
    if edge_min != edge_max:
        edge_mag = (edge_mag - edge_min) / (edge_max - edge_min)
    else:
        edge_mag = np.zeros_like(edge_mag)
    
    # Create enhanced upscaled data using edge information
    enhanced_data = np.zeros_like(upscaled_data)
    for i in range(num_bands):
        # Smooth the upscaled data
        smoothed = gaussian_filter(upscaled_data[i], sigma=0.5)
        
        # Add edge details from lidar
        # The weight controls how much edge detail to add
        data_min, data_max = smoothed.min(), smoothed.max()
        if data_min != data_max:
            weight = 0.3
            enhanced_data[i] = smoothed + weight * edge_mag * (data_max - data_min)
            
            # Ensure data is within original range
            enhanced_data[i] = np.clip(enhanced_data[i], data_min, data_max)
        else:
            enhanced_data[i] = smoothed
    
    # Prepare output metadata
    out_meta = sentinel_meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'height': lidar_data.shape[0],
        'width': lidar_data.shape[1],
        'transform': lidar_meta['transform'],
        'count': num_bands,
        'crs': lidar_meta['crs']
    })
    
    # Write the enhanced upscaled data
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(enhanced_data)
    
    return enhanced_data, upscaled_data, sentinel_data

def visualize_results(original, upscaled, enhanced, output_path="upscaling_comparison.png", bands=(0, 1, 2)):
    """
    Visualize the original, upscaled, and enhanced data
    For RGB visualization, use bands=(0, 1, 2)
    For false color (NIR), use appropriate band combination
    """
    # Create RGB composites for visualization
    # Handle single and multi-band data
    if original.shape[0] >= 3 and len(bands) >= 3:
        rgb_original = np.dstack([original[bands[i]] for i in range(3)])
        rgb_original = exposure.rescale_intensity(rgb_original, out_range=(0, 1))
    else:
        rgb_original = exposure.rescale_intensity(original[0], out_range=(0, 1))
    
    if upscaled.shape[0] >= 3 and len(bands) >= 3:
        rgb_upscaled = np.dstack([upscaled[bands[i]] for i in range(3)])
        rgb_upscaled = exposure.rescale_intensity(rgb_upscaled, out_range=(0, 1))
    else:
        rgb_upscaled = exposure.rescale_intensity(upscaled[0], out_range=(0, 1))
    
    if enhanced.shape[0] >= 3 and len(bands) >= 3:
        rgb_enhanced = np.dstack([enhanced[bands[i]] for i in range(3)])
        rgb_enhanced = exposure.rescale_intensity(rgb_enhanced, out_range=(0, 1))
    else:
        rgb_enhanced = exposure.rescale_intensity(enhanced[0], out_range=(0, 1))
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(rgb_original)
    axes[0].set_title('Original Sentinel Data')
    axes[0].axis('off')
    
    axes[1].imshow(rgb_upscaled)
    axes[1].set_title('Upscaled Data (Bilinear)')
    axes[1].axis('off')
    
    axes[2].imshow(rgb_enhanced)
    axes[2].set_title('Enhanced Upscaled Data (with Lidar details)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    
    print(f"Visualization saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    # Paths to the input files
    sentinel_path = "data/20230215-SE2B-CGG-GBR-MS3-L3-RGB-preview.jpg"  # Change to your Sentinel file
    lidar_path = "data/DSM_TQ0075_P_12757_20230109_20230315.tif"  # Change to your lidar file
    tfw_path = "data/DSM_TQ0075_P_12757_20230109_20230315.tfw"  # TFW file
    output_path = "upscaled_sentinel.tif"
    
    # Check if inputs exist and proceed
    if all(os.path.exists(p) for p in [sentinel_path, lidar_path, tfw_path]):
        try:
            # Perform upscaling
            enhanced_data, upscaled_data, original_data = upscale_sentinel_using_lidar(
                sentinel_path, lidar_path, tfw_path, output_path
            )
            
            # Visualize results
            visualize_results(original_data, upscaled_data, enhanced_data)
            
            print(f"Upscaled image saved to {output_path}")
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
    else:
        missing = [p for p in [sentinel_path, lidar_path, tfw_path] if not os.path.exists(p)]
        print(f"Missing input files: {missing}")
        print("Please provide the correct paths to the input files.")