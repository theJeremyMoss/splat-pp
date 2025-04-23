#!/usr/bin/env python3
"""
Splatvert - Image to Black & White Pixel Art Converter

This script converts images to 320x120 black and white pixel art using
Floyd-Steinberg dithering. It handles transparency by converting transparent
pixels to white and includes options for image preprocessing.

Usage:
    python splatvert.py <input_image> [options]

Options:
    --output-path PATH    Output file path (default: input_name_converted.png)
    --size WxH           Output size in pixels (default: 320x120)
    --contrast FLOAT     Contrast adjustment factor (default: 1.2)
    --brightness FLOAT   Brightness adjustment factor (default: 1.0)
    --sharpness FLOAT   Sharpness adjustment factor (default: 1.3)
"""

from PIL import Image, ImageEnhance
import os
import argparse

def preprocess_image(img, contrast=1.2, brightness=1.0, sharpness=1.3):
    """
    Preprocess the image with adjustable enhancement parameters.
    
    Args:
        img: PIL Image object
        contrast: Contrast adjustment factor (default: 1.2)
        brightness: Brightness adjustment factor (default: 1.0)
        sharpness: Sharpness adjustment factor (default: 1.3)
    
    Returns:
        Preprocessed PIL Image object
    """
    # Handle transparency by converting to white background
    if img.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', img.size, 'white')
        if img.mode == 'RGBA':
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        else:
            background.paste(img, mask=img.split()[1])  # Use alpha channel as mask
        img = background

    # Convert to grayscale
    img = img.convert("L")
    
    # Apply enhancements
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    
    return img

def convert_to_bw_dither(input_path, output_path=None, size=(320, 120),
                        contrast=1.2, brightness=1.0, sharpness=1.3):
    """
    Convert an image to black and white pixel art using Floyd-Steinberg dithering.
    
    Args:
        input_path: Path to input image
        output_path: Path for output image (default: input_name_converted.png)
        size: Tuple of (width, height) for output image
        contrast: Contrast adjustment factor
        brightness: Brightness adjustment factor
        sharpness: Sharpness adjustment factor
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Generate default output path if none provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}_converted.png"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Open and preprocess the image
        img = Image.open(input_path)
        img = preprocess_image(img, contrast, brightness, sharpness)
        
        # Resize the image
        img = img.resize(size, Image.LANCZOS)
        
        # Apply Floyd-Steinberg dithering
        dithered = img.convert("1")  # Default PIL dithering is Floyd-Steinberg
        
        # Save the output
        dithered.save(output_path, format="PNG", optimize=True)
        return output_path
    
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def parse_size(size_str):
    """Parse size string in format WxH into tuple (width, height)."""
    try:
        width, height = map(int, size_str.lower().split('x'))
        return (width, height)
    except:
        raise argparse.ArgumentTypeError('Size must be in format WxH (e.g. 320x120)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_image', help='Input image file path')
    parser.add_argument('--output-path', help='Output file path')
    parser.add_argument('--size', type=parse_size, default='320x120',
                       help='Output size in format WxH (default: 320x120)')
    parser.add_argument('--contrast', type=float, default=1.2,
                       help='Contrast adjustment factor (default: 1.2)')
    parser.add_argument('--brightness', type=float, default=1.0,
                       help='Brightness adjustment factor (default: 1.0)')
    parser.add_argument('--sharpness', type=float, default=1.3,
                       help='Sharpness adjustment factor (default: 1.3)')
    
    args = parser.parse_args()
    
    try:
        output_path = convert_to_bw_dither(
            args.input_image,
            output_path=args.output_path,
            size=args.size,
            contrast=args.contrast,
            brightness=args.brightness,
            sharpness=args.sharpness
        )
        print(f"Conversion complete! Output saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
