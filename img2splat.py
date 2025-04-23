#!/usr/bin/env python3
import sys
import os
import argparse
from PIL import Image
import numpy as np
from enum import Enum
import math

class DrawingMode(Enum):
    TYPEWRITER = "typewriter"  # Original mode - left to right, return to start on each row
    SNAKE = "snake"  # Alternate left-to-right and right-to-left to avoid return trips
    SMART = "smart"  # Optimize path between black pixels

class Img2Splat:
    def __init__(self, 
                 drawing_mode=DrawingMode.SMART, 
                 wait_time=0.05,
                 press_time=0.1,
                 debug=False):
        """
        Initialize the image to macro converter with configurable parameters
        
        Args:
            drawing_mode: The strategy to use when plotting pixels
            wait_time: Time to wait between commands in seconds
            press_time: Time to hold button presses in seconds
            debug: Whether to print debug information
        """
        self.drawing_mode = drawing_mode
        self.wait_time = wait_time
        self.press_time = press_time
        self.debug = debug
        self.max_width = 320
        self.max_height = 120
        
    def generate_test_pattern(self, pattern, output_path):
        """
        Generate a test pattern image and convert it to a macro
        
        Args:
            pattern: The type of test pattern to generate ('circle', 'square', 'lines')
            output_path: Path to save the macro file
        """
        width, height = self.max_width, self.max_height
        img = Image.new('1', (width, height), color=1)  # 1 = white background
        pixels = np.array(img)
        
        if pattern == 'circle':
            # Draw a circle in the center
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            for y in range(height):
                for x in range(width):
                    if math.sqrt((x - center_x)**2 + (y - center_y)**2) <= radius:
                        pixels[y, x] = 0  # Set pixel to black
        
        elif pattern == 'square':
            # Draw a square in the center
            center_x, center_y = width // 2, height // 2
            size = min(width, height) // 4
            for y in range(center_y - size, center_y + size):
                for x in range(center_x - size, center_x + size):
                    if 0 <= y < height and 0 <= x < width:
                        pixels[y, x] = 0  # Set pixel to black
        
        elif pattern == 'lines':
            # Draw horizontal, vertical and diagonal lines
            # Horizontal line
            mid_y = height // 2
            for x in range(width):
                pixels[mid_y, x] = 0
                
            # Vertical line
            mid_x = width // 2
            for y in range(height):
                pixels[y, mid_x] = 0
                
            # Diagonal lines
            for i in range(min(width, height)):
                if i < width and i < height:
                    pixels[i, i] = 0  # Top-left to bottom-right
                if i < width and height - i - 1 >= 0:
                    pixels[height - i - 1, i] = 0  # Bottom-left to top-right
        
        # Convert back to image and save
        test_img = Image.fromarray(pixels.astype(np.uint8) * 255)
        test_img_path = f"test_{pattern}.png"
        test_img.save(test_img_path)
        print(f"Generated test pattern '{pattern}' and saved to {test_img_path}")
        
        # Convert the test pattern to a macro
        return self.image_to_macro(test_img_path, output_path, pixels=pixels)
        
    def image_to_macro(self, image_path, output_path=None, pixels=None):
        """
        Convert a black-and-white PNG image to an NXBT macro for Splatoon 3 posts.
        
        Args:
            image_path (str): Path to the PNG image
            output_path (str, optional): Path to save the macro file. If None, will use same name as image
            pixels (numpy.array, optional): Pre-loaded pixel data, if available
        """
        try:
            # Load image if pixels aren't provided
            if pixels is None:
                img = Image.open(image_path).convert('1')
                width, height = img.size
                print(f"Image dimensions: {width}x{height}")
                
                if width > self.max_width or height > self.max_height:
                    print(f"Warning: Image dimensions exceed {self.max_width}x{self.max_height}, it may be cropped.")
                
                # Convert to numpy array for easier processing (0 = black, 1 = white)
                pixels = np.array(img)
            else:
                height, width = pixels.shape
            
            # Determine output path
            if output_path is None:
                base_name = os.path.splitext(image_path)[0]
                output_path = f"{base_name}_macro.txt"
            
            # Generate the path according to the selected strategy
            commands = self._initial_setup_commands()
            
            if self.drawing_mode == DrawingMode.TYPEWRITER:
                path_commands = self._generate_typewriter_path(pixels, width, height)
            elif self.drawing_mode == DrawingMode.SNAKE:
                path_commands = self._generate_snake_path(pixels, width, height)
            elif self.drawing_mode == DrawingMode.SMART:
                path_commands = self._generate_smart_path(pixels, width, height)
            else:
                raise ValueError(f"Unknown drawing mode: {self.drawing_mode}")
                
            commands.extend(path_commands)
            
            # Write to file
            with open(output_path, 'w') as f:
                for cmd in commands:
                    f.write(cmd + '\n')
            
            # Calculate statistics
            total_time = self._calculate_execution_time(commands)
            
            print(f"Macro created successfully at: {output_path}")
            print(f"Total commands: {len(commands)}")
            print(f"Drawing mode: {self.drawing_mode.value}")
            print(f"Estimated execution time: {total_time/60:.2f} minutes")
            
            if self.debug:
                self._print_drawing_debug_info(pixels)
            
            return True
        
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
    
    def _initial_setup_commands(self):
        """
        Generate the initial setup commands for the macro
        """
        commands = []
        # Add initial setup - wait for connection, press A to close pairing window
        commands.append("10s")
        commands.append(f"A {self.press_time}s")
        commands.append(f"{self.wait_time}s")
        
        # Additional wait to ensure drawing area is ready
        commands.append("5s")
        return commands
    
    def _generate_typewriter_path(self, pixels, width, height):
        """
        Generate path commands using the original typewriter approach
        """
        commands = []
        current_x, current_y = 0, 0
        
        # Process each row
        for y in range(min(height, self.max_height)):
            # Skip empty rows
            if self._is_empty_row(pixels, y, width):
                if self.debug:
                    print(f"Skipping empty row {y}")
                continue
                
            # Move to the correct Y position
            y_movement = self._move_to_y(current_y, y)
            commands.extend(y_movement)
            current_y = y
            
            # Reset to the left edge if not already there
            while current_x > 0:
                commands.append(f"DPAD_LEFT {self.press_time}s")
                commands.append(f"{self.wait_time}s")
                current_x -= 1
            
            # Process this row pixel by pixel
            for x in range(min(width, self.max_width)):
                # Move to the correct X position
                while current_x < x:
                    commands.append(f"DPAD_RIGHT {self.press_time}s")
                    commands.append(f"{self.wait_time}s")
                    current_x += 1
                
                # Draw black pixels
                if pixels[y, x] == 0:  # Black pixel
                    commands.append(f"A {self.press_time}s")
                    commands.append(f"{self.wait_time}s")
        
        return commands
    
    def _generate_snake_path(self, pixels, width, height):
        """
        Generate path commands using snake pattern (alternate left-to-right and right-to-left)
        """
        commands = []
        current_x, current_y = 0, 0
        direction = 1  # 1 = right, -1 = left
        
        # Process each row
        for y in range(min(height, self.max_height)):
            # Skip empty rows
            if self._is_empty_row(pixels, y, width):
                if self.debug:
                    print(f"Skipping empty row {y}")
                continue
                
            # Move to the correct Y position
            y_movement = self._move_to_y(current_y, y)
            commands.extend(y_movement)
            current_y = y
            
            # Process this row based on the current direction
            if direction == 1:  # Left to right
                x_range = range(min(width, self.max_width))
            else:  # Right to left
                x_range = range(min(width, self.max_width) - 1, -1, -1)
            
            for x in x_range:
                # Move to the correct X position
                x_movement = self._move_to_x(current_x, x)
                commands.extend(x_movement)
                current_x = x
                
                # Draw black pixels
                if pixels[y, x] == 0:  # Black pixel
                    commands.append(f"A {self.press_time}s")
                    commands.append(f"{self.wait_time}s")
            
            # Flip direction for the next row
            direction *= -1
        
        return commands
    
    def _generate_smart_path(self, pixels, width, height):
        """
        Generate an optimized path between black pixels
        """
        commands = []
        current_x, current_y = 0, 0
        
        # Find all black pixels
        black_pixels = []
        for y in range(min(height, self.max_height)):
            for x in range(min(width, self.max_width)):
                if pixels[y, x] == 0:  # Black pixel
                    black_pixels.append((x, y))
        
        if not black_pixels:
            return commands  # No black pixels to draw
            
        # Sort black pixels by row first (to minimize Y movement)
        black_pixels.sort(key=lambda p: (p[1], p[0]))
        
        # Group pixels by row for more efficient processing
        row_groups = {}
        for x, y in black_pixels:
            if y not in row_groups:
                row_groups[y] = []
            row_groups[y].append(x)
        
        # Process each row with black pixels
        for y in sorted(row_groups.keys()):
            # Move to this row
            y_movement = self._move_to_y(current_y, y)
            commands.extend(y_movement)
            current_y = y
            
            # Sort X coordinates and find runs of consecutive pixels
            row_pixels = sorted(row_groups[y])
            runs = self._find_pixel_runs(row_pixels)
            
            # Process each run of pixels
            for run in runs:
                start_x, end_x = run
                
                # Move to the start of this run
                x_movement = self._move_to_x(current_x, start_x)
                commands.extend(x_movement)
                current_x = start_x
                
                # If it's a single pixel
                if start_x == end_x:
                    commands.append(f"A {self.press_time}s")
                    commands.append(f"{self.wait_time}s")
                else:
                    # For a run, press A and hold while moving right
                    commands.append("A")  # Press A without releasing
                    
                    # Move right to cover the run
                    for _ in range(end_x - start_x):
                        commands.append(f"DPAD_RIGHT {self.press_time}s")
                        commands.append(f"{self.wait_time}s")
                    
                    # Release A
                    commands.append(f"{self.wait_time}s")
                    commands.append("A 0s")  # Release A
                    
                    current_x = end_x
        
        return commands
    
    def _find_pixel_runs(self, sorted_x_coords):
        """
        Find runs of consecutive pixel X coordinates
        
        Args:
            sorted_x_coords: List of X coordinates sorted in ascending order
            
        Returns:
            List of tuples (start_x, end_x) for each run
        """
        if not sorted_x_coords:
            return []
            
        runs = []
        start_x = sorted_x_coords[0]
        prev_x = start_x
        
        for x in sorted_x_coords[1:]:
            if x == prev_x + 1:
                # Continue the current run
                prev_x = x
            else:
                # End the current run and start a new one
                runs.append((start_x, prev_x))
                start_x = x
                prev_x = x
        
        # Add the last run
        runs.append((start_x, prev_x))
        return runs
    
    def _move_to_y(self, current_y, target_y):
        """Generate commands to move from current Y to target Y"""
        commands = []
        
        while current_y < target_y:
            commands.append(f"DPAD_DOWN {self.press_time}s")
            commands.append(f"{self.wait_time}s")
            current_y += 1
            
        while current_y > target_y:
            commands.append(f"DPAD_UP {self.press_time}s")
            commands.append(f"{self.wait_time}s")
            current_y -= 1
            
        return commands
    
    def _move_to_x(self, current_x, target_x):
        """Generate commands to move from current X to target X"""
        commands = []
        
        while current_x < target_x:
            commands.append(f"DPAD_RIGHT {self.press_time}s")
            commands.append(f"{self.wait_time}s")
            current_x += 1
            
        while current_x > target_x:
            commands.append(f"DPAD_LEFT {self.press_time}s")
            commands.append(f"{self.wait_time}s")
            current_x -= 1
            
        return commands
    
    def _is_empty_row(self, pixels, y, width):
        """Check if a row has no black pixels"""
        for x in range(min(width, self.max_width)):
            if pixels[y, x] == 0:  # Found a black pixel
                return False
        return True
    
    def _calculate_execution_time(self, commands):
        """Calculate the total execution time of the macro in seconds"""
        total_time = 0
        for cmd in commands:
            parts = cmd.split()
            
            # Handle time specifications
            if len(parts) > 0 and parts[0].endswith('s') and parts[0][:-1].replace('.', '', 1).isdigit():
                # Command starts with a time like "10s"
                total_time += float(parts[0][:-1])
            elif len(parts) > 1 and parts[1].endswith('s') and parts[1][:-1].replace('.', '', 1).isdigit():
                # Command has a time as second parameter like "A 0.1s"
                total_time += float(parts[1][:-1])
        
        return total_time
    
    def _print_drawing_debug_info(self, pixels):
        """Print debug information about the drawing"""
        height, width = pixels.shape
        black_count = np.sum(pixels == 0)
        
        print(f"\nDEBUG INFO:")
        print(f"Total black pixels: {black_count}")
        
        # Count non-empty rows
        non_empty_rows = 0
        for y in range(min(height, self.max_height)):
            if not self._is_empty_row(pixels, y, width):
                non_empty_rows += 1
        
        print(f"Non-empty rows: {non_empty_rows}/{min(height, self.max_height)}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Convert a black-and-white PNG image to an NXBT macro for Splatoon 3 posts'
    )
    
    parser.add_argument('input_image', nargs='?', default=None,
                      help='Path to the input PNG image')
                      
    parser.add_argument('output_macro', nargs='?', default=None,
                      help='Path to save the output macro file')
                      
    parser.add_argument('--mode', choices=['typewriter', 'snake', 'smart'], default='smart',
                      help='Drawing mode: typewriter (original), snake (alternating), or smart (optimized)')
                      
    parser.add_argument('--test-pattern', choices=['circle', 'square', 'lines'], default=None,
                      help='Generate a test pattern instead of processing an input image')
                      
    parser.add_argument('--wait-time', type=float, default=0.05,
                      help='Time to wait between commands in seconds')
                      
    parser.add_argument('--press-time', type=float, default=0.1,
                      help='Time to hold button presses in seconds')
                      
    parser.add_argument('--debug', action='store_true',
                      help='Print debug information')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create the converter with the specified settings
    mode = DrawingMode(args.mode)
    converter = Img2Splat(
        drawing_mode=mode,
        wait_time=args.wait_time,
        press_time=args.press_time,
        debug=args.debug
    )
    
    # Generate a test pattern or process an input image
    if args.test_pattern:
        converter.generate_test_pattern(args.test_pattern, args.output_macro)
    elif args.input_image:
        converter.image_to_macro(args.input_image, args.output_macro)
    else:
        # Default to a sample image path
        default_path = "img/image.png"
        if os.path.exists(default_path):
            converter.image_to_macro(default_path)
        else:
            print(f"No input image specified and default '{default_path}' not found.")
            print("Usage: python img2splat.py [input_image] [output_macro.txt] [options]")
            print("Run python img2splat.py --help for more information.")