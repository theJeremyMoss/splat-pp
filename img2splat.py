#!/usr/bin/env python3
"""
Img2Splat - Convert Images to Splatoon 3 Post Drawing Macros

This script converts black and white images to NXBT controller macros for drawing
pixel art in Splatoon 3's post creation feature. It includes multiple drawing modes
for optimized performance and a simulator to preview results without a Switch.

Usage:
    python img2splat.py <input_image> [output_macro.txt] [options]

Options:
    --mode MODE          Drawing mode: typewriter, snake, or smart (default: smart)
    --test-pattern PAT   Generate a test pattern: circle, square, or lines
    --wait-time FLOAT    Time to wait between commands in seconds (default: 0.05)
    --press-time FLOAT   Time to hold button presses in seconds (default: 0.1)
    --debug              Print additional debug information
    --simulate           Simulate macro execution and save preview image

Drawing Modes:
    typewriter - Original mode, processes image left-to-right, line by line
    snake      - Alternates between left-to-right and right-to-left drawing
    smart      - Optimized path finding with efficient pixel run detection
"""

import sys
import os
import argparse
from PIL import Image
import numpy as np
from enum import Enum
import math
import time

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
            
            # Add MINUS press at the end to save the artwork
            commands.append(f"MINUS {self.press_time}s")
            commands.append(f"{self.wait_time}s")
            
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
            
            # Find min and max X for this row to avoid unnecessary movement
            min_x, max_x = self._get_row_bounds(pixels, y, width)
            
            if min_x is None:  # No black pixels in this row
                continue
                
            # Process this row based on the current direction
            if direction == 1:  # Left to right
                # Move to min_x if not already there
                x_movement = self._move_to_x(current_x, min_x)
                commands.extend(x_movement)
                current_x = min_x
                
                for x in range(min_x, max_x + 1):
                    # Draw black pixels
                    if pixels[y, x] == 0:  # Black pixel
                        commands.append(f"A {self.press_time}s")
                        commands.append(f"{self.wait_time}s")
                    
                    # Move right if not at the end
                    if x < max_x:
                        commands.append(f"DPAD_RIGHT {self.press_time}s")
                        commands.append(f"{self.wait_time}s")
                        current_x += 1
            else:  # Right to left
                # Move to max_x if not already there
                x_movement = self._move_to_x(current_x, max_x)
                commands.extend(x_movement)
                current_x = max_x
                
                for x in range(max_x, min_x - 1, -1):
                    # Draw black pixels
                    if pixels[y, x] == 0:  # Black pixel
                        commands.append(f"A {self.press_time}s")
                        commands.append(f"{self.wait_time}s")
                    
                    # Move left if not at the start
                    if x > min_x:
                        commands.append(f"DPAD_LEFT {self.press_time}s")
                        commands.append(f"{self.wait_time}s")
                        current_x -= 1
            
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
                    # Fix: Always include a duration for button presses
                    commands.append(f"A {self.press_time}s")  # Press A with duration
                    commands.append(f"{self.wait_time}s")
                    
                    # Move right while holding A for each pixel after the first
                    for _ in range(start_x + 1, end_x + 1):
                        commands.append(f"DPAD_RIGHT+A {self.press_time}s")  # Hold both buttons
                        commands.append(f"{self.wait_time}s")
                    
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
    
    def _get_row_bounds(self, pixels, y, width):
        """Find the leftmost and rightmost black pixels in a row"""
        min_x = None
        max_x = None
        
        for x in range(min(width, self.max_width)):
            if pixels[y, x] == 0:  # Black pixel
                if min_x is None:
                    min_x = x
                max_x = x
        
        return min_x, max_x
    
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

    def simulate_macro(self, image_path, output_path=None):
        """
        Simulate a macro execution by creating a virtual canvas and showing the result
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the macro file
        """
        # Create a blank canvas (all white)
        canvas = np.ones((self.max_height, self.max_width), dtype=np.uint8)
        
        # Generate the macro but don't write it to file yet
        img = Image.open(image_path).convert('1')
        width, height = img.size
        pixels = np.array(img)
        
        # Generate commands according to the selected strategy
        if self.drawing_mode == DrawingMode.TYPEWRITER:
            commands = self._generate_typewriter_path(pixels, width, height)
        elif self.drawing_mode == DrawingMode.SNAKE:
            commands = self._generate_snake_path(pixels, width, height)
        elif self.drawing_mode == DrawingMode.SMART:
            commands = self._generate_smart_path(pixels, width, height)
        
        # Simulate drawing on the canvas
        print("Simulating macro execution...")
        sim_canvas, success = self._simulate_drawing(commands)
        
        if not success:
            print("Simulation failed! There may be issues with the macro.")
            return False
        
        # Compare with original image
        original_count = np.sum(pixels == 0)
        drawn_count = np.sum(sim_canvas == 0)
        accuracy = drawn_count / original_count if original_count > 0 else 0
        
        print(f"Simulation complete!")
        print(f"Original black pixels: {original_count}")
        print(f"Drawn black pixels: {drawn_count}")
        print(f"Accuracy: {accuracy:.2%}")
        
        # Save the simulated result
        sim_img = Image.fromarray(sim_canvas * 255)
        sim_output = f"{os.path.splitext(image_path)[0]}_simulated.png"
        sim_img.save(sim_output)
        print(f"Simulated result saved to: {sim_output}")
        
        # Now generate and save the actual macro file
        if output_path:
            self.image_to_macro(image_path, output_path)
        
        return True
    
    def _simulate_drawing(self, commands):
        """
        Simulate drawing commands on a virtual canvas
        
        Args:
            commands: List of macro commands
            
        Returns:
            Tuple of (canvas, success)
        """
        canvas = np.ones((self.max_height, self.max_width), dtype=np.uint8)
        current_x, current_y = 0, 0
        a_pressed = False
        success = True
        
        for cmd in commands:
            parts = cmd.split()
            if not parts:
                continue
                
            # Parse command
            if 'DPAD_RIGHT' in parts[0]:
                if '+A' in parts[0]:  # Combined button press
                    current_x = min(current_x + 1, self.max_width - 1)
                    canvas[current_y, current_x] = 0  # Draw black pixel
                else:
                    current_x = min(current_x + 1, self.max_width - 1)
                
            elif 'DPAD_LEFT' in parts[0]:
                current_x = max(current_x - 1, 0)
                
            elif 'DPAD_DOWN' in parts[0]:
                current_y = min(current_y + 1, self.max_height - 1)
                
            elif 'DPAD_UP' in parts[0]:
                current_y = max(current_y - 1, 0)
                
            elif parts[0] == 'A':
                if len(parts) == 1:  # This would be an error
                    print("Error: 'A' command without duration")
                    success = False
                else:
                    # Draw a pixel
                    canvas[current_y, current_x] = 0
        
        return canvas, success

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
                      
    parser.add_argument('--simulate', action='store_true',
                      help='Simulate the macro execution and show the result')
    
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
    
    # Generate a test pattern, simulate, or process an input image
    if args.test_pattern:
        converter.generate_test_pattern(args.test_pattern, args.output_macro)
    elif args.input_image:
        if args.simulate:
            converter.simulate_macro(args.input_image, args.output_macro)
        else:
            converter.image_to_macro(args.input_image, args.output_macro)
    else:
        # Default to a sample image path
        default_path = "img/image.png"
        if os.path.exists(default_path):
            if args.simulate:
                converter.simulate_macro(default_path)
            else:
                converter.image_to_macro(default_path)
        else:
            print(f"No input image specified and default '{default_path}' not found.")
            print("Usage: python img2splat.py [input_image] [output_macro.txt] [options]")
            print("Run python img2splat.py --help for more information.")