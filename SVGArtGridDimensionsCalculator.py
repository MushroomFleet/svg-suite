#!/usr/bin/env python3
"""
SVGArtGridDimensionsCalculator.py - A ComfyUI node that calculates optimal rows and columns
for SVGArtGridV2 node based on desired width and height.
"""

import torch
import math

class SVGArtGridDimensionsCalculator:
    """
    Calculates optimal rows and columns for SVGArtGridV2 based on desired dimensions.
    This helps maintain proper aspect ratio and dimensions when using SVGArtGridV2.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 1}),
                "min_cells": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1, 
                                     "tooltip": "Minimum number of cells in either dimension"}),
                "max_cells": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1,
                                     "tooltip": "Maximum number of cells in either dimension"}),
                "method": (["aspect_ratio", "fixed_cell_size", "min_cells", "max_cells"], 
                          {"default": "aspect_ratio"})
            },
            "optional": {
                "target_cell_size": ("INT", {"default": 64, "min": 8, "max": 512, "step": 1,
                                           "tooltip": "Target size of each cell in pixels (for fixed_cell_size method)"})
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("rows", "cols")
    FUNCTION = "calculate_dimensions"
    CATEGORY = "SVG/ArtGrid"
    
    def calculate_aspect_ratio_dimensions(self, width, height, min_cells, max_cells):
        """
        Calculate dimensions based on maintaining aspect ratio.
        This will try to find rows and cols that preserve the original aspect ratio.
        """
        aspect_ratio = width / height
        
        # Start with a reasonable number of columns
        best_error = float('inf')
        best_rows = min_cells
        best_cols = min_cells
        
        # Try different column counts and find the one that gives closest to desired aspect ratio
        for cols in range(min_cells, max_cells + 1):
            # Calculate ideal rows based on aspect ratio
            ideal_rows = round(cols / aspect_ratio)
            
            # Ensure rows is within bounds
            rows = max(min_cells, min(ideal_rows, max_cells))
            
            # Calculate resulting dimensions
            square_size = min(width // cols, height // rows)
            result_width = cols * square_size
            result_height = rows * square_size
            result_aspect = result_width / result_height if result_height != 0 else float('inf')
            
            # Calculate error (difference from target aspect ratio)
            error = abs(aspect_ratio - result_aspect)
            
            if error < best_error:
                best_error = error
                best_rows = rows
                best_cols = cols
        
        return best_rows, best_cols
    
    def calculate_fixed_cell_size_dimensions(self, width, height, target_cell_size, min_cells, max_cells):
        """
        Calculate dimensions based on a target cell size.
        """
        # Calculate initial rows and cols based on target cell size
        cols = max(min_cells, min(round(width / target_cell_size), max_cells))
        rows = max(min_cells, min(round(height / target_cell_size), max_cells))
        
        return rows, cols
    
    def calculate_min_cells_dimensions(self, width, height, min_cells, max_cells):
        """
        Calculate dimensions using the minimum number of cells.
        This gives larger cells while ensuring at least min_cells in each dimension.
        """
        # Start with minimum cells
        rows = min_cells
        cols = min_cells
        
        return rows, cols
    
    def calculate_max_cells_dimensions(self, width, height, min_cells, max_cells):
        """
        Calculate dimensions using the maximum number of cells.
        This gives smaller cells, maximizing detail.
        """
        # Use maximum cells (but respect aspect ratio)
        aspect_ratio = width / height
        
        if aspect_ratio >= 1:  # Wider than tall
            cols = max_cells
            rows = max(min_cells, min(round(cols / aspect_ratio), max_cells))
        else:  # Taller than wide
            rows = max_cells
            cols = max(min_cells, min(round(rows * aspect_ratio), max_cells))
        
        return rows, cols
    
    def calculate_dimensions(self, width, height, min_cells, max_cells, method="aspect_ratio", target_cell_size=64):
        """
        Calculate optimal rows and columns for SVGArtGridV2 based on desired width and height.
        
        Args:
            width (int): Desired width in pixels
            height (int): Desired height in pixels
            min_cells (int): Minimum number of cells in either dimension
            max_cells (int): Maximum number of cells in either dimension
            method (str): Method to use for calculation
            target_cell_size (int): Target size of each cell in pixels (for fixed_cell_size method)
            
        Returns:
            tuple: (rows, cols) - Optimal dimensions for SVGArtGridV2
        """
        # Ensure valid inputs
        width = max(32, width)
        height = max(32, height)
        min_cells = max(1, min(min_cells, max_cells))
        max_cells = max(min_cells, max_cells)
        
        # Choose calculation method
        if method == "aspect_ratio":
            rows, cols = self.calculate_aspect_ratio_dimensions(width, height, min_cells, max_cells)
        elif method == "fixed_cell_size":
            rows, cols = self.calculate_fixed_cell_size_dimensions(width, height, target_cell_size, min_cells, max_cells)
        elif method == "min_cells":
            rows, cols = self.calculate_min_cells_dimensions(width, height, min_cells, max_cells)
        elif method == "max_cells":
            rows, cols = self.calculate_max_cells_dimensions(width, height, min_cells, max_cells)
        else:
            # Default to aspect ratio method
            rows, cols = self.calculate_aspect_ratio_dimensions(width, height, min_cells, max_cells)
        
        # Convert to int to ensure compatibility with SVGArtGridV2
        rows_int = int(rows)
        cols_int = int(cols)
        
        # Debug info
        print(f"SVGArtGridDimensionsCalculator: Width={width}, Height={height}")
        print(f"SVGArtGridDimensionsCalculator: Calculated rows={rows_int}, cols={cols_int}")
        
        # Output expected dimensions
        square_size = min(width // cols_int, height // rows_int)
        expected_width = cols_int * square_size
        expected_height = rows_int * square_size
        print(f"SVGArtGridDimensionsCalculator: Expected output dimensions: {expected_width}x{expected_height}")
        
        return (rows_int, cols_int)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SVGArtGridDimensionsCalculator": SVGArtGridDimensionsCalculator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVGArtGridDimensionsCalculator": "SVG Art Grid Dimensions Calculator"
}