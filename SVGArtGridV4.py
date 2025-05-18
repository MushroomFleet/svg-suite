#!/usr/bin/env python3
"""
SVGArtGridV4.py - An advanced ComfyUI node for generating artistic SVG grids
Building on SVGArtGridV3 with enhanced grid systems, advanced patterns, improved text features and gradient fills
"""

import random
import json
import requests
import math
import colorsys
import svgwrite
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import torch
import io
import re
import time
from typing import List, Tuple, Dict, Any, Set, Optional, Union
from enum import Enum

# Import scikit-learn for K-means clustering conditionally
SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

# Import cairosvg conditionally
CAIRO_AVAILABLE = False
try:
    import cairosvg
    CAIRO_AVAILABLE = True
except (ImportError, OSError):
    pass

# Import for edge detection
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    pass

class GridTemplate(Enum):
    """Enumeration of available grid templates."""
    REGULAR = "regular"
    GOLDEN_RATIO = "golden_ratio"
    RULE_OF_THIRDS = "rule_of_thirds"
    FIBONACCI = "fibonacci"
    FOCAL_POINT = "focal_point"
    ADAPTIVE = "adaptive"

class PatternType(Enum):
    """Enumeration of available pattern types."""
    CIRCLE = "circle"
    OPPOSITE_CIRCLES = "opposite_circles"
    CROSS = "cross"
    HALF_SQUARE = "half_square"
    DIAGONAL_SQUARE = "diagonal_square"
    QUARTER_CIRCLE = "quarter_circle"
    DOTS = "dots"
    LETTER_BLOCK = "letter_block"
    WAVE = "wave"
    SPIRAL = "spiral"
    CHEVRON = "chevron"
    CHECKERBOARD = "checkerboard"
    FRACTAL = "fractal"

class TextStyle(Enum):
    """Enumeration of available text styles."""
    NORMAL = "normal"
    OUTLINE = "outline"
    SHADOW = "shadow"
    EMBOSS = "emboss"
    GLOW = "glow"

class GradientType(Enum):
    """Enumeration of available gradient types."""
    NONE = "none"
    LINEAR = "linear"
    RADIAL = "radial"
    CONICAL = "conical"

class CellDefinition:
    """Class representing a grid cell with its position and dimensions."""
    def __init__(self, row: int, col: int, width: int = 1, height: int = 1):
        self.row = row
        self.col = col
        self.width = width
        self.height = height
        
    @property
    def end_row(self) -> int:
        return self.row + self.height - 1
        
    @property
    def end_col(self) -> int:
        return self.col + self.width - 1
        
    def covers_position(self, row: int, col: int) -> bool:
        """Check if this cell covers the given grid position."""
        return (self.row <= row <= self.end_row and
                self.col <= col <= self.end_col)
    
    def get_coords(self, square_size: int) -> Tuple[int, int, int, int]:
        """Get pixel coordinates for this cell."""
        x = self.col * square_size
        y = self.row * square_size
        width = self.width * square_size
        height = self.height * square_size
        return (x, y, width, height)

class TextBlock:
    """Class representing a text block in the grid."""
    def __init__(self, 
                 text: str,
                 position: Tuple[int, int],
                 orientation: str = "horizontal",
                 font_family: str = "monospace",
                 text_color: str = "",
                 bg_color: str = "",
                 size_multiplier: float = 1.0,
                 style: TextStyle = TextStyle.NORMAL,
                 style_params: Dict[str, Any] = None):
        self.text = text
        self.position = position  # (row, col)
        self.orientation = orientation  # "horizontal" or "vertical"
        self.font_family = font_family
        self.text_color = text_color  # Override color or empty for palette
        self.bg_color = bg_color  # Override color or empty for palette
        self.size_multiplier = size_multiplier
        self.style = style
        self.style_params = style_params or {}
        
    def get_positions(self) -> Set[Tuple[int, int]]:
        """Get grid positions covered by this text block."""
        positions = set()
        row, col = self.position
        
        for i in range(len(self.text)):
            if self.orientation == "horizontal":
                positions.add((row, col + i))
            else:  # vertical
                positions.add((row + i, col))
                
        return positions
    
    def render(self, dwg: svgwrite.Drawing, square_size: int, colors: Dict[str, str]):
        """Render the text block to the SVG."""
        foreground = self.text_color if self.text_color else colors.get("foreground", "#000000")
        background = self.bg_color if self.bg_color else colors.get("background", "#FFFFFF")
        
        for i, char in enumerate(self.text):
            if self.orientation == "horizontal":
                x = (self.position[1] + i) * square_size
                y = self.position[0] * square_size
            else:  # vertical
                x = self.position[1] * square_size
                y = (self.position[0] + i) * square_size
                
            self._render_character(dwg, x, y, square_size, char, foreground, background)
            
    def _render_character(self, dwg: svgwrite.Drawing, x: int, y: int, 
                          square_size: int, character: str, foreground: str, background: str):
        """Render a single character with the specified style."""
        group = dwg.g(class_=f"text-block-character style-{self.style.value}")
        
        # Add background first
        group.add(dwg.rect((x, y), (square_size, square_size), fill=background))
        
        # Calculate text size based on multiplier
        font_size = square_size * 0.8 * self.size_multiplier
        
        # Text position
        text_x = x + square_size/2
        text_y = y + square_size/2 + font_size*0.3
        
        # Apply different text styles
        if self.style == TextStyle.NORMAL:
            # Simple text
            group.add(dwg.text(character, 
                             insert=(text_x, text_y),
                             font_family=self.font_family, 
                             font_size=font_size,
                             font_weight="bold", 
                             fill=foreground, 
                             text_anchor="middle"))
            
        elif self.style == TextStyle.OUTLINE:
            # Create outline text
            outline_size = self.style_params.get('size', 2)
            group.add(dwg.text(character, 
                             insert=(text_x, text_y),
                             font_family=self.font_family, 
                             font_size=font_size,
                             font_weight="bold",
                             fill=foreground,
                             text_anchor="middle",
                             stroke=self.style_params.get('color', background),
                             stroke_width=outline_size))
            
        elif self.style == TextStyle.SHADOW:
            # Create shadow effect
            shadow_dx = self.style_params.get('dx', 2)
            shadow_dy = self.style_params.get('dy', 2)
            shadow_color = self.style_params.get('color', '#000000')
            
            # First render the shadow
            group.add(dwg.text(character, 
                             insert=(text_x + shadow_dx, text_y + shadow_dy),
                             font_family=self.font_family, 
                             font_size=font_size,
                             font_weight="bold", 
                             fill=shadow_color, 
                             text_anchor="middle"))
            
            # Then render the main text
            group.add(dwg.text(character, 
                             insert=(text_x, text_y),
                             font_family=self.font_family, 
                             font_size=font_size,
                             font_weight="bold", 
                             fill=foreground, 
                             text_anchor="middle"))
            
        elif self.style == TextStyle.EMBOSS:
            # Create emboss effect (highlight + shadow)
            highlight_dx = self.style_params.get('highlight_dx', -1)
            highlight_dy = self.style_params.get('highlight_dy', -1)
            highlight_color = self.style_params.get('highlight_color', '#FFFFFF')
            
            shadow_dx = self.style_params.get('shadow_dx', 1)
            shadow_dy = self.style_params.get('shadow_dy', 1)
            shadow_color = self.style_params.get('shadow_color', '#000000')
            
            # First render the shadow
            group.add(dwg.text(character, 
                             insert=(text_x + shadow_dx, text_y + shadow_dy),
                             font_family=self.font_family, 
                             font_size=font_size,
                             font_weight="bold", 
                             fill=shadow_color, 
                             text_anchor="middle"))
            
            # Then render the highlight
            group.add(dwg.text(character, 
                             insert=(text_x + highlight_dx, text_y + highlight_dy),
                             font_family=self.font_family, 
                             font_size=font_size,
                             font_weight="bold", 
                             fill=highlight_color, 
                             text_anchor="middle"))
            
            # Then render the main text
            group.add(dwg.text(character, 
                             insert=(text_x, text_y),
                             font_family=self.font_family, 
                             font_size=font_size,
                             font_weight="bold", 
                             fill=foreground, 
                             text_anchor="middle"))
            
        elif self.style == TextStyle.GLOW:
            # Create glow effect
            glow_color = self.style_params.get('color', '#FFFF00')
            glow_size = self.style_params.get('size', 3)
            
            # Create a filter for the glow
            filter_id = f"glow-filter-{x}-{y}"
            glow_filter = dwg.defs.add(dwg.filter(id=filter_id))
            
            # Add a blur effect to the filter
            glow_filter.feGaussianBlur(in_="SourceGraphic", stdDeviation=glow_size)
            
            # First render the glow
            glow_text = dwg.text(character, 
                               insert=(text_x, text_y),
                               font_family=self.font_family, 
                               font_size=font_size,
                               font_weight="bold", 
                               fill=glow_color, 
                               text_anchor="middle")
            glow_text['filter'] = f"url(#{filter_id})"
            group.add(glow_text)
            
            # Then render the main text
            group.add(dwg.text(character, 
                             insert=(text_x, text_y),
                             font_family=self.font_family, 
                             font_size=font_size,
                             font_weight="bold", 
                             fill=foreground, 
                             text_anchor="middle"))
        
        dwg.add(group)

class GradientDefinition:
    """Class representing a gradient definition."""
    def __init__(self, 
                 gradient_type: GradientType,
                 colors: List[str],
                 start_position: Tuple[float, float] = (0, 0),
                 end_position: Tuple[float, float] = (1, 1),
                 color_stops: List[float] = None,
                 spread_method: str = "pad",
                 rotation_angle: float = 0):
        self.gradient_type = gradient_type
        self.colors = colors
        self.start_position = start_position  # x1, y1 as fractions of bounding box
        self.end_position = end_position  # x2, y2 as fractions of bounding box
        self.color_stops = color_stops or [i/(len(colors)-1) for i in range(len(colors))]
        self.spread_method = spread_method  # "pad", "reflect", or "repeat"
        self.rotation_angle = rotation_angle  # For linear gradients
        
    def add_to_defs(self, dwg: svgwrite.Drawing, id_prefix: str = "") -> str:
        """Add this gradient to the SVG defs and return its ID."""
        unique_id = f"{id_prefix}gradient-{int(time.time() * 1000)}-{random.randint(0, 10000)}"
        
        if self.gradient_type == GradientType.LINEAR:
            # Apply rotation to positions
            angle_rad = math.radians(self.rotation_angle)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)
            
            x1, y1 = self.start_position
            x2, y2 = self.end_position
            
            # Rotate around center (0.5, 0.5)
            cx, cy = 0.5, 0.5
            x1r = cx + (x1 - cx) * cos_angle - (y1 - cy) * sin_angle
            y1r = cy + (x1 - cx) * sin_angle + (y1 - cy) * cos_angle
            x2r = cx + (x2 - cx) * cos_angle - (y2 - cy) * sin_angle
            y2r = cy + (x2 - cx) * sin_angle + (y2 - cy) * cos_angle
            
            # Create linear gradient
            gradient = dwg.defs.add(dwg.linearGradient(
                id=unique_id,
                x1=f"{x1r:.6f}",
                y1=f"{y1r:.6f}",
                x2=f"{x2r:.6f}",
                y2=f"{y2r:.6f}",
                gradientUnits="objectBoundingBox",
                spreadMethod=self.spread_method
            ))
            
        elif self.gradient_type == GradientType.RADIAL:
            # Create radial gradient
            cx, cy = self.end_position  # center position
            fx, fy = self.start_position  # focal position
            
            gradient = dwg.defs.add(dwg.radialGradient(
                id=unique_id,
                cx=f"{cx:.6f}",
                cy=f"{cy:.6f}",
                r="0.5",  # Half of the bounding box
                fx=f"{fx:.6f}",
                fy=f"{fy:.6f}",
                gradientUnits="objectBoundingBox",
                spreadMethod=self.spread_method
            ))
            
        elif self.gradient_type == GradientType.CONICAL:
            # Conical gradients aren't directly supported in SVG 1.1
            # Use a radial gradient as base
            gradient = dwg.defs.add(dwg.radialGradient(
                id=unique_id,
                cx="0.5",
                cy="0.5",
                r="0.5",
                fx="0.5",
                fy="0.5",
                gradientUnits="objectBoundingBox"
            ))
            
            # Approximate with many color stops
            num_stops = 16  # Use more stops for smoother transition
            for i in range(num_stops):
                angle = 2 * math.pi * i / num_stops
                # Map angle to color index
                color_idx = int((i / num_stops) * len(self.colors))
                color_idx = min(color_idx, len(self.colors) - 1)
                
                r = 0.5 + 0.5 * math.cos(angle + math.radians(self.rotation_angle))
                gradient.add_stop_color(r, self.colors[color_idx])
                
            return f"url(#{unique_id})"
            
        else:
            # For GradientType.NONE or unexpected values
            return ""
        
        # Add color stops for linear and radial gradients
        for i, color in enumerate(self.colors):
            offset = self.color_stops[i] if i < len(self.color_stops) else i/(len(self.colors)-1)
            gradient.add_stop_color(offset, color)
        
        return f"url(#{unique_id})"

class SVGArtGridV4:
    """A ComfyUI node that generates sophisticated artistic SVG grids with enhanced features"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
                
                # Grid System Settings
                "grid_template": (
                    [t.value for t in GridTemplate],
                    {"default": GridTemplate.REGULAR.value}
                ),
                "rows": ("INT", {"default": 6, "min": 1, "max": 30, "step": 1}),
                "cols": ("INT", {"default": 6, "min": 1, "max": 30, "step": 1}),
                "adaptive_grid": ("BOOLEAN", {"default": False, 
                                            "tooltip": "Analyze image to create responsive grid"}),
                "variable_cells": ("BOOLEAN", {"default": False, 
                                            "tooltip": "Enable non-uniform cell sizes"}),
                "focal_point_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                                          "tooltip": "X position of focal point (0-1)"}),
                "focal_point_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                                          "tooltip": "Y position of focal point (0-1)"}),
                
                # Color Settings
                "mode": (["palette", "composition"], {"default": "palette"}),
                "palette_index": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1,
                                        "tooltip": "-1 for image-based palette"}),
                "color_count": ("INT", {"default": 5, "min": 3, "max": 10, "step": 1}),
                "blend_factor": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                
                # Pattern Settings
                "enable_big_block": ("BOOLEAN", {"default": True}),
                "big_block_size": ("INT", {"default": 2, "min": 2, "max": 4, "step": 1}),
                
                # Basic Pattern Toggles
                "enable_circle": ("BOOLEAN", {"default": True}),
                "enable_opposite_circles": ("BOOLEAN", {"default": True}),
                "enable_cross": ("BOOLEAN", {"default": True}),
                "enable_half_square": ("BOOLEAN", {"default": True}),
                "enable_diagonal_square": ("BOOLEAN", {"default": True}),
                "enable_quarter_circle": ("BOOLEAN", {"default": True}),
                "enable_dots": ("BOOLEAN", {"default": True}),
                "enable_letter_block": ("BOOLEAN", {"default": True}),
                
                # Advanced Pattern Toggles
                "enable_wave": ("BOOLEAN", {"default": False}),
                "enable_spiral": ("BOOLEAN", {"default": False}),
                "enable_chevron": ("BOOLEAN", {"default": False}),
                "enable_checkerboard": ("BOOLEAN", {"default": False}),
                "enable_fractal": ("BOOLEAN", {"default": False}),
                "pattern_complexity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                                               "tooltip": "Controls detail level in advanced patterns"}),
                
                # Gradient Settings
                "enable_gradients": ("BOOLEAN", {"default": False}),
                "gradient_type": (
                    [g.value for g in GradientType if g != GradientType.NONE],
                    {"default": GradientType.LINEAR.value}
                ),
                "gradient_angle": ("FLOAT", {"default": 45, "min": 0, "max": 360, "step": 15}),
                
                # Output Settings
                "output_svg": ("BOOLEAN", {"default": False, "tooltip": "Also output SVG string"}),
            },
            "optional": {
                # Text Block Settings
                "text_blocks": ("STRING", {"default": "[]", "multiline": True,
                                          "tooltip": "JSON array of text block definitions"}),
                
                # Legacy/Advanced Options
                "palette_url": ("STRING", {"default": "https://unpkg.com/nice-color-palettes@3.0.0/100.json",
                                         "multiline": False}),
                "block_styles": ("STRING", {"default": "all", "multiline": False,
                                          "tooltip": "Legacy: comma-separated list or 'all'"}),
                
                # Variable Cell Definitions (JSON format)
                "cell_definitions": ("STRING", {"default": "[]", "multiline": True,
                                              "tooltip": "JSON array of custom cell definitions"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "svg_string")
    FUNCTION = "generate_art_grid"
    CATEGORY = "SVG/ArtGrid"

    def _load_color_palettes(self, palette_url):
        """Load color palettes from URL."""
        try:
            response = requests.get(palette_url)
            return response.json()
        except:
            # Fallback color palettes if URL fails
            return [
                ['#69D2E7', '#A7DBD8', '#E0E4CC', '#F38630', '#FA6900'],
                ['#FF4E50', '#FC913A', '#F9D423', '#EDE574', '#E1F5C4'],
                ['#69D2E7', '#A7DBD8', '#E0E4CC', '#B2C2C1', '#8AB8B2'],
                ['#FFFFFF', '#D9D9D9', '#BFBFBF', '#8C8C8C', '#404040']
            ]

    def _extract_palette_from_image(self, pil_image, color_count=5):
        """
        Extract a color palette from an image using k-means clustering.
        
        Args:
            pil_image: PIL Image object
            color_count (int): Number of colors to extract
            
        Returns:
            list: List of hex color codes
        """
        # Resize image (for faster processing)
        img = pil_image.copy()
        img = img.resize((150, 150))  # Smaller size for faster processing
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # If sklearn is not available, return a default palette
        if not SKLEARN_AVAILABLE:
            print("SVGArtGridV4: sklearn not available, using default palette")
            return [
                '#69D2E7', '#A7DBD8', '#E0E4CC', '#F38630', '#FA6900'
            ][:color_count]
        
        # Get pixel data as numpy array and reshape for k-means
        pixels = np.array(img)
        pixels = pixels.reshape(-1, 3)
        
        # Apply k-means clustering to find dominant colors
        kmeans = KMeans(n_clusters=color_count, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get the colors
        colors = kmeans.cluster_centers_
        
        # Convert to hex format
        hex_colors = []
        for color in colors:
            # Convert to integers
            r, g, b = [int(c) for c in color]
            # Convert to hex
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            hex_colors.append(hex_color)
        
        return hex_colors

    def _sample_image_region(self, image, x, y, width, height, sample_count=2, blend_factor=0.7):
        """
        Sample predominant colors from a region of an image.
        
        Args:
            image: PIL Image object
            x, y: Top-left coordinates of the region
            width, height: Dimensions of the region
            sample_count: Number of colors to extract
            blend_factor: How closely to follow image colors (0-1)
            
        Returns:
            dict: Dict with "foreground" and "background" color values
        """
        # If sklearn is not available, return random colors
        if not SKLEARN_AVAILABLE:
            colors = ['#69D2E7', '#F38630']
            return {
                "foreground": colors[0],
                "background": colors[1]
            }
            
        # Crop the region from the image
        region = image.crop((x, y, x + width, y + height))
        
        # Resize for faster processing
        region = region.resize((50, 50))
        
        # Convert to RGB if needed
        if region.mode != 'RGB':
            region = region.convert('RGB')
        
        # Get pixel data as numpy array
        pixels = np.array(region)
        pixels = pixels.reshape(-1, 3)
        
        # Use k-means to find dominant colors
        kmeans = KMeans(n_clusters=sample_count, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get the colors
        colors = kmeans.cluster_centers_
        
        # Apply blend factor (0 = random colors, 1 = exact image colors)
        if blend_factor < 1.0:
            for i in range(len(colors)):
                # Generate a random color
                random_color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
                # Blend with the original color
                colors[i] = colors[i] * blend_factor + random_color * (1 - blend_factor)
        
        # Convert to hex format
        hex_colors = []
        for color in colors:
            r, g, b = [int(c) for c in color]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            hex_colors.append(hex_color)
        
        # Calculate brightness for each color to determine foreground/background
        brightness = []
        for color in colors:
            # Simple brightness formula: 0.299*R + 0.587*G + 0.114*B
            bright = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            brightness.append(bright)
        
        # Sort colors by brightness
        sorted_colors = [x for _, x in sorted(zip(brightness, hex_colors))]
        
        # Return darkest as foreground, lightest as background
        return {
            "foreground": sorted_colors[0],  # Darkest color
            "background": sorted_colors[-1]  # Lightest color
        }

    def _create_background_colors(self, color_palette):
        """Create background colors by mixing colors from the palette."""
        color1 = color_palette[0].lstrip('#')
        color2 = color_palette[1].lstrip('#')
        
        # Convert hex to RGB
        r1, g1, b1 = tuple(int(color1[i:i+2], 16) for i in (0, 2, 4))
        r2, g2, b2 = tuple(int(color2[i:i+2], 16) for i in (0, 2, 4))
        
        # Mix colors (50% blend)
        r = (r1 + r2) // 2
        g = (g1 + g2) // 2
        b = (b1 + b2) // 2
        
        # Desaturate
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        s = max(0, s - 0.1)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        r, g, b = int(r*255), int(g*255), int(b*255)
        
        bg = f"#{r:02x}{g:02x}{b:02x}"
        
        # Create lighter and darker versions
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        
        l_light = min(1, l + 0.1)
        r_light, g_light, b_light = colorsys.hls_to_rgb(h, l_light, s)
        bg_inner = f"#{int(r_light*255):02x}{int(g_light*255):02x}{int(b_light*255):02x}"
        
        l_dark = max(0, l - 0.1)
        r_dark, g_dark, b_dark = colorsys.hls_to_rgb(h, l_dark, s)
        bg_outer = f"#{int(r_dark*255):02x}{int(g_dark*255):02x}{int(b_dark*255):02x}"
        
        return {"bg_inner": bg_inner, "bg_outer": bg_outer}

    def _get_two_colors(self, color_palette):
        """Get two different colors from the palette."""
        color_list = color_palette.copy()
        color_index = random.randint(0, len(color_list) - 1)
        background = color_list[color_index]
        color_list.pop(color_index)
        foreground = random.choice(color_list)
        return {"foreground": foreground, "background": background}
        
    def _create_gradient_fill(self, dwg, color_palette, gradient_type, gradient_angle, cell_id=""):
        """Create a gradient fill from the color palette."""
        # Select colors for the gradient
        colors = random.sample(color_palette, min(3, len(color_palette)))
        
        # Create gradient definition
        gradient_def = GradientDefinition(
            gradient_type=GradientType(gradient_type),
            colors=colors,
            rotation_angle=gradient_angle
        )
        
        # Add to defs and return the reference
        return gradient_def.add_to_defs(dwg, cell_id)
        
    def _generate_template_grid(self, template, rows, cols, focal_x=0.5, focal_y=0.5):
        """Generate a grid based on the selected template."""
        if template == GridTemplate.REGULAR.value:
            # Regular grid - all cells are 1x1
            return [CellDefinition(row, col) for row in range(rows) for col in range(cols)]
            
        elif template == GridTemplate.GOLDEN_RATIO.value:
            # Golden ratio grid with larger cells following the golden ratio pattern
            cells = []
            golden_ratio = 1.618
            
            # Start with a large cell in the corner
            large_width = min(3, cols // 2)
            large_height = min(int(large_width * golden_ratio), rows // 2)
            
            # Add the large cell
            cells.append(CellDefinition(0, 0, large_width, large_height))
            
            # Add medium cells
            med_width = min(2, max(1, cols - large_width))
            med_height = min(2, large_height)
            if large_width < cols:
                cells.append(CellDefinition(0, large_width, med_width, med_height))
            
            # Add remaining 1x1 cells
            for row in range(rows):
                for col in range(cols):
                    # Skip positions covered by large or medium cells
                    if any(cell.covers_position(row, col) for cell in cells):
                        continue
                    cells.append(CellDefinition(row, col))
            
            return cells
            
        elif template == GridTemplate.RULE_OF_THIRDS.value:
            # Rule of thirds grid
            cells = []
            
            # Calculate third positions
            third_row = rows // 3
            third_col = cols // 3
            
            # Create regions
            # Top-left
            cells.append(CellDefinition(0, 0, third_col, third_row))
            # Top-middle
            cells.append(CellDefinition(0, third_col, third_col, third_row))
            # Top-right
            cells.append(CellDefinition(0, 2*third_col, cols - 2*third_col, third_row))
            
            # Middle-left
            cells.append(CellDefinition(third_row, 0, third_col, third_row))
            # Middle-middle
            cells.append(CellDefinition(third_row, third_col, third_col, third_row))
            # Middle-right
            cells.append(CellDefinition(third_row, 2*third_col, cols - 2*third_col, third_row))
            
            # Bottom-left
            cells.append(CellDefinition(2*third_row, 0, third_col, rows - 2*third_row))
            # Bottom-middle
            cells.append(CellDefinition(2*third_row, third_col, third_col, rows - 2*third_row))
            # Bottom-right
            cells.append(CellDefinition(2*third_row, 2*third_col, cols - 2*third_col, rows - 2*third_row))
            
            # Ensure all positions are covered
            for row in range(rows):
                for col in range(cols):
                    if not any(cell.covers_position(row, col) for cell in cells):
                        cells.append(CellDefinition(row, col))
            
            return cells
            
        elif template == GridTemplate.FIBONACCI.value:
            # Fibonacci spiral-inspired grid
            cells = []
            
            # Fibonacci numbers for sizing
            fib = [1, 1, 2, 3, 5, 8, 13]
            
            # Scale to fit grid size
            max_dim = max(rows, cols)
            scale_factor = max_dim / fib[-1]
            scaled_fib = [max(1, min(int(f * scale_factor), max_dim)) for f in fib]
            
            # Create cells in spiral pattern
            if max_dim >= 5:
                # Add larger cells for the spiral
                size = min(scaled_fib[-2], min(rows, cols) // 2)
                cells.append(CellDefinition(0, 0, size, size))
                
                if cols > size:
                    size2 = min(scaled_fib[-3], min(rows, cols - size) // 2)
                    cells.append(CellDefinition(0, size, size2, size2))
                    
                    if rows > size2:
                        size3 = min(scaled_fib[-4], min(rows - size2, cols - size) // 2)
                        cells.append(CellDefinition(size2, size, size3, size3))
            
            # Fill remaining with 1x1 cells
            for row in range(rows):
                for col in range(cols):
                    if not any(cell.covers_position(row, col) for cell in cells):
                        cells.append(CellDefinition(row, col))
            
            return cells
            
        elif template == GridTemplate.FOCAL_POINT.value:
            # Focal point grid with larger cells near the focal point
            cells = []
            
            # Convert focal point to grid coordinates
            focal_row = int(focal_y * rows)
            focal_col = int(focal_x * cols)
            
            # Ensure valid coordinates
            focal_row = max(0, min(focal_row, rows - 1))
            focal_col = max(0, min(focal_col, cols - 1))
            
            # Create larger cell at focal point
            focal_size = min(3, min(rows - focal_row, cols - focal_col, focal_row + 1, focal_col + 1))
            
            # Adjust focal point to allow for the cell size
            focal_row = min(focal_row, rows - focal_size)
            focal_col = min(focal_col, cols - focal_size)
            
            # Add focal cell
            cells.append(CellDefinition(focal_row, focal_col, focal_size, focal_size))
            
            # Fill remaining with 1x1 cells
            for row in range(rows):
                for col in range(cols):
                    if not any(cell.covers_position(row, col) for cell in cells):
                        cells.append(CellDefinition(row, col))
            
            return cells
            
        elif template == GridTemplate.ADAPTIVE.value:
            # Placeholder for adaptive grid (without image analysis)
            # Default to a regular grid
            return [CellDefinition(row, col) for row in range(rows) for col in range(cols)]
            
        else:
            # Fallback to regular grid
            return [CellDefinition(row, col) for row in range(rows) for col in range(cols)]
    
    def _detect_edges(self, pil_image):
        """
        Detect edges in the image to inform adaptive grid generation.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            numpy.ndarray: Edge image
        """
        # If OpenCV is not available, return a simplistic edge detection
        if not CV2_AVAILABLE:
            # Convert to grayscale
            if pil_image.mode != 'L':
                gray_image = pil_image.convert('L')
            else:
                gray_image = pil_image
                
            # Use PIL's ImageFilter for simple edge detection
            edge_image = gray_image.filter(ImageFilter.FIND_EDGES)
            return np.array(edge_image)
        
        # Use OpenCV for better edge detection
        # Convert PIL image to OpenCV format
        img = np.array(pil_image)
        
        # Convert to RGB if needed (OpenCV uses BGR)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges

    def _generate_adaptive_grid(self, pil_image, rows, cols):
        """
        Generate an adaptive grid based on image content analysis.
        
        Args:
            pil_image: PIL Image object
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            list: List of CellDefinition objects
        """
        # Basic implementation - can be expanded with more sophisticated analysis
        # Detect edges in the image
        edges = self._detect_edges(pil_image)
        
        # Resize edges to match the grid dimensions
        edges_resized = cv2.resize(edges, (cols, rows)) if CV2_AVAILABLE else \
                      np.array(Image.fromarray(edges).resize((cols, rows)))
        
        # Create a grid based on edge density
        cells = []
        visited = set()
        
        # Threshold for edge density
        threshold = np.mean(edges_resized) * 0.5
        
        # Function to check if a region has low edge density
        def can_merge(start_row, start_col, width, height):
            for r in range(start_row, start_row + height):
                for c in range(start_col, start_col + width):
                    if (r, c) in visited or r >= rows or c >= cols:
                        return False
                    if edges_resized[r, c] > threshold:
                        return False
            return True
        
        # Try to create larger cells in areas with low edge density
        for row in range(rows):
            for col in range(cols):
                if (row, col) in visited:
                    continue
                
                # Try to create a 2x2 cell
                if row + 1 < rows and col + 1 < cols and can_merge(row, col, 2, 2):
                    cells.append(CellDefinition(row, col, 2, 2))
                    for r in range(row, row + 2):
                        for c in range(col, col + 2):
                            visited.add((r, c))
                # Try to create a 2x1 cell (horizontal)
                elif col + 1 < cols and can_merge(row, col, 2, 1):
                    cells.append(CellDefinition(row, col, 2, 1))
                    visited.add((row, col))
                    visited.add((row, col + 1))
                # Try to create a 1x2 cell (vertical)
                elif row + 1 < rows and can_merge(row, col, 1, 2):
                    cells.append(CellDefinition(row, col, 1, 2))
                    visited.add((row, col))
                    visited.add((row + 1, col))
                # Create a 1x1 cell
                else:
                    cells.append(CellDefinition(row, col))
                    visited.add((row, col))
        
        return cells

    def _parse_cell_definitions(self, cell_defs_json, rows, cols):
        """
        Parse user-provided custom cell definitions from JSON.
        
        Args:
            cell_defs_json: JSON string with cell definitions
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            list: List of CellDefinition objects
        """
        cells = []
        try:
            # Parse JSON
            cell_defs = json.loads(cell_defs_json)
            
            # Validate and convert to CellDefinition objects
            for cell_def in cell_defs:
                row = cell_def.get('row', 0)
                col = cell_def.get('col', 0)
                width = cell_def.get('width', 1)
                height = cell_def.get('height', 1)
                
                # Validate values
                row = max(0, min(row, rows - 1))
                col = max(0, min(col, cols - 1))
                width = max(1, min(width, cols - col))
                height = max(1, min(height, rows - row))
                
                cells.append(CellDefinition(row, col, width, height))
                
            # Check for any missing positions and fill with 1x1 cells
            covered_positions = set()
            for cell in cells:
                for r in range(cell.row, cell.row + cell.height):
                    for c in range(cell.col, cell.col + cell.width):
                        covered_positions.add((r, c))
            
            for row in range(rows):
                for col in range(cols):
                    if (row, col) not in covered_positions:
                        cells.append(CellDefinition(row, col))
                        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"SVGArtGridV4: Error parsing cell definitions: {e}")
            # Fallback to regular grid
            cells = [CellDefinition(row, col) for row in range(rows) for col in range(cols)]
            
        return cells

    def _parse_text_blocks(self, text_blocks_json):
        """
        Parse user-provided text block definitions from JSON.
        
        Args:
            text_blocks_json: JSON string with text block definitions
            
        Returns:
            list: List of TextBlock objects
        """
        blocks = []
        try:
            # Parse JSON
            text_defs = json.loads(text_blocks_json)
            
            # Validate and convert to TextBlock objects
            for text_def in text_defs:
                text = text_def.get('text', '')
                position = text_def.get('position', [1, 1])
                orientation = text_def.get('orientation', 'horizontal')
                font_family = text_def.get('font_family', 'monospace')
                text_color = text_def.get('text_color', '')
                bg_color = text_def.get('bg_color', '')
                size_multiplier = float(text_def.get('size_multiplier', 1.0))
                style_name = text_def.get('style', 'normal')
                style_params = text_def.get('style_params', {})
                
                # Validate values
                try:
                    style = TextStyle(style_name)
                except ValueError:
                    style = TextStyle.NORMAL
                
                blocks.append(TextBlock(
                    text=text,
                    position=tuple(position),
                    orientation=orientation,
                    font_family=font_family,
                    text_color=text_color,
                    bg_color=bg_color,
                    size_multiplier=size_multiplier,
                    style=style,
                    style_params=style_params
                ))
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"SVGArtGridV4: Error parsing text blocks: {e}")
            # Leave empty if there are errors
            blocks = []
            
        return blocks
        
    def _draw_circle(self, dwg, x, y, width, height, foreground, background, 
                   pattern_complexity=0.5, enable_gradients=False, gradient_fill=""):
        """Draw a circle block."""
        group = dwg.g(class_="draw-circle")
        
        # Use gradient fill if enabled and provided
        bg_fill = gradient_fill if enable_gradients and gradient_fill else background
        
        group.add(dwg.rect((x, y), (width, height), fill=bg_fill))
        group.add(dwg.circle(center=(x + width/2, y + height/2), 
                             r=min(width, height)/2, fill=foreground))
        
        # Add inner circle based on complexity
        if random.random() < pattern_complexity:
            inner_size = min(width, height) / 4
            group.add(dwg.circle(center=(x + width/2, y + height/2),
                                 r=inner_size, fill=bg_fill))
        dwg.add(group)

    def _draw_opposite_circles(self, dwg, x, y, width, height, foreground, background, 
                             pattern_complexity=0.5, enable_gradients=False, gradient_fill=""):
        """Draw opposite circles block."""
        group = dwg.g(class_="opposite-circles")
        
        # Use gradient fill if enabled and provided
        bg_fill = gradient_fill if enable_gradients and gradient_fill else background
        
        group.add(dwg.rect((x, y), (width, height), fill=bg_fill))
        
        mask_id = f"mask-{x}-{y}"
        mask = dwg.mask(id=mask_id)
        mask.add(dwg.rect((x, y), (width, height), fill="white"))
        dwg.defs.add(mask)
        
        # Adjust options based on cell shape
        if width > height:
            options = [[0, height/2, width, height/2]]  # Horizontal
        elif height > width:
            options = [[width/2, 0, width/2, height]]  # Vertical
        else:
            options = [[0, 0, width, height], [width, 0, 0, height]]  # Square - diagonal
            
        offset = random.choice(options)
        
        circle_group = dwg.g()
        
        # Adjust circle radius based on cell dimensions
        radius = min(width, height) / 2
        
        # Create more complex patterns based on complexity parameter
        if pattern_complexity > 0.7 and random.random() < 0.5:
            # Create a pattern with multiple circles
            num_circles = random.randint(3, 5)
            for i in range(num_circles):
                cx = x + (width * i / (num_circles - 1))
                cy = y + (height * i / (num_circles - 1))
                circle_group.add(dwg.circle(center=(cx, cy), r=radius/2, fill=foreground))
        else:
            # Standard opposite circles
            circle_group.add(dwg.circle(center=(x + offset[0], y + offset[1]), 
                                       r=radius, fill=foreground))
            circle_group.add(dwg.circle(center=(x + offset[2], y + offset[3]), 
                                       r=radius, fill=foreground))
            
        circle_group['mask'] = f"url(#{mask_id})"
        
        group.add(circle_group)
        dwg.add(group)

    def _draw_cross(self, dwg, x, y, width, height, foreground, background, 
                  pattern_complexity=0.5, enable_gradients=False, gradient_fill=""):
        """Draw a cross or X block."""
        group = dwg.g(class_="draw-cross")
        
        # Use gradient fill if enabled and provided
        bg_fill = gradient_fill if enable_gradients and gradient_fill else background
        
        group.add(dwg.rect((x, y), (width, height), fill=bg_fill))
        
        is_plus = random.random() < 0.5
        
        if is_plus:
            # Adjust cross thickness based on complexity
            thickness = min(width, height) / (4 - pattern_complexity * 2)
            thickness = max(thickness, min(width, height) / 6)  # Ensure minimum thickness
            
            # Horizontal bar
            group.add(dwg.rect((x, y + (height - thickness)/2), 
                              (width, thickness), fill=foreground))
            # Vertical bar
            group.add(dwg.rect((x + (width - thickness)/2, y), 
                              (thickness, height), fill=foreground))
                              
            # Add decorative elements based on complexity
            if pattern_complexity > 0.7 and random.random() < 0.5:
                # Add circle in the center
                center_size = min(width, height) / 6
                group.add(dwg.circle(center=(x + width/2, y + height/2),
                                    r=center_size, fill=bg_fill))
        else:
            # X shape with polygons for better diagonal lines
            line_width = min(width, height) / (8 - pattern_complexity * 5)
            line_width = max(line_width, min(width, height) / 12)  # Ensure minimum thickness
            
            # First diagonal
            x1, y1 = x, y
            x2, y2 = x + width, y + height
            dx, dy = x2 - x1, y2 - y1
            length = (dx**2 + dy**2)**0.5
            nx, ny = -dy/length, dx/length
            
            p1 = (x1 + nx*line_width/2, y1 + ny*line_width/2)
            p2 = (x2 + nx*line_width/2, y2 + ny*line_width/2)
            p3 = (x2 - nx*line_width/2, y2 - ny*line_width/2)
            p4 = (x1 - nx*line_width/2, y1 - ny*line_width/2)
            group.add(dwg.polygon([p1, p2, p3, p4], fill=foreground))
            
            # Second diagonal
            x1, y1 = x + width, y
            x2, y2 = x, y + height
            dx, dy = x2 - x1, y2 - y1
            length = (dx**2 + dy**2)**0.5
            nx, ny = -dy/length, dx/length
            
            p1 = (x1 + nx*line_width/2, y1 + ny*line_width/2)
            p2 = (x2 + nx*line_width/2, y2 + ny*line_width/2)
            p3 = (x2 - nx*line_width/2, y2 - ny*line_width/2)
            p4 = (x1 - nx*line_width/2, y1 - ny*line_width/2)
            group.add(dwg.polygon([p1, p2, p3, p4], fill=foreground))
        
        dwg.add(group)

    def _draw_half_square(self, dwg, x, y, width, height, foreground, background, 
                         pattern_complexity=0.5, enable_gradients=False, gradient_fill=""):
        """Draw a half square block."""
        group = dwg.g(class_="draw-half-square")
        
        # Use gradient fill if enabled and provided
        bg_fill = gradient_fill if enable_gradients and gradient_fill else background
        
        group.add(dwg.rect((x, y), (width, height), fill=bg_fill))
        
        # Choose direction based on cell shape
        if width > height * 1.5:
            # Wide cell - prefer left/right division
            direction = random.choice(['left', 'right'])
        elif height > width * 1.5:
            # Tall cell - prefer top/bottom division
            direction = random.choice(['top', 'bottom'])
        else:
            direction = random.choice(['top', 'right', 'bottom', 'left'])
        
        # Define points based on direction
        if direction == 'top':
            points = [(x, y), (x + width, y), (x + width, y + height/2), (x, y + height/2)]
        elif direction == 'right':
            points = [(x + width/2, y), (x + width, y), (x + width, y + height), (x + width/2, y + height)]
        elif direction == 'bottom':
            points = [(x, y + height/2), (x + width, y + height/2), (x + width, y + height), (x, y + height)]
        else:  # left
            points = [(x, y), (x + width/2, y), (x + width/2, y + height), (x, y + height)]
        
        # Add decorative elements based on complexity
        if pattern_complexity > 0.7 and random.random() < 0.5:
            # Create polygon for main half
            group.add(dwg.polygon(points, fill=foreground))
            
            # Add a decorative element in the other half
            if direction == 'top':
                center_x = x + width/2
                center_y = y + 3*height/4
            elif direction == 'right':
                center_x = x + width/4
                center_y = y + height/2
            elif direction == 'bottom':
                center_x = x + width/2
                center_y = y + height/4
            else:  # left
                center_x = x + 3*width/4
                center_y = y + height/2
                
            radius = min(width, height) / 6
            group.add(dwg.circle(center=(center_x, center_y), r=radius, fill=foreground))
        else:
            # Simple half square
            group.add(dwg.polygon(points, fill=foreground))
        
        dwg.add(group)

    def _draw_diagonal_square(self, dwg, x, y, width, height, foreground, background, 
                             pattern_complexity=0.5, enable_gradients=False, gradient_fill=""):
        """Draw a diagonal square block."""
        group = dwg.g(class_="draw-diagonal-square")
        
        # Use gradient fill if enabled and provided
        bg_fill = gradient_fill if enable_gradients and gradient_fill else background
        
        group.add(dwg.rect((x, y), (width, height), fill=bg_fill))
        
        # Choose direction based on cell shape
        if width != height:
            # For non-square cells, choose a direction that looks better
            is_top_left_to_bottom_right = width >= height
        else:
            is_top_left_to_bottom_right = random.random() < 0.5
        
        if is_top_left_to_bottom_right:
            points = [(x, y), (x + width, y + height), (x, y + height)]
        else:
            points = [(x + width, y), (x + width, y + height), (x, y)]
        
        # Add decorative elements based on complexity
        if pattern_complexity > 0.8 and random.random() < 0.5:
            # Create polygon for main triangle
            group.add(dwg.polygon(points, fill=foreground))
            
            # Add a smaller triangle in the opposite corner
            if is_top_left_to_bottom_right:
                small_points = [(x + width, y), (x + width, y + height/3), (x + width*2/3, y)]
            else:
                small_points = [(x, y), (x + width/3, y), (x, y + height/3)]
                
            group.add(dwg.polygon(small_points, fill=foreground))
        else:
            # Simple diagonal triangle
            group.add(dwg.polygon(points, fill=foreground))
        
        dwg.add(group)

    def _draw_quarter_circle(self, dwg, x, y, width, height, foreground, background, 
                            pattern_complexity=0.5, enable_gradients=False, gradient_fill=""):
        """Draw a quarter circle block."""
        group = dwg.g(class_="draw-quarter-circle")
        
        # Use gradient fill if enabled and provided
        bg_fill = gradient_fill if enable_gradients and gradient_fill else background
        
        group.add(dwg.rect((x, y), (width, height), fill=bg_fill))
        
        # Choose corner based on cell shape
        if width > height:
            # Wide cell - prefer left/right corners
            if x == 0:
                corner = random.choice(['top-left', 'bottom-left'])
            else:
                corner = random.choice(['top-right', 'bottom-right'])
        elif height > width:
            # Tall cell - prefer top/bottom corners
            if y == 0:
                corner = random.choice(['top-left', 'top-right'])
            else:
                corner = random.choice(['bottom-left', 'bottom-right'])
        else:
            corner = random.choice(['top-left', 'top-right', 'bottom-right', 'bottom-left'])
        
        path = dwg.path(fill=foreground)
        
        # Radius based on cell dimensions
        rx = width
        ry = height
        
        if corner == 'top-left':
            path.push(f"M {x} {y}")
            path.push(f"A {rx} {ry} 0 0 1 {x + width} {y}")
            path.push(f"L {x} {y}")
        elif corner == 'top-right':
            path.push(f"M {x + width} {y}")
            path.push(f"A {rx} {ry} 0 0 1 {x + width} {y + height}")
            path.push(f"L {x + width} {y}")
        elif corner == 'bottom-right':
            path.push(f"M {x + width} {y + height}")
            path.push(f"A {rx} {ry} 0 0 1 {x} {y + height}")
            path.push(f"L {x + width} {y + height}")
        else:  # bottom-left
            path.push(f"M {x} {y + height}")
            path.push(f"A {rx} {ry} 0 0 1 {x} {y}")
            path.push(f"L {x} {y + height}")
        
        # Add decorative elements based on complexity
        if pattern_complexity > 0.7 and random.random() < 0.5:
            group.add(path)
            
            # Add a smaller contrasting circle
            if corner == 'top-left':
                center_x = x + width * 0.7
                center_y = y + height * 0.7
            elif corner == 'top-right':
                center_x = x + width * 0.3
                center_y = y + height * 0.7
            elif corner == 'bottom-right':
                center_x = x + width * 0.3
                center_y = y + height * 0.3
            else:  # bottom-left
                center_x = x + width * 0.7
                center_y = y + height * 0.3
                
            radius = min(width, height) / 5
            group.add(dwg.circle(center=(center_x, center_y), r=radius, fill=bg_fill))
        else:
            # Simple quarter circle
            group.add(path)
        
        dwg.add(group)

    def _draw_dots(self, dwg, x, y, width, height, foreground, background, 
                  pattern_complexity=0.5, enable_gradients=False, gradient_fill=""):
        """Draw a dots block."""
        group = dwg.g(class_="draw-dots")
        
        # Use gradient fill if enabled and provided
        bg_fill = gradient_fill if enable_gradients and gradient_fill else background
        
        group.add(dwg.rect((x, y), (width, height), fill=bg_fill))
        
        # Determine number of dots based on cell size and complexity
        base_dots = 3  # Minimum number of dots in each dimension
        size_factor = max(1, min(width, height) / 20)  # Scale based on cell size
        complexity_factor = 1 + pattern_complexity * 2  # More dots with higher complexity
        
        rows = max(base_dots, int(base_dots * size_factor * complexity_factor))
        cols = max(base_dots, int(base_dots * size_factor * complexity_factor))
        
        # Limit number of dots for performance
        rows = min(rows, 10)
        cols = min(cols, 10)
        
        # Choose dot pattern
        pattern_type = random.choice(['grid', 'circle', 'random']) if pattern_complexity > 0.5 else 'grid'
        
        cell_width = width / cols
        cell_height = height / rows
        dot_radius = min(cell_width, cell_height) * 0.3
        
        if pattern_type == 'grid':
            # Regular grid of dots
            for i in range(rows):
                for j in range(cols):
                    center_x = x + (j + 0.5) * cell_width
                    center_y = y + (i + 0.5) * cell_height
                    group.add(dwg.circle(center=(center_x, center_y), r=dot_radius, fill=foreground))
        
        elif pattern_type == 'circle':
            # Dots arranged in a circular pattern
            center_x = x + width / 2
            center_y = y + height / 2
            radius = min(width, height) * 0.4
            
            for i in range(cols):
                angle = 2 * math.pi * i / cols
                dot_x = center_x + radius * math.cos(angle)
                dot_y = center_y + radius * math.sin(angle)
                group.add(dwg.circle(center=(dot_x, dot_y), r=dot_radius, fill=foreground))
                
            # Add center dot
            if random.random() < 0.7:
                group.add(dwg.circle(center=(center_x, center_y), r=dot_radius*1.5, fill=foreground))
        
        else:  # random
            # Random distribution of dots
            num_dots = rows * cols // 2  # Fewer dots for random pattern
            
            for _ in range(num_dots):
                dot_x = x + random.random() * width
                dot_y = y + random.random() * height
                dot_size = dot_radius * (0.8 + random.random() * 0.4)  # Varying sizes
                group.add(dwg.circle(center=(dot_x, dot_y), r=dot_size, fill=foreground))
        
        dwg.add(group)

    def _draw_letter_block(self, dwg, x, y, width, height, foreground, background, 
                          pattern_complexity=0.5, enable_gradients=False, gradient_fill=""):
        """Draw a letter block."""
        group = dwg.g(class_="draw-letter-block")
        
        # Use gradient fill if enabled and provided
        bg_fill = gradient_fill if enable_gradients and gradient_fill else background
        
        group.add(dwg.rect((x, y), (width, height), fill=bg_fill))
        
        # Define character sets based on complexity
        basic_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                     
        extended_chars = basic_chars + ['+', '-', '*', '/', '=', '#', '@', '&', '%', '$',
                                     '<', '>', '!', '?', '(', ')', '[', ']', '{', '}']
        
        # Choose character set based on complexity
        characters = extended_chars if pattern_complexity > 0.5 else basic_chars
        
        # Choose a character
        character = random.choice(characters)
        
        # Calculate text size based on cell dimensions
        font_size = min(width, height) * 0.8
        
        # For very wide or tall cells, adjust size
        if width > height * 2 or height > width * 2:
            font_size = min(width, height) * 1.2
        
        # Text position
        text_x = x + width / 2
        text_y = y + height / 2 + font_size * 0.3
        
        # Add decorative elements based on complexity
        if pattern_complexity > 0.8 and random.random() < 0.3:
            # Add a background shape
            shape_type = random.choice(['circle', 'square', 'diamond'])
            
            if shape_type == 'circle':
                radius = min(width, height) * 0.45
                group.add(dwg.circle(center=(x + width/2, y + height/2), 
                                    r=radius, fill=foreground))
                text_color = bg_fill
            elif shape_type == 'square':
                size = min(width, height) * 0.8
                group.add(dwg.rect((x + (width-size)/2, y + (height-size)/2), 
                                  (size, size), fill=foreground))
                text_color = bg_fill
            else:  # diamond
                size = min(width, height) * 0.7
                points = [
                    (x + width/2, y + (height-size)/2),
                    (x + (width+size)/2, y + height/2),
                    (x + width/2, y + (height+size)/2),
                    (x + (width-size)/2, y + height/2)
                ]
                group.add(dwg.polygon(points, fill=foreground))
                text_color = bg_fill
            
            # Add the character on top with inverted color
            text = dwg.text(character, insert=(text_x, text_y),
                          font_family="monospace", font_size=font_size,
                          font_weight="bold", fill=text_color, text_anchor="middle")
        else:
            # Simple character
            text = dwg.text(character, insert=(text_x, text_y),
                          font_family="monospace", font_size=font_size,
                          font_weight="bold", fill=foreground, text_anchor="middle")
        
        group.add(text)
        dwg.add(group)

    def _draw_wave(self, dwg, x, y, width, height, foreground, background, 
                 pattern_complexity=0.5, enable_gradients=False, gradient_fill=""):
        """Draw a wave pattern."""
        group = dwg.g(class_="draw-wave")
        
        # Use gradient fill if enabled and provided
        bg_fill = gradient_fill if enable_gradients and gradient_fill else background
        
        group.add(dwg.rect((x, y), (width, height), fill=bg_fill))
        
        # Determine wave parameters based on complexity
        amplitude = height * (0.2 + pattern_complexity * 0.3)
        frequency = 1 + int(pattern_complexity * 3)  # Number of waves
        
        # Choose direction
        is_horizontal = width >= height or random.random() < 0.5
        
        if is_horizontal:
            # Create horizontal waves
            points = []
            
            # Top edge points
            x_step = width / 20  # Use enough points for a smooth curve
            for i in range(21):
                x_pos = x + i * x_step
                y_offset = amplitude * math.sin(frequency * math.pi * i / 20)
                points.append((x_pos, y + y_offset))
                
            # Bottom edge points (in reverse)
            for i in range(20, -1, -1):
                x_pos = x + i * x_step
                y_offset = amplitude * math.sin(frequency * math.pi * i / 20)
                # Flip the wave for bottom edge
                points.append((x_pos, y + height - y_offset))
                
            group.add(dwg.polygon(points, fill=foreground))
            
        else:
            # Create vertical waves
            points = []
            
            # Left edge points
            y_step = height / 20
            for i in range(21):
                y_pos = y + i * y_step
                x_offset = amplitude * math.sin(frequency * math.pi * i / 20)
                points.append((x + x_offset, y_pos))
                
            # Right edge points (in reverse)
            for i in range(20, -1, -1):
                y_pos = y + i * y_step
                x_offset = amplitude * math.sin(frequency * math.pi * i / 20)
                # Flip the wave for right edge
                points.append((x + width - x_offset, y_pos))
                
            group.add(dwg.polygon(points, fill=foreground))
        
        dwg.add(group)

    def _draw_spiral(self, dwg, x, y, width, height, foreground, background, 
                    pattern_complexity=0.5, enable_gradients=False, gradient_fill=""):
        """Draw a spiral pattern."""
        group = dwg.g(class_="draw-spiral")
        
        # Use gradient fill if enabled and provided
        bg_fill = gradient_fill if enable_gradients and gradient_fill else background
        
        group.add(dwg.rect((x, y), (width, height), fill=bg_fill))
        
        # Center of the spiral
        center_x = x + width / 2
        center_y = y + height / 2
        
        # Determine spiral parameters
        max_radius = min(width, height) * 0.45
        turns = 2 + pattern_complexity * 3  # More turns with higher complexity
        points_per_turn = 20  # Points to generate per turn
        
        # Spiral thickness
        thickness = max_radius * (0.1 + pattern_complexity * 0.1)
        
        # Generate spiral points
        points = []
        angle_step = 2 * math.pi / points_per_turn
        
        # Function to calculate spiral radius at a given angle
        def spiral_radius(angle, max_r, turns):
            return max_r * (angle / (2 * math.pi * turns))
        
        # Generate outer edge of spiral
        for i in range(int(points_per_turn * turns) + 1):
            angle = i * angle_step
            radius = spiral_radius(angle, max_radius, turns)
            x_pos = center_x + radius * math.cos(angle)
            y_pos = center_y + radius * math.sin(angle)
            points.append((x_pos, y_pos))
        
        # Generate inner edge of spiral (in reverse)
        for i in range(int(points_per_turn * turns), -1, -1):
            angle = i * angle_step
            radius = spiral_radius(angle, max_radius, turns) - thickness
            radius = max(0, radius)  # Ensure non-negative radius
            x_pos = center_x + radius * math.cos(angle)
            y_pos = center_y + radius * math.sin(angle)
            points.append((x_pos, y_pos))
        
        # Create the spiral
        if len(points) > 3:  # Need at least 3 points for a polygon
            group.add(dwg.polygon(points, fill=foreground))
        
        dwg.add(group)

    def _draw_chevron(self, dwg, x, y, width, height, foreground, background, 
                     pattern_complexity=0.5, enable_gradients=False, gradient_fill=""):
        """Draw a chevron pattern."""
        group = dwg.g(class_="draw-chevron")
        
        # Use gradient fill if enabled and provided
        bg_fill = gradient_fill if enable_gradients and gradient_fill else background
        
        group.add(dwg.rect((x, y), (width, height), fill=bg_fill))
        
        # Determine chevron parameters
        is_horizontal = width >= height
        
        # Number of chevrons based on complexity and cell size
        num_chevrons = 1 + int(pattern_complexity * 3)
        
        # Chevron sharpness (0.5 = triangle, 0 = rectangle)
        sharpness = 0.2 + pattern_complexity * 0.3
        
        if is_horizontal:
            # Horizontal chevrons
            chevron_height = height / num_chevrons
            
            for i in range(num_chevrons):
                y_start = y + i * chevron_height
                y_end = y_start + chevron_height
                
                # Calculate chevron points
                points = [
                    (x, y_start),  # Top-left
                    (x + width, y_start),  # Top-right
                    (x + width, y_end),  # Bottom-right
                    (x + width * (1 - sharpness), (y_start + y_end) / 2),  # Right point
                    (x, y_end)  # Bottom-left
                ]
                
                # Draw alternating chevrons
                fill_color = foreground if i % 2 == 0 else bg_fill
                group.add(dwg.polygon(points, fill=fill_color))
        else:
            # Vertical chevrons
            chevron_width = width / num_chevrons
            
            for i in range(num_chevrons):
                x_start = x + i * chevron_width
                x_end = x_start + chevron_width
                
                # Calculate chevron points
                points = [
                    (x_start, y),  # Top-left
                    (x_end, y),  # Top-right
                    (x_end, y + height),  # Bottom-right
                    (x_start, y + height),  # Bottom-left
                    ((x_start + x_end) / 2, y + height * (1 - sharpness))  # Bottom point
                ]
                
                # Draw alternating chevrons
                fill_color = foreground if i % 2 == 0 else bg_fill
                group.add(dwg.polygon(points, fill=fill_color))
        
        dwg.add(group)

    def _draw_checkerboard(self, dwg, x, y, width, height, foreground, background, 
                          pattern_complexity=0.5, enable_gradients=False, gradient_fill=""):
        """Draw a checkerboard pattern."""
        group = dwg.g(class_="draw-checkerboard")
        
        # Use gradient fill if enabled and provided
        bg_fill = gradient_fill if enable_gradients and gradient_fill else background
        
        group.add(dwg.rect((x, y), (width, height), fill=bg_fill))
        
        # Determine grid size based on complexity and cell dimensions
        base_size = 2  # Minimum checkerboard size
        size_factor = max(1, min(width, height) / 20)  # Scale based on cell size
        complexity_factor = 1 + pattern_complexity * 3  # More checks with higher complexity
        
        grid_size = max(base_size, int(base_size * size_factor * complexity_factor))
        grid_size = min(grid_size, 8)  # Limit for performance
        
        # Calculate square dimensions
        square_width = width / grid_size
        square_height = height / grid_size
        
        # Draw the checkerboard
        for i in range(grid_size):
            for j in range(grid_size):
                # Determine if this square should be filled
                if (i + j) % 2 == 0:
                    square_x = x + j * square_width
                    square_y = y + i * square_height
                    group.add(dwg.rect((square_x, square_y), 
                                     (square_width, square_height), 
                                     fill=foreground))
        
        dwg.add(group)

    def _draw_fractal(self, dwg, x, y, width, height, foreground, background, 
                     pattern_complexity=0.5, enable_gradients=False, gradient_fill=""):
        """Draw a simple fractal-like pattern (simplified Sierpinski)."""
        group = dwg.g(class_="draw-fractal")
        
        # Use gradient fill if enabled and provided
        bg_fill = gradient_fill if enable_gradients and gradient_fill else background
        
        group.add(dwg.rect((x, y), (width, height), fill=bg_fill))
        
        # Determine fractal depth based on complexity
        max_depth = min(3, int(1 + pattern_complexity * 3))
        
        # Choose fractal type
        fractal_type = random.choice(['triangle', 'square'])
        
        if fractal_type == 'triangle':
            # Sierpinski triangle
            def draw_sierpinski(x1, y1, x2, y2, x3, y3, depth):
                if depth == 0:
                    # Draw a triangle
                    group.add(dwg.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=foreground))
                else:
                    # Calculate midpoints
                    mid_x1 = (x1 + x2) / 2
                    mid_y1 = (y1 + y2) / 2
                    mid_x2 = (x2 + x3) / 2
                    mid_y2 = (y2 + y3) / 2
                    mid_x3 = (x3 + x1) / 2
                    mid_y3 = (y3 + y1) / 2
                    
                    # Recursively draw three smaller triangles
                    draw_sierpinski(x1, y1, mid_x1, mid_y1, mid_x3, mid_y3, depth - 1)
                    draw_sierpinski(mid_x1, mid_y1, x2, y2, mid_x2, mid_y2, depth - 1)
                    draw_sierpinski(mid_x3, mid_y3, mid_x2, mid_y2, x3, y3, depth - 1)
            
            # Initial triangle points
            if width > height:
                # For wide cells, use a wide triangle
                x1, y1 = x, y + height  # Bottom-left
                x2, y2 = x + width, y + height  # Bottom-right
                x3, y3 = x + width / 2, y  # Top-middle
            else:
                # For tall cells, use a tall triangle
                x1, y1 = x, y + height  # Bottom-left
                x2, y2 = x + width, y + height  # Bottom-right
                x3, y3 = x + width / 2, y  # Top-middle
            
            # Draw the fractal
            draw_sierpinski(x1, y1, x2, y2, x3, y3, max_depth)
            
        else:  # square
            # Simplified "Sierpinski carpet"
            def draw_carpet(x, y, size, depth):
                if depth == 0:
                    # Draw a square
                    group.add(dwg.rect((x, y), (size, size), fill=foreground))
                else:
                    # Calculate new size
                    new_size = size / 3
                    
                    # Draw 8 smaller squares (skipping the center)
                    for i in range(3):
                        for j in range(3):
                            # Skip the center square
                            if i == 1 and j == 1:
                                continue
                                
                            # Calculate position
                            new_x = x + j * new_size
                            new_y = y + i * new_size
                            
                            # Recursively draw smaller square
                            draw_carpet(new_x, new_y, new_size, depth - 1)
            
            # Initial square size
            size = min(width, height)
            
            # Center the pattern
            start_x = x + (width - size) / 2
            start_y = y + (height - size) / 2
            
            # Draw the fractal
            draw_carpet(start_x, start_y, size, max_depth)
        
        dwg.add(group)

    def _generate_block(self, dwg, cell: CellDefinition, square_size: int, color_palette, 
                       enabled_styles, pil_image=None, mode="palette", blend_factor=0.7,
                       pattern_complexity=0.5, enable_gradients=False, gradient_type=GradientType.LINEAR,
                       gradient_angle=45, skip_positions=None):
        """Generate a single block for a cell."""
        # Calculate pixel coordinates
        x, y, width, height = cell.get_coords(square_size)
        
        # Skip if this position is in the text blocks
        if skip_positions and any((cell.row + i, cell.col + j) in skip_positions 
                                for i in range(cell.height) for j in range(cell.width)):
            return
        
        # Create a mapping of pattern types to drawing functions
        pattern_map = {
            PatternType.CIRCLE.value: self._draw_circle,
            PatternType.OPPOSITE_CIRCLES.value: self._draw_opposite_circles,
            PatternType.CROSS.value: self._draw_cross,
            PatternType.HALF_SQUARE.value: self._draw_half_square,
            PatternType.DIAGONAL_SQUARE.value: self._draw_diagonal_square,
            PatternType.QUARTER_CIRCLE.value: self._draw_quarter_circle,
            PatternType.DOTS.value: self._draw_dots,
            PatternType.LETTER_BLOCK.value: self._draw_letter_block,
            PatternType.WAVE.value: self._draw_wave,
            PatternType.SPIRAL.value: self._draw_spiral,
            PatternType.CHEVRON.value: self._draw_chevron,
            PatternType.CHECKERBOARD.value: self._draw_checkerboard,
            PatternType.FRACTAL.value: self._draw_fractal
        }
        
        # Filter for available styles that are enabled
        available_styles = [style for style in enabled_styles if style in pattern_map]
        if not available_styles:
            # Default to circle if no styles are enabled
            available_styles = [PatternType.CIRCLE.value]
        
        # Choose a random pattern from available styles
        pattern_type = random.choice(available_styles)
        draw_func = pattern_map[pattern_type]
        
        # Determine colors based on mode
        if mode == "composition" and pil_image:
            # Sample colors from this region of the image
            colors = self._sample_image_region(pil_image, x, y, width, height, 
                                             sample_count=2, blend_factor=blend_factor)
            foreground = colors["foreground"]
            background = colors["background"]
        else:
            # Use palette colors
            colors = self._get_two_colors(color_palette)
            foreground = colors["foreground"]
            background = colors["background"]
        
        # Create gradient fill if enabled
        gradient_fill = ""
        if enable_gradients and random.random() < 0.7:  # 70% chance of using gradient for enabled cells
            gradient_fill = self._create_gradient_fill(
                dwg, color_palette, gradient_type, gradient_angle, 
                cell_id=f"cell-{cell.row}-{cell.col}"
            )
        
        # Draw the pattern
        draw_func(dwg, x, y, width, height, foreground, background, 
                 pattern_complexity, enable_gradients, gradient_fill)

    def _generate_grid_blocks(self, dwg, cells, square_size, color_palette, enabled_styles, 
                            pil_image=None, mode="palette", blend_factor=0.7, 
                            pattern_complexity=0.5, enable_gradients=False, 
                            gradient_type=GradientType.LINEAR, gradient_angle=45,
                            skip_positions=None):
        """Generate blocks for all cells in the grid."""
        for cell in cells:
            self._generate_block(
                dwg, cell, square_size, color_palette, enabled_styles, 
                pil_image, mode, blend_factor, pattern_complexity, 
                enable_gradients, gradient_type, gradient_angle, skip_positions
            )

    def _generate_big_block(self, dwg, cells, square_size, color_palette, enabled_styles, 
                          big_block_size, pattern_complexity=0.5, enable_gradients=False, 
                          gradient_type=GradientType.LINEAR, gradient_angle=45,
                          skip_positions=None):
        """Generate a big block, avoiding text block positions."""
        # Use only basic patterns for big blocks
        basic_styles = [
            PatternType.CIRCLE.value,
            PatternType.OPPOSITE_CIRCLES.value,
            PatternType.CROSS.value,
            PatternType.HALF_SQUARE.value,
            PatternType.DIAGONAL_SQUARE.value,
            PatternType.QUARTER_CIRCLE.value,
            PatternType.LETTER_BLOCK.value
        ]
        
        # Filter for available styles that are enabled
        available_styles = [style for style in enabled_styles if style in basic_styles]
        if not available_styles:
            # Default to circle if no styles are enabled
            available_styles = [PatternType.CIRCLE.value]
        
        # Find all cells with size 1x1 (candidates for replacement)
        single_cells = [cell for cell in cells if cell.width == 1 and cell.height == 1]
        
        if not single_cells:
            return False  # No suitable cells found
        
        # Find a valid position for the big block
        attempts = 0
        max_attempts = 50  # Prevent infinite loop
        
        while attempts < max_attempts and single_cells:
            # Choose a random starting cell
            start_cell_idx = random.randint(0, len(single_cells) - 1)
            start_cell = single_cells[start_cell_idx]
            
            # Check if we can fit a big block here
            can_fit = True
            
            # Check skip positions
            if skip_positions:
                for r in range(start_cell.row, start_cell.row + big_block_size):
                    for c in range(start_cell.col, start_cell.col + big_block_size):
                        if (r, c) in skip_positions:
                            can_fit = False
                            break
                    if not can_fit:
                        break
            
            # Check other cells
            if can_fit:
                for cell in cells:
                    if cell == start_cell:
                        continue
                        
                    for r in range(start_cell.row, start_cell.row + big_block_size):
                        for c in range(start_cell.col, start_cell.col + big_block_size):
                            if cell.covers_position(r, c):
                                can_fit = False
                                break
                        if not can_fit:
                            break
                    if not can_fit:
                        break
            
            if can_fit:
                # Create the big block
                big_cell = CellDefinition(
                    start_cell.row, start_cell.col, 
                    big_block_size, big_block_size
                )
                
                # Generate a random color pair
                colors = self._get_two_colors(color_palette)
                
                # Create gradient fill if enabled
                gradient_fill = ""
                if enable_gradients and random.random() < 0.5:
                    gradient_fill = self._create_gradient_fill(
                        dwg, color_palette, gradient_type, gradient_angle, 
                        cell_id=f"big-block"
                    )
                
                # Calculate coordinates
                x, y, width, height = big_cell.get_coords(square_size)
                
                # Choose a random pattern
                pattern_type = random.choice(available_styles)
                draw_func = {
                    PatternType.CIRCLE.value: self._draw_circle,
                    PatternType.OPPOSITE_CIRCLES.value: self._draw_opposite_circles,
                    PatternType.CROSS.value: self._draw_cross,
                    PatternType.HALF_SQUARE.value: self._draw_half_square,
                    PatternType.DIAGONAL_SQUARE.value: self._draw_diagonal_square,
                    PatternType.QUARTER_CIRCLE.value: self._draw_quarter_circle,
                    PatternType.LETTER_BLOCK.value: self._draw_letter_block
                }[pattern_type]
                
                # Draw the pattern
                draw_func(dwg, x, y, width, height, colors["foreground"], colors["background"],
                         pattern_complexity, enable_gradients, gradient_fill)
                
                return True  # Successfully generated big block
            
            # Remove this cell from candidates and try again
            single_cells.pop(start_cell_idx)
            attempts += 1
        
        return False  # Could not find a valid position
        
    def _svg_to_png(self, svg_string, width, height):
        """Convert SVG string to PNG image using cairosvg."""
        if CAIRO_AVAILABLE:
            try:
                png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'),
                                        output_width=width, output_height=height)
                image = Image.open(io.BytesIO(png_data))
                return image
            except Exception as e:
                print(f"SVGArtGridV4: cairosvg conversion failed: {e}")
        
        # Fallback to simple PIL representation
        print("SVGArtGridV4: Using PIL fallback for SVG rendering (cairo not available)")
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        # Draw a simple grid as fallback
        grid_size = 10
        for i in range(grid_size + 1):
            x = i * (width // grid_size)
            draw.line([(x, 0), (x, height)], fill='black', width=1)
        for j in range(grid_size + 1):
            y = j * (height // grid_size)
            draw.line([(0, y), (width, y)], fill='black', width=1)
        return image
        
    def _tensor_to_pil(self, tensor):
        """Convert a PyTorch tensor to a PIL image."""
        # ComfyUI provides tensors in format [batch, height, width, channel]
        # Take the first image if it's batched
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        # Make sure tensor is on CPU and convert to numpy
        img_np = tensor.cpu().numpy()
        
        # Scale values from [0,1] to [0,255] and convert to uint8
        img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to PIL - the tensor is already in HWC format in ComfyUI
        return Image.fromarray(img_np)
    
    def generate_art_grid(self, image, seed, grid_template, rows, cols, adaptive_grid, variable_cells,
                        focal_point_x, focal_point_y, mode, palette_index, color_count, blend_factor, 
                        enable_big_block, big_block_size, enable_circle, enable_opposite_circles, 
                        enable_cross, enable_half_square, enable_diagonal_square, enable_quarter_circle, 
                        enable_dots, enable_letter_block, enable_wave, enable_spiral, enable_chevron, 
                        enable_checkerboard, enable_fractal, pattern_complexity, enable_gradients, 
                        gradient_type, gradient_angle, output_svg, text_blocks="[]", 
                        palette_url="https://unpkg.com/nice-color-palettes@3.0.0/100.json",
                        block_styles="all", cell_definitions="[]"):
        """Generate the SVG art grid using the input image."""
        # Set random seed
        seed = seed % (2**32)
        random.seed(seed)
        
        # Check image tensor dimensions
        if len(image.shape) != 4:
            raise ValueError(f"Expected 4D tensor for image, got shape {image.shape}")
        
        # Get dimensions from the tensor directly (ComfyUI format is BHWC)
        batch_size, height, width, channels = image.shape
        
        # Convert image tensor to PIL image
        pil_image = self._tensor_to_pil(image)
        
        # Calculate square size to fit the canvas
        square_size = min(width // rows, height // cols)
        svg_width = rows * square_size
        svg_height = cols * square_size
        
        # Handle invalid square size
        if square_size <= 0:
            print(f"SVGArtGridV4: Warning - Invalid square size {square_size}. Resetting to default.")
            square_size = 20  # Default fallback size
            svg_width = rows * square_size
            svg_height = cols * square_size
        
        # Load or extract color palettes based on mode
        if mode == "palette":
            # Extract palette from the input image
            if palette_index == -1:
                color_palette = self._extract_palette_from_image(pil_image, color_count)
            else:
                # Use traditional palette from JSON
                palettes = self._load_color_palettes(palette_url)
                palette_idx = min(palette_index, len(palettes) - 1)
                color_palette = palettes[palette_idx]
        else:
            # In composition mode, we still need a basic palette for background
            palettes = self._load_color_palettes(palette_url)
            palette_idx = random.randint(0, len(palettes) - 1)
            color_palette = palettes[palette_idx]
        
        # Create list of enabled pattern types
        enabled_styles = []
        if enable_circle:
            enabled_styles.append(PatternType.CIRCLE.value)
        if enable_opposite_circles:
            enabled_styles.append(PatternType.OPPOSITE_CIRCLES.value)
        if enable_cross:
            enabled_styles.append(PatternType.CROSS.value)
        if enable_half_square:
            enabled_styles.append(PatternType.HALF_SQUARE.value)
        if enable_diagonal_square:
            enabled_styles.append(PatternType.DIAGONAL_SQUARE.value)
        if enable_quarter_circle:
            enabled_styles.append(PatternType.QUARTER_CIRCLE.value)
        if enable_dots:
            enabled_styles.append(PatternType.DOTS.value)
        if enable_letter_block:
            enabled_styles.append(PatternType.LETTER_BLOCK.value)
        if enable_wave:
            enabled_styles.append(PatternType.WAVE.value)
        if enable_spiral:
            enabled_styles.append(PatternType.SPIRAL.value)
        if enable_chevron:
            enabled_styles.append(PatternType.CHEVRON.value)
        if enable_checkerboard:
            enabled_styles.append(PatternType.CHECKERBOARD.value)
        if enable_fractal:
            enabled_styles.append(PatternType.FRACTAL.value)
            
        # If no styles are enabled, default to circle
        if not enabled_styles:
            enabled_styles = [PatternType.CIRCLE.value]
        
        # For backward compatibility - parse block styles from text
        if block_styles.lower() != "all":
            legacy_styles = [s.strip() for s in block_styles.split(',')]
            # Map old style names to new enum values
            style_map = {
                'circle': PatternType.CIRCLE.value,
                'opposite_circles': PatternType.OPPOSITE_CIRCLES.value,
                'cross': PatternType.CROSS.value,
                'half_square': PatternType.HALF_SQUARE.value,
                'diagonal_square': PatternType.DIAGONAL_SQUARE.value,
                'quarter_circle': PatternType.QUARTER_CIRCLE.value,
                'dots': PatternType.DOTS.value,
                'letter_block': PatternType.LETTER_BLOCK.value
            }
            legacy_mapped = [style_map.get(s, s) for s in legacy_styles if s in style_map]
            
            # If specific styles were provided, use the intersection
            if legacy_mapped:
                enabled_styles = [s for s in enabled_styles if s in legacy_mapped]
                
                # If now empty (no overlap), fall back to specified legacy styles
                if not enabled_styles:
                    enabled_styles = legacy_mapped
        
        # Create SVG
        dwg = svgwrite.Drawing(size=(f"{svg_width}px", f"{svg_height}px"), profile='full')
        dwg.defs.add(dwg.style("svg * { shape-rendering: crispEdges; }"))
        
        # Add background gradient
        bg_colors = self._create_background_colors(color_palette)
        gradient = dwg.defs.add(dwg.radialGradient(id="background_gradient"))
        gradient.add_stop_color(0, bg_colors["bg_inner"])
        gradient.add_stop_color(1, bg_colors["bg_outer"])
        dwg.add(dwg.rect((0, 0), (svg_width, svg_height), fill="url(#background_gradient)"))
        
        # Generate cells based on template and settings
        cells = []
        
        if variable_cells:
            if cell_definitions != "[]":
                # Use custom cell definitions
                cells = self._parse_cell_definitions(cell_definitions, rows, cols)
            elif adaptive_grid and grid_template == GridTemplate.ADAPTIVE.value:
                # Generate adaptive grid based on image content
                cells = self._generate_adaptive_grid(pil_image, rows, cols)
            else:
                # Use selected template
                cells = self._generate_template_grid(grid_template, rows, cols, focal_point_x, focal_point_y)
        else:
            # Regular grid - all cells are 1x1
            cells = [CellDefinition(row, col) for row in range(rows) for col in range(cols)]
        
        # Parse text blocks
        text_block_objects = self._parse_text_blocks(text_blocks)
        text_positions = set()
        
        # Get positions covered by text blocks
        for block in text_block_objects:
            text_positions.update(block.get_positions())
        
        # Generate grid blocks based on mode
        if mode == "composition":
            # Resize image to match grid dimensions
            composition_img = pil_image.resize((svg_width, svg_height))
            # Generate grid from image composition
            self._generate_grid_blocks(
                dwg, cells, square_size, color_palette, enabled_styles,
                composition_img, mode, blend_factor, pattern_complexity,
                enable_gradients, gradient_type, gradient_angle, text_positions
            )
        else:
            # Generate traditional grid using palette
            self._generate_grid_blocks(
                dwg, cells, square_size, color_palette, enabled_styles,
                None, mode, blend_factor, pattern_complexity,
                enable_gradients, gradient_type, gradient_angle, text_positions
            )
        
        # Add big block if enabled (only in palette mode for better visual results)
        if enable_big_block and mode == "palette":
            self._generate_big_block(
                dwg, cells, square_size, color_palette, enabled_styles,
                big_block_size, pattern_complexity, enable_gradients,
                gradient_type, gradient_angle, text_positions
            )
        
        # Render text blocks
        for block in text_block_objects:
            # Get colors from palette if not overridden
            colors = self._get_two_colors(color_palette)
            block.render(dwg, square_size, colors)
        
        # Get SVG string
        svg_string = dwg.tostring()
        
        # Convert to PNG for ComfyUI
        image = self._svg_to_png(svg_string, width, height)
            
        # Convert PIL image to tensor format for ComfyUI (BHWC format)
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Make sure we have a 3D array (HWC)
        if len(image_np.shape) == 2:  # Grayscale
            image_np = np.expand_dims(image_np, axis=2)
            
        # Create tensor and add batch dimension if needed (to BHWC)
        image_tensor = torch.from_numpy(image_np)
        if len(image_tensor.shape) == 3:  # HWC -> BHWC
            image_tensor = image_tensor.unsqueeze(0)
        
        # Always return a non-empty SVG string, even when output_svg is False
        # This helps downstream nodes that might expect a valid SVG
        return (image_tensor, svg_string)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SVGArtGridV4": SVGArtGridV4
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVGArtGridV4": "SVG Art Grid V4"
}
