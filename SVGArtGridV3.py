#!/usr/bin/env python3
"""
SVGArtGridV3.py - A ComfyUI node that generates artistic SVG grids with image input
Enhanced version of SVGArtGridV2 with individual grid type toggles and text block feature
"""

import random
import json
import requests
import math
import colorsys
import svgwrite
from PIL import Image, ImageDraw
import numpy as np
import torch
import io
from typing import List, Tuple, Dict, Any

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

class SVGArtGridV3:
    """A ComfyUI node that generates Kandinsky-style artistic SVG grids with enhanced customization options"""
    
    BLOCK_STYLES = [
        'circle', 'opposite_circles', 'cross', 'half_square', 
        'diagonal_square', 'quarter_circle', 'dots', 'letter_block'
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 6, "min": 1, "max": 30, "step": 1}),
                "cols": ("INT", {"default": 6, "min": 1, "max": 30, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
                "palette_index": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1,
                                        "tooltip": "-1 for random palette"}),
                "mode": (["palette", "composition"], {"default": "palette"}),
                "color_count": ("INT", {"default": 5, "min": 3, "max": 10, "step": 1}),
                "blend_factor": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "big_block": ("BOOLEAN", {"default": True}),
                "big_block_size": ("INT", {"default": 2, "min": 2, "max": 3, "step": 1}),
                "output_svg": ("BOOLEAN", {"default": False, "tooltip": "Also output SVG string"}),
                # Individual grid type toggles
                "enable_circle": ("BOOLEAN", {"default": True}),
                "enable_opposite_circles": ("BOOLEAN", {"default": True}),
                "enable_cross": ("BOOLEAN", {"default": True}),
                "enable_half_square": ("BOOLEAN", {"default": True}),
                "enable_diagonal_square": ("BOOLEAN", {"default": True}),
                "enable_quarter_circle": ("BOOLEAN", {"default": True}),
                "enable_dots": ("BOOLEAN", {"default": True}),
                "enable_letter_block": ("BOOLEAN", {"default": True}),
                # Text block parameters
                "enable_text_block": ("BOOLEAN", {"default": False}),
                "text_block_orientation": (["horizontal", "vertical"], {"default": "horizontal"}),
                "text_block_characters": ("STRING", {"default": "COMFY", "multiline": False, 
                                                  "tooltip": "Up to 8 characters will be used"}),
                "text_color_override": ("STRING", {"default": "", "multiline": False,
                                                "tooltip": "Optional hex color (e.g. #FF0000) for text. Leave empty to use palette"}),
                "text_bg_color_override": ("STRING", {"default": "", "multiline": False,
                                                   "tooltip": "Optional hex color for background. Leave empty to use palette"}),
            },
            "optional": {
                "palette_url": ("STRING", {"default": "https://unpkg.com/nice-color-palettes@3.0.0/100.json",
                                         "multiline": False}),
                "block_styles": ("STRING", {"default": "all", "multiline": False,
                                          "tooltip": "Legacy: comma-separated list or 'all'. Use grid type toggles instead."}),
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
            print("SVGArtGridV3: sklearn not available, using default palette")
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

    def _draw_circle(self, dwg, x, y, square_size, foreground, background):
        """Draw a circle block."""
        group = dwg.g(class_="draw-circle")
        group.add(dwg.rect((x, y), (square_size, square_size), fill=background))
        group.add(dwg.circle(center=(x + square_size/2, y + square_size/2), 
                             r=square_size/2, fill=foreground))
        
        if random.random() < 0.3:
            group.add(dwg.circle(center=(x + square_size/2, y + square_size/2),
                                 r=square_size/4, fill=background))
        dwg.add(group)

    def _draw_opposite_circles(self, dwg, x, y, square_size, foreground, background):
        """Draw opposite circles block."""
        group = dwg.g(class_="opposite-circles")
        group.add(dwg.rect((x, y), (square_size, square_size), fill=background))
        
        mask_id = f"mask-{x}-{y}"
        mask = dwg.mask(id=mask_id)
        mask.add(dwg.rect((x, y), (square_size, square_size), fill="white"))
        dwg.defs.add(mask)
        
        options = [[0, 0, square_size, square_size], [square_size, 0, 0, square_size]]
        offset = random.choice(options)
        
        circle_group = dwg.g()
        circle_group.add(dwg.circle(center=(x + offset[0], y + offset[1]), 
                                   r=square_size/2, fill=foreground))
        circle_group.add(dwg.circle(center=(x + offset[2], y + offset[3]), 
                                   r=square_size/2, fill=foreground))
        circle_group['mask'] = f"url(#{mask_id})"
        
        group.add(circle_group)
        dwg.add(group)

    def _draw_cross(self, dwg, x, y, square_size, foreground, background):
        """Draw a cross or X block."""
        group = dwg.g(class_="draw-cross")
        group.add(dwg.rect((x, y), (square_size, square_size), fill=background))
        
        is_plus = random.random() < 0.5
        
        if is_plus:
            group.add(dwg.rect((x, y + square_size/3), 
                              (square_size, square_size/3), fill=foreground))
            group.add(dwg.rect((x + square_size/3, y), 
                              (square_size/3, square_size), fill=foreground))
        else:
            width = square_size / 6
            
            # First diagonal
            x1, y1 = x, y
            x2, y2 = x + square_size, y + square_size
            dx, dy = x2 - x1, y2 - y1
            length = (dx**2 + dy**2)**0.5
            nx, ny = -dy/length, dx/length
            
            p1 = (x1 + nx*width/2, y1 + ny*width/2)
            p2 = (x2 + nx*width/2, y2 + ny*width/2)
            p3 = (x2 - nx*width/2, y2 - ny*width/2)
            p4 = (x1 - nx*width/2, y1 - ny*width/2)
            group.add(dwg.polygon([p1, p2, p3, p4], fill=foreground))
            
            # Second diagonal
            x1, y1 = x + square_size, y
            x2, y2 = x, y + square_size
            dx, dy = x2 - x1, y2 - y1
            length = (dx**2 + dy**2)**0.5
            nx, ny = -dy/length, dx/length
            
            p1 = (x1 + nx*width/2, y1 + ny*width/2)
            p2 = (x2 + nx*width/2, y2 + ny*width/2)
            p3 = (x2 - nx*width/2, y2 - ny*width/2)
            p4 = (x1 - nx*width/2, y1 - ny*width/2)
            group.add(dwg.polygon([p1, p2, p3, p4], fill=foreground))
        
        dwg.add(group)

    def _draw_half_square(self, dwg, x, y, square_size, foreground, background):
        """Draw a half square block."""
        group = dwg.g(class_="draw-half-square")
        group.add(dwg.rect((x, y), (square_size, square_size), fill=background))
        
        direction = random.choice(['top', 'right', 'bottom', 'left'])
        
        if direction == 'top':
            points = [(x, y), (x + square_size, y), (x + square_size, y + square_size/2), (x, y + square_size/2)]
        elif direction == 'right':
            points = [(x + square_size/2, y), (x + square_size, y), (x + square_size, y + square_size), (x + square_size/2, y + square_size)]
        elif direction == 'bottom':
            points = [(x, y + square_size/2), (x + square_size, y + square_size/2), (x + square_size, y + square_size), (x, y + square_size)]
        else:
            points = [(x, y), (x + square_size/2, y), (x + square_size/2, y + square_size), (x, y + square_size)]
        
        group.add(dwg.polygon(points, fill=foreground))
        dwg.add(group)

    def _draw_diagonal_square(self, dwg, x, y, square_size, foreground, background):
        """Draw a diagonal square block."""
        group = dwg.g(class_="draw-diagonal-square")
        group.add(dwg.rect((x, y), (square_size, square_size), fill=background))
        
        is_top_left_to_bottom_right = random.random() < 0.5
        
        if is_top_left_to_bottom_right:
            points = [(x, y), (x + square_size, y + square_size), (x, y + square_size)]
        else:
            points = [(x + square_size, y), (x + square_size, y + square_size), (x, y)]
        
        group.add(dwg.polygon(points, fill=foreground))
        dwg.add(group)

    def _draw_quarter_circle(self, dwg, x, y, square_size, foreground, background):
        """Draw a quarter circle block."""
        group = dwg.g(class_="draw-quarter-circle")
        group.add(dwg.rect((x, y), (square_size, square_size), fill=background))
        
        corner = random.choice(['top-left', 'top-right', 'bottom-right', 'bottom-left'])
        path = dwg.path(fill=foreground)
        
        if corner == 'top-left':
            path.push(f"M {x} {y}")
            path.push(f"A {square_size} {square_size} 0 0 1 {x + square_size} {y}")
            path.push(f"L {x} {y}")
        elif corner == 'top-right':
            path.push(f"M {x + square_size} {y}")
            path.push(f"A {square_size} {square_size} 0 0 1 {x + square_size} {y + square_size}")
            path.push(f"L {x + square_size} {y}")
        elif corner == 'bottom-right':
            path.push(f"M {x + square_size} {y + square_size}")
            path.push(f"A {square_size} {square_size} 0 0 1 {x} {y + square_size}")
            path.push(f"L {x + square_size} {y + square_size}")
        else:
            path.push(f"M {x} {y + square_size}")
            path.push(f"A {square_size} {square_size} 0 0 1 {x} {y}")
            path.push(f"L {x} {y + square_size}")
        
        group.add(path)
        dwg.add(group)

    def _draw_dots(self, dwg, x, y, square_size, foreground, background):
        """Draw a dots block."""
        group = dwg.g(class_="draw-dots")
        group.add(dwg.rect((x, y), (square_size, square_size), fill=background))
        
        num_dots = random.choice([4, 9, 16])
        
        if num_dots == 4:
            rows, cols = 2, 2
        elif num_dots == 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 4
        
        cell_size = square_size / rows
        dot_radius = cell_size * 0.3
        
        for i in range(rows):
            for j in range(cols):
                center_x = x + (i + 0.5) * cell_size
                center_y = y + (j + 0.5) * cell_size
                group.add(dwg.circle(center=(center_x, center_y), r=dot_radius, fill=foreground))
        
        dwg.add(group)

    def _draw_letter_block(self, dwg, x, y, square_size, foreground, background):
        """Draw a letter block."""
        group = dwg.g(class_="draw-letter-block")
        group.add(dwg.rect((x, y), (square_size, square_size), fill=background))
        
        characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                     '+', '-', '*', '/', '=', '#', '@', '&', '%', '$']
        
        character = random.choice(characters)
        text = dwg.text(character, insert=(x + square_size/2, y + square_size/2 + square_size*0.3),
                        font_family="monospace", font_size=square_size*0.8,
                        font_weight="bold", fill=foreground, text_anchor="middle")
        group.add(text)
        dwg.add(group)

    def _draw_text_character(self, dwg, x, y, square_size, character, foreground, background):
        """Draw a custom character for the text block feature."""
        group = dwg.g(class_="text-block-character")
        group.add(dwg.rect((x, y), (square_size, square_size), fill=background))
        
        text = dwg.text(character, insert=(x + square_size/2, y + square_size/2 + square_size*0.3),
                      font_family="monospace", font_size=square_size*0.8,
                      font_weight="bold", fill=foreground, text_anchor="middle")
        group.add(text)
        dwg.add(group)

    def _generate_little_block(self, dwg, i, j, square_size, color_palette, enabled_styles, 
                              skip_positions=None):
        """Generate a single block in the grid, respecting the enabled styles."""
        # Skip if this position is reserved for text block
        if skip_positions and (i, j) in skip_positions:
            return
        
        colors = self._get_two_colors(color_palette)
        
        style_map = {
            'circle': self._draw_circle,
            'opposite_circles': self._draw_opposite_circles,
            'cross': self._draw_cross,
            'half_square': self._draw_half_square,
            'diagonal_square': self._draw_diagonal_square,
            'quarter_circle': self._draw_quarter_circle,
            'dots': self._draw_dots,
            'letter_block': self._draw_letter_block
        }
        
        # Filter for available styles that are enabled
        available_styles = [style for style in enabled_styles if style in style_map]
        if not available_styles:
            # Default to circle if no styles are enabled
            available_styles = ['circle']
        
        style_func = style_map[random.choice(available_styles)]
        x_pos = i * square_size
        y_pos = j * square_size
        
        style_func(dwg, x_pos, y_pos, square_size, colors["foreground"], colors["background"])

    def _generate_composition_block(self, dwg, i, j, square_size, pil_image, enabled_styles, 
                                  blend_factor=0.7, skip_positions=None):
        """Generate a block based on image composition, respecting enabled styles."""
        # Skip if this position is reserved for text block
        if skip_positions and (i, j) in skip_positions:
            return
            
        x_pos = i * square_size
        y_pos = j * square_size
        
        # Sample colors from this region of the image
        colors = self._sample_image_region(pil_image, x_pos, y_pos, square_size, square_size, 
                                         sample_count=2, blend_factor=blend_factor)
        
        style_map = {
            'circle': self._draw_circle,
            'opposite_circles': self._draw_opposite_circles,
            'cross': self._draw_cross,
            'half_square': self._draw_half_square,
            'diagonal_square': self._draw_diagonal_square,
            'quarter_circle': self._draw_quarter_circle,
            'dots': self._draw_dots,
            'letter_block': self._draw_letter_block
        }
        
        # Filter for available styles that are enabled
        available_styles = [style for style in enabled_styles if style in style_map]
        if not available_styles:
            # Default to circle if no styles are enabled
            available_styles = ['circle']
        
        style_func = style_map[random.choice(available_styles)]
        style_func(dwg, x_pos, y_pos, square_size, colors["foreground"], colors["background"])

    def _generate_grid(self, dwg, num_rows, num_cols, square_size, color_palette, enabled_styles, 
                      skip_positions=None):
        """Generate the grid of blocks, respecting enabled styles."""
        for i in range(num_rows):
            for j in range(num_cols):
                self._generate_little_block(dwg, i, j, square_size, color_palette, enabled_styles, 
                                          skip_positions)

    def _generate_composition_grid(self, dwg, pil_image, num_rows, num_cols, square_size, enabled_styles, 
                                 blend_factor=0.7, skip_positions=None):
        """Generate a grid based on image composition, respecting enabled styles."""
        for i in range(num_rows):
            for j in range(num_cols):
                self._generate_composition_block(dwg, i, j, square_size, pil_image, enabled_styles, 
                                              blend_factor, skip_positions)

    def _generate_big_block(self, dwg, num_rows, num_cols, square_size, color_palette, enabled_styles, 
                          multiplier, skip_positions=None):
        """Generate a big block, respecting enabled styles and avoiding text block positions."""
        colors = self._get_two_colors(color_palette)
        
        style_map = {
            'circle': self._draw_circle,
            'opposite_circles': self._draw_opposite_circles,
            'cross': self._draw_cross,
            'half_square': self._draw_half_square,
            'diagonal_square': self._draw_diagonal_square,
            'quarter_circle': self._draw_quarter_circle,
            'letter_block': self._draw_letter_block
        }
        
        # Filter dots out and ensure we have enabled styles
        available_styles = [style for style in enabled_styles if style in style_map and style != 'dots']
        if not available_styles:
            # Default to circle if no styles are enabled
            available_styles = ['circle']
        
        # Find a valid position for the big block (not overlapping with text block)
        attempts = 0
        max_attempts = 50  # Prevent infinite loop
        
        while attempts < max_attempts:
            i = random.randint(0, num_rows - multiplier)
            j = random.randint(0, num_cols - multiplier)
            
            # Check if any position in the big block is reserved
            overlap = False
            if skip_positions:
                for di in range(multiplier):
                    for dj in range(multiplier):
                        if (i + di, j + dj) in skip_positions:
                            overlap = True
                            break
                    if overlap:
                        break
            
            if not overlap:
                # Found a valid position
                x_pos = i * square_size
                y_pos = j * square_size
                big_square_size = multiplier * square_size
                
                style_func = style_map[random.choice(available_styles)]
                style_func(dwg, x_pos, y_pos, big_square_size, colors["foreground"], colors["background"])
                return True  # Successfully generated
            
            attempts += 1
        
        # Could not find a valid position
        return False

    def _generate_text_block(self, dwg, square_size, color_palette, characters, orientation, 
                           text_color_override="", text_bg_color_override="",
                           grid_starting_pos=(1, 1)):
        """
        Generate a character text block starting at the specified position.
        
        Args:
            dwg: SVG drawing object
            square_size: Size of each grid square
            color_palette: Color palette to use
            characters: String of characters to display (up to 8)
            orientation: 'horizontal' or 'vertical'
            text_color_override: Optional hex color to override text color
            text_bg_color_override: Optional hex color to override background color
            grid_starting_pos: (i, j) grid position to start from (default: 1,1)
            
        Returns:
            set: Set of grid positions used by the text block
        """
        # Determine colors for the text block
        colors = self._get_two_colors(color_palette)
        
        # Use override colors if provided
        foreground = text_color_override if text_color_override and text_color_override.startswith('#') else colors["foreground"]
        background = text_bg_color_override if text_bg_color_override and text_bg_color_override.startswith('#') else colors["background"]
        
        # Initialize list to track positions used by text block
        text_positions = set()
        
        # Ensure we have at most 8 characters
        chars_to_use = characters[:8]
        # Pad with spaces if less than 8 characters provided
        chars_to_use = chars_to_use.ljust(8)
        
        start_i, start_j = grid_starting_pos
        
        # Draw the text block
        for idx, char in enumerate(chars_to_use):
            if orientation == 'horizontal':
                i, j = start_i + idx, start_j
            else:  # vertical
                i, j = start_i, start_j + idx
                
            # Add position to used positions
            text_positions.add((i, j))
            
            # Calculate pixel position
            x = i * square_size
            y = j * square_size
            
            # Draw the character
            self._draw_text_character(dwg, x, y, square_size, char, foreground, background)
            
        return text_positions

    def _svg_to_png(self, svg_string, width, height):
        """Convert SVG string to PNG image using cairosvg."""
        if CAIRO_AVAILABLE:
            try:
                png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'),
                                        output_width=width, output_height=height)
                image = Image.open(io.BytesIO(png_data))
                return image
            except Exception as e:
                print(f"SVGArtGridV3: cairosvg conversion failed: {e}")
        
        # Fallback to simple PIL representation
        print("SVGArtGridV3: Using PIL fallback for SVG rendering (cairo not available)")
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

    def generate_art_grid(self, image, rows, cols, seed, palette_index, mode, color_count, blend_factor,
                         big_block, big_block_size, output_svg,
                         enable_circle, enable_opposite_circles, enable_cross, enable_half_square,
                         enable_diagonal_square, enable_quarter_circle, enable_dots, enable_letter_block,
                         enable_text_block, text_block_orientation, text_block_characters,
                         text_color_override="", text_bg_color_override="",
                         palette_url="https://unpkg.com/nice-color-palettes@3.0.0/100.json",
                         block_styles="all"):
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
        
        # Create list of enabled styles based on boolean parameters
        enabled_styles = []
        if enable_circle:
            enabled_styles.append('circle')
        if enable_opposite_circles:
            enabled_styles.append('opposite_circles')
        if enable_cross:
            enabled_styles.append('cross')
        if enable_half_square:
            enabled_styles.append('half_square')
        if enable_diagonal_square:
            enabled_styles.append('diagonal_square')
        if enable_quarter_circle:
            enabled_styles.append('quarter_circle')
        if enable_dots:
            enabled_styles.append('dots')
        if enable_letter_block:
            enabled_styles.append('letter_block')
            
        # If no styles are enabled, default to all styles
        if not enabled_styles:
            enabled_styles = self.BLOCK_STYLES
        
        # For backward compatibility - parse block styles from text
        if block_styles.lower() != "all":
            legacy_styles = [s.strip() for s in block_styles.split(',')]
            # If specific styles were provided, use the intersection
            if legacy_styles:
                enabled_styles = [s for s in enabled_styles if s in legacy_styles]
                
                # If now empty (no overlap), fall back to specified legacy styles
                if not enabled_styles:
                    enabled_styles = legacy_styles
        
        # Create SVG
        dwg = svgwrite.Drawing(size=(f"{svg_width}px", f"{svg_height}px"), profile='full')
        dwg.defs.add(dwg.style("svg * { shape-rendering: crispEdges; }"))
        
        # Add background gradient
        bg_colors = self._create_background_colors(color_palette)
        gradient = dwg.defs.add(dwg.radialGradient(id="background_gradient"))
        gradient.add_stop_color(0, bg_colors["bg_inner"])
        gradient.add_stop_color(1, bg_colors["bg_outer"])
        dwg.add(dwg.rect((0, 0), (svg_width, svg_height), fill="url(#background_gradient)"))
        
        # Initialize text block positions (if enabled)
        text_positions = set()
        if enable_text_block and rows >= 9 and cols >= 9:  # Ensure grid is large enough for 8 chars
            # Generate text block and get the positions it uses
            text_positions = self._generate_text_block(
                dwg, square_size, color_palette, 
                text_block_characters, text_block_orientation,
                text_color_override, text_bg_color_override
            )
        
        # Generate grid based on mode, avoiding text block positions
        if mode == "composition":
            # Resize image to match grid dimensions
            composition_img = pil_image.resize((svg_width, svg_height))
            # Generate grid from image composition
            self._generate_composition_grid(dwg, composition_img, rows, cols, square_size, 
                                          enabled_styles, blend_factor, text_positions)
        else:
            # Generate traditional grid using palette
            self._generate_grid(dwg, rows, cols, square_size, color_palette, 
                              enabled_styles, text_positions)
        
        # Add big block if enabled (only in palette mode for better visual results)
        if big_block and mode == "palette" and not (enable_text_block and text_positions):
            self._generate_big_block(dwg, rows, cols, square_size, color_palette, 
                                   enabled_styles, big_block_size, text_positions)
        
        # Get SVG string
        svg_string = dwg.tostring()
        
        # Convert to PNG for ComfyUI
        image = self._svg_to_png(svg_string, width, height)
        
        # Ensure we have a valid SVG string even when square_size is too small
        if square_size <= 0:
            print(f"SVGArtGridV3: Warning - Invalid square size {square_size}. Resetting to default.")
            square_size = 20  # Default fallback size
            svg_width = rows * square_size
            svg_height = cols * square_size
            
            # Recreate the SVG with default size if needed
            dwg = svgwrite.Drawing(size=(f"{svg_width}px", f"{svg_height}px"), profile='full')
            dwg.defs.add(dwg.style("svg * { shape-rendering: crispEdges; }"))
            dwg.add(dwg.rect((0, 0), (svg_width, svg_height), fill="#FFFFFF"))
            svg_string = dwg.tostring()
            
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
    "SVGArtGridV3": SVGArtGridV3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVGArtGridV3": "SVG Art Grid V3"
}
