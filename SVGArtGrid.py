#!/usr/bin/env python3
"""
SVGArtGrid.py - A ComfyUI node that generates artistic SVG grids
Based on DinskyPlus.py template with SVG_ArtGrid.py logic
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

# Import cairosvg conditionally
CAIRO_AVAILABLE = False
try:
    import cairosvg
    CAIRO_AVAILABLE = True
except (ImportError, OSError):
    pass

class SVGArtGrid:
    """A ComfyUI node that generates Kandinsky-style artistic SVG grids"""
    
    BLOCK_STYLES = [
        'circle', 'opposite_circles', 'cross', 'half_square', 
        'diagonal_square', 'quarter_circle', 'dots', 'letter_block'
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 800, "min": 100, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 800, "min": 100, "max": 4096, "step": 1}),
                "rows": ("INT", {"default": 6, "min": 1, "max": 20, "step": 1}),
                "cols": ("INT", {"default": 6, "min": 1, "max": 20, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
                "palette_index": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1,
                                        "tooltip": "-1 for random palette"}),
                "big_block": ("BOOLEAN", {"default": True}),
                "big_block_size": ("INT", {"default": 2, "min": 2, "max": 3, "step": 1}),
                "output_svg": ("BOOLEAN", {"default": False, "tooltip": "Also output SVG string"}),
            },
            "optional": {
                "palette_url": ("STRING", {"default": "https://unpkg.com/nice-color-palettes@3.0.0/100.json",
                                         "multiline": False}),
                "block_styles": ("STRING", {"default": "all", "multiline": False,
                                          "tooltip": "Comma-separated list or 'all'"}),
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

    def _generate_little_block(self, dwg, i, j, square_size, color_palette, block_styles):
        """Generate a single block in the grid."""
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
        
        available_styles = [style for style in block_styles if style in style_map]
        if not available_styles:
            available_styles = self.BLOCK_STYLES
        
        style_func = style_map[random.choice(available_styles)]
        x_pos = i * square_size
        y_pos = j * square_size
        
        style_func(dwg, x_pos, y_pos, square_size, colors["foreground"], colors["background"])

    def _generate_grid(self, dwg, num_rows, num_cols, square_size, color_palette, block_styles):
        """Generate the grid of blocks."""
        for i in range(num_rows):
            for j in range(num_cols):
                self._generate_little_block(dwg, i, j, square_size, color_palette, block_styles)

    def _generate_big_block(self, dwg, num_rows, num_cols, square_size, color_palette, block_styles, multiplier):
        """Generate a big block."""
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
        
        available_styles = [style for style in block_styles if style in style_map and style != 'dots']
        if not available_styles:
            available_styles = [s for s in self.BLOCK_STYLES if s != 'dots']
        
        x_pos = random.randint(0, num_rows - multiplier) * square_size
        y_pos = random.randint(0, num_cols - multiplier) * square_size
        big_square_size = multiplier * square_size
        
        style_func = style_map[random.choice(available_styles)]
        style_func(dwg, x_pos, y_pos, big_square_size, colors["foreground"], colors["background"])

    def _svg_to_png(self, svg_string, width, height):
        """Convert SVG string to PNG image using cairosvg."""
        if CAIRO_AVAILABLE:
            try:
                png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'),
                                        output_width=width, output_height=height)
                image = Image.open(io.BytesIO(png_data))
                return image
            except Exception as e:
                print(f"SVGArtGrid: cairosvg conversion failed: {e}")
        
        # Fallback to simple PIL representation
        print("SVGArtGrid: Using PIL fallback for SVG rendering (cairo not available)")
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

    def generate_art_grid(self, width, height, rows, cols, seed, palette_index, 
                         big_block, big_block_size, output_svg,
                         palette_url="https://unpkg.com/nice-color-palettes@3.0.0/100.json",
                         block_styles="all"):
        """Generate the SVG art grid."""
        # Set random seed
        seed = seed % (2**32)
        random.seed(seed)
        
        # Calculate square size to fit the canvas
        square_size = min(width // rows, height // cols)
        svg_width = rows * square_size
        svg_height = cols * square_size
        
        # Load color palettes
        palettes = self._load_color_palettes(palette_url)
        if palette_index == -1:
            palette_idx = random.randint(0, len(palettes) - 1)
        else:
            palette_idx = min(palette_index, len(palettes) - 1)
        color_palette = palettes[palette_idx]
        
        # Parse block styles
        if block_styles.lower() == "all":
            selected_styles = self.BLOCK_STYLES
        else:
            selected_styles = [s.strip() for s in block_styles.split(',')]
        
        # Create SVG
        dwg = svgwrite.Drawing(size=(f"{svg_width}px", f"{svg_height}px"), profile='full')
        dwg.defs.add(dwg.style("svg * { shape-rendering: crispEdges; }"))
        
        # Add background gradient
        bg_colors = self._create_background_colors(color_palette)
        gradient = dwg.defs.add(dwg.radialGradient(id="background_gradient"))
        gradient.add_stop_color(0, bg_colors["bg_inner"])
        gradient.add_stop_color(1, bg_colors["bg_outer"])
        dwg.add(dwg.rect((0, 0), (svg_width, svg_height), fill="url(#background_gradient)"))
        
        # Generate grid
        self._generate_grid(dwg, rows, cols, square_size, color_palette, selected_styles)
        
        # Add big block if enabled
        if big_block:
            self._generate_big_block(dwg, rows, cols, square_size, color_palette, 
                                   selected_styles, big_block_size)
        
        # Get SVG string
        svg_string = dwg.tostring()
        
        # Convert to PNG for ComfyUI
        image = self._svg_to_png(svg_string, width, height)
        
        # Convert PIL image to tensor format for ComfyUI
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)
        
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        return (image_tensor, svg_string if output_svg else "")

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SVGArtGrid": SVGArtGrid
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVGArtGrid": "SVG Art Grid Generator"
}
