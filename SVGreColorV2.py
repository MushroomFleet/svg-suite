import os
import re
import json
import math
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Register the SVG namespace to avoid problems with parsing
ET.register_namespace("", "http://www.w3.org/2000/svg")

# Get the path to the themes directory
THEMES_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "themes"

def get_theme_files() -> List[str]:
    """
    Scan the themes directory for JSON files and return a list of theme names.
    """
    theme_files = []
    if THEMES_DIR.exists() and THEMES_DIR.is_dir():
        for file in THEMES_DIR.glob("*.json"):
            theme_files.append(file.stem)
    return sorted(theme_files) if theme_files else ["pastel_theme"]  # Default to pastel_theme if no themes found

def load_theme(theme_name: str) -> Dict:
    """
    Load a color theme from a JSON file.
    """
    theme_path = THEMES_DIR / f"{theme_name}.json"
    
    if not theme_path.exists():
        raise FileNotFoundError(f"Theme {theme_name} not found")
    
    with open(theme_path, 'r') as f:
        return json.load(f)

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:  # Handle shorthand hex notation
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to LAB color space for better color comparison."""
    # First convert RGB to XYZ
    r, g, b = [x/255.0 for x in rgb]
    
    # Convert sRGB to linear RGB
    r = r/12.92 if r <= 0.04045 else ((r+0.055)/1.055)**2.4
    g = g/12.92 if g <= 0.04045 else ((g+0.055)/1.055)**2.4
    b = b/12.92 if b <= 0.04045 else ((b+0.055)/1.055)**2.4
    
    # Convert to XYZ
    x = r*0.4124 + g*0.3576 + b*0.1805
    y = r*0.2126 + g*0.7152 + b*0.0722
    z = r*0.0193 + g*0.1192 + b*0.9505
    
    # Convert XYZ to Lab
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883
    
    x = x**(1/3) if x > 0.008856 else 7.787*x + 16/116
    y = y**(1/3) if y > 0.008856 else 7.787*y + 16/116
    z = z**(1/3) if z > 0.008856 else 7.787*z + 16/116
    
    L = 116*y - 16
    a = 500*(x - y)
    b = 200*(y - z)
    
    return (L, a, b)

def color_distance(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    """Calculate the distance between two colors in LAB space (Delta E)."""
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(lab1, lab2)))

def find_closest_color(hex_color: str, palette: List[str]) -> str:
    """Find the closest color in palette to hex_color."""
    # If the color is already in the palette, return it
    if hex_color.lower() in [c.lower() for c in palette]:
        return hex_color
    
    # Special case for 'none' or 'transparent'
    if hex_color.lower() in ['none', 'transparent']:
        return hex_color
        
    # Convert target color to LAB
    try:
        rgb = hex_to_rgb(hex_color)
        lab = rgb_to_lab(rgb)
    except ValueError:
        # Return original color if conversion fails
        return hex_color
    
    # Find the closest color
    closest_color = None
    min_distance = float('inf')
    
    for color in palette:
        try:
            color_rgb = hex_to_rgb(color)
            color_lab = rgb_to_lab(color_rgb)
            
            distance = color_distance(lab, color_lab)
            
            if distance < min_distance:
                min_distance = distance
                closest_color = color
        except ValueError:
            # Skip colors that can't be converted
            continue
    
    return closest_color if closest_color else hex_color

def extract_colors_from_svg(svg_string: str) -> List[str]:
    """
    Extract all unique fill colors from an SVG string.
    Returns a list of hex color codes.
    """
    try:
        # Parse the SVG as XML
        root = ET.fromstring(svg_string)
        
        # Find all elements with style attribute or fill attribute
        colors = set()
        
        # Method 1: Extract from style attribute
        style_pattern = re.compile(r'fill:\s*([#][0-9a-fA-F]{6}|[#][0-9a-fA-F]{3})')
        
        # Process all elements
        for element in root.iter():
            # Check style attribute
            style = element.get('style')
            if style:
                matches = style_pattern.findall(style)
                for color in matches:
                    colors.add(color)
            
            # Check fill attribute
            fill = element.get('fill')
            if fill and (fill.startswith('#') or fill.lower() in ['none', 'transparent']):
                colors.add(fill)
        
        # Return sorted list of colors
        return sorted(list(colors))
    
    except Exception as e:
        print(f"Error extracting colors from SVG: {e}")
        return []

def replace_colors_in_svg(svg_string: str, color_map: Dict[str, str]) -> str:
    """
    Replace colors in an SVG string according to the provided color_map.
    Returns the modified SVG string.
    """
    try:
        # Parse the SVG as XML
        root = ET.fromstring(svg_string)
        
        # Method 1: Replace in style attribute using regex
        style_pattern = re.compile(r'fill:\s*([#][0-9a-fA-F]{6}|[#][0-9a-fA-F]{3})')
        
        # Process all elements
        for element in root.iter():
            # Check and replace in style attribute
            style = element.get('style')
            if style:
                # Find all color matches in the style
                def replace_match(match):
                    color = match.group(1)
                    if color in color_map:
                        return f"fill: {color_map[color]}"
                    return match.group(0)
                
                new_style = style_pattern.sub(replace_match, style)
                if new_style != style:
                    element.set('style', new_style)
            
            # Check and replace in fill attribute
            fill = element.get('fill')
            if fill and fill in color_map:
                element.set('fill', color_map[fill])
        
        # Convert back to string
        return ET.tostring(root, encoding='unicode')
    
    except Exception as e:
        print(f"Error replacing colors in SVG: {e}")
        return svg_string  # Return original if there's an error

def apply_theme_to_svg(svg_string: str, theme_name: str) -> str:
    """
    Apply a color theme to an SVG string.
    """
    try:
        # Load the theme
        theme = load_theme(theme_name)
        palette = theme["colors"]
        
        # Extract original colors
        original_colors = extract_colors_from_svg(svg_string)
        
        # Create color mapping
        color_map = {}
        for color in original_colors:
            closest = find_closest_color(color, palette)
            if closest:
                color_map[color] = closest
        
        # Apply the color mapping
        return replace_colors_in_svg(svg_string, color_map)
        
    except Exception as e:
        print(f"Error applying theme to SVG: {e}")
        return svg_string  # Return original if there's an error


# Node for SVG theme-based color transformation
class SVGThemeColorizer:
    """
    Apply a color theme to an SVG from predefined JSON theme files.
    Themes are loaded from the 'themes' directory.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "theme": (get_theme_files(), {"default": "pastel_theme"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "apply_theme"
    CATEGORY = "💎TOSVG/Advanced"

    def apply_theme(self, svg_string, theme):
        """Apply a color theme to an SVG string."""
        try:
            # Apply the theme
            themed_svg = apply_theme_to_svg(svg_string, theme)
            return (themed_svg,)
        except Exception as e:
            print(f"Error in SVG theme coloring: {e}")
            return (svg_string,)  # Return original in case of error


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "SVGThemeColorizer": SVGThemeColorizer
}

# NODE DISPLAY NAMES
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVGThemeColorizer": "SVG Theme Colorizer"
}
