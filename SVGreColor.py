import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Union

# Register the SVG namespace to avoid problems with parsing
ET.register_namespace("", "http://www.w3.org/2000/svg")

# Helper functions for color extraction and manipulation
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


# Node for SVG color extraction
class SVGColorExtractor:
    """
    Extract all fill colors used in an SVG.
    Useful for identifying colors before replacement or for analytics.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "extract_colors"
    CATEGORY = "💎TOSVG/Advanced"

    def extract_colors(self, svg_string):
        """Extract all unique fill colors from an SVG string"""
        colors = extract_colors_from_svg(svg_string)
        return (colors,)


# Node for SVG color replacement
class SVGColorReplacer:
    """
    Replace specific colors in an SVG with new colors.
    Supports multiple color pairs for batch processing.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "color_mapping": ("STRING", {"multiline": True, "default": "#FF0000:#0000FF"}),
            },
            "optional": {
                "apply_to_fill_attribute": ("BOOLEAN", {"default": True}),
                "apply_to_style_attribute": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "replace_colors"
    CATEGORY = "💎TOSVG/Advanced"

    def replace_colors(self, svg_string, color_mapping, apply_to_fill_attribute=True, apply_to_style_attribute=True):
        """Replace colors in an SVG string according to the provided color_mapping"""
        
        # Parse color mapping string
        # Format: "#FF0000:#0000FF,#00FF00:#00FFFF"
        color_map = {}
        
        try:
            # Split by commas for multiple replacements
            pairs = color_mapping.split(',')
            
            for pair in pairs:
                pair = pair.strip()
                if ':' in pair:
                    source, target = pair.split(':')
                    source = source.strip()
                    target = target.strip()
                    color_map[source] = target
        
            # Perform the replacement
            modified_svg = replace_colors_in_svg(svg_string, color_map)
            
            return (modified_svg,)
        
        except Exception as e:
            print(f"Error processing color mapping: {e}")
            return (svg_string,)  # Return original if there's an error


# Node for SVG batch color operations
class SVGBatchColorReplacer:
    """
    Apply a collection of color transformation presets to an SVG.
    Creates multiple variations based on the selected presets.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "transformation": (["invert", "grayscale", "sepia", "custom"], {"default": "invert"}),
            },
            "optional": {
                "custom_mapping": ("STRING", {"multiline": True, "default": "#FF0000:#0000FF,#00FF00:#FFFF00"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "batch_replace_colors"
    CATEGORY = "💎TOSVG/Advanced"

    def batch_replace_colors(self, svg_string, transformation, custom_mapping=""):
        """Apply a preset color transformation to an SVG"""
        results = []
        
        try:
            # Extract original colors 
            original_colors = extract_colors_from_svg(svg_string)
            
            # Define preset transformations
            if transformation == "invert":
                # Invert each color (subtract from FFFFFF)
                color_map = {}
                for color in original_colors:
                    if color.startswith('#') and len(color) == 7:  # Only process standard hex colors
                        r = 255 - int(color[1:3], 16)
                        g = 255 - int(color[3:5], 16)
                        b = 255 - int(color[5:7], 16)
                        color_map[color] = f"#{r:02X}{g:02X}{b:02X}"
                
                inverted_svg = replace_colors_in_svg(svg_string, color_map)
                results.append(inverted_svg)
            
            elif transformation == "grayscale":
                # Convert each color to grayscale
                color_map = {}
                for color in original_colors:
                    if color.startswith('#') and len(color) == 7:  # Only process standard hex colors
                        r = int(color[1:3], 16)
                        g = int(color[3:5], 16)
                        b = int(color[5:7], 16)
                        # Calculate grayscale (luminance formula)
                        gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                        color_map[color] = f"#{gray:02X}{gray:02X}{gray:02X}"
                
                grayscale_svg = replace_colors_in_svg(svg_string, color_map)
                results.append(grayscale_svg)
            
            elif transformation == "sepia":
                # Apply sepia tone to each color
                color_map = {}
                for color in original_colors:
                    if color.startswith('#') and len(color) == 7:  # Only process standard hex colors
                        r = int(color[1:3], 16)
                        g = int(color[3:5], 16)
                        b = int(color[5:7], 16)
                        # Calculate sepia
                        r_sepia = min(255, int(0.393 * r + 0.769 * g + 0.189 * b))
                        g_sepia = min(255, int(0.349 * r + 0.686 * g + 0.168 * b))
                        b_sepia = min(255, int(0.272 * r + 0.534 * g + 0.131 * b))
                        color_map[color] = f"#{r_sepia:02X}{g_sepia:02X}{b_sepia:02X}"
                
                sepia_svg = replace_colors_in_svg(svg_string, color_map)
                results.append(sepia_svg)
            
            elif transformation == "custom":
                # Parse custom mapping
                color_map = {}
                pairs = custom_mapping.split(',')
                
                for pair in pairs:
                    pair = pair.strip()
                    if ':' in pair:
                        source, target = pair.split(':')
                        source = source.strip()
                        target = target.strip()
                        color_map[source] = target
                
                custom_svg = replace_colors_in_svg(svg_string, color_map)
                results.append(custom_svg)
        
        except Exception as e:
            print(f"Error in batch color processing: {e}")
            results.append(svg_string)  # Include original in case of error
        
        # Return original SVG as well if we haven't added any results
        if not results:
            results.append(svg_string)
            
        return (results,)


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "SVGColorExtractor": SVGColorExtractor,
    "SVGColorReplacer": SVGColorReplacer,
    "SVGBatchColorReplacer": SVGBatchColorReplacer
}

# NODE DISPLAY NAMES 
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVGColorExtractor": "SVG Color Extractor",
    "SVGColorReplacer": "SVG Color Replacer",
    "SVGBatchColorReplacer": "SVG Batch Color Effects"
}
