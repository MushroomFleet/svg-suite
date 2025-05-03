import os
import re
import time
import random
import colorsys
from typing import List, Tuple, Dict, Any, Optional, Union

class SVGStringReplace:
    """
    Basic string replacement operations for SVG strings.
    Can replace text patterns, colors, or attributes throughout an SVG.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "find_pattern": ("STRING", {"default": "", "multiline": False}),
                "replace_with": ("STRING", {"default": "", "multiline": False}),
                "case_sensitive": ("BOOLEAN", {"default": True}),
                "use_regex": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "limit_replacements": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, 
                                              "display": "slider"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "replace_in_svg"
    CATEGORY = "💎TOSVG/StringEdit"

    def replace_in_svg(self, svg_string, find_pattern, replace_with, 
                      case_sensitive=True, use_regex=False, 
                      limit_replacements=0):
        """
        Replace occurrences of find_pattern with replace_with in the SVG string.
        
        Args:
            svg_string: The SVG content as a string
            find_pattern: The text to find
            replace_with: The text to replace with
            case_sensitive: Whether the search should be case sensitive
            use_regex: Whether to interpret find_pattern as a regular expression
            limit_replacements: Maximum number of replacements (0 = no limit)
            
        Returns:
            The modified SVG string
        """
        if not find_pattern:
            return (svg_string,)
            
        count = 0 if limit_replacements <= 0 else limit_replacements
        
        if use_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            modified_svg = re.sub(find_pattern, replace_with, svg_string, count=count, flags=flags)
        else:
            # For non-regex, we need to implement case sensitivity and max replacements manually
            if not case_sensitive:
                # For case insensitive, use regex after escaping the pattern
                escaped_pattern = re.escape(find_pattern)
                modified_svg = re.sub(escaped_pattern, replace_with, svg_string, 
                                     count=count, flags=re.IGNORECASE)
            else:
                # For case sensitive, use normal string replace with count
                if count > 0:
                    # Count the occurrences to respect the limit
                    parts = svg_string.split(find_pattern)
                    modified_svg = find_pattern.join(parts[:count+1])
                    if len(parts) > count+1:
                        modified_svg += find_pattern.join(parts[count+1:])
                else:
                    modified_svg = svg_string.replace(find_pattern, replace_with)
        
        return (modified_svg,)

class SVGColorManipulation:
    """
    Manipulates colors in SVG string.
    Can replace specific colors, adjust brightness/saturation,
    and perform other color transformations.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "operation": (["replace_color", "adjust_brightness", "adjust_saturation", 
                              "grayscale", "invert_colors", "shift_hue"], {"default": "replace_color"}),
                "target_color": ("STRING", {"default": "#000000", "multiline": False}),
            },
            "optional": {
                "replacement_color": ("STRING", {"default": "#FF0000", "multiline": False}), 
                "brightness_factor": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 5.0, "step": 0.1}),
                "saturation_factor": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 5.0, "step": 0.1}),
                "hue_shift_degrees": ("INT", {"default": 180, "min": 0, "max": 360, "step": 1}),
                "affect_fills": ("BOOLEAN", {"default": True}),
                "affect_strokes": ("BOOLEAN", {"default": True}),
                "color_tolerance": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "manipulate_colors"
    CATEGORY = "💎TOSVG/StringEdit"

    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _rgb_to_hex(self, rgb):
        """Convert RGB tuple to hex color."""
        return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
    
    def _rgb_to_hsl(self, rgb):
        """Convert RGB tuple to HSL tuple."""
        r, g, b = [x/255.0 for x in rgb]
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return (h, s, l)
    
    def _hsl_to_rgb(self, hsl):
        """Convert HSL tuple to RGB tuple."""
        h, s, l = hsl
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return (int(r*255), int(g*255), int(b*255))
    
    def _is_similar_color(self, color1, color2, tolerance):
        """Check if two colors are similar within tolerance."""
        if color1.startswith('#') and color2.startswith('#'):
            rgb1 = self._hex_to_rgb(color1)
            rgb2 = self._hex_to_rgb(color2)
            
            diff = sum(abs(a - b) for a, b in zip(rgb1, rgb2))
            return diff <= tolerance*3  # Multiply by 3 since we have 3 channels
        return color1 == color2
    
    def _find_all_colors(self, svg_string):
        """Find all color attributes in the SVG"""
        # Pattern for fill and stroke colors in hex
        hex_pattern = r'(fill|stroke)="(#[0-9a-fA-F]{3,6})"'
        # Pattern for fill and stroke colors in rgb
        rgb_pattern = r'(fill|stroke)="rgb\((\d+),\s*(\d+),\s*(\d+)\)"'
        
        hex_colors = re.findall(hex_pattern, svg_string)
        rgb_colors = re.findall(rgb_pattern, svg_string)
        
        return hex_colors, rgb_colors

    def manipulate_colors(self, svg_string, operation, target_color,
                         replacement_color="#FF0000", brightness_factor=1.2,
                         saturation_factor=1.2, hue_shift_degrees=180,
                         affect_fills=True, affect_strokes=True,
                         color_tolerance=0):
        """
        Manipulate colors in the SVG based on the selected operation.
        """
        modified_svg = svg_string
        
        # Define patterns for color attributes
        fill_pattern = r'fill="([^"]+)"'
        stroke_pattern = r'stroke="([^"]+)"'
        
        # For replace_color operation
        if operation == "replace_color":
            def replace_match(match):
                attr_type, color = match.groups()
                # Skip if we're not affecting this type
                if (attr_type == "fill" and not affect_fills) or (attr_type == "stroke" and not affect_strokes):
                    return match.group(0)
                
                # Replace if it matches the target color within tolerance
                if self._is_similar_color(color, target_color, color_tolerance):
                    return f'{attr_type}="{replacement_color}"'
                return match.group(0)
            
            # Apply to hex colors
            modified_svg = re.sub(r'(fill|stroke)="(#[0-9a-fA-F]{3,6})"', 
                                replace_match, modified_svg)
        
        # For brightness adjustment
        elif operation == "adjust_brightness":
            def adjust_brightness(match):
                attr_type, color = match.groups()
                # Skip if we're not affecting this type
                if (attr_type == "fill" and not affect_fills) or (attr_type == "stroke" and not affect_strokes):
                    return match.group(0)
                
                # Adjust brightness for hex colors
                if color.startswith('#'):
                    rgb = self._hex_to_rgb(color)
                    adjusted_rgb = tuple(min(255, int(c * brightness_factor)) for c in rgb)
                    adjusted_color = self._rgb_to_hex(adjusted_rgb)
                    return f'{attr_type}="{adjusted_color}"'
                return match.group(0)
            
            # Apply to hex colors
            modified_svg = re.sub(r'(fill|stroke)="(#[0-9a-fA-F]{3,6})"', 
                                adjust_brightness, modified_svg)
            
        # For saturation adjustment
        elif operation == "adjust_saturation":
            def adjust_saturation(match):
                attr_type, color = match.groups()
                # Skip if we're not affecting this type
                if (attr_type == "fill" and not affect_fills) or (attr_type == "stroke" and not affect_strokes):
                    return match.group(0)
                
                # Adjust saturation for hex colors
                if color.startswith('#'):
                    rgb = self._hex_to_rgb(color)
                    hsl = self._rgb_to_hsl(rgb)
                    adjusted_hsl = (hsl[0], min(1.0, hsl[1] * saturation_factor), hsl[2])
                    adjusted_rgb = self._hsl_to_rgb(adjusted_hsl)
                    adjusted_color = self._rgb_to_hex(adjusted_rgb)
                    return f'{attr_type}="{adjusted_color}"'
                return match.group(0)
            
            # Apply to hex colors
            modified_svg = re.sub(r'(fill|stroke)="(#[0-9a-fA-F]{3,6})"', 
                                adjust_saturation, modified_svg)
            
        # For grayscale conversion
        elif operation == "grayscale":
            def convert_to_grayscale(match):
                attr_type, color = match.groups()
                # Skip if we're not affecting this type
                if (attr_type == "fill" and not affect_fills) or (attr_type == "stroke" and not affect_strokes):
                    return match.group(0)
                
                # Convert to grayscale for hex colors
                if color.startswith('#'):
                    rgb = self._hex_to_rgb(color)
                    # Standard grayscale conversion formula
                    gray = int(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
                    gray_hex = f"#{gray:02x}{gray:02x}{gray:02x}"
                    return f'{attr_type}="{gray_hex}"'
                return match.group(0)
            
            # Apply to hex colors
            modified_svg = re.sub(r'(fill|stroke)="(#[0-9a-fA-F]{3,6})"', 
                                convert_to_grayscale, modified_svg)
            
        # For color inversion
        elif operation == "invert_colors":
            def invert_color(match):
                attr_type, color = match.groups()
                # Skip if we're not affecting this type
                if (attr_type == "fill" and not affect_fills) or (attr_type == "stroke" and not affect_strokes):
                    return match.group(0)
                
                # Invert color for hex colors
                if color.startswith('#'):
                    rgb = self._hex_to_rgb(color)
                    inverted_rgb = tuple(255 - c for c in rgb)
                    inverted_color = self._rgb_to_hex(inverted_rgb)
                    return f'{attr_type}="{inverted_color}"'
                return match.group(0)
            
            # Apply to hex colors
            modified_svg = re.sub(r'(fill|stroke)="(#[0-9a-fA-F]{3,6})"', 
                                invert_color, modified_svg)
            
        # For hue shifting
        elif operation == "shift_hue":
            def shift_hue(match):
                attr_type, color = match.groups()
                # Skip if we're not affecting this type
                if (attr_type == "fill" and not affect_fills) or (attr_type == "stroke" and not affect_strokes):
                    return match.group(0)
                
                # Shift hue for hex colors
                if color.startswith('#'):
                    rgb = self._hex_to_rgb(color)
                    hsl = self._rgb_to_hsl(rgb)
                    # Shift hue by the specified degrees (0-360 -> 0-1 range)
                    new_hue = (hsl[0] + (hue_shift_degrees / 360.0)) % 1.0
                    shifted_hsl = (new_hue, hsl[1], hsl[2])
                    shifted_rgb = self._hsl_to_rgb(shifted_hsl)
                    shifted_color = self._rgb_to_hex(shifted_rgb)
                    return f'{attr_type}="{shifted_color}"'
                return match.group(0)
            
            # Apply to hex colors
            modified_svg = re.sub(r'(fill|stroke)="(#[0-9a-fA-F]{3,6})"', 
                                shift_hue, modified_svg)
        
        return (modified_svg,)

class SVGAttributeManipulation:
    """
    Manipulates attributes in SVG elements.
    Can add, modify, or remove attributes from specific elements.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "operation": (["add_attribute", "modify_attribute", "remove_attribute", 
                             "add_class", "remove_class", "add_style", "remove_style"],
                             {"default": "add_attribute"}),
                "element_selector": ("STRING", {"default": "path", "multiline": False,
                                             "tooltip": "CSS selector or element name (e.g., 'path', 'circle', 'rect')"}),
            },
            "optional": {
                "attribute_name": ("STRING", {"default": "opacity", "multiline": False}),
                "attribute_value": ("STRING", {"default": "0.5", "multiline": False}),
                "class_name": ("STRING", {"default": "highlight", "multiline": False}),
                "style_property": ("STRING", {"default": "opacity", "multiline": False}),
                "style_value": ("STRING", {"default": "0.5", "multiline": False}),
                "match_index": ("INT", {"default": -1, "min": -1, "max": 1000, 
                                      "tooltip": "-1 to affect all matching elements"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "manipulate_attributes"
    CATEGORY = "💎TOSVG/StringEdit"

    def _add_or_modify_attribute(self, element_open_tag, attr_name, attr_value):
        """Add or modify an attribute in an element's opening tag."""
        # Check if the attribute already exists
        attr_pattern = fr'{attr_name}\s*=\s*"[^"]*"'
        if re.search(attr_pattern, element_open_tag):
            # Modify existing attribute
            return re.sub(attr_pattern, f'{attr_name}="{attr_value}"', element_open_tag)
        else:
            # Add new attribute before the closing '>' of the tag
            return element_open_tag.replace('>', f' {attr_name}="{attr_value}">', 1)
    
    def _remove_attribute(self, element_open_tag, attr_name):
        """Remove an attribute from an element's opening tag."""
        # Match the attribute and any whitespace before it
        attr_pattern = fr'\s*{attr_name}\s*=\s*"[^"]*"'
        return re.sub(attr_pattern, '', element_open_tag)
    
    def _modify_class_attribute(self, element_open_tag, class_name, add=True):
        """Add or remove a class from the class attribute."""
        # Check if class attribute exists
        class_match = re.search(r'class\s*=\s*"([^"]*)"', element_open_tag)
        
        if class_match:
            classes = class_match.group(1).split()
            if add and class_name not in classes:
                classes.append(class_name)
            elif not add and class_name in classes:
                classes.remove(class_name)
            
            # Replace the class attribute with the new value
            new_class_attr = f'class="{" ".join(classes)}"'
            return re.sub(r'class\s*=\s*"[^"]*"', new_class_attr, element_open_tag)
        elif add:
            # Add new class attribute if it doesn't exist and we're adding
            return element_open_tag.replace('>', f' class="{class_name}">', 1)
        return element_open_tag
    
    def _modify_style_attribute(self, element_open_tag, style_prop, style_value, add=True):
        """Add or remove a style property from the style attribute."""
        # Check if style attribute exists
        style_match = re.search(r'style\s*=\s*"([^"]*)"', element_open_tag)
        
        if style_match:
            style_text = style_match.group(1)
            # Parse style properties
            style_props = {}
            for item in style_text.split(';'):
                if ':' in item:
                    prop, val = item.split(':', 1)
                    style_props[prop.strip()] = val.strip()
            
            # Modify style properties
            if add:
                style_props[style_prop] = style_value
            elif style_prop in style_props and not add:
                del style_props[style_prop]
            
            # Format back to style string
            style_str = '; '.join([f"{p}: {v}" for p, v in style_props.items()])
            if style_str:
                style_str += ';'
            
            # Replace style attribute
            new_style_attr = f'style="{style_str}"'
            return re.sub(r'style\s*=\s*"[^"]*"', new_style_attr, element_open_tag)
        elif add:
            # Add new style attribute if it doesn't exist and we're adding
            return element_open_tag.replace('>', f' style="{style_prop}: {style_value};">', 1)
        return element_open_tag

    def manipulate_attributes(self, svg_string, operation, element_selector,
                             attribute_name="opacity", attribute_value="0.5",
                             class_name="highlight", style_property="opacity",
                             style_value="0.5", match_index=-1):
        """
        Manipulate attributes of SVG elements based on the selected operation.
        """
        # Find all occurrences of the specified element
        if element_selector.endswith('/'):
            element_selector = element_selector[:-1]
        
        # Simple selector handling for basic tag names
        element_pattern = fr'<{element_selector}[^>]*>'
        element_matches = list(re.finditer(element_pattern, svg_string))
        
        if not element_matches:
            # No matching elements found
            return (svg_string,)
        
        modified_svg = svg_string
        for i, match in enumerate(element_matches):
            # Skip if not the specified index and index is not -1 (all)
            if match_index != -1 and i != match_index:
                continue
                
            original_tag = match.group(0)
            modified_tag = original_tag
            
            # Apply the requested operation
            if operation == "add_attribute" or operation == "modify_attribute":
                modified_tag = self._add_or_modify_attribute(original_tag, attribute_name, attribute_value)
            
            elif operation == "remove_attribute":
                modified_tag = self._remove_attribute(original_tag, attribute_name)
            
            elif operation == "add_class":
                modified_tag = self._modify_class_attribute(original_tag, class_name, add=True)
            
            elif operation == "remove_class":
                modified_tag = self._modify_class_attribute(original_tag, class_name, add=False)
            
            elif operation == "add_style":
                modified_tag = self._modify_style_attribute(original_tag, style_property, style_value, add=True)
            
            elif operation == "remove_style":
                modified_tag = self._modify_style_attribute(original_tag, style_property, style_value, add=False)
            
            # Replace the original tag with the modified one
            modified_svg = modified_svg.replace(original_tag, modified_tag, 1)
        
        return (modified_svg,)

class SVGPathManipulation:
    """
    Specialized node for manipulating SVG path data.
    Provides options to scale, rotate, translate and modify SVG paths.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "operation": (["scale", "rotate", "translate", "flip_horizontal", "flip_vertical", 
                               "simplify", "reverse"], {"default": "scale"}),
            },
            "optional": {
                "scale_x": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "scale_y": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "rotation_degrees": ("FLOAT", {"default": 45.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "translate_x": ("INT", {"default": 10, "min": -1000, "max": 1000, "step": 1}),
                "translate_y": ("INT", {"default": 10, "min": -1000, "max": 1000, "step": 1}),
                "center_x": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "center_y": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "path_selector": ("STRING", {"default": "", "multiline": False,
                                          "tooltip": "Leave empty for all paths or provide ID/class selector"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "manipulate_paths"
    CATEGORY = "💎TOSVG/StringEdit"

    def _get_svg_paths(self, svg_string, path_selector=""):
        """Extract all path elements and their data attributes."""
        # Basic path extraction
        path_pattern = r'<path([^>]*)d="([^"]*)"([^>]*)>'
        paths = re.findall(path_pattern, svg_string)
        
        # If selector is empty, return all paths
        if not path_selector:
            return paths
        
        # Filter paths based on selector (simple ID or class filtering)
        filtered_paths = []
        for path in paths:
            attrs_before, path_data, attrs_after = path
            attrs = attrs_before + attrs_after
            
            # Check for ID selector
            if path_selector.startswith('#'):
                id_value = path_selector[1:]
                id_pattern = fr'id="{id_value}"'
                if re.search(id_pattern, attrs):
                    filtered_paths.append(path)
            
            # Check for class selector
            elif path_selector.startswith('.'):
                class_value = path_selector[1:]
                class_pattern = fr'class="[^"]*{class_value}[^"]*"'
                if re.search(class_pattern, attrs):
                    filtered_paths.append(path)
        
        return filtered_paths if filtered_paths else paths
    
    def _apply_transform_to_path(self, path_data, operation, **kwargs):
        """Apply transformation to a path's data attribute."""
        # This is a simplistic implementation that works for basic cases
        # A full SVG path parser would be more robust but complex
        
        if operation == "scale":
            scale_x = kwargs.get('scale_x', 1.0)
            scale_y = kwargs.get('scale_y', 1.0)
            
            # Scale coordinates in path data
            # This is a naive approach that works for simple cases
            path_data = re.sub(r'(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)', 
                              lambda m: f"{float(m.group(1)) * scale_x:.1f},{float(m.group(2)) * scale_y:.1f}", 
                              path_data)
            
        elif operation == "rotate":
            degrees = kwargs.get('rotation_degrees', 0)
            center_x = kwargs.get('center_x', 0)
            center_y = kwargs.get('center_y', 0)
            
            # Convert to radians
            radians = degrees * (3.14159 / 180.0)
            cos_theta = round(math.cos(radians), 6)
            sin_theta = round(math.sin(radians), 6)
            
            # For simple rotation, we can add a transform attribute
            # but for actual path data, we'd need to rotate each point
            # This is extremely simplified
            def rotate_point(match):
                x = float(match.group(1))
                y = float(match.group(2))
                
                # Translate to origin
                x -= center_x
                y -= center_y
                
                # Rotate
                new_x = x * cos_theta - y * sin_theta
                new_y = x * sin_theta + y * cos_theta
                
                # Translate back
                new_x += center_x
                new_y += center_y
                
                return f"{new_x:.1f},{new_y:.1f}"
            
            path_data = re.sub(r'(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)', rotate_point, path_data)
            
        elif operation == "translate":
            tx = kwargs.get('translate_x', 0)
            ty = kwargs.get('translate_y', 0)
            
            # Translate coordinates in path data
            def translate_point(match):
                x = float(match.group(1))
                y = float(match.group(2))
                return f"{x + tx:.1f},{y + ty:.1f}"
            
            path_data = re.sub(r'(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)', translate_point, path_data)
            
        elif operation == "flip_horizontal":
            # Flip across y-axis (x becomes -x)
            def flip_h(match):
                x = float(match.group(1))
                y = float(match.group(2))
                return f"{-x:.1f},{y:.1f}"
            
            path_data = re.sub(r'(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)', flip_h, path_data)
            
        elif operation == "flip_vertical":
            # Flip across x-axis (y becomes -y)
            def flip_v(match):
                x = float(match.group(1))
                y = float(match.group(2))
                return f"{x:.1f},{-y:.1f}"
            
            path_data = re.sub(r'(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)', flip_v, path_data)
            
        elif operation == "reverse":
            # Attempt to reverse the path direction
            # This is a simplistic approach
            commands = re.findall(r'([MLHVZCSTQAmlhvzcstqa])([^MLHVZCSTQAmlhvzcstqa]*)', path_data)
            if commands:
                reversed_commands = commands[::-1]
                # First command should be 'M'
                for i, (cmd, params) in enumerate(reversed_commands):
                    if cmd.upper() == 'M' and i > 0:
                        first_cmd = reversed_commands.pop(i)
                        reversed_commands.insert(0, first_cmd)
                        break
                
                path_data = ''.join([cmd + params for cmd, params in reversed_commands])
        
        return path_data

    def manipulate_paths(self, svg_string, operation, 
                        scale_x=1.5, scale_y=1.5,
                        rotation_degrees=45.0, 
                        translate_x=10, translate_y=10,
                        center_x=0, center_y=0,
                        path_selector=""):
        """
        Apply transformations to SVG paths.
        """
        # Get all paths in the SVG
        paths = self._get_svg_paths(svg_string, path_selector)
        
        if not paths:
            return (svg_string,)
        
        modified_svg = svg_string
        
        for attrs_before, path_data, attrs_after in paths:
            original_path = f'<path{attrs_before}d="{path_data}"{attrs_after}>'
            
            # Apply the transformation to the path data
            transformed_path_data = self._apply_transform_to_path(
                path_data, 
                operation, 
                scale_x=scale_x,
                scale_y=scale_y,
                rotation_degrees=rotation_degrees,
                translate_x=translate_x,
                translate_y=translate_y,
                center_x=center_x,
                center_y=center_y
            )
            
            # Create the new path element
            new_path = f'<path{attrs_before}d="{transformed_path_data}"{attrs_after}>'
            
            # Replace in the SVG
            modified_svg = modified_svg.replace(original_path, new_path, 1)
        
        return (modified_svg,)

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "SVGStringReplace": SVGStringReplace,
    "SVGColorManipulation": SVGColorManipulation,
    "SVGAttributeManipulation": SVGAttributeManipulation,
    "SVGPathManipulation": SVGPathManipulation,
}

# NODE DISPLAY NAMES
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVGStringReplace": "SVG String Replace",
    "SVGColorManipulation": "SVG Color Manipulation",
    "SVGAttributeManipulation": "SVG Attribute Manipulation",
    "SVGPathManipulation": "SVG Path Manipulation",
}