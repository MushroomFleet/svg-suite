import os
import json
import re
import shutil
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Union, Tuple

# Define the folder for storing style presets
STYLES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles")

class SVGStyler:
    """
    Node for applying predefined styles from JSON files to SVG strings.
    Styles are loaded from the 'styles' directory and can be selected via dropdown.
    """
    @classmethod
    def INPUT_TYPES(cls):
        # Create styles folder if it doesn't exist
        os.makedirs(STYLES_FOLDER, exist_ok=True)
        
        # Scan for available style files
        style_files = cls._get_available_styles()
        
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "style_preset": (style_files, {"default": style_files[0] if style_files else "none"}),
                "style_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "exclude_selectors": ("STRING", {"default": "", "multiline": True, 
                                              "tooltip": "Comma-separated list of selectors to exclude (e.g., '#bg,circle')"}),
                "apply_only_to": ("STRING", {"default": "", "multiline": True, 
                                          "tooltip": "Comma-separated list of selectors to exclusively apply style to"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "apply_style"
    CATEGORY = "💎TOSVG/Styler"

    @staticmethod
    def _get_available_styles():
        """Scan the styles folder for available JSON style files."""
        if not os.path.exists(STYLES_FOLDER):
            return ["none"]
            
        style_files = []
        for filename in os.listdir(STYLES_FOLDER):
            if filename.endswith('.json'):
                style_files.append(filename)
        
        return style_files if style_files else ["none"]

    def _load_style(self, style_name):
        """Load a style preset from a JSON file."""
        if style_name == "none":
            return None
            
        style_path = os.path.join(STYLES_FOLDER, style_name)
        
        try:
            with open(style_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading style '{style_name}': {str(e)}")
            return None
    
    def _parse_svg(self, svg_string):
        """Parse SVG string to XML."""
        try:
            return ET.fromstring(svg_string)
        except ET.ParseError:
            # Try removing DOCTYPE declaration if present
            svg_string = re.sub(r'<!DOCTYPE[^>]*>', '', svg_string)
            return ET.fromstring(svg_string)
    
    def _find_elements(self, root, selector):
        """Find elements in the SVG that match the selector."""
        if selector == "*":
            # Return all elements except the root svg
            return [elem for elem in root.iter() if elem != root]
            
        if selector.startswith('#'):
            # ID selector
            element_id = selector[1:]
            return [elem for elem in root.iter() if elem.get('id') == element_id]
            
        elif selector.startswith('.'):
            # Class selector
            class_name = selector[1:]
            return [elem for elem in root.iter() 
                   if elem.get('class') and class_name in elem.get('class').split()]
            
        elif ':' in selector:
            # Handle simple pseudo-selectors like nth-child
            base_selector, pseudo = selector.split(':', 1)
            
            if 'nth-child' in pseudo:
                # Extract the pattern (e.g., "3n+1")
                pattern = re.search(r'nth-child\(([^)]+)\)', pseudo)
                if pattern:
                    pattern = pattern.group(1)
                    
                    # Find elements matching the base selector
                    if base_selector:
                        elements = self._find_elements(root, base_selector)
                    else:
                        elements = [elem for elem in root.iter() if elem != root]
                    
                    # Apply the nth-child pattern
                    if pattern == "odd":
                        return elements[::2]
                    elif pattern == "even":
                        return elements[1::2]
                    elif "n+" in pattern:
                        # Handle patterns like "3n+1"
                        parts = pattern.split('n+')
                        if len(parts) == 2:
                            n = int(parts[0]) if parts[0] else 1
                            offset = int(parts[1])
                            return [elem for i, elem in enumerate(elements) 
                                   if i % n == offset - 1]
                    elif "n-" in pattern:
                        # Handle patterns like "3n-1"
                        parts = pattern.split('n-')
                        if len(parts) == 2:
                            n = int(parts[0]) if parts[0] else 1
                            offset = int(parts[1])
                            return [elem for i, elem in enumerate(elements) 
                                   if i % n == n - offset]
                    elif pattern.isdigit():
                        # Handle simple index like "2"
                        index = int(pattern) - 1
                        return [elements[index]] if 0 <= index < len(elements) else []
            
            return []
            
        elif ',' in selector:
            # Group selector (e.g., "path, circle")
            results = []
            for s in selector.split(','):
                results.extend(self._find_elements(root, s.strip()))
            return results
            
        else:
            # Tag selector
            return [elem for elem in root.iter() if elem.tag.endswith(selector)]
    
    def _apply_style_rule(self, elements, rule, style_strength=1.0):
        """Apply a style rule to a list of elements."""
        if not elements:
            return
            
        for elem in elements:
            # Apply attributes
            if 'attributes' in rule:
                for attr, value in rule['attributes'].items():
                    # Skip if attribute is a URL reference and we're using partial style strength
                    is_url_ref = isinstance(value, str) and value.startswith('url(#')
                    
                    if style_strength >= 1.0 or not is_url_ref:
                        # For attributes other than fill/stroke colors, just set them directly
                        if attr not in ['fill', 'stroke'] or not isinstance(value, str) or not value.startswith('#'):
                            elem.set(attr, value)
                        else:
                            # For colors, we can blend them if style_strength < 1
                            if style_strength < 1.0 and attr in elem.attrib and elem.get(attr).startswith('#'):
                                # Get original color
                                orig_color = elem.get(attr)
                                
                                # Convert hex colors to RGB
                                orig_rgb = self._hex_to_rgb(orig_color)
                                new_rgb = self._hex_to_rgb(value)
                                
                                # Blend colors based on style_strength
                                blended_rgb = self._blend_colors(orig_rgb, new_rgb, style_strength)
                                
                                # Convert back to hex and set
                                elem.set(attr, self._rgb_to_hex(blended_rgb))
                            else:
                                # Just set the color directly
                                elem.set(attr, value)
            
            # Apply styles to style attribute
            if 'styles' in rule:
                # Get existing style
                style_attr = elem.get('style', '')
                
                # Parse existing style into a dict
                style_dict = {}
                if style_attr:
                    for style_item in style_attr.split(';'):
                        if ':' in style_item:
                            prop, val = style_item.split(':', 1)
                            style_dict[prop.strip()] = val.strip()
                
                # Add or update style properties
                for prop, value in rule['styles'].items():
                    # For full strength, just set the style
                    if style_strength >= 1.0:
                        style_dict[prop] = value
                    else:
                        # For partial strength, only override if not already set
                        if prop not in style_dict:
                            style_dict[prop] = value
                
                # Convert back to string and set
                new_style = '; '.join([f"{prop}: {val}" for prop, val in style_dict.items()])
                if new_style:
                    elem.set('style', new_style)
    
    def _add_defs(self, root, defs_list):
        """Add definitions (defs) to the SVG."""
        if not defs_list:
            return
            
        # Find or create defs element
        defs_elem = root.find('.//defs')
        if defs_elem is None:
            defs_elem = ET.SubElement(root, 'defs')
        
        # Add each definition
        for def_item in defs_list:
            def_type = def_item.get('type', '')
            def_id = def_item.get('id', '')
            
            if not def_type or not def_id:
                continue
                
            # Create the definition element
            def_elem = ET.SubElement(defs_elem, def_type)
            def_elem.set('id', def_id)
            
            # Set attributes
            if 'attributes' in def_item:
                for attr, value in def_item['attributes'].items():
                    def_elem.set(attr, str(value))
            
            # Add gradient stops
            if 'stops' in def_item and def_type in ['linearGradient', 'radialGradient']:
                for stop in def_item['stops']:
                    stop_elem = ET.SubElement(def_elem, 'stop')
                    stop_elem.set('offset', stop.get('offset', '0%'))
                    stop_elem.set('stop-color', stop.get('color', '#000000'))
                    if 'opacity' in stop:
                        stop_elem.set('stop-opacity', str(stop['opacity']))
            
            # Add child elements (for filters, patterns, etc.)
            if 'children' in def_item:
                self._add_children(def_elem, def_item['children'])
    
    def _add_children(self, parent, children):
        """Recursively add child elements to a parent element."""
        for child in children:
            child_type = child.get('type', '')
            if not child_type:
                continue
                
            child_elem = ET.SubElement(parent, child_type)
            
            # Set attributes
            if 'attributes' in child:
                for attr, value in child['attributes'].items():
                    child_elem.set(attr, str(value))
            
            # Recursively add children
            if 'children' in child:
                self._add_children(child_elem, child['children'])
    
    def _add_background(self, root, background):
        """Add a background rectangle to the SVG."""
        if not background:
            return
            
        # Get SVG dimensions
        width = root.get('width', '100%')
        height = root.get('height', '100%')
        
        # Create background rectangle
        bg_rect = ET.Element('rect')
        bg_rect.set('x', '0')
        bg_rect.set('y', '0')
        bg_rect.set('width', width)
        bg_rect.set('height', height)
        
        # Set fill attributes
        if 'fill' in background:
            bg_rect.set('fill', background['fill'])
        
        if 'pattern' in background:
            bg_rect.set('fill', background['pattern'])
        
        # Insert as first child of SVG
        root.insert(0, bg_rect)
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        
        # Handle 3-digit hex
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
            
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _rgb_to_hex(self, rgb):
        """Convert RGB tuple to hex color."""
        return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
    
    def _blend_colors(self, color1, color2, factor):
        """Blend two RGB colors."""
        return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))

    def apply_style(self, svg_string, style_preset, style_strength=1.0, 
                   exclude_selectors="", apply_only_to=""):
        """
        Apply a style preset to an SVG string.
        
        Args:
            svg_string: The SVG content as a string
            style_preset: Filename of the style preset to apply
            style_strength: How strongly to apply the style (0.0-1.0)
            exclude_selectors: Comma-separated list of selectors to exclude
            apply_only_to: Comma-separated list of selectors to exclusively apply style to
            
        Returns:
            The styled SVG string
        """
        # If style is "none" or not found, return the original
        if style_preset == "none":
            return (svg_string,)
            
        # Load the style preset
        style = self._load_style(style_preset)
        if not style:
            return (svg_string,)
            
        try:
            # Parse the SVG
            root = self._parse_svg(svg_string)
            
            # Parse exclude selectors
            excluded = [s.strip() for s in exclude_selectors.split(',')] if exclude_selectors else []
            
            # Parse apply_only_to selectors
            apply_only = [s.strip() for s in apply_only_to.split(',')] if apply_only_to else []
            
            # Add definitions first
            if 'defs' in style:
                self._add_defs(root, style['defs'])
            
            # Add background if specified
            if 'background' in style:
                self._add_background(root, style['background'])
            
            # Apply style rules
            if 'rules' in style:
                for rule in style['rules']:
                    selector = rule.get('selector')
                    if not selector:
                        continue
                        
                    # Skip if selector is in excluded list
                    if any(s in selector for s in excluded):
                        continue
                        
                    # Skip if apply_only is specified and selector is not in it
                    if apply_only and not any(s in selector for s in apply_only):
                        continue
                        
                    # Find matching elements
                    elements = self._find_elements(root, selector)
                    
                    # Apply the rule
                    self._apply_style_rule(elements, rule, style_strength)
            
            # Convert back to string
            styled_svg = ET.tostring(root, encoding='unicode')
            return (styled_svg,)
            
        except Exception as e:
            print(f"Error applying style: {str(e)}")
            return (svg_string,)

class SVGCreateStyle:
    """
    Node for creating and saving a new style preset from the current SVG.
    Extracts styles from an SVG and saves them as a reusable preset.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "style_name": ("STRING", {"default": "my_custom_style"}),
                "description": ("STRING", {"default": "Custom style created from SVG"}),
                "author": ("STRING", {"default": "ComfyUI User"}),
            },
            "optional": {
                "selectors_to_include": ("STRING", {"default": "path,circle,rect", "multiline": True,
                                                 "tooltip": "Comma-separated list of selectors to include"}),
                "extract_defs": ("BOOLEAN", {"default": True}),
                "extract_colors_only": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "create_style"
    CATEGORY = "💎TOSVG/Styler"
    
    def _parse_svg(self, svg_string):
        """Parse SVG string to XML."""
        try:
            return ET.fromstring(svg_string)
        except ET.ParseError:
            # Try removing DOCTYPE declaration if present
            svg_string = re.sub(r'<!DOCTYPE[^>]*>', '', svg_string)
            return ET.fromstring(svg_string)
    
    def _get_elements(self, root, selectors):
        """Get elements matching the selectors."""
        elements = []
        for selector in selectors:
            if selector == '*':
                # All elements except the root
                elements.extend([elem for elem in root.iter() if elem != root])
            elif selector.startswith('#'):
                # ID selector
                id_value = selector[1:]
                elements.extend([elem for elem in root.iter() if elem.get('id') == id_value])
            elif selector.startswith('.'):
                # Class selector
                class_value = selector[1:]
                elements.extend([elem for elem in root.iter() 
                               if elem.get('class') and class_value in elem.get('class').split()])
            else:
                # Tag selector
                elements.extend([elem for elem in root.iter() if elem.tag.endswith(selector)])
        return elements
    
    def _extract_style_rules(self, elements, extract_colors_only=False):
        """Extract style rules from elements."""
        rules = []
        
        # Group elements by tag
        elements_by_tag = {}
        for elem in elements:
            tag = elem.tag.split('}')[-1]  # Handle namespaces
            if tag not in elements_by_tag:
                elements_by_tag[tag] = []
            elements_by_tag[tag].append(elem)
        
        # Create rules for each tag type
        for tag, tag_elements in elements_by_tag.items():
            rule = {
                "selector": tag,
                "attributes": {}
            }
            
            # Extract common attributes
            attrs_to_extract = ['fill', 'stroke', 'stroke-width', 'opacity']
            if not extract_colors_only:
                attrs_to_extract.extend(['d', 'cx', 'cy', 'r', 'rx', 'ry', 'x', 'y', 'width', 'height',
                                        'transform', 'filter', 'mask', 'clip-path'])
            
            for attr in attrs_to_extract:
                # Check if most elements have this attribute
                attr_values = [elem.get(attr) for elem in tag_elements if attr in elem.attrib]
                if attr_values and len(attr_values) >= len(tag_elements) * 0.5:
                    # Use the most common value
                    most_common = max(set(attr_values), key=attr_values.count)
                    rule["attributes"][attr] = most_common
            
            # Extract style attribute
            if not extract_colors_only:
                style_attr_elements = [elem for elem in tag_elements if 'style' in elem.attrib]
                if style_attr_elements:
                    # Parse style attributes
                    style_props = {}
                    for elem in style_attr_elements:
                        style_attr = elem.get('style', '')
                        for item in style_attr.split(';'):
                            if ':' in item:
                                prop, val = item.split(':', 1)
                                prop = prop.strip()
                                val = val.strip()
                                if prop not in style_props:
                                    style_props[prop] = []
                                style_props[prop].append(val)
                    
                    # Add common style properties
                    if style_props:
                        rule["styles"] = {}
                        for prop, values in style_props.items():
                            if len(values) >= len(style_attr_elements) * 0.5:
                                most_common = max(set(values), key=values.count)
                                rule["styles"][prop] = most_common
            
            # Add rule if it has attributes
            if rule["attributes"] or rule.get("styles"):
                rules.append(rule)
        
        return rules
    
    def _extract_defs(self, root):
        """Extract definitions from the SVG."""
        defs = []
        
        # Find defs element
        defs_elem = root.find('.//defs')
        if defs_elem is None:
            return defs
            
        # Process each definition
        for child in defs_elem:
            tag = child.tag.split('}')[-1]  # Handle namespaces
            
            # Skip non-standard elements
            if tag not in ['linearGradient', 'radialGradient', 'filter', 'pattern', 'mask', 'clipPath']:
                continue
                
            # Create definition object
            def_obj = {
                "type": tag,
                "id": child.get('id', ''),
                "attributes": {}
            }
            
            # Extract attributes
            for attr, value in child.attrib.items():
                if attr != 'id':
                    def_obj["attributes"][attr] = value
            
            # Handle specific definition types
            if tag in ['linearGradient', 'radialGradient']:
                # Extract gradient stops
                stops = []
                for stop in child.findall('.//stop'):
                    stop_obj = {
                        "offset": stop.get('offset', '0%'),
                        "color": stop.get('stop-color', '#000000')
                    }
                    if 'stop-opacity' in stop.attrib:
                        stop_obj["opacity"] = float(stop.get('stop-opacity'))
                    stops.append(stop_obj)
                
                if stops:
                    def_obj["stops"] = stops
            
            elif tag in ['filter', 'pattern', 'mask', 'clipPath']:
                # Extract child elements
                children = []
                for grandchild in child:
                    gc_tag = grandchild.tag.split('}')[-1]
                    gc_obj = {
                        "type": gc_tag,
                        "attributes": {}
                    }
                    
                    # Extract attributes
                    for attr, value in grandchild.attrib.items():
                        gc_obj["attributes"][attr] = value
                    
                    children.append(gc_obj)
                
                if children:
                    def_obj["children"] = children
            
            # Add definition if valid
            if def_obj["id"]:
                defs.append(def_obj)
        
        return defs

    def create_style(self, svg_string, style_name, description, author,
                    selectors_to_include="path,circle,rect", 
                    extract_defs=True, extract_colors_only=False):
        """
        Create a style preset from an SVG string.
        
        Args:
            svg_string: The SVG content as a string
            style_name: Name for the new style
            description: Description of the style
            author: Author of the style
            selectors_to_include: Comma-separated list of selectors to extract
            extract_defs: Whether to extract definitions
            extract_colors_only: Whether to extract only color information
            
        Returns:
            A message indicating success or failure
        """
        # Create styles folder if it doesn't exist
        os.makedirs(STYLES_FOLDER, exist_ok=True)
        
        try:
            # Parse the SVG
            root = self._parse_svg(svg_string)
            
            # Parse selectors
            selectors = [s.strip() for s in selectors_to_include.split(',')]
            
            # Get elements matching the selectors
            elements = self._get_elements(root, selectors)
            
            # Extract style rules
            rules = self._extract_style_rules(elements, extract_colors_only)
            
            # Create style object
            style = {
                "name": style_name,
                "description": description,
                "author": author,
                "version": "1.0",
                "rules": rules
            }
            
            # Extract definitions if requested
            if extract_defs:
                defs = self._extract_defs(root)
                if defs:
                    style["defs"] = defs
            
            # Convert style name to filename
            filename = style_name.lower().replace(' ', '_') + '.json'
            filepath = os.path.join(STYLES_FOLDER, filename)
            
            # Save the style
            with open(filepath, 'w') as f:
                json.dump(style, f, indent=2)
            
            return (f"Style '{style_name}' created successfully as '{filename}'.",)
            
        except Exception as e:
            return (f"Error creating style: {str(e)}",)

class SVGStylesManager:
    """
    Node for managing style presets - listing, copying, deleting, etc.
    """
    @classmethod
    def INPUT_TYPES(cls):
        # Create styles folder if it doesn't exist
        os.makedirs(STYLES_FOLDER, exist_ok=True)
        
        # Scan for available style files
        style_files = cls._get_available_styles()
        
        return {
            "required": {
                "operation": (["list_styles", "copy_style", "delete_style", "preview_style"], {"default": "list_styles"}),
                "style_name": (style_files, {"default": style_files[0] if style_files else "none"}),
            },
            "optional": {
                "new_style_name": ("STRING", {"default": "custom_style", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "manage_styles"
    CATEGORY = "💎TOSVG/Styler"

    @staticmethod
    def _get_available_styles():
        """Get list of available style files."""
        if not os.path.exists(STYLES_FOLDER):
            return ["none"]
            
        style_files = []
        for filename in os.listdir(STYLES_FOLDER):
            if filename.endswith('.json'):
                style_files.append(filename)
        
        return style_files if style_files else ["none"]

    def manage_styles(self, operation, style_name, new_style_name="custom_style"):
        """
        Manage style presets.
        
        Args:
            operation: The operation to perform
            style_name: Name of the style to work with
            new_style_name: New name for copied styles
            
        Returns:
            A string with the result of the operation
        """
        # Create styles folder if it doesn't exist
        os.makedirs(STYLES_FOLDER, exist_ok=True)
        
        try:
            if operation == "list_styles":
                # List all available styles with info
                styles_info = []
                for filename in self._get_available_styles():
                    if filename == "none":
                        continue
                        
                    filepath = os.path.join(STYLES_FOLDER, filename)
                    try:
                        with open(filepath, 'r') as f:
                            style = json.load(f)
                            styles_info.append(f"• {style.get('name', filename)}: {style.get('description', 'No description')} (Author: {style.get('author', 'Unknown')})")
                    except:
                        styles_info.append(f"• {filename}: Could not read style info")
                
                if styles_info:
                    return ("Available Styles:\n" + "\n".join(styles_info),)
                else:
                    return ("No styles found. Create some with SVGCreateStyle node.",)
            
            elif operation == "copy_style":
                # Check if style exists
                if style_name == "none":
                    return ("No style selected.",)
                    
                style_path = os.path.join(STYLES_FOLDER, style_name)
                if not os.path.exists(style_path):
                    return (f"Style '{style_name}' not found.",)
                
                # Load the style
                with open(style_path, 'r') as f:
                    style = json.load(f)
                
                # Create a copy with new name
                style['name'] = new_style_name
                
                # Save with new filename
                new_filename = new_style_name.lower().replace(' ', '_') + '.json'
                new_path = os.path.join(STYLES_FOLDER, new_filename)
                
                with open(new_path, 'w') as f:
                    json.dump(style, f, indent=2)
                
                return (f"Style '{style_name}' copied to '{new_filename}'.",)
            
            elif operation == "delete_style":
                # Check if style exists
                if style_name == "none":
                    return ("No style selected.",)
                    
                style_path = os.path.join(STYLES_FOLDER, style_name)
                if not os.path.exists(style_path):
                    return (f"Style '{style_name}' not found.",)
                
                # Delete the style
                os.remove(style_path)
                
                return (f"Style '{style_name}' deleted.",)
            
            elif operation == "preview_style":
                # Generate a preview of the style as SVG code
                if style_name == "none":
                    return ("No style selected.",)
                    
                style_path = os.path.join(STYLES_FOLDER, style_name)
                if not os.path.exists(style_path):
                    return (f"Style '{style_name}' not found.",)
                
                # Load the style
                with open(style_path, 'r') as f:
                    style = json.load(f)
                
                # Format style info
                result = []
                result.append(f"Style Name: {style.get('name', 'Unnamed')}")
                result.append(f"Description: {style.get('description', 'No description')}")
                result.append(f"Author: {style.get('author', 'Unknown')}")
                result.append(f"Version: {style.get('version', '1.0')}")
                result.append("\nRules:")
                
                for i, rule in enumerate(style.get('rules', [])):
                    result.append(f"\nRule {i+1}:")
                    result.append(f"  Selector: {rule.get('selector', 'unknown')}")
                    
                    if 'attributes' in rule:
                        result.append("  Attributes:")
                        for attr, value in rule['attributes'].items():
                            result.append(f"    {attr}: {value}")
                    
                    if 'styles' in rule:
                        result.append("  Styles:")
                        for prop, value in rule['styles'].items():
                            result.append(f"    {prop}: {value}")
                
                if 'defs' in style:
                    result.append("\nDefinitions:")
                    for i, def_item in enumerate(style['defs']):
                        result.append(f"\nDef {i+1}:")
                        result.append(f"  Type: {def_item.get('type', 'unknown')}")
                        result.append(f"  ID: {def_item.get('id', 'unknown')}")
                
                return ("\n".join(result),)
            
            else:
                return (f"Unknown operation: {operation}",)
                
        except Exception as e:
            return (f"Error managing styles: {str(e)}",)

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "SVGStyler": SVGStyler,
    "SVGCreateStyle": SVGCreateStyle,
    "SVGStylesManager": SVGStylesManager,
}

# NODE DISPLAY NAMES
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVGStyler": "SVG Styler",
    "SVGCreateStyle": "Create SVG Style",
    "SVGStylesManager": "Manage SVG Styles",
}