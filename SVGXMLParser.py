import os
import time
import random
import math
from typing import List, Tuple, Dict, Any, Optional, Union
import xml.etree.ElementTree as ET
import re

# XML Namespaces for SVG
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}

class SVGXMLParser:
    """
    Base SVG XML parser that provides common utilities for XML manipulation.
    """
    @staticmethod
    def parse_svg(svg_string):
        """Parse SVG string to XML Element Tree."""
        # Handle namespace prefixes for proper parsing
        try:
            # Try with namespace handling
            root = ET.fromstring(svg_string)
            return root
        except ET.ParseError as e:
            # Fall back to direct parsing if namespace handling fails
            try:
                # Remove doctype if it exists
                svg_clean = re.sub(r'<!DOCTYPE[^>]*>', '', svg_string)
                root = ET.fromstring(svg_clean)
                return root
            except Exception as e2:
                raise ValueError(f"Failed to parse SVG: {str(e)} -> {str(e2)}")
    
    @staticmethod
    def element_to_string(element):
        """Convert Element Tree to string."""
        return ET.tostring(element, encoding='unicode')
    
    @staticmethod
    def find_elements(root, selector):
        """Find elements matching the given selector."""
        # Simple selector parsing
        if selector.startswith('#'):
            # ID selector
            id_value = selector[1:]
            found = []
            for elem in root.iter():
                if elem.get('id') == id_value:
                    found.append(elem)
            return found
            
        elif selector.startswith('.'):
            # Class selector
            class_value = selector[1:]
            found = []
            for elem in root.iter():
                classes = elem.get('class', '').split()
                if class_value in classes:
                    found.append(elem)
            return found
            
        else:
            # Tag selector
            return root.findall(f".//{selector}", SVG_NS)
    
    @staticmethod
    def get_attributes(element):
        """Get all attributes of an element."""
        return element.attrib
    
    @staticmethod
    def set_attribute(element, name, value):
        """Set an attribute on an element."""
        element.set(name, value)
    
    @staticmethod
    def create_element(tag, attributes=None):
        """Create a new element with specified attributes."""
        elem = ET.Element(tag)
        if attributes:
            for name, value in attributes.items():
                elem.set(name, value)
        return elem

class SVGElementSelector:
    """
    Node for selecting SVG elements using CSS-like selectors and outputting information.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "selector": ("STRING", {"default": "path", "multiline": False}),
                "output_type": (["element_count", "element_ids", "element_classes", 
                               "element_attributes", "selected_elements"], {"default": "element_count"}),
            },
            "optional": {
                "attribute_filter": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT",)
    FUNCTION = "select_elements"
    CATEGORY = "💎TOSVG/XMLParser"

    def select_elements(self, svg_string, selector, output_type, attribute_filter=""):
        """
        Select SVG elements and return information based on the selected output type.
        """
        try:
            # Parse SVG
            parser = SVGXMLParser()
            root = parser.parse_svg(svg_string)
            
            # Find elements matching the selector
            elements = parser.find_elements(root, selector)
            
            # Filter by attribute if specified
            if attribute_filter:
                attr_name, attr_value = attribute_filter.split('=', 1) if '=' in attribute_filter else (attribute_filter, None)
                filtered_elements = []
                for elem in elements:
                    if attr_value is None:
                        # Check if attribute exists
                        if attr_name in elem.attrib:
                            filtered_elements.append(elem)
                    else:
                        # Check if attribute has specific value
                        if elem.get(attr_name) == attr_value:
                            filtered_elements.append(elem)
                elements = filtered_elements
            
            # Process elements based on output type
            if output_type == "element_count":
                return (f"Found {len(elements)} elements matching '{selector}'", len(elements))
                
            elif output_type == "element_ids":
                ids = [elem.get('id', '[no-id]') for elem in elements]
                return (', '.join(ids), len(elements))
                
            elif output_type == "element_classes":
                classes = [elem.get('class', '[no-class]') for elem in elements]
                return (', '.join(classes), len(elements))
                
            elif output_type == "element_attributes":
                attributes = []
                for i, elem in enumerate(elements):
                    elem_attrs = parser.get_attributes(elem)
                    attr_str = ', '.join(f"{k}=\"{v}\"" for k, v in elem_attrs.items())
                    attributes.append(f"Element {i}: {elem.tag} - {attr_str}")
                return ('\n'.join(attributes), len(elements))
                
            elif output_type == "selected_elements":
                # Return string representation of the selected elements
                elements_str = ""
                for i, elem in enumerate(elements):
                    elements_str += f"Element {i}: {parser.element_to_string(elem)}\n\n"
                return (elements_str, len(elements))
            
            return (f"Unknown output type: {output_type}", 0)
            
        except Exception as e:
            return (f"Error selecting elements: {str(e)}", 0)

class SVGElementManipulator:
    """
    Node for manipulating SVG elements using ElementTree operations.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "operation": (["add_element", "remove_element", "replace_element", 
                             "modify_element", "duplicate_element", "reorder_element"],
                             {"default": "modify_element"}),
                "selector": ("STRING", {"default": "path", "multiline": False}),
            },
            "optional": {
                "new_tag": ("STRING", {"default": "rect", "multiline": False}),
                "attributes": ("STRING", {"default": "width=\"100\" height=\"100\" x=\"10\" y=\"10\" fill=\"red\"", 
                                       "multiline": True}),
                "content": ("STRING", {"default": "", "multiline": True}),
                "target_index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "reorder_position": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "duplicate_count": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "manipulate_elements"
    CATEGORY = "💎TOSVG/XMLParser"

    def _parse_attributes(self, attr_string):
        """Parse attribute string into a dictionary."""
        attributes = {}
        
        # Handle empty attribute string
        if not attr_string:
            return attributes
            
        # Simple attribute parser
        pairs = re.findall(r'(\w+)\s*=\s*"([^"]*)"', attr_string)
        for name, value in pairs:
            attributes[name] = value
            
        return attributes

    def manipulate_elements(self, svg_string, operation, selector,
                           new_tag="rect", attributes="", content="",
                           target_index=0, reorder_position=0, duplicate_count=1):
        """
        Manipulate SVG elements based on the selected operation.
        """
        try:
            # Parse SVG
            parser = SVGXMLParser()
            root = parser.parse_svg(svg_string)
            
            # Find elements matching the selector
            elements = parser.find_elements(root, selector)
            
            if not elements:
                return (f"No elements found matching '{selector}'",)
            
            # Parse attributes
            parsed_attributes = self._parse_attributes(attributes)
            
            # Apply operation
            if operation == "add_element":
                # Create new element
                new_element = parser.create_element(new_tag, parsed_attributes)
                if content:
                    new_element.text = content
                
                # Add to parent of the selected element
                if len(elements) > target_index:
                    parent = root if elements[target_index].getparent() is None else elements[target_index].getparent()
                    parent.append(new_element)
                
            elif operation == "remove_element":
                # Remove elements
                for i, elem in enumerate(elements):
                    if i == target_index or target_index == -1:
                        parent = root.find(f".//{elem.tag}/..") if elem.getparent() is None else elem.getparent()
                        if parent is not None:
                            parent.remove(elem)
                
            elif operation == "replace_element":
                # Replace element with a new one
                if len(elements) > target_index:
                    elem = elements[target_index]
                    parent = root if elem.getparent() is None else elem.getparent()
                    
                    # Create new element
                    new_element = parser.create_element(new_tag, parsed_attributes)
                    if content:
                        new_element.text = content
                    
                    # Position new element at same position
                    index = list(parent).index(elem)
                    parent.remove(elem)
                    parent.insert(index, new_element)
                
            elif operation == "modify_element":
                # Modify element attributes
                for i, elem in enumerate(elements):
                    if i == target_index or target_index == -1:
                        for name, value in parsed_attributes.items():
                            parser.set_attribute(elem, name, value)
                        if content:
                            elem.text = content
                
            elif operation == "duplicate_element":
                # Duplicate elements
                if len(elements) > target_index:
                    elem = elements[target_index]
                    parent = root if elem.getparent() is None else elem.getparent()
                    index = list(parent).index(elem)
                    
                    for i in range(duplicate_count):
                        # Create a deep copy by converting to string and back
                        elem_str = parser.element_to_string(elem)
                        new_elem = parser.parse_svg(elem_str)
                        
                        # Add random suffix to IDs to avoid duplicates
                        for e in new_elem.iter():
                            if 'id' in e.attrib:
                                e.attrib['id'] = f"{e.attrib['id']}_{i+1}"
                        
                        parent.insert(index + i + 1, new_elem)
                
            elif operation == "reorder_element":
                # Reorder element (move up or down in the DOM)
                if len(elements) > target_index:
                    elem = elements[target_index]
                    parent = root if elem.getparent() is None else elem.getparent()
                    
                    if parent is not None:
                        children = list(parent)
                        if elem in children:
                            index = children.index(elem)
                            new_index = max(0, min(len(children) - 1, index + reorder_position))
                            
                            if new_index != index:
                                parent.remove(elem)
                                parent.insert(new_index, elem)
            
            # Convert modified SVG back to string
            modified_svg = parser.element_to_string(root)
            return (modified_svg,)
            
        except Exception as e:
            return (f"Error manipulating elements: {str(e)}",)

class SVGElementStyler:
    """
    Node for styling SVG elements with CSS.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "selector": ("STRING", {"default": "path", "multiline": False}),
                "style_method": (["inline_style", "class_style", "add_stylesheet"], {"default": "inline_style"}),
            },
            "optional": {
                "style_properties": ("STRING", {"default": "fill: red; stroke: black; stroke-width: 2;", 
                                             "multiline": True}),
                "class_name": ("STRING", {"default": "highlighted", "multiline": False}),
                "target_index": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1,
                                      "tooltip": "-1 to affect all matching elements"}),
                "css_rules": ("STRING", {"default": ".highlighted { fill: red; stroke: black; }\n#special { opacity: 0.5; }", 
                                      "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "style_elements"
    CATEGORY = "💎TOSVG/XMLParser"

    def style_elements(self, svg_string, selector, style_method,
                     style_properties="", class_name="highlighted",
                     target_index=-1, css_rules=""):
        """
        Apply styling to SVG elements.
        """
        try:
            # Parse SVG
            parser = SVGXMLParser()
            root = parser.parse_svg(svg_string)
            
            # Find elements matching the selector
            elements = parser.find_elements(root, selector)
            
            if not elements:
                return (f"No elements found matching '{selector}'",)
            
            if style_method == "inline_style":
                # Apply inline styles directly to elements
                for i, elem in enumerate(elements):
                    if target_index == -1 or i == target_index:
                        # Get existing style
                        existing_style = elem.get('style', '')
                        
                        # Merge styles (simple approach)
                        new_style = existing_style
                        if existing_style and not existing_style.endswith(';'):
                            new_style += '; '
                        new_style += style_properties
                        
                        # Set the style attribute
                        elem.set('style', new_style)
            
            elif style_method == "class_style":
                # Add class to elements
                for i, elem in enumerate(elements):
                    if target_index == -1 or i == target_index:
                        # Get existing classes
                        existing_classes = elem.get('class', '').split()
                        
                        # Add new class if not present
                        if class_name not in existing_classes:
                            existing_classes.append(class_name)
                        
                        # Set the class attribute
                        elem.set('class', ' '.join(existing_classes))
                
                # Check if we need to add a style element with the class definition
                if css_rules:
                    self._add_or_update_stylesheet(root, css_rules)
            
            elif style_method == "add_stylesheet":
                # Add or update stylesheet in the SVG
                if css_rules:
                    self._add_or_update_stylesheet(root, css_rules)
            
            # Convert modified SVG back to string
            modified_svg = parser.element_to_string(root)
            return (modified_svg,)
            
        except Exception as e:
            return (f"Error styling elements: {str(e)}",)
    
    def _add_or_update_stylesheet(self, root, css_rules):
        """Add or update a stylesheet in the SVG."""
        # Find existing style element
        style_elem = None
        for elem in root.findall(".//style", SVG_NS):
            style_elem = elem
            break
        
        if style_elem is None:
            # Create new style element
            style_elem = ET.Element("style")
            style_elem.set("type", "text/css")
            
            # Add to the SVG document (after defs if it exists, or as first child)
            defs = root.find(".//defs", SVG_NS)
            if defs is not None:
                defs_index = list(root).index(defs)
                root.insert(defs_index + 1, style_elem)
            else:
                # Add as first element
                root.insert(0, style_elem)
        
        # Set or append CSS rules
        if style_elem.text:
            style_elem.text += "\n" + css_rules
        else:
            style_elem.text = css_rules

class SVGDefs:
    """
    Node for managing SVG defs section (gradients, filters, etc.).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "operation": (["add_linear_gradient", "add_radial_gradient", 
                             "add_pattern", "add_filter", "add_mask",
                             "add_clip_path", "remove_def"], {"default": "add_linear_gradient"}),
                "def_id": ("STRING", {"default": "gradient1", "multiline": False}),
            },
            "optional": {
                "pattern_width": ("INT", {"default": 20, "min": 1, "max": 1000, "step": 1}),
                "pattern_height": ("INT", {"default": 20, "min": 1, "max": 1000, "step": 1}),
                "gradient_stops": ("STRING", {"default": "0%:#ff0000;50%:#00ff00;100%:#0000ff", 
                                           "multiline": True}),
                "gradient_x1": ("STRING", {"default": "0%", "multiline": False}),
                "gradient_y1": ("STRING", {"default": "0%", "multiline": False}),
                "gradient_x2": ("STRING", {"default": "100%", "multiline": False}),
                "gradient_y2": ("STRING", {"default": "100%", "multiline": False}),
                "gradient_r": ("STRING", {"default": "50%", "multiline": False}),
                "gradient_cx": ("STRING", {"default": "50%", "multiline": False}),
                "gradient_cy": ("STRING", {"default": "50%", "multiline": False}),
                "filter_elements": ("STRING", {"default": "<feGaussianBlur stdDeviation=\"2\"/>", 
                                            "multiline": True}),
                "mask_content": ("STRING", {"default": "<rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" fill=\"white\"/>", 
                                         "multiline": True}),
                "clip_path_content": ("STRING", {"default": "<circle cx=\"50\" cy=\"50\" r=\"50\"/>", 
                                              "multiline": True}),
                "pattern_content": ("STRING", {"default": "<rect width=\"10\" height=\"10\" fill=\"red\"/><rect x=\"10\" y=\"10\" width=\"10\" height=\"10\" fill=\"blue\"/>", 
                                            "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "manage_defs"
    CATEGORY = "💎TOSVG/XMLParser"

    def _ensure_defs_element(self, root):
        """Ensure that a defs element exists, create if not."""
        defs = root.find(".//defs", SVG_NS)
        if defs is None:
            defs = ET.Element("defs")
            root.insert(0, defs)
        return defs
    
    def _parse_gradient_stops(self, stops_str):
        """Parse gradient stops string into a list of (offset, color) tuples."""
        stops = []
        for stop in stops_str.split(';'):
            if ':' in stop:
                offset, color = stop.split(':', 1)
                stops.append((offset.strip(), color.strip()))
        return stops

    def manage_defs(self, svg_string, operation, def_id,
                  pattern_width=20, pattern_height=20,
                  gradient_stops="0%:#ff0000;100%:#0000ff",
                  gradient_x1="0%", gradient_y1="0%",
                  gradient_x2="100%", gradient_y2="100%",
                  gradient_r="50%", gradient_cx="50%", gradient_cy="50%",
                  filter_elements="", mask_content="", clip_path_content="",
                  pattern_content=""):
        """
        Manage SVG defs section (add/remove gradients, filters, etc.).
        """
        try:
            # Parse SVG
            parser = SVGXMLParser()
            root = parser.parse_svg(svg_string)
            
            # Get or create defs element
            defs = self._ensure_defs_element(root)
            
            # Remove existing def with same ID if it exists
            for elem in defs.findall(".//*[@id='{}']".format(def_id)):
                defs.remove(elem)
            
            # Add new def based on operation
            if operation == "add_linear_gradient":
                # Create linear gradient
                gradient = ET.Element("linearGradient")
                gradient.set("id", def_id)
                gradient.set("x1", gradient_x1)
                gradient.set("y1", gradient_y1)
                gradient.set("x2", gradient_x2)
                gradient.set("y2", gradient_y2)
                
                # Add stops
                stops = self._parse_gradient_stops(gradient_stops)
                for offset, color in stops:
                    stop = ET.Element("stop")
                    stop.set("offset", offset)
                    stop.set("stop-color", color)
                    gradient.append(stop)
                
                defs.append(gradient)
            
            elif operation == "add_radial_gradient":
                # Create radial gradient
                gradient = ET.Element("radialGradient")
                gradient.set("id", def_id)
                gradient.set("cx", gradient_cx)
                gradient.set("cy", gradient_cy)
                gradient.set("r", gradient_r)
                
                # Add stops
                stops = self._parse_gradient_stops(gradient_stops)
                for offset, color in stops:
                    stop = ET.Element("stop")
                    stop.set("offset", offset)
                    stop.set("stop-color", color)
                    gradient.append(stop)
                
                defs.append(gradient)
            
            elif operation == "add_pattern":
                # Create pattern
                pattern = ET.Element("pattern")
                pattern.set("id", def_id)
                pattern.set("width", str(pattern_width))
                pattern.set("height", str(pattern_height))
                pattern.set("patternUnits", "userSpaceOnUse")
                
                # Add pattern content
                if pattern_content:
                    # Parse content
                    try:
                        content_root = parser.parse_svg(f"<svg>{pattern_content}</svg>")
                        for child in content_root:
                            pattern.append(child)
                    except Exception as e:
                        # If content parse fails, add it as text (may not work)
                        pattern.text = pattern_content
                
                defs.append(pattern)
            
            elif operation == "add_filter":
                # Create filter
                filter_elem = ET.Element("filter")
                filter_elem.set("id", def_id)
                
                # Add filter elements
                if filter_elements:
                    # Parse filter elements
                    try:
                        content_root = parser.parse_svg(f"<svg>{filter_elements}</svg>")
                        for child in content_root:
                            filter_elem.append(child)
                    except Exception as e:
                        # If content parse fails, add it as text (may not work)
                        filter_elem.text = filter_elements
                
                defs.append(filter_elem)
            
            elif operation == "add_mask":
                # Create mask
                mask = ET.Element("mask")
                mask.set("id", def_id)
                
                # Add mask content
                if mask_content:
                    # Parse content
                    try:
                        content_root = parser.parse_svg(f"<svg>{mask_content}</svg>")
                        for child in content_root:
                            mask.append(child)
                    except Exception as e:
                        # If content parse fails, add it as text (may not work)
                        mask.text = mask_content
                
                defs.append(mask)
            
            elif operation == "add_clip_path":
                # Create clipPath
                clip_path = ET.Element("clipPath")
                clip_path.set("id", def_id)
                
                # Add clip path content
                if clip_path_content:
                    # Parse content
                    try:
                        content_root = parser.parse_svg(f"<svg>{clip_path_content}</svg>")
                        for child in content_root:
                            clip_path.append(child)
                    except Exception as e:
                        # If content parse fails, add it as text (may not work)
                        clip_path.text = clip_path_content
                
                defs.append(clip_path)
            
            elif operation == "remove_def":
                # Already handled above (by removing the def with specified ID)
                pass
            
            # Convert modified SVG back to string
            modified_svg = parser.element_to_string(root)
            return (modified_svg,)
            
        except Exception as e:
            return (f"Error managing defs: {str(e)}",)

class SVGViewBox:
    """
    Node for manipulating SVG viewBox and dimensions.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "operation": (["set_viewbox", "set_dimensions", "auto_fit", 
                             "crop_to_content", "center_content"],
                             {"default": "set_viewbox"}),
            },
            "optional": {
                "min_x": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 1.0}),
                "min_y": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 1.0}),
                "width": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 10000.0, "step": 1.0}),
                "height": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 10000.0, "step": 1.0}),
                "keep_aspect_ratio": ("BOOLEAN", {"default": True}),
                "padding": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "manipulate_viewbox"
    CATEGORY = "💎TOSVG/XMLParser"

    def _get_content_bounds(self, root):
        """Calculate bounds of all visible content in the SVG."""
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for elem in root.iter():
            # Check common elements with positional attributes
            tag = elem.tag
            
            # Handle positioned elements like rect, circle, ellipse, etc.
            if tag.endswith('rect'):
                x = float(elem.get('x', '0'))
                y = float(elem.get('y', '0'))
                width = float(elem.get('width', '0'))
                height = float(elem.get('height', '0'))
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + width)
                max_y = max(max_y, y + height)
            
            elif tag.endswith('circle'):
                cx = float(elem.get('cx', '0'))
                cy = float(elem.get('cy', '0'))
                r = float(elem.get('r', '0'))
                min_x = min(min_x, cx - r)
                min_y = min(min_y, cy - r)
                max_x = max(max_x, cx + r)
                max_y = max(max_y, cy + r)
            
            elif tag.endswith('ellipse'):
                cx = float(elem.get('cx', '0'))
                cy = float(elem.get('cy', '0'))
                rx = float(elem.get('rx', '0'))
                ry = float(elem.get('ry', '0'))
                min_x = min(min_x, cx - rx)
                min_y = min(min_y, cy - ry)
                max_x = max(max_x, cx + rx)
                max_y = max(max_y, cy + ry)
            
            elif tag.endswith('line'):
                x1 = float(elem.get('x1', '0'))
                y1 = float(elem.get('y1', '0'))
                x2 = float(elem.get('x2', '0'))
                y2 = float(elem.get('y2', '0'))
                min_x = min(min_x, x1, x2)
                min_y = min(min_y, y1, y2)
                max_x = max(max_x, x1, x2)
                max_y = max(max_y, y1, y2)
            
            elif tag.endswith('path'):
                # Basic path parsing for bounds is complex
                # This is a simplified approach that may not be accurate for all paths
                d = elem.get('d', '')
                if d:
                    # Extract coordinates from path data
                    coords = re.findall(r'[-+]?\d*\.\d+|\d+', d)
                    coords = [float(c) for c in coords]
                    
                    # If we found coordinates, assume they alternate x,y
                    for i in range(0, len(coords), 2):
                        if i+1 < len(coords):
                            min_x = min(min_x, coords[i])
                            min_y = min(min_y, coords[i+1])
                            max_x = max(max_x, coords[i])
                            max_y = max(max_y, coords[i+1])
        
        # If no elements found, use defaults
        if min_x == float('inf'):
            min_x, min_y = 0, 0
            max_x, max_y = 100, 100
        
        return min_x, min_y, max_x, max_y

    def manipulate_viewbox(self, svg_string, operation,
                         min_x=0.0, min_y=0.0, width=100.0, height=100.0,
                         keep_aspect_ratio=True, padding=10.0):
        """
        Manipulate SVG viewBox and dimensions.
        """
        try:
            # Parse SVG
            parser = SVGXMLParser()
            root = parser.parse_svg(svg_string)
            
            # Get the SVG element (root)
            svg_element = root
            
            if operation == "set_viewbox":
                # Set viewBox attribute
                svg_element.set('viewBox', f"{min_x} {min_y} {width} {height}")
            
            elif operation == "set_dimensions":
                # Set width and height attributes
                svg_element.set('width', str(width))
                svg_element.set('height', str(height))
                
                # Optionally update viewBox if it exists
                if 'viewBox' in svg_element.attrib:
                    vb_parts = svg_element.get('viewBox').split()
                    if len(vb_parts) == 4:
                        # Keep min_x and min_y, update width and height
                        svg_element.set('viewBox', f"{vb_parts[0]} {vb_parts[1]} {width} {height}")
            
            elif operation == "auto_fit":
                # Calculate content bounds
                min_x, min_y, max_x, max_y = self._get_content_bounds(root)
                
                # Add padding
                min_x -= padding
                min_y -= padding
                max_x += padding
                max_y += padding
                
                # Calculate new dimensions
                new_width = max_x - min_x
                new_height = max_y - min_y
                
                # Set viewBox
                svg_element.set('viewBox', f"{min_x} {min_y} {new_width} {new_height}")
                
                # Set dimensions
                svg_element.set('width', str(new_width))
                svg_element.set('height', str(new_height))
            
            elif operation == "crop_to_content":
                # Calculate content bounds
                min_x, min_y, max_x, max_y = self._get_content_bounds(root)
                
                # Add padding
                min_x -= padding
                min_y -= padding
                max_x += padding
                max_y += padding
                
                # Calculate new dimensions
                new_width = max_x - min_x
                new_height = max_y - min_y
                
                # Set viewBox to crop to content
                svg_element.set('viewBox', f"{min_x} {min_y} {new_width} {new_height}")
                
                # Keep original dimensions but update viewBox
                if 'width' in svg_element.attrib and 'height' in svg_element.attrib:
                    orig_width = float(svg_element.get('width', new_width))
                    orig_height = float(svg_element.get('height', new_height))
                    
                    if keep_aspect_ratio:
                        # Maintain aspect ratio of the original dimensions
                        aspect = orig_width / orig_height
                        content_aspect = new_width / new_height
                        
                        if content_aspect > aspect:
                            # Content is wider, adjust height
                            orig_height = orig_width / content_aspect
                        else:
                            # Content is taller, adjust width
                            orig_width = orig_height * content_aspect
                    
                    svg_element.set('width', str(orig_width))
                    svg_element.set('height', str(orig_height))
            
            elif operation == "center_content":
                # Calculate content bounds
                min_x, min_y, max_x, max_y = self._get_content_bounds(root)
                
                # Calculate content center
                content_cx = (min_x + max_x) / 2
                content_cy = (min_y + max_y) / 2
                
                # Get current viewBox
                vb_parts = svg_element.get('viewBox', f"0 0 {width} {height}").split()
                if len(vb_parts) == 4:
                    vb_x = float(vb_parts[0])
                    vb_y = float(vb_parts[1])
                    vb_w = float(vb_parts[2])
                    vb_h = float(vb_parts[3])
                    
                    # Calculate viewBox center
                    vb_cx = vb_x + vb_w / 2
                    vb_cy = vb_y + vb_h / 2
                    
                    # Calculate translation needed to center content
                    dx = vb_cx - content_cx
                    dy = vb_cy - content_cy
                    
                    # Apply translation to all elements
                    for elem in root.iter():
                        if elem is not svg_element:  # Skip the SVG element itself
                            # Get existing transform
                            transform = elem.get('transform', '')
                            
                            # Add translation transform
                            if transform:
                                elem.set('transform', f"{transform} translate({dx},{dy})")
                            else:
                                elem.set('transform', f"translate({dx},{dy})")
            
            # Convert modified SVG back to string
            modified_svg = parser.element_to_string(root)
            return (modified_svg,)
            
        except Exception as e:
            return (f"Error manipulating viewBox: {str(e)}",)

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "SVGElementSelector": SVGElementSelector,
    "SVGElementManipulator": SVGElementManipulator,
    "SVGElementStyler": SVGElementStyler,
    "SVGDefs": SVGDefs,
    "SVGViewBox": SVGViewBox,
}

# NODE DISPLAY NAMES
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVGElementSelector": "SVG Element Selector",
    "SVGElementManipulator": "SVG Element Manipulator",
    "SVGElementStyler": "SVG Element Styler",
    "SVGDefs": "SVG Definitions Manager",
    "SVGViewBox": "SVG ViewBox Manager",
}