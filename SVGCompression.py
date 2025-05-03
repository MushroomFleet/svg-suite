import os
import io
import time
from typing import Dict, Tuple, List, Any, Optional, Union
import numpy as np

# Attempt to import SVGCompress and Scour with error handling
svg_compress_available = False
scour_available = False

try:
    from SVGCompress import compress_by_method
    svg_compress_available = True
except ImportError:
    print("Warning: SVGCompress package not found. Advanced SVG Compression nodes will not function.")
    # Define a placeholder for the module to avoid errors
    def compress_by_method(*args, **kwargs):
        raise ImportError("SVGCompress package is not installed. Please install it with: pip install SVGCompress")

try:
    from scour import scour
    scour_available = True
except ImportError:
    print("Warning: Scour package not found. SVG optimization nodes will not function.")
    # Create a placeholder module to avoid errors
    class ScourPlaceholder:
        def sanitizeOptions(self):
            return type('obj', (object,), {})
        
        def scourString(self, *args, **kwargs):
            raise ImportError("Scour package is not installed. Please install it with: pip install scour")
    
    scour = ScourPlaceholder()

# Node for SVG compression using SVGCompress package
class SVGCompressAdvanced:
    """
    Advanced SVG compression using the SVGCompress package.
    Provides multiple methods to reduce SVG file size by:
    - Removing tiny polygons
    - Simplifying shapes using Ramer-Douglas-Peucker algorithm
    - Merging adjacent or overlapping shapes
    
    Status: {availability}
    
    Required packages: SVGCompress, shapely, rdp, svg.path, lxml
    Install with: pip install SVGCompress
    """.format(availability="Available" if svg_compress_available else "Not Available - Package missing")
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "compression_type": (["delete", "simplify", "merge"], {"default": "simplify"}),
                "curve_fidelity": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "pre_select": ("BOOLEAN", {"default": False}),
                "selection_criteria": (["bboxarea", "circumference"], {"default": "bboxarea"}),
                "selection_threshold": ("FLOAT", {"default": 300.0, "min": 0.1, "max": 10000.0, "step": 0.1}),
                "optimize_after": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Parameters for deletion-based compression
                "delete_threshold": ("FLOAT", {"default": 100.0, "min": 0.1, "max": 10000.0, "step": 0.1}),
                "delete_criteria": (["bboxarea", "circumference"], {"default": "bboxarea"}),
                
                # Parameters for simplification-based compression
                "simplify_epsilon": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                
                # Parameters for merge-based compression
                "merge_epsilon": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "buffer_distance": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "operation_key": (["hull", "union"], {"default": "hull"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "compress_svg"
    CATEGORY = "💎TOSVG/Advanced"

    def compress_svg(self, svg_string, compression_type, curve_fidelity, pre_select, 
                    selection_criteria, selection_threshold, optimize_after,
                    delete_threshold=100.0, delete_criteria="bboxarea",
                    simplify_epsilon=1.0, merge_epsilon=5.0, buffer_distance=5.0,
                    operation_key="hull"):
        """
        Compress an SVG string using the SVGCompress package
        """
        # Check if SVGCompress is available
        if not svg_compress_available:
            print("Error: SVGCompress package is not installed. Cannot perform SVG compression.")
            return (svg_string,)
            
        # Create a temporary file for the input SVG
        input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 f"temp_input_{time.time()}.svg")
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  f"temp_output_{time.time()}.svg")
        
        try:
            # Write the SVG string to the temporary file
            with open(input_path, 'w', encoding='utf-8') as f:
                f.write(svg_string)
            
            # Setup parameters based on compression_type
            kwargs = {}
            selection_tuple = (selection_criteria, selection_threshold) if pre_select else ('', '')
            
            if compression_type == 'delete':
                kwargs['threshold'] = delete_threshold
                kwargs['criteria'] = delete_criteria
            
            elif compression_type == 'simplify':
                kwargs['epsilon'] = simplify_epsilon
            
            elif compression_type == 'merge':
                kwargs['epsilon'] = merge_epsilon
                kwargs['bufferDistance'] = buffer_distance
                kwargs['operation_key'] = operation_key
            
            # Setup optimization options
            optimize_options = {
                'enable-viewboxing': True,
                'enable-id-stripping': True,
                'enable-comment-stripping': True,
                'shorten-ids': True,
                'indent': 'none'
            } if optimize_after else None
            
            # Perform compression
            compress_by_method(
                filename=input_path,
                compression_type=compression_type,
                curve_fidelity=curve_fidelity,
                outputfile=output_path,
                pre_select=pre_select,
                selection_tuple=selection_tuple,
                optimize=optimize_after,
                optimize_options=optimize_options,
                **kwargs
            )
            
            # Read the output file
            with open(output_path, 'r', encoding='utf-8') as f:
                compressed_svg = f.read()
            
            return (compressed_svg,)
        
        finally:
            # Clean up temporary files
            for file_path in [input_path, output_path]:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {file_path}: {e}")

# Node for SVG optimization using Scour
class SVGScourOptimize:
    """
    SVG optimization using the Scour library.
    Reduces file size by cleaning up markup and removing unnecessary information.
    
    Status: {availability}
    
    Required packages: scour
    Install with: pip install scour
    """.format(availability="Available" if scour_available else "Not Available - Package missing")
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "enable_viewboxing": ("BOOLEAN", {"default": True}),
                "enable_id_stripping": ("BOOLEAN", {"default": True}),
                "enable_comment_stripping": ("BOOLEAN", {"default": True}),
                "shorten_ids": ("BOOLEAN", {"default": True}),
                "indent_type": (["none", "space", "tab"], {"default": "none"}),
            },
            "optional": {
                "strip_xml_prolog": ("BOOLEAN", {"default": True}),
                "strip_xml_space_attribute": ("BOOLEAN", {"default": True}),
                "remove_metadata": ("BOOLEAN", {"default": True}),
                "remove_descriptive_elements": ("BOOLEAN", {"default": True}),
                "strip_ids_prefix": ("STRING", {"default": ""}),
                "simplify_colors": ("BOOLEAN", {"default": True}),
                "precision": ("INT", {"default": 5, "min": 0, "max": 10, "step": 1}),
                "newlines": ("BOOLEAN", {"default": False}),
                "sort_attrs": ("BOOLEAN", {"default": False}),
                "group_create": ("BOOLEAN", {"default": False}),
                "protect_ids_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "optimize_svg"
    CATEGORY = "💎TOSVG/Advanced"

    def optimize_svg(self, svg_string, enable_viewboxing, enable_id_stripping, 
                    enable_comment_stripping, shorten_ids, indent_type,
                    strip_xml_prolog=True, strip_xml_space_attribute=True,
                    remove_metadata=True, remove_descriptive_elements=True,
                    strip_ids_prefix="", simplify_colors=True,
                    precision=5, newlines=False, sort_attrs=False,
                    group_create=False, protect_ids_prefix=""):
        """
        Optimize an SVG string using the Scour library
        """
        # Check if Scour is available
        if not scour_available:
            print("Error: Scour package is not installed. Cannot perform SVG optimization.")
            return (svg_string,)
            
        # Configure Scour options
        options = scour.sanitizeOptions()
        
        # Required parameters
        options.enable_viewboxing = enable_viewboxing
        options.enable_id_stripping = enable_id_stripping
        options.enable_comment_stripping = enable_comment_stripping
        options.shorten_ids = shorten_ids
        options.indent_type = indent_type
        
        # Optional parameters
        options.strip_xml_prolog = strip_xml_prolog
        options.strip_xml_space_attribute = strip_xml_space_attribute
        options.remove_metadata = remove_metadata
        options.remove_descriptive_elements = remove_descriptive_elements
        options.strip_ids_prefix = strip_ids_prefix
        options.simplify_colors = simplify_colors
        options.digits = precision
        options.newlines = newlines
        options.sort_attrs = sort_attrs
        options.group_create = group_create
        options.protect_ids_prefix = protect_ids_prefix
        
        # Perform the optimization
        svg_file = io.StringIO(svg_string)
        optimized_svg = scour.scourString(svg_string, options)
        
        return (optimized_svg,)

# Node for basic quick SVG optimization presets
class SVGOptimizePresets:
    """
    Quick optimization presets for SVG using Scour library.
    Provides common optimization configurations through simple presets.
    
    Status: {availability}
    
    Required packages: scour
    Install with: pip install scour
    """.format(availability="Available" if scour_available else "Not Available - Package missing")
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "preset": (["default", "better", "maximum", "compressed"], {"default": "default"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "optimize_svg_preset"
    CATEGORY = "💎TOSVG/Advanced"

    def optimize_svg_preset(self, svg_string, preset):
        """
        Optimize SVG using predefined presets
        """
        # Check if Scour is available
        if not scour_available:
            print("Error: Scour package is not installed. Cannot perform SVG optimization.")
            return (svg_string,)
            
        # Configure Scour options based on preset
        options = scour.sanitizeOptions()
        
        if preset == "default":
            # Default preset - minimal changes
            pass  # Use default options
            
        elif preset == "better":
            # Better preset - viewboxing for IE compatibility
            options.enable_viewboxing = True
            
        elif preset == "maximum":
            # Maximum preset - aggressive optimization
            options.enable_viewboxing = True
            options.enable_id_stripping = True
            options.enable_comment_stripping = True
            options.shorten_ids = True
            options.indent_type = 'none'
            options.strip_xml_prolog = True
            options.remove_metadata = True
            options.remove_descriptive_elements = True
            options.simplify_colors = True
            
        elif preset == "compressed":
            # Compressed preset - maximum + extra compression
            options.enable_viewboxing = True
            options.enable_id_stripping = True
            options.enable_comment_stripping = True
            options.shorten_ids = True
            options.indent_type = 'none'
            options.strip_xml_prolog = True
            options.remove_metadata = True
            options.remove_descriptive_elements = True
            options.simplify_colors = True
            options.strip_xml_space_attribute = True
            options.remove_unreferenced_ids = True
            options.remove_default_attributes = True
            
        # Perform the optimization
        optimized_svg = scour.scourString(svg_string, options)
        
        return (optimized_svg,)

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "SVGCompressAdvanced": SVGCompressAdvanced,
    "SVGScourOptimize": SVGScourOptimize,
    "SVGOptimizePresets": SVGOptimizePresets,
}

# NODE DISPLAY NAMES 
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVGCompressAdvanced": "Advanced SVG Compression",
    "SVGScourOptimize": "SVG Optimize (Scour)",
    "SVGOptimizePresets": "SVG Optimize Presets",
}
