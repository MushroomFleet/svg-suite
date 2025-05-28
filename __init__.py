"""
svg-suite: Advanced SVG conversion nodes for ComfyUI
"""

import os
import importlib.util
import sys

# Initialize empty dictionaries for all mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Helper function to safely import a module
def import_module_safely(module_name, file_path):
    try:
        if os.path.exists(file_path):
            # Use importlib to import the module from file path
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Get the mappings if they exist
                class_mappings = getattr(module, 'NODE_CLASS_MAPPINGS', {})
                display_mappings = getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS', {})
                
                # Update our mappings
                NODE_CLASS_MAPPINGS.update(class_mappings)
                NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)
                
                print(f"Successfully imported {module_name} nodes")
                return True
        else:
            print(f"File not found: {file_path}")
            return False
    except Exception as e:
        print(f"Error importing {module_name}: {e}")
        return False

# Get the current directory - using direct path rather than __file__
current_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) if '__file__' in globals() else os.getcwd())

# List of modules to import
modules = [
    ('svg_Advanced', 'svg_Advanced.py'),
    ('SVGCompression', 'SVGCompression.py'),
    ('SVGreColor', 'SVGreColor.py'),
    ('SVGStyler', 'SVGStyler.py'),
    ('SVG_StringEdit', 'SVG_StringEdit.py'),
    ('SVGXMLParser', 'SVGXMLParser.py'),
    ('SVGreColorV2', 'SVGreColorV2.py'),
    ('SVGArtGrid', 'SVGArtGrid.py'),
    ('SVGArtGridV2', 'SVGArtGridV2.py'),
    ('SVGArtGridV3', 'SVGArtGridV3.py'),
    ('SVGArtGridV4', 'SVGArtGridV4.py'),
    ('SVGArtGridDimensionsCalculator', 'SVGArtGridDimensionsCalculator.py'),
    ('ZenkaiSVG_V1', 'ZenkaiSVG_V1.py')
]

# Import each module
for module_name, file_name in modules:
    module_path = os.path.join(current_dir, file_name)
    import_module_safely(module_name, module_path)

# Print the available nodes for debugging
print(f"Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
print(f"Available display names: {list(NODE_DISPLAY_NAME_MAPPINGS.keys())}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
