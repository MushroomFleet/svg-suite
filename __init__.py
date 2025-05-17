"""
svg-suite: Advanced SVG conversion nodes for ComfyUI
"""

# Initialize empty dictionaries for all mappings
ADVANCED_NODE_MAPPINGS = {}
ADVANCED_DISPLAY_MAPPINGS = {}
COMPRESS_NODE_MAPPINGS = {}
COMPRESS_DISPLAY_MAPPINGS = {}
COLOR_NODE_MAPPINGS = {}
COLOR_DISPLAY_MAPPINGS = {}
COLOR_V2_NODE_MAPPINGS = {}
COLOR_V2_DISPLAY_MAPPINGS = {}
STYLER_NODE_MAPPINGS = {}
STYLER_DISPLAY_MAPPINGS = {}
STRING_EDIT_NODE_MAPPINGS = {}
STRING_EDIT_DISPLAY_MAPPINGS = {}
XML_PARSER_NODE_MAPPINGS = {}
XML_PARSER_DISPLAY_MAPPINGS = {}
ARTGRID_NODE_MAPPINGS =  {}
ARTGRID_DISPLAY_MAPPINGS = {}
ARTGRID_V2_NODE_MAPPINGS =  {}
ARTGRID_V2_DISPLAY_MAPPINGS = {}

# Try to import each module separately
try:
    from .svg_Advanced import NODE_CLASS_MAPPINGS as ADVANCED_NODE_MAPPINGS
    from .svg_Advanced import NODE_DISPLAY_NAME_MAPPINGS as ADVANCED_DISPLAY_MAPPINGS
    print("Successfully imported svg_Advanced nodes")
except Exception as e:
    print(f"Error importing svg_Advanced: {e}")

try:
    from .SVGCompression import NODE_CLASS_MAPPINGS as COMPRESS_NODE_MAPPINGS
    from .SVGCompression import NODE_DISPLAY_NAME_MAPPINGS as COMPRESS_DISPLAY_MAPPINGS
    print("Successfully imported SVGCompression nodes")
except Exception as e:
    print(f"Error importing SVGCompression: {e}")

try:
    from .SVGreColor import NODE_CLASS_MAPPINGS as COLOR_NODE_MAPPINGS
    from .SVGreColor import NODE_DISPLAY_NAME_MAPPINGS as COLOR_DISPLAY_MAPPINGS
    print("Successfully imported SVGreColor nodes")
except Exception as e:
    print(f"Error importing SVGreColor: {e}")

try:
    from .SVGStyler import NODE_CLASS_MAPPINGS as STYLER_NODE_MAPPINGS
    from .SVGStyler import NODE_DISPLAY_NAME_MAPPINGS as STYLER_DISPLAY_MAPPINGS
    print("Successfully imported SVGStyler nodes")
except Exception as e:
    print(f"Error importing SVGStyler: {e}")

try:
    from .SVG_StringEdit import NODE_CLASS_MAPPINGS as STRING_EDIT_NODE_MAPPINGS
    from .SVG_StringEdit import NODE_DISPLAY_NAME_MAPPINGS as STRING_EDIT_DISPLAY_MAPPINGS
    print("Successfully imported SVG_StringEdit nodes")
except Exception as e:
    print(f"Error importing SVG_StringEdit: {e}")

try:
    from .SVGXMLParser import NODE_CLASS_MAPPINGS as XML_PARSER_NODE_MAPPINGS
    from .SVGXMLParser import NODE_DISPLAY_NAME_MAPPINGS as XML_PARSER_DISPLAY_MAPPINGS
    print("Successfully imported SVGXMLParser nodes")
except Exception as e:
    print(f"Error importing SVGXMLParser: {e}")

try:
    from .SVGreColorV2 import NODE_CLASS_MAPPINGS as COLOR_V2_NODE_MAPPINGS
    from .SVGreColorV2 import NODE_DISPLAY_NAME_MAPPINGS as COLOR_V2_DISPLAY_MAPPINGS
    print("Successfully imported SVGreColorV2 nodes")
except Exception as e:
    print(f"Error importing SVGreColorV2: {e}")

try:
    from .SVGArtGrid import NODE_CLASS_MAPPINGS as ARTGRID_NODE_MAPPINGS
    from .SVGArtGrid import NODE_DISPLAY_NAME_MAPPINGS as ARTGRID_DISPLAY_MAPPINGS
    print("Successfully imported SVGArtGrid nodes")
except Exception as e:
    print(f"Error importing SVGArtGrid: {e}")

try:
    from .SVGArtGridV2 import NODE_CLASS_MAPPINGS as ARTGRID_V2_NODE_MAPPINGS
    from .SVGArtGridV2 import NODE_DISPLAY_NAME_MAPPINGS as ARTGRID_V2_DISPLAY_MAPPINGS
    print("Successfully imported SVGArtGridV2 nodes")
except Exception as e:
    print(f"Error importing SVGArtGridV2: {e}")


# Merge the node mappings
NODE_CLASS_MAPPINGS = {
    **ADVANCED_NODE_MAPPINGS, 
    **COMPRESS_NODE_MAPPINGS, 
    **COLOR_NODE_MAPPINGS,
    **COLOR_V2_NODE_MAPPINGS,
    **STYLER_NODE_MAPPINGS,
    **STRING_EDIT_NODE_MAPPINGS,
    **XML_PARSER_NODE_MAPPINGS,
    **ARTGRID_NODE_MAPPINGS,
    **ARTGRID_V2_NODE_MAPPINGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **ADVANCED_DISPLAY_MAPPINGS, 
    **COMPRESS_DISPLAY_MAPPINGS, 
    **COLOR_DISPLAY_MAPPINGS,
    **COLOR_V2_DISPLAY_MAPPINGS,
    **STYLER_DISPLAY_MAPPINGS,
    **STRING_EDIT_DISPLAY_MAPPINGS,
    **XML_PARSER_DISPLAY_MAPPINGS,
    **ARTGRID_DISPLAY_MAPPINGS,
    **ARTGRID_V2_DISPLAY_MAPPINGS
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
