import vtracer
import os
import time
import folder_paths
import numpy as np
from PIL import Image
from typing import List, Tuple
import torch
from io import BytesIO
import fitz
import random

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ConvertRasterToVectorAdvanced:
    """
    Advanced node for converting raster images to SVG with full parameter control.
    Combines both color and binary conversion modes into a single node.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colormode": (["color", "binary"], {"default": "color"}),
                "hierarchical": (["stacked", "cutout"], {"default": "stacked"}),
                "mode": (["spline", "polygon", "none"], {"default": "spline"}),
                "filter_speckle": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1}),
                "color_precision": ("INT", {"default": 6, "min": 0, "max": 10, "step": 1}),
                "layer_difference": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1}),
                "corner_threshold": ("INT", {"default": 60, "min": 0, "max": 180, "step": 1}),
                "length_threshold": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "max_iterations": ("INT", {"default": 10, "min": 1, "max": 70, "step": 1}),
                "splice_threshold": ("INT", {"default": 45, "min": 0, "max": 180, "step": 1}),
                "path_precision": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert_to_svg"
    CATEGORY = "💎TOSVG/Advanced"

    def convert_to_svg(self, image, colormode, hierarchical, mode, filter_speckle, color_precision, 
                       layer_difference, corner_threshold, length_threshold, max_iterations, 
                       splice_threshold, path_precision):
        
        svg_strings = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _image = tensor2pil(i)
            
            if _image.mode != 'RGBA':
                alpha = Image.new('L', _image.size, 255)
                _image.putalpha(alpha)

            pixels = list(_image.getdata())
            size = _image.size

            # Parameters to include depend on colormode
            params = {
                "colormode": colormode,
                "mode": mode,
                "filter_speckle": filter_speckle,
                "corner_threshold": corner_threshold,
                "length_threshold": length_threshold,
                "splice_threshold": splice_threshold,
                "path_precision": path_precision,
            }
            
            # Add color-specific parameters only if colormode is "color"
            if colormode == "color":
                params.update({
                    "hierarchical": hierarchical,
                    "color_precision": color_precision,
                    "layer_difference": layer_difference,
                    "max_iterations": max_iterations,
                })

            svg_str = vtracer.convert_pixels_to_svg(
                pixels,
                size=size,
                **params
            )
            
            svg_strings.append(svg_str)

        return (svg_strings,)

class ConvertImageFileToSVG:
    """
    Convert an image file directly to SVG using the file path.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "", "multiline": False}),
                "colormode": (["color", "binary"], {"default": "color"}),
                "hierarchical": (["stacked", "cutout"], {"default": "stacked"}),
                "mode": (["spline", "polygon", "none"], {"default": "spline"}),
                "filter_speckle": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1}),
                "color_precision": ("INT", {"default": 6, "min": 0, "max": 10, "step": 1}),
                "layer_difference": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1}),
                "corner_threshold": ("INT", {"default": 60, "min": 0, "max": 180, "step": 1}),
                "length_threshold": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "max_iterations": ("INT", {"default": 10, "min": 1, "max": 70, "step": 1}),
                "splice_threshold": ("INT", {"default": 45, "min": 0, "max": 180, "step": 1}),
                "path_precision": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
                "save_output": ("BOOLEAN", {"default": False}),
                "output_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert_file_to_svg"
    CATEGORY = "💎TOSVG/Advanced"

    def convert_file_to_svg(self, image_path, colormode, hierarchical, mode, filter_speckle, color_precision, 
                           layer_difference, corner_threshold, length_threshold, max_iterations, 
                           splice_threshold, path_precision, save_output, output_path):
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found at: {image_path}")
        
        # Parameters to include depend on colormode
        params = {
            "colormode": colormode,
            "mode": mode,
            "filter_speckle": filter_speckle,
            "corner_threshold": corner_threshold,
            "length_threshold": length_threshold,
            "splice_threshold": splice_threshold,
            "path_precision": path_precision,
        }
        
        # Add color-specific parameters only if colormode is "color"
        if colormode == "color":
            params.update({
                "hierarchical": hierarchical,
                "color_precision": color_precision,
                "layer_difference": layer_difference,
                "max_iterations": max_iterations,
            })
        
        # If saving output is requested and an output path is provided
        if save_output and output_path:
            # Use the direct file conversion method
            vtracer.convert_image_to_svg_py(image_path, output_path, **params)
            
            # Read the saved SVG to return its content
            with open(output_path, 'r') as f:
                svg_string = f.read()
        else:
            # Convert using pixels to avoid creating a temporary file
            img = Image.open(image_path).convert('RGBA')
            pixels = list(img.getdata())
            size = img.size
            
            svg_string = vtracer.convert_pixels_to_svg(pixels, size=size, **params)
        
        return (svg_string,)

class ConvertRawBytesToSVG:
    """
    Convert raw image bytes to SVG.
    Useful for integration with API calls or other nodes that produce image bytes.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_bytes": ("STRING", {"forceInput": True}),
                "image_format": (["jpg", "png", "webp"], {"default": "png"}),
                "colormode": (["color", "binary"], {"default": "color"}),
                "hierarchical": (["stacked", "cutout"], {"default": "stacked"}),
                "mode": (["spline", "polygon", "none"], {"default": "spline"}),
                "filter_speckle": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1}),
                "color_precision": ("INT", {"default": 6, "min": 0, "max": 10, "step": 1}),
                "layer_difference": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1}),
                "corner_threshold": ("INT", {"default": 60, "min": 0, "max": 180, "step": 1}),
                "length_threshold": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "max_iterations": ("INT", {"default": 10, "min": 1, "max": 70, "step": 1}),
                "splice_threshold": ("INT", {"default": 45, "min": 0, "max": 180, "step": 1}),
                "path_precision": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert_bytes_to_svg"
    CATEGORY = "💎TOSVG/Advanced"

    def convert_bytes_to_svg(self, image_bytes, image_format, colormode, hierarchical, mode, filter_speckle, 
                            color_precision, layer_difference, corner_threshold, length_threshold, 
                            max_iterations, splice_threshold, path_precision):
        
        # Convert string to bytes if needed
        if isinstance(image_bytes, str):
            image_bytes = image_bytes.encode()
        
        # Parameters to include depend on colormode
        params = {
            "colormode": colormode,
            "mode": mode,
            "filter_speckle": filter_speckle,
            "corner_threshold": corner_threshold,
            "length_threshold": length_threshold,
            "splice_threshold": splice_threshold,
            "path_precision": path_precision,
        }
        
        # Add color-specific parameters only if colormode is "color"
        if colormode == "color":
            params.update({
                "hierarchical": hierarchical,
                "color_precision": color_precision,
                "layer_difference": layer_difference,
                "max_iterations": max_iterations,
            })
        
        try:
            # Use vtracer's direct bytes to SVG conversion
            svg_string = vtracer.convert_raw_image_to_svg(image_bytes, img_format=image_format, **params)
            return (svg_string,)
        except Exception as e:
            # Fallback method if direct bytes conversion fails
            try:
                img = Image.open(BytesIO(image_bytes)).convert('RGBA')
                pixels = list(img.getdata())
                size = img.size
                
                svg_string = vtracer.convert_pixels_to_svg(pixels, size=size, **params)
                return (svg_string,)
            except Exception as inner_e:
                raise ValueError(f"Failed to convert image bytes to SVG: {str(e)}, Fallback error: {str(inner_e)}")

class SVGAdvancedPreview:
    """
    Enhanced preview node for SVG with additional options.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "background_color": ("STRING", {"default": "transparent"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "svg_preview"
    CATEGORY = "💎TOSVG/Advanced"

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz1234567890") for x in range(5))
        self.compress_level = 4

    def save_images(self, images, filename_prefix="SVGAdvancedPreview"):
        """Save images as in the original SaveImage class"""
        filename_prefix += self.prefix_append
        results = []
        
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            metadata = None
            if hasattr(img, 'text') and isinstance(img.text, dict):
                metadata = {**img.text}
                
            filename = f"{filename_prefix}_{self.type}_.png"
            save_path = os.path.join(self.output_dir, filename)
            
            img.save(save_path, compress_level=self.compress_level, pnginfo=metadata)
            results.append({
                "filename": filename,
                "subfolder": self.type,
                "type": self.type,
            })
            
        return {"ui": {"images": results}}

    def svg_preview(self, svg_string, scale=1.0, background_color="transparent"):
        """Generate a preview of the SVG with options for scaling and background color"""
        
        # Create SVG with background color if specified
        if background_color != "transparent":
            # Extract width and height from the SVG
            import re
            width_match = re.search(r'width="(\d+(?:\.\d+)?)', svg_string)
            height_match = re.search(r'height="(\d+(?:\.\d+)?)', svg_string)
            
            if width_match and height_match:
                width = float(width_match.group(1))
                height = float(height_match.group(1))
                
                # Insert a background rectangle as the first element in the SVG
                rect_insert_point = svg_string.find('>', svg_string.find('<svg')) + 1
                bg_rect = f'<rect x="0" y="0" width="{width}" height="{height}" fill="{background_color}"/>'
                svg_string = svg_string[:rect_insert_point] + bg_rect + svg_string[rect_insert_point:]
        
        # Open SVG document with pymupdf
        doc = fitz.open(stream=svg_string.encode('utf-8'), filetype="svg")
        page = doc.load_page(0)
        
        # Apply scaling if not 1.0
        if scale != 1.0:
            matrix = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=matrix)
        else:
            pix = page.get_pixmap()

        # Convert to PIL image
        image_data = pix.tobytes("png")
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")

        # Convert to tensor
        preview = pil2tensor(pil_image)

        # Return the image using the save_images method
        return self.save_images([preview], "SVGAdvancedPreview")

class SaveSVGAdvanced:
    """
    Enhanced save node for SVG with additional options.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {"forceInput": True}),              
                "filename_prefix": ("STRING", {"default": "ComfyUI_SVG_Advanced"}),
            },
            "optional": {
                "append_timestamp": ("BOOLEAN", {"default": True}),
                "custom_output_path": ("STRING", {"default": "", "multiline": False}),
                "optimize_svg": ("BOOLEAN", {"default": False}),
                "minify": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "💎TOSVG/Advanced"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_svg_file"

    def generate_unique_filename(self, prefix, timestamp=False):
        """Generate a unique filename with optional timestamp"""
        if timestamp:
            timestamp_str = time.strftime("%Y%m%d%H%M%S")
            return f"{prefix}_{timestamp_str}.svg"
        else:
            return f"{prefix}.svg"

    def optimize_svg_content(self, svg_content, minify=False):
        """
        Basic SVG optimization:
        - Remove comments
        - Remove unnecessary whitespace if minify is True
        """
        import re
        
        # Remove XML comments
        svg_content = re.sub(r'<!--[\s\S]*?-->', '', svg_content)
        
        if minify:
            # Remove newlines and extra spaces between tags
            svg_content = re.sub(r'>\s+<', '><', svg_content)
            # Remove other unnecessary whitespace
            svg_content = re.sub(r'\s{2,}', ' ', svg_content)
            # Remove space before closing tags
            svg_content = re.sub(r'\s+/>', '/>', svg_content)
        
        return svg_content

    def save_svg_file(self, svg_string, filename_prefix="ComfyUI_SVG_Advanced", 
                      append_timestamp=True, custom_output_path="",
                      optimize_svg=False, minify=False):
        """Save SVG to file with additional options for optimization"""
        
        output_path = custom_output_path if custom_output_path else self.output_dir
        os.makedirs(output_path, exist_ok=True)
        
        unique_filename = self.generate_unique_filename(f"{filename_prefix}", append_timestamp)
        final_filepath = os.path.join(output_path, unique_filename)
        
        # Optimize SVG if requested
        if optimize_svg:
            svg_string = self.optimize_svg_content(svg_string, minify)
        
        with open(final_filepath, "w") as svg_file:
            svg_file.write(svg_string)
        
        ui_info = {"ui": {"saved_svg": unique_filename, "path": final_filepath}}
        return ui_info

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "ConvertRasterToVectorAdvanced": ConvertRasterToVectorAdvanced,
    "ConvertImageFileToSVG": ConvertImageFileToSVG,
    "ConvertRawBytesToSVG": ConvertRawBytesToSVG,
    "SVGAdvancedPreview": SVGAdvancedPreview,
    "SaveSVGAdvanced": SaveSVGAdvanced,
}

# NODE DISPLAY NAMES
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConvertRasterToVectorAdvanced": "Convert to SVG (Advanced)",
    "ConvertImageFileToSVG": "Image File to SVG",
    "ConvertRawBytesToSVG": "Image Bytes to SVG",
    "SVGAdvancedPreview": "SVG Preview (Advanced)",
    "SaveSVGAdvanced": "Save SVG (Advanced)",
}
