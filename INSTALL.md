# SVG Suite Installation Guide

This guide will help you install and set up the SVG Suite extension for ComfyUI.

## Prerequisites

Before installing SVG Suite, ensure you have:

1. A working installation of ComfyUI
2. Python 3.8 or newer
3. Git (optional, for cloning the repository)

## Installation Methods

### Method 1: Manual Installation

1. Download this repository either as a ZIP file or by cloning it:
   ```
   git clone https://github.com/your-username/svg-suite.git
   ```

2. Place the entire `svg-suite` folder in your ComfyUI's `custom_nodes` directory:
   ```
   path/to/ComfyUI/custom_nodes/
   ```

3. Install the required dependencies:
   ```
   pip install -r path/to/ComfyUI/custom_nodes/svg-suite/requirements.txt
   ```

4. Restart ComfyUI

### Method 2: Installation via ComfyUI Manager

If you have [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed:

1. Open ComfyUI
2. Open the Manager panel
3. Search for "SVG Suite"
4. Click "Install"
5. Restart ComfyUI

## Verifying Installation

After installation:

1. Start or restart ComfyUI
2. Check the node browser for the new "💎TOSVG/Advanced" category
3. You should see all the new SVG Suite nodes:
   - Convert to SVG (Advanced)
   - Image File to SVG
   - Image Bytes to SVG
   - SVG Preview (Advanced)
   - Save SVG (Advanced)
   - Advanced SVG Compression
   - SVG Optimize (Scour)
   - SVG Optimize Presets

## Troubleshooting

If you encounter issues:

### Missing Dependencies

If you see error messages about missing modules:

For basic SVG conversion:
```
pip install vtracer PyMuPDF pillow numpy torch
```

For SVG compression and optimization:
```
pip install SVGCompress scour svg.path shapely rdp lxml
```

### Node Not Appearing

If the nodes don't appear in ComfyUI:

1. Check that the `svg-suite` folder is correctly placed in the `custom_nodes` directory
2. Verify that the `__init__.py` file exists in the `svg-suite` folder
3. Restart ComfyUI completely (not just refreshing the browser)
4. Check the ComfyUI console/terminal for any error messages

### SVG Conversion Issues

If SVG conversion fails:

1. Ensure your image is in a supported format (PNG, JPG, WebP)
2. Try the binary mode which is faster and more reliable for line art
3. Adjust the conversion parameters, particularly reducing `filter_speckle` if details are being lost

### SVG Compression Issues

If SVG compression fails:

1. Try using the SVG Optimize Presets node with the "default" or "better" preset first
2. For SVGCompress operations, start with the "simplify" compression type before trying "merge"
3. If using pre-selection, ensure your threshold values aren't too restrictive
4. For complex SVGs, you may need to increase curve_fidelity

## Dependencies

SVG Suite depends on:

### Core Dependencies
- vtracer: Core library for image to SVG conversion
- PyMuPDF (fitz): For SVG preview generation
- PIL (Pillow): For image processing
- numpy: For array operations
- torch: For tensor operations with ComfyUI

### SVG Compression Dependencies
- SVGCompress: For advanced SVG shape simplification and compression
- scour: For SVG optimization and cleaning
- svg.path: For SVG path parsing and manipulation
- shapely: For geometric operations on SVG shapes
- rdp: Implementation of the Ramer-Douglas-Peucker algorithm
- lxml: For XML/SVG parsing and manipulation

## Updating

To update to the latest version:

1. Delete the existing `svg-suite` folder from your `custom_nodes` directory
2. Download or clone the latest version
3. Follow the installation steps again
