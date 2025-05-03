# SVG Suite for ComfyUI

SVG Suite is an advanced set of nodes for converting images to SVG in ComfyUI, expanding upon the functionality of ComfyUI-ToSVG.

## Features

This extension enhances the original SVG conversion capabilities with:

### 1. Advanced Unified Conversion

The `Convert to SVG (Advanced)` node combines both color and binary conversion modes with full parameter control:

- **Unified Interface**: Choose between color and binary modes in a single node
- **Dynamic Parameters**: Parameters adjust automatically based on selected colormode
- **Full Control**: Access to all vtracer parameters for precise SVG generation

### 2. Direct File Processing

The `Image File to SVG` node allows direct conversion from image files:

- **File Path Input**: Process images from anywhere on your system
- **Direct Output**: Optionally save SVG directly to a specified path
- **Parameter Control**: Full control over conversion parameters

### 3. Raw Bytes Processing

The `Image Bytes to SVG` node enables conversion from raw image data:

- **Format Support**: Process JPG, PNG, and WebP image bytes
- **API Integration**: Useful for integration with other nodes that produce image bytes
- **Resilient Processing**: Includes fallback mechanisms for error handling

### 4. Enhanced Preview & Save

- **Advanced Preview**: Scale and add background colors to SVG previews
- **Optimization Options**: Minify and optimize SVG output
- **Custom Paths**: Save to custom locations with flexible naming options

### 5. SVG Compression & Optimization

The suite now includes powerful SVG compression and optimization capabilities:

#### Advanced SVG Compression

The `Advanced SVG Compression` node leverages the SVGCompress library to reduce SVG file size through:

- **Multiple Compression Methods**:
  - **Delete**: Remove tiny polygons below a size threshold
  - **Simplify**: Reduce polygon complexity using the Ramer-Douglas-Peucker algorithm
  - **Merge**: Combine adjacent or overlapping shapes

- **Precision Control**: Fine-tune the compression with parameters like curve fidelity, epsilon values, and buffer distances

#### SVG Optimization with Scour

The `SVG Optimize (Scour)` node provides granular control over SVG optimization:

- **Markup Cleaning**: Remove unnecessary elements, attributes, and metadata
- **ID Management**: Strip or shorten element IDs
- **Formatting Options**: Control indentation, whitespace, and newlines
- **Precision Control**: Adjust decimal precision for coordinates

#### Quick Optimization Presets

The `SVG Optimize Presets` node offers one-click optimization with predefined configurations:

- **Default**: Minimal safe changes
- **Better**: Improved compatibility with older browsers
- **Maximum**: Aggressive optimization for significantly smaller files
- **Compressed**: Maximum optimization plus additional compression techniques

## Usage

1. Add any of the nodes from the "💎TOSVG/Advanced" category to your workflow
2. Connect image data, file paths, or image bytes as required
3. Adjust parameters to control the SVG conversion process
4. Apply compression and optimization as needed
5. Preview and save your SVG output

## Parameters Explained

### Main Conversion Parameters

- **colormode**: Choose between "color" (multi-color SVG) or "binary" (black and white)
- **hierarchical**: For color mode, controls how colors are stacked ("stacked" or "cutout")
- **mode**: Shape mode - "spline" (smooth curves), "polygon" (straight lines), or "none"
- **filter_speckle**: Size threshold to filter small artifacts (higher = more filtering)

### Color Processing (Color Mode Only)

- **color_precision**: Controls color quantization (higher = more colors)
- **layer_difference**: Color difference threshold between layers (higher = fewer layers)
- **max_iterations**: Maximum number of iterations for color processing

### Shape Processing

- **corner_threshold**: Angle threshold for corners (higher = fewer corners)
- **length_threshold**: Length threshold for paths (higher = simpler paths)
- **splice_threshold**: Angle threshold for path splicing
- **path_precision**: Decimal precision for path coordinates (higher = more precise but larger file)

### Compression Parameters

- **compression_type**: Method for reducing SVG complexity ("delete", "simplify", or "merge")
- **curve_fidelity**: Level of detail for curve conversion (higher = more detailed but larger file)
- **pre_select**: Whether to perform pre-selection of polygons before compression
- **selection_criteria**: Method for selecting polygons ("bboxarea" or "circumference")
- **selection_threshold**: Size threshold for polygon selection
- **operation_key**: For merge operations, method to use ("hull" or "union")

### Optimization Parameters

- **enable_viewboxing**: Enable viewBox attribute for better scaling
- **enable_id_stripping**: Remove unnecessary ID attributes
- **enable_comment_stripping**: Remove comments from SVG code
- **shorten_ids**: Shorten ID names to reduce file size
- **indent_type**: Control indentation style ("none", "space", or "tab")
- **simplify_colors**: Convert colors to simpler formats where possible
- **precision**: Number of decimal places for coordinates (lower = smaller file but less precision)

## Compression and Optimization Workflow

For optimal results with complex SVGs:

1. First convert your image to SVG using one of the conversion nodes
2. Apply the `Advanced SVG Compression` node to reduce complexity
3. Further reduce file size with the `SVG Optimize (Scour)` node
4. Preview the result and adjust parameters as needed
5. Save the optimized SVG with the `Save SVG (Advanced)` node

## Comparison with Original ComfyUI-ToSVG

SVG Suite expands upon the original implementation by:

1. Combining separate color and BW nodes into a unified interface
2. Adding direct file processing capabilities
3. Supporting raw image bytes conversion
4. Providing SVG optimization and minification
5. Enhancing preview capabilities with scaling and background options
6. Adding advanced SVG compression and optimization

## Requirements

- ComfyUI
- vtracer
- pymupdf (fitz)
- PIL
- numpy
- torch
- SVGCompress
- scour
- svg.path
- shapely
- rdp
- lxml

## Credits

This extension builds upon the ComfyUI-ToSVG project, enhancing its functionality with additional features and parameters.
