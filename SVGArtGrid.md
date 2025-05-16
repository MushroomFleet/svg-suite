# SVG Art Grid Generator

A ComfyUI custom node that generates artistic SVG grids inspired by Kandinsky-style abstract art.

## What This Node Does

The SVG Art Grid Generator creates visually appealing grids of abstract geometric patterns, using SVG (Scalable Vector Graphics) format. It allows you to:

- Generate consistent geometric patterns arranged in a customizable grid
- Use various block styles like circles, crosses, squares, and more
- Customize colors using pre-defined palettes
- Create standout "big blocks" for visual emphasis
- Get both image output for your ComfyUI workflow and SVG string output for further use

The node produces abstract art that can be used as backgrounds, design elements, or creative starting points for further image generation.

## How This Node Works

The SVG Art Grid Generator works through the following process:

1. **Configuration**: The node takes user inputs for dimensions, grid size, seed, and styling preferences.

2. **Color Selection**: It loads color palettes from a specified URL (defaulting to "nice-color-palettes") or uses fallback palettes if the URL is unreachable.

3. **Grid Generation**: 
   - Calculates grid square size based on width, height, and number of rows/columns
   - Creates an SVG canvas with a subtle gradient background
   - For each cell in the grid, randomly selects a block style and colors from the palette
   - Optionally adds a larger "big block" spanning multiple grid cells

4. **Block Styling**: Each grid cell is filled with one of eight possible block styles:
   - Circle: A circle within a square
   - Opposite Circles: Two circles in opposite corners
   - Cross: Either a plus (+) or an X shape
   - Half Square: A square divided in half
   - Diagonal Square: A square with a diagonal split
   - Quarter Circle: A quarter circle in one corner
   - Dots: A regular pattern of dots
   - Letter Block: A random character or symbol

5. **Rendering**: The SVG is rendered to an image compatible with ComfyUI's image format (tensor) and optionally outputs the SVG string.

## Step-by-Step Instructions for Use

### Basic Setup

1. Add the "SVG Art Grid Generator" node to your ComfyUI workflow.
2. Connect the "image" output to your workflow where an image input is needed.
3. If needed, connect the "svg_string" output to nodes that accept string inputs (ensure "output_svg" is set to True).
4. Adjust parameters as desired (see Parameter Explanations below).
5. Run your workflow to generate the SVG art grid.

### Creating Variations

1. To create different variations with the same settings, simply change the "seed" value.
2. To try different color schemes, adjust the "palette_index" (use -1 for random palettes).
3. For different geometric styles, customize the "block_styles" parameter.

### Saving Your SVG

1. Set "output_svg" to True to get the SVG string.
2. You can use the SVG string output with a "Save Text" node to save as an .svg file.
3. Alternatively, save the image output directly as a standard image format.

## Detailed Parameter Explanations

### Required Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| width | INT | 800 | 100-4096 | Width of the output image in pixels |
| height | INT | 800 | 100-4096 | Height of the output image in pixels |
| rows | INT | 6 | 1-20 | Number of horizontal grid cells |
| cols | INT | 6 | 1-20 | Number of vertical grid cells |
| seed | INT | 0 | 0-4294967295 | Random seed for reproducible results |
| palette_index | INT | -1 | -1 to 100 | Color palette selection (-1 for random) |
| big_block | BOOLEAN | True | - | Whether to include one larger block |
| big_block_size | INT | 2 | 2-3 | Size multiplier for the big block |
| output_svg | BOOLEAN | False | - | Whether to output SVG string in addition to image |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| palette_url | STRING | "https://unpkg.com/nice-color-palettes@3.0.0/100.json" | URL for loading color palettes |
| block_styles | STRING | "all" | Comma-separated list of block styles to use, or "all" |

### Block Styles Available

- `circle`: A filled circle within a square, sometimes with a smaller circle inside
- `opposite_circles`: Two circular segments in opposite corners
- `cross`: Either a plus (+) or an X shape
- `half_square`: A square with one half filled (top, right, bottom, or left)
- `diagonal_square`: A square with a diagonal split
- `quarter_circle`: A quarter circle in one corner of the square
- `dots`: A grid of dots (4x4, 3x3, or 2x2)
- `letter_block`: A character or symbol centered in the square

## Examples

### Example 1: Basic Grid

```
Parameters:
- width: 800
- height: 800
- rows: 6
- cols: 6
- seed: 42
- palette_index: 5
- big_block: False
- block_styles: "all"
```

This creates a uniform 6x6 grid of various block styles using the 6th color palette, without any big blocks.

### Example 2: Minimal Design

```
Parameters:
- width: 1000
- height: 1000
- rows: 10
- cols: 10
- seed: 123
- palette_index: 0
- big_block: True
- big_block_size: 3
- block_styles: "circle,half_square,quarter_circle"
```

This creates a more minimal design with only circular and square-based patterns, and one large 3x3 feature block.

### Example 3: Letter Pattern

```
Parameters:
- width: 600
- height: 600
- rows: 5
- cols: 5
- seed: 789
- palette_index: 12
- big_block: True
- big_block_size: 2
- block_styles: "letter_block"
```

This creates a grid of random letters and symbols with one 2x2 large letter.

### Example 4: Geometric Pattern

```
Parameters:
- width: 1200
- height: 800
- rows: 8
- cols: 5
- seed: 555
- palette_index: -1
- big_block: True
- big_block_size: 2
- block_styles: "cross,diagonal_square,dots"
```

This creates a rectangular grid focused on geometric patterns with random color selection and one larger feature.

## Tips for Best Results

1. **Balanced Grid Size**: For square outputs, use equal values for rows and columns. For rectangular outputs, adjust proportionally.

2. **Custom Block Selection**: Limit block styles for more cohesive designs (e.g., "circle,dots" for a more minimal look).

3. **Color Control**: Experiment with different palette_index values to find pleasing color combinations.

4. **Feature Prominence**: The big_block adds visual interest - try different positions (controlled by seed) and sizes.

5. **SVG Advantage**: When output_svg is enabled, you can use the SVG string in external editors for further customization.

6. **Rendering Note**: For optimal SVG rendering, the node uses cairosvg if available. If not, it falls back to a simpler PIL-based representation.
