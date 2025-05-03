# SVG Style Presets

This folder contains style presets for the `SVG_Styler.py` module in ComfyUI.

## Creating Your Own Style Presets

Each style preset is a JSON file that defines rules for styling different elements in your SVG. You can create your own presets by following the structure below.

### Basic Structure

```json
{
  "name": "My Style",
  "description": "Description of what this style does",
  "author": "Your Name",
  "version": "1.0",
  "rules": [
    {
      "selector": "path",
      "attributes": {
        "fill": "#FF0000",
        "stroke": "#000000",
        "stroke-width": "2"
      },
      "styles": {
        "opacity": "0.8"
      }
    }
  ],
  "defs": [
    // SVG definitions like gradients, filters, etc.
  ]
}
```

### Example Style: Neon Glow

```json
{
  "name": "Neon Glow",
  "description": "Vibrant neon colors with glow effects",
  "author": "ComfyUI",
  "version": "1.0",
  "rules": [
    {
      "selector": "path",
      "attributes": {
        "stroke": "#00FFFF",
        "stroke-width": "2",
        "fill": "none",
        "filter": "url(#neonGlow)"
      }
    }
  ],
  "defs": [
    {
      "type": "filter",
      "id": "neonGlow",
      "attributes": {
        "x": "-20%",
        "y": "-20%",
        "width": "140%",
        "height": "140%"
      },
      "children": [
        {
          "type": "feGaussianBlur",
          "attributes": {
            "stdDeviation": "5",
            "result": "blur"
          }
        }
      ]
    }
  ]
}
```

## Structure Explanation

### Required Fields

- `name`: Display name of the style
- `description`: Brief description of what the style does
- `author`: Creator of the style
- `version`: Version information
- `rules`: Array of styling rules to apply

### Rules

Each rule consists of:

- `selector`: CSS-like selector to match elements
  - Simple tag names: `path`, `circle`, `rect`
  - Multiple elements: `path, circle`
  - ID selector: `#myElement`
  - Class selector: `.myClass`
  - All elements: `*`
  - Pseudo-selectors (limited): `path:nth-child(odd)`, `rect:nth-child(3n+1)`

- `attributes`: Object with SVG attributes to set
  - Common attributes: `fill`, `stroke`, `stroke-width`, `opacity`
  - Can include references to definitions: `filter: "url(#myFilter)"`

- `styles`: Optional CSS styles to add via the `style` attribute

### Definitions (`defs`)

SVG definitions for gradients, filters, patterns, etc.:

- `type`: Type of definition (`linearGradient`, `radialGradient`, `filter`, etc.)
- `id`: Unique identifier for the definition
- `attributes`: Attributes for the definition element
- `stops`: For gradients, an array of color stops
- `children`: Child elements (for filters, patterns, etc.)

## Tips for Creating Styles

1. **Start with existing styles**: Use the provided examples as templates
2. **Use the SVGCreateStyle node**: Extract styles from existing SVGs
3. **Test incrementally**: Build your style gradually and test with different SVGs
4. **Combine selectors**: Group similar elements with comma-separated selectors
5. **Use style_strength**: Adjust how strongly styles are applied with the `style_strength` parameter

## Common Selectors

- `path`: Vector paths (most common in SVGs)
- `circle`, `ellipse`: Round shapes
- `rect`: Rectangles
- `polygon`, `polyline`: Multi-point shapes
- `text`: Text elements
- `g`: Groups of elements
- `*`: All elements

Enjoy creating your own SVG styles!