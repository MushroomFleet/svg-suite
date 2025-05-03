# SVG Theme Files

This directory contains JSON theme files for use with the SVGThemeColorizer node. Each file defines a collection of colors that can be applied to SVG images to transform their appearance.

## Theme File Structure

Theme files must be valid JSON with the following structure:

```json
{
  "name": "Theme Name",
  "description": "A short description of the theme",
  "colors": [
    "#HEXCOLOR1",
    "#HEXCOLOR2",
    "#HEXCOLOR3",
    ...
  ]
}
```

- **name**: A human-readable name for the theme
- **description**: A brief description of the theme's appearance or purpose
- **colors**: An array of hexadecimal color codes that define the palette

## Creating Your Own Themes

To create a custom theme:

1. Create a new `.json` file in this directory (e.g., `my_theme.json`)
2. Follow the structure above, defining your theme name, description, and an array of colors
3. Save the file - the theme will automatically appear in the SVGThemeColorizer dropdown menu

## How Colors Are Applied

When a theme is applied to an SVG, the SVGThemeColorizer node:

1. Extracts all fill colors from the SVG
2. Finds the closest matching color in the theme palette for each original color
3. Replaces the original colors with the corresponding theme colors

The color matching uses LAB color space for perceptually accurate matching.
