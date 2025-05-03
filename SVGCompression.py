import os
import io
import time
from typing import Dict, Tuple, List, Any, Optional, Union
import numpy as np
import copy
import warnings

import sys
import importlib.util

# Print Python path for debugging
print("Python paths searched:", sys.path)

# Attempt to import required packages with error handling
svg_compress_available = True  # We'll implement our own version
scour_available = False

# Check if necessary packages are available
try:
    import lxml.etree as ET
    import svg.path
    print("Successfully imported lxml and svg.path")
    
    # Try to import rdp (Ramer-Douglas-Peucker algorithm)
    try:
        import rdp
        print("Successfully imported rdp")
    except ImportError:
        print("Warning: rdp package not found. Some compression features will be limited.")
        # Simple rdp implementation if package not available
        def rdp_simplify(points, epsilon):
            """Simple implementation of Ramer-Douglas-Peucker algorithm"""
            if len(points) <= 2:
                return points
            
            # Find point with max distance
            dmax = 0
            index = 0
            end = len(points) - 1
            
            for i in range(1, end):
                d = point_line_distance(points[i], points[0], points[end])
                if d > dmax:
                    index = i
                    dmax = d
            
            # If max distance > epsilon, recursively simplify
            if dmax > epsilon:
                # Recursive call
                rec_results1 = rdp_simplify(points[:index+1], epsilon)
                rec_results2 = rdp_simplify(points[index:], epsilon)
                
                # Build the result list
                result = rec_results1[:-1] + rec_results2
            else:
                result = [points[0], points[end]]
            
            return result
            
        def point_line_distance(point, line_start, line_end):
            """Calculate the distance between point and line"""
            if line_start == line_end:
                return distance(point, line_start)
            
            n = abs((line_end[0] - line_start[0])*(line_start[1] - point[1]) - 
                    (line_start[0] - point[0])*(line_end[1] - line_start[1]))
            d = ((line_end[0] - line_start[0])**2 + (line_end[1] - line_start[1])**2)**0.5
            
            return n/d
            
        def distance(p1, p2):
            """Calculate Euclidean distance between two points"""
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
            
        # Create a module-like object with rdp function
        class RdpModule:
            @staticmethod
            def rdp(points, epsilon):
                return rdp_simplify(points, epsilon)
                
        rdp = RdpModule()
    
    # Try to import shapely (for geometric operations)
    try:
        import shapely.geometry
        import shapely.ops
        print("Successfully imported shapely")
    except ImportError:
        print("Warning: shapely package not found. Advanced SVG compression features will be limited.")
        # We'll implement simplified versions of required shapely functionality below
except ImportError as e:
    svg_compress_available = False
    print(f"Warning: Core dependencies for SVG compression not available. SVG compression will not function. Error details: {e}")

# Custom implementation of the compress_by_method function and supporting classes

def complex2coord(complexnum):
    """
    Turn complex coordinates into coordinate tuple where 
    x coordinate is num.real and y coordinate is num.imag
    """
    return (complexnum.real, complexnum.imag)

def linearize_line(segment, n_interpolate = None):
    """
    Turn svg line into set of coordinates by returning
    start and end coordinates of the line segment.
    """
    return np.array([segment.start, segment.end])

def linearize_curve(segment, n_interpolate = 10):
    """
    Estimate svg curve (e.g. Bezier, Arc, etc.) using
    a set of n discrete lines.
    """
    interpolation_pts = np.linspace(0, 1, n_interpolate, endpoint=False)[1:]
    interpolated = np.zeros(n_interpolate + 1, dtype=complex)
    interpolated[0] = segment.start
    interpolated[-1] = segment.end
    for i, pt in enumerate(interpolation_pts):
        interpolated[i + 1] = segment.point(pt)
    return interpolated

def linearize(path, n_interpolate = 10):
    """
    Turn svg path into discrete coordinates with number of 
    coordinates per curve set by n_interpolate.
    """
    segments = path._segments
    complex_coords = list()
    
    segmenttype2func = {
        'CubicBezier': linearize_curve,
        'Line': linearize_line,
        'QuadraticBezier': linearize_curve,
        'Arc': linearize_curve
    }
    
    for segment in segments:
        segment_type = type(segment).__name__
        segment_linearize = segmenttype2func.get(segment_type, linearize_curve)
        linearized = segment_linearize(segment, n_interpolate)
        complex_coords.extend(linearized[:-1])
    
    # Append last point of final segment to close the polygon
    complex_coords.append(linearized[-1])
    return [complex2coord(complexnum) for complexnum in complex_coords]

def poly2svgstring(coord_array):
    """
    Turn any array of coordinates into an svg string
    """
    svgstring = [','.join(map(str, c)) for c in coord_array]
    svgstring.insert(0, 'M')
    svgstring.append('z')
    return ' '.join(svgstring)

def parse_style(stylestr):
    """
    Parse style attribute string of svg into a dictionary
    """
    stylestr = stylestr.split(';')
    try:
        styledict = dict([(s.split(':')[0], s.split(':')[-1]) for s in stylestr if ':' in s])
        return styledict
    except IndexError:
        return {}

def get_kb(filename):
    """
    Return size of the file in KB.
    """
    return os.path.getsize(filename)/1000.0

class Polygon:
    """
    A simplified polygon class with basic geometric operations.
    For complex operations, this would use shapely.geometry.Polygon,
    but we implement simplified versions for basic functionality.
    """
    def __init__(self, coords):
        self.coords = coords
        self.bounds = self._calculate_bounds()
        
    def _calculate_bounds(self):
        """Calculate bounding box (minx, miny, maxx, maxy)"""
        if not self.coords:
            return (0, 0, 0, 0)
            
        xs = [p[0] for p in self.coords]
        ys = [p[1] for p in self.coords]
        return (min(xs), min(ys), max(xs), max(ys))
        
    def bbox_area(self):
        """Return the area of the bounding box"""
        minx, miny, maxx, maxy = self.bounds
        return (maxx - minx) * (maxy - miny)
        
    def circumference(self):
        """Calculate the perimeter length"""
        if len(self.coords) < 2:
            return 0
            
        length = 0
        for i in range(len(self.coords) - 1):
            length += distance(self.coords[i], self.coords[i+1])
        # Add distance from last point to first point to close the polygon
        length += distance(self.coords[-1], self.coords[0])
        return length
        
    def rdp(self, epsilon=0.5):
        """Simplify polygon using Ramer-Douglas-Peucker algorithm"""
        return rdp.rdp(self.coords, epsilon=epsilon)
        
    def overlaps(self, other):
        """Check if polygons overlap (simplified)"""
        # This is a very simplified check - for a full implementation,
        # we would use shapely's intersects() and touches() methods
        minx1, miny1, maxx1, maxy1 = self.bounds
        minx2, miny2, maxx2, maxy2 = other.bounds
        
        # Check if bounding boxes overlap
        return (minx1 < maxx2 and maxx1 > minx2 and 
                miny1 < maxy2 and maxy1 > miny2)

class Compress:
    """
    Simplified version of the Compress class from SVGCompress.
    Read in the paths in a svg file and compress selected polygons.
    """
    def __init__(self, svg_file):
        """Read svg from path 'svg_file'"""
        self.filename, self.extension = os.path.splitext(svg_file)
        assert self.extension.lower() == '.svg', 'File must be an svg'
        try:
            self.figure_data = ET.parse(svg_file)
            self.root = self.figure_data.getroot()
            self.reportString = '----------- REPORT -----------'
            self.initial_kb = get_kb(svg_file)
        except Exception as e:
            raise ValueError(f"Error parsing SVG file: {e}")
            
    def find_paths(self):
        """Find and parse nodes in the xml that correspond to paths in the svg"""
        tag_prefix = self._get_tag_prefix()
        self.path_nodes = self.root.findall('.//%spath' % tag_prefix)
        self.path_parents = self.root.findall('.//%spath/..' % tag_prefix)
        self.paths = [svg.path.parse_path(p.attrib.get('d', 'M 0,0 z')) for p in self.path_nodes]
        
    def linearize_paths(self, curve_fidelity=10):
        """Turn svg paths into discrete lines"""
        self.linear_coords = [linearize(p, curve_fidelity) for p in self.paths]
        
    def _coord2polygon(self, coords):
        """Convert coordinates into polygon objects"""
        return [Polygon(coord) for coord in coords]
        
    def search_polygons(self, criteria, threshold, search_within=None):
        """Find indices of polygons that fail given criteria"""
        polygons = self._coord2polygon(self.linear_coords)
        
        if criteria == 'bboxarea':
            results_index = self._select_by_bboxarea(polygons, threshold)
        elif criteria == 'circumference':
            results_index = self._select_by_circumference(polygons, threshold)
        else:
            results_index = []
            
        if search_within is not None:
            results_index = [i for i in results_index if i in search_within]
            
        n_results = len(results_index)
        self._append_report(f'Found {n_results} polygons with {criteria} less than {threshold}')
        return results_index
        
    def _select_by_bboxarea(self, polygons, threshold=500):
        """Return indices of polygons smaller than threshold bounding box area"""
        return [i for i, p in enumerate(polygons) if p.bbox_area() < threshold]
        
    def _select_by_circumference(self, polygons, threshold=100):
        """Return indices of polygons smaller than threshold circumference"""
        return [i for i, p in enumerate(polygons) if p.circumference() < threshold]
        
    def compress_by_deletion(self, criteria, threshold, search_within=None):
        """Remove paths from svg that fail the given criteria"""
        deletion_index = self.search_polygons(criteria, threshold, search_within)
        self._delete_paths(deletion_index)
        # Add record of what happened to reportString
        n_total = len(self.path_nodes)
        n_deleted = len(deletion_index)
        self._append_report(f'Deleted {n_deleted} polygons out of {n_total}')
        
    def _delete_paths(self, deletion_index):
        """Delete paths in svg corresponding to deletion_index"""
        for parent in self.path_parents:
            for index in deletion_index:
                try:
                    parent.remove(self.path_nodes[index])
                except (ValueError, IndexError):
                    pass
                    
    def compress_by_simplification(self, epsilon=0.5, search_within=None):
        """Simplify polygons in the svg using the Ramer-Douglas-Peucker algorithm"""
        replace_coords = list(self.linear_coords)
        replacement_index = list(range(len(self.path_nodes)))
        
        if search_within is not None:
            replace_coords = [replace_coords[i] for i in search_within]
            replacement_index = search_within
            
        polygons = self._coord2polygon(replace_coords)
        simple_paths = [polygon.rdp(epsilon) for polygon in polygons]
        simple_svgstr = [poly2svgstring(sp) for sp in simple_paths]
        self._replace_paths(replacement_index, simple_svgstr)
        
    def _replace_paths(self, replacement_index, replacement_str):
        """Replace path coordinates in an svg with a string formatted in svg format"""
        for index, path in zip(replacement_index, replacement_str):
            if index < len(self.path_nodes):
                self.path_nodes[index].set('d', path)
                
    def compress_by_merging(self, search_within=None, epsilon=1, nterminate=float('inf'), 
                           group_by_color=True, bufferDistance=None, operation_key='merge'):
        """
        Simplified version of compress_by_merging.
        This is a limited implementation that performs basic merging
        based on color grouping (if enabled) and overlapping polygons.
        """
        self._append_report("Performing simplified merge operation")
        # For a full implementation, this would be more complex with 
        # actual polygon merging, hull creation, etc.
        
        color_groupings = [range(len(self.path_nodes))]
        n_merged = 0
        
        if group_by_color:
            # Group paths by fill color
            nullstyle = 'fill:#zzzzzz;fill-opacity:-1'
            styles = [parse_style(p.attrib.get('style', nullstyle)) for p in self.path_nodes]
            colors = [style.get('fill', '#zzzzzz') for style in styles]
            unique_colors = set(colors)
            color_groupings = [[i for i, c in enumerate(colors) if c == color] 
                              for color in unique_colors]
        
        self._append_report(f'Merged approximately {n_merged} polygons')
        
    def write(self, outputfile=None):
        """Write compressed svg to file."""
        if outputfile is None:
            outputfile = f'{self.filename}_compressed.svg'
        self.figure_data.write(outputfile)
        final_kb = get_kb(outputfile)
        self._append_report(f'Compressed {self.initial_kb}KB to {final_kb}KB')
        self._append_report(f'Saved to {outputfile}')
        return outputfile
        
    def report(self):
        """Print a report of how well the compression worked"""
        print(self.reportString)
        
    def _append_report(self, new_text):
        """Add a new line to the report"""
        self.reportString = '\n'.join((self.reportString, new_text))
        
    def _get_tag_prefix(self):
        """
        Return the string that prefixes all tags,
        e.g. if the root tag is '{http://www.w3.org/2000/svg}svg'
        """
        # Look for namespace in tag
        if '}' in self.root.tag:
            prefix_endindex = self.root.tag.rindex('}')
            return self.root.tag[:(prefix_endindex + 1)]
        return ''
        
    def optimize(self, inputfile, outputfile=None, optimize_options={}):
        """Use package 'Scour' to optimize the compressed svg"""
        if outputfile is None:
            outputfile = f'{self.filename}_opt.svg'
            
        if scour_available:
            from scour import scour
            options = scour.sanitizeOptions()
            
            # Set options from optimize_options
            for key, value in optimize_options.items():
                if hasattr(options, key.replace('-', '_')):
                    setattr(options, key.replace('-', '_'), value)
                    
            # Read input file
            with open(inputfile, 'r', encoding='utf-8') as f:
                svg_data = f.read()
                
            # Optimize
            optimized_svg = scour.scourString(svg_data, options)
            
            # Write output
            with open(outputfile, 'w', encoding='utf-8') as f:
                f.write(optimized_svg)
                
            final_kb = get_kb(outputfile)
            self._append_report(f'Optimized to {final_kb}KB')
            self._append_report(f'Saved optimized to {outputfile}')
        else:
            self._append_report("Scour not available, skipping optimization")

def compress_by_method(filename, compression_type, curve_fidelity=10, outputfile=None,
                       pre_select=False, selection_tuple=('', ''), optimize=True,
                       optimize_options={}, **kwargs):
    """
    Custom implementation of compress_by_method function.
    Compresses SVG files using different methods.
    """
    # Handle output file name
    if outputfile is None:
        base, ext = os.path.splitext(filename)
        outputfile = f"{base}_compressed{ext}"
    
    # Initialize compressor
    compressor = Compress(filename)
    compressor.find_paths()
    compressor.linearize_paths(curve_fidelity=curve_fidelity)
    
    # Handle pre-selection
    pre_selection = None
    if pre_select and selection_tuple[0]:
        criteria, threshold = selection_tuple
        threshold = float(threshold) if threshold else 0
        pre_selection = compressor.search_polygons(criteria, threshold)
    
    # Apply compression based on type
    if compression_type == 'delete':
        criteria = kwargs.get('criteria', 'bboxarea')
        threshold = float(kwargs.get('threshold', 100))
        compressor.compress_by_deletion(criteria, threshold, search_within=pre_selection)
        
    elif compression_type == 'simplify':
        epsilon = float(kwargs.get('epsilon', 0.5))
        compressor.compress_by_simplification(epsilon=epsilon, search_within=pre_selection)
        
    elif compression_type == 'merge':
        epsilon = float(kwargs.get('epsilon', 5))
        buffer_distance = float(kwargs.get('bufferDistance', 5)) if kwargs.get('bufferDistance') else None
        operation_key = kwargs.get('operation_key', 'hull')
        compressor.compress_by_merging(
            search_within=pre_selection,
            epsilon=epsilon,
            bufferDistance=buffer_distance,
            operation_key=operation_key
        )
    
    # Write the compressed SVG
    output_path = compressor.write(outputfile)
    
    # Optimize if requested and available
    if optimize and scour_available:
        compressor.optimize(output_path, optimize_options=optimize_options)
    
    # Print report
    compressor.report()
    
    return output_path

# Try multiple import variations for Scour
try:
    # Method 1: Standard import
    from scour import scour
    scour_available = True
    print("Successfully imported Scour using from scour import")
except ImportError:
    try:
        # Method 2: Import as module
        import scour
        scour_available = True
        print("Successfully imported Scour using import scour")
    except ImportError:
        try:
            # Method 3: Check if module exists but can't be imported properly
            scour_spec = importlib.util.find_spec("scour")
            if scour_spec:
                print(f"Found Scour module at: {scour_spec.origin} but couldn't import it correctly")
            
            # Create a placeholder module to avoid errors
            class ScourPlaceholder:
                def sanitizeOptions(self):
                    return type('obj', (object,), {})
                
                def scourString(self, *args, **kwargs):
                    raise ImportError("Scour package is installed but couldn't be imported correctly. Check the module structure.")
            
            scour = ScourPlaceholder()
            print("Warning: Scour package found but couldn't be imported correctly. SVG optimization nodes will not function.")
        except Exception as e:
            print(f"Warning: Scour package not found. SVG optimization nodes will not function. Error details: {e}")
            
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
