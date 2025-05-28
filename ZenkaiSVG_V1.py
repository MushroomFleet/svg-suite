import os
import random

class ZenkaiSVG_V1:
    @classmethod
    def INPUT_TYPES(cls):
        # Find ComfyUI root directory by looking for main.py
        current_dir = os.path.dirname(__file__)
        comfyui_root = current_dir
        for _ in range(5):
            parent = os.path.dirname(comfyui_root)
            if os.path.exists(os.path.join(parent, 'main.py')):
                comfyui_root = parent
                break
            comfyui_root = parent
            
        svgstore_folder = os.path.join(comfyui_root, 'svgstore')
        
        # Get subdirectories in svgstore folder
        subdirectories = []
        if os.path.exists(svgstore_folder):
            subdirectories = [d for d in os.listdir(svgstore_folder) 
                            if os.path.isdir(os.path.join(svgstore_folder, d))]
        
        if not subdirectories:
            subdirectories = ["No subdirectories found"]
        
        return {
            "required": {
                "subfolder": (subdirectories,),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF
                }),
                "num_svgs": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "mode": (["sequential", "random"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_svg"
    CATEGORY = "DJZ-Nodes"

    def load_svg(self, subfolder, seed, num_svgs, mode="sequential"):
        # Find ComfyUI root directory
        current_dir = os.path.dirname(__file__)
        comfyui_root = current_dir
        for _ in range(5):
            parent = os.path.dirname(comfyui_root)
            if os.path.exists(os.path.join(parent, 'main.py')):
                comfyui_root = parent
                break
            comfyui_root = parent
            
        svgstore_folder = os.path.join(comfyui_root, 'svgstore')
        subfolder_path = os.path.join(svgstore_folder, subfolder)

        if not os.path.exists(subfolder_path):
            return (["SVG subfolder not found."],)

        # Get all SVG files in the subfolder
        svg_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.svg')]
        
        if not svg_files:
            return (["No SVG files found in the selected subfolder."],)
            
        total_files = len(svg_files)
        
        if mode == "sequential":
            # Sequential mode: Directly map seed to file index with looping
            selected_files = []
            for i in range(num_svgs):
                # Calculate file index based on seed, looping if needed
                file_index = (seed + i) % total_files
                selected_files.append(svg_files[file_index])
        else:
            # Random mode: Use seed for reproducible randomness
            random.seed(seed)
            selected_files = random.sample(svg_files, min(num_svgs, total_files))

        # Load SVG content
        svg_contents = []
        for svg_file in selected_files:
            file_path = os.path.join(subfolder_path, svg_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    svg_content = f.read()
                    svg_contents.append(svg_content)
            except Exception as e:
                svg_contents.append(f"Error loading {svg_file}: {str(e)}")

        return (svg_contents,)

    @classmethod
    def IS_CHANGED(cls, subfolder, seed, num_svgs, mode):
        return float(seed)  # This ensures the node updates when the seed changes

NODE_CLASS_MAPPINGS = {
    "ZenkaiSVG_V1": ZenkaiSVG_V1
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZenkaiSVG_V1": "Zenkai SVG V1"
}