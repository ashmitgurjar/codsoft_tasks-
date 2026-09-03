import os
import random
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Tuple, Optional


class HandwritingRenderer:
    """Visual Handwriting Document Generator. Renders text as handwritten visual pages."""

    PAPER_STYLES = {
        "ruled": {"bg": (252, 250, 242), "line": (195, 215, 235), "margin_line": (235, 120, 120)},
        "parchment": {"bg": (245, 235, 210), "line": (210, 195, 165), "margin_line": (190, 160, 120)},
        "clean": {"bg": (255, 255, 255), "line": (240, 240, 240), "margin_line": (230, 230, 230)},
        "grid": {"bg": (250, 250, 250), "line": (220, 230, 240), "margin_line": (240, 180, 180)},
    }

    INK_COLORS = {
        "blue": (28, 55, 155),
        "black": (25, 25, 30),
        "red": (175, 35, 35),
        "sepia": (85, 50, 25),
        "green": (25, 115, 60),
    }

    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path

    def _get_font(self, font_size: int) -> ImageFont.ImageFont:
        """Load handwriting font or fallback to system font."""
        if self.font_path and os.path.exists(self.font_path):
            try:
                return ImageFont.truetype(self.font_path, font_size)
            except Exception:
                pass
                
        # Try finding system handwriting/cursive font or standard TrueType font
        candidate_fonts = [
            "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
            "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
            "/System/Library/Fonts/Supplemental/Chalkboard.ttc",
            "/System/Library/Fonts/Supplemental/Snell Roundhand.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
        
        for path in candidate_fonts:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, font_size)
                except Exception:
                    continue
                    
        return ImageFont.load_default()

    def render_page(
        self,
        text: str,
        paper_style: str = "ruled",
        ink_color: str = "blue",
        font_size: int = 24,
        line_spacing: int = 36,
        margin_left: int = 70,
        margin_top: int = 60,
        canvas_width: int = 900,
        canvas_height: int = 1100,
        baseline_jitter: float = 1.5,
        slant_jitter: float = 0.5,
    ) -> Image.Image:
        """
        Render string text into a visual handwritten paper page image.
        
        Args:
            text: Text content to render.
            paper_style: 'ruled', 'parchment', 'clean', or 'grid'.
            ink_color: 'blue', 'black', 'red', 'sepia', or 'green'.
            font_size: Size of handwriting text.
            line_spacing: Distance between lines in pixels.
            margin_left: Left margin offset in pixels.
            margin_top: Top margin offset in pixels.
            canvas_width: Page width in pixels.
            canvas_height: Page height in pixels.
            baseline_jitter: Random vertical variation per character for natural organic look.
            slant_jitter: Random horizontal offset per character.
            
        Returns:
            PIL Image object.
        """
        style = self.PAPER_STYLES.get(paper_style, self.PAPER_STYLES["ruled"])
        ink = self.INK_COLORS.get(ink_color, self.INK_COLORS["blue"])
        
        # Base image canvas
        img = Image.new("RGB", (canvas_width, canvas_height), color=style["bg"])
        draw = ImageDraw.Draw(img)
        
        # Draw paper background ruling lines
        if paper_style == "ruled":
            # Horizontal ruling lines
            for y in range(margin_top + line_spacing, canvas_height - 30, line_spacing):
                draw.line([(0, y), (canvas_width, y)], fill=style["line"], width=1)
            # Red vertical margin line
            draw.line([(margin_left - 15, 0), (margin_left - 15, canvas_height)], fill=style["margin_line"], width=2)
            
        elif paper_style == "grid":
            grid_size = line_spacing
            for x in range(0, canvas_width, grid_size):
                draw.line([(x, 0), (x, canvas_height)], fill=style["line"], width=1)
            for y in range(0, canvas_height, grid_size):
                draw.line([(0, y), (canvas_width, y)], fill=style["line"], width=1)
            draw.line([(margin_left - 15, 0), (margin_left - 15, canvas_height)], fill=style["margin_line"], width=2)

        # Load handwriting font
        font = self._get_font(font_size)
        
        # Word wrapping and character placement
        current_x = margin_left
        current_y = margin_top + font_size // 4
        max_x = canvas_width - 40
        
        words = text.split(" ")
        
        for word_idx, word in enumerate(words):
            # Calculate word width
            try:
                bbox = font.getbbox(word + " ")
                word_w = bbox[2] - bbox[0]
            except Exception:
                word_w = len(word + " ") * (font_size * 0.6)
                
            # Line wrap check
            if current_x + word_w > max_x and current_x > margin_left:
                current_x = margin_left
                current_y += line_spacing
                
            if current_y + line_spacing > canvas_height - 40:
                # Stop rendering if exceeding single page height bounds
                break
                
            # Render character by character with organic handwriting jitter
            for char in word + " ":
                if char == "\n":
                    current_x = margin_left
                    current_y += line_spacing
                    continue
                    
                # Calculate character bounding box
                try:
                    char_bbox = font.getbbox(char)
                    char_w = char_bbox[2] - char_bbox[0]
                except Exception:
                    char_w = font_size * 0.5
                    
                if char_w == 0:
                    char_w = font_size * 0.3
                    
                # Add subtle handwriting stroke variations
                y_offset = random.uniform(-baseline_jitter, baseline_jitter)
                x_offset = random.uniform(-slant_jitter, slant_jitter)
                
                # Organic ink opacity / color micro-fluctuations
                ink_variation = (
                    max(0, min(255, ink[0] + random.randint(-10, 10))),
                    max(0, min(255, ink[1] + random.randint(-10, 10))),
                    max(0, min(255, ink[2] + random.randint(-10, 10))),
                )
                
                draw.text(
                    (current_x + x_offset, current_y + y_offset),
                    char,
                    fill=ink_variation,
                    font=font,
                )
                
                current_x += char_w
                
        return img
