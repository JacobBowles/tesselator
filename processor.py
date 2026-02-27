from PIL import Image, ImageDraw
import math
import numpy as np

class PatternProcessor:
    def __init__(self, image_path, h_tiles):
        self.img = Image.open(image_path).convert("RGBA")
        self.h_tiles = h_tiles
        self.aspect_ratio = self.img.height / self.img.width

    def render(self, vessel, dpi=300):
        mm_to_px = dpi / 25.4
        tile_w = vessel.c_top / self.h_tiles
        tile_h = tile_w * self.aspect_ratio

        # Calculate physical canvas size
        out_w_mm = 2 * vessel.r_top * math.sin(vessel.theta_total / 2)
        out_h_mm = vessel.r_top - (vessel.r_bottom * math.cos(vessel.theta_total / 2))
        
        w_px, h_px = int(out_w_mm * mm_to_px), int(out_h_mm * mm_to_px)
        
        # 1. Create a white background (saving that toner!)
        output_data = np.full((h_px, w_px, 4), 255, dtype=np.uint8) # Fill with White
        
        # 2. Create Coordinate Grids
        x_indices = np.arange(w_px)
        y_indices = np.arange(h_px)
        x_grid, y_grid = np.meshgrid(x_indices, y_indices)
        
        u_mm = x_grid / mm_to_px
        v_mm = y_grid / mm_to_px
        
        # 3. Vectorized Inverse Mapping
        du = u_mm - out_w_mm / 2
        r = np.sqrt(du**2 + (vessel.r_top - v_mm)**2)
        theta = np.arctan2(du, vessel.r_top - v_mm)
        
        # Mask for pixels inside the "Fan"
        mask = (r >= vessel.r_bottom) & (r <= vessel.r_top) & (np.abs(theta) <= vessel.theta_total / 2)
        
        # 4. Map to Source Pixel Coordinates
        sx = ((theta + vessel.theta_total / 2) / vessel.theta_total) * vessel.c_top
        sy = vessel.r_top - r
        
        src_x_px = ((sx % tile_w) / tile_w * self.img.width).astype(int) % self.img.width
        src_y_px = ((sy % tile_h) / tile_h * self.img.height).astype(int) % self.img.height
        
        # 5. Fast Pixel Transfer
        src_data = np.array(self.img)
        output_data[mask] = src_data[src_y_px[mask], src_x_px[mask]]
        
        canvas = Image.fromarray(output_data)
        self._draw_guides(canvas, vessel, out_w_mm, mm_to_px)
        return canvas

    def _draw_guides(self, canvas, vessel, out_w_mm, mm_to_px):
            draw = ImageDraw.Draw(canvas)
            
            # 1. Draw Arcs (Red)
            for r in [vessel.r_top, vessel.r_bottom]:
                points = []
                for t in np.linspace(-vessel.theta_total/2, vessel.theta_total/2, 100):
                    px = (vessel.r_top * np.sin(t) + out_w_mm/2) * mm_to_px
                    py = (vessel.r_top - r * np.cos(t)) * mm_to_px
                    points.append((px, py))
                draw.line(points, fill=(255, 0, 0, 255), width=5)

            # 2. Draw Seams (Blue)
            for t_sign in [-1, 1]:
                t = (vessel.theta_total / 2) * t_sign
                px_top = (vessel.r_top * np.sin(t) + out_w_mm/2) * mm_to_px
                py_top = (vessel.r_top - vessel.r_top * np.cos(t)) * mm_to_px
                px_bot = (vessel.r_top * np.sin(t) + out_w_mm/2) * mm_to_px
                py_bot = (vessel.r_top - vessel.r_bottom * np.cos(t)) * mm_to_px
                draw.line([(px_top, py_top), (px_bot, py_bot)], fill=(0, 0, 255, 255), width=5)

            # 3. Draw 50mm Scale Bar (Black) in the bottom left
            margin = 10 * mm_to_px
            bar_len = 50 * mm_to_px
            bar_y = canvas.height - margin
            draw.line([(margin, bar_y), (margin + bar_len, bar_y)], fill=(0, 0, 0, 255), width=8)
            draw.text((margin, bar_y - 25), "50mm SCALE CHECK", fill=(0, 0, 0, 255))
            points = []
            t = (vessel.theta_total / 2) * t_sign
            for r in [vessel.r_top, vessel.r_bottom]:
                px = (vessel.r_top * np.sin(t) + out_w_mm/2) * mm_to_px
                py = (vessel.r_top - r * np.cos(t)) * mm_to_px
                points.append((px, py))
            draw.line(points, fill=(0, 0, 255, 255), width=5)