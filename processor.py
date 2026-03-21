from PIL import Image, ImageDraw
import math
import numpy as np


class PatternProcessor:
    def __init__(self, image_path, h_tiles):
        self.img = Image.open(image_path).convert("RGBA")
        self.h_tiles = h_tiles
        self.aspect_ratio = self.img.height / self.img.width

    # ------------------------------------------------------------------
    # Shared vertical accumulation
    # ------------------------------------------------------------------

    def _build_F(self, vessel):
        """F(s) = integral_0^s  N / (c(s') * aspect_ratio)  ds'"""
        n_fine = 10000
        s_fine = np.linspace(0.0, vessel.s_total, n_fine)
        c_fine = vessel.get_circumference_at(s_fine)
        integrand = self.h_tiles / (c_fine * self.aspect_ratio)
        ds = vessel.s_total / (n_fine - 1)
        F_fine = np.concatenate(([0.0], np.cumsum(
            0.5 * (integrand[:-1] + integrand[1:]) * ds
        )))
        return s_fine, F_fine

    # ------------------------------------------------------------------
    # Top segment geometry — drives all fan layout
    # ------------------------------------------------------------------

    def _top_seg_geometry(self, vessel):
        """
        The fan is defined by the TOP segment of the vessel only.
        This matches the stamp approach: we align top trapezoid edges.

        Returns (r_top_seg, r_bottom_seg, theta_seg, theta_step, slice_w_mm)
        """
        s0 = float(vessel._arc_lengths[0])
        s1 = float(vessel._arc_lengths[1])
        c0 = float(vessel.get_circumference_at(s0))
        c1 = float(vessel.get_circumference_at(s1))
        h  = s1 - s0

        if abs(c0 - c1) < 1e-6:
            # Cylinder — use a very large radius
            r_top_seg = 1e9
        else:
            r_top_seg = (c0 * h) / (c0 - c1)

        r_bottom_seg = r_top_seg - vessel.s_total
        theta_seg    = c0 / r_top_seg
        theta_step   = theta_seg / self.h_tiles
        slice_w_mm   = c0 / self.h_tiles

        return r_top_seg, r_bottom_seg, theta_seg, theta_step, slice_w_mm

    # ------------------------------------------------------------------
    # Slice render — flat rectangle debug output
    # ------------------------------------------------------------------

    def render_slice(self, vessel, dpi=300):
        mm_to_px = dpi / 25.4
        _, _, _, _, slice_w_mm = self._top_seg_geometry(vessel)

        h_mm = vessel.s_total
        w_px = max(1, int(round(slice_w_mm * mm_to_px)))
        h_px = max(1, int(round(h_mm * mm_to_px)))

        output_data = np.full((h_px, w_px, 4), 255, dtype=np.uint8)
        s_fine, F_fine = self._build_F(vessel)

        y_px_arr    = np.arange(h_px)
        s_arr       = y_px_arr / mm_to_px
        c_arr       = vessel.get_circumference_at(s_arr)
        tile_w_arr  = c_arr / self.h_tiles
        sy_frac_arr = np.interp(s_arr, s_fine, F_fine)

        x_mm_arr     = np.arange(w_px) / mm_to_px
        x_mm_grid    = x_mm_arr[np.newaxis, :]
        tile_w_grid  = tile_w_arr[:, np.newaxis]
        sy_frac_grid = sy_frac_arr[:, np.newaxis]

        offset  = (slice_w_mm - tile_w_grid) / 2
        sx_frac = (x_mm_grid - offset) / tile_w_grid
        mask    = (x_mm_grid >= offset) & (x_mm_grid < offset + tile_w_grid)

        src_x_px = (sx_frac % 1.0 * self.img.width).astype(int)  % self.img.width
        src_y_px = (sy_frac_grid % 1.0 * self.img.height
                    * np.ones((1, w_px))).astype(int) % self.img.height

        src_data = np.array(self.img)
        output_data[mask] = src_data[src_y_px[mask], src_x_px[mask]]

        canvas = Image.fromarray(output_data)
        draw   = ImageDraw.Draw(canvas)

        c_arr2     = vessel.get_circumference_at(np.arange(h_px) / mm_to_px)
        tile_w2    = c_arr2 / self.h_tiles
        prev_l = prev_r = None
        for row in range(0, h_px, 2):
            off = (slice_w_mm - float(tile_w2[row])) / 2
            lx  = max(0, min(int(off * mm_to_px), w_px-1))
            rx  = max(0, min(int((off + float(tile_w2[row])) * mm_to_px), w_px-1))
            if prev_l:
                draw.line([prev_l, (lx, row)], fill=(255,0,0,255), width=2)
                draw.line([prev_r, (rx, row)], fill=(255,0,0,255), width=2)
            prev_l, prev_r = (lx, row), (rx, row)

        for s_k in vessel._arc_lengths[1:-1]:
            y_k = int(float(s_k) * mm_to_px)
            if 0 <= y_k < h_px:
                draw.line([(0,y_k),(w_px-1,y_k)], fill=(0,80,200,200), width=2)

        draw.line([(2,h_px-4),(2+int(10*mm_to_px),h_px-4)], fill=(0,0,0,255), width=3)
        draw.text((2, h_px-20), "10mm", fill=(0,0,0,255))
        return canvas

    # ------------------------------------------------------------------
    # Fan render — fast inverse mapping
    # ------------------------------------------------------------------

    def render(self, vessel, dpi=300):
        """
        Fast inverse mapping — mathematically identical to the stamp approach
        but processes all pixels simultaneously with NumPy.

        For each output pixel:
          1. Polar coords: r, theta  (fan defined by top segment geometry)
          2. Which slice: i = floor((theta + theta_seg/2) / theta_step)
          3. Centre angle of that slice: theta_i = -theta_seg/2 + (i+0.5)*theta_step
          4. Local angle within slice: theta_local = theta - theta_i
          5. Arc distance from slice centre: arc_mm = r * theta_local
          6. x position in slice: x_mm = arc_mm + slice_w_mm/2
          7. Mask: x_mm must be in [offset, offset + tile_w(s)]
                   where tile_w(s) = c(s)/N and offset = (slice_w - tile_w)/2
          8. sx_frac = (x_mm - offset) / tile_w(s)
          9. sy_frac from F(s) integral
        """
        mm_to_px = dpi / 25.4

        r_top, r_bottom, theta_seg, theta_step, slice_w_mm = \
            self._top_seg_geometry(vessel)

        out_w_mm    = 2 * r_top * math.sin(theta_seg / 2)
        out_h_mm    = r_top - r_bottom * math.cos(theta_seg / 2)
        w_px        = int(round(out_w_mm * mm_to_px))
        h_px        = int(round(out_h_mm * mm_to_px))

        output_data = np.full((h_px, w_px, 4), 255, dtype=np.uint8)

        # Pixel grids
        x_grid, y_grid = np.meshgrid(np.arange(w_px), np.arange(h_px))
        u_mm = x_grid / mm_to_px
        v_mm = y_grid / mm_to_px

        # Polar coords (O is at canvas (out_w_mm/2, r_top))
        du    = u_mm - out_w_mm / 2
        dv    = r_top - v_mm          # distance above O
        r     = np.sqrt(du**2 + dv**2)
        theta = np.arctan2(du, dv)    # angle from vertical
        s     = r_top - r             # arc length from top

        # Fan boundary mask
        fan_mask = (
            (r >= r_bottom) &
            (r <= r_top) &
            (np.abs(theta) <= theta_seg / 2)
        )

        # Which slice does each pixel belong to?
        i_slice = np.floor((theta + theta_seg / 2) / theta_step).astype(int)
        i_slice = np.clip(i_slice, 0, self.h_tiles - 1)

        # Centre angle of that slice
        theta_i = -theta_seg / 2 + (i_slice + 0.5) * theta_step

        # Local angle and arc distance from slice centre
        theta_local = theta - theta_i
        arc_mm      = r * theta_local             # mm from slice centre line

        # x position in slice image coords (0 = left edge of slice)
        x_mm = arc_mm + slice_w_mm / 2

        # Local circumference and tile width at this height
        c_s      = vessel.get_circumference_at(s)
        c_s      = np.where(c_s < 1e-6, 1e-6, c_s)
        tile_w   = c_s / self.h_tiles             # mm

        # Centred strip within slice
        offset   = (slice_w_mm - tile_w) / 2
        strip_mask = (x_mm >= offset) & (x_mm < offset + tile_w)

        mask = fan_mask & strip_mask

        # Source coordinates
        sx_frac  = (x_mm - offset) / tile_w
        src_x_px = (sx_frac % 1.0 * self.img.width).astype(int) % self.img.width

        s_fine, F_fine = self._build_F(vessel)
        sy_frac  = np.interp(s, s_fine, F_fine)
        src_y_px = (sy_frac % 1.0 * self.img.height).astype(int) % self.img.height

        src_data = np.array(self.img)
        output_data[mask] = src_data[src_y_px[mask], src_x_px[mask]]

        canvas = Image.fromarray(output_data)
        self._draw_guides(canvas, vessel, out_w_mm, r_top, r_bottom,
                          theta_seg, theta_step, slice_w_mm, mm_to_px)
        return canvas

    # ------------------------------------------------------------------
    # Guide lines
    # ------------------------------------------------------------------

    def _draw_guides(self, canvas, vessel, out_w_mm, r_top, r_bottom,
                     theta_total, theta_step, slice_w_mm, mm_to_px):
        draw = ImageDraw.Draw(canvas)
        self._draw_arcs(draw, out_w_mm, r_top, r_bottom, theta_total, mm_to_px)
        self._draw_seams(draw, out_w_mm, r_top, r_bottom, theta_total, mm_to_px)
        self._draw_junction_arcs(draw, vessel, out_w_mm, r_top, theta_total, mm_to_px)
        self._draw_seam_lines(draw, vessel, out_w_mm, r_top, r_bottom, theta_total, theta_step, slice_w_mm, mm_to_px)
        self._draw_scale_bar(draw, canvas, mm_to_px)

    def _fan_pt(self, t, r_val, out_w_mm, r_top, mm_to_px):
        px = (r_val * math.sin(t) + out_w_mm / 2) * mm_to_px
        py = (r_top - r_val * math.cos(t)) * mm_to_px
        return (px, py)

    def _draw_arcs(self, draw, out_w_mm, r_top, r_bottom, theta_total, mm_to_px):
        for r in [r_top, r_bottom]:
            pts = [self._fan_pt(t, r, out_w_mm, r_top, mm_to_px)
                   for t in np.linspace(-theta_total/2, theta_total/2, 200)]
            draw.line(pts, fill=(255, 0, 0, 255), width=5)

    def _draw_seams(self, draw, out_w_mm, r_top, r_bottom, theta_total, mm_to_px):
        for t_sign in [-1, 1]:
            t = (theta_total / 2) * t_sign
            draw.line([self._fan_pt(t, r_top,    out_w_mm, r_top, mm_to_px),
                       self._fan_pt(t, r_bottom, out_w_mm, r_top, mm_to_px)],
                      fill=(0, 0, 255, 255), width=5)

    def _draw_junction_arcs(self, draw, vessel, out_w_mm, r_top, theta_total, mm_to_px):
        for k in range(1, len(vessel._arc_lengths) - 1):
            s_k = float(vessel._arc_lengths[k])
            r_k = r_top - s_k
            pts = [self._fan_pt(t, r_k, out_w_mm, r_top, mm_to_px)
                   for t in np.linspace(-theta_total/2, theta_total/2, 200)]
            draw.line(pts, fill=(0, 80, 200, 180), width=2)

    def _draw_seam_lines(self, draw, vessel, out_w_mm, r_top, r_bottom, theta_total, theta_step, slice_w_mm, mm_to_px):
        """
        Cut lines at each slice edge — straight line segments between
        the edge points at each CSV arc_length measurement.
        At each s, the edge is at theta_i +/- (c(s)/N/2) / r.
        """
        N = self.h_tiles

        # Sample at each CSV arc length only
        s_samples = vessel._arc_lengths
        r_samples  = r_top - s_samples
        c_samples  = vessel.get_circumference_at(s_samples)
        tile_w     = c_samples / N

        for i in range(N):
            theta_i = -theta_total / 2 + (i + 0.5) * theta_step
            for side in [-1, 1]:
                pts = []
                for j in range(len(s_samples)):
                    r  = float(r_samples[j])
                    tw = float(tile_w[j])
                    t  = theta_i + side * (tw / 2) / r
                    pts.append(self._fan_pt(t, r, out_w_mm, r_top, mm_to_px))
                draw.line(pts, fill=(180, 0, 0, 255), width=3)

    def _draw_scale_bar(self, draw, canvas, mm_to_px):
        margin  = 10 * mm_to_px
        bar_len = 50 * mm_to_px
        bar_y   = canvas.height - margin
        draw.line([(margin, bar_y), (margin + bar_len, bar_y)],
                  fill=(0, 0, 0, 255), width=8)
        draw.text((margin, bar_y - 25), "50mm SCALE CHECK",
                  fill=(0, 0, 0, 255))