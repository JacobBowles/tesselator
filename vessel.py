import numpy as np

class TaperedVessel:
    def __init__(self, c_top, c_bottom, slant_height):
        """
        c_top: Top circumference (mm)
        c_bottom: Bottom circumference (mm)
        slant_height: The 'arc length' down the profile (mm)
        """
        self.c_top = c_top
        self.c_bottom = c_bottom
        self.s = slant_height
        
        # Calculate the 'Radii of Development' (the flat pattern radii)
        # Using the frustum unwrapping formula
        self.r_top = (c_top * slant_height) / (c_top - c_bottom)
        self.r_bottom = self.r_top - slant_height
        
        # Total angle of the unwrapped fan in radians
        self.theta_total = c_top / self.r_top

    def get_source_coords(self, u_mm, v_mm, canvas_width_mm):
        """
        Maps a point on the physical paper (u, v) back to 
        coordinates on the original design (sx, sy).
        """
        # Center the fan on the output canvas
        du = u_mm - canvas_width_mm / 2
        
        # Convert Cartesian paper coordinates to Polar fan coordinates
        r = np.sqrt(du**2 + (self.r_top - v_mm)**2)
        theta = np.arctan2(du, self.r_top - v_mm)
        
        # Check if the point is actually on the paper template
        if not (self.r_bottom <= r <= self.r_top) or abs(theta) > self.theta_total / 2:
            return None, None

        # sx = horizontal position along the circumference
        # sy = vertical position along the slant height
        sx = ((theta + self.theta_total / 2) / self.theta_total) * self.c_top
        sy = self.r_top - r
        return sx, sy