import numpy as np


class TaperedVessel:
    def __init__(self, df):
        """
        df: pandas DataFrame with 'arc_length' and 'circumference' columns.
        Rows define the vessel profile from top (s=0) downward.
        """
        # Sort by arc_length just in case rows are out of order
        df = df.sort_values('arc_length').reset_index(drop=True)

        self._arc_lengths = df['arc_length'].to_numpy(dtype=float)
        self._circumferences = df['circumference'].to_numpy(dtype=float)

        # Top and bottom of the overall vessel
        self.c_top = self._circumferences[0]
        self.c_bottom = self._circumferences[-1]
        self.s_total = self._arc_lengths[-1] - self._arc_lengths[0]

        # The widest circumference anywhere on the vessel — used for wedge calculations
        self.c_max = float(np.max(self._circumferences))

        # Overall fan geometry — derived from the first and last measurement.
        # This defines the outer and inner arcs of the flat template.
        # If c_top == c_bottom (pure cylinder) r_top would be infinite;
        # we guard against that but note a pure cylinder is an edge case
        # we can revisit if needed.
        if abs(self.c_top - self.c_bottom) < 1e-6:
            raise ValueError(
                "Top and bottom circumferences are equal (pure cylinder). "
                "Add at least one measurement point with a different circumference."
            )

        self.r_top = (self.c_top * self.s_total) / (self.c_top - self.c_bottom)
        self.r_bottom = self.r_top - self.s_total
        self.theta_total = self.c_top / self.r_top

    # ------------------------------------------------------------------
    # Profile queries
    # ------------------------------------------------------------------

    def get_circumference_at(self, s):
        """
        Linearly interpolates circumference at arc-length s (scalar or array).
        s is measured from the top rim (s=0).
        Values outside the measured range are clamped to the end values.
        """
        return np.interp(s, self._arc_lengths, self._circumferences)

    # ------------------------------------------------------------------
    # Inverse mapping — paper coordinates → pattern coordinates
    # ------------------------------------------------------------------

    def get_source_coords(self, u_mm, v_mm, canvas_width_mm):
        """
        Maps a point on the physical paper (u, v) back to
        coordinates on the original design (sx, sy).

        u_mm, v_mm  : arrays of paper coordinates in mm
        canvas_width_mm : total width of the output canvas in mm

        Returns
        -------
        sx : horizontal position in pattern space (0 → c_top)
        sy : vertical position in pattern space (0 → s_total)
             — note this is the arc-length down the vessel profile
        """
        # Shift origin so the fan is centred horizontally
        du = u_mm - canvas_width_mm / 2

        # Convert Cartesian paper coordinates to polar fan coordinates
        r = np.sqrt(du**2 + (self.r_top - v_mm)**2)
        theta = np.arctan2(du, self.r_top - v_mm)

        # Boolean mask — True only for pixels that lie inside the fan
        mask = (
            (r >= self.r_bottom) &
            (r <= self.r_top) &
            (np.abs(theta) <= self.theta_total / 2)
        )

        # Map polar → arc-length pattern coordinates
        sx = ((theta + self.theta_total / 2) / self.theta_total) * self.c_top
        sy = self.r_top - r   # sy == 0 at top rim, increases downward

        return sx, sy, mask