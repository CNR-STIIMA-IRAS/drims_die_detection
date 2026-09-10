"""
PlotlyVisualizer
================
Generates interactive 3D point-cloud plots saved as self-contained HTML files.
Requires the optional *plotly* package.
"""

from __future__ import annotations

import numpy as np

from .die_detector_params import DieDetectorParams

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class PlotlyVisualizer:
    """Generates interactive 3D HTML plots of the scene and detected die pose.

    Parameters
    ----------
    params : DieDetectorParams
        Uses ``debug``, ``save``, ``output_dir``.
    """

    def __init__(self, params: DieDetectorParams | None = None) -> None:
        self.params = params or DieDetectorParams()
        self._log = self._make_logger()

    # ──────────────────────────────────────────────────────────────────────
    def visualize(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        die_points: np.ndarray | None = None,
        plane_model: tuple | None = None,
        centroid: np.ndarray | None = None,
        die_centroid: np.ndarray | None = None,
        die_axes: tuple | None = None,
        die_size: float | None = None,
        title: str = "3D Point Cloud & Segmented Die Pose",
        output_html_path: str | None = None,
        max_points: int = 15_000,
    ):
        """Build the Plotly figure.

        Parameters
        ----------
        points : (N, 3) — full scene point cloud
        colors : (N, 3) float — RGB [0,1]
        die_points : (M, 3) optional — segmented die blob
        plane_model : (A,B,C,D) optional — for future plane rendering
        centroid : (3,) — top-face TF origin
        die_centroid : (3,) — die body centroid TF origin
        die_axes : (x_ax, y_ax, z_ax) — die reference frame axes
        die_size : float — die side length in metres
        title : str
        output_html_path : str | None — if set (and params.save), saved here
        max_points : int — subsample limit for scene cloud

        Returns
        -------
        fig : plotly.graph_objects.Figure or None
        """
        if not HAS_PLOTLY:
            print("[PlotlyVisualizer] WARNING: plotly not installed — 3D HTML plots will not be generated. "
                  "Install with: pip install plotly")
            return None

        d_size = die_size if die_size is not None else self.params.die_size_m

        # Subsample
        N = len(points)
        if N > max_points:
            idx = np.random.choice(N, max_points, replace=False)
            pts_sub = points[idx]
            cols_sub = colors[idx]
        else:
            pts_sub, cols_sub = points, colors

        rgb_strs = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in cols_sub]

        fig = go.Figure()

        # Scene cloud
        fig.add_trace(go.Scatter3d(
            x=pts_sub[:, 0], y=pts_sub[:, 1], z=pts_sub[:, 2],
            mode="markers",
            marker=dict(size=2, color=rgb_strs, opacity=0.5),
            name="Scene Point Cloud",
        ))

        # Die blob
        if die_points is not None and len(die_points) > 0:
            die_sub = die_points if len(die_points) <= 3000 else \
                die_points[np.random.choice(len(die_points), 3000, replace=False)]
            fig.add_trace(go.Scatter3d(
                x=die_sub[:, 0], y=die_sub[:, 1], z=die_sub[:, 2],
                mode="markers",
                marker=dict(size=5, color="red", symbol="diamond", opacity=0.9),
                name="Segmented Die Blob",
            ))

        # Camera origin + axes
        VL = 0.15
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers+text",
            marker=dict(size=7, color="black", symbol="circle"),
            text=["Camera Origin (0,0,0)"], textposition="top center",
            name="Camera Origin",
        ))
        for axis_vec, color, label in [
            ([VL, 0, 0], "red",   "Cam X"),
            ([0, VL, 0], "green", "Cam Y"),
            ([0, 0, VL], "blue",  "Cam Z"),
        ]:
            fig.add_trace(go.Scatter3d(
                x=[0, axis_vec[0]], y=[0, axis_vec[1]], z=[0, axis_vec[2]],
                mode="lines+text", line=dict(color=color, width=6),
                text=["", label], name=f"{label} axis",
            ))

        # Die TF axes
        if centroid is not None and die_axes is not None:
            x_ax, y_ax, z_ax = die_axes
            arm = 0.16

            if die_centroid is None:
                die_centroid = centroid - (d_size / 2.0) * z_ax

            for origin, symbol, text_label, name_prefix, width, colors_axs in [
                (centroid,     "circle",  "Top Face TF",   "Top Face",   10,
                 ("red", "green", "blue")),
                (die_centroid, "diamond", "Die Centroid TF", "Die Centroid", 8,
                 ("darkred", "darkgreen", "darkblue")),
            ]:
                fig.add_trace(go.Scatter3d(
                    x=[origin[0]], y=[origin[1]], z=[origin[2]],
                    mode="markers+text",
                    marker=dict(size=14,
                                color="white" if "Top" in name_prefix else "yellow",
                                line=dict(color="black", width=3),
                                symbol=symbol),
                    text=[text_label], textposition="top center",
                    name=f"{name_prefix} TF Origin",
                ))
                for ax_vec, c, ax_name in zip([x_ax, y_ax, z_ax], colors_axs, ["X", "Y", "Z"]):
                    fig.add_trace(go.Scatter3d(
                        x=[origin[0], origin[0] + arm * ax_vec[0]],
                        y=[origin[1], origin[1] + arm * ax_vec[1]],
                        z=[origin[2], origin[2] + arm * ax_vec[2]],
                        mode="lines", line=dict(color=c, width=width),
                        name=f"{name_prefix} {ax_name}-axis",
                    ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (m)", yaxis_title="Y (m)",
                zaxis_title="Z (depth m: top→bottom)",
                zaxis=dict(autorange="reversed"),
                aspectmode="data",
                camera=dict(eye=dict(x=0.5, y=-1.5, z=-1.5)),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        if output_html_path and self.params.save:
            fig.write_html(output_html_path)
            self._log(f"Saved 3D plot → {output_html_path}")
        elif output_html_path:
            fig.write_html(output_html_path)   # respect explicit path even if save=False

        return fig

    # Static alias
    @classmethod
    def visualize_point_cloud_3d(cls, **kwargs):
        return cls().visualize(**kwargs)

    def _make_logger(self):
        tag = "[PlotlyVisualizer]"
        if self.params.debug:
            return lambda msg: print(f"{tag} {msg}")
        return lambda msg: None
