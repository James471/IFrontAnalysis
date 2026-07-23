import os
import re

import numpy as np
import matplotlib.pyplot as pl
from astropy import units as u

from .util import create_movie
from .IonizationFrontSnapshot import IonizationFrontSnapshot


# Maps field_name -> default video output filename
_FIELD_VIDEO_NAMES = {
    "n_photon":    "radiation_evolution",
    "n_e":         "scalar0_evolution",
    "n_HI":        "scalar1_evolution",
    "n_HII":       "scalar2_evolution",
    "x_HI":        "x_HI_evolution",
    "gasDensity":  "density_evolution",
    "temperature": "temperature_evolution",
    "velocity":    "velocity_evolution",
    "v_x":         "v_x_evolution",
    "v_y":         "v_y_evolution",
    "v_z":         "v_z_evolution",
    "cs":          "cs_evolution",
    "pressure":    "pressure_evolution",
    "E_IR":        "E_IR_evolution",
    "E_optical":   "E_optical_evolution",
    "E_ion":       "E_ion_evolution",
}


class IonizationFront:
    def __init__(self, path, outdir=None, start_pattern="plt", ending_number=None, step=1,
                 analytical=None, save_pdf=True):
        self.path = path
        self.outdir = outdir
        self.start_pattern = start_pattern
        self.ending_number = ending_number
        self.step = step
        self.analytical = analytical
        # Whether field maps are also written as PDF alongside PNG. PDFs double the
        # render time and are wasted for video frames, so the driver disables them
        # for the whole-directory (map/video) workflow.
        self.save_pdf = save_pdf
        if self.outdir is None:
            self.outdir = os.path.join(self.path, "Plots")
        os.makedirs(self.outdir, exist_ok=True)
        self.plotfile_pathlist = self.get_plotfile_pathlist()
        self.snapshot_list = [IonizationFrontSnapshot(p, analytical=self.analytical)
                              for p in self.plotfile_pathlist]
        self.time_list = [s.ds.current_time.value for s in self.snapshot_list]

    # ── internal helpers ─────────────────────────────────────────────────────

    def _t_char_s(self):
        """t_char as an astropy Quantity in seconds, or 1 s when no analytical is set."""
        if self.analytical is None:
            return 1 * u.s
        t = self.analytical.t_char
        return t.to(u.s) if hasattr(t, 'to') else t * u.s

    def _r_char_pc(self):
        """r_char as an astropy Quantity in pc (requires analytical to be set)."""
        r = self.analytical.r_char
        return (r * u.cm).to(u.pc) if not hasattr(r, 'unit') else r.to(u.pc)

    # ── plotfile discovery ───────────────────────────────────────────────────

    def get_plotfile_pathlist(self):
        files = [f for f in os.listdir(self.path) if re.match(rf"{self.start_pattern}\d+", f)]
        files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        if self.ending_number is not None:
            files = [f for f in files if int(re.findall(r'\d+', f)[0]) <= self.ending_number]
        return [os.path.join(self.path, f) for f in files[::self.step]]

    # ── quantity maps & videos ───────────────────────────────────────────────

    def get_quantity_range(self, field_name):
        los, his = [], []
        for snapshot in self.snapshot_list:
            lo, hi = snapshot.get_quantity_range(field_name)
            if lo is not None:
                los.append(lo)
            if hi is not None:
                his.append(hi)
        return (min(los) if los else None,
                max(his) if his else None)

    def create_quantity_plots(self, field_name, vmin=None, vmax=None, cmap="viridis",
                              redo=False, plot_analytical=False, plot_eff=False,
                              nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        lo, hi = self.get_quantity_range(field_name)
        vmin = vmin if vmin is not None else lo
        vmax = vmax if vmax is not None else hi
        return [s.create_quantity_map(field_name, vmin=vmin, vmax=vmax, cmap=cmap, redo=redo,
                                      plot_analytical=plot_analytical, plot_eff=plot_eff,
                                      nolog=nolog, plot_front=plot_front,
                                      front_lb=front_lb, front_ub=front_ub,
                                      save_pdf=self.save_pdf)
                for s in self.snapshot_list]

    def create_quantity_video(self, field_name, output_filename=None, fps=10,
                              vmin=None, vmax=None, cmap="viridis", redo=False,
                              plot_analytical=False, plot_eff=False, nolog=False,
                              plot_front=False, front_lb=0.01, front_ub=0.99):
        if output_filename is None:
            output_filename = _FIELD_VIDEO_NAMES.get(field_name, f"{field_name}_evolution")
        paths = self.create_quantity_plots(field_name, vmin=vmin, vmax=vmax, cmap=cmap,
                                           redo=redo, plot_analytical=plot_analytical,
                                           plot_eff=plot_eff, nolog=nolog,
                                           plot_front=plot_front, front_lb=front_lb,
                                           front_ub=front_ub)
        create_movie(paths, self.outdir, output_filename, framerate=fps)
        print(f"Saved: {os.path.join(self.outdir, output_filename)}")

    # ── radius history ───────────────────────────────────────────────────────

    def get_radius_history(self, lb=0.01, ub=0.99):
        t_arr = np.array(self.time_list) * u.s
        r_median, r_16th, r_84th = [], [], []
        for snapshot in self.snapshot_list:
            r_med, r_16, r_84 = snapshot.get_front_radius(lb=lb, ub=ub)
            r_median.append(r_med)
            r_16th.append(r_16)
            r_84th.append(r_84)
        return (t_arr,
                np.array(r_median) * u.pc,
                np.array(r_16th)   * u.pc,
                np.array(r_84th)   * u.pc)

    def plot_radius_history(self, plot_analytical=True, lb=0.01, ub=0.99, fig=None, ax=None, label=None):
        if fig is None:
            fig, ax = pl.subplots()
        t_arr, r_median, r_16th, r_84th = self.get_radius_history(lb=lb, ub=ub)
        t_char_s = self._t_char_s()
        t_label  = self.analytical.t_char_label if self.analytical else "s"
        t_norm   = t_arr / t_char_s
        ax.plot(t_norm, r_median, label=label)
        ax.fill_between(t_norm.value, r_16th.value, r_84th.value, alpha=0.3)
        if plot_analytical and self.analytical is not None:
            r_an = self.analytical.get_analytical_radius_history_causal(t_arr)
            ax.plot(t_norm.value, r_an.to('pc').value, color='black', linestyle="--")
        ax.set_xlabel(rf"$t / ${t_label}")
        ax.set_ylabel("Radius (pc)")
        return fig, ax

    # ── effective radius history ─────────────────────────────────────────────

    def get_effective_radius_history(self):
        t_arr        = np.array(self.time_list) * u.s
        # s.get_effective_radius() returns a unyt_quantity; astropy's Quantity([...])
        # silently drops units when wrapping a list of those, so convert to plain
        # floats in pc first.
        r_effective  = np.array([float(s.get_effective_radius().to('pc').value)
                                 for s in self.snapshot_list]) * u.pc
        r_analytical = (self.analytical.get_analytical_radius_history_causal(t_arr)
                        if self.analytical is not None else None)
        return t_arr, r_effective, r_analytical

    def get_normalized_effective_radius_history(self):
        if self.analytical is None:
            raise ValueError("Analytical solution is required to compute the normalized effective radius history")
        t_arr, r_effective, _ = self.get_effective_radius_history()
        return t_arr / self._t_char_s(), r_effective / self._r_char_pc()

    def plot_normalized_effective_radius_history(self, fig=None, ax=None, label=None):
        if self.analytical is None:
            raise ValueError("Analytical solution is required to plot the normalized effective radius history")
        if fig is None:
            fig, ax = pl.subplots()
        t_hat, r_hat = self.get_normalized_effective_radius_history()
        ax.plot(t_hat, r_hat, label=label)
        ax.set_xlabel(rf"$t / ${self.analytical.t_char_label}")
        ax.set_ylabel(rf"$r_{{\mathrm{{eff}}}} / ${self.analytical.r_char_label}")
        return fig, ax

    def plot_effective_radius_history(self, plot_analytical=True, fig=None, ax=None, label=None):
        if fig is None:
            fig, ax = pl.subplots()
        t_arr, r_effective, r_analytical = self.get_effective_radius_history()
        if self.analytical is not None:
            t_norm  = t_arr / self._t_char_s()
            ax.set_xlabel(rf"$t / ${self.analytical.t_char_label}")
        else:
            t_norm  = t_arr
            ax.set_xlabel("Time (s)")
        ax.plot(t_norm, r_effective, label=label)
        if plot_analytical and r_analytical is not None:
            ax.plot(t_norm.value, r_analytical.to('pc').value, color='black', linestyle="--")
        ax.set_ylabel(r"$r_{\rm eff}$ (pc)")
        return fig, ax

    def plot_analytical_comparison_history(self, fig=None, ax=None):
        """Plot the numerical effective radius alongside all three analytical
        D-type front solutions (pure Spitzer, Krumholz & Matzner 2009 radiation
        pressure, and KM09 with the multi-group r_ch boost), all vs. time in
        seconds since Spitzer's t_s and KM09's t_ch are different normalisations
        and cannot share a single dimensionless x-axis.

        Requires self.analytical to be a DTypeAnalytical instance (used as the
        source of Q, n_H, and optical_to_ionizing_fraction for the three curves).
        """
        if self.analytical is None:
            raise ValueError("Analytical solution is required to plot the analytical comparison history")
        from .DTypeAnalytical import DTypeAnalytical

        if fig is None:
            fig, ax = pl.subplots()
        t_arr, r_effective, _ = self.get_effective_radius_history()
        t_s_val = t_arr.to(u.s).value

        Q, n_H = self.analytical.Q, self.analytical.n_H
        f_opt  = self.analytical.optical_to_ionizing_fraction
        spitzer   = DTypeAnalytical(Q, n_H, radiation_pressure=False)
        km09      = DTypeAnalytical(Q, n_H, radiation_pressure=True, mg=False)
        km09_mg   = DTypeAnalytical(Q, n_H, radiation_pressure=True, mg=True,
                                     optical_to_ionizing_fraction=f_opt)

        ax.plot(t_s_val, r_effective.to('pc').value, label=r"$r_{\rm eff}$ (numerical)")
        ax.plot(t_s_val, spitzer.get_spitzer_radius_history(t_arr).to('pc').value,
               color='black', linestyle="--", label="Spitzer")
        ax.plot(t_s_val, km09.get_radpres_radius_history(t_arr).to('pc').value,
               color='tab:red', linestyle=":", label="KM09")
        ax.plot(t_s_val, km09_mg.get_radpres_radius_history(t_arr).to('pc').value,
               color='tab:purple', linestyle="-.", label="KM09 (mg)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Radius (pc)")
        ax.legend()
        return fig, ax

    def get_effective_radius_error_history(self):
        if self.analytical is None:
            raise ValueError("Analytical solution is required to compute the effective radius error history")
        t_arr, r_effective, r_analytical = self.get_effective_radius_history()
        dx    = self.snapshot_list[0].ds.index.get_smallest_dx().to('pc')
        t_hat = t_arr / self._t_char_s()
        delta = ((r_effective - r_analytical).to('pc') / dx).value
        return t_hat, delta

    def plot_effective_radius_error_history(self, fig=None, ax=None, label=None):
        if fig is None:
            fig, ax = pl.subplots()
        t_hat, delta = self.get_effective_radius_error_history()
        ax.plot(t_hat, delta, label=label)
        ax.axhline(-np.sqrt(3), color='black', linestyle='--')
        ax.axhline( np.sqrt(3), color='black', linestyle='--')
        ax.set_xlabel(rf"$t / ${self.analytical.t_char_label}")
        ax.set_ylabel(r"$\Delta r / \Delta x$")
        ax.legend()
        return fig, ax

    # ── mass history ─────────────────────────────────────────────────────────

    def get_mass_in_sphere_history(self):
        t_arr       = np.array(self.time_list) * u.s
        m_in_sphere = np.array([s.get_mass_in_sphere().value for s in self.snapshot_list]) * u.g
        return t_arr, m_in_sphere

    # ── max velocity history ─────────────────────────────────────────────────

    def get_max_velocity_history(self):
        t_arr   = np.array(self.time_list) * u.s
        v_max   = np.array([float(s.get_quantity_range("velocity")[1])
                            for s in self.snapshot_list]) * (u.cm / u.s)
        return t_arr, v_max

    def plot_max_velocity_history(self, nolog=False, fig=None, ax=None, label=None):
        if fig is None:
            fig, ax = pl.subplots()
        t_arr, v_max = self.get_max_velocity_history()
        if self.analytical is not None:
            t_norm = t_arr / self._t_char_s()
            ax.set_xlabel(rf"$t / ${self.analytical.t_char_label}")
        else:
            t_norm = t_arr
            ax.set_xlabel("Time (s)")
        ax.plot(t_norm, v_max, label=label)
        if not nolog:
            ax.set_yscale("log")
        ax.set_ylabel(r"Max velocity (cm s$^{-1}$)")
        return fig, ax
