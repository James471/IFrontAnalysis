import os
import re
import numpy as np
import yt
import matplotlib.pyplot as pl
from astropy import units as u

from .util import create_movie
from .StromgrenSphereSnapshot import StromgrenSphereSnapshot

class StromgrenSphere:
    def __init__(self, path, outdir=None, start_pattern="plt", ending_number=None, step=1, analytical=None):
        self.path = path
        self.outdir = outdir
        self.start_pattern = start_pattern
        self.ending_number = ending_number
        self.step = step
        self.analytical = analytical
        if self.outdir is None:
            self.outdir = os.path.join(self.path, "Plots")
        os.makedirs(self.outdir, exist_ok=True)
        self.plotfile_pathlist = self.get_plotfile_pathlist()
        self.snapshot_list = [StromgrenSphereSnapshot(plotfile_path, analytical=self.analytical) for plotfile_path in self.plotfile_pathlist]
        self.time_list = [snapshot.ds.current_time.value for snapshot in self.snapshot_list]

    def get_plotfile_pathlist(self):
        files = [f for f in os.listdir(self.path) if re.match(f"{self.start_pattern}\d+", f)]
        files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        if self.ending_number is not None:
            files = [f for f in files if int(re.findall(r'\d+', f)[0]) <= self.ending_number]
        files = files[::self.step]
        return [os.path.join(self.path, f) for f in files]
    
    def get_quantity_range(self, field_name):
        vmin, vmax = None, None
        for snapshot in self.snapshot_list:
            snapshot_vmin, snapshot_vmax = snapshot.get_quantity_range(field_name)
            if vmin is None:
                vmin, vmax = snapshot_vmin, snapshot_vmax
            else:
                vmin = min(vmin, snapshot_vmin)
                vmax = max(vmax, snapshot_vmax)
        return vmin, vmax
    
    def create_quantity_plots(self, field_name, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        plot_paths = []
        qty_range = self.get_quantity_range(field_name)
        if vmin is None:
            vmin = qty_range[0]
        if vmax is None:
            vmax = qty_range[1]
        for snapshot in self.snapshot_list:
            plot_paths.append(snapshot.create_quantity_map(field_name, vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub))
        return plot_paths

    def create_radiation_plots(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        return self.create_quantity_plots("n_photon", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)

    def create_scalar0_plots(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        return self.create_quantity_plots("n_e", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)

    def create_scalar1_plots(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        return self.create_quantity_plots("n_HI", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)

    def create_scalar2_plots(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        return self.create_quantity_plots("n_HII", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)

    def create_x_HI_plots(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        return self.create_quantity_plots("x_HI", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)

    def create_density_plots(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        return self.create_quantity_plots("gasDensity", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)

    def create_radiation_video(self, output_filename="radiation_evolution", fps=10, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        radiation_plot_paths = self.create_radiation_plots(vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        create_movie(radiation_plot_paths, self.outdir, output_filename, framerate=fps)

    def create_scalar_videos(self, output_filename_prefix="scalar_evolution", fps=10, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        scalar_plot0_paths = self.create_scalar0_plots(vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        scalar_plot1_paths = self.create_scalar1_plots(vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        scalar_plot2_paths = self.create_scalar2_plots(vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        create_movie(scalar_plot0_paths, self.outdir, f"{output_filename_prefix}_0", framerate=fps)
        create_movie(scalar_plot1_paths, self.outdir, f"{output_filename_prefix}_1", framerate=fps)
        create_movie(scalar_plot2_paths, self.outdir, f"{output_filename_prefix}_2", framerate=fps)

    def create_x_HI_video(self, output_filename="x_HI_evolution", fps=10, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        x_HI_plot_paths = self.create_x_HI_plots(vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        create_movie(x_HI_plot_paths, self.outdir, output_filename, framerate=fps)

    def create_density_video(self, output_filename="density_evolution", fps=10, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        density_plot_paths = self.create_density_plots(vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        create_movie(density_plot_paths, self.outdir, output_filename, framerate=fps)

    def get_radius_history(self, lb=0.01, ub=0.99):
        """Get radius history data from snapshots.
        
        Returns:
            t_arr: Time array in seconds
            r_median: Median radius in pc
            r_16th: 16th percentile radius in pc
            r_84th: 84th percentile radius in pc
        """
        t_arr = np.array(self.time_list) * u.s
        r_median, r_16th, r_84th = [], [], []
        for snapshot in self.snapshot_list:
            r_med, r_16, r_84 = snapshot.get_front_radius(lb=lb, ub=ub)
            r_median.append(r_med)
            r_16th.append(r_16)
            r_84th.append(r_84)
        r_median = np.array(r_median) * u.pc
        r_16th = np.array(r_16th) * u.pc
        r_84th = np.array(r_84th) * u.pc
        return t_arr, r_median, r_16th, r_84th
    
    def plot_radius_history(self, plot_analytical=True, lb=0.01, ub=0.99, fig=None, ax=None, label=None):
        """Plot radius history with optional analytical solution.
        
        Parameters:
            plot_analytical: Whether to plot analytical solution
            lb, ub: Lower and upper bounds for front radius calculation
            fig, ax: Matplotlib figure and axis objects
            label: Label for the plot
        
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        if fig is None:
            fig, ax = pl.subplots()
        t_arr, r_median, r_16th, r_84th = self.get_radius_history(lb=lb, ub=ub)
        if self.analytical:
            t_rec = self.analytical.t_rec
        else:
            t_rec = 1 * u.s
        ax.plot(t_arr / t_rec, r_median, label=label)
        ax.fill_between(t_arr / t_rec, r_16th.value, r_84th.value, alpha=0.3)
        if plot_analytical and self.analytical is not None:
            r_analytical = self.analytical.get_analytical_radius_history_causal(np.asarray(t_arr) * u.s)
            ax.plot(t_arr / t_rec, r_analytical.to('pc').value, color='black', linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Radius (pc)")
        return fig, ax

    def get_effective_radius_history(self):
        """Get effective radius history data from snapshots.
        
        Returns:
            t_arr: Time array in seconds
            r_effective: Effective radius in pc
            r_analytical: Analytical radius in pc (or None)
        """
        t_arr = np.array(self.time_list) * u.s
        r_effective = np.array([snapshot.get_effective_radius() for snapshot in self.snapshot_list]) * u.pc
        r_analytical = None
        if self.analytical is not None:
            r_analytical = self.analytical.get_analytical_radius_history_causal(np.asarray(t_arr) * u.s)
        return t_arr, r_effective, r_analytical

    def get_effective_radius_error_history(self):
        """Get normalized effective-radius error history.
        
        Returns:
            t_hat: Time array normalized by the recombination time
            delta: Normalized error array, (r_effective - r_analytical) / dx
        """
        if self.analytical is None:
            raise ValueError("Analytical solution is required to compute the effective radius error history")
        t_arr, r_effective, r_analytical = self.get_effective_radius_history()
        dx = self.snapshot_list[0].ds.index.get_smallest_dx().to('pc')
        t_rec = self.analytical.t_rec.to(u.s)
        t_hat = t_arr / t_rec
        delta = ((r_effective - r_analytical).to('pc') / dx).value
        return t_hat, delta
    
    def plot_effective_radius_history(self, plot_analytical=True, fig=None, ax=None, label=None):
        """Plot effective radius history with optional analytical solution.
        
        Parameters:
            plot_analytical: Whether to plot analytical solution
            fig, ax: Matplotlib figure and axis objects
            label: Label for the plot
        
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        if fig is None:
            fig, ax = pl.subplots()
        t_arr, r_effective, r_analytical = self.get_effective_radius_history()
        t_rec = self.analytical.t_rec if self.analytical is not None else 1
        ax.plot(t_arr/t_rec, r_effective, label=label)
        if plot_analytical and r_analytical is not None:
            ax.plot(t_arr/t_rec, r_analytical.to('pc').value, color='black', linestyle="--")
        if self.analytical is not None:
            ax.set_xlabel("t/t_rec")
        else:
            ax.set_xlabel("Time (s)")
        ax.set_ylabel("r_effective (pc)")
        return fig, ax

    def plot_effective_radius_error_history(self, fig=None, ax=None, label=None):
        """Plot normalized effective-radius error history.
        
        Parameters:
            fig, ax: Matplotlib figure and axis objects
            label: Label for the error curve

        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        if fig is None:
            fig, ax = pl.subplots()
        t_hat, delta = self.get_effective_radius_error_history()
        ax.plot(t_hat, delta, label=label)
        ax.axhline(-np.sqrt(3), color='black', linestyle='--')
        ax.axhline(np.sqrt(3), color='black', linestyle='--')
        ax.set_xlabel(r"$t/t_{\mathrm{rec}}$")
        ax.set_ylabel(r"$\Delta \mathrm{r}/\mathrm{dx}$")
        ax.legend()
        return fig, ax