import os
import re
import numpy as np
import yt

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