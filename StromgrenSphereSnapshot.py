import os
import re
import numpy as np
import yt
from astropy import units as u

from .Constants import Constants

def e_density(field, data):
    return data['scalar_0'] / Constants.m_e

def HI_density(field, data):
    return data['scalar_1'] / Constants.m_HI

def HII_density(field, data):
    return data['scalar_2'] / Constants.m_HII

def photon_density(field, data):
    freq_low, freq_high = 3.29e15, 1.50e16
    avg_freq = (freq_low + freq_high) / 2
    return data['radEnergy-Group0'] / (Constants.hplanck * avg_freq)

def x_HI(field, data):
    n_HI = data['scalar_1'] / Constants.m_HI
    n_HII = data['scalar_2'] / Constants.m_HII
    return n_HI / (n_HI + n_HII)

def temperature(field, data):
    """
    Calculate temperature from internal energy using the EOS formula.
    Follows the prescription from actual_eos.H:

    # Get the total density
    rho_total = data['gasDensity']
    T = e_int / (k_B * sum(n_i * f_i / 2))
    where f_i = 1 / (gamma_i - 1) for each species
    """
    
    # Get number densities
    n_HI = data['scalar_1'] / Constants.m_HI
    n_HII = data['scalar_2'] / Constants.m_HII
    n_e = data['scalar_0'] / Constants.m_e
    
    e_int = data['gasInternalEnergy']
    
    # Calculate sum of n_i * f_i / 2, where f_i = 1 / (gamma_i - 1)
    # For most species (monatomic ideal gas): gamma = 5/3, so f_i = 3/2
    # Assuming all species have gamma = 5/3 (monatomic)
    gamma = 5.0 / 3.0
    gamma_HI, gamma_HII, gamma_e = gamma, gamma, gamma
    
    # Sum of number fractions weighted by degrees of freedom
    n_total = n_HI + n_HII + n_e
    sum_n_f_over_2 = (n_HI * (1 / (gamma_HI - 1)) + n_HII * (1 / (gamma_HII - 1)) + n_e * (1 / (gamma_e - 1)))
    sum_n_f_over_2 /= n_total

    # Calculate temperature
    temp = e_int / (Constants.k_B * sum_n_f_over_2)

    return temp

def v_x(field, data):
    px = data['x-GasMomentum']
    rho = data['gasDensity']
    return px / rho

def v_y(field, data):
    py = data['y-GasMomentum']
    rho = data['gasDensity']
    return py / rho

def v_z(field, data):
    pz = data['z-GasMomentum']
    rho = data['gasDensity']
    return pz / rho

def velocity_magnitude(field, data):
    vx = v_x(field, data)
    vy = v_y(field, data)
    vz = v_z(field, data)
    return np.sqrt(vx**2 + vy**2 + vz**2)

def cs(field, data):
    # Get number densities
    n_HI = data['scalar_1'] / Constants.m_HI
    n_HII = data['scalar_2'] / Constants.m_HII
    n_e = data['scalar_0'] / Constants.m_e
    rhotot = data['gasDensity']
    e_int = data['gasInternalEnergy']
    
    # Calculate sum of n_i * f_i / 2, where f_i = 1 / (gamma_i - 1)
    # For most species (monatomic ideal gas): gamma = 5/3, so f_i = 3/2
    # Assuming all species have gamma = 5/3 (monatomic)
    gamma = 5.0 / 3.0
    gamma_HI, gamma_HII, gamma_e = gamma, gamma, gamma
    
    # Sum of number fractions weighted by degrees of freedom
    n_total = n_HI + n_HII + n_e
    sum_n_f_over_2 = (n_HI * (1 / (gamma_HI - 1)) + n_HII * (1 / (gamma_HII - 1)) + n_e * (1 / (gamma_e - 1)))
    sum_n_f_over_2 /= n_total

    p = e_int / sum_n_f_over_2
    cs = np.sqrt(1 + 1/sum_n_f_over_2) * np.sqrt(p / rhotot)
    return cs

def pressure(field, data):
    # Get number densities
    n_HI = data['scalar_1'] / Constants.m_HI
    n_HII = data['scalar_2'] / Constants.m_HII
    n_e = data['scalar_0'] / Constants.m_e
    
    e_int = data['gasInternalEnergy']
    
    # Calculate sum of n_i * f_i / 2, where f_i = 1 / (gamma_i - 1)
    # For most species (monatomic ideal gas): gamma = 5/3, so f_i = 3/2
    # Assuming all species have gamma = 5/3 (monatomic)
    gamma = 5.0 / 3.0
    gamma_HI, gamma_HII, gamma_e = gamma, gamma, gamma
    
    # Sum of number fractions weighted by degrees of freedom
    n_total = n_HI + n_HII + n_e
    sum_n_f_over_2 = (n_HI * (1 / (gamma_HI - 1)) + n_HII * (1 / (gamma_HII - 1)) + n_e * (1 / (gamma_e - 1)))
    sum_n_f_over_2 /= n_total

    p = e_int / sum_n_f_over_2
    return p

class StromgrenSphereSnapshot:
    def __init__(self, path, outdir=None, analytical=None):
        self.path = path
        self.outdir = outdir
        self.analytical = analytical
        self.ds = yt.load(path)
        self.ds.add_field(("boxlib", "n_e"), function=e_density, units="dimensionless", sampling_type="cell")
        self.ds.add_field(("boxlib", "n_HI"), function=HI_density, units="dimensionless", sampling_type="cell")
        self.ds.add_field(("boxlib", "n_HII"), function=HII_density, units="dimensionless", sampling_type="cell")
        self.ds.add_field(("boxlib", "n_photon"), function=photon_density, units="dimensionless", sampling_type="cell")
        self.ds.add_field(("boxlib", "x_HI"), function=x_HI, units="dimensionless", sampling_type="cell")
        self.ds.add_field(("boxlib", "temperature"), function=temperature, units="dimensionless", sampling_type="cell")
        self.ds.add_field(("boxlib", "v_x"), function=v_x, units="dimensionless", sampling_type="cell")
        self.ds.add_field(("boxlib", "v_y"), function=v_y, units="dimensionless", sampling_type="cell")
        self.ds.add_field(("boxlib", "v_z"), function=v_z, units="dimensionless", sampling_type="cell")
        self.ds.add_field(("boxlib", "velocity"), function=velocity_magnitude, units="dimensionless", sampling_type="cell")
        self.ds.add_field(("boxlib", "cs"), function=cs, units="dimensionless", sampling_type="cell")
        self.ds.add_field(("boxlib", "pressure"), function=pressure, units="dimensionless", sampling_type="cell")
        self.ad = self.ds.all_data()
        if self.outdir is None:
            self.outdir = os.path.join(self.path, "plots")
        os.makedirs(self.outdir, exist_ok=True)

    def get_front_radius(self, lb=0.01, ub=0.99):
        ad = self.ad
        r = np.sqrt(ad['x']**2 + ad['y']**2 + ad['z']**2)
        mask = (ad['x_HI'] > lb) & (ad['x_HI'] < ub)
        r_masked = r[mask]
        if len(r_masked) == 0:
            return 0, 0, 0
        return np.percentile(r_masked, 50).to('pc'), np.percentile(r_masked, 16).to('pc'), np.percentile(r_masked, 84).to('pc')

    def get_quantity_range(self, field_name):
        field_data = self.ad[field_name]
        vmin, vmax = np.min(field_data), np.max(field_data)
        return vmin, vmax

    def create_quantity_map(self, field_name, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = os.path.join(self.outdir, f"{field_name}.png")
        if os.path.exists(outpath) and not redo:
            return outpath
        bottom = self.ds.domain_left_edge[2]
        center = self.ds.domain_center.copy()
        dz = self.ds.index.get_smallest_dx()
        center[2] = bottom + 0.5 * dz
        plot = yt.SlicePlot(self.ds, "z", field_name, center=center)
        if vmin is None:
            vmin = 'min'
        if vmax is None:
            vmax = 'max'
        plot.set_zlim(field_name, zmin=vmin, zmax=vmax)
        if nolog:
            plot.set_log(field_name, False)
        plot.set_cmap(field_name, cmap)
        plot.set_colorbar_label(field_name, field_name)
        if plot_analytical and self.analytical:
            time = self.ds.current_time.to("s").value
            t_rec = self.analytical.t_rec.to(u.s).value
            r = self.analytical.get_analytical_radius_history_causal(np.asarray(time) * u.s)
            # r = self.analytical.r_s * (1 - np.exp(-time / t_rec))**(1/3)
            r = r.to(u.pc).value
            plot.annotate_sphere(self.ds.domain_left_edge, radius=(r, "pc"), circle_args={"color": "black", "linewidth": 4, "linestyle": "dashed"})
            plot.annotate_text((0.05, 0.95), f"t = {time/t_rec:.2f} t_rec", coord_system="axis")
        else:
            plot.annotate_text((0.05, 0.95), f"t = {(self.ds.current_time.value*u.s).to('Myr').value:.2f} Myr", coord_system="axis")
        if plot_front:
            r_med, r_low, r_high = self.get_front_radius(lb=front_lb, ub=front_ub)
            r_med = r_med
            plot.annotate_sphere(self.ds.domain_left_edge, radius=(r_med, "pc"), circle_args={"color": "red", "linewidth": 2})
        plot.save(outpath)
        return outpath

    def get_effective_radius(self):
        ad = self.ad
        x_HI = ad['x_HI']
        volume = ad['cell_volume']
        total_volume = np.sum(volume * (1 - x_HI))
        r_effective = (3 * 8 * total_volume / (4 * np.pi))**(1/3)
        return r_effective.to('pc')

    def create_radiation_map(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = self.create_quantity_map("n_photon", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        return outpath

    def create_scalar0_map(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = self.create_quantity_map("n_e", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        return outpath

    def create_scalar1_map(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = self.create_quantity_map("n_HI", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        return outpath

    def create_scalar2_map(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = self.create_quantity_map("n_HII", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        return outpath

    def create_x_HI_map(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = self.create_quantity_map("x_HI", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        return outpath

    def create_density_map(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = self.create_quantity_map("gasDensity", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        return outpath

    def create_temperature_map(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = self.create_quantity_map("temperature", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        return outpath
    
    def create_vx_map(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = self.create_quantity_map("v_x", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        return outpath
    
    def create_vy_map(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = self.create_quantity_map("v_y", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        return outpath
    
    def create_vz_map(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = self.create_quantity_map("v_z", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        return outpath
    
    def create_velocity_map(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = self.create_quantity_map("velocity", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        return outpath
    
    def create_cs_map(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = self.create_quantity_map("cs", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        return outpath
    
    def create_pressure_map(self, vmin=None, vmax=None, cmap="viridis", redo=False, plot_analytical=False, nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99):
        outpath = self.create_quantity_map("pressure", vmin=vmin, vmax=vmax, cmap=cmap, redo=redo, plot_analytical=plot_analytical, nolog=nolog, plot_front=plot_front, front_lb=front_lb, front_ub=front_ub)
        return outpath