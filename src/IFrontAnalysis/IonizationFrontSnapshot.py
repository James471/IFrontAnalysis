import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: safe in spawned worker processes

import numpy as np
import yt
from astropy import units as u
from .Constants import Constants

_TEXT_ARGS = {"color": "white", "fontsize": 14}
_INSET_BOX_ARGS = {"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.8, "edgecolor": "none"}

# Physical units of each plotted field. The field functions return raw cgs
# values (yt registers them as "dimensionless"), so these strings describe the
# quantity actually plotted. Fields absent from this map are genuinely
# dimensionless (e.g. x_HI) and get no unit appended to the colorbar label.
_FIELD_UNITS = {
    "n_e":         "cm$^{-3}$",
    "n_HI":        "cm$^{-3}$",
    "n_HII":       "cm$^{-3}$",
    "n_photon":    "cm$^{-3}$",
    "temperature": "K",
    "pressure":    "dyn cm$^{-2}$",
    "cs":          "cm s$^{-1}$",
    "v_x":         "cm s$^{-1}$",
    "v_y":         "cm s$^{-1}$",
    "v_z":         "cm s$^{-1}$",
    "velocity":    "cm s$^{-1}$",
    "gasDensity":  "g cm$^{-3}$",
    "E_IR":        "erg cm$^{-3}$",
    "E_optical":   "erg cm$^{-3}$",
    "E_ion":       "erg cm$^{-3}$",
}

# ── derived field functions ──────────────────────────────────────────────────

def e_density(field, data):
    return data['scalar_0'] / Constants.m_e

def HI_density(field, data):
    return data['scalar_1'] / Constants.m_HI

def HII_density(field, data):
    return data['scalar_2'] / Constants.m_HII

def photon_density(field, data):
    avg_freq = (3.29e15 + 1.50e16) / 2
    return data['radEnergy-Group0'] / (Constants.hplanck * avg_freq)

def E_IR(field, data):
    return data['radEnergy-Group0']

def E_optical(field, data):
    return data['radEnergy-Group1']

def E_ion(field, data):
    return data['radEnergy-Group2']

def x_HI(field, data):
    n_HI  = data['scalar_1'] / Constants.m_HI
    n_HII = data['scalar_2'] / Constants.m_HII
    return n_HI / (n_HI + n_HII)

# gamma = 5/3 for every species, so 1/(gamma-1) = 3/2 everywhere.
# sum_n_f = n_HI * 3/2 + n_HII * 3/2 + n_e * 3/2  (un-normalised, per-cell)
# sum_n_f_norm = sum_n_f / n_total                  (normalised, used by cs/pressure)
_INV_GAMMA_MINUS1 = 1.0 / (5.0 / 3.0 - 1.0)  # = 1.5

def _sum_n_f(data):
    """Sum of n_i / (gamma_i - 1) over all species (not normalised by n_total)."""
    n_HI  = data['scalar_1'] / Constants.m_HI
    n_HII = data['scalar_2'] / Constants.m_HII
    n_e   = data['scalar_0'] / Constants.m_e
    return (n_HI + n_HII + n_e) * _INV_GAMMA_MINUS1


def temperature(field, data):
    return data['gasInternalEnergy'] / (Constants.k_B * _sum_n_f(data))

def pressure(field, data):
    # p = e_int / (sum_n_f / n_total) = e_int / _INV_GAMMA_MINUS1
    return data['gasInternalEnergy'] / _INV_GAMMA_MINUS1

def cs(field, data):
    p      = pressure(field, data)
    rhotot = data['gasDensity']
    # cs^2 = (1 + 1/f_norm) * p/rho  where f_norm = _INV_GAMMA_MINUS1
    return np.sqrt((1.0 + 1.0 / _INV_GAMMA_MINUS1) * p / rhotot)

def v_x(field, data):
    return data['x-GasMomentum'] / data['gasDensity']

def v_y(field, data):
    return data['y-GasMomentum'] / data['gasDensity']

def v_z(field, data):
    return data['z-GasMomentum'] / data['gasDensity']

def velocity_magnitude(field, data):
    vx = data['x-GasMomentum'] / data['gasDensity']
    vy = data['y-GasMomentum'] / data['gasDensity']
    vz = data['z-GasMomentum'] / data['gasDensity']
    return np.sqrt(vx**2 + vy**2 + vz**2)


# ── snapshot class ───────────────────────────────────────────────────────────

class IonizationFrontSnapshot:
    def __init__(self, path, outdir=None, analytical=None):
        self.path = path
        self.analytical = analytical
        self.ds = yt.load(path)
        # radEnergy-Group1/2 only exist in multi-group (e.g. DTypeFrontMG)
        # plotfiles. Registering E_optical/E_ion against a single-group plotfile
        # makes yt raise YTFieldNotFound as soon as any field is touched, so gate
        # those two derived fields on the groups actually being present.
        raw_fields = {f[1] for f in self.ds.field_list}
        derived_fields = [
            ("n_e",       e_density),
            ("n_HI",      HI_density),
            ("n_HII",     HII_density),
            ("n_photon",  photon_density),
            ("E_IR",      E_IR),
            ("x_HI",      x_HI),
            ("temperature", temperature),
            ("v_x",       v_x),
            ("v_y",       v_y),
            ("v_z",       v_z),
            ("velocity",  velocity_magnitude),
            ("cs",        cs),
            ("pressure",  pressure),
        ]
        if "radEnergy-Group1" in raw_fields:
            derived_fields.append(("E_optical", E_optical))
        if "radEnergy-Group2" in raw_fields:
            derived_fields.append(("E_ion", E_ion))
        for name, fn in derived_fields:
            self.ds.add_field(("boxlib", name), function=fn,
                              units="dimensionless", sampling_type="cell")
        self.ad = self.ds.all_data()
        self.outdir = outdir if outdir is not None else os.path.join(path, "plots")
        os.makedirs(self.outdir, exist_ok=True)
        # Cache of (min, max) per field. Deriving a field over all_data() is
        # expensive and the range is needed by both the global-range scan and the
        # slice plot, so memoise it to avoid re-deriving the same field twice.
        self._range_cache = {}

    # ── analysis helpers ─────────────────────────────────────────────────────

    def get_front_radius(self, lb=0.01, ub=0.99):
        r = np.sqrt(self.ad['x']**2 + self.ad['y']**2 + self.ad['z']**2)
        mask = (self.ad['x_HI'] > lb) & (self.ad['x_HI'] < ub)
        r_masked = r[mask]
        if len(r_masked) == 0:
            return 0, 0, 0
        return (np.percentile(r_masked, 50).to('pc'),
                np.percentile(r_masked, 16).to('pc'),
                np.percentile(r_masked, 84).to('pc'))

    def get_quantity_range(self, field_name):
        cached = self._range_cache.get(field_name)
        if cached is not None:
            return cached
        field_data = self.ad[field_name]
        rng = (np.min(field_data), np.max(field_data))
        self._range_cache[field_name] = rng
        return rng

    def get_effective_radius(self):
        total_volume = np.sum(self.ad['cell_volume'] * (1 - self.ad['x_HI']))
        return (((3 * 8 * total_volume) / (4 * np.pi))**(1/3)).to('pc')

    def get_mass_in_sphere(self):
        r = self.get_effective_radius()
        ad = self.ad
        density = ad['gasDensity'].value * u.g / u.cm**3
        volume  = ad['cell_volume'].value * u.cm**3
        r_cell  = np.sqrt(ad['x']**2 + ad['y']**2 + ad['z']**2)
        return np.sum(density[r_cell < r] * volume[r_cell < r])

    # ── plotting ─────────────────────────────────────────────────────────────

    def create_quantity_map(self, field_name, vmin=None, vmax=None, cmap="viridis",
                            redo=False, plot_analytical=False, plot_eff=False,
                            nolog=False, plot_front=False, front_lb=0.01, front_ub=0.99,
                            save_pdf=True):
        outpath = os.path.join(self.outdir, f"{field_name}.png")
        if os.path.exists(outpath) and not redo:
            return outpath

        center = self.ds.domain_center.copy()
        center[2] = self.ds.domain_left_edge[2] + 0.5 * self.ds.index.get_smallest_dx()
        plot = yt.SlicePlot(self.ds, "z", field_name, center=center)

        # Fall back to linear scale when the field has no positive dynamic range
        # (e.g. velocity is identically zero in the first frames of a D-type
        # front). A log/symlog norm would otherwise raise "No finite data
        # points." once yt masks out the non-positive values.
        fmin, fmax = self.get_quantity_range(field_name)
        force_linear = nolog or not (float(fmax) > 0.0 and float(fmin) < float(fmax))

        plot.set_zlim(field_name, zmin=vmin or 'min', zmax=vmax or 'max')
        if force_linear:
            plot.set_log(field_name, False)
        plot.set_cmap(field_name, cmap)
        unit = _FIELD_UNITS.get(field_name)
        label = f"{field_name} [{unit}]" if unit else field_name
        plot.set_colorbar_label(field_name, label)

        time = self.ds.current_time.to("s").value
        if self.analytical:
            t_char = self.analytical.t_char
            t_char_s = t_char.to(u.s).value if hasattr(t_char, 'to') else t_char
            label = self.analytical.t_char_label
            plot.annotate_text((0.05, 0.95), f"t = {time/t_char_s:.0f} {label}",
                               coord_system="axis", text_args=_TEXT_ARGS,
                               inset_box_args=_INSET_BOX_ARGS)
        else:
            t_myr = (time * u.s).to('Myr').value
            plot.annotate_text((0.05, 0.95), f"t = {t_myr:.2f} Myr",
                               coord_system="axis", text_args=_TEXT_ARGS,
                               inset_box_args=_INSET_BOX_ARGS)

        if plot_analytical and self.analytical:
            r = self.analytical.get_analytical_radius_history_causal(
                    np.asarray(time) * u.s).to(u.pc).value
            plot.annotate_sphere(self.ds.domain_left_edge, radius=(r, "pc"),
                                 circle_args={"color": "black", "linewidth": 4, "linestyle": "dashed"})

        if plot_front:
            r_med, _, _ = self.get_front_radius(lb=front_lb, ub=front_ub)
            plot.annotate_sphere(self.ds.domain_left_edge, radius=(r_med, "pc"),
                                 circle_args={"color": "red", "linewidth": 2})
        if plot_eff:
            r_eff = self.get_effective_radius()
            plot.annotate_sphere(self.ds.domain_left_edge, radius=(r_eff, "pc"),
                                 circle_args={"color": "blue", "linewidth": 2})
        plot.save(outpath)
        print(f"Saved: {outpath}")
        if save_pdf:
            pdfpath = outpath.replace(".png", ".pdf")
            plot.save(pdfpath)
            print(f"Saved: {pdfpath}")
        return outpath
