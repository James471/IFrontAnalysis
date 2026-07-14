import numpy as np
import astropy.units as u
import matplotlib.pyplot as pl


class IonizationFrontFromCSV:
    """Read the `dtype_front_radii.csv` file written directly by the DTypeFront /
    DTypeFrontRadPres test problems (columns: time, r_effective, r_spitzer[,
    r_radpres]) and reproduce the effective-radius plots that IonizationFront
    normally computes from plotfiles.

    This avoids re-running a simulation with plotfiles enabled (plotfile_interval
    != -1) just to re-check the front radius against an updated analytical
    solution: the C++ test already writes r_effective, r_spitzer, and (for
    DTypeFrontRadPres) r_radpres at every step, so this class only needs the CSV
    and the domain cell size (for the error plot) to reproduce those plots.
    """

    def __init__(self, csv_path, dx_pc, analytical=None, use_radpres=False):
        """
        Parameters:
            csv_path:    path to dtype_front_radii.csv
            dx_pc:       smallest cell size, in pc (plain float), used for the
                         effective-radius error plot (Delta r / dx)
            analytical:  a DTypeAnalytical instance (or None). Used only for
                         t_char / r_char labels and normalisation; the analytical
                         radius itself is read from the CSV, not recomputed, so
                         it always matches exactly what the simulation compared
                         against.
            use_radpres: if True, use the r_radpres column (Krumholz & Matzner
                         2009 radiation-pressure solution) as "the" analytical
                         radius for plotting/error purposes instead of r_spitzer.
                         Requires the CSV to have been written by
                         DTypeFrontRadPres (which includes r_radpres).
        """
        self.csv_path = csv_path
        self.dx_pc = dx_pc
        self.analytical = analytical
        self.use_radpres = use_radpres

        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        self.t_arr = np.atleast_1d(data["time"]) * u.s
        self.r_effective = np.atleast_1d(data["r_effective"]) * u.cm
        self.r_spitzer = np.atleast_1d(data["r_spitzer"]) * u.cm
        if use_radpres:
            if "r_radpres" not in data.dtype.names:
                raise ValueError(
                    f"{csv_path} has no r_radpres column; it was written by DTypeFront "
                    "rather than DTypeFrontRadPres, or use_radpres=True was requested "
                    "with the wrong CSV."
                )
            self.r_radpres = np.atleast_1d(data["r_radpres"]) * u.cm
        else:
            self.r_radpres = None

    # ── internal helpers ─────────────────────────────────────────────────────

    def _t_char_s(self):
        if self.analytical is None:
            return 1 * u.s
        t = self.analytical.t_char
        return t.to(u.s) if hasattr(t, "to") else t * u.s

    def _t_char_label(self):
        return self.analytical.t_char_label if self.analytical is not None else "s"

    def _r_analytical(self):
        """The CSV column to compare r_effective against: r_radpres if
        use_radpres, else r_spitzer. Read directly from the CSV (not recomputed),
        so it exactly matches what the simulation itself compared against."""
        return self.r_radpres if self.use_radpres else self.r_spitzer

    # ── effective radius history ─────────────────────────────────────────────

    def get_effective_radius_history(self):
        return self.t_arr, self.r_effective.to(u.pc), self._r_analytical().to(u.pc)

    def get_normalized_effective_radius_history(self):
        if self.analytical is None:
            raise ValueError("Analytical solution is required to compute the normalized effective radius history")
        r_char_pc = (self.analytical.r_char * u.cm).to(u.pc)
        return self.t_arr / self._t_char_s(), self.r_effective.to(u.pc) / r_char_pc

    def plot_normalized_effective_radius_history(self, fig=None, ax=None, label=None):
        if self.analytical is None:
            raise ValueError("Analytical solution is required to plot the normalized effective radius history")
        if fig is None:
            fig, ax = pl.subplots()
        t_hat, r_hat = self.get_normalized_effective_radius_history()
        ax.plot(t_hat, r_hat, label=label)
        ax.set_xlabel(rf"$t / ${self._t_char_label()}")
        ax.set_ylabel(rf"$r_{{\mathrm{{eff}}}} / ${self.analytical.r_char_label}")
        return fig, ax

    def plot_effective_radius_history(self, plot_analytical=True, fig=None, ax=None, label=None):
        if fig is None:
            fig, ax = pl.subplots()
        t_arr, r_effective, r_analytical = self.get_effective_radius_history()
        if self.analytical is not None:
            t_norm = t_arr / self._t_char_s()
            ax.set_xlabel(rf"$t / ${self._t_char_label()}")
        else:
            t_norm = t_arr
            ax.set_xlabel("Time (s)")
        ax.plot(t_norm, r_effective, label=label)
        if plot_analytical:
            ax.plot(t_norm.value, r_analytical.value, color="black", linestyle="--")
        ax.set_ylabel(r"$r_{\rm eff}$ (pc)")
        return fig, ax

    def get_effective_radius_error_history(self):
        t_arr, r_effective, r_analytical = self.get_effective_radius_history()
        t_hat = t_arr / self._t_char_s()
        delta = ((r_effective - r_analytical) / (self.dx_pc * u.pc)).value
        return t_hat, delta

    def plot_effective_radius_error_history(self, fig=None, ax=None, label=None):
        if fig is None:
            fig, ax = pl.subplots()
        t_hat, delta = self.get_effective_radius_error_history()
        ax.plot(t_hat, delta, label=label)
        ax.axhline(-np.sqrt(3), color="black", linestyle="--")
        ax.axhline(np.sqrt(3), color="black", linestyle="--")
        ax.set_xlabel(rf"$t / ${self._t_char_label()}")
        ax.set_ylabel(r"$\Delta r / \Delta x$")
        ax.legend()
        return fig, ax
