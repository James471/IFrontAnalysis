import numpy as np
import astropy.units as u
from .Constants import Constants


def _lambda_rec(T):
    if T < 100.0:
        return 0.0
    return 6.1e-10 * Constants.k_B * T * np.power(T, -0.89)


def _lambda_ion_ff(T):
    return 1.4e-27 * np.sqrt(T) + 1.0e-19 * np.exp(-118348.0 / T)


def _net_energy_ionized(T, n_e):
    alpha_B = 2.6e-13 * np.power(T / 1.0e4, -0.7)
    epsilon = 6.4e-12
    photoheating = alpha_B * n_e * n_e * epsilon
    recombination_cooling = n_e * n_e * _lambda_rec(T)
    ion_ff_cooling = n_e * n_e * _lambda_ion_ff(T)
    return photoheating - recombination_cooling - ion_ff_cooling


def _compute_equilibrium_temperature_ionized(n_e):
    T_lo, T_hi = 1000.0, 1.0e5
    assert _net_energy_ionized(T_lo, n_e) > 0.0 and _net_energy_ionized(T_hi, n_e) < 0.0, \
        "Temperature brackets do not straddle root"
    for _ in range(10000):
        T_mid = 0.5 * (T_lo + T_hi)
        if _net_energy_ionized(T_mid, n_e) > 0.0:
            T_lo = T_mid
        else:
            T_hi = T_mid
        if (T_hi - T_lo) < 1.0:
            break
    return 0.5 * (T_lo + T_hi)


class DTypeAnalytical:
    """Analytical solution for D-type ionization fronts using the 4/7 Spitzer solution,
    with an optional radiation-pressure-driven solution from Krumholz & Matzner (2009,
    ApJ, 703, 1352).

    Gas-pressure (Spitzer) front position: R(t) = R_s * (1 + 7*t / (4*t_s))^(4/7)

    alpha_B is derived from the equilibrium temperature of the ionized region,
    which is computed self-consistently from the photoheating / cooling balance.

    Radiation-pressure option (radiation_pressure=True): the combined radiation +
    gas-pressure front radius (their Eq. 13) is

        R(t) = ( R_rad(t)^p + R_spitzer(t)^p )^(1/p),   p = (7 - k_rho)/2,

    where R_spitzer(t) is the Spitzer solution above and R_rad(t) is the pure
    radiation-pressure momentum-driven solution (their Eq. 11, embedded/spherical
    case, dimensional form). The shell momentum equals the radiant momentum,
    M_sh * Rdot = f_trap * L * t / c, with M_sh = (4/3) pi rho0 R^3 (k_rho = 0);
    integrating with R(0) = 0 gives the closed form

        R_rad(t) = [ 3 * f_trap * L / (2 pi c rho0) ]^(1/4) * sqrt(t),

    with bolometric luminosity L = psi * Q * eps0 (eps0 = 13.6 eV) and ambient mass
    density rho0 = mu_amb * m_p * n_H. This dimensional form requires no paper-fiducial
    constants (alpha_B, T_II, phi) beyond alpha_B(T_eq) already used for the Spitzer
    limb, so R_rad and R_spitzer combine on a common footing: the combined curve
    reduces EXACTLY to the Spitzer curve as t -> infinity and to the pure radiation
    solution as t -> 0. This mirrors the implementation in
    src/problems/DTypeFrontRadPres/testDTypeFrontRadPres.cpp.

    f_trap = 1 (direct ionizing radiation pressure only) is the physically consistent
    default for simulations that deposit only the momentum of absorbed ionizing
    photons (the paper's Table/figures use the fiducial f_trap = 2).
    """

    def __init__(self, Q, n_H, radiation_pressure=False, f_trap=1.0, psi=1.0, mu_amb=1.4):
        """
        Parameters:
            Q:      Ionizing photon rate (s^-1), dimensionless float in CGS
            n_H:    Hydrogen number density (cm^-3), dimensionless float in CGS
            radiation_pressure: if True, get_analytical_radius_history_causal returns
                    the Krumholz & Matzner (2009) combined radiation + gas-pressure
                    solution instead of the pure Spitzer solution.
            f_trap: radiation trapping factor (see class docstring). Default 1.0
                    (direct ionizing momentum only).
            psi:    ratio of bolometric to ionizing power, L = psi * Q * eps0.
                    Default 1.0 (luminosity comes entirely from ionizing photons).
            mu_amb: atomic mass per H nucleus of the ambient neutral gas. Default 1.4
                    (standard cosmic composition).
        """
        self.Q = Q
        self.n_H = n_H
        self.radiation_pressure = radiation_pressure
        self.f_trap = f_trap
        self.psi = psi
        self.mu_amb = mu_amb

        # Equilibrium temperature in ionized cavity (n_e ≈ n_H for fully ionized H)
        self.T_eq = _compute_equilibrium_temperature_ionized(n_H)

        # alpha_B derived from T_eq (not a free parameter)
        self.alpha_B = 2.6e-13 * np.power(self.T_eq / 1.0e4, -0.7)

        # Sound speed in ionized region (mu = 0.5 for fully ionized hydrogen)
        mu = 0.5
        self.c_i = np.sqrt(Constants.k_B * self.T_eq / (mu * Constants.m_p))

        # Characteristic radius and timescales
        self.r_s = np.power(3.0 * Q / (4.0 * np.pi * self.alpha_B * n_H**2), 1.0 / 3.0)
        self.t_s = self.r_s / self.c_i
        self.t_rec = 1.0 / (self.alpha_B * n_H)
        self.V_s = (4.0 / 3.0) * np.pi * self.r_s**3
        # Generic aliases used by IonizationFrontSnapshot / IonizationFront
        self.r_char = self.r_s
        self.t_char = self.t_s
        self.t_char_label = r"$t_{\rm s}$"
        self.r_char_label = r"$r_{\rm s}$"

        # Radiation-pressure quantities (Krumholz & Matzner 2009), computed regardless of
        # radiation_pressure so callers can inspect them (e.g. via get_radpres_radius_history).
        eps0 = 13.6 * Constants.ev2erg
        self.L = self.psi * Q * eps0
        self.rho0 = self.mu_amb * Constants.m_p * n_H

    def _spitzer_radius_cm(self, t_s_val):
        """Spitzer gas-pressure radius in cm, given time array in seconds (plain floats)."""
        return self.r_s * np.power(1.0 + 7.0 * t_s_val / (4.0 * self.t_s), 4.0 / 7.0)

    def _radpres_radius_cm(self, t_s_val):
        """Pure radiation-pressure momentum-driven radius in cm (dimensional Eq. 11,
        embedded, k_rho = 0), given time array in seconds (plain floats)."""
        t_s_val = np.asarray(t_s_val, dtype=float)
        R_rad = np.zeros_like(t_s_val)
        positive = t_s_val > 0.0
        R_rad[positive] = (
            np.power(3.0 * self.f_trap * self.L / (2.0 * np.pi * Constants.c * self.rho0), 0.25)
            * np.sqrt(t_s_val[positive])
        )
        return R_rad

    def get_spitzer_radius_history(self, t_array):
        """Return the pure Spitzer (gas-pressure) front radius at each time in t_array.

        Parameters:
            t_array: array with astropy time units

        Returns:
            Quantity array of radii in pc
        """
        t_s_val = t_array.to(u.s).value
        return (self._spitzer_radius_cm(t_s_val) * u.cm).to(u.pc)

    def get_radpres_radius_history(self, t_array):
        """Return the Krumholz & Matzner (2009) combined radiation + gas-pressure front
        radius at each time in t_array (their Eq. 13, dimensional, anchor-free form; see
        class docstring).

        Parameters:
            t_array: array with astropy time units

        Returns:
            Quantity array of radii in pc
        """
        t_s_val = t_array.to(u.s).value
        R_rad_cm = self._radpres_radius_cm(t_s_val)
        R_gas_cm = self._spitzer_radius_cm(t_s_val)
        p = 3.5  # (7 - k_rho) / 2, k_rho = 0
        R_cm = np.power(np.power(R_rad_cm, p) + np.power(R_gas_cm, p), 1.0 / p)
        return (R_cm * u.cm).to(u.pc)

    def get_analytical_radius_history_causal(self, t_array):
        """Return the D-type front radius at each time in t_array.

        Returns the Krumholz & Matzner (2009) radiation-pressure solution if
        radiation_pressure=True was passed to the constructor, otherwise the pure
        Spitzer (gas-pressure) solution.

        Parameters:
            t_array: array with astropy time units

        Returns:
            Quantity array of radii in pc
        """
        if self.radiation_pressure:
            return self.get_radpres_radius_history(t_array)
        return self.get_spitzer_radius_history(t_array)
