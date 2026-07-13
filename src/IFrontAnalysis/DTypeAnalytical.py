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
    """Analytical solution for D-type ionization fronts using the 4/7 Spitzer solution.

    Front position: R(t) = R_s * (1 + 7*t / (4*t_s))^(4/7)

    alpha_B is derived from the equilibrium temperature of the ionized region,
    which is computed self-consistently from the photoheating / cooling balance.
    """

    def __init__(self, Q, n_H):
        """
        Parameters:
            Q:   Ionizing photon rate (s^-1), dimensionless float in CGS
            n_H: Hydrogen number density (cm^-3), dimensionless float in CGS
        """
        self.Q = Q
        self.n_H = n_H

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

    def get_analytical_radius_history_causal(self, t_array):
        """Return the D-type front radius at each time in t_array.

        Parameters:
            t_array: array with astropy time units

        Returns:
            Quantity array of radii in pc
        """
        t_s_val = t_array.to(u.s).value
        R_cm = self.r_s * np.power(1.0 + 7.0 * t_s_val / (4.0 * self.t_s), 4.0 / 7.0)
        return (R_cm * u.cm).to(u.pc)
