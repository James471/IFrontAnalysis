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

        R(t) = r_ch * ( x_rad(tau)^p + x_gas(tau)^p )^(1/p),   p = (7 - k_rho)/2,

    with tau = t / t_ch and, for k_rho = 0 (uniform ambient medium),

        x_rad(tau) = (2 tau^2)^(1/4)
        x_gas(tau) = (49/36 tau^2)^(2/7)

    r_ch and t_ch are KM09's own characteristic radiation-pressure scales (their
    Eq. 9), NOT r_s/t_s. x_gas is the LATE-TIME ASYMPTOTE of the Spitzer solution
    (i.e. r_s * (7t / 4t_s)^(4/7), dropping the "1 +" term) expressed in units of
    r_ch -- it is NOT the full Spitzer solution get_spitzer_radius_history returns.
    Using the full "1 + 7t/4ts" Spitzer term here would double-count the initial
    R-type/transient growth already encoded in x_rad and break the exact reduction
    to the KM09 gas-pressure limit as t -> infinity. This mirrors the implementation
    in src/problems/DTypeFront/testDTypeFront.cpp (computeAfterTimestep), which does
    not expose f_trap, psi, or mu_amb: the ambient mass density feeding t_ch is
    n_H * m_HI (pure atomic hydrogen, no mean-molecular-weight factor) and the
    bolometric luminosity is fixed to the ionizing photon luminosity; this class
    matches that.

    Multi-group option (mg=True): boosts r_ch and t_ch to account for the extra
    momentum deposited by the reprocessed optical band in testDTypeFrontMG.cpp
    (SetRadEnergySource), which injects L_star_optical = optical_to_ionizing_fraction
    * L_star_ion alongside the ionizing luminosity L_star_ion = Q1 * e1. Writing
    Q2 * e2 = L_star_optical, the ratio Q2*e2 / (Q1*e1) = optical_to_ionizing_fraction, so

        r_ch(mg) = r_ch * (1 + Q2*e2 / (Q1*e1))^2
                 = r_ch * (1 + optical_to_ionizing_fraction)^2
        t_ch(mg) = t_ch / sqrt(1 + Q2*e2 / (Q1*e1))
                 = t_ch / sqrt(1 + optical_to_ionizing_fraction),

    where r_ch and t_ch on the right are the plain (non-mg) values computed from
    Q1, e1 alone (t_ch is NOT simply re-derived from the boosted r_ch via KM09's
    Eq. 9 relation -- that would apply the r_ch^4 boost to t_ch too, which is a
    different, much larger correction than the explicit factor above).
    """

    def __init__(self, Q, n_H, radiation_pressure=False, mg=False, optical_to_ionizing_fraction=0.1):
        """
        Parameters:
            Q:      Ionizing photon rate (s^-1), dimensionless float in CGS
            n_H:    Hydrogen number density (cm^-3), dimensionless float in CGS
            radiation_pressure: if True, get_analytical_radius_history_causal returns
                    the Krumholz & Matzner (2009) combined radiation + gas-pressure
                    solution instead of the pure Spitzer solution.
            mg:     if True, boost r_ch for the extra momentum deposited by the
                    reprocessed optical band (see class docstring). Only affects
                    the radiation-pressure solution.
            optical_to_ionizing_fraction: L_star_optical / L_star_ion, as set by
                    stromgen.optical_to_ionizing_fraction in testDTypeFrontMG.cpp.
                    Only used when mg=True.
        """
        self.Q = Q
        self.n_H = n_H
        self.radiation_pressure = radiation_pressure
        self.mg = mg
        self.optical_to_ionizing_fraction = optical_to_ionizing_fraction

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
        # eps is the mean photon energy of the ionizing band (matches
        # RadSystem::GetChemBandQuanta(0) in testDTypeFront.cpp).
        avg_freq = (3.29e15 + 1.50e16) / 2.0
        self.eps = Constants.hplanck * avg_freq
        self.rho0 = n_H * Constants.m_HI

        # KM09's own characteristic radiation-pressure scales (their Eq. 9), NOT r_s/t_s.
        self.r_ch = (Q * self.eps**2 * self.alpha_B
                     / (12.0 * np.pi * Constants.k_B**2 * self.T_eq**2 * Constants.c**2))
        self.t_ch = np.sqrt(4.0 * np.pi * self.rho0 * self.r_ch**4 * Constants.c
                             / (3.0 * Q * self.eps))
        if self.mg:
            self.r_ch = self.r_ch * (1.0 + self.optical_to_ionizing_fraction)**2
            self.t_ch = self.t_ch / np.sqrt(1.0 + self.optical_to_ionizing_fraction)

    def _spitzer_radius_cm(self, t_s_val):
        """Spitzer gas-pressure radius in cm, given time array in seconds (plain floats)."""
        return self.r_s * np.power(1.0 + 7.0 * t_s_val / (4.0 * self.t_s), 4.0 / 7.0)

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
        radius at each time in t_array (their Eq. 13, k_rho = 0), exactly mirroring
        testDTypeFront.cpp::computeAfterTimestep: tau = t / t_ch,
        x_rad = (2 tau^2)^(1/4), x_gas = (49/36 tau^2)^(2/7),
        R = r_ch * (x_rad^3.5 + x_gas^3.5)^(2/7).

        Note x_gas is the LATE-TIME ASYMPTOTE of the Spitzer solution (no "1 +"
        term), not get_spitzer_radius_history's full expression -- see class
        docstring.

        Parameters:
            t_array: array with astropy time units

        Returns:
            Quantity array of radii in pc
        """
        t_s_val = t_array.to(u.s).value
        tau = t_s_val / self.t_ch
        x_rad = np.power(2.0 * tau**2, 1.0 / 4.0)
        x_gas = np.power(49.0 / 36.0 * tau**2, 2.0 / 7.0)
        p = 3.5  # (7 - k_rho) / 2, k_rho = 0
        x = np.power(np.power(x_rad, p) + np.power(x_gas, p), 2.0 / 7.0)
        R_cm = self.r_ch * x
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
