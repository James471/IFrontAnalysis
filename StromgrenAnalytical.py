import numpy as np
from scipy.integrate import solve_ivp
import astropy.units as u

class StromgrenAnalytical:
    def __init__(self, Q, n_H, alpha_B):
        self.Q = Q
        self.n_H = n_H
        self.alpha_B = alpha_B
        self.r_s = (3 * Q / (4 * np.pi * alpha_B * n_H**2))**(1/3)
        self.V_s = self.r_s**3 * (4/3) * np.pi
        self.t_rec = 1 / (alpha_B * n_H)

    def get_analytical_radius_history_causal(self, t_array):
        def rhs(t, R, Q, n_H):
            alpha_B = 2.6e-13
            c = 3e10
            num = Q - (4 / 3) * np.pi * R**3 * alpha_B * n_H**2
            den = Q / c + 4 * np.pi * R**2 * n_H
            return num / den

        t_span = (0, 5 * self.t_rec.value)
        R0 = [0]

        sol = solve_ivp(rhs, t_span, R0, args=(self.Q.value, self.n_H.value), dense_output=True)
        t_eval = t_array.to(u.s).value
        R_eval = sol.sol(t_eval)[0] * u.cm
        return R_eval.to(u.pc)