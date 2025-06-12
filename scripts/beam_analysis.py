import numpy as np
import matplotlib.pyplot as plt
from abc import ABC


class BeamAnalysis(ABC):
    """
    Class to analyze a rotating beam with variable cross-section and tip forces.
    
    :param L: Length of the beam [m]
    :param D_base: Outer diameter at base [m]
    :param D_tip: Outer diameter at tip [m]
    :param t_base: Thickness at base [m]
    :param t_tip: Thickness at tip [m]
    :param E: Young's modulus [Pa]
    :param rho: Density of material [kg/m³]
    :param M_app: Applied moment at pivot [Nm]
    :param Fy: Vertical force applied at tip [N]
    :param Fx: Horizontal force applied at tip [N]
    """

    def __init__(
        self, 
        L:float, 
        D_base:float, 
        D_tip:float,
        t_base:float, 
        t_tip:float,
        E:float, 
        rho:float, 
        ys:float = 0,
        M:float = 0,
        Fy:float = 0, 
        Fx:float = 0, 
        check_points:int = 10000,
        gravity:float = -9.81
        ):
        
        """
        Initialize the beam analysis with given parameters.
        
        :param L: Length of the beam [m]
        :param D_base: Outer diameter at base [m]
        :param D_tip: Outer diameter at tip [m]
        :param t_base: Thickness at base [m]
        :param t_tip: Thickness at tip [m]
        :param E: Young's modulus [Pa]
        :param rho: Density of material [kg/m³]
        :param M: Applied moment at pivot [Nm]
        :param Fy: Vertical force applied at tip [N]
        :param Fx: Horizontal force applied at tip [N]
        :param check_points: Number of points to evaluate along the beam
        """
        
        self.L = L
        self.D_base = D_base
        self.D_tip = D_tip
        self.t_base = t_base
        self.t_tip = t_tip
        self.E = E
        self.rho = rho
        self.M = M
        self.Fy = Fy
        self.Fx = Fx
        self.g = gravity
        self.ys = ys
        
        # Derived properties
        self.x = np.linspace(0, self.L, check_points)
        self.A = self.cross_section_area(self.x)
        self.m = self.mass()
        self.J = self.moment_of_inertia_at_pivot()

    def cross_section_area(self, x:float) -> float:
        """
        Calculate area A(x) at position x assuming linear variation
        
        :params x: Position along the beam [m]
        :return: Cross-sectional area at position x [m²]
        """
        
        D = self.D_base - (self.D_base - self.D_tip) * (x / self.L)
        t = self.t_base - (self.t_base - self.t_tip) * (x / self.L)
        r_ext = D / 2
        r_int = r_ext - t
        return np.pi * (r_ext**2 - r_int**2)
    
    def moment_of_inertia(self, x:float) -> float:
        """
        Calculate I(x) at position x assuming linear variation
        
        :param x: Position along the beam [m]
        :return: Moment of inertia at position x [m⁴]
        """
        
        D = self.D_base - (self.D_base - self.D_tip) * (x / self.L)
        t = self.t_base - (self.t_base - self.t_tip) * (x / self.L)
        r_ext = D / 2
        r_int = r_ext - t
        return (np.pi / 4) * (r_ext**4 - r_int**4)
    
    def mass(self) -> float:
        """
        Total mass of the beam assuming linear variation in section
        
        :return: Mass of the beam [kg]
        """
        
        return np.trapezoid(self.A, self.x) * self.rho
    
    def moment_of_inertia_at_pivot(self) -> float:
        """
        Moment of inertia around pivot using parallel axis theorem
        
        :return: Moment of inertia at pivot [kg·m²]
        """
        
        return np.trapezoid(self.A*(self.x**2), self.x) * self.rho
    
    def angular_acceleration(self) -> float:
        """
        Compute angular acceleration due to total moment
        
        :return: Angular acceleration [rad/s²]
        """
        
        weight_moment = self.g * np.trapezoid(self.A*(self.x), self.x) * self.rho
        force_moment = self.Fy * self.L
        total_moment = self.M + weight_moment + force_moment
        return total_moment / self.J
    
    def internal_forces(self, x:np.ndarray) -> tuple:
        """Calculate bending moment M(x) and shear V(x) along the beam
        
        :param x: Array of positions along the beam [m]
        :return: Tuple of bending moment M(x) and shear force V(x) at each position
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        alpha = self.angular_acceleration()
        
        # Distributed load due to inertia and gravity at each point x
        p = self.rho * self.cross_section_area(x) * (- alpha * x + self.g)
        
        V = np.zeros_like(x)
        M = np.zeros_like(x)
        
        for i in range(len(x)):
            # Shear force from distributed load (from x[i] to tip)
            V_dist = np.trapezoid(p[i:], x[i:])
            V_point = self.Fy  # Vertical concentrated force at tip
            V[i] = V_dist + V_point
            
            # Bending moment from distributed load
            dx = x[i:] - x[i]
            M_dist = np.trapezoid(p[i:] * dx, x[i:])
            
            # Moment from concentrated force
            dist_from_tip = self.L - x[i]
            M_point = self.Fy * dist_from_tip
            
            M[i] = M_dist + M_point
        
        return M, V

    def stresses_and_strain(self, x:np.ndarray) -> tuple:
        """
        Calculate max stress and strain along the beam 
        
        :param x: Array of positions along the beam [m]
        :return: Tuple of stress and strain at each position
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        
        M, V = self.internal_forces(x)
        sigma = np.zeros_like(M)
        epsilon = np.zeros_like(M)

        for i in range(len(x)):
            D_at_x = self.D_base - (self.D_base - self.D_tip) * (x[i] / self.L) # Calculate D(x)
            y_max_at_x = D_at_x / 2
            I = self.moment_of_inertia(x[i])
            sigma[i] = (M[i] * y_max_at_x / I) # Use y_max_at_x
            epsilon[i] = sigma[i] / self.E
        return sigma, epsilon

    def elastic_curve(self, x:np.ndarray) -> np.ndarray:
        """Calculate elastic curve using curvature integration
        
        :param x: Array of positions along the beam [m]
        :return: Tuple of positions and deflections along the beam
        :rtype: np.ndarray
        """
        EI = np.array([self.E * self.moment_of_inertia(xi) for xi in x])
        M, _ = self.internal_forces(x)
        curvature = M / EI
        v = np.zeros_like(x)
        theta = np.zeros_like(x)
        
        for i in range(1, len(x)):
            theta[i] = theta[i-1] + curvature[i-1] * (x[i] - x[i-1])
            v[i] = v[i-1] + theta[i] * (x[i] - x[i-1])
        
        return v
    
    def absorbed_energy(self) -> float:
            """
            Calculate the total elastic strain energy absorbed by the beam.
            
            :return: Total absorbed energy [J]
            """
            M, _ = self.internal_forces(self.x)
            EI = np.array([self.E * self.moment_of_inertia(xi) for xi in self.x])
            
            # Strain energy density (dU) = (M^2) / (2 * EI)
            strain_energy_density = (M**2) / (2 * EI)
            
            # Integrate along the beam length to get total energy
            total_energy = np.trapezoid(strain_energy_density, self.x)
            
            return total_energy
    
    def plot_results(self, suffix:str = ""):
        """
        Generates plots for bending moment, max stress, and elastic curve.
        """
        
        M, _ = self.internal_forces(self.x)
        sigma, _ = self.stresses_and_strain(self.x)
        v = self.elastic_curve(self.x)
        
        # Bending Moment
        axs[0].plot(self.x, M, label=suffix)
        axs[0].set_title("Bending Moment $M(x)$")
        axs[0].set_ylabel("Moment [Nm]")
        axs[0].grid(True)
        axs[0].legend()

        # Max Stress
        axs[1].plot(self.x, sigma*1e-6, label=suffix)
        axs[1].set_title("Max Stress $\\sigma_{max}(x)$")
        axs[1].set_ylabel("Stress [MPa]")
        axs[1].grid(True)
        axs[1].legend()

        # Elastic Curve
        axs[2].plot(self.x, v*1e3, label=suffix)
        axs[2].set_title("Elastic Curve")
        axs[2].set_ylabel("Deflection [mm]")
        axs[2].set_xlabel("Position along beam [m]")
        axs[2].grid(True)
        axs[2].legend()
        
    def print_summary(self, material_name:str = ""):
        """Print structural analysis summary"""
        
        alpha = self.angular_acceleration()
        print(f"Structural Analysis Summary ({material_name})")
        print("-"*40)
        # print(f"Beam length: {self.L} m")
        print(f"Total rod mass: {self.m:.2f} kg")
        # print(f"Applied torque: {self.M} Nm")
        # print(f"Vertical tip force: {self.Fy} N")
        # print(f"Horizontal tip force: {self.Fx} N")
        # print(f"Moment of inertia at pivot (J): {self.J:.3f} kg·m²")
        # print(f"Moment of inertia at root: {self.moment_of_inertia(0):.10f} m⁴")
        print(f"Angular acceleration: {alpha:.4f} rad/s²")
        print(f"Max bending moment: {max(np.abs(self.internal_forces(self.x)[0])):.2f} Nm")
        print(f"Max stress: {max(np.abs(self.stresses_and_strain(self.x)[0]))*1e-6:.2f} MPa")
        print(f"Max deflection: {max(np.abs(self.elastic_curve(self.x)))*1e3:.3f} mm")
        print(f"Total absorbed energy: {self.absorbed_energy():.6f} J") # Added line
        print(f"Security factor: {self.ys / max(np.abs(self.stresses_and_strain(self.x)[0])):.2f}")

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    
    # Define common beam parameters
    common_params = {
        "L": 0.8,
        "D_base": 25 * 1e-3,
        "D_tip": 20 * 1e-3,
        "t_base": 2 * 1e-3,
        "t_tip": 2 * 1e-3,
        "M": 5,
        "Fy": -2.0,
        "Fx": 0,
        "gravity": -9.81,
    }
    
    # Define material properties
    materials = {
        "Steel": {"E": 200e9, "rho": 7850, "ys": 250e6},
        "Aluminum 6061-T6": {"E": 70e9, "rho": 2700, "ys": 240e6},
        "Fiberglass ": {"E": 22e9, "rho": 1750, "ys": 30e6},
        "Carbon Fiber": {"E": 150e9, "rho": 1800, "ys": 4830e6},
        "PVC": {"E": 3e9, "rho": 1400, "ys": 50e6},

    }

    fig, axs = plt.subplots(3, 1, figsize=(8, 7))
    axs:list[plt.Axes] = axs.flatten()
    
    for material_name, props in materials.items():
        
        print("="*40)
        beam_params = {**common_params, **props} # Merge common params with material-specific params
        beam = BeamAnalysis(**beam_params)
        
        beam.print_summary(material_name=material_name)
        beam.plot_results(suffix=f"{material_name}") # Uncomment to plot for each material    
    
    plt.show()