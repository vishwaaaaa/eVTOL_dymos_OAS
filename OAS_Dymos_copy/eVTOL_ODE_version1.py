########################### This scripts has the ODE for eVTOL trajectory optimization using OAS ########################
# by Vishwa Mohan Tiwari 
# This Group is contains
#	- AeroForce component (from OpenAeroStruct)
# 	- Dynamics (eVTOL dynamics by shamsheer, without the CL and CD models)

import numpy as np
import openmdao.api as om
from aero import AeroForce

"""
eVTOL ODE model
"""
input_arg_1 = 0.0

input_arg_2 = 'ns'

# Some specifications
prop_rad = 0.75
wing_S = 9.
wing_span = 6.
num_blades = 3.
blade_chord = 0.1
num_props = 8

### Initialization dictionary ########
input_dict = {'T_guess': 9.8 * 725 * 1,  # initial thrust guess
				'x_dot_initial': 67.,  # initial horizontal speed
				'y_dot_initial': 0.,  # initial vertical speed
				'y_initial': 305,  # initial vertical displacement
				'A_disk': np.pi * prop_rad ** 2 * num_props,  # total propeller disk area
				'AR': wing_span ** 2 / (0.5 * wing_S),  # aspect ratio of each wing
				'e': 0.68,  # span efficiency factor of each wing
				't_over_c': 0.12,  # airfoil thickness-to-chord ratio
				'S': wing_S,  # total wing reference area
				'CD0': 0.35 / wing_S,  # coefficient of drag of the fuselage, gear, etc.
				'm': 715.,  # mass of aircraft
				'a0': 5.9,  # airfoil lift-curve slope
				'alpha_stall': 15. / 180. * np.pi,  # wing stall angle
				'rho': 1.225,  # air density
				'induced_velocity_factor': int(input_arg_1) / 100.,  # induced-velocity factor
				'stall_option': input_arg_2,  # stall option: 's' allows stall, 'ns' does not
				'R': prop_rad,  # propeller ra

				'solidity': num_blades * blade_chord / np.pi / prop_rad,  # solidity
				'omega': 136. / prop_rad,  # angular rotation rate
				'prop_CD0': 0.012,  # CD0 for prop profile power
				'k_elec': 0.9,  # electrical and mechanical losses factor
				'k_ind': 1.2,  # induced-losses factor
				'nB': num_blades,  # number of blades per propeller
				'bc': blade_chord,  # representative blade chord
				'n_props': num_props  # number of propellers
				}
######################
class eVTOL2DODE(om.Group):
    """
    Computes the aerodynamic force in the wind frame, 
    
    Inputs
    ----------
    v : number of nodes
        nn
    aero_model : str
    #       simple
    OAS_surface : dict
        surface
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('aero_model', types=str, default='simple')  # "simple" for "OAS_aero" or "OAS_AS"
        self.options.declare('OAS_surface', types=dict, default={}, desc='Surface dict for OAS')    # not needed if aero_model == "simple"
        self.options.declare('input_dict',types=dict,default={}, desc='input dictionary for the parameters of eVTOL dynamics')
    def setup(self):
        nn = self.options['num_nodes']  # number of nodes

        # --- aerodynamic model ---
        # input: v (airspeed), alpha, S (ref area). (not connecting density!)
        # outputs: f_lift, f_drag
        if self.options['aero_model'] == "simple":
            # use simple aero model (linear CL slope & polar drag)
            self.add_subsystem(name='aero', subsys=AeroForce(num_nodes=nn), promotes_inputs=['v', 'alpha', ('S', 'Sref')])
        
        else:
            raise RuntimeError('Option `aero_model` must be `simple`')

        # --- flight dynamics ---
        # input: m, v, gam, chi, alpha, theta, L, D, thrust
        # output: x_dot, y_dot, z_dot, v_dot, gam_dot, chi_dot
        self.add_subsystem(name='fd',
                           subsys=Dynamics(num_nodes=nn,input_dict=input_dict), promotes_inputs=['vx', 'vy', 'theta', 'power'])# promotes_inputs=['m', 'v', 'gam', 'chi', 'alpha', 'theta', ('T', 'thrust')]
        self.connect('aero.CD', 'fd.CD')
        self.connect('aero.CL', 'fd.CL')
        #self.connect('aero.aoa', 'fd.aoa')
        # self.add_input('CL', val=np.ones(num_nodes))  # state rates for x
        # self.add_input('CD', val=np.ones(num_nodes))

class Dynamics(om.ExplicitComponent):
    """
    This is the OpenMDAO component that takes the design variables and computes
    the objective function and other quantities of interest.

    Parameters
    ----------
    powers : array
        Electrical power distribution as a function of time
    thetas : array
        Wing-angle-to-vertical distribution as a function of time
    flight_time : float
        Duration of flight

    Returns
    -------
    x_dot : float
        Final horizontal speed
    y_dot : float
        Final vertical speed
    x : float
        Final horizontal position
    y : float
        Final vertical position
    y_min : float
        Minimum vertical displacement
    u_prop_min : float
        Minimum propeller freestream inflow velocity
    energy : float
        Electrical energy consumed
    aoa_max : float
        Maximum effective angle of attack
    aoa_min : float
        Minimum effective angle of attack
    acc_max : float
        Maximum acceleration magnitude
    """

    def initialize(self):
        # declare the input dict provided in the run script
        self.options.declare('input_dict', types=dict)
        self.options.declare('num_nodes', types=int)

    def setup(self):
        input_dict = self.options['input_dict']
        num_nodes = self.options['num_nodes']

        # give variable names to user-specified values from input dict
        self.x_dot_initial = input_dict['x_dot_initial']  # initial horizontal speed
        self.y_dot_initial = input_dict['y_dot_initial']  # initial vertical speed
        self.y_initial = input_dict['y_initial']  # initial vertical displacement
        self.A_disk = input_dict['A_disk']  # total propeller disk area
        self.T_guess = input_dict['T_guess']  # initial thrust guess
        self.alpha_stall = input_dict['alpha_stall']  # wing stall angle
        self.CD0 = input_dict['CD0']  # coefficient of drag of the fuselage, gear, etc.
        self.AR = input_dict['AR']  # aspect ratio
        self.e = input_dict['e']  # span efficiency factor of each wing
        self.rho = input_dict['rho']  # air density
        self.S = input_dict['S']  # total wing reference area
        self.m = input_dict['m']  # mass of aircraft
        self.a0 = input_dict['a0']  # airfoil lift-curve slope
        self.t_over_c = input_dict['t_over_c']  # airfoil thickness-to-chord ratio
        self.v_factor = input_dict['induced_velocity_factor']  # induced-velocity factor
        self.stall_option = input_dict['stall_option']  # stall option: 's' allows stall, 'ns' does not
        # self.num_steps = input_dict['num_steps']  # number of time steps
        self.R = input_dict['R']  # propeller radius
        self.solidity = input_dict['solidity']  # propeller solidity
        self.omega = input_dict['omega']  # propeller angular speed
        self.prop_CD0 = input_dict['prop_CD0']  # CD0 for propeller profile power
        self.k_elec = input_dict['k_elec']  # factor for mechanical and electrical losses
        self.k_ind = input_dict['k_ind']  # factor for induced losses
        self.nB = input_dict['nB']  # number of blades per propeller
        self.bc = input_dict['bc']  # representative blade chord
        self.n_props = input_dict['n_props']  # number of propellers

        self.quartic_poly_coeffs, pts = give_curve_fit_coeffs(self.a0, self.AR, self.e)

        # openmdao innameputs to the component
        self.add_input('power', val=np.ones(num_nodes))  # control
        self.add_input('theta', val=np.ones(num_nodes))  # control
        # openmdao outputs from the component

        # state history inputs that some calcs need
        self.add_input('vx', val=np.ones(num_nodes))
        self.add_input('vy', val=np.ones(num_nodes))

		# From the Aero component of the ODE Group

        self.add_input('CL', val=np.ones(num_nodes))  # state rates for x
        self.add_input('CD', val=np.ones(num_nodes))

        self.add_output('x_dot', val=np.ones(num_nodes))  # state rates for x
        self.add_output('y_dot', val=np.ones(num_nodes))  # state rates for y

        self.add_output('a_x', val=np.ones(num_nodes))  # state rates for v_x
        self.add_output('a_y', val=np.ones(num_nodes))  # state rates for v_y

        self.add_output('energy_dot', val=np.ones(num_nodes))  # state rates for energy

        # additional intermediate variables we want to track
        self.add_output('acc', val=np.ones(num_nodes))
        self.add_output('atov', val=np.ones(num_nodes))
        #self.add_output('CL', val=np.ones(num_nodes))
        #self.add_output('CD', val=np.ones(num_nodes))
        self.add_output('L_wings', val=np.ones(num_nodes))
        self.add_output('D_wings', val=np.ones(num_nodes))
        self.add_output('D_fuse', val=np.ones(num_nodes))
        self.add_output('aoa', val=np.ones(num_nodes))
        self.add_output('aoa_prop', val=np.ones(num_nodes))
        self.add_output('v_i', val=np.ones(num_nodes))
        self.add_output('N', val=np.ones(num_nodes))
        self.add_output('thrust', val=np.ones(num_nodes))
        self.add_output('u_inf_prop', val=np.ones(num_nodes))

        # some internal variables variables
        # self.thrusts = np.ones(num_nodes, dtype=complex) # thrusts
        # self.atov = np.ones(num_nodes, dtype=complex) # freestream angles to vertical
        # self.CL = np.zeros(num_nodes, dtype=complex) # wing lift coefficients
        # self.CD = np.zeros(num_nodes, dtype=complex) # wing drag coefficients

        # self.aoa = np.zeros(num_nodes, dtype=complex) # effective wing angles of attack

        # use complex step for partial derivatives
        self.declare_partials('*', '*', method='cs')
        self.declare_coloring(method='cs', per_instance=True, show_sparsity=True, show_summary=True)

        # Partial derivative coloring
        self.declare_coloring(wrt=['*'], method='cs', tol=1.0E-15, num_full_jacs=5,
                              show_summary=True, show_sparsity=True, min_improve_pct=10.)

    def compute(self, inputs, outputs):
        thrust = self.T_guess

        # time integration
        for i in range(self.options['num_nodes']):
            power = inputs['power'][i]
            theta = inputs['theta'][i]
            x_dot = inputs['vx'][i]
            y_dot = inputs['vy'][i]
            CL = inputs['CL'][i]
            CD = inputs['CD'][i]

            # the freestream angle relative to the vertical is
            atov = c_atan2(x_dot, y_dot)
            # the freestream speed is
            v_inf = (x_dot ** 2 + y_dot ** 2) ** 0.5

            outputs['u_inf_prop'][i] = u_inf_prop = v_inf * np.cos(atov - theta)
            u_parallel = v_inf * np.sin(atov - theta)

            mu = u_parallel / (self.omega * self.R)
            CP_profile = self.solidity * self.prop_CD0 / 8. * (1 + 4.6 * mu ** 2)
            P_disk = self.k_elec * power - CP_profile * (
                        self.rho * self.A_disk * (self.omega * self.R) ** 3)

			########################## Thrust Calc #################
            thrust, vi = Thrust(u_inf_prop, P_disk, self.A_disk, thrust, self.rho, self.k_ind)
            outputs['thrust'][i] = thrust


			############### NORMAL FORCE #####################
            Normal_F = Normal_force(v_inf, self.R, thrust / self.n_props, atov - theta, self.rho,
                                    self.nB, self.bc)

            # step = change(atov, v_inf, dt, theta, thrust, self.alpha_stall, self.CD0, self.AR, self.e, self.rho, self.S, self.m, self.a0, self.t_over_c, self.quartic_poly_coeffs, vi, self.v_factor, self.n_props * Normal_F)
            step = change(atov, v_inf, theta, thrust, self.alpha_stall, self.CD0, self.AR, self.e,
                          self.rho, self.S, self.m, self.a0, self.t_over_c,
                          self.quartic_poly_coeffs, vi, self.v_factor, self.n_props * Normal_F,CL,CD)
#np.array([delta_xdot, delta_ydot, aoa_blown, L, D_wings, D_fuse])
            outputs['acc'][i] = ((step[0]) ** 2 + (step[1]) ** 2) ** 0.5 / 9.81
            outputs['atov'][i] = atov
            #outputs['CL'][i] = step[2]
            #outputs['CD'][i] = step[3]

            outputs['v_i'][i] = vi * self.v_factor
            outputs['aoa'][i] = step[2]
            outputs['L_wings'][i] = step[3]
            outputs['D_wings'][i] = step[4]
            outputs['D_fuse'][i] = step[5]
            outputs['N'][i] = self.n_props * Normal_F
            outputs['aoa_prop'][i] = atov - theta

            outputs['x_dot'][i] = inputs['vx'][i]
            outputs['y_dot'][i] = inputs['vy'][i]

            outputs['a_x'][i] = step[0]
            outputs['a_y'][i] = step[1]

            outputs['energy_dot'][i] = power


################## Thrust and Normal Force calculation ###########
def Thrust(u0, power, A, T, rho, kappa):
    """
    This computes the thrust and induced velocity at the propeller disk.
    This uses formulas from propeller momentum theory.

    Parameters
    ----------
    u0 : float
        Freestream speed normal to the propeller disk
    power : float
        Power supplied to the propeller disk
    A : float
        Propeller disk area
    T : float
        Thrust guess
    rho : float
        Air density
    kappa: float
        Corection factor for non-uniform inflow and tip effects

    Returns
    -------
    thrust : float
        Thrust
    v_i : float
        Induced velocity at the propeller disk
    """

    T_old = T + 10.
    thrust = T

    # iteration loop to solve for the thrust as a function of power
    while np.abs(T_old.real - thrust.real) > 1e-10:
        T_old = thrust

        ### FPI (Fixed point iteration)
        # T_new = power / (u0 + kappa * (-u0/2 + 0.5 * (u0**2 + 2 * thrust / rho / A)**0.5))
        # thrust = thrust + (T_new - thrust) * 0.5

        # Newton-Raphson
        root_term = (u0 ** 2 + 2 * thrust / rho / A) ** 0.5
        R = power - thrust * (u0 + kappa * (-u0 / 2 + 0.5 * root_term))
        R_prime = -u0 - kappa * (-u0 / 2 + 0.5 * root_term + 0.5 * thrust / rho / A / root_term)
        thrust = T_old - R / R_prime

    # the induced velocity (i.e., velocity added at the disk) is
    v_i = (-u0 / 2 + (u0 ** 2 / 4. + thrust / 2 / rho / A) ** 0.5)

    if u0.real < 0:
        # This model is not valid if the freestream speed is negative
        print("FREESTREAM SPEED IS NEGATIVE!", thrust, v_i)

    return thrust, v_i


def Normal_force(u0, radius, thrust, alpha, rho, nB, bc):
    """
    This computes the normal force developed by each propeller due to the incidence angle of the flow.
    These equations are from "Propeller at high incidence" by de Young, 1965.

    Parameters
    ----------
    u0 : float
        Freestream speed
    radius : float
        Propeller radius
    thrust : float
        Propeller thrust
    alpha: float
        Incidence angle
    rho : float
        Air density
    nB : float
        Number of blades
    bc : float
        Effective blade chord

    Returns
    -------
    normal_force : float
        Normal force generated by one propeller
    """

    # conversion factor to convert from m to ft becasue these emperical formulas use imperial units
    m2f = 3.28

    # propeller 0.75R pitch angle as a function of freestream
    beta = 10 + u0 / 67. * 25

    u0 = u0 * m2f * np.cos(alpha)
    Diam = 2 * radius * m2f
    rho = rho * 0.00194
    c = bc * m2f

    q = 0.5 * rho * u0 ** 2
    A_d = np.pi * Diam ** 2 / 4.
    Tc = thrust / q / A_d
    f = 1 + 0.5 * ((1 + Tc) ** .5 - 1) + Tc / 4. / (2 + Tc)
    sigma = 4 * nB / 3 / np.pi * c / Diam
    slope = 4.25 * sigma / (1 + 2 * sigma) * np.sin(
        beta / 180. * np.pi + 8. / 180. * np.pi) * f * q * A_d

    normal_force = slope * np.tan(alpha) / 2.2046 * 9.81

    return normal_force


def change(atov, v_inf, theta, T, alpha_stall, CD0, AR, e, rho, S, m, a0, t_over_c, coeffs, v_i,
           v_factor, Normal_F, CL, CD):
    """
    This computes the change in velocity for each time step.

    Parameters
    ----------
    atov : float
        Freestream angle to the vertical
    v_inf : float
        Freestream speed
    dt : float
        Time step size
    theta : float
        Wing angle to the vertical
    T : float
        Thrust
    alpha_stall : float
        Stall angle of attack
    CD0 : float
        Parasite drag coefficient for fuse, gear, etc.
    AR : float
        Aspect ratio
    e : float
        Span efficiency factor
    rho : float
        Air density
    S : float
        Wing planform area
    m : float
        Mass of the aircraft
    a0 : float
        Airfoil lift-curve slope
    t_over_c : float
        Thickness-to-chord ratio
    coeffs : array
        Curve-fit polynomial coefficients for the drag coefficient below 27.5 deg
    v_i : float
        Induced-velocity value from the propellers
    v_factor : float
        Induced-velocity factor
    Normal_F : float
        Total propeller forces normal to the propeller axes
	CL : float
        Wing lift coefficient
    CD : float
        Wing drag coefficient

    Returns
    -------
    delta_xdot : float
        Change in horizontal velocity
    delta_ydot : float
        Change in vertical velocity
    aoa_blown : float
        Effective angle of attack with prop wash
    L : float
        Total lift force of the wings
    D_wings : float
        Total drag force of the wings
    D_fuse : float
        Drag force of the fuselage
    """

    # use angle of attack of wing to estimate CL and CD
    aoa = atov - theta
    v_chorwise = v_inf * np.cos(aoa)
    v_normal = v_inf * np.sin(aoa)

    v_chorwise += v_i * v_factor
    v_blown = (v_chorwise ** 2 + v_normal ** 2) ** 0.5
    aoa_blown = c_atan2(v_normal, v_chorwise)

    # CL = CLfunc(aoa_blown, alpha_stall, AR, e, a0, t_over_c)
    # CD = CDfunc(aoa_blown, AR, e, alpha_stall, coeffs, a0, t_over_c)

    # compute lift and drag forces
    L = 0.5 * rho * v_blown ** 2 * CL * S
    D_wings = 0.5 * rho * v_blown ** 2 * CD * S
    D_fuse = 0.5 * rho * v_inf ** 2 * CD0 * S

    # compute horizontal and vertical changes in velocity
    delta_xdot = (T * np.sin(theta) - D_fuse * np.sin(atov) - D_wings * np.sin(
        theta + aoa_blown) - L * np.cos(theta + aoa_blown) - Normal_F * np.cos(theta)) / m
    delta_ydot = (T * np.cos(theta) - D_fuse * np.cos(atov) - D_wings * np.cos(
        theta + aoa_blown) + L * np.sin(theta + aoa_blown) + Normal_F * np.sin(
        theta) - m * 9.81) / m

    return np.array([delta_xdot, delta_ydot, aoa_blown, L, D_wings, D_fuse])

def c_atan2(x, y):
    """ This is an arctan2 function that works with the complex-step method."""
    a = x.real
    b = x.imag
    c = y.real
    d = y.imag

    if np.iscomplex(x) or np.iscomplex(y):
        return complex(np.arctan2(a, c), (c * b - a * d) / (a ** 2 + c ** 2))
    else:
        return np.arctan2(a, c)
    

def give_curve_fit_coeffs(a0, AR, e):
    """
    This gives the coefficients for the quartic least-squares curve fit that is used for each wing's
    coefficient of drag below 27.5 deg.

    Parameters
    ----------
    a0 : float
        Airfoil lift-curve slope in 1/rad
    AR : float
        Aspect ratio
    e : float
        Span efficiency factor

    Returns
    -------
    quartic_poly_coeffs : array
        Coefficients of the curve fit
    data_pts : array
        Data points that are fitted
    """

    cla = a0 / (1 + a0 / (np.pi * e * AR))

    data_pts = np.array([[16. / 180. * np.pi, 0.1],  # Tangler--Ostowari points
                         [20. / 180. * np.pi, 0.175],  # Tangler--Ostowari points
                         [25. / 180. * np.pi, 0.275],  # Tangler--Ostowari points
                         [27.5 / 180. * np.pi, 0.363],  # Tangler--Ostowari points
                         [12. / 180. * np.pi,
                          0.015 + (cla * 12. / 180. * np.pi) ** 2 / np.pi / AR / e],
                         [10. / 180. * np.pi,
                          0.012 + (cla * 10. / 180. * np.pi) ** 2 / np.pi / AR / e],
                         [8. / 180. * np.pi,
                          0.0095 + (cla * 8. / 180. * np.pi) ** 2 / np.pi / AR / e],
                         [6. / 180. * np.pi,
                          0.008 + (cla * 6. / 180. * np.pi) ** 2 / np.pi / AR / e],
                         [4. / 180. * np.pi,
                          0.007 + (cla * 4. / 180. * np.pi) ** 2 / np.pi / AR / e],
                         [2. / 180. * np.pi,
                          0.0062 + (cla * 2. / 180. * np.pi) ** 2 / np.pi / AR / e],
                         [0. / 180. * np.pi, 0.006]])

    new_fit_matrix = np.array([[1, data_pts[0, 0] ** 2, data_pts[0, 0] ** 4],
                               [1, data_pts[1, 0] ** 2, data_pts[1, 0] ** 4],
                               [1, data_pts[2, 0] ** 2, data_pts[2, 0] ** 4],
                               [1, data_pts[3, 0] ** 2, data_pts[3, 0] ** 4],
                               [1, data_pts[4, 0] ** 2, data_pts[4, 0] ** 4],
                               [1, data_pts[5, 0] ** 2, data_pts[5, 0] ** 4],
                               [1, data_pts[6, 0] ** 2, data_pts[6, 0] ** 4],
                               [1, data_pts[7, 0] ** 2, data_pts[7, 0] ** 4],
                               [1, data_pts[8, 0] ** 2, data_pts[8, 0] ** 4],
                               [1, data_pts[9, 0] ** 2, data_pts[9, 0] ** 4],
                               [1, data_pts[10, 0] ** 2, data_pts[10, 0] ** 4]])

    quartic_poly_coeffs = np.linalg.solve(np.dot(new_fit_matrix.T, new_fit_matrix),
                                          np.dot(new_fit_matrix.T, data_pts[:, 1]))

    return quartic_poly_coeffs, data_pts