import openmdao.api as om
import dymos as dm
import numpy as np
from openaerostruct.geometry.utils import generate_mesh



import sys

#sys.path.insert(0, "../../ode")

from evtol_dynamics_comp import Dynamics

from eVTOL_ODE_version1 import eVTOL2DODE

sys.path.insert(0,"../../ode")

import verify_data

if __name__ == '__main__':
    input_arg_1 = 0.0

    input_arg_2 = 'ns'

    # Some specifications
    prop_rad = 0.75
    wing_S = 9.
    wing_span = 6.
    num_blades = 3.
    blade_chord = 0.1
    num_props = 8

    # User-specified input dictionary
    input_dict = {'T_guess': 9.8 * 725 * 1.2,  # initial thrust guess
                  'x_dot_initial': 0.,  # initial horizontal speed
                  'y_dot_initial': 0.01,  # initial vertical speed
                  'y_initial': 0.01,  # initial vertical displacement
                  'A_disk': np.pi * prop_rad ** 2 * num_props,  # total propeller disk area
                  'AR': wing_span ** 2 / (0.5 * wing_S),  # aspect ratio of each wing
                  'e': 0.68,  # span efficiency factor of each wing
                  't_over_c': 0.12,  # airfoil thickness-to-chord ratio
                  'S': wing_S,  # total wing reference area
                  'CD0': 0.35 / wing_S,  # coefficient of drag of the fuselage, gear, etc.
                  'm': 725.,  # mass of aircraft
                  'a0': 5.9,  # airfoil lift-curve slope
                  'alpha_stall': 15. / 180. * np.pi,  # wing stall angle
                  'rho': 1.225,  # air density
                  'induced_velocity_factor': int(input_arg_1) / 100.,  # induced-velocity factor
                  'stall_option': input_arg_2,  # stall option: 's' allows stall, 'ns' does not
                  'R': prop_rad,  # propeller radius
                  'solidity': num_blades * blade_chord / np.pi / prop_rad,  # solidity
                  'omega': 136. / prop_rad,  # angular rotation rate
                  'prop_CD0': 0.012,  # CD0 for prop profile power
                  'k_elec': 0.9,  # electrical and mechanical losses factor
                  'k_ind': 1.2,  # induced-losses factor
                  'nB': num_blades,  # number of blades per propeller
                  'bc': blade_chord,  # representative blade chord
                  'n_props': num_props  # number of propellers
                  }
    input_dict_d = {'T_guess': 9.8 * 725 * 1,  # initial thrust guess
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
################################### Aero Dictionary ##########################

    # weight
    W0 = 2.0  # total weight, electric (weight does not change)

    # Create a dictionary to store options about the surface
    mesh_dict = {
        "num_y": 5,
        "num_x": 2,
        "wing_type": "rect",
        "symmetry": True,
        "span": 3.0,
        "chord": 0.3,
        "span_cos_spacing": 1.0,
        "chord_cos_spacing": 1.0,
    }

    # Generate half-wing mesh of rectangular wing
    mesh = generate_mesh(mesh_dict)

    # spanwise control points
    ncp = 5

    # Create a dictionary with info and options about the aerodynamic lifting surface
    surface = {
        # Wing definition
        "name": "wing",  # name of the surface
        "symmetry": True,  # if true, model one half of wing reflected across the plane y = 0
        "S_ref_type": "wetted",  # how we compute the wing area,  can be 'wetted' or 'projected'
        "fem_model_type": "tube",
        "twist_cp": np.zeros((ncp)),    # [deg]
        "thickness_cp": np.ones((ncp)) * 0.003,  # [meters]
        "radius_cp": np.ones(ncp) * 0.03,
        "mesh": mesh,
        # Aerodynamic performance of the lifting surface at
        # an angle of attack of 0 (alpha=0).
        # These CL0 and CD0 values are added to the CL and CD
        # obtained from aerodynamic analysis of the surface to get
        # the total CL and CD.
        # These CL0 and CD0 values do not vary wrt alpha.
        "CL0": 0.0,  # CL of the surface at alpha=0
        "CD0": 0.015,  # CD of the surface at alpha=0
        # Airfoil properties for viscous drag calculation
        "k_lam": 0.05,  # percentage of chord with laminar
        # flow, used for viscous drag
        "t_over_c_cp": np.array([0.15]),  # thickness over chord ratio (NACA0015)
        "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
        # thickness
        "with_viscous": True,
        "with_wave": False,  # if true, compute wave drag
        # Material properties taken from http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp
        "E": 85.0e9,
        "G": 25.0e9,
        "yield": 350.0e6,
        "mrho": 1.6e3,
        "fem_origin": 0.35,  # normalized chordwise location of the spar
        "wing_weight_ratio": 1.0,  # multiplicative factor on the computed structural weight
        "struct_weight_relief": True,  # True to add the weight of the structure to the loads on the structure
        "distributed_fuel_weight": False,
        # Constraints
        "exact_failure_constraint": False,  # if false, use KS function
    }

    p = om.Problem()

    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    # phase = dm.Phase(transcription=dm.GaussLobatto(num_segments=10, order=3, solve_segments=False,
    #                                                compressed=False),
    #                  ode_class=Dynamics,
    #                  ode_init_kwargs={'input_dict': input_dict})  ##the vectorized dynamics

    # traj.add_phase('phase0', phase)

    # phase.set_time_options(fix_initial=True, duration_bounds=(5, 60), duration_ref=30)
    # phase.add_state('x', fix_initial=True, rate_source='x_dot', ref0=0, ref=900, defect_ref=100)
    # phase.add_state('y', fix_initial=True, rate_source='y_dot', ref0=0, ref=300, defect_ref=300)
    # phase.add_state('vx', fix_initial=True, rate_source='a_x', ref0=0, ref=10)
    # phase.add_state('vy', fix_initial=True, rate_source='a_y', ref0=0, ref=10)
    # phase.add_state('energy', fix_initial=True, rate_source='energy_dot', ref0=0, ref=1E7, defect_ref=1E5)

    # phase.add_control('power', lower=1e3, upper=311000, ref0=1e3, ref=311000, rate_continuity=False)
    # phase.add_control('theta', lower=0., upper=3 * np.pi / 4, ref0=0, ref=3 * np.pi / 4,
    #                   rate_continuity=False)

    # phase.add_timeseries_output(['CL', 'CD'])

    # # Objective
    # # phase.add_objective('energy', loc='final', ref0=0, ref=1E7)

    # # Boundary Constraints
    # phase.add_boundary_constraint('y', loc='final', lower=305,
    #                               ref=100)  # Constraint for the final vertical displacement
    # phase.add_boundary_constraint('x', loc='final', equals=900,
    #                               ref=100)  # Constraint for the final horizontal displacement
    # phase.add_boundary_constraint('x_dot', loc='final', equals=67.,
    #                               ref=100)  # Constraint for the final horizontal speed

    # # Path Constraints
    # phase.add_path_constraint('y', lower=0., upper=305,
    #                           ref=300)  # Constraint for the minimum vertical displacement
    # phase.add_path_constraint('acc', upper=0.3,
    #                           ref=1.0)  # Constraint for the acceleration magnitude
    # phase.add_path_constraint('aoa', lower=-np.radians(15), upper=np.radians(15), ref0=-np.radians(15),
    #                           ref=np.radians(15))  # Constraint for the angle of attack
    # phase.add_path_constraint('thrust', lower=10, ref0=10,
    #                           ref=100)  # Constraint for the thrust magnitude

#####################################################################
    tx = dm.GaussLobatto(num_segments=10, order=3, solve_segments=False,
                                                   compressed=False)
    nn = tx.grid_data.num_nodes
    print('pass');
    phase1 = dm.Phase(transcription=dm.GaussLobatto(num_segments=10, order=3, solve_segments=False,
                                                   compressed=False),
                     ode_class=eVTOL2DODE,
                     ode_init_kwargs={'aero_model':'simple','OAS_surface':surface,'input_dict': input_dict_d})  ##the vectorized dynamics
    print("Pass");
    traj.add_phase('phase1', phase1)

    phase1.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),duration_ref=100, units='s')
    phase1.add_state('x', fix_initial=False, rate_source='fd.x_dot', ref0=900, ref=1400, defect_ref=100)
    phase1.add_state('y', fix_initial=False, rate_source='fd.y_dot', ref0=0, ref=305, defect_ref=300)
    phase1.add_state('vx', fix_initial=False, rate_source='fd.a_x', ref0=0, ref=10)
    phase1.add_state('vy', fix_initial=False, rate_source='fd.a_y', ref0=0, ref=10)
    phase1.add_state('energy', fix_initial=False, rate_source='fd.energy_dot', ref0=0, ref=1E7, defect_ref=1E5)

    phase1.add_control('power', lower=1e3, upper=311000, ref0=1e3, ref=311000, rate_continuity=False)
    phase1.add_control('theta', lower=0., upper=3 * np.pi / 4, ref0=0, ref=3 * np.pi / 4,
                      rate_continuity=False)

    #phase1.add_timeseries_output(['CL', 'CD'])

    # Objective
    phase1.add_objective('energy', loc='final', ref0=0, ref=1E7)

    # Boundary Constraints
    phase1.add_boundary_constraint('y', loc='initial', lower=305,
                                  ref=100)  # Constraint for the final vertical displacement
    phase1.add_boundary_constraint('x', loc='initial', equals=900,
                                  ref=100)  # Constraint for the final horizontal displacement
    phase1.add_boundary_constraint('fd.x_dot', loc='initial', equals=67.,
                                  ref=100)  # Constraint for the final horizontal speed
    phase1.add_boundary_constraint('y', loc='final', lower=305,
                                  ref=100)  # Constraint for the final vertical displacement
    phase1.add_boundary_constraint('x', loc='final', equals=1400,
                                  ref=100)  # Constraint for the final horizontal displacement
    phase1.add_boundary_constraint('fd.x_dot', loc='final', equals=67.,
                                  ref=100)  # Constraint for the final horizontal speed

    # Path Constraints
    phase1.add_path_constraint('y', lower=305., upper=305,
                              ref=300)  # Constraint for the minimum vertical displacement
    phase1.add_path_constraint('fd.acc', upper=0.3,
                              ref=1.0)  # Constraint for the acceleration magnitude
    phase1.add_path_constraint('fd.aoa', lower=-np.radians(15), upper=np.radians(15), ref0=-np.radians(15),
                              ref=np.radians(15))  # Constraint for the angle of attack
    phase1.add_path_constraint('fd.thrust', lower=10, ref0=10,
                              ref=100)  # Constraint for the thrust magnitude
    phase1.add_path_constraint('vx', lower=10, ref0=10,
                               ref=100)  # Constraint for the thrust magnitude
    
    # Link Phases (link time and all state variables)
    # traj.link_phases(phases=['phase0', 'phase1'], vars=['*','power','theta'])

    # # Setup the driver
    p.driver = om.pyOptSparseDriver()

    # p.driver.options['optimizer'] = 'SNOPT'
    # p.driver.opt_settings['Major optimality tolerance'] = 1e-4
    # p.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    # p.driver.opt_settings['Major iterations limit'] = 1000
    # p.driver.opt_settings['Minor iterations limit'] = 100_000_000
    # p.driver.opt_settings['iSumm'] = 6

    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.opt_settings['max_iter'] = 1000
    p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
    p.driver.opt_settings['print_level'] = 5
    p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    p.driver.opt_settings['tol'] = 5.0E-5

    p.driver.declare_coloring(tol=1.0E-8)

    p.setup()

    # p.set_val('traj.phase0.t_initial', 0.0)
    # p.set_val('traj.phase0.t_duration', 30)
    # p.set_val('traj.phase0.states:x', phase.interpolate(ys=[0, 900], nodes='state_input'))
    # p.set_val('traj.phase0.states:y', phase.interpolate(ys=[0.01, 300], nodes='state_input'))
    # p.set_val('traj.phase0.states:vx', phase.interpolate(ys=[0, 60], nodes='state_input'))
    # p.set_val('traj.phase0.states:vy', phase.interpolate(ys=[0.01, 10], nodes='state_input'))
    # p.set_val('traj.phase0.states:energy', phase.interpolate(ys=[0, 1E7], nodes='state_input'))

    # p.set_val('traj.phase0.controls:power', phase.interpolate(xs=np.linspace(0, 28.368, 500),
    #                                                           ys=verify_data.powers.ravel(),
    #                                                           nodes='control_input'))
    # p.set_val('traj.phase0.controls:theta', phase.interpolate(xs=np.linspace(0, 28.368, 500),
    #                                                           ys=verify_data.thetas.ravel(),
    #                                                           nodes='control_input'))

    # p.set_val('traj.phase0.controls:power', 200000.0)
    # p.set_val('traj.phase0.controls:theta', phase.interpolate(ys=[0.001, np.radians(85)], nodes='control_input'))
    
    p.set_val('traj.phase1.t_initial', 30.0)
    p.set_val('traj.phase1.t_duration', 30)
    p.set_val('traj.phase1.states:x', phase1.interpolate(ys=[900, 1400], nodes='state_input'))
    p.set_val('traj.phase1.states:y', phase1.interpolate(ys=[300, 305], nodes='state_input'))
    p.set_val('traj.phase1.states:vx', phase1.interpolate(ys=[60, 60], nodes='state_input'))
    p.set_val('traj.phase1.states:vy', phase1.interpolate(ys=[0., 0], nodes='state_input'))
    p.set_val('traj.phase1.states:energy', phase1.interpolate(ys=[1E7, 2E7], nodes='state_input'))

    p.set_val('traj.phase1.controls:power', phase1.interpolate(xs=np.linspace(28.368, 2*28.368, 500),
                                                              ys=verify_data.powers.ravel(),
                                                              nodes='control_input'))
    p.set_val('traj.phase1.controls:theta', phase1.interpolate(xs=np.linspace(28.368, 2*28.368, 500),
                                                              ys=verify_data.thetas.ravel(),
                                                              nodes='control_input'))

    p.set_val('traj.phase1.controls:power', 200000.0)
    p.set_val('traj.phase1.controls:theta', phase1.interpolate(ys=[np.radians(85), np.radians(90)], nodes='control_input'))
    dm.run_problem(p, run_driver=True, simulate=True)
    om.n2(p)