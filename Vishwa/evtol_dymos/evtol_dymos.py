import openmdao.api as om
import dymos as dm
import numpy as np

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

import sys

sys.path.insert(0, "../ode")

from evtol_dynamics_comp_climb import Dynamics as D_climb
from evtol_dynamics_comp_cruise import Dynamics as D_cruise
from evtol_dynamics_comp_cruise import c_atan2

import verify_data

excomp1 = om.ExecComp('y = x1 - x2', x1=0, x2=0)
excomp2 = om.ExecComp('y = x1 - x2', x1=0, x2=0)

class AOAComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('vx', val=1.)
        self.add_input('vy', val=1.)
        self.add_input('theta', val=0.9*np.pi/2.0) # HARD CODED FOR NOW, ADJUST IF CHANGED
        self.add_output('aoa', val=1.)
        self.add_output('v_inf', val=1.)

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        x_dot = inputs["vx"]
        y_dot = inputs["vy"]
        theta = inputs["theta"]
        atov = c_atan2(x_dot, y_dot)
        outputs["aoa"] = (atov - theta) * 180. / np.pi
        outputs["v_inf"] = (x_dot**2 + y_dot**2) ** (0.5)

if __name__ == '__main__':
    # =========================================================================
    # Trajectory Setup
    # =========================================================================
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

    # =========================================================================
    # OAS Setup
    # =========================================================================
    # Create a dictionary to store options about the surface
    mesh_dict = {
        "num_y": 35,
        "num_x": 11,
        "wing_type": "rect",
        "symmetry": True,
        "span": 10.0,
        "chord": 1,
        "span_cos_spacing": 1.0,
        "chord_cos_spacing": 1.0,
    }

    # Generate half-wing mesh of rectangular wing
    mesh = generate_mesh(mesh_dict)

    # Define input surface dictionary for our wing
    surface = {
        # Wing definition
        "name": "wing",  # name of the surface
        "type": "aero",
        "symmetry": True,  # if true, model one half of wing
        # reflected across the plane y = 0
        "S_ref_type": "projected",  # how we compute the wing area,
        # can be 'wetted' or 'projected'
        # "chord_cp": np.ones(10),  # Define chord using 3 B-spline cp's
        # "twist_cp": np.zeros(10),  # Define twist using 3 B-spline cp's
        # distributed along span
        "mesh": mesh,
        # Aerodynamic performance of the lifting surface at
        # an angle of attack of 0 (alpha=0).
        # These CL0 and CD0 values are added to the CL and CD
        # obtained from aerodynamic analysis of the surface to get
        # the total CL and CD.
        # These CL0 and CD0 values do not vary wrt alpha.
        "CL0": 0.0,  # CL of the surface at alpha=0
        "CD0": 0.0,  # CD of the surface at alpha=0
        # Airfoil properties for viscous drag calculation
        "k_lam": 0.05,  # percentage of chord with laminar
        # flow, used for viscous drag
        "t_over_c": 0.12,  # thickness over chord ratio (NACA0015)
        "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
        # thickness
        "with_viscous": False,  # if true, compute viscous drag,
        "with_wave": False,
    }  # end of surface dictionary

    name = surface["name"]

    # Add geometry to the problem as the name of the surface.
    # These groups are responsible for manipulating the geometry of the mesh,
    # in this case spanwise twist.
    geom_group = Geometry(surface=surface)

    # =========================================================================
    # Problem Setup
    # =========================================================================
    p = om.Problem()
    # -------------------------------------------------------------------------
    # Exec Comps
    # -------------------------------------------------------------------------
    comp1 = om.IndepVarComp('vx', 40.0)
    p.model.add_subsystem('comp1', comp1, promotes=['*'])
    comp2 = om.IndepVarComp('vy', 0.0)
    p.model.add_subsystem('comp2', comp2, promotes=['*'])
    p.model.add_subsystem('AOA', AOAComp())

    # -------------------------------------------------------------------------
    # OAS
    # -------------------------------------------------------------------------
    indep_var_comp = om.IndepVarComp()
    # indep_var_comp.add_output("v", val=248.136, units="m/s")  # Freestream Velocity
    # indep_var_comp.add_output("alpha", val=5.0, units="deg")  # Angle of Attack
    indep_var_comp.add_output("beta", val=0.0, units="deg")  # Sideslip angle
    indep_var_comp.add_output("omega", val=np.zeros(3), units="deg/s")  # Rotation rate
    indep_var_comp.add_output("Mach_number", val=0.0)  # Freestream Mach number
    indep_var_comp.add_output("re", val=1.0e6, units="1/m")  # Freestream Reynolds number
    indep_var_comp.add_output("rho", val=1.25, units="kg/m**3")  # Freestream air density
    indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")  # Aircraft center of gravity
    # Add vars to model, promoting is a quick way of automatically connecting inputs
    # and outputs of different OpenMDAO components
    p.model.add_subsystem("flight_vars", indep_var_comp, promotes=["*"])
    p.model.add_subsystem(name, geom_group)

    # Create the aero point group for this flight condition and add it to the model
    aero_group = AeroPoint(surfaces=[surface], rotational=True)
    point_name = "aero_point_0"
    p.model.add_subsystem(
        point_name, aero_group, promotes_inputs=["v", "alpha", "beta", "omega", "Mach_number", "re", "rho", "cg"]
    )
    p.model.aero_point_0.set_input_defaults('alpha',0)
    p.model.aero_point_0.set_input_defaults('v',0)

    # Connect the mesh from the geometry component to the analysis point
    p.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")

    # Perform the connections with the modified names within the
    # 'aero_states' group.
    p.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")

    # -------------------------------------------------------------------------
    # Trajectory
    # -------------------------------------------------------------------------
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    # =========================================================================
    # Climb
    # =========================================================================
    phase0 = dm.Phase(transcription=dm.GaussLobatto(num_segments=10, order=3, solve_segments=False,
                                                   compressed=False),
                     ode_class=D_climb,
                     ode_init_kwargs={'input_dict': input_dict})

    traj.add_phase('phase0', phase0)

    phase0.set_time_options(fix_initial=True, duration_bounds=(5, 60), duration_ref=30)
    phase0.add_state('x', fix_initial=True, rate_source='x_dot', ref0=0, ref=900, defect_ref=100)
    phase0.add_state('y', fix_initial=True, rate_source='y_dot', ref0=0, ref=300, defect_ref=300)
    phase0.add_state('vx', fix_initial=True, rate_source='a_x', ref0=0, ref=10)
    phase0.add_state('vy', fix_initial=True, rate_source='a_y', ref0=0, ref=10)
    phase0.add_state('energy', fix_initial=True, rate_source='energy_dot', ref0=0, ref=1E7, defect_ref=1E5)

    phase0.add_control('power', lower=1e3, upper=311000, ref0=1e3, ref=311000, rate_continuity=False)
    phase0.add_control('theta', lower=0., upper=3 * np.pi / 4, ref0=0, ref=3 * np.pi / 4,
                      rate_continuity=False)

    phase0.add_timeseries_output(['CL', 'CD'])

    # Boundary Constraints
    phase0.add_boundary_constraint('y', loc='final', lower=305,
                                  ref=100)  # Constraint for the final vertical displacement
    phase0.add_boundary_constraint('x', loc='final', equals=900,
                                  ref=100)  # Constraint for the final horizontal displacement

    # Path Constraints
    phase0.add_path_constraint('y', lower=0., upper=305,
                              ref=300)  # Constraint for the minimum vertical displacement
    phase0.add_path_constraint('acc', upper=0.3,
                              ref=1.0)  # Constraint for the acceleration magnitude
    phase0.add_path_constraint('aoa', lower=-np.radians(15), upper=np.radians(15), ref0=-np.radians(15),
                              ref=np.radians(15))  # Constraint for the angle of attack
    phase0.add_path_constraint('thrust', lower=10, ref0=10,
                              ref=100)  # Constraint for the thrust magnitude

    # =========================================================================
    # Climb
    # =========================================================================
    phase1 = dm.Phase(transcription=dm.GaussLobatto(num_segments=10, order=3, solve_segments=False,
                                                   compressed=False),
                     ode_class=D_cruise,
                     ode_init_kwargs={'input_dict': input_dict})

    traj.add_phase('phase1', phase1)

    phase1.set_time_options(fix_initial=False, duration_bounds=(5, 60), duration_ref=30)
    phase1.add_state('x', fix_initial=False, rate_source='x_dot', ref0=0, ref=900, defect_ref=100)
    phase1.add_state('y', fix_initial=False, rate_source='y_dot', ref0=0, ref=300, defect_ref=300)

    phase1.add_state('energy', fix_initial=False, rate_source='energy_dot', ref0=0, ref=1E7, defect_ref=1E5)

    phase1.add_control('power', lower=0, upper=311000, ref0=1e3, ref=10000, rate_continuity=False)

    phase1.add_parameter('vx', val=67., opt=True)
    phase1.add_parameter('vy', val=0.0, opt=True)
    phase1.add_parameter('theta', val=0.9*np.pi/2, opt=False)

    phase1.add_parameter('CL', val=1.0, opt=False)
    phase1.add_parameter('CD', val=1.0, opt=False)

    # phase1.add_timeseries_output(['CL', 'CD'])

    # Objective
    phase1.add_objective('energy', loc='final', ref0=0, ref=1E7)
    # phase1.add_objective('time', loc='final', ref=1.0)

    # Boundary Constraints
    phase1.add_boundary_constraint('y', loc='final', equals=305,
                                  ref=100)  # Constraint for the final vertical displacement
    phase1.add_boundary_constraint('x', loc='final', equals=3000,
                                  ref=100)  # Constraint for the final horizontal displacement
    phase1.add_boundary_constraint('acc', loc='final', equals=0, )

    # Path Constraints
    phase1.add_path_constraint('y', lower=305., upper=305,
                              ref=300)  # Constraint for the minimum vertical displacement
    phase1.add_path_constraint('acc', upper=0.3,
                              ref=1.0)  # Constraint for the acceleration magnitude
    phase1.add_path_constraint('aoa', lower=-np.radians(15), upper=np.radians(15), ref0=-np.radians(15),
                              ref=np.radians(15))  # Constraint for the angle of attack
    phase1.add_path_constraint('thrust', lower=0.1, ref0=11,
                              ref=100)  # Constraint for the thrust magnitude

    traj.link_phases(['phase0', 'phase1'], vars=["*"])
    traj.link_phases(['phase0', 'phase1'], vars=["power"])

    p.model.connect("vx", "AOA.vx")
    p.model.connect("vx", "traj.phase1.parameters:vx")
    p.model.connect("vy", "AOA.vy")
    p.model.connect("vy", "traj.phase1.parameters:vy")

    p.model.connect("AOA.v_inf", "v")
    p.model.connect("AOA.aoa", "alpha")

    p.model.connect("aero_point_0.wing_perf.CL", "traj.phase1.parameters:CL")
    p.model.connect("aero_point_0.wing_perf.CD", "traj.phase1.parameters:CD")

    p.model.add_subsystem("vx_con", excomp1)
    p.model.connect("traj.phase0.timeseries.states:vx", "vx_con.x1", src_indices=[-1])
    p.model.connect("vx", "vx_con.x2")
    p.model.add_subsystem("vy_con", excomp2)
    p.model.connect("traj.phase0.timeseries.states:vy", "vy_con.x1", src_indices=[-1])
    p.model.connect("vy", "vy_con.x2")

    p.model.add_constraint("vx_con.y", equals=0.0)
    p.model.add_constraint("vy_con.y", equals=0.0)

    p.model.add_design_var("vx", lower=30.0)
    p.model.add_design_var("vy", lower=-10.0)

    # # Setup the driver
    # p.driver = om.ScipyOptimizeDriver()
    # p.driver.options['optimizer'] = 'SLSQP'
    # p.driver.options['debug_print'] = ['desvars', 'objs']
    # p.driver.options['tol'] = 1e-9
    # p.driver.options['maxiter'] = 1000

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

    # p.driver.declare_coloring(tol=1.0E-8)
    p.setup()

    p.set_val('traj.phase0.t_initial', 0.0)
    p.set_val('traj.phase0.t_duration', 30)
    p.set_val('traj.phase0.states:x', phase0.interpolate(ys=[0, 900], nodes='state_input'))
    p.set_val('traj.phase0.states:y', phase0.interpolate(ys=[0.01, 300], nodes='state_input'))
    p.set_val('traj.phase0.states:vx', phase0.interpolate(ys=[0, 60], nodes='state_input'))
    p.set_val('traj.phase0.states:vy', phase0.interpolate(ys=[0.01, 10], nodes='state_input'))
    p.set_val('traj.phase0.states:energy', phase0.interpolate(ys=[0, 1E7], nodes='state_input'))

    p.set_val('traj.phase0.controls:power', phase0.interpolate(xs=np.linspace(0, 28.368, 500),
                                                              ys=verify_data.powers.ravel(),
                                                              nodes='control_input'))
    p.set_val('traj.phase0.controls:theta', phase0.interpolate(xs=np.linspace(0, 28.368, 500),
                                                              ys=verify_data.thetas.ravel(),
                                                              nodes='control_input'))

    p.set_val('traj.phase0.controls:power', 200000.0)
    p.set_val('traj.phase0.controls:theta', phase0.interpolate(ys=[0.001, np.radians(85)], nodes='control_input'))

    # Cruise
    p.set_val('traj.phase1.t_initial', 30.0)
    p.set_val('traj.phase1.t_duration', 60)
    p.set_val('traj.phase1.states:x', phase1.interpolate(ys=[900, 3000], nodes='state_input'))
    p.set_val('traj.phase1.states:y', phase1.interpolate(ys=[300, 300], nodes='state_input'))
    p.set_val('traj.phase1.states:energy', phase1.interpolate(ys=[1E7, 1E7], nodes='state_input'))
    p.set_val('traj.phase1.controls:power', 100.0)

    dm.run_problem(p, run_driver=True, simulate=True)

    print(p.get_val('traj.phase1.parameters:vx'))
    print(p.get_val('traj.phase1.parameters:vy'))
    print(p.get_val('traj.phase1.parameters:theta'))
    print(p.get_val('aero_point_0.wing_perf.CL'))
    print(p.get_val('aero_point_0.wing_perf.CD'))

