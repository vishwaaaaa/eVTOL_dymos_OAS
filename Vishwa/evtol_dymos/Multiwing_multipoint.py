import numpy as np

import openmdao.api as om

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

# Create a dictionary to store options about the surface
# Total number of nodes to use in the spanwise (num_y) and
# chordwise (num_x) directions. Vary these to change the level of fidelity.
num_y = 21
num_x = 3

# Create a mesh dictionary to feed to generate_mesh to actually create
# the mesh array.
mesh_dict = {
    "num_y": num_y,
    "num_x": num_x,
    "wing_type": "rect",
    "symmetry": True,
    "span_cos_spacing": 0.9,
    "span": 3.11,
    "root_chord": 0.3,
}

mesh = generate_mesh(mesh_dict)

# Apply camber to the mesh
camber = 1 - np.linspace(-1, 1, num_x) ** 2
camber *= 0.3 * 0.05

for ind_x in range(num_x):
    mesh[ind_x, :, 2] = camber[ind_x]

# Introduce geometry manipulation variables to define the ScanEagle shape
zshear_cp = np.zeros(20)
zshear_cp[0] = 0

###########
xshear_cp = np.zeros(20)
xshear_cp[0] = 0.0

chord_cp = np.ones(20)
chord_cp[0] = 1
chord_cp[-1] = 1
chord_cp[-2] = 1

radius_cp = 0.01 * np.ones(20)

# Define wing parameters
surf_dict = {
    # Wing definition
    "name": "wing",  # name of the surface
    "symmetry": True,  # if true, model one half of wing
    # reflected across the plane y = 0
    "S_ref_type": "wetted",  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    "fem_model_type": "tube",
    "taper": 1,
    "zshear_cp": zshear_cp,
    "xshear_cp": xshear_cp,
    "chord_cp": chord_cp,
    "sweep": 5.0,
    "twist_cp": np.array([2.5, 2.5, 5.0]),  # np.zeros((3)),
    "thickness_cp": np.ones((3)) * 0.008,
    # Give OAS the radius and mesh from before
    #"radius_cp": radius_cp,  # for having the lifted wings in the edges
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
    "t_over_c_cp": np.array([0.12]),  # thickness over chord ratio
    "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
    # thickness
    "with_viscous": True,
    "with_wave": False,  # if true, compute wave drag
}

# Create a dictionary to store options about the surface
# mesh_dict = {"num_y": 7, "num_x": 2, "wing_type": "rect", "symmetry": True, "offset": np.array([10, 0.0, 0.0])}

# mesh = generate_mesh(mesh_dict)

# surf_dict2 = {
#     # Wing definition
#     "name": "tail",  # name of the surface
#     "symmetry": True,  # if true, model one half of wing
#     # reflected across the plane y = 0
#     "S_ref_type": "wetted",  # how we compute the wing area,
#     # can be 'wetted' or 'projected'
#     "twist_cp": np.array([2.5, 2.5, 5.0]),# twist_cp,
#     "mesh": mesh,
#     # Aerodynamic performance of the lifting surface at
#     # an angle of attack of 0 (alpha=0).
#     # These CL0 and CD0 values are added to the CL and CD
#     # obtained from aerodynamic analysis of the surface to get
#     # the total CL and CD.
#     # These CL0 and CD0 values do not vary wrt alpha.
#     "CL0": 0.0,  # CL of the surface at alpha=0
#     "CD0": 0.0,  # CD of the surface at alpha=0
#     "fem_origin": 0.35,
#     # Airfoil properties for viscous drag calculation
#     "k_lam": 0.05,  # percentage of chord with laminar
#     # flow, used for viscous drag
#     "t_over_c_cp": np.array([0.15]),  # thickness over chord ratio (NACA0015)
#     "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
#     # thickness
#     "with_viscous": True,  # if true, compute viscous drag
#     "with_wave": False,  # if true, compute wave drag
# }

num_y = 21
num_x = 3

# Create a mesh dictionary to feed to generate_mesh to actually create
# the mesh array.
mesh_dict = {
    "num_y": num_y,
    "num_x": num_x,
    "wing_type": "rect",
    "symmetry": True,
    "span_cos_spacing": 0.9,
    "span": 3.11,
    "root_chord": 0.3,
    "offset": np.array([1, 0.0, 0.0])
}

mesh = generate_mesh(mesh_dict)

# Apply camber to the mesh
camber = 1 - np.linspace(-1, 1, num_x) ** 2
camber *= 0.3 * 0.05

for ind_x in range(num_x):
    mesh[ind_x, :, 2] = camber[ind_x]

# Introduce geometry manipulation variables to define the ScanEagle shape
zshear_cp = np.zeros(20)
zshear_cp[0] = 0.3

###########
xshear_cp = np.zeros(20)
xshear_cp[0] = 0.15

chord_cp = np.ones(20)
chord_cp[0] = 0.5
chord_cp[-1] = 1.5
chord_cp[-2] = 1.3

radius_cp = 0.01 * np.ones(20)

# Define wing parameters
surf_dict2 = {
    # Wing definition
    "name": "tail",  # name of the surface
    "symmetry": True,  # if true, model one half of wing
    # reflected across the plane y = 0
    "S_ref_type": "wetted",  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    "fem_model_type": "tube",
    "taper": 0.9,
    "zshear_cp": zshear_cp,
    "xshear_cp": xshear_cp,
    "chord_cp": chord_cp,
    "sweep": 5.0,
    "twist_cp": np.array([2.5, 2.5, 5.0]),  # np.zeros((3)),
    "thickness_cp": np.ones((3)) * 0.008,
    # Give OAS the radius and mesh from before
    "radius_cp": radius_cp,
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
    "t_over_c_cp": np.array([0.12]),  # thickness over chord ratio
    "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
    # thickness
    "with_viscous": True,
    "with_wave": False,  # if true, compute wave drag
}


#############################
surfaces = [surf_dict, surf_dict2]

# Create the problem and the model group
prob = om.Problem()

indep_var_comp = om.IndepVarComp()

indep_var_comp.add_output("v", val=248.136, units="m/s")
indep_var_comp.add_output("alpha", val=5.0, units="deg")
indep_var_comp.add_output("Mach_number", val=0.84)
indep_var_comp.add_output("re", val=1.0e6, units="1/m")
indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.array([0.2, 0.1, 0.0]), units="m")
prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

# Loop over each surface in the surfaces list
for surface in surfaces:
    geom_group = Geometry(surface=surface)

    # Add tmp_group to the problem as the name of the surface.
    # Note that is a group and performance group for each
    # individual surface.
    prob.model.add_subsystem(surface["name"], geom_group)

# Loop through and add a certain number of aero points
for i in range(1,2):
    # Create the aero point group and add it to the model
    aero_group = AeroPoint(surfaces=surfaces)
    point_name = "aero_point_{}".format(i)
    prob.model.add_subsystem(point_name, aero_group)

    # Connect flow properties to the analysis point
    prob.model.connect("v", point_name + ".v")
    prob.model.connect("alpha", point_name + ".alpha")
    prob.model.connect("Mach_number", point_name + ".Mach_number")
    prob.model.connect("re", point_name + ".re")
    prob.model.connect("rho", point_name + ".rho")
    prob.model.connect("cg", point_name + ".cg")




    # Connect the parameters within the model for each aero point
    for surface in surfaces:
        name = surface["name"]

        # Connect the mesh from the geometry component to the analysis point
        prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")

        # Perform the connections with the modified names within the
        # 'aero_states' group.
        prob.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")

        prob.model.connect(name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c")

# Set up the problem
prob.setup()

# Set the optimizer type
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["tol"] = 1e-7

# Record data from this problem so we can visualize it using plot_wing


recorder = om.SqliteRecorder("aero1.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options["record_derivatives"] = True
prob.driver.recording_options["includes"] = ["*"]
                                             



# Setup problem and add design variables.



prob.model.add_design_var("wing.sweep", lower=0.0, upper=10.0)
prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=15.0)
prob.model.add_design_var("tail.sweep", lower=0.0, upper=10.0)
prob.model.add_design_var("tail.twist_cp", lower=-10.0, upper=15.0)
prob.model.add_constraint("aero_point_1"+ ".wing_perf.CL", equals=0.7)
prob.model.add_objective("aero_point_1"+ ".wing_perf.CD", scaler=1e4)
# Set up the problem
prob.setup()

# Use this if you just want to run analysis and not optimization
# prob.run_model()


# Actually run the optimization problem
#om.n2(prob)
prob.run_driver()