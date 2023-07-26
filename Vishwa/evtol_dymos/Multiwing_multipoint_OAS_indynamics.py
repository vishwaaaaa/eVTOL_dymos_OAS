import numpy as np

import openmdao.api as om

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

import matplotlib.pyplot as plt

from utils import Scalars2Vector, Vectors2Matrix, Arrays2Dto3D, Arrays3Dto4D

class OASAeroMulti(om.Group):

    """
    Computes CL and CD of the wing using OAS aerodynamics analysis

    Inputs: v, alpha (vector)
    Outputs: CL, CD (vectors)
    """

    def initialize(self):

        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('OAS_surface_1', types=dict, desc='Surface dict for OAS for the wing')
        self.options.declare('OAS_surface_2', types=dict, desc='Surface dict for OAS for the tail')
        #self.options.declare('flag_AS', types=bool, desc='True for aerostructural analysis, False for aerodynamic analysis')



    def setup(self):
        nn = self.options['num_nodes']
        surf_dict = self.options['OAS_surface_1']
        surf_dict2 = self.options['OAS_surface_2']
        surfaces = [surf_dict,surf_dict2]


        #-------setup OAS model--------#
        #  Time-independent settings 

        indep_var_comp = om.IndepVarComp()

        #indep_var_comp.add_output("v", val=248.136, units="m/s")
        #indep_var_comp.add_output("alpha", val=5.0, units="deg")
        indep_var_comp.add_output("Mach_number", val=0.84)
        indep_var_comp.add_output("re", val=1.0e6, units="1/m")
        #indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
        indep_var_comp.add_output("cg", val=np.array([0.2, 0.1, 0.0]), units="m")
        #prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

        # Add this Independent Variable Component to the problem model
        self.add_subsystem("prob_vars", indep_var_comp, promotes=['*'])

        # this component does nothing, but this is a hack necessary to connect the vector inputs (v, alpha, rho, m) into a bunch of scalars at each point.
        vec_comp = om.ExecComp(['v_vector = v', 'alpha_vector = alpha', 'rho_vector = rho'],
                               v_vector={'shape': (nn), 'units': 'm/s'},
                               v={'shape': (nn), 'units': 'm/s'},
                               alpha_vector={'shape': (nn), 'units': 'deg'},
                               alpha={'shape': (nn), 'units': 'deg'},
                               rho_vector={'shape': (nn), 'units': 'kg/m**3'},
                               rho={'shape': (nn), 'units': 'kg/m**3'},
                               has_diag_partials=True)
        
        self.add_subsystem('vector_in', vec_comp, promotes_inputs=['v', 'alpha', 'rho'])



        # Loop over each surface in the surfaces list
        for surface in surfaces:
            geom_group = Geometry(surface=surface)

            # Add tmp_group to the problem as the name of the surface.
            # Note that is a group and performance group for each
            # individual surface.
            self.add_subsystem(surface["name"], geom_group)

        # Loop through and add a certain number of aero points
        for i in range(nn):
            # Create the aero point group and add it to the model
            aero_group = AeroPoint(surfaces=surfaces)
            point_name = "aero_point_{}".format(i)

            # can try promoting the S_ref, only needed for the lift force calculation
            # if i == 0:
            #         # promote S_ref output for later use
            #         promotes_outputs = [(surf_dict2["name"]+ '.S_ref','S_ref')]
            # else:
            #         promotes_outputs = []

            self.add_subsystem(point_name, aero_group,promotes_inputs=["Mach_number","re","cg"])

            # Connect flow properties to the analysis point
            # prob.model.connect("v", point_name + ".v")
            # prob.model.connect("alpha", point_name + ".alpha")
            #prob.model.connect("Mach_number", point_name + ".Mach_number")
            #prob.model.connect("re", point_name + ".re")
            # prob.model.connect("rho", point_name + ".rho")
            #prob.model.connect("cg", point_name + ".cg")
            self.connect('vector_in.rho_vector', point_name + '.rho', src_indices=i)
            self.connect('vector_in.v_vector', point_name + '.v', src_indices=i)
            self.connect('vector_in.alpha_vector', point_name + '.alpha', src_indices=i)




            # Connect the parameters within the model for each aero point
            for surface in surfaces:
                name = surface["name"]

                # Connect the mesh from the geometry component to the analysis point
                self.connect(name + ".mesh", point_name + "." + name + ".def_mesh")

                # Perform the connections with the modified names within the
                # 'aero_states' group.
                self.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")

                self.connect(name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c")

        # Output CL and CD vectors
        self.add_subsystem('CL_vector', Scalars2Vector(num_nodes=nn, units=None), promotes_outputs=[('vector', 'CL')])
        self.add_subsystem('CD_vector', Scalars2Vector(num_nodes=nn, units=None), promotes_outputs=[('vector', 'CD')])
        # connect outputs of each aero point into the vectors
        for i in range(nn):
            point_name = "aero_point_" + str(i)
            # self.connect(point_name + ".wing_perf.CL", 'CL_vector.scalar' + str(i))
            self.connect(point_name + ".CL", 'CL_vector.scalar' + str(i))

            # self.connect(point_name + ".wing_perf.CD", 'CD_vector.scalar' + str(i))
            self.connect(point_name + ".CD", 'CD_vector.scalar' + str(i))
        # Set up the problem
        # prob.setup()

        # # Set the optimizer type
        # prob.driver = om.ScipyOptimizeDriver()
        # prob.driver.options["tol"] = 1e-7

        # # Record data from this problem so we can visualize it using plot_wing


        # recorder = om.SqliteRecorder("aero1.db")
        # prob.driver.add_recorder(recorder)
        # prob.driver.recording_options["record_derivatives"] = True
        # prob.driver.recording_options["includes"] = ["*"]
                                                    



        # # Setup problem and add design variables.



        # prob.model.add_design_var("wing.sweep", lower=0.0, upper=10.0)
        # prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=15.0)
        # prob.model.add_design_var("tail.sweep", lower=0.0, upper=10.0)
        # prob.model.add_design_var("tail.twist_cp", lower=-10.0, upper=15.0)
        # prob.model.add_constraint("aero_point_1"+ ".wing_perf.CL", equals=0.7)
        # prob.model.add_objective("aero_point_1"+ ".wing_perf.CD", scaler=1e4)
        # # Set up the problem
        # prob.setup()

        # # Use this if you just want to run analysis and not optimization
        # # prob.run_model()


        # # Actually run the optimization problem
        # #om.n2(prob)
        # prob.run_driver()



##----- AOA component------##
class AOAComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num')
    def setup(self):
        nn = self.options['num']
        self.add_input('vx', val=1*np.ones(nn))
        self.add_input('vy', val=1*np.ones(nn))
        self.add_input('theta', val=0.9*np.pi/2.0*np.ones(nn)) # HARD CODED FOR NOW, ADJUST IF CHANGED
        self.add_output('aoa', val=1*np.ones(nn))
        self.add_output('v_inf', val=1*np.ones(nn))

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num']
        for i in range(nn):
            x_dot = inputs["vx"][i]
            y_dot = inputs["vy"][i]
            theta = inputs["theta"][i]
            atov = c_atan2(x_dot, y_dot)
            outputs["aoa"][i] = (atov - theta) * 180. / np.pi
            outputs["v_inf"][i] = (x_dot**2 + y_dot**2) ** (0.5)

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

class Dynamics(om.ExplicitComponent):
    """
    Dummy dynamics class for Multipoint testing 

    """
    def initialize(self):
        self.options.declare('num_nodes',types=int)
    
    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('CL', val=np.ones(num_nodes))
        self.add_input('CD', val=np.ones(num_nodes))
        self.add_input('vx', val=np.ones(num_nodes))
        self.add_input('vy', val=np.ones(num_nodes))
        self.add_input('theta', val=np.ones(num_nodes))

        self.add_output('scaledCL',val = np.ones(num_nodes))
        self.add_output('scaledCD', val=np.ones(num_nodes))

        self.declare_partials('*','*', method='cs')


    def compute(self, inputs, outputs):
        
        # time integration
        for i in range(self.options['num_nodes']):
            CL = inputs['CL']
            CD = inputs['CD']
            

            outputs['scaledCL'] = CL*2
            outputs['scaledCD'] = CD*2





#------------------------

if __name__=='__main__':
      ## defining surface perimeters for the tail and the wing 
    #========================SURFACE-1 (wing)=====================================#
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



    #========================SURFACE-2 (Tail)=====================================#
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
    p = om.Problem()
    
    ind_ver_compm = om.IndepVarComp()
    ind_ver_compm.add_output('vx',np.array([50, 60, 60]), units='m/s')
    ind_ver_compm.add_output('vy',np.array([0.707, 1.4, 1.8]), units='m/s')
    ind_ver_compm.add_output('theta', np.array([np.pi*7/8, np.pi*7/8, np.pi*7/8]), units='deg')

    p.model.add_subsystem('probvars',ind_ver_compm,promotes=['vx','vy','theta'])
    p.model.add_subsystem(name='AOA',subsys=AOAComp(num=3), promotes_inputs=['vx','vy','theta'],promotes_outputs=[('v_inf','v'),('aoa','alpha')])
    p.model.add_subsystem(name='aero', subsys=OASAeroMulti(num_nodes=3, OAS_surface_1=surf_dict, OAS_surface_2 = surf_dict2), promotes_inputs=['v', 'alpha','rho'])
    p.model.add_subsystem(name='fd', subsys=Dynamics(num_nodes=3), promotes_inputs=['vx','vy','theta'])
    p.model.connect('aero.CL','fd.CL')
    p.model.connect('aero.CD','fd.CD')
    p.setup(check=True)
    #p.set_val('alpha', np.array([1, 2, 3]), units='deg')
    #p.set_val('v', np.array([50, 60, 70]), units='m/s')
    # p.set_val('vx', np.array([50, 60, 60]), units='m/s')
    # p.set_val('vy', np.array([0.707, 1.4, 1.8]), units='m/s')
    # p.set_val('theta', np.array([np.pi*7/8, np.pi*7/8, np.pi*7/8]), units='deg')

    p.set_val('rho', val=1.2 * np.ones(3), units="kg/m**3")

    p.run_model()
    #plt.plot(p.get_val('aero.CL'),np.arange(1,4))
    om.n2(p)
