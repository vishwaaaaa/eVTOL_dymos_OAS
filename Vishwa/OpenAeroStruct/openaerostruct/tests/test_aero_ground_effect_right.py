import unittest


class Test(unittest.TestCase):
    def test(self):
        # docs checkpoint 0
        from openaerostruct.geometry.utils import generate_mesh

        from openmdao.utils.assert_utils import assert_near_equal
        from openaerostruct.utils.testing import assert_check_totals

        # Create a dictionary to store options about the mesh
        mesh_dict = {"num_y": 7, "num_x": 2, "wing_type": "CRM", "symmetry": True, "num_twist_cp": 5}

        # Generate the aerodynamic mesh based on the previous dictionary
        mesh, twist_cp = generate_mesh(mesh_dict)
        left_problem = self.setup_problem(mesh, twist_cp)
        left_problem.run_driver()

        # flip each surface to lie on right
        flip_mesh = mesh[:, ::-1, :]
        flip_mesh[:, :, 1] *= -1.0
        right_problem = self.setup_problem(flip_mesh, twist_cp)
        right_problem.run_driver()

        # docs checkpoint 1
        assert_near_equal(
            left_problem["aero_point_0.wing_perf.CD"][0], right_problem["aero_point_0.wing_perf.CD"][0], 1e-6
        )
        assert_near_equal(
            left_problem["aero_point_0.wing_perf.CL"][0], right_problem["aero_point_0.wing_perf.CL"][0], 1e-6
        )
        assert_near_equal(left_problem["aero_point_0.CM"][1], right_problem["aero_point_0.CM"][1], 1e-6)

        left_problem["height_agl"] = right_problem["height_agl"] = 10.0
        left_problem.run_driver()
        right_problem.run_driver()
        assert_near_equal(
            left_problem["aero_point_0.wing_perf.CD"][0], right_problem["aero_point_0.wing_perf.CD"][0], 1e-6
        )
        assert_near_equal(
            left_problem["aero_point_0.wing_perf.CL"][0], right_problem["aero_point_0.wing_perf.CL"][0], 1e-6
        )
        assert_near_equal(left_problem["aero_point_0.CM"][1], right_problem["aero_point_0.CM"][1], 1e-6)

        totals = right_problem.check_totals(
            of=["aero_point_0.wing_perf.CD", "aero_point_0.wing_perf.CL"],
            wrt=["wing.twist_cp", "height_agl"],
            compact_print=True,
            out_stream=None,
        )
        assert_check_totals(totals, atol=1e-2, rtol=1e-5)

    def setup_problem(self, mesh, twist_cp):
        """
        Setup identical OpenMDAO problems based on an input surface mesh and twist array.
        """
        import numpy as np

        import openmdao.api as om

        from openaerostruct.geometry.geometry_group import Geometry
        from openaerostruct.aerodynamics.aero_groups import AeroPoint

        # Create a dictionary with info and options about the aerodynamic
        # lifting surface
        surface = {
            # Wing definition
            "name": "wing",  # name of the surface
            "symmetry": True,  # if true, model one half of wing
            # reflected across the plane y = 0
            "groundplane": True,
            "S_ref_type": "wetted",  # how we compute the wing area,
            # can be 'wetted' or 'projected'
            "fem_model_type": "tube",
            "twist_cp": twist_cp,
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
            "with_viscous": True,  # if true, compute viscous drag
            "with_wave": False,  # if true, compute wave drag
        }

        # Create the OpenMDAO problem
        prob = om.Problem()

        # Create an independent variable component that will supply the flow
        # conditions to the problem.
        indep_var_comp = om.IndepVarComp()
        indep_var_comp.add_output("v", val=248.136, units="m/s")
        indep_var_comp.add_output("alpha", val=5.0, units="deg")
        indep_var_comp.add_output("Mach_number", val=0.84)
        indep_var_comp.add_output("re", val=1.0e6, units="1/m")
        indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
        indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")
        indep_var_comp.add_output("height_agl", val=8000.0, units="m")

        # Add this IndepVarComp to the problem model
        prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

        # Create and add a group that handles the geometry for the
        # aerodynamic lifting surface
        geom_group = Geometry(surface=surface)
        prob.model.add_subsystem(surface["name"], geom_group)

        # Create the aero point group, which contains the actual aerodynamic
        # analyses
        aero_group = AeroPoint(surfaces=[surface])
        point_name = "aero_point_0"
        prob.model.add_subsystem(
            point_name, aero_group, promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg", "height_agl"]
        )

        name = surface["name"]

        # Connect the mesh from the geometry component to the analysis point
        prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")

        # Perform the connections with the modified names within the
        # 'aero_states' group.
        prob.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")

        prob.model.connect(name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c")

        # Import the Scipy Optimizer and set the driver of the problem to use
        # it, which defaults to an SLSQP optimization method
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options["tol"] = 1e-9

        # Setup problem and add design variables, constraint, and objective
        prob.model.add_design_var("height_agl", lower=10.0, upper=8000.0)
        prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=15.0)
        prob.model.add_constraint(point_name + ".wing_perf.CL", equals=0.5)
        prob.model.add_objective(point_name + ".wing_perf.CD", scaler=1e4)

        # Set up and run the optimization problem
        prob.setup()

        return prob


if __name__ == "__main__":
    unittest.main()
