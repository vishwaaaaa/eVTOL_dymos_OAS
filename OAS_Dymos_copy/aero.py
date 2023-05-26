import numpy as np
import openmdao.api as om

class AeroForce(om.Group):
    """
    Computes the aerodynamic force in the wind frame, 
    
    Parameters
    ----------
    v : float
        air-relative velocity (m/s)
    #  sos : float
    #      local speed of sound (m/s)
    rho : float
        atmospheric density (kg/m**3)
    alpha : float
         angle of attack (rad)
    S : float
        aerodynamic reference area (m**2)
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='CL_comp', subsys=CLComp(num_nodes=nn),
                           promotes_inputs=['alpha'], promotes_outputs=['CL'])

        self.add_subsystem(name='CD_comp', subsys=CDComp(num_nodes=nn),
                           promotes_inputs=['CL'], promotes_outputs=['CD'])

        self.add_subsystem(name='q_comp', subsys=DynamicPressureComp(num_nodes=nn),
                           promotes_inputs=['rho','v'], promotes_outputs=['q'])

        self.add_subsystem(name='lift_drag_force_comp', subsys=LiftDragForceComp(num_nodes=nn),
                           promotes_inputs=['CL', 'CD', 'q', 'S'],
                           promotes_outputs=['f_lift', 'f_drag'])


class CLComp(om.ExplicitComponent):
    # computes lift coefficient, assuming constant CL_alpha (ignore Mach dependence)
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self._Cl_alpha = 0.25   # 1/deg, hardcoded!

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('alpha', shape=(nn,), desc='angle of attck', units='deg')
        self.add_output(name='CL', val=np.ones(nn), desc='lift coefficient', units=None)
        self.declare_partials(of='CL', wrt='alpha', rows=np.arange(nn), cols=np.arange(nn))

    def compute(self, inputs, outputs):
        outputs['CL'] = self._Cl_alpha * inputs['alpha']

    def compute_partials(self, inputs, partials):
        partials['CL', 'alpha'] = self._Cl_alpha


class CDComp(om.ExplicitComponent):
    # computes drag coefficient, ignoring Mach dependence
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self._Cd0 = 0.0065
        self._AR = 15.  # aspect ratio
        self._e = 0.9  # oswald span efficiency
 
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('CL', shape=(nn,), desc='lift coefficient', units=None)
        self.add_output(name='CD', val=np.zeros(nn), desc='drag coefficient', units=None)
        self.declare_partials(of='CD', wrt='CL', rows=np.arange(nn), cols=np.arange(nn))

    def compute(self, inputs, outputs):
        outputs['CD'] = self._Cd0 + inputs['CL']**2 / (np.pi * self._e * self._AR)

    def compute_partials(self, inputs, partials):
        partials['CD', 'CL'] = 2. * inputs['CL'] / (np.pi * self._e * self._AR)


class DynamicPressureComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input(name='rho', val=0.5 * np.ones(nn), desc='atmospheric density', units='kg/m**3')
        self.add_input(name='v', shape=(nn,), desc='air-relative velocity', units='m/s')
        #self.add_input(name='vy', shape=(nn,), desc='air-relative velocity', units='m/s')
        self.add_output(name='q', shape=(nn,), desc='dynamic pressure', units='N/m**2')
        ar = np.arange(nn)
        self.declare_partials(of='q', wrt='rho', rows=ar, cols=ar)
        self.declare_partials(of='q', wrt='v', rows=ar, cols=ar)
        #self.declare_partials(of='q', wrt='vy', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['q'] = 0.5 * inputs['rho'] * inputs['v']**2

    def compute_partials(self, inputs, partials):
        partials['q', 'rho'] = 0.5 * inputs['v']**2  #inputs['v'] ** 2
        partials['q', 'v'] = inputs['rho'] * inputs['v']#inputs['v']
        #partials['q', 'vy'] = inputs['rho'] * input['vy']


class LiftDragForceComp(om.ExplicitComponent):
    """
    Compute the aerodynamic forces on the vehicle in the wind axis frame
    (lift, drag, cross) force. Cross (side) force is assumed 0.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input(name='CL', val=np.zeros(nn,), desc='lift coefficient', units=None)
        self.add_input(name='CD', val=np.zeros(nn,), desc='drag coefficient', units=None)
        self.add_input(name='q', val=np.zeros(nn,), desc='dynamic pressure', units='N/m**2')
        self.add_input(name='S', shape=(1,), desc='aerodynamic reference area', units='m**2')
        self.add_output(name='f_lift', shape=(nn,), desc='aerodynamic lift force', units='N')
        self.add_output(name='f_drag', shape=(nn,), desc='aerodynamic drag force', units='N')

        ar = np.arange(nn)
        self.declare_partials(of='f_lift', wrt='q', rows=ar, cols=ar)
        self.declare_partials(of='f_lift', wrt='S', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='f_lift', wrt='CL', rows=ar, cols=ar)
        self.declare_partials(of='f_drag', wrt='q', rows=ar, cols=ar)
        self.declare_partials(of='f_drag', wrt='S', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='f_drag', wrt='CD', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        q = inputs['q']
        S = inputs['S']
        CL = inputs['CL']
        CD = inputs['CD']

        qS = q * S
        outputs['f_lift'] = qS * CL
        outputs['f_drag'] = qS * CD

    def compute_partials(self, inputs, partials):
        q = inputs['q']
        S = inputs['S']
        CL = inputs['CL']
        CD = inputs['CD']

        qS = q * S
        partials['f_lift', 'q'] = S * CL
        partials['f_lift', 'S'] = q * CL
        partials['f_lift', 'CL'] = qS
        partials['f_drag', 'q'] = S * CD
        partials['f_drag', 'S'] = q * CD
        partials['f_drag', 'CD'] = qS