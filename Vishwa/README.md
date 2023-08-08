# Simulating an eVTOL takeoff and transition with (and without Dymos)

This problem falls into the [unsteady analysis][unsteady-class] class of optimization problems. 
The first thing that needs to be done when addressing these class of problems is to 
develop an ordinary differential equation that governs the dynamics. 

In this problem, an ODE for the eVTOL aircraft was provided by Shamsheer Chuahan. 
You can checkout the details in his paper [here](https://www.researchgate.net/publication/337259571_Tilt-Wing_eVTOL_Takeoff_Trajectory_Optimization). 
Shamsheer also provided his [original implementation](https://bitbucket.org/shamsheersc19/tilt_wing_evtol_takeoff), which included an Euler time integration built inside an OpenMDAO component. 


# Shooting VS. Implicit Methods

## Shooting Methods a.k.a MDF
In a shooting method, the optimizer gets to see the control schedule as a design variable. 
Then, given that control schedule the entire time history is simulated out and the objective and constraints can be evaluated. 

We can relate this in the MDO context to the Multidisciplinary Design Feasible (MDF) architecture. 
Normally MDF implies that some sort of solver is in place converging governing equations, and that the optimizer is being shown a **continuous feasible space** where those governing equations are solved. 

What are the governing equations in the context of an Euler integration, and what are the associated state variables? 
The state variables are the collection of discrete values that represent the state-time-history of your system. 
If you were modeling a simple cannonball problem, you might have 2 states: x and y. 
If you took 10 time steps then you'd have 10 state variables for x and 10 for y. 
So what then are the residuals? 
Assuming you have some ODE function like this: 

x_dot, y_dot = f(time, control, x, y)

You can define the i-th residual like this: 

R_xi = x_dot(time_i, control_i, x_i, y_i) * delta_time - x_i+1
R_yi = x_dot(time_i, control_i, x_i, y_i) * delta_time - y_i+1

You don't have to actually solve the time series as a big implicit system (though you can if you want to), but the residual exist none the less which makes the connection to the MDF architecture. 
Regardless whether you find time-history using an time-stepping approach or the implicit residual form, since the optimizer always see a fully complete time history its an MDF approach. 


## Implicit Collocation a.k.a SAND
In an implicit collocation method, you treat both the controls and the state time-history as design variables for the optimizer. 
Then, because you've now added a bunch of new degrees of freedom to the optimizer you also provide it new equality constraints (sometimes called defect-constraints in the optimal control world) that must also be satisfied. 
If you were using an Euler based time integration scheme, the defect constraints would be exactly the same as the residuals 
given above. 

In the MDO world, the defect constraints could also be called the governing equations of the system. 
When you give the governing equations to the optimizer as equality constraints that is called the Simultaneous Analysis and Design (SAND) architecture. 

The major practical change by using this approach is that the optimizer has a much larger space to navigate through, 
since it can now violate physics. 
This can be really helpful if you happen to have a problem with a non-continuous feasible space. 


## Which is better Shooting or Implicit?

There is no simple answer here. Generally speaking, implicit methods are faster and commonly find better answers. 
However, for some problems implicit methods can be numerically finicky and really sensitive to scaling of design vars and constraints. 
Shooting methods are easier to set up, and when they work are much less sensitive to scaling. 
However, they also have their own numerical challenges if there are any singularities in your ODE (e.g. if you divide by sin(angle-of-attack)) or if you have a bumpy non-contiguous search space. 

Both have uses, and we tested out several solutions to this problem to see what happened. 





[unsteady-class]: ../../solution_approaches/unsteady_analysis.md
