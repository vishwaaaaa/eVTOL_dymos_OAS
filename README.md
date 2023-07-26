# eVTOL_dymos_OAS

## working_OAS_dymos_traj
This branch contains the updated code for the trajectory optimization of eVTOL. 
```
└───OAS_Dymos_copy/
│
└───Vishwa/
│   └───OpenAeroStruct/
│   └───evtol_dymos/
│   │   └───coloringfiles/
│   │   └───reports/
│   │   └───evtol_OAS_thurstcorrection.py     <---- main run script
│   │   └───plot_results_both.py               <---- main plotting script
│
└───evtol_explicit_time_integration/
│
└───ode/
│   └───evtol_dynamics_comp_climb.py     <----contains the ode for the take-off phase
│   └───evtol_dynamics_comp_climb_new.py     <----contains the ode for the cruise with OAS aerodynamics analysis
└───original

```
In the main run script we define two phases: ```Phase0 (take-off)``` and ```Phase1 (cruise)```. The initial parameters for the both the phases are defined in ```input_dict``` and ```input_dict_d``` dictionaries. The states variables in this optimization are ``` x, y, vx, vy, energy``` and the controls are ```power``` and $\theta$ . 
```phase0``` uses the first principle based model for CL and CD calculations during the flight.
```phase1``` uses OAS based aerodynamic analysis for CL and CD calculations during the flight.The lighting surfaces closely mimic 'Airbus Vahana' as shown in the figure below:
![image](https://github.com/vishwaaaaa/eVTOL_dymos_OAS/assets/122648757/3274a514-ff39-45c0-87f9-9612dd320263)
