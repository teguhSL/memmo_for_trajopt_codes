import numpy as np
import json
import trajoptpy
import time

def define_request(time_step=30, manip_name='active', constraint_type ='joint', init_type='straight_line', start_fixed=True, coeffs = [1]):
    request = {
      "basic_info" : {
        "n_steps" : time_step,
        "manip" : manip_name, # see below for valid values
        "start_fixed" : start_fixed # i.e., DOF values at first timestep are fixed based on current robot state
      },
      "costs" : [
      {
        "type" : "joint_vel", # joint-space velocity cost
        "params": {"coeffs" : coeffs} # a list of length one is automatically expanded to a list of length n_dofs
      },
      {
        "type" : "collision",
        "name" :"cont_coll", # shorten name so printed table will be prettier
        "params" : {
          "continuous" : True,
          "coeffs" : [20], # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
          "dist_pen" : [0.005] # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
        }
      },
      {
        "type" : "collision",
        "name" :"dist_coll", # shorten name so printed table will be prettier
        "params" : {
          "continuous" : False,
          "coeffs" : [20], # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
          "dist_pen" : [0.005] # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
        }
      }
      ],
      # BEGIN init
      "init_info" : {
          "type" : init_type, # straight line in joint space.
      }
      # END init
    }
    
    request["constraints"] = [  ]

    return request

def add_constraint(request, constraint_type, frame_name, target, timestep):
    target = target.tolist()
    constraint = dict()
    if constraint_type == 'pose':
        xyz_target = target[0:3]
        quat_target = target[3:]
        if len(quat_target) == 0: quat_target = [1,0,0,0]
        constraint['params'] = dict()
        constraint['params']['link'] = frame_name
        constraint['params']['timestep'] = timestep
        constraint['params']['wxyz'] =  quat_target
        constraint['params']['xyz'] = xyz_target
        #constraint['params']['rot_coeffs'] = [0,0,0]
        #constraint['params']['pos_coeffs'] = [10,10,10]
        constraint['type'] = 'pose' 
        request['constraints'].append(constraint)
    elif constraint_type == 'joint':
        constraint['type'] = 'joint' 
        constraint['params'] = dict()
        constraint['params']['vals'] = target
        request['constraints'].append(constraint)
    else:
        print('Constraint {} is not defined'.format(constraint_type))
    return request
        
def set_init(request, init_type, init_values):
    if init_type == 'straight_line':
        request['init_info']['endpoint'] = init_values.tolist()
    elif init_type == 'given_traj':
        request["init_info"]["data"] = [row.tolist() for row in init_values]
        request["init_info"]["type"] = init_type
    else:
        print('Initialization {} is not defined'.format(init_type))
        return None
    return request

def run_opt(request,env):
    s = json.dumps(request) 
    prob = trajoptpy.ConstructProblem(s, env)
    tic = time.time()
    result = trajoptpy.OptimizeProblem(prob) # do optimization
    toc = time.time()
    print("Optimization is completed in {} s!".format(toc-tic))
    duration = toc-tic
    return duration, result


    