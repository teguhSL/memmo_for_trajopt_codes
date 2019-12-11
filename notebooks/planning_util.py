import numpy as np
import openravepy
import trajoptpy.kin_utils as ku
from trajoptpy.check_traj import traj_is_safe
import time

def add_box(env, x,y,z, box_size = 0.04):
    box = openravepy.RaveCreateKinBody(env,'')
    box.InitFromBoxes(np.array([ [x,y,z,box_size,box_size,box_size]]),True)
    box.SetName('box')
    env.Add(box,True)
    return box

def check_col_with_box(env,x,y,z, box_size = 0.04):
    box = add_box(env, x,y,z,box_size)
    collision =  env.CheckCollision(box)
    env.RemoveKinBody(box)
    return collision


def generate_random_xyz(limit):
    x_lim,y_lim,z_lim = limit[0], limit[1], limit[2]
    x = np.random.rand()*(x_lim[1] - x_lim[0]) + x_lim[0]
    y = np.random.rand()*(y_lim[1] - y_lim[0]) + y_lim[0]
    z = np.random.rand()*(z_lim[1] - z_lim[0]) + z_lim[0]
    return x,y,z

def get_random_pose(env,limits, manip, frame_name="r_gripper_tool_frame"):
    robot = env.GetRobots()[0]
    is_success = False
    while is_success is not True:
        #Select which input space
        index = np.random.choice(limits.keys())
        x,y,z = generate_random_xyz(limits[index])
        
        #simple check for collision with a box
        if check_col_with_box(env,x,y,z):
            continue
            
        #do IK here
        xyz = [x,y,z]
        quat = [1,0,0,0] # wxyz
        pose = openravepy.matrixFromPose( np.r_[quat, xyz] )
        joint_angle = ku.ik_for_link(pose, manip, frame_name,
          filter_options = openravepy.IkFilterOptions.CheckEnvCollisions)
        if joint_angle is None:continue
        
        #further check for collision
        robot.SetDOFValues(joint_angle, manip.GetArmIndices())
        if env.CheckCollision(robot) or robot.CheckSelfCollision(): continue
            
        is_success = True
        
    xyz =  np.array([x,y,z])     
    return joint_angle, xyz

def get_random_pose_both(env,limits, right_arm, left_arm):
    robot = env.GetRobots()[0]
    init_joint = robot.GetActiveDOFValues()
    is_success = False
    
    while is_success is not True:
        #Get right pose
        robot.SetActiveDOFValues(init_joint)
        target_joint_right,xyz_target_right = get_random_pose(env,limits,right_arm, frame_name='r_gripper_tool_frame')
        #Get left pose
        robot.SetActiveDOFValues(init_joint)
        target_joint_left,xyz_target_left = get_random_pose(env,limits,left_arm, frame_name='l_gripper_tool_frame')
        #Combine both
        target_joint = np.concatenate([target_joint_right,target_joint_left])
        
        #check for collision
        robot.SetActiveDOFValues(target_joint)
        if env.CheckCollision(robot) or robot.CheckSelfCollision(): 
            print ('The IK solution has a collision!')
            continue
        else:
            is_success = True
    return target_joint_right,xyz_target_right, target_joint_left,xyz_target_left, target_joint
            
def remove_boxes(env,num):
    env.RemoveKinBodyByName('box')
    for i in range(num):
        env.RemoveKinBodyByName('box'+str(i))
        
        
def check_traj(env,result, target, threshold = 0.01):
    #check trajectory with the target as joint configuration
    robot = env.GetRobots()[0]
    traj = result.GetTraj()
    
    #check for collision
    if traj_is_safe(traj, robot) is not True:
        print "There is a collision within the trajectory!"
        return False

    #check optimal solver status
    if (result.GetStatus()[0] is not 0):
        print "Optimization is not converged!"
        return False

    #todo: put condition on either pose or joint constraints 

    #check target for joint constraints
    if (np.linalg.norm(target - traj[-1]) > threshold):
        print ('Target joint is not reached!')
        return False
    
    #Otherwise, return True
    return True

def check_traj_pose(env,result, target_left, target_right, threshold = 0.01):
    #check trajectory with Cartesian target
    robot = env.GetRobots()[0]
    traj = result.GetTraj()
    
    #check for collision
    if traj_is_safe(traj, robot) is not True:
        print "There is a collision within the trajectory!"
        return False

    #check optimal solver status
    if (result.GetStatus()[0] is not 0):
        print "Optimization is not converged!"
        return False

    robot.SetActiveDOFValues(traj[-1])
    T_left = robot.GetLink('l_gripper_tool_frame').GetTransform()
    T_right = robot.GetLink('r_gripper_tool_frame').GetTransform()
    pose_left = np.concatenate([T_left[0:3,3].flatten(), openravepy.quatFromRotationMatrix(T_left)])
    pose_right = np.concatenate([T_right[0:3,3].flatten(), openravepy.quatFromRotationMatrix(T_right)])
    
    #check target for joint constraints
    if (np.linalg.norm(target_left - pose_left) > threshold):
        print target_left,pose_left
        print ('Left pose is not reached!')
        return False

    if (np.linalg.norm(target_right - pose_right) > threshold):
        print target_right,pose_right
        print ('Right pose is not reached!')
        return False

    #Otherwise, return True
    return True

def plot_traj(env,traj, timestep = 0.2):
    robot = env.GetRobots()[0]
    for p in traj:
        robot.SetActiveDOFValues(p)
        time.sleep(timestep)
        
def print_result(results, method_names, with_func=False):
    print(' Method \t| Success Rate \t| Conv. Time \t| Traj. Cost \t| Func. Evals \t| QP Solves')
    for method in method_names:
        successes = np.array(results[method]['successes'][:])
        success = np.count_nonzero(successes)

        comp_times = np.array(results[method]['comp_times'][:])[successes]
        costs = np.array(results[method]['costs'][:])[successes]

        success_mean = success*100.0/len(successes)

        time_mean = np.sum(comp_times)/success
        time_std = np.std(comp_times)

        cost_mean = np.sum(costs)/success
        cost_std = np.std(costs)

        if with_func:
            func_evals = np.array(results[method]['func_evals'][:])[successes]
            func_mean = np.max(func_evals)/success
            func_std = np.std(func_evals)

            print('{0} \t& {1:.1f} \t& {2:.2f}$\\pm${3:.2f} \t& {4:.2f}$\\pm${5:.2f} \t & {6:.2f}$\\pm${7:.2f} \t \\\\'.format(method, success_mean, \
                                            time_mean, time_std, cost_mean, cost_std, func_mean, func_std)) 
        else:
            print('{0} \t& {1:.1f} \t& {2:.2f}$\\pm${3:.2f} \t& {4:.2f}$\\pm${5:.2f} \\\\'.format(method, success_mean, time_mean, time_std, cost_mean, cost_std)) 
            