import numpy as np
import pickle
import trajoptpy.math_utils as mu

class Regressor():
    def __init__(self, is_transform=None):
        self.is_transform = is_transform
        self.pca = None
        
    def save_to_file(self,filename):
        f = open(filename + '.pkl', 'wb')
        pickle.dump(self.__dict__,f)
        f.close()
       
    def load_from_file(self,filename):
        f = open(filename + '.pkl', 'rb')
        self.__dict__ = pickle.load(f)
            
#Nearest Neighbor Regressor
class NN_Regressor(Regressor):
    def fit(self,x,y):
        self.x = x.copy()
        self.y = y.copy()
        
    def nearest(self,x_i):
        dists = []
        for x_j in self.x:
            dists.append(np.linalg.norm(x_i-x_j))
        max_index = np.argmin(dists)
        return max_index
    
    def predict(self,x):
        y_index = self.nearest(x)
        y_cur = self.y[y_index:y_index+1,:].copy()
        return y_cur, np.array([[0]])
    

#Straight Line Planner
class Straight_Regressor(Regressor):
    def __init__(self, dof, n_steps, is_transform=None):
        self.dof = dof
        self.n_steps = n_steps
        self.is_transform = is_transform
    def predict(self, init_joint, target_joint):
        inittraj = np.empty((self.n_steps, self.dof))
        inittraj = mu.linspace2d(init_joint, target_joint, self.n_steps)
        return inittraj,np.array([[0]])
            
    def predict_with_waypoint(self, init_joint, target_joint, waypoint = None, waypoint_step = 0):
        inittraj = np.empty((self.n_steps, self.dof))
        inittraj[:waypoint_step+1] = mu.linspace2d(init_joint, waypoint, waypoint_step+1)
        inittraj[waypoint_step:] = mu.linspace2d(waypoint, target_joint, self.n_steps - waypoint_step)
        return inittraj,np.array([[0]])

    
#GPy GP Regressor
import GPy


class GPy_Regressor(Regressor):
    def __init__(self, dim_input, is_transform = False):
        self.is_transform = is_transform #whether the output should be transformed or not. Possible option: PCA, RBF, etc. 
        self.dim_input = dim_input
        
    def fit(self,x,y, num_restarts = 10):
        kernel = GPy.kern.RBF(input_dim=self.dim_input, variance=0.1,lengthscale=0.3, ARD=True) + GPy.kern.White(input_dim=self.dim_input)
        self.gp = GPy.models.GPRegression(x, y, kernel)
        self.gp.optimize_restarts(num_restarts=num_restarts)
            
    def predict(self,x):
        y,cov = self.gp.predict(x)
        return y,cov
    
#Sparse GP Regressor
class Sparse_GPy_Regressor(Regressor):
    def __init__(self, num_z = 100, is_transform = False):
        self.zdim = num_z
        self.is_transform = is_transform
    def fit(self,x,y):
        Z = x[0:self.zdim]
        self.sparse_gp = GPy.models.SparseGPRegression(x, y, Z=Z)
        self.sparse_gp.optimize('bfgs')
    def predict(self,x):
        y,cov = self.sparse_gp.predict(x)
        return y,cov
    
#BGMR
import pbdlib as pbd
import pdb
class DP_GLM_Regressor(Regressor):
    def fit(self,x,y, n_components = 10, n_init = 20 , weight_type = 'dirichlet_process'):
        self.x_joint = np.concatenate([x, y], axis=1)
        self.n_joint = self.x_joint.shape[1]
        self.n_in = x.shape[1]
        self.n_out = y.shape[1]
        self.joint_model = pbd.VBayesianGMM({'n_components':n_components, 'n_init':n_init, 'reg_covar': 0.00006 ** 2,
     'covariance_prior': 0.00002 ** 2 * np.eye(self.n_joint),'mean_precision_prior':1e-9,'weight_concentration_prior_type':weight_type})
        self.joint_model.posterior(data=self.x_joint, dp=False, cov=np.eye(self.n_joint))
    def predict(self,x, return_gmm=True, return_more = False):
        result = self.joint_model.condition(x, slice(0, self.n_in), slice(self.n_in, self.n_joint),return_gmm = return_gmm) #
        #pdb.set_trace()
        
        if return_gmm:
            if return_more:
                return result[0], result[1], result[2] 
            else:
                index = np.argmax(result[0])
                return result[1][index], result[2][index]
        else:
            return result[0], result[1]