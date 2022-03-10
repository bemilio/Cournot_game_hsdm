from operator import index

from operators.backwardStep import backwardStep
from operators.projectLinSpace import projectLinSpace
from numpy import random
# from numba import jit
import numpy as np
import warnings
import threading 
import time
import debugpy
import networkx as nx
from cmath import inf
from scipy import sparse
import matplotlib.pyplot as plt


class Param:
    def __init__(self, T_horiz, N_iter, use_hsdm, \
            N_agents, N_markets, stepsize_primal=0.1, exponent_hsdm=0.8, stepsize_hsdm=1):
        self.stepsizes = self.Stepsizes(stepsize_primal,exponent_hsdm)
        self.N_iter = N_iter
        self.T_horiz = T_horiz
        self.use_hsdm = use_hsdm
        self.N_agents = N_agents
        self.stepsize_hsdm = stepsize_hsdm
        self.N_markets = N_markets
        
    class Stepsizes:
        def __init__(self, stepsize_primal, exponent_hsdm):
            self.primal = stepsize_primal
            self.exponent_hsdm = exponent_hsdm

class CommonVariables:
    def __init__(self, d, T_horiz):
        self.gen_shared_cost(d, T_horiz)

    def gen_shared_cost(self, d, T_horiz):
        self.C = sparse.csc_matrix(np.kron(np.eye(T_horiz), np.diagflat(d) ) )

class Agent:
    def __init__(self, id, param, markets_i, quad_cost_i, lin_cost_i, min_prod_i,  P_max, max_prod, common_variables):
        self.id = id
        self.param = param
        T_horiz = param.T_horiz
        self.N_dec_var =  len(markets_i)* T_horiz
        random.seed()
        Ai = np.matrix(np.zeros( ( param.N_markets, len(markets_i)) ))
        I_mT = np.matrix(np.eye( param.N_markets ))
        for i in range(len(markets_i)):
            Ai[:, i] = I_mT[:, markets_i[i]]

        self.loc_to_sigma = sparse.csc_matrix(np.kron(np.eye(T_horiz), Ai) )
        self.aggreg_cost = self.AggregCost(common_variables.C, self.loc_to_sigma) #this matrix is rectangular (maps from sigma to local var.)
        self.local_obj = self.LocalObjective(quad_cost_i, lin_cost_i, T_horiz, P_max, common_variables.C, self.loc_to_sigma)
        self.local_const = self.LocalConst(T_horiz, len(markets_i), min_prod_i)
        self.shared_const = self.SharedConst(Ai, T_horiz, max_prod, param.N_agents)
        self.x = np.matrix(np.random.rand(self.N_dec_var)).T
        self.N_iter = param.N_iter
        self.N_shared_constr = self.shared_const.Ai.shape[0]

    class AggregCost:
        def __init__(self, C, loc_to_sigma):
            self.C = (loc_to_sigma.T * C)

    class LocalObjective:
        def __init__(self, quad_cost_i, lin_cost_i, T_horiz, P_max, C, loc_to_sigma):
            Ai_bar = loc_to_sigma
            self.Q = sparse.csc_matrix( np.kron(np.eye(T_horiz), np.diagflat(quad_cost_i))) +  Ai_bar.T * C * Ai_bar
            q = np.matrix( lin_cost_i  ).T
            self.q = np.kron(np.ones((T_horiz, 1)), q) #-  Ai_bar.T * np.kron(np.ones((T_horiz, 1)),P_max)

    class LocalConst:
        def __init__(self, T_horiz, n_local_markets, bi):
            # Production greater than minimum value
            A_min_prod = np.kron( np.ones( (1,T_horiz) ), - np.eye(n_local_markets) ) 
            b_min_prod = -bi
            self.A = np.vstack( (A_min_prod, -1*np.eye( T_horiz * n_local_markets ), 1*np.eye( T_horiz * n_local_markets ) ) ) 
            self.b = np.matrix( np.vstack( (b_min_prod,  100* ( np.ones((T_horiz* n_local_markets, 1)) ), 100* ( np.ones((T_horiz* n_local_markets, 1)) )  )  ) )
            self.Aeq = (1, n_local_markets * T_horiz ) # dummy
            self.beq =np.matrix([0])
            self.Aall_sparse = sparse.csc_matrix(sparse.vstack((self.A, self.Aeq)))
            self.u= np.vstack((self.b, self.beq))
            self.l= np.vstack((-np.inf * np.ones((self.b.shape[0],1)), self.beq ))

    class SharedConst:
        def __init__(self, Ai, T_horiz, max_prod, N_agents):
            Ai_bar = np.kron(Ai,np.eye(T_horiz)) 
            self.Ai = Ai_bar
            self.bi = np.kron(max_prod/N_agents, np.ones((T_horiz, 1)) ) 

    def compute_linear_dual_cost(self, dual_var):
        q_dual = self.shared_const.Ai.transpose() * dual_var
        return(q_dual)

    def compute_linear_aggregative_cost(self, sigma, x):
        return( self.aggreg_cost.C * (sigma - self.loc_to_sigma * x))

    def compute_constraint_violation(self, x):
        violation = self.shared_const.Ai * x - self.shared_const.bi
        return(violation)

    def project_local_constraints(self,x):
        (proj_x, solution_status) =projectLinSpace(self.local_const.Aall_sparse,\
             self.local_const.l, self.local_const.u, x)
        return(proj_x)

    def compute_pseudogradient_i(self, x, dual_var, sigma):
        q_loc = self.local_obj.q
        q_dual = self.compute_linear_dual_cost(dual_var)
        q_aggr = self.compute_linear_aggregative_cost(sigma, x)
        pseudogr = self.local_obj.Q * x + q_loc + q_dual + q_aggr
        return(pseudogr)

    def compute_contribution_to_aggregation(self):
        return(self.loc_to_sigma * self.x)

################ Main loop #######################
    def threaded_fun(self, lambda_shared, sigma, completed_iteration, completed_hsdm, traffic_light, sel_fun_gradient, cond_var_primal, cond_var_dual, cond_var_hsdm, cond_var_aggregate):
        # Main loop
        # Q = sparse.csc_matrix(self.local_obj.Q)
        Q = self.local_obj.Q
        for k in range(self.N_iter):
            step_hsdm = self.param.stepsize_hsdm /((k+1)**self.param.stepsizes.exponent_hsdm)
            self.dual_var = lambda_shared # this is updated by the aggregator
            # linear part of cost
            q_loc = self.local_obj.q
            q_dual = self.compute_linear_dual_cost(self.dual_var)
            q_aggr = self.compute_linear_aggregative_cost(sigma, self.x)
            q = q_loc + q_dual + q_aggr
            # Proximal point
            x_new = backwardStep(Q, q, self.local_const.Aall_sparse, self.local_const.l, self.local_const.u, self.x, self.param.stepsizes.primal)[0]
            self.x = x_new
            with cond_var_primal:
                completed_iteration[self.id]=k
                cond_var_primal.notify_all()
            with cond_var_dual:
                while not traffic_light[self.id]:
                    # wait for the aggregator to signal that the gradients are ready
                    cond_var_dual.wait()
                    # time.sleep(0.00001)
                traffic_light[self.id] = False
            # Performs HSDM step
            with cond_var_hsdm:
                if self.param.use_hsdm:
                    self.x = x_new -  step_hsdm* sel_fun_gradient[self.id]
                completed_hsdm[self.id]=k
                cond_var_hsdm.notify_all()
            # wait for agg. variable update
            with cond_var_aggregate:
                while not traffic_light[self.id]:
                    cond_var_aggregate.wait()
                traffic_light[self.id] = False
################################################

    def run(self, lambda_shared, sigma, completed_iteration, completed_hsdm, traffic_light, sel_fun_gradient, cond_var_primal, cond_var_dual, cond_var_hsdm, cond_var_aggregate):
        self.thread = threading.Thread(target = self.threaded_fun,  \
            args=(lambda_shared, sigma, completed_iteration, completed_hsdm, traffic_light, sel_fun_gradient, cond_var_primal, cond_var_dual, cond_var_hsdm, cond_var_aggregate))
        self.thread.start()
