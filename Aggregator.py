
from cmath import inf
import re
import threading 
import numpy as np
from scipy import sparse
import time
class Aggregator:
    def __init__(self, agents, N_markets, T_horiz, regulated_agent, weight_reg_ag, lin_weight_reg_ag, regulated_timestep):
        # the aggregator logs the primal trajectories
        self.x_traj={}
        self.select_fun = self.SelectFun(agents, N_markets, T_horiz, regulated_agent, weight_reg_ag, lin_weight_reg_ag, regulated_timestep)
        self.avg_time =0
        pass
    def run(self, sigma, completed_iteration, completed_hsdm, traffic_light, agents, lambda_shared, sel_fun_gradient, cv_primal, cv_dual, cv_hsdm, cv_aggregate,  dual_stepsize=0.01):
        self.thread = threading.Thread(target = self.threaded_fun, args=(sigma, completed_iteration, completed_hsdm, traffic_light, agents,\
             lambda_shared, dual_stepsize, sel_fun_gradient, cv_primal, cv_dual, cv_hsdm, cv_aggregate))
        self.thread.start()

    def threaded_fun(self, sigma, completed_iteration, completed_hsdm, traffic_light, agents, lambda_shared, dual_stepsize, sel_fun_gradient, cv_primal, cv_dual, cv_hsdm, cv_aggregate):
            iter_index = 0
            # Wait for agents to complete their iteration, 
            # then update sigma and give green light for next iteration
            # logging residuals
            self.residual= np.zeros((1+int(agents[0].N_iter/20), 1))
            self.cost_history= np.zeros((1+int(agents[0].N_iter/20), 1))
            for agent in agents:
                self.x_traj.update({agent.id: np.zeros((agent.N_dec_var, agent.N_iter))})
            self.avg_time = 0
            while(iter_index < agents[0].N_iter): 
                    time1 = time.perf_counter() 
                    # check if every agent has completed the current iteration
                    with cv_primal:
                        while not np.all(completed_iteration>=iter_index):
                            cv_primal.wait()
                    with cv_dual:
                        sigma.fill(0)
                        for agent in agents:
                            sigma[:] = sigma + agent.compute_contribution_to_aggregation()
                            self.x_traj[agent.id][:, iter_index]= np.ravel(agent.x)
                            lambda_shared[:] = lambda_shared + dual_stepsize * (agent.shared_const.Ai * agent.x - agent.shared_const.bi)
                        self.select_fun.update_selection_fun_gradient(sel_fun_gradient, agents) 
                        lambda_shared[:] = np.maximum(lambda_shared[:], 0)   
                        # update residuals
                        if iter_index%20 ==0:
                            residual_x = np.zeros((len(agents),1))
                            residual_lambda = np.zeros((len(agents),1))
                            constraint_violation=np.array([])
                            for agent in agents: 
                                x_init=np.copy(agent.x)
                                pgrad_i = np.matrix(agent.compute_pseudogradient_i(np.copy(agent.x), np.copy(agent.dual_var), np.copy(sigma)))
                                if np.linalg.norm(agent.x- x_init)>0.0001:
                                    print("arresting")
                                x_after_pgrad = np.copy(agent.x)
                                x_transformed=np.matrix(agent.project_local_constraints(np.copy(agent.x) -  pgrad_i))
                                if np.linalg.norm(agent.x- x_after_pgrad)>0.0001:
                                    print("arresting")
                                residual_x[agent.id][0] = np.linalg.norm(agent.x - x_transformed , inf )
                                constraint_violation = constraint_violation + agent.compute_constraint_violation(agent.x) if constraint_violation.size else agent.compute_constraint_violation(agent.x) 
                                
                            lambda_transformed = lambda_shared + constraint_violation
                            lambda_transformed = np.maximum(lambda_transformed, 0)
                            residual_lambda[agent.id][0]  = np.linalg.norm( lambda_shared - lambda_transformed, inf )
                            self.residual[int(iter_index/20)][0] = np.max( (np.amax(residual_x), np.amax(residual_lambda)) )
                            self.cost_history[int(iter_index/20)][0] = self.select_fun.evaluate(agents)
                        traffic_light.fill(True)
                        cv_dual.notify_all()
                    # wait for HSDM step
                    with cv_hsdm:
                        while not np.all(completed_hsdm>=iter_index):
                            cv_hsdm.wait()
                    with cv_aggregate:
                        sigma.fill(0)
                        for agent in agents:
                            sigma[:] = sigma + agent.compute_contribution_to_aggregation()
                        traffic_light.fill(True)
                        cv_aggregate.notify_all()
                    time2 = time.perf_counter()
                    self.avg_time = (self.avg_time* (iter_index) + (time2-time1)) /(iter_index+1)
                    if iter_index%20==0:
                        print("Completed iteration ", iter_index, " avg iteration time: ", self.avg_time, "Residual: ", self.residual[int(iter_index/20)][0])
                    iter_index = iter_index+1
                    

    class SelectFun:
        def __init__(self, agents, N_markets, T_horiz, regulated_agents, weight_reg_ag, lin_weight_reg_ag, regulated_timestep):
            self.Q_i=[]
            self.q_i=[]
            self.N_dec_var_tot = 0
            for agent in agents:
                self.N_dec_var_tot = self.N_dec_var_tot + agent.N_dec_var
            Q = np.matrix(np.zeros((self.N_dec_var_tot, self.N_dec_var_tot)))
            index =0
            for agent in agents:
                if agent.id in regulated_agents:
                    Sel_reg_t = np.eye(agent.N_dec_var)
                    N_dec_var_per_timestep = int(agent.N_dec_var/T_horiz)
                    index_t = 0
                    for t in range(T_horiz):
                        if t not in regulated_timestep:
                            Sel_reg_t[index_t: index_t + N_dec_var_per_timestep, :] = 0* Sel_reg_t[index_t: index_t + N_dec_var_per_timestep, :]
                        index_t = index_t + N_dec_var_per_timestep
                    Q[index:index + agent.N_dec_var, index:index + agent.N_dec_var] = np.diag(weight_reg_ag[agent.id]) * Sel_reg_t
                index = index + agent.N_dec_var
            self.Q=sparse.csc_matrix(Q)
            index = 0
            for agent in agents:
                self.Q_i.append( self.Q [index: index + agent.N_dec_var][:])
                if agent.id in regulated_agents:
                    self.q_i.append(np.matrix(lin_weight_reg_ag[agent.id]))
                else:
                    self.q_i.append(np.matrix(np.zeros( (agent.N_dec_var, 1) ) ))
                index = index + agent.N_dec_var
            self.x_complete=np.matrix(np.zeros((self.N_dec_var_tot, 1)))
            self.q=np.matrix(np.zeros((self.N_dec_var_tot, 1)))
            index = 0
            for agent in agents:
                self.q[index:index + agent.N_dec_var, 0]=self.q_i[agent.id][:]
                index = index + agent.N_dec_var


        def update_selection_fun_gradient(self, sel_fun_gradient, agents):
            index = 0
            for agent in agents:
                self.x_complete[index:index + agent.N_dec_var]=agent.x
                index = index + agent.N_dec_var
            for agent in agents:
                sel_fun_gradient[agent.id][:]=self.Q_i[agent.id] * self.x_complete + self.q_i[agent.id]

        def evaluate(self, agents):
            index = 0
            for agent in agents:
                self.x_complete[index:index + agent.N_dec_var]=agent.x
                index = index + agent.N_dec_var
            return 0.5*np.transpose(self.x_complete)*self.Q*self.x_complete + np.transpose(self.q) * self.x_complete 