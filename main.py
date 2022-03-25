import threading
import numpy as np
import Agent
from Aggregator import Aggregator
import Aggregator
import networkx as nx
from threading import Lock
import matplotlib as mpl
mpl.interactive(True)
import matplotlib.pyplot as plt
import time
import random
from cmath import inf
from operators.projectLinSpace import projectLinSpace
import pickle
from checkSolutionsSpace import checkSolutionsSpace,checkSolutionsSpace_QP
from scipy import sparse
import sys

if __name__ == '__main__':
    run_hsdm = True 
    run_PPP = True 
    exponent_hsdm_to_test = [ 0.6, 0.8, 1.0]
    N_random_initial_states = 3
    print("Initializing problem...")
    N=6   # N agents
    T_horiz = 1 # Horizon of the multi-period Cournot game
    N_markets = 3
    n_reg_agents= 2# int(N/3)  # Number of "regulated" agents
    n_reg_timesteps=1 #int(T_horiz/3)  # Number of "regulated" timesteps

    print("Done")
    # Parameters of algorithm
    N_iter=500

    # containers for saved variables
    x_hsdm={}
    x_not_hsdm={}
    x0={}
    residual_hsdm={}
    residual_not_hsdm={}

    cost_hsdm={}
    cost_hsdm_history={}
    cost_not_hsdm={}
    cost_not_hsdm_history={}
    if len(sys.argv) < 2:
        seed = 1
        job_id=0
    else:
        seed=int(sys.argv[1])
        job_id = int(sys.argv[2])
    random.seed(seed)
    print("Running with  random seed = ", seed)
    for n_init in range(N_random_initial_states):
        for n_agent in range(N):
            x0.update({ (n_init, n_agent): np.matrix(10* (-0.5+(np.random.rand(N_markets* T_horiz)))).T })
    # Determine markets in which agents compete and production costs
    n_i = [] # number of markets in which agent i participates 
    # A_i = [] # agent production -> markets
    quad_cost_i = [] # Quadratic prod. penalty for each product
    lin_cost_i = [] # Linear prod. penalty
    min_prod_i = [] # minimum production of each product for agent  i
    markets_i = [] # markets in which i participates
    d = np.matrix(np.random.rand(N_markets, 1 )) # Price stiffness
    P_max = 0*np.matrix(np.random.rand(N_markets, 1 ))  # max price
    max_prod = 5 * np.ones((N_markets, 1)) # market saturation level
    regulated_agent = random.sample(range(N), n_reg_agents)
    regulated_timestep = random.sample(range(T_horiz), n_reg_timesteps)
    weight_reg_ag={}
    lin_weight_reg_ag={}
    lin_cost = -10 + 10 * np.random.rand(N_markets)
    for i in range(N):
        n_i.append(N_markets) #n_i.append(np.random.random_integers(1, N_markets, 1).item())
        markets_i.append(random.sample(range(N_markets), n_i[i]))
        quad_cost_i.append(np.zeros((n_i[i], 1)))
        lin_cost_i.append(lin_cost)
        min_prod_i.append(-1 + 2*np.random.rand(n_i[i], 1))

    for i in regulated_agent:
        weight_reg_ag.update({ i: 1 + np.random.rand(n_i[i], 1) })
        lin_weight_reg_ag.update({ i: -1 + 2 * np.random.rand(n_i[i], 1)  })

    stepsize_primal=2*max(d).item()*N
    dual_stepsize = 0.1
    stepsize_hsdm = 1
    for n_init in range(N_random_initial_states):
        print("Testing initial state with index", n_init)
        for test in range(len(exponent_hsdm_to_test)):
            exponent_hsdm = exponent_hsdm_to_test[test]
            print("Exponent of HSDM set to", exponent_hsdm)        
            # Simulation with HSDM
            common_variables = Agent.CommonVariables(d, T_horiz)
            if run_hsdm:
                time_1 = time.perf_counter() 
                agents_hsdm = []
                traffic_light = np.zeros((N,1))
                completed_iteration = -1*np.ones((N,1))
                completed_hsdm = -1*np.ones((N,1))
                print("Initializing agents for HSDM run...")
                param = Agent.Param(T_horiz=T_horiz, N_iter=N_iter, use_hsdm=True, \
                        N_agents= N, N_markets=N_markets, stepsize_primal=stepsize_primal, \
                        exponent_hsdm=exponent_hsdm, stepsize_hsdm=stepsize_hsdm)
                for i in range(N): 
                    agents_hsdm.append(Agent.Agent(i,  param, markets_i[i], quad_cost_i[i], lin_cost_i[i], min_prod_i[i], P_max, max_prod, common_variables, x0=x0[(n_init, i)]))
                print("Done")
                print("Checking dimension of solution space")
                x_osqp = checkSolutionsSpace_QP(agents_hsdm,common_variables)
                sigma_HSDM = np.matrix(np.zeros((N_markets * T_horiz, 1) ))
                sel_fun_gradient = []
                for agent in agents_hsdm:
                    sel_fun_gradient.append(np.matrix(np.zeros((agent.N_dec_var, 1) )))
                lambda_shared = np.matrix(np.zeros((agents_hsdm[0].N_shared_constr, 1) ))
                aggregator_hsdm=Aggregator.Aggregator(agents_hsdm, N_markets, T_horiz, regulated_agent, weight_reg_ag, lin_weight_reg_ag, regulated_timestep)
                time_2 = time.perf_counter() 
                # initialization_time.update( { (N,g) : (time_2 - time_1) } )
                print("Running agent threads for HSDM...")
                cv_primal = threading.Condition()
                cv_dual = threading.Condition()
                cv_hsdm = threading.Condition()
                cv_aggregate = threading.Condition()
                for agent in agents_hsdm:
                    agent.run(lambda_shared, sigma_HSDM, completed_iteration, completed_hsdm,  traffic_light, sel_fun_gradient, cv_primal, cv_dual, cv_hsdm, cv_aggregate)
                aggregator_hsdm.run(sigma_HSDM, completed_iteration, completed_hsdm, traffic_light, agents_hsdm, lambda_shared, sel_fun_gradient,cv_primal, cv_dual, cv_hsdm, cv_aggregate, dual_stepsize=dual_stepsize)
                while not completed_iteration[0]==N_iter-1:
                    time.sleep(1)
                # iteration_time.update( {(N, g) : aggregator_hsdm.avg_time })

                # Store results
                for agent in agents_hsdm:
                    x_hsdm.update({(agent.id, test, n_init):  agent.x})
                cost_hsdm.update({ (test, n_init): aggregator_hsdm.select_fun.evaluate(agents_hsdm) })
                residual_hsdm.update({ (test, n_init): aggregator_hsdm.residual})
                cost_hsdm_history.update({ (test, n_init): aggregator_hsdm.cost_history })

        ###############################
        # Simulation without HSDM
        if run_PPP:
            time_1 = time.perf_counter() 
            agents_not_hsdm = []
            traffic_light = np.zeros((N,1))
            completed_iteration = -1*np.ones((N,1))
            completed_hsdm = -1*np.ones((N,1))
            print("Initializing agents for PPP run...")
            param = Agent.Param(T_horiz=T_horiz, N_iter=N_iter, use_hsdm=False, \
                    N_agents= N, N_markets=N_markets, stepsize_primal=stepsize_primal, \
                    exponent_hsdm=exponent_hsdm, stepsize_hsdm=stepsize_hsdm)
            for i in range(N): 
                agents_not_hsdm.append(Agent.Agent(i,  param, markets_i[i], quad_cost_i[i], lin_cost_i[i], min_prod_i[i], P_max, max_prod, common_variables, x0=x0[(n_init, i)]))
            print("Done")
            sigma_not_HSDM = np.matrix(np.zeros((N_markets * T_horiz, 1) ))
            sel_fun_gradient_not_hsdm = []
            for agent in agents_not_hsdm:
                sel_fun_gradient_not_hsdm.append(np.matrix(np.zeros((agent.N_dec_var, 1) )))
            lambda_shared = np.matrix(np.zeros((agents_not_hsdm[0].N_shared_constr, 1) ))
            aggregator_not_hsdm=Aggregator.Aggregator(agents_not_hsdm, N_markets, T_horiz, regulated_agent, weight_reg_ag, lin_weight_reg_ag, regulated_timestep)
            time_2 = time.perf_counter() 
            # initialization_time.update( { (N,g) : (time_2 - time_1) } )
            print("Running agent threads for PPP...")
            cv_primal = threading.Condition()
            cv_dual = threading.Condition()
            cv_hsdm = threading.Condition()
            cv_aggregate = threading.Condition()
            for agent in agents_not_hsdm:
                agent.run(lambda_shared, sigma_not_HSDM, completed_iteration, completed_hsdm, traffic_light, sel_fun_gradient_not_hsdm, cv_primal, cv_dual, cv_hsdm, cv_aggregate)
            aggregator_not_hsdm.run(sigma_not_HSDM, completed_iteration, completed_hsdm, traffic_light, agents_not_hsdm, lambda_shared, sel_fun_gradient_not_hsdm, cv_primal, cv_dual, cv_hsdm, cv_aggregate,  dual_stepsize=dual_stepsize)
            while not completed_iteration[0]==N_iter-1:
                time.sleep(1)
            ################################
            # Storing results
            
            for agent in agents_not_hsdm:
                x_not_hsdm.update({(agent.id, n_init):  agent.x})
            cost_not_hsdm.update( { (n_init): aggregator_not_hsdm.select_fun.evaluate(agents_not_hsdm) })
            residual_not_hsdm.update({ (n_init): aggregator_not_hsdm.residual })
            cost_not_hsdm_history.update({ (n_init): aggregator_not_hsdm.cost_history })
################################
if not run_hsdm:
    sigma_HSDM=[]
if not run_PPP:
    sigma_not_hsdm=[]

if run_PPP and run_hsdm:
    filename = "saved_sol"+str(job_id)+".pkl"
    f= open(filename, 'wb')  
    pickle.dump([x_hsdm, x_not_hsdm, residual_hsdm, residual_not_hsdm, sigma_HSDM, \
                sigma_not_HSDM, cost_hsdm, cost_not_hsdm, cost_hsdm_history, cost_not_hsdm_history, T_horiz, exponent_hsdm_to_test, N, x_osqp], f)
    f.close

