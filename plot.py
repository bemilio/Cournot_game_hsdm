from functools import total_ordering
import matplotlib as mpl
from cmath import inf
import os

mpl.interactive(True)
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 14
})
import networkx as nx
import numpy as np
import pickle
from scipy import sparse

check_test = False

# directory = '/Users/ebenenati/Repositories/Results_hsdm_ppp_academic/Statistics_16_mar/'
directory='/Users/ebenenati/surfdrive/TUDelft/Simulations/PPP_hsdm_academic/Statistics_multiple_beta/'
x_hsdm_all=[]
x_not_hsdm_all=[]
residual_hsdm_all=[]
residual_not_hsdm_all=[]
cost_hsdm_all=[]
cost_not_hsdm_all=[]
cost_hsdm_history_all=[]
cost_not_hsdm_history_all=[]
for filename in os.listdir(directory):
    if filename.find('.pkl')>=0:
        f=open(directory+filename, 'rb')
        # f = open('/Users/ebenenati/Repositories/Results_MDP/local_results/80_agents_5000It/80_agents_5000It.pkl', 'rb')
        x_hsdm, x_not_hsdm, residual_hsdm, residual_not_hsdm, sigma_HSDM, \
            sigma_not_hsdm, cost_hsdm, cost_not_hsdm, cost_hsdm_history, cost_not_hsdm_history, T_horiz, tested_hsdm_exponents, N_agents, x_osqp = pickle.load(f)
        f.close()
        x_hsdm_all.append(x_hsdm)
        x_not_hsdm_all.append(x_not_hsdm)
        residual_hsdm_all.append(residual_hsdm)
        residual_not_hsdm_all.append(residual_not_hsdm)
        cost_hsdm_all.append(cost_hsdm)
        cost_not_hsdm_all.append(cost_not_hsdm)
        cost_hsdm_history_all.append(cost_hsdm_history)
        cost_not_hsdm_history_all.append(cost_not_hsdm_history)
# sanity check
# test = 0
# sigma_computed = np.zeros((x_hsdm[(0, test)].shape[0], x_hsdm[(0, test)].shape[1]))
# for agent_id in range(N_agents):
#     sigma_computed = sigma_computed + x_hsdm[(agent_id, test)]
# print('For hsdm The difference between computed and saved sigma is: ', np.linalg.norm(sigma_computed - sigma_HSDM))
# sigma_computed_not_hsdm = np.zeros((x_hsdm[(0, test)].shape[0], x_hsdm[(0, test)].shape[1]))
# for agent_id in range(N_agents):
#     sigma_computed_not_hsdm = sigma_computed_not_hsdm + x_not_hsdm[(agent_id, test)]
# print('For hsdm The difference between computed and saved sigma is: ', np.linalg.norm(sigma_computed_not_hsdm - sigma_not_hsdm))

hsdm_exponent_to_plot = 1
if hsdm_exponent_to_plot in tested_hsdm_exponents:
    index_exponent_to_plot = tested_hsdm_exponents.index(hsdm_exponent_to_plot)
else:
    raise ValueError("The requested HSDM exponent to plot is not present in these simulations")

N_problems = len(x_hsdm_all)
N_exponents_hsdm = 1+max([key[0] for key in list(cost_hsdm_all[0].keys()) ])
N_init_conditions_tested = 1+max([key[1] for key in list(cost_hsdm_all[0].keys())])

cost_improvement_rel=[]
cost_improvement_rel=np.zeros((N_problems, N_init_conditions_tested))
for problem in range(N_problems):
    for index_init_state in range(N_init_conditions_tested):
        cost_improvement_rel[problem, index_init_state] = (cost_not_hsdm_all[problem][(index_init_state)]  - cost_hsdm_all[problem][(index_exponent_to_plot, index_init_state)]) /cost_not_hsdm_all[problem][index_init_state] \
              if cost_not_hsdm_all[problem][index_init_state]>0.0001 else 0

distance_optimizers=[]
distance_optimizers=np.zeros((N_problems, N_init_conditions_tested))
for problem in range(N_problems):
    for index_init_state in range(N_init_conditions_tested):
        stacked_x_hsdm=np.matrix([])
        stacked_x_not_hsdm=np.matrix([])
        for index_agent in range(N_agents):
            stacked_x_hsdm = np.vstack(( stacked_x_hsdm, x_hsdm_all[problem][(index_agent, index_exponent_to_plot, index_init_state )] )) if stacked_x_hsdm.size else x_hsdm_all[problem][(index_agent, index_exponent_to_plot, index_init_state )] 
            stacked_x_not_hsdm = np.vstack( ( stacked_x_not_hsdm, x_not_hsdm_all[problem][(index_agent, index_init_state )] ) ) if stacked_x_not_hsdm.size else x_not_hsdm_all[problem][(index_agent, index_init_state )]
        distance_optimizers[problem, index_init_state] = np.linalg.norm(stacked_x_hsdm  - stacked_x_not_hsdm, inf) 

## Plot residual for all exponents

len_residual = residual_hsdm_all[0][(0,0)].shape[0]
residual_hsdm_stacked={}
for i in range(N_exponents_hsdm):
    residual_hsdm_stacked.update({tested_hsdm_exponents[i]: np.matrix(np.zeros((N_init_conditions_tested*N_problems, len_residual))) })
residual_not_hsdm_stacked = np.matrix(np.zeros((N_init_conditions_tested*N_problems, len_residual))) 
index = 0


for i_problem in range(N_problems):
    for i_initial_cond in range(N_init_conditions_tested):
        for i_exponent in range(N_exponents_hsdm):
            residual_hsdm_stacked[tested_hsdm_exponents[i_exponent]][index, :] = np.ravel(residual_hsdm_all[i_problem][(i_exponent, i_initial_cond)])
        residual_not_hsdm_stacked[index,:] = np.ravel(residual_not_hsdm_all[i_problem][i_initial_cond])
        index = index +1

average_residual_hsdm = np.mean(residual_hsdm_stacked[tested_hsdm_exponents[index_exponent_to_plot]], axis =0)
average_residual_not_hsdm = np.mean(residual_not_hsdm_stacked, axis =0)
min_residual_hsdm = np.min(residual_hsdm_stacked[tested_hsdm_exponents[index_exponent_to_plot]], axis =0)
min_residual_not_hsdm = np.min(residual_not_hsdm_stacked, axis =0)
max_residual_hsdm = np.max(residual_hsdm_stacked[tested_hsdm_exponents[index_exponent_to_plot]], axis =0)
max_residual_not_hsdm = np.max(residual_not_hsdm_stacked, axis =0)

x_total_hsdm = np.matrix([])
N=len(x_hsdm)

fig, ax = plt.subplots(figsize=(6, 2.5), layout='constrained') 
N_iter = (average_residual_hsdm.shape[1]-1) * 20
colors = ["#e377c2", "#2ca02c", "#1f77b4"]
for i_exponent in range(N_exponents_hsdm):
    average_residual_hsdm = np.mean(residual_hsdm_stacked[tested_hsdm_exponents[i_exponent]], axis =0)
    ax.loglog(np.arange(0, N_iter,20), np.ravel(average_residual_hsdm[:,:-1]), color=colors[i_exponent], label=tested_hsdm_exponents[i_exponent])
ax.loglog(np.arange(0, N_iter,20), np.ravel(average_residual_not_hsdm[:,:-1]), color="#ff7f0e", label="PPP")
# ax.fill_between(np.arange(0, N_iter,20), np.ravel(min_residual_hsdm[:,:-1]), np.ravel(max_residual_hsdm[:,:-1]), alpha=0.2)
# ax.fill_between(np.arange(0, N_iter,20), np.ravel(min_residual_not_hsdm[:,:-1]), np.ravel(max_residual_not_hsdm[:,:-1]), alpha=0.2)
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.ylim((10**(-8), 10**(1)))
plt.xlim((10**(1), 5*10**(4)))
plt.grid()
# plt.title('Residual')
plt.legend(loc="lower left", fontsize=12)
plt.savefig('Residual_exponents.pdf')  

## Plot residual for a particular exponent


fig, ax = plt.subplots(figsize=(6, 2.5), layout='constrained') 
N_iter = (average_residual_hsdm.shape[1]-1) * 20
ax.loglog(np.arange(0, N_iter,20), np.ravel(average_residual_hsdm[:,:-1]), color="#1f77b4", label="Algorithm 1")
ax.loglog(np.arange(0, N_iter,20), np.ravel(average_residual_not_hsdm[:,:-1]), color="#ff7f0e", label="PPP")
ax.fill_between(np.arange(0, N_iter,20), np.ravel(min_residual_hsdm[:,:-1]), np.ravel(max_residual_hsdm[:,:-1]), alpha=0.2)
ax.fill_between(np.arange(0, N_iter,20), np.ravel(min_residual_not_hsdm[:,:-1]), np.ravel(max_residual_not_hsdm[:,:-1]), alpha=0.2)
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.ylim((10**(-8), 10**(1)))
plt.xlim((10**(1), 5*10**(4)))
plt.grid()

# plt.title('Residual')
plt.legend(loc="lower left", fontsize=12)
plt.savefig('Residual_exponents.pdf') 





## Plot cost for all exponents

len_residual = residual_hsdm_all[0][(0,0)].shape[0]
cost_hsdm_stacked={}
for i in range(N_exponents_hsdm):
    cost_hsdm_stacked.update({tested_hsdm_exponents[i]: np.matrix(np.zeros((N_init_conditions_tested*N_problems, len_residual))) })
cost_not_hsdm_stacked = np.matrix(np.zeros((N_init_conditions_tested*N_problems, len_residual))) 
index = 0

for i_problem in range(N_problems):
    for i_initial_cond in range(N_init_conditions_tested):
        for i_exponent in range(N_exponents_hsdm):
            cost_hsdm_stacked[tested_hsdm_exponents[i_exponent]][index, :] = np.ravel(cost_hsdm_history_all[i_problem][(i_exponent, i_initial_cond)])
        cost_not_hsdm_stacked[index,:] = np.ravel(cost_not_hsdm_history_all[i_problem][i_initial_cond])
        index = index +1

average_cost_not_hsdm = np.mean(cost_not_hsdm_stacked, axis =0)

x_total_hsdm = np.matrix([])
N=len(x_hsdm)

fig, ax = plt.subplots(figsize=(6, 2.5), layout='constrained') 
N_iter = (average_residual_hsdm.shape[1]-1) * 20
colors = ["#e377c2", "#2ca02c", "#1f77b4"]
for i_exponent in range(N_exponents_hsdm):
    average_cost_hsdm = np.mean(cost_hsdm_stacked[tested_hsdm_exponents[i_exponent]], axis =0)
    ax.plot(np.arange(0, N_iter,20), np.ravel(average_cost_hsdm[:,:-1]), color=colors[i_exponent], label=tested_hsdm_exponents[i_exponent])
ax.loglog(np.arange(0, N_iter,20), np.ravel(average_cost_not_hsdm[:,:-1]), color="#ff7f0e", label="PPP")
# ax.fill_between(np.arange(0, N_iter,20), np.ravel(min_residual_hsdm[:,:-1]), np.ravel(max_residual_hsdm[:,:-1]), alpha=0.2)
# ax.fill_between(np.arange(0, N_iter,20), np.ravel(min_residual_not_hsdm[:,:-1]), np.ravel(max_residual_not_hsdm[:,:-1]), alpha=0.2)
plt.xlabel('Iteration')
plt.ylabel('Cost')
# plt.ylim((10**(-8), 10**(1)))
# plt.xlim((10**(1), 5*10**(4)))
plt.grid()
# plt.title('Residual')
plt.legend(loc="lower left", fontsize=12)
plt.savefig('Cost_exponents.pdf')  




# sigma_not_electric_hsdm=[]
# sigma_not_electric_not_hsdm=[]
# test = 0

# cost_advantage = []
# distance_from_ppp = []
# for test in range(N_random_tests):
#     x_total_hsdm = np.matrix([])
#     for i in range(N_agents):
#         x_total_hsdm=np.vstack((x_total_hsdm, x_hsdm[(i, test)])) if x_total_hsdm.size else x_hsdm[(i, test)]
#     x_total_not_hsdm = np.matrix([])
#     for i in range(N_agents):
#         x_total_not_hsdm=np.vstack((x_total_not_hsdm, x_not_hsdm[(i , test)])) if x_total_not_hsdm.size else x_not_hsdm[(i, test)]
#     distance_from_ppp.append(np.linalg.norm(x_total_hsdm-x_total_not_hsdm, inf))
#     if cost_not_hsdm[test]> 0.01:
#         cost_advantage.append( (cost_not_hsdm[test] - cost_hsdm[test])  / cost_not_hsdm[test])
#     else:
#         cost_advantage.append(0)
# fig, ax = plt.subplots(figsize=(5, 1.8), layout='constrained') 
# a= round(min(distance_from_ppp), 2)
# b= round(max(distance_from_ppp)+0.01, 2)
# step=(b-a)/10
# bins = np.arange(a, b+step, step)
# plt.hist(distance_from_ppp,rwidth=0.5, bins=bins)
# plt.xticks(bins)
# plt.xlabel("$\|x_{HSDM}-x_{PPP}\|_{\infty}$")
# plt.ylabel("Occurrencies")
# ax.xlim=([a,b+step])
# ax.set_xticks=bins
# ax.set_xticklabels=bins
# plt.show()
# plt.title("Distance between HSDM and PPP")

# print("The distance between HSDM and OSQP solution is, ", np.linalg.norm(x_total_hsdm - x_osqp, inf))
# print("The distance between PPP and OSQP solution is, ", np.linalg.norm(x_total_not_hsdm - x_osqp, inf))
# print("The distance between PPP and HSDM solution is, ", np.linalg.norm(x_total_not_hsdm - x_total_hsdm, inf))



# fig, ax = plt.subplots(figsize=(5, 1.8), layout='constrained') 
# cost_advantage_percentage=[ cost.item()*100 for cost in cost_advantage]
# a= round(min(cost_advantage_percentage))
# b= round(max(cost_advantage_percentage)+1)
# step=(b-a)/10
# bins = np.arange(0, b+step, step)
# plt.hist( cost_advantage_percentage, rwidth=0.5, bins=bins)
# plt.xlabel("Selection function value decrease ($\%$)")
# plt.ylabel("Occurrencies")
# ax.xlim=([a,b+step])
# # ax.set_xticks=bins
# plt.xticks(bins)
# # ax.set_xticklabels=bins
# plt.title("Advantage of HSDM")

fig, ax = plt.subplots(figsize=(6,5), layout='constrained') 
avg_improvement=np.zeros((N_problems,1))
for problem in range(N_problems):
    avg_improvement[problem] = np.median(cost_improvement_rel[problem, :])
sorted_idx = np.argsort(np.ravel(avg_improvement))
plt.boxplot(100*cost_improvement_rel[sorted_idx].T, sym="", whis =10000, vert=False, medianprops=dict(color='k'))
plt.barh([i+1 for i in range(N_problems)], np.ravel(100*avg_improvement[sorted_idx]), color="#1f77b4")
# ax.set_xticklabels=binsplt.title("Advantage of HSDM")

plt.yticks([])
plt.grid(axis='x')
plt.ylabel('GNE problem instance')
plt.xlabel(r'$ \frac{\phi(x_{\mathrm{PPP}})-\phi(x^{\star})}{\phi(x_{\mathrm{PPP}})} (\%)$ ')
plt.xlim((0, 200))
plt.savefig('Advantage.pdf') 

fig, ax = plt.subplots(figsize=(6,5), layout='constrained') 
avg_distance=np.zeros((N_problems,1))
for problem in range(N_problems):
    avg_distance[problem] = np.median(distance_optimizers[problem, :])
plt.boxplot(distance_optimizers[sorted_idx].T, sym="", whis =10000, vert=False, medianprops=dict(color='k'))
plt.barh([i+1 for i in range(N_problems)], np.ravel(avg_distance[sorted_idx]), color="#1f77b4")
# ax.set_xticklabels=binsplt.title("Advantage of HSDM")

plt.yticks([])
plt.grid(axis='x')
plt.ylabel('GNE problem instance')
plt.xlabel(r'$ \|x_{\mathrm{PPP}}-x^{\star}\|_{\infty}$ ')
plt.xlim((0, 5))
plt.savefig('Distance.pdf') 
plt.show(block=True)
