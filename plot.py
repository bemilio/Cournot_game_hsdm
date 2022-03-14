from functools import total_ordering
from re import X
import matplotlib as mpl
from cmath import inf

mpl.interactive(True)
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
import networkx as nx
import numpy as np
import pickle
from scipy import sparse

check_test = False

f=open('saved_sol.pkl', 'rb')
# f = open('/Users/ebenenati/Repositories/Results_MDP/local_results/80_agents_5000It/80_agents_5000It.pkl', 'rb')
x_hsdm, x_not_hsdm, residual_hsdm, residual_not_hsdm, sigma_HSDM, \
    sigma_not_hsdm, cost_hsdm, cost_not_hsdm, T_horiz, N_random_tests, N_agents, x_osqp = pickle.load(f)
f.close()

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


x_total_hsdm = np.matrix([])
N=len(x_hsdm)

fig, ax = plt.subplots(figsize=(5, 1.8), layout='constrained') 
N_iter = (len(residual_hsdm)-1) * 20
ax.loglog(np.arange(0, N_iter,20), residual_hsdm[:-1], label="HSDM+PPP")
ax.loglog(np.arange(0, N_iter,20),residual_not_hsdm[:-1], label="PPP")
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.ylim((10**(-6), 10**(1)))
# plt.title('Residual')
plt.legend()
plt.savefig('Residual.pdf')  

sigma_not_electric_hsdm=[]
sigma_not_electric_not_hsdm=[]
test = 0

cost_advantage = []
distance_from_ppp = []
for test in range(N_random_tests):
    x_total_hsdm = np.matrix([])
    for i in range(N_agents):
        x_total_hsdm=np.vstack((x_total_hsdm, x_hsdm[(i, test)])) if x_total_hsdm.size else x_hsdm[(i, test)]
    x_total_not_hsdm = np.matrix([])
    for i in range(N_agents):
        x_total_not_hsdm=np.vstack((x_total_not_hsdm, x_not_hsdm[(i , test)])) if x_total_not_hsdm.size else x_not_hsdm[(i, test)]
    distance_from_ppp.append(np.linalg.norm(x_total_hsdm-x_total_not_hsdm, inf))
    if cost_not_hsdm[test]> 0.01:
        cost_advantage.append( (cost_not_hsdm[test] - cost_hsdm[test])  / cost_not_hsdm[test])
    else:
        cost_advantage.append(0)
fig, ax = plt.subplots(figsize=(5, 1.8), layout='constrained') 
a= round(min(distance_from_ppp), 2)
b= round(max(distance_from_ppp)+0.01, 2)
step=(b-a)/10
bins = np.arange(a, b+step, step)
plt.hist(distance_from_ppp,rwidth=0.5, bins=bins)
plt.xticks(bins)
plt.xlabel("$\|x_{HSDM}-x_{PPP}\|_{\infty}$")
plt.ylabel("Occurrencies")
ax.xlim=([a,b+step])
ax.set_xticks=bins
ax.set_xticklabels=bins
plt.show()
plt.title("Distance between HSDM and PPP")

print("The distance between HSDM and OSQP solution is, ", np.linalg.norm(x_total_hsdm - x_osqp, inf))
print("The distance between PPP and OSQP solution is, ", np.linalg.norm(x_total_not_hsdm - x_osqp, inf))
print("The distance between PPP and HSDM solution is, ", np.linalg.norm(x_total_not_hsdm - x_total_hsdm, inf))



fig, ax = plt.subplots(figsize=(5, 1.8), layout='constrained') 
cost_advantage_percentage=[ cost.item()*100 for cost in cost_advantage]
a= round(min(cost_advantage_percentage))
b= round(max(cost_advantage_percentage)+1)
step=(b-a)/10
bins = np.arange(0, b+step, step)
plt.hist( cost_advantage_percentage, rwidth=0.5, bins=bins)
plt.xlabel("Selection function value decrease ($\%$)")
plt.ylabel("Occurrencies")
ax.xlim=([a,b+step])
# ax.set_xticks=bins
plt.xticks(bins)
# ax.set_xticklabels=bins
plt.title("Advantage of HSDM")

plt.show(block=True)
