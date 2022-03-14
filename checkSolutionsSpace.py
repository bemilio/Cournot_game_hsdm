import numpy as np
import matplotlib.pyplot as plt
import osqp
from cmath import inf
from scipy import sparse
from scipy import optimize

def checkSolutionsSpace(agents_hsdm, shared_var):
    N=len(agents_hsdm)
    N_dec_var_i = agents_hsdm[0].N_dec_var
    N_local_constr = agents_hsdm[0].local_const.Aeq.shape[0]
    Q = np.matrix(np.zeros( (N*N_dec_var_i, N*N_dec_var_i) ))
    A_eq = np.matrix(np.zeros( (N*N_local_constr , N*N_dec_var_i) ))
    q = np.matrix([])
    b=np.matrix([])
    for i in range(len(agents_hsdm)):
        Q[i*N_dec_var_i: i*N_dec_var_i + N_dec_var_i,  i*N_dec_var_i:i*N_dec_var_i + N_dec_var_i ] = agents_hsdm[i].local_obj.Q.todense()
        q = np.vstack((q, agents_hsdm[i].local_obj.q)) if q.size else agents_hsdm[i].local_obj.q
        A_eq[i*N_local_constr: i*N_local_constr + N_local_constr,  i*N_dec_var_i:i*N_dec_var_i + N_dec_var_i ]  = agents_hsdm[i].local_const.Aeq.todense()
        b = np.vstack((b, agents_hsdm[i].local_const.beq )) if b.size else agents_hsdm[i].local_const.beq
        rank_A=np.linalg.matrix_rank(agents_hsdm[i].local_obj.Q.todense())
        rank_b=np.linalg.matrix_rank(np.block([agents_hsdm[i].local_obj.Q.todense(), agents_hsdm[i].local_obj.q]))
    Q_complete = Q +  np.kron( (np.matrix(np.ones((N, N))) - np.matrix(np.eye(N))) , shared_var.C.todense() )

    KKT_A=np.block( [ [ Q_complete, np.transpose(A_eq) ], [ A_eq, np.zeros( (N*N_local_constr, N*N_local_constr) ) ]   ] )
    KKT_b = np.vstack( (-q, b ) )

    rank_A=np.linalg.matrix_rank(KKT_A)
    rank_b=np.linalg.matrix_rank(np.block( [KKT_A, KKT_b] ))
    rank_Aeq=np.linalg.matrix_rank(A_eq)
    rank_Aeqb=np.linalg.matrix_rank(np.block( [A_eq, b] ))

    rank_Q=np.linalg.matrix_rank(Q_complete)
    rank_Qq=np.linalg.matrix_rank(np.block( [Q_complete, -q] ))

    # Q_single = np.block( [ [agents_hsdm[0].local_obj.Q.todense(), shared_var.C.todense() ] , [shared_var.C.todense(), agents_hsdm[1].local_obj.Q.todense() ]  ]   )
    # rank_Q_single=np.linalg.matrix_rank( Q_single )
    # rank_Qq_single=np.linalg.matrix_rank(np.block( [Q_single, np.vstack( (agents_hsdm[0].local_obj.q, agents_hsdm[1].local_obj.q) )  ] ))


    print("The size of the matrix is ", KKT_A.shape[0], "The rank of the coeff. matrix is, ", rank_A, " The rank of the ext. matrix is, ", rank_b)
    print("The size of the matrix Q is ", Q_complete.shape[0], "The rank of the coeff. matrix is, ", rank_Q, " The rank of the ext. matrix is, ", rank_Qq)
    # print("The size of the matrix Q for a single agent is ", Q_single.shape[0], "The rank of the coeff. matrix is, ", rank_Q_single, " The rank of the ext. matrix is, ", rank_Qq_single)
    print("The size of the matrix Aeq is ", A_eq.shape[0], "The rank of the coeff. matrix is, ", rank_Aeq, " The rank of the ext. matrix is, ", rank_Aeqb)


    # print("Stopping...")


def checkSolutionsSpace_QP(agents_hsdm, shared_var):
    N=len(agents_hsdm)
    n_dec_var_tot = 0
    n_loc_constr_eq_tot = 0
    n_loc_constr_tot = 0
    for agent in agents_hsdm:
        n_dec_var_tot = n_dec_var_tot + agent.N_dec_var
        n_loc_constr_eq_tot = n_loc_constr_eq_tot + agent.local_const.Aeq.shape[0]
        n_loc_constr_tot = n_loc_constr_tot + agent.local_const.A.shape[0]
    Q = np.matrix(np.zeros( (n_dec_var_tot, n_dec_var_tot) ))
    A_eq = np.matrix(np.zeros( (n_loc_constr_eq_tot ,n_dec_var_tot) ))
    A = np.matrix(np.zeros( (n_loc_constr_tot, n_dec_var_tot) ))
    q = np.matrix([])
    b_eq=np.matrix([])
    b=np.matrix([])
    index_Q = 0
    index_A = 0
    index_Aeq = 0
    for i in range(len(agents_hsdm)):
        N_dec_var_i= agents_hsdm[i].N_dec_var
        N_local_constr_eq = agents_hsdm[i].local_const.Aeq.shape[0]
        N_local_constr = agents_hsdm[i].local_const.A.shape[0]
        Q[index_Q: index_Q + N_dec_var_i,  index_Q: index_Q + N_dec_var_i] = agents_hsdm[i].local_obj.Q.todense()
        q = np.vstack((q, agents_hsdm[i].local_obj.q)) if q.size else agents_hsdm[i].local_obj.q
        A_eq[index_Aeq: index_Aeq + N_local_constr_eq, index_Q: index_Q + N_dec_var_i ]  = agents_hsdm[i].local_const.Aeq.todense()
        A[index_A: index_A + N_local_constr, index_Q: index_Q + N_dec_var_i]  = agents_hsdm[i].local_const.A.todense()
        b_eq = np.vstack((b_eq, agents_hsdm[i].local_const.beq )) if b_eq.size else agents_hsdm[i].local_const.beq
        b = np.vstack((b, agents_hsdm[i].local_const.b )) if b.size else agents_hsdm[i].local_const.b
        index_Q = index_Q + N_dec_var_i
        index_A = index_A + N_local_constr
        index_Aeq = index_Aeq + N_local_constr_eq
    C_complete = np.matrix(np.zeros((n_dec_var_tot, n_dec_var_tot)))
    index_row = 0
    for i in range(len(agents_hsdm)):
        N_dec_var_i = agents_hsdm[i].N_dec_var
        index_col = 0
        for j in range(len(agents_hsdm)):
            N_dec_var_j = agents_hsdm[j].N_dec_var
            if i!=j:
                C_complete[index_row: index_row + N_dec_var_i, index_col: index_col + N_dec_var_j] = (agents_hsdm[i].loc_to_sigma.T * shared_var.C *  agents_hsdm[j].loc_to_sigma).todense()
            index_col = index_col + N_dec_var_j
        index_row = index_row + N_dec_var_i

    Q_complete = sparse.csc_matrix(Q + C_complete)
    x_opt=[]
    A_all= sparse.csc_matrix(np.vstack((A, A_eq)))
    u=np.vstack((b, b_eq))
    l=np.vstack((-np.inf * np.ones((b.shape[0],1)), b_eq ))
    for i in range(100):
        solved = False
        while(not solved):
            m = osqp.OSQP()
            x0 = 10*np.random.rand(Q_complete.shape[0], 1)
            y0 = 10*np.random.rand(A_all.shape[0], 1)
            m.setup(P=Q_complete, q=q, A=A_all, l=l, u=u, verbose=False, max_iter=3000, eps_abs=10**(-8), eps_rel=10**(-8))
            m.warm_start(x=x0, y=y0)
            results = m.solve()  
            if results.info.status != 'solved':
                print("Not solved...")
            else:
                x_opt.append(np.transpose(np.matrix(results.x)))
                solved = True
    max_dist = 0
    for i in range(len(x_opt)):
        for j in range(len(x_opt)):
            if i!=j:
                max_dist = max(max_dist, np.linalg.norm(x_opt[i] -x_opt[j], inf))
    print("maximum distance between solutions: ,", max_dist)
    return(x_opt[0])


