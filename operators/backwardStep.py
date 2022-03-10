import numpy as np
import osqp
from scipy import sparse

def backwardStep(Q, q, A_all, l, u, x0, alpha):
    # Proximal point operator for a quadratic cost and linear set
    # min 1/2 x'Qx + x'q + alpha/2|| x-x0 ||^2 ; x\in Ax<=b

    for attempts in range(5): # sometimes OSQP gets stuck, but if you reset the optimization it solves it. - yes, really
        m = osqp.OSQP()
        P = (Q + alpha*sparse.identity(Q.shape[0], format='csc'))
        # A_all = sparse.csc_matrix(np.vstack((A, Aeq)))
        q2 = q-alpha*x0
        #m.setup(P=P, q=q2, A=A_all, l=l, u=u, verbose=True)
        m.setup(P=P, q=q2, A=A_all, l=l, u=u, verbose=False, warm_start=False, max_iter=3000, eps_abs=10**(-6), eps_rel=10**(-6))
        results = m.solve()   
        if not (results.info.status == 'maximum iterations reached'):
            break
        else:
            if attempts<2:
                print("Warning: OSQP did not solve, trying attempt", attempts + 1)
            else:
                print("Warning: OSQP did not solve, giving up")

    return(np.transpose(np.matrix(results.x)), results.info.status)

