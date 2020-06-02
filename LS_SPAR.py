from scipy.linalg import *
from scipy.sparse import *
from numpy import *
from cmath import *
import math
from queue import *
from statistics import *
from guppy import hpy
import time 
import os
import sys
import bisect
#import cvxpy as cp
    
def columnsum(M):
    r = array([0]*len(M[0]))
    for i in range(len(M)):
        r = r + M[i]
    return r

def columnvariance(M):
    [m,n] = shape(M)
    v = [0]*n
    for i in range(n):
        v[i] = variance(M[:,i])
    s = array([0]*n)
    s = [v[j] ** 0.5 for j in range(n)]
    return [s,v]


def columnvariance_mk2(M):
    [m,n] = shape(M)
    s = [0]*n
    for i in range(n):
        s[i] = statistics.stdev(M[:,i])
    v = array([0]*n)
    v = [s[j] ** 2 for j in range(n)]
    return [s,v]
    

def columnmin(M):
    values = [0]*len(M[0])
    indexes = [0]*len(M[0])
    for j in range(len(M[0])):
        val = inf
        index = 0
        for i in range(len(M)):
            if M[i][j] < val:
                val = M[i][j]
                index = i
        values[j] = val
        indexes[j] = index
    
    return [values,indexes]
        
def f_LI(A,b,x):

    return norm(A.dot(x)-b) ** 2;

def f_QU(A,c,x):
    
    return sum(((A.dot(x)) ** 2 - c) ** 2)

def g_LI(A,b,x,S):
    """
     This function finds the index of the variable (among the indices set S) 
     that causes the greatest decrease in the objective function
     value 
     ||Ax-b||^2
     The output is the three-dimensional vector out comprising the new value of the index,
     the new objective function value, and the index causing the greatest
     decrease
    
     Based on the paper
     Amir Beck and Yonina Eldar, "Sparsity Constrained Nonlinear Optimization: Optimality Conditions and Algorithms"
     -----------------------------------------------------------------------
     Copyright (2012): Amir Beck and Yonina Eldar
     
     Distributed under the terms of 
     the GNU General Public License 2.0.
     
     Permission to use, copy, modify, and distribute this software for
     any purpose without fee is hereby granted, provided that this entire
     notice is included in all copies of any software which is or includes
     a copy or modification of this software and in all copies of the
     supporting documentation for such software.
     This software is being provided "as is", without any express or
     implied warranty.  In particular, the authors do not make any
     representation or warranty of any kind concerning the merchantability
     of this software or its fitness for any particular purpose."
    -------------------------------------------------------------
     INPUT
    ===================================================
     A ............ the matrix A associated with the objective function
     b ............ the vector b associated with the objective fucntion
     x ............ current point
     S ............ indices set (from which an index should be chosen)
    
     OUTPUT
     ====================================================
     out ........... a 3-dimensional vector containing out(1) - the new value
                     of the chosen variable, out(2) - the new objective
                     function value and out(3) - the index of the chosen
                     variable.
    """

    g = transpose(A).dot((A.dot(x)-b))
    norm_square_vector = transpose([columnsum(A ** 2)])

    g_S = g[S]

    norm_square_vector_S = norm_square_vector[S]
    
    val_all = (g_S ** 2)/norm_square_vector_S
    
    stam = max(val_all) 
    ind = list(val_all).index(stam)
    
    if type(S) == ndarray or type(S) == list or type(S) == range:
        val = x[S[ind]] - g[S[ind]]/norm_square_vector[S[ind]]
        fun_val = norm(A.dot(x) - b - transpose([(g[S[ind]]/norm_square_vector[S[ind]])*(transpose(A[:,S[ind]]))])) ** 2
        out = zeros([3,1])
        out[0] = val
        out[1] = fun_val
        out[2] = S[ind]
    else:
        val = x[S] - g[S]/norm_square_vector[S]
        fun_val = norm(A.dot(x) - b - transpose([(g[S]/norm_square_vector[S])*(transpose(A[:,S]))])) ** 2
        out = zeros([3,1])
        out[0] = val
        out[1] = fun_val
        out[2] = S

    


    return out

def g_QU(A,c,x,Q):
    """
     This function finds the index of the variable (among the indices set Q) 
     that causes the greatest decrease in the objective function
     \sum_{i=1}^m ((A(i,:)*x)^2-c)^2
     The output is the three-dimensional vector out comprising the new value of the index,
     the new objective function value, and the index causing the greatest
     decrease
    
     Based on the paper
     Amir Beck and Yonina Eldar, "Sparsity Constrained Nonlinear Optimization: Optimality Conditions and Algorithms"
     -----------------------------------------------------------------------
     Copyright (2012): Amir Beck and Yonina Eldar
     
     Distributed under the terms of 
     the GNU General Public License 2.0.
     
     Permission to use, copy, modify, and distribute this software for
     any purpose without fee is hereby granted, provided that this entire
     notice is included in all copies of any software which is or includes
     a copy or modification of this software and in all copies of the
     supporting documentation for such software.
     This software is being provided "as is", without any express or
     implied warranty.  In particular, the authors do not make any
     representation or warranty of any kind concerning the merchantability
     of this software or its fitness for any particular purpose."
    -------------------------------------------------------------
     INPUT
    ===================================================
     A ............ the matrix A associated with the objective function
     c ............ the vector c associated with the objective fucntion
     x ............ current point
     S ............ indices set (from which an index should be chosen)
    
     OUTPUT
     ====================================================
     out ........... a 3-dimensional vector containing out(1) - the new value
                     of the chosen variable, out(2) - the new objective
                     function value and out(3) - the index of the chosen
                     variable.
    """
    
    s = shape(A)
    n = s[1]
    A2 = A ** 2
    A3 = A ** 3
    A4 = A ** 4
    w1 = transpose([columnsum(A4)])
    w2 = transpose(A2).dot(c)
    u1 = A.dot(x)
    u2 = u1 ** 2;
    u4 = u2 ** 2;
    v5 = w1 
    v4 = 4 * transpose(A3).dot(u1)
    v3 = 6 * transpose(A2).dot(u2)-2*w2;
    v2 = 4 * transpose(A).dot(((u2-c)*u1))
    v1 = sum((u2-c) ** 2)*ones([n,1])

    V = hstack((v1,v2,v3,v4,v5))
    
    
    [out2, fun] = solve_minimum_quartic(V)
    

    fun_limit = fun[Q]
    
    fval = min(fun[Q])
    resh_funq = reshape(fun[Q],[1,len(fun[Q])])
    d1_funq = resh_funq[0]
    ind = -1
    for i in range(len(d1_funq)):
        if d1_funq[i] == fval:
            ind = i
    out = zeros([3,1])
    if type(Q) == ndarray or type(Q) == list or type(Q) == range:
        out[0] = x[Q[ind]]+out2[Q[ind]]
        out[1] = fval
        out[2] = Q[ind]
    else:
        out[0] = x[Q]+out2[Q]
        out[1] = fval
        out[2] = Q
    return out  

def gradient_LI(A,c,x):
    return 2*transpose(A).dot(A.dot(x)-c)

def gradient_QU(A,c,x):
    
    return 4 * transpose(A).dot(((A.dot(x)) * ((A.dot(x)) ** 2 - c )))

def solve_cubic(a):
    

    A0 = a[:,0]/a[:,3]
    A1 = a[:,1]/a[:,3]
    A2 = a[:,2]/a[:,3]
    
    Q = 1/9 * (3 * A1 - A2 ** 2)
    R = 1/54 * (9 * A1 * A2 - 27 * A0 - 2 * A2 ** 3)
    D = Q ** 3 + R ** 2
    sqrtd = array([sqrt(i) for i in D])
    S = ( R + sqrtd ) ** (1/3)
    T = -1 * Q/S
    
    z1 = -1/3 * A2 + (S+T)
    z2 = -1/3 * A2 - 1/2 * (S+T) + 1/2 * sqrt(-3) * (S-T)
    z3 = -1/3 * A2 - 1/2 * (S+T) - 1/2 * sqrt(-3) * (S-T)
    
 
    z1 = reshape(z1,[len(z1),1])
    z2 = reshape(z1,[len(z2),1])
    z3 = reshape(z1,[len(z3),1])
    out = hstack((z1,z2,z3))
    return out

def solve_minimum_quartic(a):
    """
    this function minimizes  quartic  funcctions of the form
    a(i,1)+a(i,2)*x+a(i,3)*x^2+a(i,4)*x^3+a(i,5)*x^4
    """
    s = shape(a)
    m = s[0]
    n = s[1]


    out = solve_cubic(hstack((reshape(a[:,1],[m,1]),reshape(2*a[:,2],[m,1]),reshape(3*a[:,3],[m,1])\
         ,reshape(4*a[:,4],[m,1]))))
    ro = real(out)
    
    v = kron(reshape(a[:,0],[m,1]),ones([1,3]))+kron(reshape(a[:,1],[m,1]),ones([1,3]))*ro+\
    kron(reshape(a[:,2],[m,1]),ones([1,3])) * ro ** 2  + kron(reshape(a[:,3],[m,1]),ones([1,3])) * ro ** 3 + \
    kron(reshape(a[:,4],[m,1]),ones([1,3])) * ro ** 4
    

    [fun , ind] = columnmin(transpose(v))
    fun = transpose([fun])
    S = coo_matrix((ones(m),(arange(0,m),ind)),shape = (m,3))
    out = transpose([columnsum(transpose((S.toarray()*ro)))])
            
    """
    fun=full(sum((S.*)')');
    out=ro(ind);  
    """
    
    return   [out,fun]

def IHT(f,g,s,L,x0,N):
    """
    
     This function employs N iterations of the iterative hard-thresholding method for solving the sparsity-constrained 
     problem min{f(x):||x||_0 <=s}
    
     Based on the paper
     Amir Beck and Yonina Eldar, "Sparsity Constrained Nonlinear Optimization: Optimality Conditions and Algorithms"
     -----------------------------------------------------------------------
     Copyright (2012): Amir Beck and Yonina Eldar
     
     Distributed under the terms of 
     the GNU General Public License 2.0.
     
     Permission to use, copy, modify, and distribute this software for
     any purpose without fee is hereby granted, provided that this entire
     notice is included in all copies of any software which is or includes
     a copy or modification of this software and in all copies of the
     supporting documentation for such software.
     This software is being provided "as is", without any express or
     implied warranty.  In particular, the authors do not make any
     representation or warranty of any kind concerning the merchantability
     of this software or its fitness for any particular purpose."
    -------------------------------------------------------------
     INPUT
    ===================================================
     f ............ the objective function
     g ............ the gradient of the objective function
     s ............ the sparsity level 
     L ............ upper bound on the Lipschitz constant
     x0 ........... initial vector 
     N ............ number of iterations 
    
     OUTPUT
     ====================================================
     x ............. the output of the IHT method
     fval .......... objective function value of the obtained vector
    
    """
    
    """
    [xiht_coef,funval_iht] = IHT(lambda x: norm(A.dot(x)-b) ** 2, lambda y :  2 * transpose(A).dot(A.dot(y)-b),s\
        
        , 2 * max(eigvals(transpose(A).dot(A)))+ 0.1, x00 , 1000)
    
    A little note for how you should call this function

    """
    
    n = len(x0)
    x = copy(x0)
    for i in range(0,N):
        x = x-1/L*g(x)
        isort = sorted(range(len(x)), key=lambda k: abs(x[k]))
        xsort = sorted(abs(x))
        xnew = zeros([n,1])
        xnew[isort[n-s:n]] = x[isort[n-s:n]]
        x = xnew
        print('iter = {:5d} value = {:5.10f}'.format(i,f(x)))
        
    fx = f(x)
    return [x,fx]

def  greedy_sparse_simplex(f,g,s,N,x0,A,b):
    """
     This function employes the greedy sparse simplex method on the problem
     (P) min \{ f(x): ||x||_0 <=s\}
     
     Based on the paper
     Amir Beck and Yonina Eldar, "Sparsity Constrained Nonlinear Optimization: Optimality Conditions and Algorithms"
     -----------------------------------------------------------------------
     Copyright (2012): Amir Beck and Yonina Eldar
     
     Distributed under the terms of 
     the GNU General Public License 2.0.
     
     Permission to use, copy, modify, and distribute this software for
     any purpose without fee is hereby granted, provided that this entire
     notice is included in all copies of any software which is or includes
     a copy or modification of this software and in all copies of the
     supporting documentation for such software.
     This software is being provided "as is", without any express or
     implied warranty.  In particular, the authors do not make any
     representation or warranty of any kind concerning the merchantability
     of this software or its fitness for any particular purpose."
    -------------------------------------------------------------
     INPUT
    ===================================================
     f ............ the objective function (which is a function of the
                    decision variable vector x)
     g ............ a function performing one-dimensional optimization of 
                    the function f. Its input consists of the pair (x,S) where
                    x is the input decision variables vector and S is a set of
                    indices on which the optimization is performed
     s ............ sparsity level
     N ............ maximum number of iterations
     x0 ........... initial vector
    
     OUTPUT
     ====================================================
     X ............ The sequence of iterates generated by the method
     fun_val ...... The obtained function value (of the last iterate)
    
    """
    
    """
    
    [Xgreed,funval_greed] = greedy_sparse_simplex(f_LI,g_LI,s,20000,x00,A,b)
    #xgreed_coef = Xgreed[:,[-1]]
    
    A little note for how you should call this function
    """
    n = len(x0)
    x = copy(x0)    
    
    X = array([[]])
    fold = Inf
    fun_val = -Inf
    iter_stuck = 0
    for iteration in range(0,N):
        if (abs(fold - fun_val) < 1e-8):
            iter_stuck = iter_stuck +1
            
        if (iter_stuck == 5):
            break
        
        fold = fun_val
    
        d = dia_matrix(x).nnz
    
        if (d > s):
            ok = 1
            x[s:n] = 0
        if shape(X)[1] != 0:
            X = hstack((X,x))
        else:
            X = x
        if (d < s):
            ok = 0
            min_funval = Inf
            out = g(A,b,x,range(0,n))
            val = out[0]
            fun_val = out[1]
            ind = out[2]
            if (fun_val < min_funval):
                min_index = ind
                min_funval = fun_val
                min_val = val
            min_index = int(min_index)  
            x[min_index] = min_val
            
        if (d == s):
            I1 = x.nonzero()
            I1 = I1[0]
            min_funval = Inf
            for i in range(0,s):
                xtilde = copy(x)
                xtilde[I1[i]] = 0
                out = g(A,b,xtilde,range(0,n))
                val = out[0]
                fun_val = out[1]
                ind = out[2]
                
                if (fun_val < min_funval):
                    min_index_out = I1[i]
                    min_index_in = ind
                    min_funval = fun_val
                    min_val = val
                    
            ok = 0
            min_index_in = int(min_index_in)
            if x[min_index_in] == 0:
                ok = 1
                
            x[min_index_out] = 0
            
            x[min_index_in] = min_val
        fun_val = f(A,b,x)
        print('iter = {:3d}  fun_val = {:5.5f}   change = {:d}'.format(iteration,fun_val,ok))
    fun_val = f(A,b,x)
    
    return [X,fun_val]


def  partial_sparse_simplex(f,f_grad,g,s,N,x0,A,c):
    """
    
    #[Xpartial,funval_partial] = partial_sparse_simplex(f_LI,gradient_LI,g_LI,s,1000,x00,A,b)
    #xpartial_coef = Xpartial[:,[-1]]  
    
    A little note for how you should call this function
    
    """

    n = len(x0)
    x = copy(x0)
    
    X = array([[]])
    fold = Inf
    fun_val = -Inf
    iter_stuck = 0
    
    for iteration in range(0,N):
        if (abs(fold-fun_val) < 1e-8):
            iter_stuck = iter_stuck+1
            
        if (iter_stuck == 5):
            break
        
        fold = fun_val
    
        d = dia_matrix(x).nnz
    
        if (d > s):
            P = random.permutation(n)
            x[P[0:n-s]] = 0
            ok = 0
            
        if shape(X)[1] != 0:
            X = hstack((X,x))
        else:
            X = copy(x)
        if (d < s):
            min_funval = Inf
            out = g(A,c,x,range(0,n))
            min_val = out[0]
            min_funval = out[1]
            min_index = out[2]
            x[min_index] = min_val
            ok = 0
        if (d == s):
            ok = 0
            I1 = x.nonzero()
            I1 = I1[0]
            I0 = array(list(set(range(0,n)).difference(set(I1))))
          
            out = g(A,c,x,I1)
            min_val = out[0]
            min_funval = out[1]
            min_index = out[2]
    
    
            stam = min(abs(x[I1]))
            resh_xI1 = reshape(x[I1],[1,len(x[I1])])
            d1_xI1 = resh_xI1[0]
            ind = -1
            for i in range(len(d1_xI1)):
                if d1_xI1[i] == stam:
                    ind = i
                    
            i_index = I1[ind]
            gradient = f_grad(A,c,x)
            stam = max(abs(gradient[I0]))
            ind = list(abs(gradient[I0])).index(stam)
            
            j_index = I0[ind]
            xtilde = copy(x)
            xtilde[i_index] = 0
            
            out = g(A,c,xtilde,j_index)
            val = out[0]
            fun_val = out[1]
            min_index = int(min_index)
            if (fun_val < min_funval+1e-8):
                ok = 1 
                x[i_index] = 0
                x[j_index] = val
            else:
                x[min_index] = min_val
                
        fun_val = f(A,c,x)
        print('iter= {:3d}  fun_val = {:5.5f} change = {:d}'.format(iteration,fun_val,ok))
    fun_val = f(A,c,x)
    
    return [X,fun_val]


class LSSPAR:
    def __init__(self,A,b,s):
        """ 
        initilization of parameters 
        -------------------------------------------------------------------------------------
        A       = m x n matrix consisting of m observations of n independent variables
        b       = m x 1 column vector consisting of m observed dependent variable
        s       = sparsity level
        
        Solver parameters
        -------------------------------------------------------------------------------------
        P       = Indexes of possible choices for independent variable
        C       = Chosen set of independent variables
        
        
        If you get a pythonic error, it will probably because of how unfriendly is python with usage of 
        vectors, like lists cannot be sliced with indeces but arrays can be done, hstack converts everything
        to floating points, registering a vector and randomly trying to make it 2 dimensional takes too much
        unnecessary work, some functions do inverse of hstack, randomly convert numbers to integers
        
        Please contact selim.aktas@ug.bilkent.edu.tr for any bugs, errors and recommendations.
        """
        
        for i in range(len(A)):
            for j in range(len(A[0])):
                if math.isnan(A[i,j]) or abs(A[i,j]) == Inf:
                    print("Matrix A has NAN or Inf values, SVD will not converge")
                    break
        for i in range(len(A)):
            for j in range(len(A[0])):
                if type(A[i,j]) != float64:
                    print("Matrix A should be registered as float64, otherwise computations will be wrong\
for example; Variance will be negative")
            
        if shape(shape(b)) == (1,):
            print("you did not register the vector b as a vector of size m x 1, you are making a pythonic\
                  error, please reshape the vector b")
        elif shape(A)[0] != shape(b)[0]:
            print("Dimensions of A and b must match, A is",shape(A)[0],"x",shape(A)[1],"matrix", \
                  "b is",shape(b)[0],"x 1 vector, make sure" ,shape(A)[0],"=",shape(b)[0])
        elif shape(A)[0] <= shape(A)[1]:
            print("The linear system is supposed to be overdetermined, given matrix A of size m x n,",\
                  "the condition m > n should be satisfied") 
        elif shape(b)[1] != 1:
            print("input argument b should be a vector of size m x 1, where m is the row size of the \
                  A")
        elif type(A) != ndarray:
            print("A should be a numpy ndarray")
        elif type(s) != int:
            print("s should be an integer")
        else:
            self.A = A
            self.b = b
            self.bv = b[:,0]
            self.s = s
            """ registering the matrix A independent variable values, vector b dependent variable values 
            and s sparsity level """
            self.m = shape(A)[0]
            self.n = shape(A)[1]
            """ saving the shape of matrix A explicitly not to repeat use of shape() function """
            self.best_feasible = Inf
            """ initializing the best feasible point to Infinity """
            self.solcoef = None
            """ initializing the solution coefficients, they are stored in the order given by solset """
            self.solset = None
            """ initializing the solution set, they are saved by their indexes """
            self.node = 0
            """ initializing the number of end nodes or equivalently number of branching done """
            self.check = 0
            """ initializing number of nodes visited """
            self.means = columnsum(A)/self.m
            """ storing the mean values of each independent variable which will be used for 
            selection criteria in solver algortihms"""
            sterror,variances = columnvariance(A)
            self.sterror = array(sterror)
            self.variances = array(variances)
            """ storing the standard deviation and variance of each independent variable which will
            be used for selection criteria in solver algortihms"""
            self.rem_qsize = 0
            """ initializing remaining queue size after the algorithm finishes """
            self.out = 1
            """ initializing the output choice of algorithm """
            self.heur = False
            """ initializing the heuristic choice"""
            self.iter = 1000 
            """ initializing the heuristic iterations"""
            self.initial_point = ones([self.n,1])
            """ initializing the initial point of search for heuristic algorithms to be all ones """
            self.SST = variance(b[:,0])*(self.m-1)
            self.original_stdout = sys.stdout
            """  saving original stdout, after manipulating it for changing values of out, we should be
            able to recover it """
            self.enumerate = "m-st-lsc"
            """ initializing the enumeartion to (mean+standard deviation)*lstsq coefficient"""
            self.search = "best"
            """ initializing the search type to best first search """
            self.solver = "qr"
            """ initializing the least squares solver to qr"""
            self.lapack_driver = "gelsy"
            """ overwriting the default lstsq lapack driver """
            self.ill = False
            if linalg.matrix_rank(A) < self.n:
                print("Matrix A, either has linearly dependent columns or seriously ill-conditioned\
you can proceed and solve the problem but, accuracy and precision of the solution\
is not guaranteed")
                self.ill = True
            """ A = Q*R, """
            q,r = qr(A)
            self.q = q
            self.r = r
            self.rtilde = r[0:self.n,:]
            self.qb = self.q.T.dot(self.b)
            self.qbt = self.qb[0:self.n,[0]]
            self.qbr = self.qb[self.n:,[0]]
            self.permanentresidual = norm(self.qbr) ** 2
            self.cov = A.T.dot(A)
            self.tablelookup = {}
            self.tablelookupqri = {}
            """ lookup tables for solving all subset problem """
            self.many = 4
            """ initializing the number of solutions for multiple subset problem """
            self.residual_squared = []
            self.indexes = []
            self.coefficients = []
            """ initializing the arrays of solution parameters """
            
            """ the following parameters are only used by qp_mip1 which is not part of LSSPAR"""
            #self.bigm = max(abs(lstsq(A,b)[0]))*4  
            #self.verbose = False
            #self.solver2 = "MOSEK"
 
    """
    This function was only used for comparison. It is not part of LSSPAR, also it requires CVXPY and a solver. 
    def qp_mip1(self,ind,ce = []):
        
        A_i = vstack((eye(self.n),-1*eye(self.n)))
        A_iy = vstack((-1*self.bigm*eye(self.n),-1*self.bigm*eye(self.n)))
        A_iy2 = ones([1,self.n])
        
        
        b_i = zeros([2*self.n,1])
        #b_i2 = vstack((self.s,-1*self.s))
        b_i2 = self.s
        
        x = cp.Variable((self.n,1))
        y = cp.Variable((self.n,1), integer = True) 
        
        cost = cp.Minimize(cp.sum_squares(self.A @ x - self.b))
        if ce == []:
            constraints = [ A_i @ x + A_iy @ y <= b_i, A_iy2 @ y <= b_i2, eye(self.n) @ y <= ones([self.n,1]) ]
        else:
            k = zeros([self.n,1])
            k[ce] = 1
            CE = vstack((diag(k),-1*diag(k)))
            CEb = vstack((k,-1*k))
            constraints = [ A_i @ x + A_iy @ y <= b_i, A_iy2 @ y <= b_i2, eye(self.n) @ y <= ones([self.n,1]), CE @ y <= CEb ]
        
        prob = cp.Problem(cost,constraints)
        prob.solve(solver = self.solver2,verbose = self.verbose)
        
        self.residual_squared.append(prob.value)
        ind = where(abs(x.value[:self.n,0]) > 1e-6)[0]
        self.indexes.append(ind)
        self.coefficients.append(x.value[ind,0])
        return [x.value[:self.n,0],prob.value,y.value[:self.n,0]]
    """
    
    def qr_lstsq(self,ind):
        
        l = len(ind) 
        t = max(ind)+1
        """ since R is upper triangular, rows beyond t are 0 """
        qs,rs = qr(self.rtilde[:t,ind])
        rhs = transpose(qs).dot(self.qb[:t,[0]])
        x = solve_triangular(rs[:l,:l],rhs[:l,0])
        res = norm(rhs[l:]) ** 2 + norm(self.qb[t:]) ** 2
        """ using QR to solve a least squares problem """
        return [x,res]
    

    def qr_lstsql(self,ind):
        check = str(ind)
        if check in self.tablelookup:
            return self.tablelookup[check]
        """ table look up for all subsets problem """
        l = len(ind)
        t = max(ind)+1
        """ since R is upper triangular, rows beyond t are 0 """
        qs,rs = qr(self.rtilde[:t,ind])
        rhs = transpose(qs).dot(self.qb[:t,[0]])
        x = solve_triangular(rs[:l,:l],rhs[:l,0])
        res = norm(rhs[l:]) ** 2 +  norm(self.qb[t:]) ** 2
        """ using QR to solve a least squares problem """
        self.tablelookup[check] = [x,res]
        """ registering the solution to the table """
        return [x,res]

    
    def lstsq_qrl(self,ind):
        check = str(ind)
        if check in self.tablelookup:
            return self.tablelookup[check]
        """ table look up for all subsets problem """
        l = len(ind)
        t = max(ind)+1
        """ since R is upper triangular, rows beyond t are 0 """
        sol =  lstsq(self.rtilde[:t,ind],self.qb[:t],lapack_driver = self.lapack_driver)
        """ gelsy routine proved to be faster in our experience """
        
        x = sol[0][:,0]
        res = norm(self.rtilde[:t,ind].dot(x)-self.qb[:t,0])**2 +  norm(self.qb[t:]) ** 2
        self.tablelookup[check] = [x,res]
        """ registering the solution """
        return [x,res]
    
    def lstsq_qr(self,ind):
        l = len(ind)
        t = max(ind)+1
        """ since R is upper triangular, rows beyond t are 0 """
        sol =  lstsq(self.rtilde[:t,ind],self.qb[:t],lapack_driver = self.lapack_driver)
        x = sol[0][:,0]
        res = norm(self.rtilde[:t,ind].dot(x)-self.qb[:t,0])**2 +  norm(self.qb[t:]) ** 2
        """ using QR to solve a least squares problem """
        return [x,res]
    
    def lstsq(self,ind):
        sol = lstsq(self.A[:,ind],self.bv,lapack_driver = self.lapack_driver)
        x = sol[0]
        res = sol[1]
        """ pure numpy lstsq to solve  least squares problem """
        return [x,res]
    
    def lstsqi(self,ind):
        sol = lstsq(self.A[:,ind],self.bv,lapack_driver = self.lapack_driver)
        x = sol[0]
        res = norm(self.A.dot(x)-self.b) ** 2
        """ pure numpy lstsq to solve ill conditioned least squares problem, since lstsq does not return a residual value when the
        problem is ill conditioend"""
        return [x,res]
    
    """
            
    side quest, less flop = more cpu time, but executed by python = much worse than doing more under Fortran
    
    def rotate(x):
        if x[1] == 0:
            return array( [[1,0],[0,1]])
        if x[0] == 0:
            return array( [[0,1],[1,0]])
        if x[0] <= x[1]:
            cot = x[0]/x[1]
            s = 1/math.sqrt(1+cot ** 2)
            c = s * cot
            return array([[c, s],[-s, c]])
        else:
            tan = x[1]/x[0]
            c = 1/math.sqrt(1+tan ** 2)
            s = c * tan
            return array([[c, s],[-s, c]])
    
    def qr_givens_lstsq(ind,qb,r):
        l = len(ind)
        t = max(ind)+1
        m,n = shape(r)
        
        a = array(sorted(ind))
        b = arange(l)
        b = a - b
        
        s_ind = sorted(range(l),key = ind.__getitem__)
        ret_ind = sorted(range(l), key = s_ind.__getitem__)
        rtilde = hstack((r[:t,a],qb[:t,:]))
    
        
        skip = bisect.bisect_left(b,1)
        
        for i in range(skip,l):
            for j in range(ind[i],i,-1):
                rtilde[j-1:j+1,:] = rotate(rtilde[j-1:j+1,i]).dot( rtilde[j-1:j+1,:])
                
        x = solve_triangular(rtilde[:l,:l],rtilde[:l,-1])
        x = [x[i] for i in ret_ind]
        res = norm(rtilde[l:,[-1]]) ** 2 + norm(qb[t:]) ** 2
        
        return [x,res]
    """
    def experimentalsolver(self,P,C,d = 0):
        # This initial code was experiemental, for cleaner code use solve_sp_qr_mk3 
        """ 
        type(C) == int and all these unnecessary if statements are justified because python 
        ndarrays get lost when you slice the last elements , they even lose the dimensionality which 
        explains the reason behind   ## code_1 C1 = reshape(P[:,[0]],[self.m,1]) 
        rest of the shape, read it as dimension, controls are for the same reason as C , but they dont have
        initialization problem so I try to dodge the "pythonic" errors with couple of ifs, logic comes
        from the fact that P has a lot of independent variables and it will only go down, then we keep
        track of how many variables are left in our hands to not fall short, and we branch, find a feasible
        point and kill other nodes if we have a good feasible solution better than lower bound
        
        then everything is saved ofc, although this function does not need to return anything, I made it 
        return the same result as best_feasible which is actually sum of squared errors 
        
        """
        print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
         
            
        if type(C) == int:
            if shape(P)[1] > self.s:
                lower = lstsq(P,self.b)
                lower_bound = lower[1][0]
                if low >= self.best_feasible:
                    return Inf
                
                C1 = reshape(P[:,[0]],[self.m,1]) # code_1
                P1 = P[:,range(1,len(P[0]))]
                 
                self.node += 1
                first = self.experimentalsolver(P1,C1,d)
                second = self.experimentalsolver(P1,C2,d)
                return min(first,second)
            else:
                csolset = lstsq(P,self.b)
                if low < self.best_feasible:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = P
                return csolset[1][0]
        elif shape(C)[1] < self.s:
            if shape(hstack((P,C)))[0] <= self.s:
                csolset = lstsq(C,self.b)
                if low < self.best_feasible:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = P+C
                return csolset[1][0]
            else:
                lower = lstsq(hstack((P,C)),self.b)
                lower_bound = lower[1][0]
                if low >= self.best_feasible:
                    return Inf
                C1 = hstack((C,P[:,[0]]))
                P1 = P[:,range(1,len(P[0]))]
                 
                self.node += 1
                first = self.experimentalsolver(P1,C1,d)
                second = self.experimentalsolver(P1,C2,d-1)
                return min(first,second)
        elif shape(C)[1] == self.s:
            csolset = lstsq(C,self.b)
            if low < self.best_feasible:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = C
            return csolset[1][0]
        
    def solve_dfs(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
        """ this function does depth first search
        
        Branch and bound done lexicografigly
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        self.check += 1
        if low >= self.best_feasible:
            return Inf
        if len_c < self.s:
            if len_c + len_p <= self.s:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = P+C
                return low
            else:
                C1 = C + P[0:1]
                P1 = P[1:len(P)]
                self.node += 1
                lower2 =self.lstsq(P1+C)
                if len_c1 == self.s:
                    sol = self.lstsq(C1)
                    first = self.solve_dfs(P1,C1,sol[1][0],sol[0],len_c1,len_p)
                    second = self.solve_dfs(P1,C,lower2[1][0],lower2[0],len_c,len_p)
                    return min(first,second)
                else:
                    first = self.solve_dfs(P1,C1,low,coef,len_c1,len_p)
                    second = self.solve_dfs(P1,C,lower2[1][0],lower2[0],len_c,len_p)
                    return min(first,second)
        else:
            self.best_feasible = low
            self.solcoef = coef
            self.solset = C
            return low

    def solve_dfs_mk2(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
        
        """ this function does depth first search
        
        Branch and bound done according to |xbar(j)*a(j)| value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        self.check += 1
        if low >= self.best_feasible:
            return Inf
        if len_c < self.s:
            if len_c + len_p <= self.s:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = P+C
                return low
            else:
                xbar = self.means[P]
                
                bb_dec = abs(xbar*coef)
                 
                 
                l_index_bb = argmax(bb_dec)
                r_index_bb = P[l_index_bb]
                C1 = C + [r_index_bb]
                 
                  
                P1 = P[:]
                
                del P1[l_index_bb]
                coef = delete(coef,l_index_bb,0)
                self.node += 1
                lower2 =self.lstsq(P1+C)
                len_c1 = len_c +1
                len_p = len_p -1
                if len_c1 == self.s:
                    sol = self.lstsq(C1)
                    first = self.solve_dfs_mk2(P1,C1,sol[1][0],sol[0],len_c1,len_p)
                    second = self.solve_dfs_mk2(P1,C,lower2[1][0],lower2[0][0:len_p,0],len_c,len_p)
                    return min(first,second)
                else:
                    first = self.solve_dfs_mk2(P1,C1,low,coef,len_c1,len_p)
                    second = self.solve_dfs_mk2(P1,C,lower2[1][0],lower2[0][0:len_p,0],len_c,len_p)
                    return min(first,second)
        else:
            self.best_feasible = low
            self.solcoef = coef
            self.solset = C
            return low

    def solve_dfs_mk3(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
        
        """ this function does depth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        self.check += 1
        if low >= self.best_feasible:
            return Inf
        if len_c < self.s:
            if len_c + len_p <= self.s:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = P+C
                return low
            else:
                xbar = self.means[P]
                sdx = self.sterror[P]
                
                bb_dec = abs((sdx+xbar)*coef)
                 
                 
                l_index_bb = argmax(bb_dec)
                r_index_bb = P[l_index_bb]
                C1 = C + [r_index_bb]
                 
                  
                P1 = P[:]
                
                del P1[l_index_bb]
                coef = delete(coef,l_index_bb,0)
                self.node += 1
                lower2 =self.lstsq(P1+C)
                len_c1 = len_c +1
                len_p = len_p -1
                if len_c1 == self.s:
                    sol = self.lstsq(C1)
                    first = self.solve_dfs_mk3(P1,C1,sol[1][0],sol[0],len_c1,len_p)
                    second = self.solve_dfs_mk3(P1,C,lower2[1][0],lower2[0][0:len_p,0],len_c,len_p)
                    return min(first,second)
                else:
                    first = self.solve_dfs_mk3(P1,C1,low,coef,len_c1,len_p)
                    second = self.solve_dfs_mk3(P1,C,lower2[1][0],lower2[0][0:len_p,0],len_c,len_p)
                    return min(first,second)
        else:
            self.best_feasible = low
            self.solcoef = coef
            self.solset = C
            return low

    def solve_dfs_mk4(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
        
        """ this function does depth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        self.check += 1
        if low >= self.best_feasible:
            return Inf
        if len_c < self.s:
            if len_c + len_p <= self.s:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = P+C
                return low
            else:
                cov = inv(self.cov[P+C,:][:,P+C])
                var = diagonal(cov)[0:len_p]
                bb_dec = coef/var
                 
                 
                l_index_bb = argmax(bb_dec)
                r_index_bb = P[l_index_bb]
                C1 = C + [r_index_bb]
                 
                  
                P1 = P[:]
                
                del P1[l_index_bb]
                
                coef = delete(coef,l_index_bb,0)
                
                self.node += 1
                lower2 =self.lstsq(P1+C)
                len_c1 = len_c +1
                len_p = len_p -1
                if len_c1 == self.s:
                    sol = self.lstsq(C1)
                    first = self.solve_dfs_mk4(P1,C1,sol[1][0],sol[0],len_c1,len_p)
                    second = self.solve_dfs_mk4(P1,C,lower2[1][0],lower2[0][0:len_p,0],len_c,len_p)
                    return min(first,second)
                else:
                    first = self.solve_dfs_mk4(P1,C1,low,coef,len_c1,len_p)
                    second = self.solve_dfs_mk4(P1,C,lower2[1][0],lower2[0][0:len_p,0],len_c,len_p)
                    return min(first,second)
        else:
            self.best_feasible = low
            self.solcoef = coef
            self.solset = C
            return low
        
    def solve_bfs(self,P,C = []):
        """ this function does breadth first search
        
        Branch and bound done lexicografigly
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        q = Queue()
        f = self.lstsq(P+C)
        q.put([P,C,f[1][0],f[0],len(C),len(P)])
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            self.check += 1
            if low >= self.best_feasible:
                continue
            if len_c < self.s:
                if len_c + len_p <= self.s:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = P+C
                else:
                    C1 = C + P[0:1]
                    P1 = P[1:len(P)]
                    self.node += 1
                    lower2 =self.lstsq(P1+C)
                    len_c1 = len_c +1
                    len_p = len_p -1
                    if len_c1  == self.s:
                        sol = self.lstsq(C1)
                        q.put([P1,C1,sol[1][0],sol[0],len_c1,len_p])
                        q.put([P1,C,lower2[1][0],lower2[0],len_c,len_p])
                    else:
                        q.put([P1,C1,low,coef,len_c1,len_p])
                        q.put([P1,C,lower2[1][0],lower2[0],len_c,len_p])
            else:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = C

    def solve_bfs_mk2(self,P,C = []):
        """ this function does breadth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        q = Queue()
        f = self.lstsq(P+C)
        lenp = len(P)
        q.put([P,C,f[1][0],f[0][0:lenp,0],len(C),lenp])
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            self.check += 1
            if low >= self.best_feasible:
                continue
            if len_c < self.s:
                if len_c + len_p <= self.s:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = P+C
                else:
                    xbar = self.means[P]
                    
                    bb_dec = abs(xbar*coef)
                     
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    C1 = C + [r_index_bb]
                     
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    
                    coef = delete(coef,l_index_bb,0)
                    
                    self.node += 1
                    lower2 =self.lstsq(P1+C)
                    len_c1 = len_c +1
                    len_p = len_p -1
                    if len_c1  == self.s:
                        sol = self.lstsq(C1)
                        q.put([P1,C1,sol[1][0],sol[0],len_c1,len_p])
                        q.put([P1,C,lower2[1][0],lower2[0][0:len_p,0],len_c,len_p])
                    else:
                        q.put([P1,C1,low,coef,len_c1,len_p])
                        q.put([P1,C,lower2[1][0],lower2[0][0:len_p,0],len_c,len_p])
            else:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = C
         
    def solve_bfs_mk3(self,P,C = []):
        """ this function does breadth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        q = Queue()
        f = self.lstsq(P+C)
        lenp = len(P)
        q.put([P,C,f[1][0],f[0][0:lenp,0],len(C),lenp])
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            self.check += 1
            if low >= self.best_feasible:
                continue
             
            if len_c < self.s:
                if len_c + len_p <= self.s:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = P+C
                else:
                    xbar = self.means[P]
                    sdx = self.sterror[P]
                    
                    bb_dec = abs((sdx+xbar)*coef)
                     
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    C1 = C + [r_index_bb]
                     
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    self.node += 1
                    lower2 =self.lstsq(P1+C)
                    len_c1 = len_c +1
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq(C1)
                        q.put([P1,C1,sol[1][0],sol[0],len_c1,len_p])
                        q.put([P1,C,lower2[1][0],lower2[0][0:len_p,0],len_c,len_p])
                    else:
                        q.put([P1,C1,low,coef,len_c1,len_p])
                        q.put([P1,C,lower2[1][0],lower2[0][0:len_p,0],len_c,len_p])
            else:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = C


    def solve_bfs_mk4(self,P,C = []):
        """ this function does breadth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        q = Queue()
        f = self.lstsq(P+C)
        lenp = len(P)
        q.put([P,C,f[1][0],f[0][0:lenp,0],len(C),lenp])
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            self.check += 1
            if low >= self.best_feasible:
                continue
            if len_c < self.s:
                if len_c + len_p <= self.s:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = P+C
                else:
                    cov = inv(self.cov[P+C,:][:,P+C])
                    var = diagonal(cov)[0:len_p]
                    bb_dec = coef/var
                     
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    C1 = C + [r_index_bb]
                     
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    self.node += 1
                    lower2 =self.lstsq(P1+C)
                    len_c1 = len_c +1
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq(C1)
                        q.put([P1,C1,sol[1][0],sol[0],len_c1,len_p])
                        q.put([P1,C,lower2[1][0],lower2[0][0:len_p,0],len_c,len_p])
                    else:
                        q.put([P1,C1,low,coef,len_c1,len_p])
                        q.put([P1,C,lower2[1][0],lower2[0][0:len_p,0],len_c,len_p])
            else:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = C

    def solve_dfsi(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
        """ this function does depth first search
        
        Branch and bound done lexicografigly
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        self.check += 1
        if low >= self.best_feasible:
            return Inf
        if len_c < self.s:
            if len_c + len_p <= self.s:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = P+C
                return low
            else:
                C1 = C + P[0:1]
                P1 = P[1:len(P)]
                self.node += 1
                lower2 =self.lstsqi(P1+C)

                len_c1 = len_c +1 
                len_p = len_p -1
                if len_c == self.s:
                    sol = self.lstsqi(C1)
                    sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                    first = self.solve_dfsi(P1,C1,sol_sse,sol[0],len_c1,len_p)
                    second = self.solve_dfsi(P1,C,lower2[1],lower2[0],len_c,len_p)
                    return min(first,second)
                else:
                    first = self.solve_dfsi(P1,C1,low,coef,len_c1,len_p)
                    second = self.solve_dfsi(P1,C,lower2[1],lower2[0],len_c,len_p)
                    return min(first,second)
        else:
            self.best_feasible = low
            self.solcoef = coef
            self.solset = C
            return low

    def solve_dfsi_mk2(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
        
        """ this function does depth first search
        
        Branch and bound done according to |xbar(j)*a(j)| value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        self.check += 1
        if low >= self.best_feasible:
            return Inf
        if len_c < self.s:
            if len_c + len_p <= self.s:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = P+C
                return low
            else:
                xbar = self.means[P]
                
                bb_dec = abs(xbar*coef)
                 
                 
                l_index_bb = argmax(bb_dec)
                r_index_bb = P[l_index_bb]
                C1 = C + [r_index_bb]
                 
                  
                P1 = P[:]
                
                del P1[l_index_bb]
                coef = delete(coef,l_index_bb,0)
                self.node += 1
                lower2 =self.lstsqi(P1+C)

                len_c1 = len_c +1 
                len_p = len_p -1
                if len_c1 == self.s:
                    sol = self.lstsqi(C1)
                    sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                    first = self.solve_dfsi_mk2(P1,C1,sol_sse,sol[0],len_c1,len_p)
                    second = self.solve_dfsi_mk2(P1,C,lower2[1],lower2[0][0:len_p,0],len_c,len_p)
                    return min(first,second)
                else:
                    first = self.solve_dfsi_mk2(P1,C1,low,coef,len_c1,len_p)
                    second = self.solve_dfsi_mk2(P1,C,lower2[1],lower2[0][0:len_p,0],len_c,len_p)
                    return min(first,second)
        else:
            self.best_feasible = low
            self.solcoef = coef
            self.solset = C
            return low

    def solve_dfsi_mk3(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
        
        """ this function does depth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        self.check += 1
        if low >= self.best_feasible:
            return Inf
        if len_c < self.s:
            if len_c + len_p <= self.s:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = P+C
                return low
            else:
                xbar = self.means[P]
                sdx = self.sterror[P]
                
                bb_dec = abs((sdx+xbar)*coef)
                 
                 
                l_index_bb = argmax(bb_dec)
                r_index_bb = P[l_index_bb]
                C1 = C + [r_index_bb]
                 
                  
                P1 = P[:]
                
                del P1[l_index_bb]
                coef = delete(coef,l_index_bb,0)
                self.node += 1
                lower2 =self.lstsqi(P1+C)
                
                len_c1 = len_c +1 
                len_p = len_p -1
                if len_c1 == self.s:
                    sol = self.lstsqi(C1)
                    sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                    first = self.solve_dfsi_mk3(P1,C1,sol_sse,sol[0],len_c1,len_p)
                    second = self.solve_dfsi_mk3(P1,C,lower2[1],lower2[0][0:len_p,0],len_c,len_p)
                    return min(first,second)
                else:
                    first = self.solve_dfsi_mk3(P1,C1,low,coef,len_c1,len_p)
                    second = self.solve_dfsi_mk3(P1,C,lower2[1],lower2[0][0:len_p,0],len_c,len_p)
                    return min(first,second)
        else:
            self.best_feasible = low
            self.solcoef = coef
            self.solset = C
            return low

    def solve_dfsi_mk4(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
        
        """ this function does depth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        self.check += 1
        if low >= self.best_feasible:
            return Inf
        if len_c < self.s:
            if len_c + len_p <= self.s:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = P+C
                return low
            else:
                cov = inv(self.cov[P+C,:][:,P+C])
                var = diagonal(cov)[0:len_p,0]
                bb_dec = coef/var

                 
                 
                l_index_bb = argmax(bb_dec)
                r_index_bb = P[l_index_bb]
                C1 = C + [r_index_bb]
                 
                  
                P1 = P[:]
                
                del P1[l_index_bb]
                
                coef = delete(coef,l_index_bb,0)
                
                self.node += 1
                lower2 =self.lstsqi(P1+C)

                len_c1 = len_c +1 
                len_p = len_p -1
                if len_c1 == self.s:
                    sol = self.lstsqi(C1)
                    sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                    first = self.solve_dfsi_mk4(P1,C1,sol_sse,sol[0],len_c1,len_p)
                    second = self.solve_dfsi_mk4(P1,C,lower2[1],lower2[0][0:len_p,0],len_c,len_p)
                    return min(first,second)
                else:
                    first = self.solve_dfsi_mk4(P1,C1,low,coef,len_c1,len_p)
                    second = self.solve_dfsi_mk4(P1,C,lower2[1],lower2[0][0:len_p,0],len_c,len_p)
                    return min(first,second)
        else:
            self.best_feasible = low
            self.solcoef = coef
            self.solset = C
            return low
        
    def solve_bfsi(self,P,C = []):
        """ this function does breadth first search
        
        Branch and bound done lexicografigly
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        q = Queue()
        f = self.lstsqi(P+C)
        q.put([P,C,f[1],f[0],len(C),len(P)])
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            self.check += 1
            if low >= self.best_feasible:
                continue
             
            [P,C,low,coef,len_c,len_p] = q.get()
            if len_c < self.s:
                if len_c + len_p <= self.s:
                    self.best_feasible = cost
                    self.solcoef = coef
                    self.solset = P+C
                else:
                    C1 = C+ P[0:1]
                    P1 = P[1:len(P)]
                     
                    self.node += 1
                    lower2 =self.lstsqi(P1+C)
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsqi(C1)
                        sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                        q.put([P1,C1,sol_sse,sol[0],len_c1,len_p])
                        q.put([P1,C,f[1],lower2[0],len_c,len_p])
                    else:
                        q.put([P1,C1,low,coef,len_c1,len_p])
                        q.put([P1,C,f[1],lower2[0],len_c,len_p])
            else:
                self.best_feasible = cost
                self.solcoef = coef
                self.solset = C

    def solve_bfsi_mk2(self,P,C = []):
        """ this function does breadth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        q = Queue()
        f = self.lstsqi(P+C)
        q.put([P,C,f[1],f[0][0:lenp,0],len(C),len(P)])
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            self.check += 1
            if low >= self.best_feasible:
                continue
            if len_c < self.s:
                if len_c + len_p <= self.s:
                    self.best_feasible = cost
                    self.solcoef = coef
                    self.solset = P+C
                else:
                    xbar = self.means[P]
                    
                    bb_dec = abs(xbar*coef)
                     
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    C1 = C + [r_index_bb]
                     
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    self.node += 1
                    lower2 =self.lstsqi(P1+C)
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsqi(C1)
                        sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                        q.put([P1,C1,sol_sse,sol[0],len_c1,len_p])
                        q.put([P1,C,f[1],lower2[0][0:len_p],len_c,len_p])
                    else:
                        q.put([P1,C1,low,coef,len_c1,len_p])
                        q.put([P1,C,f[1],lower2[0][0:len_p],len_c,len_p])
            else:
                self.best_feasible = cost
                self.solcoef = coef
                self.solset = C
         
    def solve_bfsi_mk3(self,P,C = []):
        """ this function does breadth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        q = Queue()
        f = self.lstsqi(P+C)
        q.put([P,C,f[1],f[0][0:lenp],len(C),len(P)])
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            self.check += 1
            if low >= self.best_feasible:
                continue
            if len_c < self.s:
                if len_c + len_p <= self.s:
                    self.best_feasible = cost
                    self.solcoef = coef
                    self.solset = P+C
                else:
                    xbar = self.means[P]
                    sdx = self.sterror[P]
                    
                    bb_dec = abs((sdx+xbar)*coef)
                     
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    C1 = C + [r_index_bb]
                     
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    self.node += 1
                    lower2 =self.lstsqi(P1+C)
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsqi(C1)
                        sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                        q.put([P1,C1,sol_sse,sol[0],len_c1,len_p])
                        q.put([P1,C,f[1],lower2[0][0:len_p,0],len_c,len_p])
                    else:
                        q.put([P1,C1,low,coef,len_c1,len_p])
                        q.put([P1,C,f[1],lower2[0][0:len_p,0],len_c,len_p])
            else:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = C

    def solve_bfsi_mk4(self,P,C = []):
        """ this function does breadth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        q = Queue()
        f = self.lstsqi(P+C)
        cost = f_LI(self.A,self.b,f[0])
        lenp = len(P)
        q.put([P,C,cost,f[0][0:lenp,0],len(C),lenp])
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            self.check += 1
            if low >= self.best_feasible:
                continue
             
            if len_c < self.s:
                if len_c + len_p <= self.s:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = P+C
                else:
                    
                    cov = inv(self.cov[P+C,:][:,P+C])
                    var = diagonal(cov)[0:len_p]
                    bb_dec = coef/var
                     
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    C1 = C + [r_index_bb]
                     
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    self.node += 1
                    lower2 =self.lstsqi(P1+C)

                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsqi(C1)
                        sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                        q.put([P1,C1,sol_sse,sol[0],len_c1,len_p])
                        q.put([P1,C,lower2[1],lower2[0][0:len_p,0],len_c,len_p])
                    else:
                        q.put([P1,C1,low,coef,len_c1,len_p])
                        q.put([P1,C,lower2[1],lower2[0][0:len_p,0],len_c,len_p])
            else:
                 
                self.best_feasible = low
                self.solcoef = coef
                self.solset = C

    def solve_dfs_qr(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
        """ this function does depth first search
        
        Branch and bound done lexicografigly
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        self.check += 1
        if low >= self.best_feasible:
            return Inf
        if len_c < self.s:
            if len_c + len_p <= self.s:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = P+C
                return low
            else:
                C1 = C + P[0:1]
                P1 = P[1:len(P)]
                self.node += 1
                lower2 = self.qr_lstsq(P1+C)
                len_c1 = len_c +1 
                len_p = len_p -1
                if len_c1 == self.s:
                    sol =self.qr_lstsq(C1)
                    first = self.solve_dfs_qr(P1,C1,sol[1],sol[0],len_c1,len_p)
                    second = self.solve_dfs_qr(P1,C,lower2[1],lower2[0],len_c,len_p)
                    return min(first,second)
                else:
                    first = self.solve_dfs_qr(P1,C1,low,coef,len_c1,len_p)
                    second = self.solve_dfs_qr(P1,C,lower2[1],lower2[0],len_c,len_p)
                    return min(first,second)
        else:
            self.best_feasible = low
            self.solcoef = coef
            self.solset = C
            return low

    def solve_dfs_qr_mk2(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
        
        """ this function does depth first search
        
        Branch and bound done according to |xbar(j)*a(j)| value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        self.check += 1
        if low >= self.best_feasible:
            return Inf
        if len_c < self.s:
            if len_c + len_p <= self.s:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = P+C
                return low
            else:
                xbar = self.means[P]
                
                bb_dec = abs(xbar*coef)
                 
                 
                l_index_bb = argmax(bb_dec)
                r_index_bb = P[l_index_bb]
                C1 = C + [r_index_bb]
                 
                  
                P1 = P[:]
                
                del P1[l_index_bb]
                coef = delete(coef,l_index_bb,0)
                self.node += 1
                lower2 =self.qr_lstsq(P1+C)
                len_c1 = len_c +1 
                len_p = len_p -1
                if len_c1 == self.s:
                    sol =self.qr_lstsq(C1)
                    first = self.solve_dfs_qr_mk2(P1,C1,sol[1],sol[0],len_c1,len_p)
                    second = self.solve_dfs_qr_mk2(P1,C,lower2[1],lower2[0][0:len_p],len_c,len_p)
                    return min(first,second)
                else:
                    first = self.solve_dfs_qr_mk2(P1,C1,low,coef,len_c1,len_p)
                    second = self.solve_dfs_qr_mk2(P1,C,lower2[1],lower2[0][0:len_p],len_c,len_p)
                    return min(first,second)
        else:
            self.best_feasible = low
            self.solcoef = coef
            self.solset = C
            return low

    def solve_dfs_qr_mk3(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
        
        """ this function does depth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        self.check += 1
        if low >= self.best_feasible:
            return Inf
        if len_c < self.s:
            if len_c + len_p <= self.s:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = P+C
                return low
            else:
                xbar = self.means[P]
                sdx = self.sterror[P]
                
                bb_dec = abs((sdx+xbar)*coef)
                 
                 
                l_index_bb = argmax(bb_dec)
                r_index_bb = P[l_index_bb]
                C1 = C + [r_index_bb]
                 
                P1 = P[:]
                
                del P1[l_index_bb]
                coef = delete(coef,l_index_bb,0)
                self.node += 1
                lower2 = self.qr_lstsq(P1+C)
                len_c1 = len_c +1 
                len_p = len_p -1
                if len_c1 == self.s:
                    sol = self.qr_lstsq(C1)
                    first = self.solve_dfs_qr_mk3(P1,C1,sol[1],sol[0],len_c1,len_p)
                    second = self.solve_dfs_qr_mk3(P1,C,lower2[1],lower2[0][0:len_p],len_c,len_p)
                    return min(first,second)
                else:
                    first = self.solve_dfs_qr_mk3(P1,C1,low,coef,len_c1,len_p)
                    second = self.solve_dfs_qr_mk3(P1,C,lower2[1],lower2[0][0:len_p],len_c,len_p)
                    return min(first,second)
        else:
            self.best_feasible = low
            self.solcoef = coef
            self.solset = C
            return low

    def solve_dfs_qr_mk4(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
        
        """ this function does depth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        self.check += 1
        if low >= self.best_feasible:
            return Inf
        if len_c < self.s:
            if len_c + len_p <= self.s:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = P+C
                return low
            else:
                cov = inv(self.cov[P+C,:][:,P+C])
                var = diagonal(cov)[0:len_p]
                bb_dec = coef/var
  
                 
                 
                l_index_bb = argmax(bb_dec)
                r_index_bb = P[l_index_bb]
                C1 = C + [r_index_bb]
                 
                  
                P1 = P[:]
                
                del P1[l_index_bb]
                
                coef = delete(coef,l_index_bb,0)
                self.node += 1
                lower2 = self.qr_lstsq(P1+C)
                len_c1 = len_c +1 
                len_p = len_p -1
                if len_c1 == self.s:
                    sol = self.qr_lstsq(C1)
                    first = self.solve_dfs_qr_mk4(P1,C1,sol[1],sol[0],len_c1,len_p)
                    second = self.solve_dfs_qr_mk4(P1,C,lower2[1],lower2[0][0:len_p],len_c,len_p)
                    return min(first,second)
                else:
                    first = self.solve_dfs_qr_mk4(P1,C1,low,coef,len_c1,len_p)
                    second = self.solve_dfs_qr_mk4(P1,C,lower2[1],lower2[0][0:len_p],len_c,len_p)
                    return min(first,second)
        else:
            self.best_feasible = low
            self.solcoef = coef
            self.solset = C
            return low
        
        
    def solve_bfs_qr(self,P,C = []):
        """ this function does breadth first search
        
        Branch and bound done lexicografigly
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        q = Queue()
        f =self.qr_lstsq(P+C)
        lenp = len(P)
        q.put([P,C,f[1],f[0][0:lenp],len(C),lenp])
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            self.check += 1
            if low >= self.best_feasible:
                continue
             
            if len_c < self.s:
                if len_c + len_p <= self.s:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = P+C
                else:
                    C1 = C+ P[0:1]
                    P1 = P[1:len(P)]
                    self.node += 1
                    lower2 = self.qr_lstsq(P1+C)
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_lstsq(C1)
                        q.put([P1,C1,sol[1],sol[0],len_c1,len_p])
                        q.put([P1,C,lower2[1],lower2[0],len_c,len_p])
                    else:
                        q.put([P1,C1,low,coef,len_c1,len_p])
                        q.put([P1,C,lower2[1],lower2[0],len_c,len_p])
            else:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = C

    def solve_bfs_qr_mk2(self,P,C = []):
        """ this function does breadth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        q = Queue()
        f =self.qr_lstsq(P+C)
        lenp = len(P)
        q.put([P,C,f[1],f[0][0:lenp],len(C),lenp])
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            self.check += 1
            if low >= self.best_feasible:
                continue
            if len_c < self.s:
                if len_c + len_p <= self.s:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = P+C
                else:
                    xbar = self.means[P]
                    
                    bb_dec = abs(xbar*coef)
                     
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    C1 = C + [r_index_bb]
                     
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    self.node += 1
                    lower2 = self.qr_lstsq(P1+C)
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_lstsq(C1)
                        q.put([P1,C1,sol[1],sol[0],len_c1,len_p])
                        q.put([P1,C,lower2[1],lower2[0][0:len_p],len_c,len_p])
                    else:
                        q.put([P1,C1,low,coef,len_c1,len_p])
                        q.put([P1,C,lower2[1],lower2[0][0:len_p],len_c,len_p])
            else:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = C
         
    def solve_bfs_qr_mk3(self,P,C = []):
        """ this function does breadth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        q = Queue()
        f = self.qr_lstsq(P+C)
        lenp = len(P)
        q.put([P,C,f[1],f[0][0:lenp],len(C),lenp])
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            self.check += 1
            if low >= self.best_feasible:
                continue
             
            if len_c < self.s:
                if len_c + len_p <= self.s:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = P+C
                else:
                    xbar = self.means[P]
                     
                    sdx = self.sterror[P]
                    
                    bb_dec = abs((sdx+xbar)*coef)
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    C1 = C + [r_index_bb]
                     
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    self.node += 1
                    lower2 = self.qr_lstsq(P1+C)
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_lstsq(C1)
                        q.put([P1,C1,sol[1],sol[0],len_c1,len_p])
                        q.put([P1,C,lower2[1],lower2[0][0:len_p],len_c,len_p])
                    else:
                        q.put([P1,C1,low,coef,len_c1,len_p])
                        q.put([P1,C,lower2[1],lower2[0][0:len_p],len_c,len_p])
            else:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = C
                    
    def solve_bfs_qr_mk4(self,P,C = []):
        """ this function does breadth first search
        
        Branch and bound done according to (|xbar(j)| + varx(j))*a(j) value
        for more detail look at solve_sp_mk3 
        
        This is never recommended to be used, use solve_sp_qr_mk3 instead"""
        q = Queue()
        f = self.qr_lstsq(P+C)
        lenp = len(P)
        q.put([P,C,f[1],f[0][0:lenp],len(C),lenp])
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            self.check += 1
            if low >= self.best_feasible:
                continue
             
            if len_c < self.s:
                if len_c + len_p <= self.s:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = P+C
                else:
                    cov = inv(self.cov[P+C,:][:,P+C])
                    var = diagonal(cov)[0:len_p]
                    bb_dec = coef/var

                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    C1 = C + [r_index_bb]
                     
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    self.node += 1
                    lower2 = self.qr_lstsq(P1+C)
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_lstsq(C1)
                        q.put([P1,C1,sol[1],sol[0],len_c1,len_p])
                        q.put([P1,C,lower2[1],lower2[0][0:len_p],len_c,len_p])
                    else:
                        q.put([P1,C1,low,coef,len_c1,len_p])
                        q.put([P1,C,lower2[1],lower2[0][0:len_p],len_c,len_p])
            else:
                self.best_feasible = low
                self.solcoef = coef
                self.solset = C


    def solve_bb_experimental(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        """ this algorithm does branch and bound with greedy logic, depending on the SSE, it does 
        bfs (breadth first search) or dfs (depth first search) so best first search in other words
        
        Branch and bound done lexicografigly
        """
        q = PriorityQueue()
        q.put([1.5,[P,C,d]])
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            self.check += 1
            if low >= self.best_feasible:
                continue
            print("qsize")
            print(q.qsize())
             
            [low,[P,C,d]] = q.get()
            print("proof I am working",str(self.node),"many nodes by now",str(q.qsize()),",qsizebt\
#w",str(self.best_feasible),"thisis best feasible,current lowerbound",str(low))
            if len_c < self.s:
                if len_c + len_p <= self.s:
                     
                    self.check += 1
                    if low < self.best_feasible:
                        self.best_feasible = low
                        self.solcoef = coef
                        self.solset = P+C
                else:
                    lower1 = lstsq(self.A[:,P+C],self.b)
                    lower_bound1 = lower1[1][0]
                    self.check += 1
                    if lower_bound1 >= self.best_feasible:
                        continue
                    C1 = C+ P[0:1]
                    P1 = P[1:len(P)]
                     
                    self.node += 1
                    lower2 =  lstsq(self.A[:,[int(x) for x in hstack((P1,C2))]],self.b)
                    lower2[1] = lower2[1][0]
                    q.put([lower_bound1 ,[P1,C1,d]])
                    q.put([lower2[1] ,[P1,C2,d-1]])
            else:
                self.check += 1
                 
                if low < self.best_feasible:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = C
                    
    def solve_sp_experimental(self,P,C = [], low = 0,  coef = None, len_c = 0, len_p = -1):
        """
        This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independent variable
        if |x(j)*a(j)| is the highest then it has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        this code is experimental for readability look at mk2
        """
        q = PriorityQueue()
        q.put([0,[P,C,d]])
        i = 1
        while q.qsize() > 0:
            [P,C,low,coef,len_c,len_p] = q.get()
            self.check += 1
            if low >= self.best_feasible:
                continue
            i += 1
             
            [low,[P,C,d]] = q.get()
            print("lowerbound for now",low,"at depth",d)
            if len_c < self.s:
                if len_c + len_p <= self.s:
                    self.check += 1
                     
                    if csolset[1][0] <= self.best_feasible:
                        self.best_feasible = low
                        self.solcoef = coef
                        self.solset = P+C
                        break
                else:
                    lower1 = lstsq(self.A[:,P+C],self.b)
                    lower_bound1 = lower1[1][0]
                    self.check += 1
                    if lower_bound1 >= self.best_feasible:
                        continue
                    gwa = reshape(self.means[P],[1,len(P)])
                    zulul = reshape(lower1[0][range(len(P))],[1,len(P)])
                    bb_dec = gwa[0]*zulul[0]
                     
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    C1 = C + [r_index_bb]
                     
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                     
                    self.node += 1
                    lower2 = lstsq(self.A[:,[int(x) for x in hstack((P1,C2))]],self.b)
                    lower2[1] = lower2[1][0]
                    if len_c +1 == self.s:
                        sol = self.lstsq(C1)
                        sol_sse = sol[1][0]
                        q.put([sol_sse,[[],C1,d]])
                        q.put([lower2[1],[P1,C2,d-1]])
                    else:
                        first = q.put([lower_bound1 ,[P1,C1,d]])
                        second = q.put([lower2[1] ,[P1,C2,d-1]])
            else:
                 
                self.check += 1
                if csolset[1][0] <= self.best_feasible:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = C
                    break
                
    def solve_sp(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p )
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.best_feasible = low
                    """ register SSE, sum of squares error."""
                    self.solcoef = coef
                    """ register Least Squares coefficients of the variables in the order 
                    that is given by solset"""
                    self.solset = P+C
                    """ register the solution set variables' indexes, it is ordered and the 
                    coefficient of each variable is in the same index in solcoef"""
                    self.rem_qsize = q.qsize()
                    """ register the number of unvisited nodes """
                    break
                else:
                    C1 = C+ P[0:1]
                    P1 = P[1:len(P)]
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                """ register the solution if it is better than any previous feasible solution """
                self.best_feasible = low
                """ register SSE, sum of squares error."""
                self.solcoef = coef
                """ register Least Squares coefficients of the variables in the order 
                that is given by solset"""
                self.solset = C
                """ register the solution set variables' indexes, it is ordered and the 
                coefficient of each variable is in the same index in solcoef"""
                self.rem_qsize = q.qsize()
                """ register the number of unvisited nodes """
                break
                    
    def solve_spm(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        
        
        This is modified solve_sp for multiple subsets
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p )
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    C1 = C+ P[0:1]
                    P1 = P[1:len(P)]
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break
            
    def solve_sp_mk2(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.best_feasible = low
                    """ register SSE, sum of squares error."""
                    self.solcoef = coef
                    """ register Least Squares coefficients of the variables in the order 
                    that is given by solset"""
                    self.solset = P+C
                    """ register the solution set variables' indexes, it is ordered and the 
                    coefficient of each variable is in the same index in solcoef"""
                    self.rem_qsize = q.qsize()
                    """ register the number of unvisited nodes """
                    break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    bb_dec = abs(xbar*coef)
                    """ bb_dec is the decision vector, logic is explained above
                     absolute value because coef might be negative """
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                """ register the solution if it is better than any previous feasible solution """
                self.best_feasible = low
                """ register SSE, sum of squares error."""
                self.solcoef = coef
                """ register Least Squares coefficients of the variables in the order 
                that is given by solset"""
                self.solset = C
                """ register the solution set variables' indexes, it is ordered and the 
                coefficient of each variable is in the same index in solcoef"""
                self.rem_qsize = q.qsize()
                """ register the number of unvisited nodes """
                break

    def solve_sp_mk2m(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        
        This is modified mk2 for multiple subsets
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    bb_dec = abs(xbar*coef)
                    """ bb_dec is the decision vector, logic is explained above
                     absolute value because coef might be negative """
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break
               
    def solve_sp_mk3(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.best_feasible = low
                    """ register SSE, sum of squares error."""
                    self.solcoef = coef
                    """ register Least Squares coefficients of the variables in the order 
                    that is given by solset"""
                    self.solset = P+C
                    """ register the solution set variables' indexes, it is ordered and the 
                    coefficient of each variable is in the same index in solcoef"""
                    self.rem_qsize = q.qsize()
                    """ register the number of unvisited nodes """
                    break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    sdx = self.sterror[P]
                    """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                    bb_dec = abs((sdx+xbar)*coef)
                    """ bb_dec is the decision vector, logic is explained above"""
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                     
                    """ this unnecessary copying of P, I did not prefer to pass P, it might be ok,
                    but some times python passes by reference and it is dangerous because one node
                    changes vector P and other nodes use the new changed P, rather than the original
                    P passed to it """      
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                """ register the solution if it is better than any previous feasible solution """
                self.best_feasible = low
                """ register SSE, sum of squares error."""
                self.solcoef = coef
                """ register Least Squares coefficients of the variables in the order 
                that is given by solset"""
                self.solset = C
                """ register the solution set variables' indexes, it is ordered and the 
                coefficient of each variable is in the same index in solcoef"""
                self.rem_qsize = q.qsize()
                """ register the number of unvisited nodes """
                break
            
    def solve_sp_mk3m(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        
        This is modified mk3 for multiple subsets
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    sdx = self.sterror[P]
                    """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                    bb_dec = abs((sdx+xbar)*coef)
                    """ bb_dec is the decision vector, logic is explained above"""
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                     
                    """ this unnecessary copying of P, I did not prefer to pass P, it might be ok,
                    but some times python passes by reference and it is dangerous because one node
                    changes vector P and other nodes use the new changed P, rather than the original
                    P passed to it """      
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break

    def solve_sp_mk4(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p)
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.best_feasible = low
                    """ register SSE, sum of squares error."""
                    self.solcoef = coef
                    """ register Least Squares coefficients of the variables in the order 
                    that is given by solset"""
                    self.solset = P+C
                    """ register the solution set variables' indexes, it is ordered and the 
                    coefficient of each variable is in the same index in solcoef"""
                    self.rem_qsize = q.qsize()
                    """ register the number of unvisited nodes """
                    break
                else:
                    cov = inv(self.cov[P+C,:][:,P+C])
                    var = diagonal(cov)[0:len_p]
                    bb_dec = coef/var
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                """ register the solution if it is better than any previous feasible solution """
                self.best_feasible = low
                """ register SSE, sum of squares error."""
                self.solcoef = coef
                """ register Least Squares coefficients of the variables in the order 
                that is given by solset"""
                self.solset = C
                """ register the solution set variables' indexes, it is ordered and the 
                coefficient of each variable is in the same index in solcoef"""
                self.rem_qsize = q.qsize()
                """ register the number of unvisited nodes """
                break

    def solve_sp_mk4m(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        
        This is modified mk4 for multiple subsets
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p)
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    cov = inv(self.cov[P+C,:][:,P+C])
                    var = diagonal(cov)[0:len_p]
                    bb_dec = coef/var
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break
               
                
    def solve_sp_mk0(self,P ,C = []):
        """ 
        This algorithm finds all subsets of size 1 to s, with pure qr decomposition which is slower than svd, because q is formed 
        explicitly and makes the algorithm run slower. Best first search and m-st-lsc is used.
        
        For faster algortihm, use mk0s.
        
        """

        L = [0]*(self.s+1)
        for i in range(self.s+1):
            L[i] = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.qr_lstsq(P+C)
        self.tqsize = []
        lenp = len(P)
        lenc = len(C)
        for i in range(len(L)):
            L[i].put([f[1],[P,C,f[0][0:lenp],lenc,lenp]])
        """ initialization of the first problem """
        s = self.s
        i = 0
        while i < s:
            i += 1
            self.s = i
            print("Started solving for sparsity level ",self.s)
            count_best = 0
            self.node = 0
            while True:
                """ termination condition of the problem if we visit all the nodes then search is over """
                [low,[P,C,coef,len_c,len_p]] = L[i].get()
                """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
                print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-2:-1])
                if len_c < self.s:
                    if len_p + len_c <= self.s:
                        self.residual_squared.append(low)
                        self.indexes.append(P+C)
                        self.coefficients.append(coef)
                        count_best += 1
                        if count_best == self.many:
                            self.tqsize.append(L[i].qsize())
                            break
                    else:
                        xbar = self.means[P]
                        """ xbar is a vector length len(p), it retrieves the mean for each variable """
                        sdx = self.sterror[P]
                        """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                        bb_dec = abs((sdx+xbar)*coef)
                        """ bb_dec is the decision vector, logic is explained above"""
                        l_index_bb = argmax(bb_dec)
                        """ find index of the largest value in decision vector """
                        r_index_bb = P[l_index_bb]
                        """ find index of the variable by index of the largest value in decision vector 
                        this is the chosen variable for this node """ 
                        C1 = C + [r_index_bb]
                        coef = delete(coef,l_index_bb,0)
                        """ add the new chosen variable to the solution 
                        We also use old C, where chosen variable is not added""" 
                          
                        P1 = P[:]
                        del P1[l_index_bb]
                        """ erasing the chosen variable from the possible variables' list 
                        reminder: this stores the variables by their indexes"""
                        self.node += 1
                        """ update number of nodes visited """
                        lower2 =self.qr_lstsql(P1+C)
                        """ calculate lower bound of the second solution where C is the old chosen variable 
                        list and p1 is the possible variable (indexes ) list where the chosen variable for 
                        this node is erased """
                        len_p -= 1
                        len_c1 = len_c +1 
                        if len_c1 == self.s:
                            """ fix here """
                            sol = self.qr_lstsq(C1)
                            L[i].put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                            L[i].put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                        else:
                            """ if the length of the chosen variable list is not equal to sparsity level, 
                            then it is lower than sparsity level. We create two new nodes where first node
                            is the node where chosen variable for this node is in the solution and the second 
                            where the chosen variable for this node is erased from the problem """
                            
                            L[i].put([low ,[P1,C1,coef,len_c1,len_p]])
                            L[i].put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
                else:
                    self.residual_squared.append(low)
                    self.indexes.append(C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.tqsize.append(L[i].qsize())
                        break

                    
    def solve_sp_mk0s(self,P ,C = []):
        """ 
        
        This algorithm finds all subsets of size 1 to s, after initial qr, svd is used to find least squares solution, it is faster because
        q,r requires q to be formed explicitly which makes it slower. Best first search and m-st-lsc is used.
        
        """

        L = [0]*(self.s+1)
        for i in range(self.s+1):
            L[i] = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsq_qr(P+C)
        self.tqsize = []
        lenp = len(P)
        lenc = len(C)
        for i in range(len(L)):
            L[i].put([f[1],[P,C,f[0][0:lenp],lenc,lenp]])
        """ initialization of the first problem """
        s = self.s
        i = 0
        while i < s:
            i += 1
            self.s = i
            print("Started solving for sparsity level ",self.s)
            count_best = 0
            self.node = 0
            while True:
                """ termination condition of the problem if we visit all the nodes then search is over """
                [low,[P,C,coef,len_c,len_p]] = L[i].get()
                """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
                print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-2:-1])
                if len_c < self.s:
                    if len_p + len_c <= self.s:
                        self.residual_squared.append(low)
                        self.indexes.append(P+C)
                        self.coefficients.append(coef)
                        count_best += 1
                        if count_best == self.many:
                            self.tqsize.append(L[i].qsize())
                            break
                    else:
                        xbar = self.means[P]
                        """ xbar is a vector length len(p), it retrieves the mean for each variable """
                        sdx = self.sterror[P]
                        """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                        bb_dec = abs((sdx+xbar)*coef)
                        """ bb_dec is the decision vector, logic is explained above"""
                        l_index_bb = argmax(bb_dec)
                        """ find index of the largest value in decision vector """
                        r_index_bb = P[l_index_bb]
                        """ find index of the variable by index of the largest value in decision vector 
                        this is the chosen variable for this node """ 
                        C1 = C + [r_index_bb]
                        coef = delete(coef,l_index_bb,0)
                        """ add the new chosen variable to the solution 
                        We also use old C, where chosen variable is not added""" 
                          
                        P1 = P[:]
                        del P1[l_index_bb]
                        """ erasing the chosen variable from the possible variables' list 
                        reminder: this stores the variables by their indexes"""
                        self.node += 1
                        """ update number of nodes visited """
                        lower2 = self.lstsq_qrl(P1+C)
                        """ calculate lower bound of the second solution where C is the old chosen variable 
                        list and p1 is the possible variable (indexes ) list where the chosen variable for 
                        this node is erased """
                        len_p -= 1
                        len_c1 = len_c +1 
                        if len_c1 == self.s:
                            """ fix here """
                            sol = self.lstsq_qr(C1)
                            L[i].put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                            L[i].put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                        else:
                            """ if the length of the chosen variable list is not equal to sparsity level, 
                            then it is lower than sparsity level. We create two new nodes where first node
                            is the node where chosen variable for this node is in the solution and the second 
                            where the chosen variable for this node is erased from the problem """
                            
                            L[i].put([low ,[P1,C1,coef,len_c1,len_p]])
                            L[i].put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
                else:
                    self.residual_squared.append(low)
                    self.indexes.append(C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.tqsize.append(L[i].qsize())
                        break
                    
                    
                
    def solve_spi(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsqi(P+C)
        lenp = len(P)
        cost = f_LI(self.A,self.b,f[0])
        q.put([cost,[P,C,f[0][0:lenp,0],len(C),lenp]])
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p )
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.best_feasible = low
                    """ register SSE, sum of squares error."""
                    self.solcoef = coef
                    """ register Least Squares coefficients of the variables in the order 
                    that is given by solset"""
                    self.solset = P+C
                    """ register the solution set variables' indexes, it is ordered and the 
                    coefficient of each variable is in the same index in solcoef"""
                    self.rem_qsize = q.qsize()
                    """ register the number of unvisited nodes """
                    break
                else:
                    C1 = C+ P[0:1]
                    P1 = P[1:len(P)]
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsqi(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """

                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsqi(C1)
                        sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                        q.put([sol_sse,[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
            else:
                """ register the solution if it is better than any previous feasible solution """
                self.best_feasible = low
                """ register SSE, sum of squares error."""
                self.solcoef = coef
                """ register Least Squares coefficients of the variables in the order 
                that is given by solset"""
                self.solset = C
                """ register the solution set variables' indexes, it is ordered and the 
                coefficient of each variable is in the same index in solcoef"""
                self.rem_qsize = q.qsize()
                """ register the number of unvisited nodes """
                break

    def solve_spim(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsqi(P+C)
        lenp = len(P)
        cost = f_LI(self.A,self.b,f[0])
        q.put([cost,[P,C,f[0][0:lenp,0],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p )
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    C1 = C+ P[0:1]
                    P1 = P[1:len(P)]
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsqi(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """

                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsqi(C1)
                        sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                        q.put([sol_sse,[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break
            
    def solve_spi_mk2(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsqi(P+C)
        cost = f_LI(self.A,self.b,f[0])
        lenp = len(P)
        q.put([cost,[P,C,f[0][0:lenp,0],len(C),lenp]])
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.best_feasible = low
                    """ register SSE, sum of squares error."""
                    self.solcoef = coef
                    """ register Least Squares coefficients of the variables in the order 
                    that is given by solset"""
                    self.solset = P+C
                    """ register the solution set variables' indexes, it is ordered and the 
                    coefficient of each variable is in the same index in solcoef"""
                    self.rem_qsize = q.qsize()
                    """ register the number of unvisited nodes """
                    break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    bb_dec = abs(xbar*coef)
                    """ bb_dec is the decision vector, logic is explained above"""
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                     
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsqi(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """

                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsqi(C1)
                        sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                        q.put([sol_sse,[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p,0],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p,0],len_c,len_p]])
            else:
                """ register the solution if it is better than any previous feasible solution """
                self.best_feasible = low
                """ register SSE, sum of squares error."""
                self.solcoef = coef
                """ register Least Squares coefficients of the variables in the order 
                that is given by solset"""
                self.solset = C
                """ register the solution set variables' indexes, it is ordered and the 
                coefficient of each variable is in the same index in solcoef"""
                self.rem_qsize = q.qsize()
                """ register the number of unvisited nodes """
                break

    def solve_spi_mk2m(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsqi(P+C)
        cost = f_LI(self.A,self.b,f[0])
        lenp = len(P)
        q.put([cost,[P,C,f[0][0:lenp,0],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    bb_dec = abs(xbar*coef)
                    """ bb_dec is the decision vector, logic is explained above"""
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                     
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsqi(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """

                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsqi(C1)
                        sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                        q.put([sol_sse,[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p,0],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p,0],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break
            
            
    def solve_spi_mk3(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsqi(P+C)
        cost = f_LI(self.A,self.b,f[0])
        lenp = len(P)
        q.put([cost,[P,C,f[0][0:lenp,0],len(C),lenp]])
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p )
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.best_feasible = low
                    """ register SSE, sum of squares error."""
                    self.solcoef = coef
                    """ register Least Squares coefficients of the variables in the order 
                    that is given by solset"""
                    self.solset = P+C
                    """ register the solution set variables' indexes, it is ordered and the 
                    coefficient of each variable is in the same index in solcoef"""
                    self.rem_qsize = q.qsize()
                    """ register the number of unvisited nodes """
                    break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    sdx = self.sterror[P]
                    """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                    
                    bb_dec = abs((sdx+xbar)*coef)
                    """ bb_dec is the decision vector, logic is explained above"""
                    
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsqi(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """

                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsqi(C1)
                        sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                        q.put([sol_sse,[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p,0],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p,0],len_c,len_p]])
            else:
                """ register the solution if it is better than any previous feasible solution """
                self.best_feasible = low
                """ register SSE, sum of squares error."""
                self.solcoef = coef
                """ register Least Squares coefficients of the variables in the order 
                that is given by solset"""
                self.solset = C
                """ register the solution set variables' indexes, it is ordered and the 
                coefficient of each variable is in the same index in solcoef"""
                self.rem_qsize = q.qsize()
                """ register the number of unvisited nodes """
                break

    def solve_spi_mk3m(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsqi(P+C)
        cost = f_LI(self.A,self.b,f[0])
        lenp = len(P)
        q.put([cost,[P,C,f[0][0:lenp,0],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p )
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    sdx = self.sterror[P]
                    """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                    
                    bb_dec = abs((sdx+xbar)*coef)
                    """ bb_dec is the decision vector, logic is explained above"""
                    
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsqi(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """

                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsqi(C1)
                        sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                        q.put([sol_sse,[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p,0],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p,0],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break       
                
    def solve_spi_mk4(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsqi(P+C)
        cost = f_LI(self.A,self.b,f[0])
        lenp = len(P)
        q.put([cost,[P,C,f[0][0:lenp,0],len(C),lenp]])
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p)
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.best_feasible = low
                    """ register SSE, sum of squares error."""
                    self.solcoef = coef
                    """ register Least Squares coefficients of the variables in the order 
                    that is given by solset"""
                    self.solset = P+C
                    """ register the solution set variables' indexes, it is ordered and the 
                    coefficient of each variable is in the same index in solcoef"""
                    self.rem_qsize = q.qsize()
                    """ register the number of unvisited nodes """
                    break
                else:
                    cov = inv(self.cov[P+C,:][:,P+C])
                    var = diagonal(cov)[0:len_p]
                    bb_dec = coef/var
                     
                     
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsqi(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """

                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsqi(C1)
                        sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                        q.put([sol_sse,[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p,0],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p,0],len_c,len_p]])
            else:
                """ register the solution if it is better than any previous feasible solution """
                self.best_feasible = low
                """ register SSE, sum of squares error."""
                self.solcoef = coef
                """ register Least Squares coefficients of the variables in the order 
                that is given by solset"""
                self.solset = C
                """ register the solution set variables' indexes, it is ordered and the 
                coefficient of each variable is in the same index in solcoef"""
                self.rem_qsize = q.qsize()
                """ register the number of unvisited nodes """
                break

    def solve_spi_mk4m(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsqi(P+C)
        cost = f_LI(self.A,self.b,f[0])
        lenp = len(P)
        q.put([cost,[P,C,f[0][0:lenp,0],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p)
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    cov = inv(self.cov[P+C,:][:,P+C])
                    var = diagonal(cov)[0:len_p]
                    bb_dec = coef/var
                     
                     
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                      
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 =self.lstsqi(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """

                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsqi(C1)
                        sol_sse = f_LI(self.A[:,C1],self.b,sol[0])
                        q.put([sol_sse,[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p,0],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p,0],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break
            

    def solve_sp_qr(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        f = self.qr_lstsq(P+C)
        q.put([f[1],[P,C,f[0],len(C),len(P)]])
        while q.qsize() > 0:
            [low,[P,C,coef,len_c,len_p]] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p )
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.best_feasible = low
                    """ register SSE, sum of squares error."""
                    self.solcoef = coef
                    """ register Least Squares coefficients of the variables in the order 
                    that is given by solset"""
                    self.solset = P+C
                    """ register the solution set variables' indexes, it is ordered and the 
                    coefficient of each variable is in the same index in solcoef"""
                    self.rem_qsize = q.qsize()
                    """ register the number of unvisited nodes """
                    break
                else:
                    C1 = C+ P[0:1]
                    P1 = P[1:len(P)]
                     
                    self.node += 1
                    lower2 = self.qr_lstsq(P1+C)
                    lower2[1] = lower2[1]
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                    else:
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
            else:
                if low <= self.best_feasible:
                    self.best_feasible = low
                    self.solcoef = coef
                    self.solset = C
                    self.rem_qsize = q.qsize()
                    break

    def solve_sp_qrm(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        f = self.qr_lstsq(P+C)
        q.put([f[1],[P,C,f[0],len(C),len(P)]])
        count_best = 0
        while q.qsize() > 0:
            [low,[P,C,coef,len_c,len_p]] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p )
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    C1 = C+ P[0:1]
                    P1 = P[1:len(P)]
                     
                    self.node += 1
                    lower2 = self.qr_lstsq(P1+C)
                    lower2[1] = lower2[1]
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                    else:
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break

    def solve_sp_qr_svdm(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        f = self.lstsq_qr(P+C)
        q.put([f[1],[P,C,f[0],len(C),len(P)]])
        count_best = 0
        while q.qsize() > 0:
            [low,[P,C,coef,len_c,len_p]] = q.get()
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p )
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    C1 = C+ P[0:1]
                    P1 = P[1:len(P)]
                     
                    self.node += 1
                    lower2 = self.lstsq_qr(P1+C)
                    lower2[1] = lower2[1]
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq_qr(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                    else:
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break
                
    def solve_sp_qr_mk2(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.qr_lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.best_feasible = low
                    """ register SSE, sum of squares error."""
                    self.solcoef = coef
                    """ register Least Squares coefficients of the variables in the order 
                    that is given by solset"""
                    self.solset = P+C
                    """ register the solution set variables' indexes, it is ordered and the 
                    coefficient of each variable is in the same index in solcoef"""
                    self.rem_qsize = q.qsize()
                    """ register the number of unvisited nodes """
                    break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    bb_dec = abs(xbar*coef)
                    """ bb_dec is the decision vector, logic is explained above"""
                     
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                    
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 = self.qr_lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                """ register the solution if it is better than any previous feasible solution """
                self.best_feasible = low
                """ register SSE, sum of squares error."""
                self.solcoef = coef
                """ register Least Squares coefficients of the variables in the order 
                that is given by solset"""
                self.solset = C
                """ register the solution set variables' indexes, it is ordered and the 
                coefficient of each variable is in the same index in solcoef"""
                self.rem_qsize = q.qsize()
                """ register the number of unvisited nodes """
                break

    def solve_sp_qr_mk2m(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.qr_lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    bb_dec = abs(xbar*coef)
                    """ bb_dec is the decision vector, logic is explained above"""
                     
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                    
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 = self.qr_lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break

    def solve_sp_qr_svd_mk2m(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsq_qr(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    bb_dec = abs(xbar*coef)
                    """ bb_dec is the decision vector, logic is explained above"""
                     
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                    
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 = self.lstsq_qr(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq_qr(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break
                
    def solve_sp_qr_mk3(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f =  self.qr_lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            #print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.best_feasible = low
                    """ register SSE, sum of squares error."""
                    self.solcoef = coef
                    """ register Least Squares coefficients of the variables in the order 
                    that is given by solset"""
                    self.solset = P+C
                    """ register the solution set variables' indexes, it is ordered and the 
                    coefficient of each variable is in the same index in solcoef"""
                    self.rem_qsize = q.qsize()
                    """ register the number of unvisited nodes """
                    break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    sdx = self.sterror[P]
                    """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                    bb_dec = abs((sdx+xbar)*coef)
                    """ bb_dec is the decision vector, logic is explained above"""
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 = self.qr_lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                """ register the solution if it is better than any previous feasible solution """
                self.best_feasible = low
                """ register SSE, sum of squares error."""
                self.solcoef = coef
                """ register Least Squares coefficients of the variables in the order 
                that is given by solset"""
                self.solset = C
                """ register the solution set variables' indexes, it is ordered and the 
                coefficient of each variable is in the same index in solcoef"""
                self.rem_qsize = q.qsize()
                """ register the number of unvisited nodes """
                break
            
            
    def solve_sp_qr_mk3m(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f =  self.qr_lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    sdx = self.sterror[P]
                    """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                    bb_dec = abs((sdx+xbar)*coef)
                    """ bb_dec is the decision vector, logic is explained above"""
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 = self.qr_lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break

    def solve_sp_qr_svd_mk3m(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f =  self.lstsq_qr(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    sdx = self.sterror[P]
                    """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                    bb_dec = abs((sdx+xbar)*coef)
                    """ bb_dec is the decision vector, logic is explained above"""
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 = self.lstsq_qr(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq_qr(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break
                
    def solve_sp_qr_mk4(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.qr_lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.best_feasible = low
                    """ register SSE, sum of squares error."""
                    self.solcoef = coef
                    """ register Least Squares coefficients of the variables in the order 
                    that is given by solset"""
                    self.solset = P+C
                    """ register the solution set variables' indexes, it is ordered and the 
                    coefficient of each variable is in the same index in solcoef"""
                    self.rem_qsize = q.qsize()
                    """ register the number of unvisited nodes """
                    break
                else:
                    cov = inv(self.cov[P+C,:][:,P+C])
                    var = diagonal(cov)[0:len_p]
                    bb_dec = coef/var
                     
                     
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                   
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 = self.qr_lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                """ register the solution if it is better than any previous feasible solution """
                self.best_feasible = low
                """ register SSE, sum of squares error."""
                self.solcoef = coef
                """ register Least Squares coefficients of the variables in the order 
                that is given by solset"""
                self.solset = C
                """ register the solution set variables' indexes, it is ordered and the 
                coefficient of each variable is in the same index in solcoef"""
                self.rem_qsize = q.qsize()
                """ register the number of unvisited nodes """
                break

    def solve_sp_qr_mk4m(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.qr_lstsq(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    cov = inv(self.cov[P+C,:][:,P+C])
                    var = diagonal(cov)[0:len_p]
                    bb_dec = coef/var
                     
                     
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                   
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 = self.qr_lstsq(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_lstsq(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break

    def solve_sp_qr_svd_mk4m(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be LSE coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This is the suggested algoritm for general use. If you have any other purpose, experimenting and
        etc. you can use any algorithm given above.
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.lstsq_qr(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        """ initialization of the first problem """
        count_best = 0
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-1:])
            if len_c < self.s:
                if len_p + len_c <= self.s:
                    self.residual_squared.append(low)
                    self.indexes.append(P+C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize = q.qsize()
                        break
                else:
                    cov = inv(self.cov[P+C,:][:,P+C])
                    var = diagonal(cov)[0:len_p]
                    bb_dec = coef/var
                     
                     
                     
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]
                    
                    C1 = C + [r_index_bb]
                    coef = delete(coef,l_index_bb,0)
                   
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 = self.lstsq_qr(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.lstsq_qr(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0][0:len_p],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0][0:len_p],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize = q.qsize()
                    break
                
          
    def solve(self,C = []):
        
        if not type(C) == list and C != []:
            print("C should be a list, C is taken empty list now ")
            C = []
        elif C != [] and (max(C) >= self.n or min(C) < 0):
            print("Values of C should be valid, in the range 0 <= n-1, C is taken empty list")
            C = []
        elif len(set(C)) != len(C):
            print("Values of C should be unique, C is taken empty list")
            C = []
        elif len(C) > self.s:
            print(" Length of C cannot be greater than spartsity level s,C is taken empty list")
            C = []
            
        if self.enumerate not in ["m-st-lsc","m-lsc","lexi","stat"]:
            print("Invalid choice of enumeration")
            return None
        elif self.search not in ["best","depth","breadth"]:
            print("Invalid choice of search")
            return None
        elif self.solver not in ["qr","svd"]:
            print("Invalid choice of solver")
            return None
        elif type(self.iter) != int:
            print("Iteration should be an integer")
            return None
        elif self.out not in [0,1,2]:
            print("OUT parameter should be a integer >=0  and <= 2")
            return None
        elif type(self.initial_point) != ndarray or shape(self.initial_point) != (self.n,1):
            print("initial_point should be an array of size (self.n,1) not (self.n,) careful with \
python syntax of arrays")
            return None

        mem = hpy()
        """ memory object """
        mem.heap()
        """ check the objects that are in memory right now """
        mem.setrelheap()
        """ referencing this point, memory usage will be calculated """
        t0 = time.process_time()
        """ referencing this point, cpu usage will be calculated """
        
        if self.out != 2:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout    
        """ whether user wants to print every step of the algorithm or not """
        
        
        """
        if self.heur == True:
            [Xgreed,funval_greed] = greedy_sparse_simplex(f_LI,g_LI,self.s,self.iter,\
            self.initial_point,self.A,self.b)
            
            xgreed_coef = Xgreed[:,[-1]]
            
            [Xpartial,funval_partial] = partial_sparse_simplex(f_LI,gradient_LI,g_LI,self.s,self.iter,\
            self.initial_point,self.A,self.b)
            
            xpartial_coef = Xpartial[:,[-1]]  
            
            if funval_greed <= funval_partial:
                self.best_heuristic = funval_greed
                self.coef_heuristic = xgreed_coef
                self.best_feasible = funval_greed
            else:
                self.best_heruistic = funval_partial
                self.coef_heuristic = xpartial_coef
                self.best_feasible = funval_partial
                
                
                benched heuristic approch
        """    
        
        """ so many if conditions below to find and call the right function """
                
        if self.enumerate == "m-st-lsc" and self.search == "best" \
        and self.solver == "svd" and self.ill == False and C == []:
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_sp_mk3(list(range(self.n)))
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "best" \
        and self.solver == "svd" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_sp_mk3(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "best" \
        and self.solver == "svd" and self.ill == True and C == []:
            
            self.solve_spi_mk3(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "best" \
        and self.solver == "svd" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_spi_mk3(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "m-st-lsc" and self.search == "best" \
        and self.solver == "qr" and self.ill == False and C == []:
            
            self.solve_sp_qr_mk3(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "best" \
        and self.solver == "qr" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_sp_qr_mk3(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "best" \
        and self.solver == "qr" and self.ill == True and C == []:
            
            self.solve_sp_qr_mk3(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "best" \
        and self.solver == "qr" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_sp_qr_mk3(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "m-st-lsc" and self.search == "depth" \
        and self.solver == "svd" and self.ill == False and C == []:
            
            f = self.lstsq(P+C)
            low = f[1][0]
            coef = f[0][0:len_p,0]
            self.solve_dfs_mk3(list(range(self.n)),[],low,coef,0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "depth" \
        and self.solver == "svd" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.lstsq(P+C)
            low = f[1][0]
            coef = f[0][0:len_p,0]
            self.solve_dfs_mk3(P,C,low,coef,len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "depth" \
        and self.solver == "svd" and self.ill == True and C == []:
            
            f = self.lstsqi(P+C)
            self.solve_dfsi_mk3(list(range(self.n)),[],f[1],f[0][0:len(P)],0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "depth" \
        and self.solver == "svd" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.lstsqi(P+C)
            self.solve_dfsi_mk3(P,C,f[1],f[0][0:len(P)],len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "m-st-lsc" and self.search == "depth" \
        and self.solver == "qr" and self.ill == False and C == []:
            
            P = list(range(self.n))
            f = self.qr_lstsq(P+C)
            low = f[1]
            coef = f[0][0:len(P)]
            self.solve_dfs_qr_mk3(list(range(self.n)),[],low,coef,0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "depth" \
        and self.solver == "qr" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            C = list(C)
            f = self.qr_lstsq(P+C)
            low = f[1]
            coef = f[0][0:len(P)]
            self.solve_dfs_qr_mk3(P,C,low,coef,len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "depth" \
        and self.solver == "qr" and self.ill == True and C == []:
            
            P = list(range(self.n))
            f = self.qr_lstsq(P+C)
            self.solve_dfs_qr_mk3(list(range(self.n)),[],f[1],f[0][0:len(P)],0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "depth" \
        and self.solver == "qr" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            C = list(C)
            f = self.qr_lstsq(P+C)
            self.solve_dfs_qr_mk3(P,C,f[1],f[0][0:len(P)],len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "m-st-lsc" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == False and C == []:
            
            self.solve_bfs_mk3(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfs_mk3(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == True and C == []:
            
            self.solve_bfsi_mk3(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfsi_mk3(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "m-st-lsc" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == False and C == []:
            
            self.solve_bfs_qr_mk3(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfs_qr_mk3(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == True and C == []:
            
            self.solve_bfs_qr_mk3(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-st-lsc" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfs_qr_mk3(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "m-lsc" and self.search == "best" \
        and self.solver == "svd" and self.ill == False and C == []:
            
            self.solve_sp_mk2(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "best" \
        and self.solver == "svd" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_sp_mk2(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "best" \
        and self.solver == "svd" and self.ill == True and C == []:
            
            self.solve_spi_mk2(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "best" \
        and self.solver == "svd" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_spi_mk2(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "m-lsc" and self.search == "best" \
        and self.solver == "qr" and self.ill == False and C == []:
            
            self.solve_sp_qr_mk2(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "best" \
        and self.solver == "qr" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_sp_qr_mk2(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "best" \
        and self.solver == "qr" and self.ill == True and C == []:
            
            self.solve_sp_qr_mk2(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "best" \
        and self.solver == "qr" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_sp_qr_mk2(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "m-lsc" and self.search == "depth" \
        and self.solver == "svd" and self.ill == False and C == []:
            
            f = self.lstsq(P+C)
            low = f[1][0]
            coef = f[0][0:len_p,0]
            self.solve_dfs_mk2(list(range(self.n)),[],low,coef,0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "depth" \
        and self.solver == "svd" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.lstsq(P+C)
            low = f[1][0]
            coef = f[0][0:len_p,0]
            self.solve_dfs_mk2(P,C,low,coef,len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "depth" \
        and self.solver == "svd" and self.ill == True and C == []:
            
            f = self.lstsqi(P+C)
            self.solve_dfsi_mk2(list(range(self.n)),[],f[1],f[0][0:len(P)],0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "depth" \
        and self.solver == "svd" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.lstsqi(P+C)
            self.solve_dfsi_mk2(P,C,f[1],f[0][0:len(P)],len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "m-lsc" and self.search == "depth" \
        and self.solver == "qr" and self.ill == False and C == []:
            
            P = list(range(self.n))
            f = self.qr_lstsq(P+C)
            low = f[1]
            coef = f[0][0:len(P)]
            self.solve_dfs_qr_mk2(list(range(self.n)),[],low,coef,0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "depth" \
        and self.solver == "qr" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.qr_lstsq(P+C)
            low = f[1]
            coef = f[0][0:len(P)]
            self.solve_dfs_qr_mk2(P,C,low,coef,len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "depth" \
        and self.solver == "qr" and self.ill == True and C == []:
            
            P = list(range(self.n))
            f =  self.qr_lstsq(P+C)
            self.solve_dfs_qr_mk2(list(range(self.n)),[],f[1],f[0][0:len(P)],0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "depth" \
        and self.solver == "qr" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.qr_lstsq(P+C)
            self.solve_dfs_qr_mk2(P,C,f[1],f[0][0:len(P)],len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "m-lsc" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == False and C == []:
            
            self.solve_bfs_mk2(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfs_mk2(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == True and C == []:
            
            self.solve_bfsi_mk2(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfsi_mk2(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "m-lsc" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == False and C == []:
            
            self.solve_bfs_qr_mk2(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfs_qr_mk2(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == True and C == []:
            
            self.solve_bfs_qr_mk2(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "m-lsc" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfs_qr_mk2(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]


        elif  self.enumerate == "lexi" and self.search == "best" \
        and self.solver == "svd" and self.ill == False and C == []:
            
            self.solve_sp(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "best" \
        and self.solver == "svd" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_sp(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "best" \
        and self.solver == "svd" and self.ill == True and C == []:
            
            self.solve_spi(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "best" \
        and self.solver == "svd" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_spi(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "lexi" and self.search == "best" \
        and self.solver == "qr" and self.ill == False and C == []:
            
            self.solve_sp_qr(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "best" \
        and self.solver == "qr" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_sp_qr(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "best" \
        and self.solver == "qr" and self.ill == True and C == []:
            
            self.solve_sp_qr(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "best" \
        and self.solver == "qr" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_sp_qr(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "lexi" and self.search == "depth" \
        and self.solver == "svd" and self.ill == False and C == []:
            
            f = self.lstsq(P+C)
            low = f[1][0]
            coef = f[0][0:len_p,0]
            self.solve_dfs(list(range(self.n)),[],low,coef,0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "depth" \
        and self.solver == "svd" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.lstsq(P+C)
            low = f[1][0]
            coef = f[0][0:len_p,0]
            self.solve_dfs(P,C,low,coef,len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "depth" \
        and self.solver == "svd" and self.ill == True and C == []:
            
            f = self.lstsqi(P+C)
            low = f_LI(self.A,self.b,f[0])
            coef = f[0][0:len_p,0]
            self.solve_dfsi(list(range(self.n)),[],low,coef,0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "depth" \
        and self.solver == "svd" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.lstsqi(P+C)
            low = f_LI(self.A,self.b,f[0])
            coef = f[0][0:len_p,0]
            self.solve_dfsi(P,C,low,coef,len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "lexi" and self.search == "depth" \
        and self.solver == "qr" and self.ill == False and C == []:
            
            P = list(range(self.n))
            f = self.qr_lstsq(P+C)
            low = f[1]
            coef = f[0][0:len(P)]
            self.solve_dfs_qr(list(range(self.n)),[],low,coef,0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "depth" \
        and self.solver == "qr" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.qr_lstsq(P+C)
            low = f[1]
            coef = f[0][0:len(P)]
            self.solve_dfs_qr(P,C,low,coef,len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "depth" \
        and self.solver == "qr" and self.ill == True and C == []:
            
            P = list(range(self.n))
            f = self.qr_lstsq(P+C)
            self.solve_dfs_qr(list(range(self.n)),[],f[1],f[0][0:len(P)],0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "depth" \
        and self.solver == "qr" and self.ill == True and C != []:
            
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.lstsqi(P+C)
            self.solve_dfs_qr(P,C,f[1],f[0][0:len(P)],len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "lexi" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == False and C == []:
            
            self.solve_bfs(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfs(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == True and C == []:
            
            self.solve_bfsi(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfsi(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "lexi" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == False and C == []:
            
            self.solve_bfs_qr(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfs_qr(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == True and C == []:
            
            self.solve_bfs_qr(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "lexi" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfs_qr(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "stat" and self.search == "best" \
        and self.solver == "svd" and self.ill == False and C == []:
            
            self.solve_sp_mk4(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "best" \
        and self.solver == "svd" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_sp_mk4(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "best" \
        and self.solver == "svd" and self.ill == True and C == []:
            
            self.solve_spi_mk4(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "best" \
        and self.solver == "svd" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_spi_mk4(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "stat" and self.search == "best" \
        and self.solver == "qr" and self.ill == False and C == []:
            
            self.solve_sp_qr_mk4(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "best" \
        and self.solver == "qr" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_sp_qr_mk4(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "best" \
        and self.solver == "qr" and self.ill == True and C == []:
            
            self.solve_sp_qr_mk4(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "best" \
        and self.solver == "qr" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_sp_qr_mk4(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "stat" and self.search == "depth" \
        and self.solver == "svd" and self.ill == False and C == []:
            
            f = self.lstsq(P+C)
            low = f[1][0]
            coef = f[0][0:len_p,0]
            self.solve_dfs_mk4(list(range(self.n)),[],low,coef,0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "depth" \
        and self.solver == "svd" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.lstsq(P+C)
            low = f[1][0]
            coef = f[0][0:len_p,0]
            self.solve_dfs_mk4(P,C,low,coef,len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "depth" \
        and self.solver == "svd" and self.ill == True and C == []:
            
            f = self.lstsqi(P+C)
            self.solve_dfsi_mk4(list(range(self.n)),[],f[1],f[0][0:len(P)],0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "depth" \
        and self.solver == "svd" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.lstsqi(P+C)
            self.solve_dfsi_mk4(P,C,f[1],f[0][0:len(P)],len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "stat" and self.search == "depth" \
        and self.solver == "qr" and self.ill == False and C == []:
            
            P = list(range(self.n))
            f = self.qr_lstsq(P+C)
            low = f[1]
            coef = f[0][0:len(P)]
            self.solve_dfs_qr_mk4(f[3],list(range(self.n)),[],low,coef,0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "depth" \
        and self.solver == "qr" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.qr_lstsq(P+C)
            low = f[1]
            coef = f[0][0:len(P)]
            self.solve_dfs_qr_mk4(f[3],P,C,low,coef,len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "depth" \
        and self.solver == "qr" and self.ill == True and C == []:
            
            P = list(range(self.n))
            f = self.qr_lstsq(P+C)
            self.solve_dfs_qr_mk4(f[3],list(range(self.n)),[],f[1],f[0][0:len(P)],0,self.n)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "depth" \
        and self.solver == "qr" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            f = self.qr_lstsq(P+C)
            self.solve_dfs_qr_mk4(f[3],P,C,f[1],f[0][0:len(P)],len(C),self.n-len(C))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "stat" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == False and C == []:
            
            self.solve_bfs_mk4(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfs_mk4(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == True and C == []:
            
            self.solve_bfsi_mk4(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "breadth" \
        and self.solver == "svd" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfsi_mk4(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]

        elif  self.enumerate == "stat" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == False and C == []:
            
            self.solve_bfs_qr_mk4(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == False and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfs_qr_mk4(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == True and C == []:
            
            self.solve_bfs_qr_mk4(list(range(self.n)))
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.best_feasible,self.solset,self.node]
        
        elif  self.enumerate == "stat" and self.search == "breadth" \
        and self.solver == "qr" and self.ill == True and C != []:
            
            P = list(range(self.n))
            for i in range(len(C)):
                P.remove(C[i])
            self.solve_bfs_qr_mk4(P,C)
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            
            self.memory = m.size
             
            x_vector = zeros([self.n,1])
            x_vector[self.solset,0] = self.solcoef
            self.x_vector = x_vector
            sys.stdout = self.original_stdout
            U,D,V = svd(self.A[:,self.solset])
            error = (D[0]/D[-1] + self.best_feasible ** 0.5 * (D[0]/D[-1]) ** 2) * (finfo(float64).eps/2)
            precision = -1*log10(error)
            print("Error bound on the norm of the x vector",error,"Number of correct digits of x ",precision.real )
            return [self.solcoef,self.bdest_feasible,self.solset,self.node]
        
    def solve_multiple(self,C = []):
        
        """ For performance concerns, only the best algorithms are offered here to use to find best k subsets for sparsity level s"""
        
        if not type(C) == list and C != []:
            print("C should be a list, C is taken empty list now ")
            C = []
        elif C != [] and (max(C) >= self.n or min(C) < 0):
            print("Values of C should be valid, in the range 0 <= n-1, C is taken empty list")
            C = []
        elif len(set(C)) != len(C):
            print("Values of C should be unique, C is taken empty list")
            C = []
        elif len(C) > self.s:
            print(" Length of C cannot be greater than spartsity level s,C is taken empty list")
            C = []
            
        if self.enumerate not in ["m-st-lsc","m-lsc","lexi","stat"]:
            print("Invalid choice of enumeration")
            return None
        elif self.solver not in ["qr","svd"]:
            print("Invalid choice of solver")
            return None
        elif self.out not in [0,1,2]:
            print("OUT parameter should be a integer >=0  and <= 2")
            return None

        elif self.many >= math.factorial(self.n)/(math.factorial(self.s)*math.factorial(self.n-self.s)):
            print("Reduce number of best subsets you want to find, it is greater than  or equal to all possibilities")
            return None
        
        mem = hpy()
        """ memory object """
        mem.heap()
        """ check the objects that are in memory right now """
        mem.setrelheap()
        """ referencing this point, memory usage will be calculated """
        t0 = time.process_time()
        """ referencing this point, cpu usage will be calculated """
        
        if self.out != 2:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout    
        """ whether user wants to print every step of the algorithm or not """
        
        P = list(range(self.n))
        if C != []:
            for i in range(len(C)):
                P.remove(C[i])
        """ Preparing the P and C that will be passed to the function """
        
        """ Another if list to find and call the right function """
        
        if  self.enumerate == "m-st-lsc" and self.solver == "qr":
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_sp_qr_mk3m(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
        elif  self.enumerate == "m-lsc" and self.solver == "qr":
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_sp_qr_mk2m(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
        elif  self.enumerate == "lexi" and self.solver == "qr":
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_sp_qrm(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
        elif  self.enumerate == "stat" and self.solver == "qr":
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_sp_qr_mk4m(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
        elif  self.enumerate == "m-st-lsc" and self.solver == "svd" and self.ill == False:
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_sp_mk3m(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
        elif  self.enumerate == "m-lsc" and self.solver == "svd" and self.ill == False:
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_sp_mk2m(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
        elif  self.enumerate == "lexi" and self.solver == "svd" and self.ill == False:
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_spm(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
            
        elif  self.enumerate == "stat" and self.solver == "svd" and self.ill == False:

            t2 = time.process_time()
            t3 = time.time()
            self.solve_sp_mk4m(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
        elif  self.enumerate == "m-st-lsc" and self.solver == "svd" and self.ill == True:
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_spi_mk3m(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
        elif  self.enumerate == "m-lsc" and self.solver == "svd" and self.ill == True:
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_spi_mk2m(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
            
        elif  self.enumerate == "lexi" and self.solver == "svd" and self.ill == True:
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_spim(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
        elif  self.enumerate == "stat" and self.solver == "svd" and self.ill == True:
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_spi_mk4m(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]

        elif  self.enumerate == "m-st-lsc" and self.solver == "qr-svd" and self.ill == False:
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_sp_qr_svd_mk3m(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
        elif  self.enumerate == "m-lsc" and self.solver == "qr-svd" and self.ill == False:
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_sp_qr_svd_mk2m(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
        elif  self.enumerate == "lexi" and self.solver == "qr-svd" and self.ill == False:
            
            t2 = time.process_time()
            t3 = time.time()
            self.solve_sp_qr_svdm(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
            
        elif  self.enumerate == "stat" and self.solver == "qr-svd" and self.ill == False:

            t2 = time.process_time()
            t3 = time.time()
            self.solve_sp_qr_svd_mk4m(P,C)
            t4 = time.time()
            finish = time.process_time()
            duration = finish-t0
            self.cpu = duration
            if self.out == 0:
                sys.stdout = open(os.devnull, 'w')
            else:
                sys.stdout = self.original_stdout
            print("CPU time of the algorithm",duration,"seconds")
            m = mem.heap()
            print(m)
            self.memory = m.size
            """ real memory usage is different than the number we store here because we use guppy package 
            to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
            process """
             
            sys.stdout = self.original_stdout
            
            return [self.residual_squared,self.indexes,self.coefficients]
        
    def solve_allsubsets(self):
        
        """ For performance concerns, only the best algorithm is offered to use for finding all best k subsets for sparsity level
        from 1 to s, """

        if self.many >= self.n:
            print("Reduce number of best subsets you want to find, it is greater than  or equal to all possibilities")
            return None
        
        mem = hpy()
        """ memory object """
        mem.heap()
        """ check the objects that are in memory right now """
        mem.setrelheap()
        """ referencing this point, memory usage will be calculated """
        t0 = time.process_time()
        """ referencing this point, cpu usage will be calculated """
        
        if self.out != 2:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout    
        """ whether user wants to print every step of the algorithm or not """
            
        P = list(range(self.n))
        C = []
        """ Preparing the P and C that will be passed to the function """
        
        
        t2 = time.process_time()
        t3 = time.time()
        self.solve_sp_mk0s(P,C)
        """ mk0 also works, but this is in general faster """
        t4 = time.time()
        finish = time.process_time()
        duration = finish-t0
        self.cpu = duration
        if self.out == 0:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout   
        print("CPU time of the algorithm",duration,"seconds")
        m = mem.heap()
        print(m)
        self.memory = m.size
        """ real memory usage is different than the number we store here because we use guppy package 
        to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
        process """
         
        sys.stdout = self.original_stdout
        
        return [self.residual_squared,self.indexes,self.coefficients]        

            
