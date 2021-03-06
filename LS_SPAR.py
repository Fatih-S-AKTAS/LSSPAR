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
        
        This Python Code implements the Algorithms described in the paper: Fatih Selim AKTAŞ, Mustafa Çelebi
        Pınar,"Provably Optimal Sparse Solutions to Overdetermined Linear Systems by Implicit Enumeration"
        
        If you get a pythonic error, it will probably because of how unfriendly is python with usage of 
        vectors, like lists cannot be sliced with indeces but arrays can be done, hstack converts everything
        to floating points, registering a vector and randomly trying to make it 2 dimensional takes too much
        unnecessary work, some functions do inverse of hstack, randomly converting numbers to integers etc.
        
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
            self.means = abs(mean(A,axis = 0))
            """ storing the mean values of each independent variable which will be used for 
            selection criteria in solver algortihms"""
            self.sterror = std(A,axis = 0)
            """ storing the standard deviation and variance of each independent variable which will
            be used for selection criteria in solver algortihms"""
            self.rem_qsize = 0
            """ initializing remaining queue size after the algorithm finishes """
            self.out = 1
            """ initializing the output choice of algorithm """
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
            
            """
            
            functions are named in the following structure:
            
            solve_(search type)(ill conditioned solver or not)_(least squares solver)_(enumeration type)(multiple subsets or not)
            
            
            search types: (bfs) breadth-first-search,  (dfs) depth-first-search, (sp) best-first-search,
            
            when the matrix is extremely ill conditioned, a slight modificiation is required to calculate the residual of the 
            system  because of how numpy's lstsq work.
            
            least squares solvers: (qr) QR decomposition, (svd) Singular Value Decomposition
            
            enumeration types: () lexicographical enumeration, (mk2)  "m-lsc", (mk3) "m-st-lsc", (mk4) statistical significance
            
            for multiple subsets simply termination condition is changed.
            
            
            functions with mk0 name are special functions that are developed for all subsets problem.
            
            
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


        P_0 = P[:]
        C_0 = C[:]
        """ saving the original P and C """
        
        f = self.qr_lstsq(P+C)
        self.tqsize = []
        lenp = len(P)
        lenc = len(C)

        s = self.s
        i = 0
        while i < s:
            i += 1
            self.s = i
            print("Started solving for sparsity level ",self.s)
            q = PriorityQueue()
            q.put([f[1],[P_0,C_0,f[0][lenc:],lenc,lenp]])
            """ starting from scratch with tablelookups """
            count_best = 0
            self.node = 0
            while True:
                """ termination condition of the problem if we visit all the nodes then search is over """
                [low,[P,C,coef,len_c,len_p]] = q.get()
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
                        self.tqsize.append(L[i].qsize())
                        break

                    
    def solve_sp_mk0s(self,P ,C = []):
        """ 
        
        This algorithm finds all subsets of size 1 to s, after initial qr, svd is used to find least squares solution, it is faster because
        q,r requires q to be formed explicitly which makes it slower. Best first search and m-st-lsc is used.
        
        """

        P_0 = P[:]
        C_0 = C[:]
        """ saving the original P and C """
        
        f = self.lstsq_qr(P+C)
        self.tqsize = []
        lenp = len(P)
        lenc = len(C)

        """ initialization of the first problem """
        s = self.s
        i = 0
        while i < s:
            i += 1
            self.s = i
            print("Started solving for sparsity level ",self.s)
            q = PriorityQueue()
            q.put([f[1],[P_0,C_0,f[0][lenc:],lenc,lenp]])
            """ starting from scratch with tablelookups """
            count_best = 0
            self.node = 0
            while q.qsize() > 1:
                """ termination condition of the problem if we visit all the nodes then search is over """
                [low,[P,C,coef,len_c,len_p]] = q.get()
                """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
                print("lowerbound for now",low,"number of nodes",self.node,"len of chosen",len_c,"len of possible",len_p,"last chosen",C[-2:-1])
                if len_c < self.s:
                    if len_p + len_c <= self.s:
                        self.residual_squared.append(low)
                        self.indexes.append(P+C)
                        self.coefficients.append(coef)
                        count_best += 1
                        if count_best == self.many:
                            self.tqsize.append(q.qsize())
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
                        self.tqsize.append(q.qsize())
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

            
