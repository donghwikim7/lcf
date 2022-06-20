import numpy as np
import scipy as sp
from scipy.optimize import basinhopping
import models_function
from scipy.optimize import rosen, differential_evolution
import sympy as sym
class Models:
    def __init__(self, model_id, training_sizes, training_scores, sizes, go_method='basin', initailparams=None):


        self.id = model_id
        self.training_sizes = training_sizes/sizes[-1]
        self.training_scores = training_scores
        self.sizes = sizes /sizes[-1]
        self.model_function = models_function.ModelsFunction(self.id).model_func
        self.n_param = models_function.ModelsFunction(self.id).n_param
        self.go_method = go_method
        self.best_beta = np.ones(self.n_param)
        self.out_of_range_penalty = 1
        self.initailparam = None
        self.boundryy = [(-1, 0), (0, 1)]
        if self.id == 'log2':
            self.boundryy = [(-1, 0), (0, 1)]
        elif self.id == 'pow2':
            self.boundryy = [(-1, 0), (-1, 0)]
        elif self.id == 'exp3':
            self.boundryy = [(-10, 100), (-10, 100), (-100, 10)]
        elif self.id == 'pow3':
            self.boundryy = [(0, 150), (0, 150), (0, 1)]
        elif self.id == 'exp4':
            self.boundryy = [(-10, 100), (-10, 100), (-100, 10), (-5, 15)]
        elif self.id == 'pow4':
            self.boundryy = [(0, 150), (0, 150), (-5, 50), (-1, 10)]
        if initailparams == None:
            arrr = []
            for (l, h) in self.boundryy:
                arrr.append(np.random.uniform(l, h, 10))
            arrr = np.array(arrr)
            self.initailparam = arrr.T
        else:
            b = False
            for i in range(len(initailparams)):
                if initailparams[i]['model'] == self.id:
                    self.initailparam = initailparams[i]['centroids']
                    b = True
                    break
            if not b:
                self.initailparam = np.random.rand(15, self.n_param)


        #initpT = np.transpose(self.initailparam)
        #self.bounds=[]
        #for i in range(len(initpT)):
          #  self.bounds.append((np.min(initpT[i]), np.max(initpT[i])))
        #self.get_upperbb = self.get_upperb()
        self.wee = 1#self.get_weight()
        #self.fprime = lambda x: sp.optimize.approx_fprime(x, self.fitness_function, 0.001)
       # self.ffprime =lambda x: sp.optimize.approx_fprime(x, self.fprime, 0.00001)
        #self.aa = self.getJacfunc()
    def get_upperb(self):
        best = self.initailparam[0]
        for i in range(len(self.initailparam)):
            if self.fitness_function_nopenal(best)>self.fitness_function_nopenal(self.initailparam[i]):
                best = self.initailparam[i]
        return best

    def get_weight(self):
        if True:
            w = np.array([])
            maxiter = len(self.training_sizes)
            max_we = 1
            for i in range(maxiter):
                w = np.append(w, (1 + (i*0.1)))
            return w
        else:
            w = np.array([])
            maxiter = len(self.sizes)
            max_we = self.weight_coefficient
            c = 1.0
            for i in range(maxiter):
                w = np.append(w, c)
                c = c * max_we
            return w

    # predict full dataset for given model
    def predict(self):
        return self.model_function(self.best_beta, self.sizes)

    def compute_out_of_range_penalty(self, beta):
        arr1 = self.model_function(beta, self.sizes[len(self.training_sizes):])
        #print(self.training_scores[-1])
        #v = 0.1
        #arr1[((arr1) < 0)] = arr1[((arr1) < 0)] - 1 - self.training_scores[-1]
        #arr1[((arr1) >= 0)&((arr1+v ) <self.training_scores[-1])]=arr1[((arr1) >= 0)&((arr1+v )  <self.training_scores[-1])] -1 - self.training_scores[-1]
        #arr1[((arr1 +v ) > self.training_scores[-1]) & (arr1 < 1)] = 0
        p = np.minimum(0.2, (self.training_scores[-1]))
        #print(self.training_scores[-1])
        #print(self.training_scores)
        if p >1:
            p = self.training_scores[-1]
        #p=0
        p=self.training_scores[-1]
        #print( self.training_scores)
        arr1[(arr1 < p)] = (arr1[(arr1 < p)] - p) * 10
        arr1[(arr1 > p) & (arr1 < 1)] = 0

        arr1[(arr1 >1)] = (arr1[(arr1 >1)] - 1) *10

        # arr1[(arr1 < 0)] = arr1[(arr1 < 0)] - 1

        return self.out_of_range_penalty * arr1

    def fitness_function_nopenal(self, beta):
        return np.mean((
            self.model_function(beta, self.training_sizes) - self.training_scores)** 2)

    def fitness_function(self, beta):
        return np.mean(
            np.append((self.model_function(beta, self.training_sizes) - self.training_scores),
                      self.compute_out_of_range_penalty(beta)) ** 2)

    def fitness_function_for_least_square(self, beta):  # this returns the residuals of the fit on the training points
        # print(self.get_weight()*np.append((self.model_function(beta, self.training_sizes) - self.training_scores),
        #                self.compute_out_of_range_penalty(beta)))
        return np.append((self.model_function(beta, self.training_sizes) - self.training_scores),
                         self.compute_out_of_range_penalty(beta))

    def get_initial_point(self):
        fails_init = 0
        best_beta = np.random.rand(self.n_param)
        error = True
        while (error):

            if fails_init > 1000:  # give up
                best_beta = np.zeros(self.n_param)
                break

            fails_init += 1
            best_beta = np.random.rand(self.n_param)
            # check for errors in initial point
            trn_error_init = np.mean(self.fitness_function_for_least_square(best_beta) ** 2)
            fun_init = self.fitness_function_for_least_square(best_beta)
            error = np.isnan(fun_init).any() or np.isinf(fun_init).any() or np.isnan(trn_error_init).any() or np.isinf(
                trn_error_init).any()

        return best_beta

    def optimise(self):
        global best_beta_new1
        if self.id == 'last1':
            self.best_beta = np.array([self.training_scores[-1]])
            return self.best_beta
        elif True:
            for i in range(len(self.initailparam)):
               if not (np.isinf(self.fitness_function_for_least_square(self.initailparam[i])).any() | np.isnan(
                       self.fitness_function_for_least_square(self.initailparam[i])).any()):
                   best_beta_new = sp.optimize.least_squares(self.fitness_function_for_least_square,
                                                             self.initailparam[i], method="lm").x
                   if np.mean(self.fitness_function_for_least_square(best_beta_new) ** 2) < np.mean(
                            self.fitness_function_for_least_square(self.best_beta) ** 2):
                        self.best_beta = best_beta_new
#

            return self.best_beta
        elif False:
            for i in range(len(self.initailparam)):
                if not (np.isinf(self.fitness_function_for_least_square(self.initailparam[i])).any() | np.isnan(
                        self.fitness_function_for_least_square(self.initailparam[i])).any()):
                    try:
                        #print(self.fitness_function(self.initailparam[i]))
                        best_beta_new1= sp.optimize.newton(self.fitness_function, self.initailparam[i])
                        #print(111)
                        #print(best_beta_new1)
                    except Exception as e:
                        best_beta_new1 = self.best_beta
                    finally:
                        best_beta_new = best_beta_new1                   #print(best_beta_new)
                    if  (False==np.isnan(best_beta_new).any())&((np.mean(self.fitness_function_for_least_square(best_beta_new) ** 2) < np.mean(
                            self.fitness_function_for_least_square(self.best_beta) ** 2))):
                        self.best_beta = best_beta_new
            return self.best_beta
        elif False:
            for i in range(len(self.initailparam)):
                if not (np.isinf(self.fitness_function_for_least_square(self.initailparam[i])).any() | np.isnan(
                        self.fitness_function_for_least_square(self.initailparam[i])).any()):
                    try:
                        # print(self.fitness_function(self.initailparam[i]))
                        best_beta_new1 = sp.optimize.newton(self.fitness_function, self.initailparam[i])
                        # print(111)
                        # print(best_beta_new1)
                    except Exception as e:
                        best_beta_new1 = self.best_beta
                    finally:
                        best_beta_new = best_beta_new1  # print(best_beta_new)
                    if (False == np.isnan(best_beta_new).any()) & (
                    (np.mean(self.fitness_function_for_least_square(best_beta_new) ** 2) < np.mean(
                            self.fitness_function_for_least_square(self.best_beta) ** 2))):
                        self.best_beta = best_beta_new

            #
            #
            self.best_beta = sp.optimize.least_squares(self.fitness_function_for_least_square,
                                                      self.best_beta, method="lm").x
            return self.best_beta
        elif False:
            for i in range(len(self.initailparam)):
                if not (np.isinf(self.fitness_function_for_least_square(self.initailparam[i])).any() | np.isnan(
                        self.fitness_function_for_least_square(self.initailparam[i])).any()):
                    best_beta_new =  sp.optimize.minimize(self.fitness_function, self.initailparam[i], method='Newton-CG',
                                             jac=self.fprime).x
                    if np.mean(self.fitness_function_for_least_square(best_beta_new) ** 2) < np.mean(
                            self.fitness_function_for_least_square(self.best_beta) ** 2):
                        self.best_beta = best_beta_new
            #
            #
            return self.best_beta

        elif False:
            self.best_beta = differential_evolution(self.fitness_function,   bounds=self.boundryy,init=self.initailparam).x
            #self.best_beta = sp.optimize.least_squares(self.fitness_function_for_least_square,
             #                                                self.best_beta, method="lm").x
            return self.best_beta
        elif False:
            self.best_beta = differential_evolution(self.fitness_function,   bounds=self.boundryy,init=self.initailparam).x
            #self.best_beta = sp.optimize.least_squares(self.fitness_function_for_least_square,
             #                                                self.best_beta, method="lm").x
            if not (np.isnan(self.best_beta).any()|np.isinf(self.fitness_function_for_least_square(self.best_beta)).any() | np.isnan(
                    self.fitness_function_for_least_square(self.best_beta)).any()):
                self.best_beta = sp.optimize.least_squares(self.fitness_function_for_least_square,
                                                           self.best_beta, method="lm").x

            return self.best_beta

    def fitness_function(self, beta):
        if  (
            np.mean((np.append((self.model_function(beta, self.training_sizes) - self.training_scores),
                        self.compute_out_of_range_penalty(beta)))) ** 2)<10000:
            return  (
            np.mean((np.append((self.model_function(beta, self.training_sizes) - self.training_scores),
                        self.compute_out_of_range_penalty(beta)))) ** 2)
        else:
            return 10001



    # predict full dataset for given model
    def predict(self):
        return self.model_function(self.best_beta, self.sizes)


if __name__ == "__main__":
    print(2)
    # b = m.optimise()
    # print(m.best_beta)

    # with gmpy2.local_context() as ctx:
    #  ctx.precision = 40000
    # x = gmpy2.exp(-1000)

    # print(gmpy2.exp(-1000))
    # print(100 * x)
    # print((np.log(np.exp(1000))))
# arr = np.array([1, 2, 3])
# print(arr[2:])