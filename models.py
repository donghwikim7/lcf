import numpy as np
import scipy as sp
from scipy.optimize import basinhopping
import models_function

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
        self.out_of_range_penalty = 2
        self.initailparam = None
        if initailparams == None:
            self.initailparam = np.random.rand(15,self.n_param )
        else:
            b = False
            for i in range(len(initailparams)):
                if initailparams[i]['model'] == self.id:
                    self.initailparam = initailparams[i]['centroids']
                    b = True
                    break
            if not b:
                self.initailparam = np.random.rand(15, self.n_param)



    def compute_out_of_range_penalty(self, beta):
        arr1 = self.model_function(beta, self.sizes[len(self.training_sizes):])
        arr1[(arr1 > 0.0) & (arr1 < 1)] = 0
        return self.out_of_range_penalty * arr1

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

            return self.best_beta
        else:
            return self.best_beta

    def fitness_function(self, beta):
        return np.mean(
            ((np.append((self.model_function(beta, self.training_sizes) - self.training_scores),
                        self.compute_out_of_range_penalty(beta)))) ** 2)



    # predict full dataset for given model
    def predict(self):
        return self.model_function(self.best_beta, self.sizes)


if __name__ == "__main__":
    import pickle
    with open('james.p', 'rb') as file:
        arr = pickle.load(file)
        print(arr)
    print(arr[0]['model'])