import glob
from scipy import special
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

#can be made into separate module, called function or something. So it will be easier to manage

def d2_eqn_diffusion(t, tau):
    return np.exp(-tau/(t*2))*(special.iv(0,tau/(t*2))+special.iv(1,tau/(t*2)))

def d1_eqn_diffusion(t, tau):
    return 1-special.erf(np.sqrt(tau/t))+np.sqrt(t/(np.pi*tau))*(1-np.exp(-tau/t))

def d3_eqn_diffusion(t, tau):
    #print(tau)
    #print(t)
    return 1-special.erf(np.sqrt(tau/t))+np.sqrt(t/(np.pi*tau))*(3-np.exp(-(tau/t)))+2*np.sqrt((t**3/(np.pi*tau**3)))*(np.exp(-tau/t)-1)

def exponential_eqn_simple(t, tau, A):
    return A*(1-np.exp(-t/tau))

def get_r2(ydat, xdata, popt, function_fit):
    residauals = ydat-function_fit(xdata, *popt)
    ss_res = np.sum(np.nan_to_num(residauals)**2)
    ss_tot = np.sum((ydat-np.mean(ydat))**2)
    return 1-(ss_res/ss_tot)

functions = {
    1 : d1_eqn_diffusion,
    2 : d2_eqn_diffusion,
    3 : d3_eqn_diffusion,
    4 : exponential_eqn_simple
}


class CurveAnalysis():

    functions = {
    1 : d1_eqn_diffusion,
    2 : d2_eqn_diffusion,
    3 : d3_eqn_diffusion,
    4 : exponential_eqn_simple
}


    def __init__(self, plotdata, function_type):
        self.int_values_list = np.array(plotdata[1])
        self.time_values = np.array(plotdata[0])
        self.first_zero = [index for index, value in enumerate(self.time_values) if value <= 0]
        self.function_t = functions[int(function_type)]
        self.nonzerotime = self.time_values[self.first_zero[-1]:]+1e-9


    def fitEquation(self):
        popt, pcov = curve_fit(self.function_t , self.time_values[self.first_zero[-1]:] , self.int_values_list[self.first_zero[-1]:])
        return popt
    
    def fitEquation_nz(self):
        popt, pcov = curve_fit(self.function_t , self.nonzerotime , self.int_values_list[self.first_zero[-1]:])
        return popt

    def plotFit(self):
        x1 = np.linspace(self.nonzerotime[0], self.nonzerotime[-1], 100) 
        plt.plot(x1, self.function_t(x1, self.fitEquation_nz()), 'r--')
    
    def returnR2(self, nonzero=False):
        #print(self.fitEquation())
        if nonzero == False:
            r2_value = get_r2(self.int_values_list[self.first_zero[-1]:], self.time_values[self.first_zero[-1]:], self.fitEquation(), self.function_t)
            return r2_value
        else:
            r2_value = get_r2(self.int_values_list[self.first_zero[-1]:], self.nonzerotime, self.fitEquation_nz(), self.function_t)
            return r2_value

    def plotCurve(self):
        plt.scatter(self.time_values[self.first_zero[-1]:],self.int_values_list[self.first_zero[-1]:])

