import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py, time, datetime, json
locator = mdates.AutoDateLocator()
timeformat = mdates.ConciseDateFormatter(locator)
from scipy.optimize import curve_fit, brentq
from lmfit import minimize, Parameters, report_fit
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#MySQL stuff
import mysql, mysql.connector
mysqlconfig = {
    "host": "phys-dots-data.physik.unibas.ch",
    "user": "logger",
    "passwd": "CryoH4ll",
    "database": "logs"
}
# END MySQL

# Constants
Rk = 25813
e0 = 1.602176634e-19
muB = 9.274009994e-24
R = 8.31446261815324      # J/K/mol  Ideal gas constant
mu0 = 1.25663706212e-6    # H/m      Vaccuum Permeability
muN = 5.050783699e-27     # J/T      Nuclear Magneton
kB = 1.380649e-23         # J/K      Boltzman constant
A0 = 6.02214076e23        # 1/mol    Avogadro's constant
L0 = 2.445e-8             # W.Ohms/K^2 Lorenz constant used in Wiedemann-Franz law (for silver wires in this simulation)

#Properties of Silver (Who knows when we're going to need these?)
Ag_atomicmass = 107.87-3 # Kg/mol

# Properties of copper
Cu_Jn = 3/2                  # nuclear spin for 63Cu and 65Cu
Cu_rho = 8930.0              # Kg/m^3 Density
Cu_atomicmass = 63.5463e-3   # Kg/mol
Cu_gamma = 0.691e-3          # J/mol/K^2 See Pobell Table 10.1
Cu_Ce = lambda Te: Cu_gamma*Te  # Electron's heat capacity / mol

abundCu63, abundCu65 = 0.6915, 1.0 - 0.6915  # Natural abundance of two stable isotopes of copper
Cu_Korringa = abundCu63*1.27  + abundCu65*1.09  # = 1.215 K.s See Pobell 3rdEd., Table 10.1
# Cu g-factor = nuclear magnetic moment / spin
# http://faculty.missouri.edu/~glaserr/8160f09/STONE_Tables.pdf or Pobell, table 10.1
mu_r63 = 2.227206         # in units of mu_N
mu_r65 = 2.3816           # in units of mu_N
g_factor63 = mu_r63/Cu_Jn
g_factor65 = mu_r65/Cu_Jn
CC63 = mu0 * g_factor63*g_factor63 * A0 * muN*muN * Cu_Jn*(Cu_Jn+1) / (3*kB)
CC65 = mu0 * g_factor65*g_factor65 * A0 * muN*muN * Cu_Jn*(Cu_Jn+1) / (3*kB)
Cu_CC = abundCu63*CC63 + abundCu65*CC65 # Molar Currie Constant in K/m A^2 mol
Cu_Cn = lambda Tn, h:(Cu_CC / mu0) * (h*h)/(Tn*Tn) # Nuclear heat capacity / mole

# Measurement specific parameters
R_wire = 8255/2
#Ec_Cu, gT_Cu = 0.7362499766572822, 21.683533093853708e-6
Ec_Cu, gT_Cu = 0.7373, 21.683533093853708e-6
# Ec_Cu, gT_Cu = 0.73650, 21.1684e-06

# Obtained from https://stackoverflow.com/questions/20339234/python-and-lmfit-how-to-fit-multiple-datasets-with-shared-parameters
def MasterEquation(Vsd, Vsd0, gT, Ec, Te, N=33):
    Vsd = np.asarray(Vsd)
    scalar_input = False
    if Vsd.ndim == 0:
        Vsd = Vsd[None]  # Makes x 1D
        scalar_input = True
    u = 2*Ec/Te
    flt = np.abs(Vsd - Vsd0) > 1e-6
    x = np.ones(len(Vsd))
    x[flt] = e0*(Vsd[flt] - Vsd0)/(N*kB*Te)

    g = np.zeros(len(Vsd))
    g[flt] = (x[flt]*np.sinh(x[flt])-4*np.sinh(x[flt]/2)**2)/(8*np.sinh(x[flt]/2)**4)
    
    ret = np.ones(len(Vsd))*gT*(1-u*g)
    ret[np.logical_not(flt)] = gT*(1 - u/6 + u*u/60 - u*u*u/630)
    
    if scalar_input:
        return np.squeeze(ret)
    return ret

def GetBFData(sensor_id, start_time, end_time):
    sql = f"SELECT UNIX_TIMESTAMP(`time`), `value` FROM `records` WHERE `time` BETWEEN FROM_UNIXTIME({start_time:.0f}) AND FROM_UNIXTIME({end_time:.0f}) AND `sensor_id`={sensor_id} ORDER BY `time` ASC"
    db = mysql.connector.connect(**mysqlconfig)
    cursor = db.cursor()
    cursor.execute(sql)
    return np.array(cursor.fetchall())

def Tcbt(g0gT, Ec, N=33):
    if np.isnan(g0gT):
        return np.nan
    if g0gT > 1 or g0gT<=0:
        return np.nan
    _dg = 1-g0gT
    f = lambda u: _dg - (u/6 - u*u/60 + u*u*u/630)
    u = brentq(f, 0, 150)
    if u==0:
        return np.nan
    return 2*Ec/u

def Tcbt_Cu(g0gT, Ec=Ec_Cu):
    g0gT = np.asarray(g0gT)
    scalar_input = False
    if g0gT.ndim == 0:
        g0gT = g0gT[None]  # Makes x 1D
        scalar_input = True
   
    T = np.array([Tcbt(g0gT[i], Ec, N=33) for i in range(len(g0gT))])
    
    #with h5py.File('./data/MCMC_Cu.h5', 'r') as f:
    #    g0gT_mcmc = np.array(f['monte_carlo/mean normalized zero bias conductance'])
    #    g0gT_std = np.array(f['monte_carlo/std normalized zero bias conductance'])
    #    temp = np.array(f['monte_carlo/temperature'])
    with h5py.File('./data/Calibration_NormalizedTemperature.h5', 'r') as f:
        g0gT_mcmc = np.array(f['monte_carlo/mean normalized zero bias conductance'])
        g0gT_std = np.array(f['monte_carlo/std normalized zero bias conductance'])
        temp = np.array(f['monte_carlo/normalized temperature'])*Ec

    slc = T<Ec # np.logical_or(np.isnan(T), 
    T[slc] = np.interp(g0gT[slc], g0gT_mcmc, temp)
    if scalar_input:
        return np.squeeze(T)
    return T

# miscellaneous functions
def MakeSmoothie(data, ws=20):
    smooth_array = np.ones(ws)
    smooth_data = np.convolve(smooth_array/np.sum(smooth_array), data, mode='valid')
    smooth_data = np.hstack(
        (smooth_data[0]*np.ones(int(ws/2)),
         smooth_data,
         smooth_data[-1]*np.ones(int(ws/2)-1)
    ))
    return smooth_data