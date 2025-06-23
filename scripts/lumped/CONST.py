import numpy as np
import sympy 

s = sympy.symbols('s', real=True)

robot_name = 'fishing_rod'
### REFERENCES FETAURES
t_f = 1
dt = 1e-4
max_step = int(np.floor(t_f/dt))
time_task = np.linspace(0, t_f, max_step + 1)
### BOOL USEFUL FOR CONTROL
WANNA_STAY_STILL = False # True # False
PRINT_ONE_ILC = True
WANNA_FB = False
WANNA_MODEL_BASED = False
PRE_TIME = True
WANNA_MODEL_BASED_INIT = False # True # False
WANNA_PRINT = True
WANNA_PLOT = True # True # False
TIME2PRINT = int(max_step/50)
TIME2SAVE = 1
ONLY_DATA_CONTROL = False
conta = 0
conta_all = 0 
append_time_dyn = []
###############################
integratorIs = 'Euler' # Euler # RungeKutta # Trapezi # VariableStep # RungeKuttaAdaptive
### MODEL FETAURES
n_state = 12 # 5 # 20 #
n_x = int(n_state * 2)
n_output = 1
m = n_output
n_actuator = 1
toll_err = 1e-2

### PLOT CONSTANTS
line_widt = 3
font_size = 1 * 15
off_set = 0.2
marker_size = 5 * 3 
line_width_robot = 5 * 3
TIME_2_PAUSE_PLOT = 1
WANNA_SAVE_BIG = False
ITER2SAVE = 50
PRINT_ONE = True
label_list_cart = [r'X',r'Y',r'Z']
color_list = ['r', 'g', 'b', 'm', 'k', 'c', 'y']
C_y = np.zeros(n_state)
A_act = C_y
A_act[0] = 1 

K_P = 100 
K_V = 10 
K_A = 1
############################# PHYSICAL CONSTANTS
n_actuator = m
p = m # number of output variables
g = 9.81

# # /fishing_rod_12_basic_u_new_9/test_fishing_rod_12_basic_u_new.csv
# k = 1e-3 * np.array([5.2, 6.8, 6.4, 5.0, 3.7, 1.9, 1.0, 0.7, 0.4, 0.05, 0.02, 0.01]) # 1e-1
# d = 5e-7 * np.array([5.2, 6.8, 6.4, 5.0, 3.7, 1.9, 1.0, 0.7, 0.4, 0.05, 0.02, 0.01]) # 1e-6

# k = 5e-2 * np.array([5.0, 15.8, 10.4, 8.0, 5.7, 2.9, 2.0, 1.0, 0.7, 0.5, 0.5, 0.5]) # 1e-1
# d = 5e-8 * np.array([5.0, 15.8, 10.4, 8.0, 5.7, 2.9, 2.0, 1.0, 0.7, 0.5, 0.5, 0.5]) # 1e-6

# L = np.array([1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]) 
# m = 1 * np.array([0.05, 0.04, 0.03, 0.03, 0.015, 0.012, 0.001, 0.01, 0.01, 0.005, 0.005, 0.005]) 
# I_zz = 1 * m * L**2  

## from urdf resize into 12. 
# m = np.array([0.0376, 0.0404, 0.02, 0.0134, 0.0101, 0.008, 0.00577, 0.005, 0.0045, 0.004, 0.0036, 0.0033, 0.003, 0.0028, 0.0026, 0.0025, 0.0024, 0.0022, 0.0021, 0.002, 0.0019])
# k_ii = 5 * np.array([0, 34.6099, 26.8363, 17.2029, 11.8908, 8.9892, 12.6067, 8.8801, 4.0401, 3.6465, 3.0458, 6.3951, 3.6361, 2.6307,\
#         3.0194, 1.1336,1.5959, 1.3670, 0.7066, 0.5041, 0.412])
# d_ii = np.array([0.191, 0.164, 0.127, 0.082, 0.056, 0.043,\
#         0.060, 0.042, 0.019, 0.017, 0.014, 0.030, 0.017, 0.012, 0.014, 0.005, 0.008, 0.006, 0.003, 0.002, 0.001])
# L = np.array([0.75, 0.2, 0.2, 0.13, 0.13, 0.13, 0.1, 0.1, 0.1, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08, 
#               0.067, 0.067, 0.067, 0.067, 0.067, 0.067])

m = 1 * np.array([0.04, 0.04, 0.01, 0.008, 0.006, 0.005, 0.004, 0.004, 0.003, 0.0028, 0.0026, 0.0022])
k = 1e0 * np.array([0, 34.61, 17.20, 11.89, 12.61, 8.88, 3.65, 3.05, 6.4, 2.63, 3.02, 1.37])
d = 5e-1 * np.array([0.19, 0.16, 0.08, 0.06, 0.06, 0.04, 0.02, 0.15, 0.3, 0.1, 0.15, 0.06])
L = np.array([0.7, 0.35, 0.35, 0.20, 0.20, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1])
I_zz = 1 * m * L**2  