import sympy
from sympy.utilities.lambdify import lambdify
from sympy import symbols, Matrix, latex, Add
from sympy.vector import *
from sympy import symbols, pprint, simplify, atan2, sqrt, diag
from sympy import  eye, symbols
import numpy as np
import inspect
import os 
import re 
from termcolor import colored

WANNA_CREATE = True # False # True
WANNA_PRINT = True


def get_function(func, name_fun, folder_name, params=None):
    """_summary_

    Args:
        func (sympy): _description_
        name_fun (string): _description_
        params (sympy, optional): _description_. Defaults to None.
    """
    if params is None:
        func = lambdify((q, L), func, modules=['numpy'])
    else:
        func = lambdify((q, L, params), func, modules=['numpy'])
        
    func_str = inspect.getsource(func)
    func_str.__name__ = name_fun
    func_str = func_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
    func_str = re.sub(r'def _lambdifygenerated', 'def ' + name_fun, func_str)

    if WANNA_CREATE: 
        if os.path.isfile(folder_name + '/' + name_fun + '.py'):
            print(name_fun + '.py already exists!')
        else:
            with open(folder_name + '/' + name_fun + '.py', 'w') as f:
                f.write('import numpy as np\n')
                f.write('from jax import jit\n')
                f.write('from jax import numpy as jnp\n\n')
                f.write(func_str)
            print(name_fun + '.py' + ' MATRIX ... DONE!')
        print(colored('===================================================================================================', 'green'))


def PCC_kine(phi, theta, dL):
    # Implement also the Delta-parametrization
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8722799&casa_token=0ApuPyuVxfoAAAAA:vDMIs8JAVWvdmRzCzBV0ZzMjopyyk0wacN0FjrPC_NJwSZ3M6GJB_1JbHLyFeqA1Ov4FDtxE2g&tag=1
    c_phi = sympy.cos(phi)
    s_phi = sympy.sin(phi)
    c_theta = sympy.cos(theta)
    s_theta = sympy.sin(theta)
    p_tip = Matrix([c_phi * (dL + L) * (c_theta - 1)/theta, s_phi * (c_theta - 1) * (dL + L)/theta, s_theta * (dL + L)/theta])
    
    A = Matrix([[c_phi**2 * (c_theta - 1) - 1, s_phi * c_phi * (c_theta - 1), -c_phi * s_theta, p_tip[0]],
            [s_phi * c_phi * (c_theta - 1), c_phi**2 * (c_theta - 1) - c_theta, -s_phi * s_theta, p_tip[1]],
            [c_phi * s_theta, s_theta * s_phi, c_theta, p_tip[2]],
            [0, 0, 0, 1]])
    
         
    return A, p_tip

def christoffel(M, q, q_dot):
    n = len(q)
    c = np.zeros((n, n, n), dtype=object)
    C = sympy.zeros(n, n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i,j,k] = (sympy.Rational(1, 2) * (sympy.diff(M[i,k], q[j]) + sympy.diff(M[i,j], q[k]) - sympy.diff(M[j,k], q[i])))
                C[i,j] += c[i,j,k] * q_dot[k]
    return C

print('\n')
print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
print('                          INIT PCC DYNAMIC')
print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
print('\n')
# print(colored('ToDo: Generalized wrt the number of segs (now it is 1 !!)', 'yellow'))
folder_name = "pcc_utils_py_jax"
path = os.getcwd()

# Check if the folder exists
folder_path = os.path.join(path, folder_name)
print('===================================================================================================')
if not os.path.exists(folder_path):
    # Create the folder if it does not exist
    os.makedirs(folder_path)
    print("Folder created successfully!")
    print('===================================================================================================')
else:
    print("Folder already exists!")
    print('===================================================================================================')

######################################################################################################################################

theta, phi, L, rho, E, r, g, kappa_theta, kappa_phi, d_ii, s, dL = symbols('theta phi L rho E r g kappa_theta kappa_phi d_ii s dL', real=True)
theta_dot, phi_dot, dL_dot = symbols('theta_dot phi_dot dL_dot', real=True)
theta_ddot, phi_ddot, dL_ddot = symbols('theta_ddot phi_ddot dL_ddot', real=True)

tol_s = 0.001
q = [phi, theta, dL]
q_dot = [phi_dot, theta_dot, dL_dot]
q_ddot = [phi_ddot, theta_ddot, dL_ddot]

T_01_pcc, p_tip_pcc = PCC_kine(phi, theta, dL)
T_01 = T_01_pcc
p_tip = p_tip_pcc
R = T_01[0:3, 0:3]
# print(latex(p_tip))

T_01_pcc_fun = lambdify((q, L), T_01_pcc, modules=['numpy'])
T_01_pcc_fun.__name__ = '_direct_kinematics_pcc'
T_01_pcc_fun_str = inspect.getsource(T_01_pcc_fun)
T_01_pcc_fun_str = T_01_pcc_fun_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
T_01_pcc_fun_str = re.sub(r'def _lambdifygenerated', 'def _direct_kinematics_pcc', T_01_pcc_fun_str)

if WANNA_CREATE:    
    if os.path.isfile(folder_name + '/_direct_kinematics_pcc.py'):
        print('_direct_kinematics_pcc.py already exists!')
    else:
        with open(folder_name + '/_direct_kinematics_pcc.py', 'w') as f:
            f.write('from jax import jit\n')
            f.write('from jax import numpy as jnp\n')
            f.write('import numpy as np\n\n')
            f.write(T_01_pcc_fun_str)
        print('POSITION PARAMETRIZATION ... DONE!')
    print('===================================================================================================')
    
    
p_tip_fun = lambdify((q, L), p_tip, modules=['numpy'])
p_tip_fun.__name__='_p_tip_pcc'
p_tip_fun_str = inspect.getsource(p_tip_fun)
p_tip_fun_str = p_tip_fun_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
p_tip_fun_str = re.sub(r'def _lambdifygenerated', 'def _p_tip_pcc', p_tip_fun_str)

if WANNA_CREATE:    
    if os.path.isfile(folder_name + '/_p_tip_pcc.py'):
        print('_p_tip_pcc.py already exists!')
    else:
        with open(folder_name + '/_p_tip_pcc.py', 'w') as f:
            f.write('from jax import jit\n')
            f.write('from jax import numpy as jnp\n')
            f.write('import numpy as np\n\n')
            f.write(p_tip_fun_str)
        print('TIP POSITION PARAMETRIZATION ... DONE!')
    print('===================================================================================================')


J_p = simplify(p_tip.jacobian(q))
yaw = atan2(R[1,0], R[0,0])
pitch = atan2(-R[2,0], sqrt(R[2,1]**2 + R[2,2]**2) )
roll = atan2(R[2,1], R[2,2])
angles = Matrix([yaw, pitch, roll])

angles_fun = lambdify((q, L), angles, modules=['numpy'])
angles_fun.__name__ = '_angles_fun'
angles_fun_str = inspect.getsource(angles_fun)
angles_fun_str = angles_fun_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
angles_fun_str = re.sub(r'def _lambdifygenerated', 'def _angles_fun', angles_fun_str)

if WANNA_CREATE:
    if os.path.isfile(folder_name + '/_angles_fun.py'):
        print('_angles_fun.py already exists!')
    else:
        with open(folder_name + '/_angles_fun.py', 'w') as f:
            f.write('import numpy as np\n')
            f.write('from jax import jit\n')
            f.write('from jax import numpy as jnp\n\n')
            f.write(angles_fun_str)
        print('ORIENTATION PARAMETRIZED ... DONE!')
    print('===================================================================================================')
    
angles_simple = Matrix([phi, theta, 0])
J_o = simplify(angles_simple.jacobian(q))
J = simplify(J_p.col_join(J_o))
J_fun = lambdify((q, L), J, modules=['numpy'])
J_fun.__name__ = '_jacobian_diff'
J_fun_str = inspect.getsource(J_fun)
J_fun_str = J_fun_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
J_fun_str = re.sub(r'def _lambdifygenerated', 'def _jacobian_diff', J_fun_str)

if WANNA_CREATE:
    if os.path.isfile(folder_name + '/_jacobian_diff.py'):
        print('_jacobian_diff.py already exists!')
    else:
        with open(folder_name + '/_jacobian_diff.py', 'w') as f:
            f.write('import numpy as np\n')
            f.write('from jax import jit\n')
            f.write('from jax import numpy as jnp\n\n')
            f.write(J_fun_str)
        print('DIFFERENTIAL KINEAMTICS ... DONE!')
    print('===================================================================================================')

I_xx, I_yy, I_zz, m_0, kappa_dL = symbols('I_xx I_yy I_zz m_0 kappa_dL', real=True, positive=True)
inertia_tensor = diag(I_xx, I_yy, I_zz)
B_lin = simplify(J_p.transpose() * m_0 * eye(3) * J_p)
B_rot = simplify(J_o.transpose() * inertia_tensor * J_o)
B = simplify(B_rot + B_lin)
B_fun = lambdify((q, L, I_xx, I_yy, I_zz, m_0), B, modules=['numpy']) 
B_fun.__name__ = '_inertia_matrix'
B_fun_str = inspect.getsource(B_fun)
B_fun_str = B_fun_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
B_fun_str = re.sub(r'def _lambdifygenerated', 'def _inertia_matrix', B_fun_str)

if WANNA_CREATE:
    if os.path.isfile(folder_name + '/_inertia_matrix.py'):
        print('_inertia_matrix.py already exists!')
    else:
        with open(folder_name + '/_inertia_matrix.py', 'w') as f:
            f.write('import numpy as np\n')
            f.write('from jax import jit\n')
            f.write('from jax import numpy as jnp\n\n')
            f.write(B_fun_str)
        print('INERTIA MATRIX ... DONE!')
    print('===================================================================================================')

C = christoffel(B, q, q_dot)
# C = simplify(C)
C_fun = lambdify((q, q_dot, L, I_xx, I_yy, I_zz, m_0), C, modules=['numpy'])
C_fun.__name__ = '_coriolis_matrix'
C_fun_str = inspect.getsource(C_fun)
C_fun_str = C_fun_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
C_fun_str = re.sub(r'def _lambdifygenerated', 'def _coriolis_matrix', C_fun_str)

if WANNA_CREATE:
    if os.path.isfile(folder_name + '/_coriolis_matrix.py'):
        print('_coriolis_matrix.py already exists!')
    else:
        with open(folder_name + '/_coriolis_matrix.py', 'w') as f:
            f.write('import numpy as np\n')
            f.write('from jax import jit\n')
            f.write('from jax import numpy as jnp\n\n')
            f.write(C_fun_str)
        print('CORIOLIS MATRIX ... DONE!')
    print('===================================================================================================')
    
U_g = m_0 * sympy.Matrix([[0, 0, -g]]) * p_tip
G = U_g.jacobian(q)
G_fun = lambdify((q, L, m_0, g), G, modules=['numpy'])
G_fun.__name__ = '_gravity_matrix'
G_fun_str = inspect.getsource(G_fun)
G_fun_str = G_fun_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
G_fun_str = re.sub(r'def _lambdifygenerated', 'def _gravity_matrix', G_fun_str)

if WANNA_CREATE:
    if os.path.isfile(folder_name + '/_gravity_matrix.py'):
        print('_gravity_matrix.py already exists!')
    else:
        with open(folder_name + '/_gravity_matrix.py', 'w') as f:
            f.write('import numpy as np\n')
            f.write('from jax import jit\n')
            f.write('from jax import numpy as jnp\n\n')
            f.write(G_fun_str)
        print('GRAVITY VECTOR ... DONE!')
    print('===================================================================================================')

U_e = sympy.Rational(1, 2) * kappa_phi * phi**2 + sympy.Rational(1, 2) * kappa_theta * theta**2 + sympy.Rational(1, 2) * kappa_dL * dL**2
K_q = sympy.Matrix([U_e]).jacobian(q)
K_fun = lambdify((q, kappa_theta, kappa_phi, kappa_dL), K_q, modules=['numpy'])
K_fun.__name__ = '_stiffness_torque'
K_fun_str = inspect.getsource(K_fun)
K_fun_str = K_fun_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
K_fun_str = re.sub(r'def _lambdifygenerated', 'def _stiffness_torque', K_fun_str)

if WANNA_CREATE:
    if os.path.isfile(folder_name + '/_stiffness_torque.py'):
        print('_stiffness_torque.py already exists!')
    else:
        with open(folder_name + '/_stiffness_torque.py', 'w') as f:
            f.write('import numpy as np\n')
            f.write('from jax import jit\n')
            f.write('from jax import numpy as jnp\n\n')
            f.write(K_fun_str)
        print('STIFFNESS MATRIX ... DONE!')
    print('===================================================================================================')
    
q_dot_M = Matrix([q_dot])
L_kin = q_dot_M * Matrix(B) * q_dot_M.T
L_pot = - (U_e + Add(*U_g))
# L = Add(*L_kin) + L_pot
# print('Lagrangian: \n{}'.format(L))

L_pot_fun = lambdify((q, q_dot, L, I_xx, I_yy, I_zz, m_0), L_kin, modules=['numpy']) 
L_pot_fun.__name__ = '_L_pot'
L_pot_fun_str = inspect.getsource(L_pot_fun)
L_pot_fun_str = L_pot_fun_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
L_pot_fun_str = re.sub(r'def _lambdifygenerated', 'def _L_pot', L_pot_fun_str)

if WANNA_CREATE:
    if os.path.isfile(folder_name + '/_L_pot.py'):
        print('_L_pot.py already exists!')
    else:
        with open(folder_name + '/_L_pot.py', 'w') as f:
            f.write('import numpy as np\n')
            f.write('from jax import jit\n')
            f.write('from jax import numpy as jnp\n\n')
            f.write(L_pot_fun_str)
            print('LAGRANGIAN POT ... DONE!')
    print('===================================================================================================')

L_kin_fun = lambdify((q, q_dot, L, I_xx, I_yy, I_zz, m_0), L_kin, modules=['numpy']) 
L_kin_fun.__name__ = '_L_kin'
L_kin_fun_str = inspect.getsource(L_kin_fun)
L_kin_fun_str = L_kin_fun_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
L_kin_fun_str = re.sub(r'def _lambdifygenerated', 'def _L_kin', L_kin_fun_str)

if WANNA_CREATE:
    if os.path.isfile(folder_name + '/_L_kin.py'):
        print('_L_kin.py already exists!')
    else:
        with open(folder_name + '/_L_kin.py', 'w') as f:
            f.write('import numpy as np\n')
            f.write('from jax import jit\n')
            f.write('from jax import numpy as jnp\n\n')
            f.write(L_kin_fun_str)
            print('LAGRANGIAN KIN ... DONE!')
    print('===================================================================================================')


D_q_dot = diag(d_ii, d_ii, d_ii) * sympy.Matrix(q_dot)
D_fun = lambdify((q_dot, d_ii), D_q_dot, modules=['numpy'])
D_fun.__name__ = '_damping_torque'
D_fun_str = inspect.getsource(D_fun)
D_fun_str = D_fun_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
D_fun_str = re.sub(r'def _lambdifygenerated', 'def _damping_torque', D_fun_str)

if WANNA_CREATE:
    if os.path.isfile(folder_name + '/_damping_torque.py'):
        print('_damping_torque.py already exists!')
    else:
        with open(folder_name + '/_damping_torque.py', 'w') as f:
            f.write('import numpy as np\n')
            f.write('from jax import jit\n')
            f.write('from jax import numpy as jnp\n\n')
            f.write(D_fun_str)
            print('DAMPING MATRIX ... DONE!')
    print('===================================================================================================')

r_pulley, r_head_1, r_head_2, theta_head_1, theta_head_2 = \
                symbols('r_pulley r_head_1 r_head_2 theta_head_1 theta_head_2', real=True, positive=True) 
f_hat = sympy.Matrix([0, 0, -1])
p_hat1 = sympy.Matrix([r_head_1 * sympy.cos(theta_head_1), r_head_1 * sympy.sin(theta_head_1), 0])
p_hat2 = sympy.Matrix([-r_head_1 * sympy.sin(theta_head_1), r_head_1 * sympy.cos(theta_head_1), 0])
p_hat3 = sympy.Matrix([-r_head_1 * sympy.cos(theta_head_1), -r_head_1 * sympy.sin(theta_head_1), 0])
p_hat4 = sympy.Matrix([r_head_1 * sympy.sin(theta_head_1), -r_head_1 * sympy.cos(theta_head_1), 0])

f1, f2, f3, f4 = symbols('f1 f2 f3 f4', real = True)
f = sympy.Matrix([f1, f2, f3, f4])

f_hat_1 = f_hat * f1
f_hat_2 = f_hat * f2
f_hat_3 = f_hat * f3
f_hat_4 = f_hat * f4
M1 = sympy.Matrix([p_hat1[1] * f_hat_1[2] - p_hat1[2] * f_hat_1[1],
                    p_hat1[2] * f_hat_1[0] - p_hat1[0] * f_hat_1[2],
                    p_hat1[0] * f_hat_1[1] - p_hat1[1] * f_hat_1[0]])

M2 = sympy.Matrix([p_hat2[1] * f_hat_2[2] - p_hat2[2] * f_hat_2[1],
                    p_hat2[2] * f_hat_2[0] - p_hat2[0] * f_hat_2[2],
                    p_hat2[0] * f_hat_2[1] - p_hat2[1] * f_hat_2[0]])

M3 = sympy.Matrix([p_hat3[1] * f_hat_3[2] - p_hat3[2] * f_hat_3[1],
                    p_hat3[2] * f_hat_3[0] - p_hat3[0] * f_hat_3[2],
                    p_hat3[0] * f_hat_3[1] - p_hat3[1] * f_hat_3[0]])

M4 = sympy.Matrix([p_hat4[1] * f_hat_4[2] - p_hat4[2] * f_hat_4[1],
                    p_hat4[2] * f_hat_4[0] - p_hat4[0] * f_hat_4[2],
                    p_hat4[0] * f_hat_4[1] - p_hat4[1] * f_hat_4[0]])

MM1 = sympy.Matrix(M1[:2]) + sympy.Matrix(M2[:2]) + sympy.Matrix(M3[:2]) + sympy.Matrix(M4[:2])
# print(shape(MM1), type(MM1))
MM2 = sympy.Matrix(MM1).jacobian(f) 

M = sympy.Matrix.hstack(
        sympy.Matrix([0,0,0,0]),
        sympy.Matrix([0,0,0,0]),
        sympy.Matrix([-1,-1,-1,-1]),
        MM2.transpose(),
        sympy.Matrix([0,0,0,0]),
    ).transpose()

P = simplify(J.transpose() * M) 
P_fun = lambdify((q, L, r_head_1, theta_head_1), P, modules=['numpy'])
P_fun.__name__ = '_tendon_matrix'
P_fun_str = inspect.getsource(P_fun)
P_fun_str = P_fun_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
P_fun_str = re.sub(r'def _lambdifygenerated', 'def _tendon_matrix', P_fun_str)

if WANNA_CREATE: 
    if os.path.isfile(folder_name + '/_tendon_matrix.py'):
        print('_tendon_matrix.py already exists!')
    else:
        with open(folder_name + '/_tendon_matrix.py', 'w') as f:
            f.write('import numpy as np\n')
            f.write('from jax import jit\n')
            f.write('from jax import numpy as jnp\n\n')
            f.write(P_fun_str)
        print('TENDON MATRIX ... DONE!')
    print('===================================================================================================')


P_pen = sympy.Matrix([[-sympy.cos(phi) * sympy.sin(theta), -sympy.sin(phi) * sympy.sin(theta), 0], 
                     [-sympy.sin(phi), sympy.cos(phi), (L + dL) * (theta - sympy.sin(theta))/theta**2],
                     [0, 0, sympy.sin(theta)/theta]])
P_pen_fun = lambdify((q, L), P_pen, modules=['numpy'])
P_pen_fun.__name__ = '_pneumatic_matrix'
P_pen_fun_str = inspect.getsource(P_pen_fun)
P_pen_fun_str = P_pen_fun_str.replace('sin', 'jnp.sin').replace('cos', 'jnp.cos').replace('array', 'np.array').replace("arctan2","jnp.arctan2").replace("pi", "jnp.pi")
P_pen_fun_str = re.sub(r'def _lambdifygenerated', 'def _pneumatic_matrix', P_pen_fun_str)
pprint(P_pen)
if WANNA_CREATE: 
    if os.path.isfile(folder_name + '/_pneumatic_matrix.py'):
        print('_pneumatic_matrix.py already exists!')
    else:
        with open(folder_name + '/_pneumatic_matrix.py', 'w') as f:
            f.write('import numpy as np\n')
            f.write('from jax import jit\n')
            f.write('from jax import numpy as jnp\n\n')
            f.write(P_pen_fun_str)
        print('PENUAMATIC MATRIX ... DONE!')
    print('===================================================================================================')


if WANNA_PRINT:
    print('\nB Inertia Matrix:')
    pprint(J)
    J_latex = latex(J)
    print(J_latex)
    print('\n\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
    print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')

    print('\nB Inertia Matrix:')
    pprint(B)
    B_latex = latex(B)
    print(B_latex)
    print('\n\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
    print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')

    print('\nDeterminant B Inertia Matrix:')
    det_B = simplify(B.det())
    pprint(det_B)
    det_B_latex = latex(det_B)
    print(det_B_latex)
    print('\n\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
    print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')

    print('\nB Inertia Matrix INV:')
    B_inv = simplify(B.inv())
    pprint(B_inv)
    B_inv_latex = latex(B_inv)
    print(B_inv_latex)
    print('\n\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
    print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')

    # print('\nJ Jacobian:')
    # pprint(J)
    # B_latex = latex(J)
    # print(B_latex)
    # print('\n\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
    # print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')

    # print('\nLgLfh Jacobian:')
    # LgLfh = simplify( J_p * B.inv() * P )
    # pprint(LgLfh)
    # LgLfh_latex = latex(LgLfh)
    # print(LgLfh_latex)
    # print('\n\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
    # print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')


print('\n')
print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
print('\t\t\t FINISH')
print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
print('\n')


