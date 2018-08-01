import numpy as np
import copy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.visualization import latex_drawer
from qiskit.extensions.standard import h, ry, barrier, cz, x, y, z
from qiskit.tools.qi.pauli import Pauli, label_to_pauli

#I added these to visualize a circuit
import shutil
import pdf2image

def circuitImage(circuit, basis="u1,u2,u3,cx"):
    """Obtain the circuit in image format
    Note: Requires pdflatex installed (to compile Latex)
    Note: Required pdf2image Python package (to display pdf as image)
    """
    filename='circuit'
    tmpdir='tmp/'
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    latex_drawer(circuit, tmpdir+filename+".tex", basis=basis)
    os.system("pdflatex -output-directory {} {}".format(tmpdir, filename+".tex"))
    images = pdf2image.convert_from_path(tmpdir+filename+".pdf")
    shutil.rmtree(tmpdir)
    return images[0]

############################################################
# Code below is used for the Nelder-Mead optimization algorithm.
# Python has a built in function for a Nelder-Mead algorithm, but
# the exit conditions for the minimizer are different. This algorithm
# works as well as the Python one.
############################################################
def findLow(data_set):
    size = len(data_set)
    z_low = 1e10
    for i in range(size):
        if data_set[i] < z_low:
            l = i
            z_low = data_set[i]

    return l
    
############################################################
def findHigh(data_set):
    size = len(data_set)
    z_high = -1e10
    for i in range(size):
        if data_set[i] > z_high:
            h = i
            z_high = data_set[i]

    return h

############################################################
def calculateAverage(data_set):
    average = 0.0
    number_of_entries = len(data_set)
    for i in range(number_of_entries):
        average += data_set[i]
        
    return average / len(data_set)

############################################################
def calculateSTDEV(data_set):
    number_of_entries = len(data_set)
    average = calculateAverage(data_set)
    stdev = 0
    for i in range(number_of_entries):
        stdev += (data_set[i] - average)*(data_set[i] - average)

    return ( stdev / (number_of_entries-1.0) )**0.5

############################################################
def centroid(data_set, high):
    
    num_points = len(data_set)
    num_vars = num_points - 1
    p_bar = [0 for i in range(num_vars)]
    
    for i in range(num_points):
        if i != high:
            for j in range(num_vars):
                p_bar[j] += data_set[i][j]

    for i in range(num_vars):
        p_bar[i] /= num_vars

    return p_bar

############################################################
def amoeba(obj_fun, initial_theta, save_step, target_stdev=1e-4, q=0.1, alpha = 1.0, beta = 0.5, gamma = 2.0, stuck = 50):

    output = []
    ########################################
    # Creating the initial simplex #########

    num_vars = len(initial_theta)
    num_points = num_vars + 1
    p_simplex = [[0] * num_vars for i in range(num_points)]
    for i in range(num_points):
        for j in range(num_vars):
            if i==num_points:
                #p_simplex[i][j] = -2.0 * n**(-0.5) * np.pi
                p_simplex[i][j] = -1.0 * n**(-0.5) + initial_theta[j]
            else:
                p_simplex[i][j] = initial_theta[j] + np.random.uniform(-np.pi/2.0, np.pi/2.0) if (i==j) else 0

                
    # Now we have our simplex of points#####
    ########################################

    p_star = [0 for i in range(num_vars)]
    p_star_star = [0 for i in range(num_vars)]
    ########################################
    # z_point = costFunction(point)
    
    z_simplex = [0 for i in range(num_points)]
    z_high = -10000000
    z_low  = 10000000
    for i in range(num_points):
        z_simplex[i] = obj_fun(p_simplex[i])
        if z_simplex[i] > z_high:
            high = i
            z_high = z_simplex[i]
        if z_simplex[i] < z_low:
            low = i
            z_low = z_simplex[i]

    stdev = calculateSTDEV(z_simplex)
    count = 0
    iteration = 0
    while (iteration < 1):
        if count%save_step == 0:
            output.append([count, z_low])
            
            print('Amoeba on count ' + str(count))
            print('The low energy is ' + str(z_low) + '\n')
            
        count += 1
        recalc_all = 0
        high = findHigh(z_simplex)#location of high point in simplex list
        low = findLow(z_simplex)#location of low point in simplex list
        p_high = p_simplex[high]
        p_low = p_simplex[low]

        z_high = obj_fun(p_simplex[high])
        z_low  = obj_fun(p_simplex[low])

        p_bar = centroid(p_simplex, high)#p_bar is the average of all points except for p_high

        #construct p_star
        for i in range(num_vars):
            p_star[i] = (1+alpha)*p_bar[i] - alpha*p_simplex[high][i]

        z_star = obj_fun(p_star)

        if z_star < z_low:
            for i in range(num_vars):
                p_star_star[i] = (1+gamma)*p_star[i] - gamma*p_bar[i]

            z_star_star = obj_fun(p_star_star)

            if z_star_star < z_low:
                for i in range(num_vars):
                    p_simplex[high][i] = p_star_star[i]
            else:
                for i in range(num_vars):
                    p_simplex[high][i] = p_star[i]

        else:
            for i in range(num_points):
                if i != high:
                    if z_star < z_simplex[i]:
                        check = 0 #z_star < z_simplex[i] for some i. We could not have possible produced a new high value
                        break
                    else:
                        check = 1 #z_star > z_simplex[i] for all i. We could have produced a new high value

            if check == 1: #we could have mae a new high point, need to check
                if z_star < z_high: #if the new point is less than the high point, use the new point as the lower high point
                    for i in range(num_vars):
                        p_high[i] = p_star[i]
                        #if z_star is higher, we ignore it and use the lower highest value
                for i in range(num_vars):
                    p_star_star[i] = beta*p_high[i] + (1-beta)*p_bar[i]

                z_star_star = obj_fun(p_star_star)
                if z_star_star > z_high: #we made a new high point. Shrink it
                    replace_all = 1
                    for i in range(num_points):
                        for j in range(num_vars):
                            p_simplex[i][j] = 0.5*(p_simplex[i][j] + p_low[j])
                else:
                    for i in range(num_vars):
                        p_high[i] = p_star_star[i]
            else:
                for i in range(num_vars):
                    p_high[i] = p_star[i]

        if recalc_all:
            for i in range(num_points):
                z_simplex[i] = obj_fun(p_simplex[i])
        else:
            z_simplex[high] = obj_fun(p_simplex[high])

        stdev = calculateSTDEV(z_simplex)
        if stdev < target_stdev or count > stuck:
            if count > stuck:
                print('Amoeba got stuck, count = ' + str(count) + '\n')
                count = 0
            else:
                iteration += 1
                print('Amoeba found min, reseting points. Starting iteration ' + str(iteration) + '\n')
            
            recalc_all = 1
            
            #or i in range(num_points):
            #   for j in range(num_vars):
            #       p_simplex[i][j] = p_bar[j] + q*(i+j)
            #   z_simplex[i] = obj_fun(p_simplex[i])
                

    low = findLow(z_simplex)
    print('Ameoba finished on count ' + str(count) + ' with energy = ' + str(z_simplex[low]) + '\n')
    output.append([count, z_simplex[low]])
    return output

############################################################
# End of Nelder-Mead code
############################################################

############################################################
# Trial circuit for a system Raff was studying
############################################################
def trial_circuit_raff_2qubits(n, theta, entangler_map, meas_string=None, measurement=True):

    q = QuantumRegister("q", n)
    c = ClassicalRegister("c", n)
    trial_circuit = QuantumCircuit(q, c)

    trial_circuit.h(q)

    trial_circuit.ry(theta, q[0])
    
    for node in entangler_map:
            for j in entangler_map[node]:
                trial_circuit.cx(q[node], q[j])
    
    return trial_circuit

############################################################
# Trial circuit used for the anharmonoic oscillator for the n=3, n=4 cases
############################################################
def trial_circuit_anh_simple(n, theta, meas_string=None, measurement=False):
    q = QuantumRegister("q", n)
    c = ClassicalRegister("c", n)
    trial_circuit = QuantumCircuit(q, c)
    
    if meas_string is None:
        meas_string = [None for x in range(n)]

    trial_circuit.ry(theta[0], q[1])
    
    if measurement:
        for j in range(n):
            trial_circuit.measure(q[j], c[j])
    
    return trial_circuit

############################################################
# Trial circuit for the two qubit SUSY anharmonic oscillator
############################################################
def trial_circuit_SUSY_n2(n, theta, entangler_map, meas_string=None, measurement=False):
    q = QuantumRegister("q", n)
    c = ClassicalRegister("c", n)
    trial_circuit = QuantumCircuit(q, c)
    if meas_string is None:
        meas_string = [None for x in range(n)]

    trial_circuit.x(q[0])
    trial_circuit.ry(theta[0],q[1])
    trial_circuit.ry(theta[1],q[1])
    #trial_circuit.h(q[1])
    trial_circuit.barrier(q)
    if measurement:
        for j in range(n):
            trial_circuit.measure(q[j], c[j])
    
    return trial_circuit

############################################################
# Trial circuit for the three qubit SUSY anharmonic oscillator
############################################################
def trial_circuit_SUSY_n3(n, theta, entangler_map, meas_string=None, measurement=False):

    num_vars = n+1
    
    q = QuantumRegister("q", n)
    c = ClassicalRegister("c", n)
    trial_circuit = QuantumCircuit(q, c)
    if meas_string is None:
        meas_string = [None for x in range(n)]

    trial_circuit.x(q[0])

    #trial_circuit.h(q[1])
    trial_circuit.ry(theta[0], q[1])

    #trial_circuit.h(q[2])
    trial_circuit.ry(theta[1], q[2])
    trial_circuit.ry(theta[2], q[2])

    trial_circuit.cu3(theta[3], 0, 0, q[1], q[2])
    
    trial_circuit.barrier(q)
    if measurement:
        for j in range(n):
            trial_circuit.measure(q[j], c[j])
    
    return trial_circuit

############################################################
# Trial circuit for the four qubit SUSY anharmonic oscillator
############################################################
def trial_circuit_SUSY_n4(n, theta, entangler_map, meas_string=None, measurement=False):

    q = QuantumRegister("q", n)
    c = ClassicalRegister("c", n)
    trial_circuit = QuantumCircuit(q,c)

    if meas_string is None:
        meas_string = [None for x in range(n)]
        
    trial_circuit.x(q[0])
    
    trial_circuit.ry(theta[0], q[1])
    trial_circuit.ry(theta[1], q[2])
    
    trial_circuit.cu3(theta[2], 0, 0, q[1], q[2])
    
    trial_circuit.ry(theta[3], q[3])
    
    trial_circuit.cu3(theta[4], 0, 0, q[2], q[3])
    
    trial_circuit.ry(theta[5], q[3])
    
    trial_circuit.cu3(theta[6], 0, 0, q[1], q[2])
    
    trial_circuit.ry(theta[7], q[2])

    trial_circuit.barrier(q)

    if measurement:
        for j in range(n):
            trial_circuit.measure(q[j], c[j])
            
    return trial_circuit
