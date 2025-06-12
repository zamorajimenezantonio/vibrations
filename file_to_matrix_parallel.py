import os
import re

filename = f"1500modes-full.spm"
#filename = "4in4out6modes.spm"
#filename = f"file2.spm"
#filename = f"file.spm"


filepath = os.path.join(os.getcwd(),f"inputs/{filename}")
text_file = open(filepath,"r")

file_lines = text_file.readlines()

def extract_text(file_lines, first_reference, second_reference = None):

    lines = []
    flag = None
    print(f"Looking for text lines between [{first_reference},{second_reference}]:")
    print(f"...")

    for line in file_lines:
        line = line.replace("\n"," ")
        if flag == True:
            lines.append(line)
        if re.match(first_reference, line): 
            flag = True
            print(f"first reference founded")
        elif second_reference != None:
            if re.match(second_reference, line):
                flag = False                
        
    number_of_lines_founded = len(lines)         
    print(f"\t{number_of_lines_founded} has been founded")
    if (second_reference == None) or (flag == True):
        print(f"EOF reached")
    elif (flag == False):
        print(f"second reference founded --> stop searching")

    return lines

#============================================================
#====================== CODE START ==========================
#============================================================

input_labels = extract_text(file_lines, r".*INPUT LABELS.*", r".*OUTPUT LABELS.*")
input_labels = input_labels[0:-1]
output_labels = extract_text(file_lines, r".*OUTPUT LABELS.*", r".*A MATRIX.*")
output_labels = output_labels[0:-1]
A_matrix = extract_text(file_lines, r".*A MATRIX.*", r".*B MATRIX.*")
B_matrix = extract_text(file_lines, r".*B MATRIX.*", r".*C MATRIX.*")
C_matrix = extract_text(file_lines, r".*C MATRIX.*", r".*D MATRIX.*")

import pandas as pd
import numpy as np

#%%

matrices={"A_matrix": A_matrix,
          "B_matrix": B_matrix,
          "C_matrix": C_matrix}

for key,value in matrices.items():
    value.pop(0)
    value.pop(-1)

#%%
del matrices

#%%
A_matrix = ''.join(A_matrix)
B_matrix = ''.join(B_matrix)
C_matrix = ''.join(C_matrix)

input_labels = (''.join(input_labels)).split()
output_labels = ''.join(output_labels)
output_labels = re.split(r'\s{2,}', output_labels)
output_labels.pop(0)
output_labels.pop(-1)

n_inputs = len(input_labels)
n_outputs = len(output_labels)
#

#%%
# A matrix preparation. It dimension should match (row, col) = (2*n_modes, 2*n_modes)
A_matrix = np.array(A_matrix.split())
A_matrix = A_matrix.astype(np.float64)

n_modes = int(np.sqrt(len(A_matrix))/2)

A_matrix = np.reshape(A_matrix,(2*n_modes,2*n_modes))

# B matrix preparation. It dimension should match (row, col) = (2*n_modes, 2*n_inputs)

B_matrix = np.array(B_matrix.split())
B_matrix = B_matrix.astype(np.float64)

B_matrix = np.reshape(B_matrix,(2*n_modes,n_inputs))

# C matrix preparation. It dimension should match (3*n_outputs, 2*n_modes)

C_matrix = np.array(C_matrix.split())
C_matrix = C_matrix.astype(np.float64)

C_matrix = np.reshape(C_matrix,(3*n_outputs,2*n_modes))

# *.spm file includes coefficients for velocity and acceleration calculation.
# C_matrix should be reduced to (row,col) = n_outputs, 2*n_modes
C_matrix = C_matrix[0:n_outputs,:]

#%% number of modes and mode simplification:
truncated_frequency = 70 #Hz
A_matrix21=A_matrix[int(len(A_matrix)/2):int(len(A_matrix)),0:int(len(A_matrix)/2)]
A_freq = np.pow(-np.diag(A_matrix21),0.5)/(2*np.pi)
filter_indexes = np.where(A_freq>truncated_frequency)
A_freq_truncated = A_freq[0:(filter_indexes[0])[0]]

# truncated model:

# A matrix truncation:
# 1st, A matrix is divided by nmodes x nmodes shape submatrices.
A_matrix11 = A_matrix[0:int(len(A_matrix)/2),0:int(len(A_matrix)/2)]
A_matrix12 = A_matrix[0:int(len(A_matrix)/2),int(len(A_matrix)/2):int(len(A_matrix))]
A_matrix21 = A_matrix[int(len(A_matrix)/2):int(len(A_matrix)),0:int(len(A_matrix)/2)]
A_matrix22 = A_matrix[int(len(A_matrix)/2):int(len(A_matrix)),int(len(A_matrix)/2):int(len(A_matrix))]
# then, the submatrices are truncated:
A_matrix11 = A_matrix11[0:(filter_indexes[0])[0],0:(filter_indexes[0])[0]]
A_matrix12 = A_matrix12[0:(filter_indexes[0])[0],0:(filter_indexes[0])[0]]
A_matrix21 = A_matrix21[0:(filter_indexes[0])[0],0:(filter_indexes[0])[0]]
A_matrix22 = A_matrix22[0:(filter_indexes[0])[0],0:(filter_indexes[0])[0]]
# finally, A matrix is built by concatenation:
A_matrix=np.concatenate((
    np.concatenate((A_matrix11,A_matrix12),axis=1),
    np.concatenate((A_matrix21,A_matrix22),axis=1)),axis=0)

# B matrix truncation:
# 1st. B matrix is subdivided:
B_matrix11 = B_matrix[0:int(len(B_matrix)/2)]
B_matrix12 = B_matrix[int(len(B_matrix)/2):int(len(B_matrix))]
# then, the submatrices are truncated:
B_matrix11 = B_matrix11[0:(filter_indexes[0])[0]]
B_matrix12 = B_matrix12[0:(filter_indexes[0])[0]]
# finally, B matrix is built by concatenation:
B_matrix=np.concatenate((B_matrix11,B_matrix12),axis=0)

# C matrix truncation:
# 1st. C matrix is subdivided:
C_matrix11 = C_matrix[:,0:int(len(np.transpose(C_matrix))/2)]
C_matrix21 = C_matrix[:,int(len(np.transpose(C_matrix))/2):int(len(np.transpose(C_matrix)))]
# then, the submatrices are truncated:
C_matrix11 = C_matrix11[:,0:(filter_indexes[0])[0]]
C_matrix21 = C_matrix21[:,0:(filter_indexes[0])[0]]
# finally, B matrix is built by concatenation:
C_matrix=np.concatenate((C_matrix11,C_matrix21),axis=1)

#%% ===============================================
#   =============CALCULATING BODE==================
#   ===============================================

# prepraring to calculate bode for each input and output
import subprocess
import scipy.io as io
import matplotlib.pyplot as plt

# discretized frequency array:

f0 = 1 # Hz
w0 = f0/(2*np.pi) # rad/s
#fend = 70 # Hz
fend = truncated_frequency
wend = 2*np.pi*fend # rad/s

finc = 0.01 # Hz
n=np.log((fend-f0)/finc)/np.log(2)
n=int(n)

# creating a bode container (row for inputs and columns for outputs)
n_inputs = len(input_labels)
n_outputs = len(output_labels)
bode_matrix=np.ndarray(shape=(n_inputs,n_outputs),dtype=object)

# row for-loop for each one of the inputs
first_input=input_labels.index(input_labels[0])
last_input=input_labels.index(input_labels[5])
first_output=output_labels.index(output_labels[0])
last_output=output_labels.index(output_labels[0])

n_inputs=np.linspace(first_input,last_input,1+last_input-first_input).astype(int)
n_outputs=np.linspace(first_output,last_output,1+last_output-first_output).astype(int)


def matlab_SISO_bode(input, output, input_labels, output_labels, A_matrix, B_matrix, C_matrix,
                     w0, wend, n, matlabcores, bode_matrix):

    print(f"Generating {output_labels[output]} output due to {input_labels[input]} input...")
    # Selection of input and output to calculate H(s):
    # 1st input (position 0):
    #input = 0
    # 1st outut (position 0):
    # output = 0

    # B matrix must be now a column vector so it needs to be transposed:
    B_matrix_i = B_matrix[:,input]
    B_matrix_i = np.reshape(B_matrix_i,(len(B_matrix_i),1))

    # C matrix must be now a row vector
    C_matrix_i = C_matrix[output,:]

    # D matrix as:
    # D_matrix = np.zeros((1,1))
    # is defined in Matlab

    # Generating reduced space-state matrices for each input and output (SISO)
    dict_reduced_matrix = {'A_matrix': A_matrix,
                        'B_matrix': B_matrix_i,
                        'C_matrix': C_matrix_i} 

    # saving matrices to be loaded using Matlab
    for A in ['A','B','C']:
        io.savemat(f"{A}_{input}_{output}.mat", {A: dict_reduced_matrix[f"{A}_matrix"]})


    matlab_command = f"matlab -nodisplay -nosplash -nojvm -minimize -r -wait [f,mag]=calculateBode({w0},{wend},{n},{input},{output},{matlabcores});exit"
    res= subprocess.call(matlab_command, shell=True)

    bode_data = np.loadtxt(f"bode_data_{input}_{output}.csv",delimiter=',')
    bode_matrix[input,output]=bode_data
    print(f"...{output_labels[output]} output due to {input_labels[input]} input done")

    return bode_matrix

# Parallelize:
from joblib import Parallel, delayed

parallel_obj = Parallel(n_jobs=6, prefer='threads')
my_result = parallel_obj(delayed(matlab_SISO_bode)(input, output, input_labels, output_labels,
                                                   A_matrix, B_matrix, C_matrix,
                                                   w0, wend, n, 
                                                   matlabcores=1, 
                                                   bode_matrix=bode_matrix) 
                                                   for input in n_inputs for output in n_outputs)

for elem in my_result:
    for row in range(len(input_labels)):
        for column in range(len(output_labels)):
            try:
                if (elem[row,column])==None:
                    elem[row,column]=0
            except:
                pass

bode_matrix = np.sum(my_result,keepdims=True)[0]

plt.title(input_labels[0])
plt.xlim([1,150])
plt.loglog((bode_matrix[0,0])[:,0],(bode_matrix[0,0])[:,1], color='r', label=output_labels[132])
plt.loglog((bode_matrix[0,1])[:,0],(bode_matrix[0,1])[:,1], color='g', label=output_labels[133])
plt.loglog((bode_matrix[0,2])[:,0],(bode_matrix[0,2])[:,1], color='b', label=output_labels[134])
plt.legend()
plt.grid(which='both')
plt.show()

due_to_fx_rms = np.sqrt(np.power((bode_matrix[54,132])[:,1],2)+
                 np.power((bode_matrix[54,133])[:,1],2)+
                 np.power((bode_matrix[54,134])[:,1],2))
due_to_fy_rms = np.sqrt(np.power((bode_matrix[55,132])[:,1],2)+
                 np.power((bode_matrix[55,133])[:,1],2)+
                 np.power((bode_matrix[55,134])[:,1],2))
due_to_fz_rms = np.sqrt(np.power((bode_matrix[56,132])[:,1],2)+
                 np.power((bode_matrix[56,133])[:,1],2)+
                 np.power((bode_matrix[56,134])[:,1],2))
total_rms = np.sqrt(np.power(due_to_fx_rms,2)+
                 np.power(due_to_fy_rms,2)+
                 np.power(due_to_fz_rms,2))

plt.semilogy((bode_matrix[54,132])[:,0],total_rms, color='black')
plt.grid(which='both')
plt.show()

plt.semilogy((bode_matrix[54,132])[:,0],total_rms/total_rms[0], color='black')
plt.grid(which='both')
plt.show()