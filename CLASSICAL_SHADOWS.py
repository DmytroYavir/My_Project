
import pennylane as qml
import pennylane.numpy as np
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(666)

def calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits):

    # applying the single-qubit Clifford circuit is equivalent to measuring a Pauli
    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]

    # sample random Pauli measurements uniformly, where 0,1,2 = X,Y,Z
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits))
    outcomes = np.zeros((shadow_size, num_qubits))

    for ns in range(shadow_size):
        # for each snapshot, add a random Pauli observable at each location
        obs = [unitary_ensemble[int(unitary_ids[ns, i])](i) for i in range(num_qubits)]
        outcomes[ns, :] = circuit_template(params, observable=obs)


    # combine the computational basis outcomes and the sampled unitaries
    return (outcomes, unitary_ids)




def snapshot_state(b_list, obs_list):

    num_qubits = len(b_list)

    # computational basis states
    zero_state = np.array([[1, 0], [0, 0]])
    one_state = np.array([[0, 0], [0, 1]])

    # local qubit unitaries
    phase_z = np.array([[1, 0], [0, -1j]], dtype=complex)
    hadamard = qml.matrix(qml.Hadamard(0))
    identity = qml.matrix(qml.Identity(0))

    # undo the rotations that were added implicitly to the circuit for the Pauli measurements
    unitaries = [hadamard, hadamard @ phase_z, identity]

    # reconstructing the snapshot state from local Pauli measurements
    rho_snapshot = [1]
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = unitaries[int(obs_list[i])]
        # applying Eq. (S44)
        local_rho = 3 * (U.conj().T @ state @ U) - identity
        rho_snapshot = np.kron(rho_snapshot, local_rho)

    return rho_snapshot


def shadow_state_reconstruction(shadow):

    num_snapshots, num_qubits = shadow[0].shape

    # classical values
    b_lists, obs_lists = shadow

    # Averaging over snapshot states.
    shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])

    return shadow_rho / num_snapshots


num_qubits=5
dev = qml.device("default.qubit", wires=num_qubits, shots=1)

@qml.qnode(dev)
def local_qubit_rotation_circuit(params, **kwargs):
    observables = kwargs.pop("observable")
    #print(observables)
    #for w in dev.wires:

    #qml.RX(params, wires=0)
    #qml.RX(np.pi, wires=1)
    #qml.RY(np.pi, wires=0)
    #qml.RY(np.pi, wires=1)
    #qml.CNOT(wires=[0, 1])
    #qml.RY(np.pi, wires=0)
    #qml.RY(np.pi, wires=1)
    #qml.RX(np.pi, wires=0)
    #qml.RX(np.pi, wires=1)

    qml.U3(params, 0, np.pi, wires=1)
    qml.CNOT(wires=[1, 3])
    qml.CNOT(wires=[3, 4])
    qml.RZ(np.pi/2, wires=1)

    #qml.X(wires=0)
    #qml.Hadamard(0)
    #qml.CNOT(wires=[0, 1])
    #qml.RX(params, wires=0)
    #qml.RY(params, wires=1)



    #qml.Hadamard(0)
    #qml.CNOT(wires=[0, 1])
    return [qml.expval(o) for o in observables]

k=1
N=2
lr=1
S2_all=[]
all_params=[]
S_page=k*np.log(2)-1/(2**(N-2*k+1))
num_snapshots = 1000
params = np.linspace(0, 2*np.pi, 50)

for i in range(len(params)):
    shadow = calculate_classical_shadow(
        local_qubit_rotation_circuit, params[i], num_snapshots, num_qubits
    )

    shadow_state = shadow_state_reconstruction(shadow)
    #print(np.round(shadow_state, decimals=6))

    S2 = (-np.log2(np.trace(shadow_state**2)))
    #print(S2)
    #if S2<S_page:
    S2_all.append(S2)
    all_params.append(params[i])
print(S2_all)
'''
plt.scatter(x=all_params,
            y=S2_all,
            #c=steps,
            #cmap=plt.cm.RdYlBu
            )
plt.show()
'''

plt.figure()

plt.plot(all_params, S2_all)
plt.show()


# get x and y vectors
x = all_params
y = S2_all

# calculate polynomial
z = np.polyfit(x, y, 3)
f = np.poly1d(z)

# calculate new x's and y's
x_new = np.linspace(x[0], x[-1], 50)
y_new = f(x_new)

plt.plot(x,y,'o', x_new, y_new)
plt.xlim([x[0]-1, x[-1] + 1 ])
plt.show()

