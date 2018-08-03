# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from QISKit ACQUA
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit_acqua.utils.variational_forms import VariationalForm


class VarFormRYSpecialDW(VariationalForm):
    """Couple of Y rotations. Useful for the double well, hopefully."""

    RYSpecialDW_CONFIGURATION = {
        'name': 'RYSpecialDW',
        'description': 'RYSpecialDW Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'ryspecialdw_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                'entanglement': {
                    'type': 'string',
                    'default': 'full',
                    'oneOf': [
                        {'enum': ['full', 'linear']}
                    ]
                },
                'entangler_map': {
                    'type': ['object', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.RYSpecialDW_CONFIGURATION.copy())
        self._num_qubits = 0
        self._depth = 0
        self._entangler_map = None
        self._initial_state = None

    def init_args(self, num_qubits, depth, entangler_map=None,
                  entanglement='full', initial_state=None):
        """
        Args:
            num_qubits (int) : number of qubits
            depth (int) : number of rotation layers, but will be set to 1
            entangler_map (dict) : dictionary of entangling gates, in the format
                                    { source : [list of targets] },
                                    or None for full entanglement.
            entanglement (str): 'full' or 'linear'
            initial_state (InitialState): an initial state object
        """
        depth = 4
        self._num_parameters = 5
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._num_qubits = num_qubits
        self._depth = depth
        if entangler_map is None:
            entangler_map = dict()
            #self._entangler_map = VariationalForm.get_entangler_map(entanglement, num_qubits)
        #else:
            #self._entangler_map = VariationalForm.validate_entangler_map(entangler_map, num_qubits)
        self._initial_state = initial_state

    def construct_circuit(self, parameters):
        """
        Construct the variational form, given its parameters.

        Args:
            parameters (numpy.ndarray): circuit parameters.

        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`

        Raises:
            ValueError: the number of parameters is incorrect.
        """
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        q = QuantumRegister(self._num_qubits, name='q')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
        else:
            circuit = QuantumCircuit(q)


        # try this for some good results circuit
        #circuit.u3(0.990248603589793,0,0,q[1])
        #circuit.u3(1.8787902135897931,0,0,q[2])
        #circuit.u3(-3.17478292,0,0,q[3])
        #circuit.cx(q[2],q[1])
        #circuit.u3(6.08966297,0,0,q[1])
        #circuit.cx(q[2],q[3])
        #circuit.u3(-0.19086582,0,0,q[2])

        circuit.u3(parameters[0],0,0,q[1])
        circuit.u3(parameters[1],0,0,q[2])
        circuit.u3(parameters[2],0,0,q[3])
        circuit.cx(q[2],q[1])
        circuit.u3(parameters[3],0,0,q[1])
        circuit.cx(q[2],q[3])
        circuit.u3(parameters[4],0,0,q[2])
        circuit.barrier(q)

        return circuit
