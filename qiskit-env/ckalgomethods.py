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
"""Algorithm function for retrieving an instance of the algorithm etc. Takes code from algomethods.py in ACQUA"""

from qiskit_acqua.algorithmerror import AlgorithmError
from qiskit_acqua._discover import (_discover_on_demand,
                                  local_algorithms,
                                  get_algorithm_instance)
from qiskit_acqua.utils.jsonutils import convert_dict_to_json,convert_json_to_dict
from qiskit_acqua.parser._inputparser import InputParser
from qiskit_acqua.input import get_input_instance
import logging
import json
import copy

logger = logging.getLogger(__name__)

def get_algorithm_ck(params, algo_input=None, json_output=False):
    """
    Retrieve algorithm as named in params, using params and algo_input as input data
    and returning a result dictionary
    Args:
        params (dict): Dictionary of params for algo and dependent objects
        algo_input(algorithminput): Main input data for algorithm. Optional, an algo may run entirely from params
        json_output(bool): False for regular python dictionary return, True for json conversion
    Returns:
        Result dictionary containing result of algorithm computation
    """
    _discover_on_demand()

    inputparser = InputParser(params)
    inputparser.parse()
    inputparser.validate_merge_defaults()
    logger.debug('Algorithm Input: {}'.format(json.dumps(inputparser.get_sections(), sort_keys=True, indent=4)))

    algo_name = inputparser.get_section_property(InputParser.ALGORITHM, InputParser.NAME)
    if algo_name is None:
        raise AlgorithmError('Missing algorithm name')

    if algo_name not in local_algorithms():
        raise AlgorithmError('Algorithm "{0}" missing in local algorithms'.format(algo_name))

    backend_cfg = None
    backend = inputparser.get_section_property(InputParser.BACKEND, InputParser.NAME)
    if backend is not None:
        backend_cfg = {k: v for k, v in inputparser.get_section(InputParser.BACKEND).items() if k != 'name'}
        backend_cfg['backend'] = backend

    algorithm = get_algorithm_instance(algo_name)
    algorithm.random_seed = inputparser.get_section_property(InputParser.PROBLEM, 'random_seed')
    if backend_cfg is not None:
        algorithm.setup_quantum_backend(**backend_cfg)

    algo_params = copy.deepcopy(inputparser.get_sections())

    if algo_input is None:
        input_name = inputparser.get_section_property('input', InputParser.NAME)
        if input_name is not None:
            algo_input = get_input_instance(input_name)
            input_params = copy.deepcopy(inputparser.get_section_properties('input'))
            del input_params[InputParser.NAME]
            convert_json_to_dict(input_params)
            algo_input.from_params(input_params)

    algorithm.init_params(algo_params, algo_input)

    return algorithm
