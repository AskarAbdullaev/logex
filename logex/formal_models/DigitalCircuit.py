from typing import Collection, Union, Callable
from itertools import product
import regex as re
import inspect

from . import tools

class DigitalCircuit():
    """
    A specific class to define a Digital Circuit.

    It consists of inputs: channel or channels, which receive an independent signal at each step.
    Internal states: internal variables, which change its values according to some logical rules.
    Outputs: channel or channels, which are read as the output of the circuit.

    There is no practical difference between states and outputs except for their names.

     Args:
        inputs (Collection[str] | str, None): 
            names of input channels, e.g. 'input' or ['in1', 'in2'], etc.
        outputs (Collection[str]):
            names of output channels, e.g. 'output', ['out1', 'out2'], etc.
        states (Collection[str]):
            names of internal circuit states, e.g. 'D', ['x', 'y'], etc.
        logical_rules (dict):
                dictionary of a specific form:
                    keys are names of states and output channels (if they are present)
                    values are functions that take exactly ONE argument of dict type and use 
                    the keys specified before to derive a logical step.
                Example of logical_rules:

                {
                'state_1': lambda x: x['input'] and not x['state_2']
                'state_2': lambda x: not x['input']
                'output': lambda x: not x['state_1] or not x['state_2']
                }
        """


    def __init__(self, inputs: Union[Collection[str], str, None], outputs: Union[Collection[str], str, None], states: Union[Collection[str], str, None],
                 logical_rules: dict[str: Callable], name: str = None):

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = 'Circuit'
        self.name = name
        
        # Process input channels:
        if not inputs:
            self.inputs = set()
        elif isinstance(inputs, str):
            self.inputs = {inputs}
        elif isinstance(inputs, Collection):
            for inp in inputs:
                assert isinstance(inp, str), f'input channel names must be str, not {type(inp)} ({inp})'
                assert len(inp) > 0, 'cannot accept empty string as an input channel name'
            self.inputs = set(inputs)
        else:
            raise AssertionError(f'inputs must be a str, Collection or None, not {type(inputs)}')
        
        # Process output channels:
        if not outputs:
            self.outputs = set()
        elif isinstance(outputs, str):
            self.outputs = {outputs}
        elif isinstance(outputs, Collection):
            for out in outputs:
                assert isinstance(out, str), f'output channel names must be str, not {type(out)} ({out})'
                assert len(out) > 0, 'cannot accept empty string as an output channel name'
            self.outputs = set(outputs)
        else:
            raise AssertionError(f'outputs must be a str, Collection or None, not {type(outputs)}')
        
        # Process state names:
        if not states:
            self.states = set()
        elif isinstance(states, str):
            self.states = {states}
        elif isinstance(states, Collection):
            for s in states:
                assert isinstance(s, str), f'state names must be str, not {type(s)} ({s})'
                assert len(s) > 0, 'cannot accept empty string as an state name'
            self.states = set(states)
        else:
            raise AssertionError(f'states must be a str, Collection or None, not {type(states)}')
  
        # Process logical rules:
        assert isinstance(logical_rules, dict), f'logical_rules must be a dict, not {type(logical_rules)}'
        for key, value in logical_rules.items():
            assert key in (self.states | self.outputs), f'logic_rules key "{key}" is neither in states nor in outputs'
            assert key not in self.inputs, 'input names cannot appear as logical_rules keys'
            if isinstance(value, bool):
                logical_rules[key] = lambda x: value
            elif isinstance(value, int):
                logical_rules[key] = lambda x: bool(value)
            elif isinstance(value, Callable):
                sig = [x for x, _ in inspect.signature(value).parameters.items()]
                assert len(sig) == 1, f'each function in logical_rules must take exactly one argument, not {sig}'
                variable = sig[0]
                code = ' ' + re.sub(r'^\s{1,100}', '', inspect.getsource(value))
                if f' {variable}' in code:
                    code_keys = re.findall(' ' + variable + r'\[([^\]]+)\]', code)
                    assert code_keys, f'argument {variable} in logical_rules[{key}] must be treated as a dict'
                    for code_key in code_keys:
                        assert code_key[1:-1] in (self.inputs | self.states), f'key "{code_key}" is neither in inputs nor in states, function code: {code}'

        self.rules = logical_rules

    def __call__(self, inputs: dict[str: Union[int, bool]]) -> dict[str: Union[int, bool]]:
        """
        Performs a step with a provided conditions:
        "inputs" is a dictionary with boolean values for input channels and internal states;

        Args:
           inputs (dict): boolean setting

        Returns:
            dict of boolean values for internal states and output channels after the step is performed
        """
        
        assert isinstance(inputs, dict), f'inputs must be a dict, not {type(inputs)}'

        for key, value in inputs.items():
            assert key in (self.states | self.inputs), f'dict key "{key}" is neither in states nor in inputs'
            # assert key not in self.outputs, 'output channel names cannot appear as dict keys'
            if isinstance(value, bool):
                pass
            elif isinstance(value, int):
                assert value in [0, 1], 'only 0 and 1 are accepted as integers for digital circuit'
        
        results = {}
        for key, function in self.rules.items():
            results.update({key: int(function(inputs))})
        return results
    
    def __str__(self):
        string = 'Digital Circuit:\n'
        string += f'Input channels: {self.inputs}\n'
        string += f'Output channels: {self.outputs}\n'
        string += f'Inner states: {self.states}\n'
        string += f'Rules:\n'

        for key, rule in self.rules.items():

            # Rule variable:
            sig = [x for x, _ in inspect.signature(rule).parameters.items()]
            variable = sig[0]

            # Rule code
            code = ' ' + re.sub(r'^\s{1,100}', '', inspect.getsource(rule))
            if 'lambda' in code:
                code = ''.join(re.split(r'lambda\s{1,5}' + variable + ':', code)[1:])
            elif 'def ' in code:
                code = ''.join(code.split('\n')[1:])

            code_keys = re.findall(' ' + variable + r'\[[^\]]+\]', code)
            for code_key in code_keys:
                code = code.replace(code_key, f' {code_key[3 + len(variable):-2]}')
            string += f'{key} =\n'
            string += code
            string += '\n' if code[-1] != '\n' else ''
        return string
    
    def __repr__(self):
        return str(self)
    
    def to_io_automaton(self, initial_state: str = None, order_of_states: list = None, order_of_inputs: list = None, order_of_outputs: list = None, name: str = None):
        """
        Derives I/O Automaton from the Digital Circuit.
        Order of states / inputs / outputs are important for the exact representation.
        For example, if there are 2 internal states 'x' and 'y', the default state names will be
        constructed as 'xy' with both values being 0 or 1.
        But you can provide an order_of_states = ['y', 'x'] and make state names follow 'yx' pattern.
        The same applies to input / output channels. Of course, if there is only one channel for input 
        or output: there is no need in ordering.
        Default ordering is always lexicographical.

        Args:
            initial_state (str, optional): what state to treat as an initial state. Defaults to None.
            order_of_states (list, optional): in what order to construct state names. Defaults to None.
            order_of_inputs (list, optional): in what order to show input channels. Defaults to None.
            order_of_outputs (list, optional): in what order to show output channels. Defaults to None.
            name (str, optional): a name for I/O Automaton. Defaults to None.

        Returns:
            instance of IOAutomaton
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'IO({self.name})'
        
        # Working with states
        if order_of_states:
            assert isinstance(order_of_states, list), f'order_of_states must be a list, not {type(order_of_states)}'
            assert not set(order_of_states) - self.states, f'order_of_states does not correspond to circuit states, must contain {self.states}'
            all_possible_states = set(product(*[[0,  1] for _ in range(len(self.states))]))
        elif not self.states:
            order_of_states = [tools.EMPTY_STATE]
            all_possible_states = {tools.EMPTY_STATE}
        else:
            order_of_states = sorted(list(self.states))
            all_possible_states = set(product(*[[0,  1] for _ in range(len(self.states))]))
        
        # Working with inputs
        if order_of_inputs:
            assert isinstance(order_of_inputs, list), f'order_of_inputs must be a list, not {type(order_of_inputs)}'
            assert not set(order_of_inputs) - self.inputs, f'order_of_inputs does not correspond to circuit inputs, must contain {self.inputs}'
            all_possible_inputs = set(product(*[[0, 1] for _ in range(len(self.inputs))]))
        elif not self.inputs:
            order_of_inputs = [tools.EMPTY_STATE]
            all_possible_inputs = {tools.EMPTY_STATE}
        else:
            order_of_inputs = sorted(list(self.inputs))
            all_possible_inputs = set(product(*[[0, 1] for _ in range(len(self.inputs))]))
        
        # Working with outputs
        if order_of_outputs:
            assert isinstance(order_of_outputs, list), f'order_of_states must be a list, not {type(order_of_outputs)}'
            assert not set(order_of_outputs) - self.outputs, f'order_of_outputs does not correspond to circuit outputs, must contain {self.outputs}'
            all_possible_outputs = set(product(*[[0, 1] for _ in range(len(self.outputs))]))
        elif not self.outputs:
            order_of_outputs = [tools.EMPTY_STATE]
            all_possible_outputs = {tools.EMPTY_STATE}
        else:
            order_of_outputs = sorted(list(self.outputs))
            all_possible_outputs = set(product(*[[0, 1] for _ in range(len(self.outputs))]))
        
        # Defining transitions
        transitions = set()
        for state in all_possible_states:
            for input_ in all_possible_inputs:

                input_dict = {}
                for state_name, state_value in zip(order_of_states, state):
                    if state_name != tools.EMPTY_STATE:
                        input_dict.update({state_name: state_value})
                for input_channel, input_value in zip(order_of_inputs, input_):
                    if input_channel != tools.EMPTY_STATE:
                        input_dict.update({input_channel: input_value})
                
                result = self(input_dict)
                result_state = [0 for _ in range(len(self.states))]
                result_output = [0 for _ in range(len(self.outputs))]
                for key, value in result.items():
                    if key in self.states:
                        result_state[order_of_states.index(key)] = value
                    elif key in self.outputs:
                        result_output[order_of_outputs.index(key)] = value

                result_output = result_output[:len(order_of_outputs)]
                result_state = result_state[:len(order_of_states)]
                
                if not self.inputs:
                    input_ = [tools.EMPTY_STATE]
                if not self.outputs:
                    result_output = [tools.EMPTY_STATE]
                transitions.add((
                    ''.join([str(s) for s in state]),
                    (''.join([str(i) for i in input_]), ''.join([str(o) for o in result_output])),
                    ''.join([str(s) for s in result_state] if self.states else tools.EMPTY_STATE)
                ))

        all_possible_outputs = set(product(*[[0, 1] for _ in range(len(order_of_outputs))]))
        all_possible_outputs = set(product(*[[0, 1] for _ in range(len(order_of_states))]))
        input_alphabet = {''.join([str(i) for i in input_]) for input_ in all_possible_inputs}
        output_alphabet = {''.join([str(o) for o in output]) for output in all_possible_outputs}
        states = {''.join([str(s) for s in state]) for state in all_possible_states}

        if not initial_state:
            initial_state = sorted(states)[0]
        else:
            assert isinstance(initial_state, str), f'initial state must be str, not {type(initial_state)}'
            assert initial_state in states, f'there is no such state "{initial_state}" in states'

        from . import IOAutomaton
        io_automaton = IOAutomaton(
            states,
            initial_state,
            transitions,
            input_alphabet,
            output_alphabet,
            name
        )

        return io_automaton
    
    def to_kripke(self, mode: str = 'i+s+o', linker: str = '',
                  order_of_states: list = None,
                  order_of_inputs: list = None,
                  order_of_outputs: list = None,
                  initial_state: str = None,
                  show_initial_states: bool = False,
                  name: str = None):
        """
        Derives a Kripke from a Digital Circuit.

        Parameters:
            name (str): Optional. How to entitle the new Kripke instance. 
            mode (str): mode of concatenation state, input and output. By default: "i+s+o".
            linker (str): a symbol to concatenate state names with. By default - empty string. By default: Kripke(previous name).
            order_of_states (list, optional): in what order to construct state names. Defaults to None.
            order_of_inputs (list, optional): in what order to show input channels. Defaults to None.
            order_of_outputs (list, optional): in what order to show output channels. Defaults to None.
            initial_state (str, optional): what state to treat as an initial state. Defaults to None.
            show_initial_states (bool): enables initial states in Kripke Structure. Defaults to False.
            name (str, optional): a name for I/O Automaton. Defaults to None.

        Returns:
            a new Kripke instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'Kripke({self.name})'
        assert isinstance(mode, str), f'mode must be a str, not {type(mode)}'
        assert len(mode.split('+')) == 3, f'mode must consist of letter "s", "i", "o" concatenated with a "+"'
        assert set(mode.split('+')) == {'i', 's', 'o'}, f'mode must consist of letter "s", "i", "o" concatenated with a "+"'
        assert isinstance(linker, str), f'linker must be a string'
        assert isinstance(show_initial_states, bool), f'show_initial_states must be bool, not {type(show_initial_states)}'

        io = self.to_io_automaton(initial_state='0' * len(self.states) if not initial_state else initial_state,
                                  order_of_inputs=order_of_inputs,
                                  order_of_outputs=order_of_outputs,
                                  order_of_states=order_of_states)
        kripke = io.to_kripke(name=name, mode=mode, linker=linker, no_annotations=True, no_initial_states=not show_initial_states)

        return kripke