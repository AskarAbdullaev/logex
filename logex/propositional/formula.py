from typing import Union, Sequence
import ast
import time

# from tqdm import tqdm
import regex as re
from pysat.solvers import Solver
import numpy as np

from . import boolean
from . import const

def get_outermost_connective(string: str, _clean: bool = True) -> tuple[str, tuple[str, str]]:
    """
    Takes a string, cleans it and finds the outermost logical connective with respect to the
    conventional precedence rules;
    Returns tuple of the form: 
        outermost_connective, (left-hand side, right-hand side)
    """
    assert isinstance(string, str), f"only str / Formula types are accepted, not {type(string)}"
    
    if _clean:
        string = clean(string)

    found = set(re.findall(const.COMBINED, string))
    if not found:
        return '', (string.replace('(', '').replace(')', ''), '')
    if len(found) == 1:
        
        conn = found.pop()
        split_string = string.replace('(', '').replace(')', '').split(conn)
        left, outer, right = split_string[0], conn, conn.join(split_string[1:])
        return outer, (left, right)

    special_symbol = '@'
    enclosures = []
    counter, _, new_enclosure = 0, np.zeros(len(string)), ''
    for i, symbol in enumerate(string):
        if symbol == '(':
            counter += 1
            if counter == 1:
                new_enclosure = '('
            else:
                new_enclosure += '('
        elif symbol == ')':
            counter -= 1
            if counter == 0:
                new_enclosure += ')'
                enclosures.append(new_enclosure[:])
                new_enclosure = ''
            else:
                new_enclosure += ')'
        elif counter >= 1:
            new_enclosure += symbol

    replacement_dict = {}
    for i, enclosure in enumerate(enclosures):
        string = string.replace(enclosure, special_symbol + chr(9312 + i))
        replacement_dict.update({special_symbol + chr(9312 + i): enclosure})  # store enclosure replacements

    split_string = [x for x in re.split(const.COMBINED, string) if x]  # split by logical connectives

    if not any(x in split_string for x in const.CONNECTIVES):
        # If there are no logical connectives - return the atomic formula
        outermost_connective = ''
        left_part, right_part = ''.join(split_string), ''
    elif not any(x in split_string for x in const.CONNECTIVES[1:]) and split_string[0] == const.NOT:
        # If the only connective is NOT in the beginning
        outermost_connective = const.NOT
        left_part, right_part = '', ''.join(split_string[1:])
    else:
        # Else find the first connective by precedence and split the formula at its index
        for connective in const.CONNECTIVES[1:]:
            for i, symbol in enumerate(split_string):
                if symbol == connective:
                    outermost_connective = connective
                    left_part, right_part = ''.join(split_string[:i]), ''.join(split_string[i + 1:])

    # Back substitution of enclosures
    for special_symbol, enclosure in replacement_dict.items():
        left_part = left_part.replace(special_symbol, enclosure)
        right_part = right_part.replace(special_symbol, enclosure)

    return outermost_connective, (left_part, right_part)


def clean(string: str):
    """
    Small pipeline for string cleaning
    """

    assert isinstance(string, str), f"only str type is accepted, not {type(string)}"
    cleaned_str = standardize(string)
    cleaned_str = cleaned_str.replace(' ', '').replace('\t', '').replace('->', '@').replace('-', '_').replace('@', '->')
    cleaned_str = ''.join([s.split('%')[0] for s in cleaned_str.split('\n')])
    cleaned_str = clean_parenthesis(cleaned_str)
    return cleaned_str

def standardize(string: str):
    """
    Making operations standard
    """
    assert isinstance(string, str), f"only str type is accepted, not {type(string)}"
    standard_str = string[:]
    for symbol, standard_symbol in const.CONNECTIVES_DICT.items():
        if symbol in standard_str:
            standard_str = standard_str.replace(symbol, standard_symbol)
    return standard_str

    
def clean_parenthesis(string: str, _clean: bool = True):
    """
    Removing redundant parenthesis

    if _clean = False, supresses initial cleaning
    """
    assert isinstance(string, str), f"only str type is accepted, not {type(string)}"
    assert isinstance(_clean, bool), f"_clean must be boolean, not {type(_clean)}"

    cleaned_string = string[:]
    if _clean:
        cleaned_string = standardize(cleaned_string)

    counter = 0
    for symbol in cleaned_string:
        if symbol == '(':
            counter += 1
        elif symbol == ')':
            counter -= 1
    if counter != 0:
        raise ValueError('Parenthesis are not matching!')
    all_enclosures = []
    for i, symbol in enumerate(cleaned_string):
        if symbol == '(':
            new_enclosure = '('
            counter = 1
            for symbol_ in cleaned_string[i + 1:]:
                if symbol_ == '(':
                    counter += 1
                elif symbol_ == ')':
                    counter -= 1
                new_enclosure += symbol_
                if counter == 0:
                    break
            if counter != 0:
                raise ValueError('Parenthesis are not matching!')
            all_enclosures.append(new_enclosure)

    if cleaned_string in all_enclosures:
        cleaned_string = cleaned_string[1:-1]

    for enclosure in all_enclosures:
        if '(' + enclosure + ')' in all_enclosures:
            cleaned_string = cleaned_string.replace('(' + enclosure + ')', enclosure)

    for enclosure in all_enclosures:
        if not any(connective in enclosure for connective in const.CONNECTIVES[1:]):
            cleaned_string = cleaned_string.replace(enclosure, enclosure[1:-1])

    while 2 * const.NOT in cleaned_string:
        cleaned_string = cleaned_string.replace(2 * const.NOT, '')
    return cleaned_string

def _clean_ast(string,  _clean: bool = True):
    """
    Cleaning of experssion using AST
    """
    assert isinstance(string, str), f"only str type is accepted, not {type(string)}"
    assert isinstance(_clean, bool), f"_clean must be boolean, not {type(_clean)}"

    string = string.replace(' ', '')
    if not string:
        return string
    
    if _clean:
        string = clean(string)

    for c, r in const.INTERPRETATION.items():
        string = string.replace(c, r)

    tree = ast.parse(string, mode='eval')
    root = tree.body
    string = ast.unparse(root)

    for c, r in const.BACK.items():
        string = string.replace(c, r)

    return string

def _get_outermost_operation(string, _clean: bool = True):
    """
    Getting the outermost operation using AST

    If _clean = False, supresses initial cleaning
    """
    assert isinstance(string, str), f"only str type is accepted, not {type(string)}"
    assert isinstance(_clean, bool), f"_clean must be boolean, not {type(_clean)}"

    if _clean:
        string = clean(string)

    found = re.findall(const.COMBINED, string)
    if not found:
        return '', (string.replace('(', '').replace(')', ''), '')
    if len(found) == 1:
        return found[0], (string.split(found[0])[0].replace('(', '').replace(')', ''),
                          string.split(found[0])[1].replace('(', '').replace(')', ''))
    
    for c, r in const.INTERPRETATION.items():
        string = string.replace(c, r)

    tree = ast.parse(string, mode='eval')
    root = tree.body

    if isinstance(root, ast.BinOp):
        operator = type(root.op).__name__
        left = ast.unparse(root.left)
        right = ast.unparse(root.right) 
    elif isinstance(root, ast.UnaryOp):
        operator = type(root.op).__name__
        left = ''
        right = ast.unparse(root.operand)
    elif isinstance(root, ast.Compare):
        operator = type(root.ops[0]).__name__
        left = ast.unparse(root.left)
        right = ast.unparse(root.comparators[0])
    else:
        operator, left, right = '', string, ''

    if operator:
        operator = const.AST_NAMES[operator]
    for c, r in const.BACK.items():
        left = left.replace(c, r)
        right = right.replace(c, r)

    return operator, (left, right)

def variables(formula: str):
    """
    Extracts variables from expression
    """
    assert isinstance(formula, str), f"only str type is accepted, not {type(formula)}"
    string = formula[:]
    string = clean(string)
    for symbol in const.CONNECTIVES + ['(', ')', const.TRUE, const.FALSE]:
        string = string.replace(symbol, ' ')
    string = re.sub(r'\s{2,200}', ' ', string)
    _variables = set(string.split(' '))
    return sorted([v for v in _variables if v])

def literals(formula: str):
    """
    Exctracts lliterals from expression
    """
    assert isinstance(formula, str), f"only str type is accepted, not {type(formula)}"
    string = formula[:]
    string = clean(string)
    for symbol in const.CONNECTIVES[1:] + ['(', ')', const.TRUE, const.FALSE]:
        string = string.replace(symbol, ' ')
    string = re.sub(r'\s{2,200}', ' ', string)
    _literals = set(string.split(' '))
    return sorted([v for v in _literals if v], key=lambda v: v.replace(const.NOT, ''))


class Formula():
    """
    Main datatype for propositional expressions

    Args:
            f (str): the formula itself
            as_cnf (bool, optional): treat as a cnf. Defaults to False.
            as_dnf (bool, optional): treat as a dnf. Defaults to False.
            as_aig (bool, optional): treat as an aig. Defaults to False.
    """

    def __init__(self, f: str, as_cnf: bool = False, as_dnf: bool = False, as_aig: bool = False):
        
        assert isinstance(f, str), f'formula must be provided a str input, not {type(f)}'
        assert isinstance(as_cnf, bool), f"as_cnf must be boolean, not {type(as_cnf)}"
        assert isinstance(as_dnf, bool), f"as_dnf must be boolean, not {type(as_dnf)}"
        assert isinstance(as_aig, bool), f"as_cnf must be boolean, not {type(as_aig)}"
        assert sum([as_cnf, as_dnf, as_aig]) <= 1, "only one marker can be active at a time: as_cnf / as_dnf / as_aig"

        self.original = f[:]
        self.string = clean(f)
        self.variables = variables(self.string)
        self.literals = literals(self.string)
        self.n_vars = len(self.variables)
        self._subformulas = "Not yet computed, use .get_subformulas()"
        self._nnf = "Not yet computed, use .to_nnf()"
        self._cnf = "Not yet computed, use .to_cnf()"
        self._aig = "Not yet computed, use .to_aig()"
        self._dnf = "Not yet computed, use .to_dnf()"
        self.is_nnf = "Not yet computed"
        self.is_cnf = "Not yet computed, use .check_cnf()"
        self.is_dnf = "Not yet computed, use .check_dnf()"
        self.is_aig = "Not yet computed, use .check_aig()"
        self._sat = "Not yet computed, use .is_sat()"
        self._unsat = "Not yet computed, use .is_unsat()"
        self._valid = "Not yet computed, use .is_valid()"
        self._refutable = "Not yet computed, use .is_refutable()"
        self._model = "Not yet computed, use .get_model()"
        self._models = "Not yet computed, use .get_model()"
        self._falsifying = "Not yet computed, use .get_falsifying()"
        self._falsifyings = "Not yet computed, use .get_falsifying()"
        self._truth_table = "Not yet computed, use .get_truth_table()"

        if const.XOR in self.string or const.EQUIVALENT in self.string or const.IMPLIES in self.string:
            self.is_cnf = False
            self.is_nnf = False
            self.is_dnf = False
            self.is_aig = False
        else:
            if const.OR in self.string:
                self.is_aig = False
            else:
                self.is_aig = True
                self._aig = self
            
            if const.NOT + '(' in self.string:
                self.is_nnf, self.is_cnf, self.is_dnf = False, False, False
            else:
                self.is_nnf = True
                self._nnf = self

        if self.n_vars == 0:
            self._model, self._models, self._falsifying, self._falsifyings = None, set(), None, set()
            self._truth_table = []
            value = self('')
            if value:
                self._valid, self._sat, self._unsat, self._refutable = True, True, False, False
            else:
                self._valid, self._sat, self._unsat, self._refutable = False, False, True, True

        if as_cnf:
            self._nnf = self
            self._cnf = self

            if self.n_vars == 0:
                self.clauses = {const.TRUE if self.is_valid() else const.FALSE}
                self.clauses_qdmacs = []
            else:
                self.clauses = set(self.string.replace('(', '').replace(')', '').split(const.AND))
                clauses_valid = []
                for clause in self.clauses:
                    if len(set(clause.split(const.OR))) == len(set(clause.replace(const.NOT, '').split(const.OR))):
                        clauses_valid.append(clause)
                self.clauses = set(clauses_valid)
                if not self.clauses:
                    self._valid = True
                    self._sat = True
                    self._refutable = False
                    self._unsat = False
                    self.clauses = {const.TRUE}
                    self._model = {v: True for v in self.variables}
                    self._models = {self._assignment_to_str(self._model)}
                    self._falsifying = None
                    self._falsifyings = set()
                else:
                    self._valid = False
                    self._refutable = True
                    self._falsifying = {v: False for v in self.variables}
                    for v in list(self.clauses)[0].split(const.OR):
                        self._falsifying.update({v: False if const.NOT not in v else True})
                    self._falsifyings = {self._assignment_to_str(self._falsifying)}
                self.literals = literals(self.string)
                self.clauses_qdmacs = []
                for clause in self.clauses:
                    qdimacs = sorted([self.variables.index(c) + 1 if '!' not in c else -(self.variables.index(c[1:]) + 1) 
                                    for c in clause.split('|')], key=abs)
                    self.clauses_qdmacs.append(qdimacs)
                
        
        if as_dnf:
            self._nnf = self
            self._dnf = self
            self.cubes = self.string.replace('(', '').replace(')', '').split(const.OR)
            self.literals = self.string.replace('(', '.').replace(')', '.').replace(const.AND, '.').replace(const.OR, '.').split('.')
            self.literals = sorted({l for l in self.literals if l}, key=lambda x: x.replace(const.NOT, ''))

    def __str__(self):
        return self.string
    
    def __repr__(self):
        return f'<Formula: {self.string}, vars: ({",".join(v for v in self.variables)})>'

    def __call__(self, assignment: Union[dict, Sequence]):
        """
        If assignemnt is not an explicit dictionary, it will parsed entry by entry
        and assigned to variables in lexicographical order.
        The order of variables can be checked by formula.variables.
        """

        assert len(assignment) == self.n_vars, f'there are {self.n_vars} variables, but {len(assignment)} values were given'
        if isinstance(assignment, Sequence):
            assignment = {variable: boolean._boolean(value) 
                          for variable, value in zip(self.variables, assignment)}
        elif isinstance(assignment, dict):
            assignment = {key: boolean._boolean(value) for key, value in assignment.items()}

        string = self._assignment_to_str(assignment)
        if string in self._models:
            return True
        elif string in self._falsifyings:
            return False

        result = self.string[:]
        for symbol, function in const.INTERPRETATION.items():
            result = result.replace(symbol, function)
        
        result = eval(result, {'boolean': boolean}, assignment)
        if isinstance(result, boolean._boolean):
            result = result.value

        if result:

            # SAT is found, model is found
            self._sat = True
            self._unsat = False
            self._model = {v: int(str(a)) for v, a in assignment.items()}

            if isinstance(self._models, str):
                self._models = set()
            self._models.add(self._assignment_to_str(assignment))

        else:
            # Refutable if found: falsifying is found
            self._valid = False
            self._refutable = True
            self._falsifying = {v: int(str(a)) for v, a in assignment.items()}

            if isinstance(self._falsifyings, str):
                self._falsifyings = set()
            self._falsifyings.add(self._assignment_to_str(assignment))
        
        return bool(result)
    
    def _assignment_to_str(self, assignment: dict):
        s = ''.join([str(int(assignment[v])) if isinstance(assignment[v], bool) 
                     else str(assignment[v]) for v in self.variables])
        return s
    
    def _str_to_assignment(self, string: str):
        a = {v: l == '1' for l, v in zip(string, self.variables)}
        return a
    
    def get_subformulas(self):
        """
        Allows to get a collection of all valid subformulas
        """

        if not isinstance(self._subformulas, str):
            return self._subformulas
        
        self.is_cnf, self.is_dnf = False, False
        self._subformulas = set()
        self._get_subformulas(self.string)
        if self.is_cnf:
            self._cnf = self
        if self.is_dnf:
            self._dnf = self
        return self._subformulas
    
    def _get_subformulas(self, f: str, previous_outer: str = None):

        outer, (l, r) = _get_outermost_operation(f, _clean = False)
        
        if self.is_nnf and outer == const.AND and previous_outer == const.OR:
            self.is_cnf = False
        if self.is_nnf and outer == const.OR and previous_outer == const.AND:
            self.is_dnf = False

        if not outer:
            return

        self._subformulas.add((f, (l, outer, r)))
        self._get_subformulas(_clean_ast(l, _clean=False), previous_outer=outer)
        self._get_subformulas(_clean_ast(r, _clean=False), previous_outer=outer)

    def simplify(self):
        """
        Allows to get rid of EQUIVALENCE, XOR, IMPLICATION and push negation inside.
        """

        if not isinstance(self._nnf, str):
            return self._nnf

        result = self.string[:]
        iteration, stop = 0, False

        # with tqdm(desc='Converting to NNF: ', unit=' rules applied') as p_bar:
        while not stop:

                stop = True
                subformulas = self.get_subformulas() if iteration == 0 else Formula(result).get_subformulas()


                for sf in sorted(subformulas, key=lambda x: len(x[0]), reverse=True):
                    l, o, r = sf[1]
                    r_o = ''
                    
                    match o:

                        case const.XOR:
                            # p_bar.update(1)
                            replacement = f'(({l}){const.OR}({r})){const.AND}({const.NOT}({l}){const.OR}{const.NOT}({r}))'
                            result = result.replace(sf[0], replacement)
                            
                        case const.EQUIVALENT:
                            # p_bar.update(1)
                            replacement = f'({const.NOT}({l}){const.OR}({r})){const.AND}(({l}){const.OR}{const.NOT}({r}))'
                            result = result.replace(sf[0], replacement)
                            
                        case const.IMPLIES:
                            # p_bar.update(1)
                            replacement = f'({const.NOT}({l}){const.OR}({r}))'
                            result = result.replace(sf[0], replacement)
                    
                        case const.NOT:
                            
                            r_o, (r_l, r_r) = _get_outermost_operation(r, _clean = False)

                            match r_o:

                                case const.AND:
                                    # p_bar.update(1)
                                    replacement = f'({const.NOT}({r_l}){const.OR}{const.NOT}({r_r}))'
                                    result = result.replace(sf[0], replacement)
                                    
                                case const.OR:
                                    # p_bar.update(1)
                                    replacement = f'({const.NOT}({r_l}){const.AND}{const.NOT}({r_r}))'
                                    result = result.replace(sf[0], replacement)
                                    
                                case const.NOT:
                                    # p_bar.update(1)
                                    replacement = f'({r_r})'
                                    result = result.replace(sf[0], replacement)
                                    
                                    
                    if o in {const.XOR, const.EQUIVALENT, const.IMPLIES} or r_o in {const.OR, const.AND}:
                        stop = False
                        iteration += 1
                        result = _clean_ast(result, _clean = False)

        result = Formula(result)
        self._nnf = result
        return self._nnf


    def to_nnf(self):
        """
        Allows to get rid of EQUIVALENCE, XOR, IMPLICATION and push negation inside.
        (Negation Normal Form)
        """

        if not isinstance(self._nnf, str):
            return self._nnf
            
        result = self.simplify()
        self._nnf = result

        return self._nnf
    
    def to_cnf(self):
        """
        Allows to construct Conjuction Normal Form (CNF)
        """

        if not isinstance(self._cnf, str):
            return self._cnf

        result = self.to_nnf().string
        stop, iteration = False, 0

        # with tqdm(desc='Converting to CNF: ', unit=' rules applied') as p_bar:
            
        while not stop:
                # print('It ', iteration)

                stop = True
                subformulas = self.to_nnf().get_subformulas() if iteration == 0 else Formula(result).get_subformulas()


                for sf in sorted(subformulas, key=lambda x: len(x[0]), reverse=False):
                    l, o, r = sf[1]

                    if o == const.OR:

                        if not const.AND in r and not const.AND in l:
                            continue

                        r_o, (r_l, r_r) = _get_outermost_operation(r, _clean = False)
                        l_o, (l_l, l_r) = _get_outermost_operation(l, _clean = False)

                        if r_o == const.AND:
                            # p_bar.update(1)
                            stop = False
                            replacement = f'(({l}){const.OR}({r_l})){const.AND}(({l}){const.OR}({r_r}))'
                            result = result.replace(sf[0], replacement)
                        
                        elif l_o == const.AND:
                            # p_bar.update(1)
                            stop = False
                            replacement = f'(({l_l}){const.OR}({r})){const.AND}(({l_r}){const.OR}({r}))'
                            result = result.replace(sf[0], replacement)
                iteration += 1
                result = _clean_ast(result, _clean=False)
                result = result.replace(2 * const.NOT, '')
                # print(result)

        result = list(set(result.replace('(', '').replace(')', '').split(const.AND)))
        result = set(filter(lambda clause: len(set(clause.replace(const.NOT, '').split(const.OR))) == len(set(clause.split(const.OR))), 
                            result))
        if not result:
            result = {const.TRUE}

        cnf = set()
        for clause in result:
            if len(set(clause.split(const.OR))) == len(set(clause.replace(const.NOT, '').split(const.OR))):
                cnf.add(clause)
        
        if not cnf:
            cnf = const.TRUE
            self._sat, self._valid, self._unsat, self._refutable = True, True, False, False
            self._falsifying, self._falsifyings = None, set()
            if isinstance(self._model, str):
                self._model = {v: True for v in self.variables}
            if isinstance(self._models, str):
                self._models = set()
            self._models.add('1' * len(self.variables))
        else:
            self._valid, self._refutable = False, True
            first_clause = list(cnf)[0].split(const.OR)
            falsifying = {v: 0 for v in self.variables}
            for v in first_clause:
                falsifying.update({v.replace(const.NOT, ''): 1 if const.NOT in v else 0})
            if isinstance(self._falsifying, str):
                self._falsifying = falsifying
            if isinstance(self._falsifyings, str):
                self._falsifyings = set()
            self._falsifyings.add(self._assignment_to_str(falsifying))
            
        result = const.AND.join(set([
            f"({clause})" if const.OR in clause else clause
              for clause in cnf]))
        result = Formula(result, as_cnf=True)

        result.is_cnf, result.is_nnf = True, True
        result._cnf = result
        result._nnf = result
        if const.OR in result.string and const.AND in result.string:
            result.is_dnf, result.is_aig = False, False
        elif const.OR in result.string:
            result.is_dnf, result.is_aig = True, False
            result._dnf = result
        elif const.AND in result.string:
            result.is_dnf, result.is_aig = True, True
            result._dnf = result
            result._aig = result
        else:
            result.is_dnf, result.is_aig = True, True
            result._dnf = result
            result._aig = result
        self._cnf = result
        
        if isinstance(self._valid, str):
            self._valid = result._valid
        if isinstance(self._refutable, str):
            self._refutable = result._refutable
            
        return self._cnf
    
    def check_cnf(self):
        """
        Allows to check if formula is in CNF form
        """
        if not isinstance(self.is_cnf, str):
            return self.is_cnf
                
        for sf in self.get_subformulas():
            l, o, r = sf[1]
            r_o, _ = _get_outermost_operation(r, _clean = False)
            l_o, _ = _get_outermost_operation(l, _clean = False)

            if o == const.OR: 
                if r_o == const.AND or l_o == const.AND:
                    return False
            elif o == const.NOT:
                if r_o or l_o:
                    return False
        
        self.is_cnf = True
        self._cnf = self
        return True
    
    def to_dnf(self):
        """
        Allows to construct Disjunction Normal Form (DNF)
        """

        if not isinstance(self._dnf, str):
            return self._dnf

        result = self.to_nnf().string
        stop, iteration = False, 0

        # with tqdm(desc='Converting to DNF: ', unit=' rules applied') as p_bar:
        while not stop:
                # print('It ', iteration)

                stop = True
                subformulas = self.to_nnf().get_subformulas() if iteration == 0 else Formula(result).get_subformulas()


                for sf in sorted(subformulas, key=lambda x: len(x[0]), reverse=False):
                    l, o, r = sf[1]
                    r_o, (r_l, r_r) = _get_outermost_operation(r, _clean = False)
                    l_o, (l_l, l_r) = _get_outermost_operation(l, _clean = False)

                    if o == const.AND:
                        
                        if r_o == const.OR:
                            # p_bar.update(1)
                            stop = False
                            replacement = f'(({l}){const.AND}({r_l})){const.OR}(({l}){const.AND}({r_r}))'
                            result = result.replace(sf[0], replacement)
                        elif l_o == const.OR:
                            # p_bar.update(1)
                            stop = False
                            replacement = f'(({l_l}){const.AND}({r})){const.OR}(({l_r}){const.AND}({r}))'
                            result = result.replace(sf[0], replacement)
                iteration += 1
                result = _clean_ast(result, _clean=False)
                result = result.replace(2 * const.NOT, '')
                
                # print(result)

        result = result.replace('(', '').replace(')', '').split(const.OR)
        result = set(filter(lambda cube: len(set(cube.replace(const.NOT, '').split(const.AND))) == len(set(cube.split(const.AND))), 
                            result))
        result = const.OR.join(set([
            f"({cube})" if const.AND in cube else cube
              for cube in result]))
        result = Formula(result, as_dnf=True)

        result.is_dnf, result.is_nnf = True, True
        result._dnf = result
        result._nnf = result
        if const.AND in result.string and const.OR in result.string:
            result.is_cnf, result.is_aig = False, False
        elif const.AND in result.string:
            result.is_cnf, result.is_aig = True, True
            result._cnf = result
            result._aig = result
        elif const.OR in result.string:
            result.is_cnf, result.is_aig = True, False
            result._cnf = result
        else:
            result.is_cnf, result.is_aig = True, True
            result._cnf = result
            result._aig = result
        self._dnf = result
        return self._dnf
    
    def to_aig(self):
        """
        Allows to construct Add Inverter Graph form AIG)
        """

        if not isinstance(self._aig, str):
            return self._aig

        result = self.to_nnf().string
        stop, iteration = False, 0

        # with tqdm(desc='Converting to AIG: ', unit=' rules applied') as p_bar:
        while not stop:

                stop = True
                subformulas = self.to_nnf().get_subformulas() if iteration == 0 else Formula(result).get_subformulas()


                for sf in sorted(subformulas, key=lambda x: len(x[0]), reverse=False):
                    l, o, r = sf[1]

                    if o == const.OR:
                        # p_bar.update(1)
                        stop = False
                        replacement = f'{const.NOT}({const.NOT}({l}){const.AND}{const.NOT}({r}))'
                        result = result.replace(sf[0], replacement)
        
                iteration += 1
                result = _clean_ast(result, _clean=False)
                result = result.replace(2 * const.NOT, '')
                

        result = Formula(result, as_aig=True)
        self._aig = result
        return self._aig
    

    def is_sat(self):
        """
        Checks SAT of formula
        """
        if not isinstance(self._sat, str):
            return self._sat
        if not isinstance(self._valid, str) and self._valid:
            return True
        if not isinstance(self._unsat, str) and self._unsat:
            return False

        if isinstance(self._cnf, str):
            self.to_cnf()
        # print(self._cnf)
        solver = Solver(name="lingeling")
        solver.append_formula(self._cnf.clauses_qdmacs)

        if solver.solve():

            self._sat, self._unsat = True, False
            model = solver.get_model()
            model_dict = {}
            for v in model:
                model_dict.update({self.variables[abs(v) - 1]: True if v > 0 else False})
            self._model = model_dict
            if isinstance(self._models, str):
                self._models = set()
            self._models.add(self._assignment_to_str(model_dict))
            solver.delete()

            return True
        
        else:
            self._sat, self._valid, self._unsat, self._refutable = False, False, True, True
            if isinstance(self._falsifying, str):
                self._falsifying = {v: True for v in self.variables}
            if isinstance(self._falsifyings, str):
                self._falsifyings = set()
            self._falsifyings.add(self._assignment_to_str(self._falsifying))
            self._model = None
            self._models = set()
            solver.delete()
        
            return False


    def is_valid(self):
        """
        Checks VALIDITY of formula
        """
        if not isinstance(self._valid, str):  # return value if already computed
            return self._valid
        if not isinstance(self._refutable, str):  # if refutable - not valid, else - valid
            return not self._refutable
        if not isinstance(self._unsat, str) and self._unsat:  # if unsat - not valid
            return False
        if not isinstance(self._sat, str) and not self._sat:  # if unsat - not valid
            return False

        cnf = self.to_cnf()
        self._valid = cnf._valid
        return self._valid
   

    def is_refutable(self):
        """
        Checks formula if REFUTABLE
        """
        if not isinstance(self._refutable, str):  # return value if already computed
            return self._refutable
        if not isinstance(self._valid, str):  # if valid - not refutable, else - refutable
            return not self._valid
        if not isinstance(self._unsat, str) and self._unsat:  # if unsat - refutable
            return True
        return not self.is_valid()

    def is_unsat(self):
        """
        Checks formula if UNSAT
        """
        if not isinstance(self._unsat, str):  # return value if already computed
            return self._unsat
        if not isinstance(self._sat, str):  # if sat - not unsat, else - unsat
            return not self._sat
        if not isinstance(self._valid, str) and self._valid:  # if valid - not unsat
            return False
        return not self.is_sat()
    
    def get_truth_table(self, print_result: bool = True):
        """
        Allows to construct a truth tables for small formulas
        """
        assert self.n_vars <= 6, f"too many variables for explicit truth table, maximum 6, but {self.n_vars} were provided"
        if not isinstance(self._truth_table, str):
            _truth_table = self._truth_table
        else:
            _truth_table = np.zeros((2 ** self.n_vars, self.n_vars + 1))
            assignments = [bin(x)[2:].rjust(self.n_vars, '0') for x in range(0, 2 ** self.n_vars)]
            for i, assignment in enumerate(assignments):
                formula_value = self(assignment)
                _truth_table[i] = [*list(assignment), formula_value]
                if formula_value:
                    if isinstance(self._model, str):
                        self._model = self._str_to_assignment(assignment)
                    self._models.add(assignment)
                else:
                    if isinstance(self._falsifying, str):
                        self._falsifying = self._str_to_assignment(assignment)
                    self._falsifyings.add(assignment)
            self._truth_table = _truth_table

            if np.all(_truth_table[:, -1] == 1):
                self._valid = True
            if np.all(_truth_table[:, -1] == 0):
                self._unsat = True
            if np.any(_truth_table[:, -1] == 1):
                self._sat = True
            if np.any(_truth_table[:, -1] == 0):
                self._refutable = True

        report = ' TRUTH TABLE '.center(60, '#') + '\n'
        report += f'Formula: {self.string}' + '\n'
        report += 'Truth Table:' + '\n'
        report += '\t\t' + ' '.join([var for var in self.variables + ['result']]) + '\n'
        lengths = [len(var) + 1 for var in self.variables + ['result']]
        for row in _truth_table:
            report += '\t\t' + ''.join([str(int(number)).center(l) for number, l in zip(row, lengths)]) + '\n'
        report += f'Valid? {self._valid}' + '\n'
        report += f'Satisfiable? {self._sat}' + '\n'
        report += f'Refutable? {self._refutable}' + '\n'
        report += f'Unsatisfiable? {self._unsat}' + '\n'
        report += '#' * 60
        
        if print_result:
            print(report)
        return _truth_table
        

    def get_model(self):
        """
        Allows to get a model if exists
        """
        if isinstance(self._model, str):
            self.is_sat()
        
        return self._model
    
    def get_all_models(self):
        """
        Allows to obtain all models for small formulas
        """

        assert self.n_vars <= 6, f"too many variables for explicit truth table, maximum 6, but {self.n_vars} were provided"
        if isinstance(self._truth_table, str):
            self.get_truth_table()
        return self._models
    
    def get_falsifying(self):
        """
        Allows to get a falsifying assignment if exists
        """
        assert self.n_vars <= 6, f"too many variables for explicit truth table, maximum 6, but {self.n_vars} were provided"
        if isinstance(self._truth_table, str):
            self.get_truth_table()
        return self._falsifying
    
    def get_all_falsifying(self):
        """
        Allows to obtain all falsifying assignments for small formulas
        """

        assert self.n_vars <= 6, f"too many variables for explicit truth table, maximum 6, but {self.n_vars} were provided"
        if isinstance(self._truth_table, str):
            self.get_truth_table()
        return self._falsifyings


    def info(self):
        """
        Generates current info about Formula instance (mainly for debugging)
        """
        s = 'Formula:'.ljust(20) + self.string + '\n'
        s += 'Original input:'.ljust(20) + self.original + '\n'
        s += 'Variables:'.ljust(20) + ', '.join(self.variables) + '\n'
        s += 'Literals:'.ljust(20) + ', '.join(self.literals) + '\n'
        s += 'N vars:'.ljust(20) + str(self.n_vars) + '\n'
        s += 'Subformulas:'.ljust(20) + (self._subformulas if isinstance(self._subformulas, str) else str(len(self._subformulas))) + '\n'
        s += 'Type:'.ljust(20)
        known_true = []
        known_false = []
        unknown = []
        for b, name in zip([self.is_cnf, self.is_dnf, self.is_aig, self.is_nnf], ['CNF', 'DNF', 'AIG', 'NNF']):
            if isinstance(b, str):
                unknown.append(f'[ ? {name} ]')
            elif b:
                known_true.append(f'[ IS {name} ]')
            else:
                known_false.append(f'[ NOT {name} ]')
        s += ' '.join(known_true) + ' '.join(known_false) + ' '.join(unknown) + '\n'
        s += 'NNF:'.ljust(20) + (self._nnf if isinstance(self._nnf, str) else self._nnf.string) + '\n'
        s += 'CNF:'.ljust(20) + (self._cnf if isinstance(self._cnf, str) else self._cnf.string) + '\n'
        s += 'DNF:'.ljust(20) + (self._dnf if isinstance(self._dnf, str) else self._dnf.string) + '\n'
        s += 'AIG:'.ljust(20) + (self._aig if isinstance(self._aig, str) else self._aig.string) + '\n'
        s += 'SAT?:'.ljust(20) + (self._sat if isinstance(self._sat, str) else str(self._sat)) + '\n'
        s += 'UNSAT?:'.ljust(20) + (self._unsat if isinstance(self._unsat, str) else str(self._unsat)) + '\n'
        s += 'VALID?:'.ljust(20) + (self._valid if isinstance(self._valid, str) else str(self._valid)) + '\n'
        s += 'REFUTABLE?:'.ljust(20) + (self._refutable if isinstance(self._refutable, str) else str(self._refutable)) + '\n'
        s += 'Model example:'.ljust(20) + str(self._model) + '\n'
        s += 'Models so far:'.ljust(20) + (self._models if isinstance(self._models, str) else str(len(self._models))) + '\n'
        s += 'Falsifying example:'.ljust(20) + str(self._falsifying) + '\n'
        s += 'Falsifying so far:'.ljust(20) + (self._falsifyings if isinstance(self._falsifyings, str) else str(len(self._falsifyings))) + '\n'
        return s
