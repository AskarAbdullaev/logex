import re
from typing import Callable, Union, Collection, Sequence

VARIABLES = {chr(i) for i in range(65, 91)}

def clean(string: str):
    """
    Small pipeline for string cleaning
    """
    assert isinstance(string, str), f"only str type is accepted, not {type(string)}"
    cleaned_str = string.replace(' ', '').replace('\t', '').replace('->', '@').replace('-', '_').replace('@', '->').replace('\n', '').replace('()', '')
    return cleaned_str

def decompose_term(tau: str):
    """
    Extracts function symbols set, variable symbols set and all valid
    sub-terms from a given term
    """

    tau = clean(tau)

    pointer, current_symbol, depth = 0, '', 0
    functions, arities, variables, depths = {}, {}, [], []
    while pointer < len(tau):
        next_letter = tau[pointer]
        if next_letter == '(':
            functions.update({depth: current_symbol})
            arities.update({current_symbol: 1})
            current_symbol = ''
            depths.append(depth)
            depth += 1
        elif next_letter == ',':
            if current_symbol:
                variables.append(current_symbol)
            current_symbol = ''
            arities[functions[depth - 1]] += 1
            depths.append(depth)
        elif next_letter == ')':
            if current_symbol:
                variables.append(current_symbol)
            current_symbol = ''
            depth -= 1
            depths.append(depth)
        else:
            current_symbol += next_letter
            depths.append(depth)
        pointer += 1
        
    if current_symbol:
        variables.append(current_symbol)

    functions = set(functions.values())
    constants = {v for v in variables if v.islower()}
    variables = set(variables) - set(constants)

    layers = [[] for _ in range(max(depths) + 1)]
    layers[0].append((tau[0], 0))
    for i, (symbol, depth) in enumerate(zip(tau[1:], depths[1:]), start=1):
        for d, _ in enumerate(layers):
            if depth >= d:
                if symbol != ',':
                    layers[d].append((symbol, i))
                else:
                    if d == depth:
                        layers[d].append(('|', i))
                    else:
                        layers[d].append((',', i))
            
            elif depth + 1 == d:
                layers[d].append(('|', i))
    
    # print(layers)
    sub_terms = set()
    sub, sub_start, sub_finish = '', 0, 0
    for layer in layers:
        for i, (symbol, index) in enumerate(layer):
            if not symbol == '|':
                if sub == '':
                    sub_start = index
                sub += symbol
            else:
                if sub != '':
                    sub_finish = layer[i-1][1] + 1
                    sub_terms.add((sub, (sub_start, sub_finish)))
                sub, sub_start, sub_finish = '', 0, 0
    if sub != '':
        sub_terms.add((sub, (sub_start, len(tau))))

    sub_terms_indices = {}
    sub_terms_at_positions = {}
    for st, (start, finish) in sub_terms:
        sub_terms_at_positions.update({start: st})
        if st not in sub_terms_indices.keys():
            sub_terms_indices.update({st: [(start, finish)]})
        else:
            sub_terms_indices[st].append((start, finish))

    return functions, arities, variables, constants, sub_terms_indices, sub_terms_at_positions

class Term():
    """
    Main class for equational reasoning
    """

    def __init__(self, term: str):

        assert isinstance(term , str), f"term must be of type str, not {type(term)}"
        self.original = term
        self.term = clean(term)
        (self.functions, self.arities, self.variables, self.constants,
                        self.sub_terms_indices, self.sub_terms_at_position) = decompose_term(self.term)
        self.is_ground_term = True if not self.variables else False
        self.sub_terms = set(self.sub_terms_indices.keys())

    def __str__(self):

        return self.term
    
    def __repr__(self):

        return f"<Term: {self.term}>"
    
    def __len__(self):
        return len(self.term)
    
    def __eq__(self, other):
        if isinstance(other, str):
            return clean(other) == self.term
        elif isinstance(other, Term):
            return self.term == other.term
        else:
            raise ValueError('Term can be only compared with a string or with another Term instance')
    
    def subterm_index(self, sub_term: str):
        """
        Getting start and end indices of a subterm
        """

        assert isinstance(sub_term, str) or isinstance(sub_term, Term), f'sub_term must be str / Term, not {type(sub_term)}'
        sub_term = clean(sub_term)
        return self.sub_terms_indices.get(sub_term, False)
        
    def subterm_at_position(self, index: int):
        """
        Getting subterm rooted at index
        """

        assert isinstance(index, int) and index >= 0, f'index must be of type int and >= 0, not {type(index)}'
        return self.sub_terms_at_position.get(index, False)
    
    def apply_substitution(self, sigma: dict):
        """
        Applying the substitution to the term
        """
        assert isinstance(sigma, dict), f'substitution sigma must be a dict, not {type(sigma)}'
        new_term = self.term[:]

        for variable, replacement in sigma.items():
            r = '@' + '@'.join(list(replacement)) + '@'
            new_term = re.sub(f'(?<!@){variable}(?!@)', r, new_term)
        return Term(new_term.replace('@', ''))
    
    def is_variable_renaming(self, sigma: dict):
        """
        Check is substitution is variable renaming
        """
        assert isinstance(sigma, dict), f'substitution sigma must be a dict, not {type(sigma)}'
        mentioned_vars = self.variables & set(sigma.keys())

        for var1 in mentioned_vars:
            for var2 in mentioned_vars:
                replace_1, replace_2 = clean(sigma[var1]), clean(sigma[var2])
                if replace_1 not in self.variables or replace_2 not in self.variables:
                    return False
                if var1 == var2 and replace_1 != replace_2:
                    return False
        return True

    def copy(self):
        """
        Make a deep copy
        """
        return Term(self.term)
    
    def __hash__(self):
        return hash(self.term)

    
    def info(self):
        """
        Information anout an instance (mainly for debugging
        """
        s = f'Term: {self.term}\n'
        s += f'Originals: {self.original}\n'
        s += f'Ground term? {self.is_ground_term}\n'
        s += f'Functions: {self.functions}\n'
        s += f'with arities: {self.arities}\n'
        s += f'Variables: {self.variables}\n'
        s += f'Constants: {self.constants}\n'
        s += f'Sub-terms: {self.sub_terms}\n'

        return s
    
def disjoin_variables(to_keep: str | Term, to_replace: str | Term):
    """
    Create a substitution to make original look like target or False if impossible
    (Algorithm 5 - Matching)

    Args:
        original_term (str | Term): _description_
        target_term (str | Term): _description_
    """

    assert isinstance(to_keep, Union[str, Term]), f'to_keep must be str / Term , not {type(to_keep)}'
    assert isinstance(to_replace, Union[str, Term]), f'to_replace term must be str / Term , not {type(to_replace)}'

    if isinstance(to_keep, str):
        to_keep = Term(to_keep)
    if isinstance(to_replace, str):
        to_replace = Term(to_replace)

     # Check that vars are different
    if to_keep.variables & to_replace.variables:
        possible_vars = VARIABLES - to_replace.variables
        sigma = {}
        for var in to_replace.variables & to_keep.variables:
            replacement = list(possible_vars)[0]
            sigma.update({var: replacement})
            possible_vars -= {replacement}
        to_replace = to_replace.apply_substitution(sigma)
    
    return to_keep, to_replace

    
def match_terms(original_term: str | Term, target_term: str | Term, generate_report: bool = True):
    """
    Create a substitution to make original look like target or False if impossible
    (Algorithm 5 - Matching)

    Args:
        original_term (str | Term): _description_
        target_term (str | Term): _description_
    """

    assert isinstance(original_term, Union[str, Term]), f'original term must be str / Term , not {type(original_term)}'
    assert isinstance(target_term, Union[str, Term]), f'target term must be str / Term , not {type(target_term)}'

    if isinstance(original_term, str):
        o = Term(original_term)
    else:
        o = original_term.copy()
    if isinstance(target_term, str):
        t = Term(target_term)
    else:
        t = target_term.copy()

    report = ' TERM MATCHING '.center(60, '#') + '\n'
    
    o, t = disjoin_variables(o, t)
    # print(f'disjoint: {o}, {t}')
    report += f'Original term: {o}\n'
    report += f'Target term: {t}\n'

    sigma = {v: v for v in o.variables}
    pointer = 0

    while pointer < min(len(o.term), len(t.term)):

        c1, c2 = o.term[pointer], t.term[pointer]
        
        if c1 != c2:
            report += 'Mismatch found:\n'
            report += f'\t{o.term}\n'
            report += f'\t{t.term}\n'
            report += '\t' + (' ' * pointer) + '^' + '\n\n'
            # print(f'c1 = {c1} != c2 = {c2}.')
            rooted_in_c1 = o.subterm_at_position(pointer)
            # print(f'rooted in c1: {rooted_in_c1}')
            
            if rooted_in_c1 not in o.variables:
                # print('c1 is function: False')
                sigma = False
                break
            if len(o.variables) == 1:
                if rooted_in_c1 in o.term[:pointer]:
                    # print('c1 was already: False')
                    sigma = False
                    break
            else:
                if re.findall(r'(<=[\(,])' + rooted_in_c1 + r'(?=[\),])', o.term):
                    # print('c1 was already: False')
                    sigma = False
                    break
            rooted_in_c2 = t.subterm_at_position(pointer)
            # print(f'Substitution {rooted_in_c1} -> {rooted_in_c2} is applied')

            report += f'Substitution added: {rooted_in_c1} -> {rooted_in_c2}\n\n'
            sigma.update({rooted_in_c1: rooted_in_c2})
            o = original_term.apply_substitution(sigma)
            # print(f'Now O = {o.term}')
        pointer += 1

    report += f'Result: {sigma}\n'
    report += '#' * 60 + '\n'
    if generate_report:
        print(report)

    return sigma

def least_general_generalization(term_1: str | Term, term_2: str | Term, generate_report: bool = True):
    """
    Find LGG

    Args:
        original_term (str | Term): _description_
        target_term (str | Term): _description_
    """

    assert isinstance(term_1, Union[str, Term]), f'term_1 must be str / Term , not {type(term_1)}'
    assert isinstance(term_2, Union[str, Term]), f'term_2 must be str / Term , not {type(term_2)}'

    if isinstance(term_1, str):
        t1 = Term(term_1)
    else:
        t1 = term_1.copy()
    if isinstance(term_2, str):
        t2 = Term(term_2)
    else:
        t2 = term_2.copy()

    sigma1, sigma2 = {}, {}

    report = ' LGG '.center(60, '#') + '\n'
    report += f'Term 1: {t1}\n'
    report += f'Term 2: {t2}\n'
    
    possible_variables = VARIABLES - (t1.variables | t2.variables)
    encountered_pairs = {}

    while True:

        pointer = 0
        while pointer < min(len(t1.term), len(t2.term)):

            c1, c2 = t1.term[pointer], t2.term[pointer]
            
            if c1 != c2:
                report += 'Mismatch found:\n'
                report += f'\t{t1.term}\n'
                report += f'\t{t2.term}\n'
                report += '\t' + (' ' * pointer) + '^' + '\n\n'

                rooted_in_c1 = t1.subterm_at_position(pointer)
                rooted_in_c2 = t2.subterm_at_position(pointer)

                if (rooted_in_c1, rooted_in_c2) not in encountered_pairs.keys():
                    new_var = possible_variables.pop()
                    sigma1.update({new_var: rooted_in_c1})
                    sigma2.update({new_var: rooted_in_c2})
                    encountered_pairs.update({(rooted_in_c1, rooted_in_c2): new_var})
                    report += f'Substitution added to sigma_1: {new_var} -> {rooted_in_c1}\n'
                    report += f'Substitution added to sigma_2: {new_var} -> {rooted_in_c2}\n'
                    report += 'Back substitution is performed\n\n'
                else:
                    new_var = encountered_pairs[(rooted_in_c1, rooted_in_c2)]
                    report += 'Substitution is already encountered.\n'
                    report += 'Back substitution is performed\n\n'

                t1_new = t1.term
                c1_start, c1_stop = t1.sub_terms_indices[rooted_in_c1][0]
                t1_new = t1_new[:c1_start] + new_var + t1_new[c1_stop:]
                t1 = Term(t1_new)

                t2_new = t2.term
                c2_start, c2_stop = t2.sub_terms_indices[rooted_in_c2][0]
                t2_new = t2_new[:c2_start] + new_var + t2_new[c2_stop:]
                t2 = Term(t2_new)

                break
 
            pointer += 1
        
        if pointer == len(t1.term):
            break

    report += f'Result: {t1.term}\nsigma_1: {sigma1}\nsigma_2: {sigma2}\n'
    report += '#' * 60 + '\n'
    if generate_report:
        print(report)

    return t1.copy(), sigma1, sigma2

def most_general_unifier(term_1: str | Term, term_2: str | Term, generate_report: bool = True):
    """
    Find MGU (Algorithm 6)

    Args:
        original_term (str | Term): _description_
        target_term (str | Term): _description_
    """

    assert isinstance(term_1, Union[str, Term]), f'term_1 must be str / Term , not {type(term_1)}'
    assert isinstance(term_2, Union[str, Term]), f'term_2 must be str / Term , not {type(term_2)}'

    if isinstance(term_1, str):
        t1 = Term(term_1)
    else:
        t1 = term_1.copy()
    if isinstance(term_2, str):
        t2 = Term(term_2)
    else:
        t2 = term_2.copy()

    report = ' MGU '.center(60, '#') + '\n'
    report += f'Term 1: {t1}\n'
    report += f'Term 2: {t2}\n'

    sigma = {}
    pointer = 0

    while pointer < min(len(t1.term), len(t2.term)):

        c1, c2 = t1.term[pointer], t2.term[pointer]
        
        if c1 != c2:
            report += 'Mismatch found:\n'
            report += f'\t{t1.term}\n'
            report += f'\t{t2.term}\n'
            report += '\t' + (' ' * pointer) + '^' + '\n\n'

            rooted_in_c1 = t1.subterm_at_position(pointer)
            rooted_in_c2 = t2.subterm_at_position(pointer)

            # print(f'Rooted in c1: {rooted_in_c1}\nRooted in c2: {rooted_in_c2}')
            
            if rooted_in_c1 not in t1.variables and rooted_in_c2 not in t2.variables:
                report += 'Both sub-terms are functions: not unifiable\n'
                sigma = False
                break
            
            if rooted_in_c1 in t1.variables:
                if rooted_in_c1 in Term(rooted_in_c2).variables:
                    report += f'Variable {rooted_in_c1} is in counter-term {rooted_in_c2}: not unifiable\n'
                    return False
                new_sub = {rooted_in_c1: rooted_in_c2}
                report += f'New substitution is added: {rooted_in_c1} -> {rooted_in_c2}\n'
                
            
            elif rooted_in_c2 in t2.variables:
                if rooted_in_c2 in Term(rooted_in_c1).variables:
                    report += f'Variable {rooted_in_c2} is in counter-term {rooted_in_c1}: not unifiable\n'
                    sigma = False
                    break
                new_sub = {rooted_in_c2: rooted_in_c1}
                report += f'New substitution is added: {rooted_in_c2} -> {rooted_in_c1}\n'
            

            t1 = t1.apply_substitution(new_sub)
            t2 = t2.apply_substitution(new_sub)
            for key, value in sigma.items():
                sigma[key] = Term(value).apply_substitution(new_sub).term
            sigma.update(new_sub)

        pointer += 1

    report += f'Result: {sigma}\n'
    if sigma:
        report += f'Final unification: {t1.term}\n'
    report += '#' * 60 + '\n'
    if generate_report:
        print(report)

    return sigma


def is_variable_renaming(sigma: dict):
    """
    Checks if substitution is a variable renaming
    """

    if len(set(sigma.keys())) != len(set(sigma.values())):
        return False
    for value in sigma.values():
        if '(' in value:
            return False
    return True
    

class DefaultTermOrder():
    """
    Default term order
    """

    def __init__(self):
        pass

    def __call__(self, term_1: str | Term, term_2: str | Term):
        """
        Find order relation
        """

        assert isinstance(term_1, Union[str, Term]), f'term_1 must be str / Term , not {type(term_1)}'
        assert isinstance(term_2, Union[str, Term]), f'term_2 must be str / Term , not {type(term_2)}'

        if isinstance(term_1, str):
            t1 = Term(term_1)
        else:
            t1 = term_1.copy()
        if isinstance(term_2, str):
            t2 = Term(term_2)
        else:
            t2 = term_2.copy()

        if len(t1.functions) == 0 or len(t2.functions) == 0:
            return None
        if len(t1.term) < len(t2.term):
            return True
        if len(t1.term) > len(t2.term):
            return False
        if len(t1.term) == len(t2.term):
            if t1.term[0] < t2.term[0]:
                return True
            if t1.term[0] > t2.term[0]:
                return False
            if t1.term[0] == t2.term[0]:
                k = 0
                arguments_1 = []
                arguments_2 = []

                for i in range(2, len(t1.term) - 1):
                    if t1.term[i-1] == ",":
                        arguments_1.append(t1.subterm_at_position(i))
                    
                for i in range(2, len(t2.term) - 1):
                    if t2.term[i-1] == "," :
                        arguments_2.append(t2.subterm_at_position(i))
                
                if len(arguments_1) == 0 or len(arguments_2) == 0:
                    return None
                
                for i in range(min(len(arguments_1), len(arguments_2))):
                    if arguments_1[i] == arguments_2[i]:
                        k += 1
                        continue
                    result = self(arguments_1[i], arguments_2[i])
                    if k > 0:
                        return result
                    else:
                        return None
                return None
            

def reduction_of_term(reduction_system: Collection, term: str, indent: int = 0, generate_report: bool = True):
    """
    Reduction algorithm (7)
    """

    assert isinstance(term, Union[str, Term]), f'term must be str / Term , not {type(term)}'
    if isinstance(term, str):
        term = Term(term)
    else:
        term = term.copy()
    
    assert isinstance(reduction_system, Collection), f'reduction system must be Collection , not {type(reduction_system)}'
    assert len(reduction_system) > 0, 'reduction system cannot be empty!'
    red_sys, left_parts, right_parts = [], [], []
    for rule in reduction_system:
        assert not isinstance(rule, str) and isinstance(rule, Sequence), f'each rule must be a sequence , not {type(rule)}'
        assert len(rule) == 2, f'each rule must be a 2-sequence , not {len(rule)}-sequence'
        left, right = rule
        assert isinstance(left, Union[str, Term]), f'rule parts must be str / Term , not {type(left)}'
        assert isinstance(right, Union[str, Term]), f'rule parts must be str / Term , not {type(right)}'
        if isinstance(left, str):
            left = Term(left)
        else:
            left = left.copy()
        if isinstance(right, str):
            right = Term(right)
        else:
            right = right.copy()
        left_parts.append(left)
        right_parts.append(right)
        red_sys.append((left, right))

    report = "\t" * indent + " REDUCTION ".center(60, '#') + "\n"
    report += "\t" * indent + f'Term: {term.term}\n'
    report += "\t" * indent + 'Reduction system: \n' + ('\n' + "\t" * indent).join(rule[0].term + " -> " + rule[1].term for rule in red_sys) + '\n\n'

    tau_prime = term.copy()
    stop = False
    step = 1

    trace = []
    while not stop:
        
        stop = True
        for s in tau_prime.sub_terms:
            for i, l in enumerate(left_parts):

                sigma = match_terms(l, s, generate_report=False)
                if not sigma:
                    continue
                s_prime = right_parts[i].apply_substitution(sigma)

                report += "\t" * indent + f"Step {step}, by using rule: [ {left_parts[i]} -> {right_parts[i]} ]:\n\t{tau_prime} ==> "

                # Not a substitution but a replacement
                new_tau_prime = tau_prime.term
                for start, stop in tau_prime.sub_terms_indices[s]:
                    new_tau_prime = new_tau_prime[:start] + s_prime.term + new_tau_prime[stop:]
                tau_prime = Term(new_tau_prime)

                report += f"{tau_prime}\n\n"

                trace.append(i)
                step += 1
                stop = False
                break

            if not stop:
                break
    
    if step == 1:
        report = "\t" * indent + f"REDUCTION ALGORITHM for term {term}: No rules applicable\n"
    

    if generate_report:
        report += f'Result: {tau_prime.term}\n'
        report += "\t" * indent + '#' * 60
        print(report)

    return tau_prime, trace

def critical_pairs(reduction_system: list, check_one_rule: int = None, 
                   indent: int = 0, generate_report: bool = True):
    """
    Takes reduction system and derived critical points according to algorithm 8.
    """

    assert isinstance(reduction_system, Collection), f'reduction system must be Collection , not {type(reduction_system)}'
    assert len(reduction_system) > 0, 'reduction system cannot be empty!'
    red_sys, left_parts, right_parts = [], [], []
    for rule in reduction_system:
        assert not isinstance(rule, str) and isinstance(rule, Sequence), f'each rule must be a sequence , not {type(rule)}'
        assert len(rule) == 2, f'each rule must be a 2-sequence , not {len(rule)}-sequence'
        left, right = rule
        assert isinstance(left, Union[str, Term]), f'rule parts must be str / Term , not {type(left)}'
        assert isinstance(right, Union[str, Term]), f'rule parts must be str / Term , not {type(right)}'
        if isinstance(left, str):
            left = Term(left)
        else:
            left = left.copy()
        if isinstance(right, str):
            right = Term(right)
        else:
            right = right.copy()
        left_parts.append(left)
        right_parts.append(right)
        red_sys.append((left, right))
    reduction_system = red_sys

    
    report = ''
    report += "\t" * indent + " CRITICAL PAIRS ALGORITHM ".center(60, '#') + '\n'
    report += "\t" * indent + 'Reduction system: \n' + ('\n' + "\t" * indent).join(rule[0].term + " -> " + rule[1].term for rule in red_sys) + '\n\n'
    report += "\t" * indent + f"Initialising Critical pairs to empty set: {set()}\n"

    C, step = set(), 0
    for i, (lambda_1, rho_1) in enumerate(reduction_system):
        for j, (lambda_2, rho_2) in enumerate(reduction_system):
            if check_one_rule:
                if not (i == check_one_rule or j == check_one_rule):
                    continue

            step += 1

            # Report
            report += "\t" * indent + f"{step}) Working with pair of rules:\n"
            report += "\t" * indent + f"\t\t\tRule {i + 1}: [ {lambda_1} -> {rho_1} ]\n"
            report += "\t" * indent + f"\t\t\tRule {j + 1}: [ {lambda_2} -> {rho_2} ]\n\n"

            common_variables = lambda_1.variables & lambda_2.variables
            used_variables = lambda_1.variables | lambda_2.variables
            sigma = {}
            pool = VARIABLES - used_variables
            for common_variable in common_variables:
                sigma.update({common_variable: pool.pop()})

            # Report
            report += "\t" * indent + f"\tλ1 [ {lambda_1} ] and λ2 [ {lambda_2} ] have {len(common_variables)} common variables: {common_variables}\n"
            report += "\t" * indent + "\tsubstitution σ = {" + ', '.join([v + ' <- ' + k for k, v in sigma.items()]) + "}\n"

            lambda_2, rho_2 = lambda_2.apply_substitution(sigma), rho_2.apply_substitution(sigma)

            # Report
            if len(sigma) > 0:
                report += "\t" * indent + f"\tNew rule 2: [ {lambda_2} -> {rho_2} ]\n"
            report += "\t" * indent + f"\n\tIterating over non-variable subterms (s) of λ1 [ {lambda_1} ]:\n"

            lambda_1_subterms = lambda_1.sub_terms.copy()
            for var in lambda_1.variables:
                if var in lambda_1_subterms:
                    lambda_1_subterms.remove(var)
            
            for k, s in enumerate(lambda_1_subterms):

                s = Term(s)

                # Report
                report += "\t" * indent + f"\n\t\t{step}.{k + 1}) Working with a subterm s [ {s} ]:\n"

                mu = most_general_unifier(s, lambda_2, generate_report=False)

                #Repoprt
                report += "\t" * indent + f"\t\t\tMost general unifier of s and λ2: μ = {mu if mu else '⊥'}\n"
                if not mu:
                    report += "\t" * indent + "\t\t\t as μ = ⊥ => nothing happens\n\n"
                    continue
                if is_variable_renaming(mu) and i == j:
                    report += "\t" * indent + "\t\t\t as μ = renaming => nothing happens\n\n"
                    continue


                mu_lambda_1 = lambda_1.apply_substitution(mu)
                mu_lambda_2 = lambda_2.apply_substitution(mu)
                mu_s = s.apply_substitution(mu)
                mu_rho_2 = rho_2.apply_substitution(mu)
                mu_rho_1 = rho_1.apply_substitution(mu)

                # Report
                report += "\t" * indent + f"\t\t\t μ(λ1) = {mu_lambda_1}\n"
                report += "\t" * indent + f"\t\t\t μ(λ2) = {mu_lambda_2}\n"
                report += "\t" * indent + f"\t\t\t μ(s) = {mu_s}\n"
                report += "\t" * indent + f"\t\t\t μ(ρ2) = {mu_rho_2}\n"
                report += "\t" * indent + f"\t\t\t μ(ρ1) = {mu_rho_1}\n"


                # Not a substitution but a replacement
                tau = mu_lambda_1.term
                for start, stop in mu_lambda_1.sub_terms_indices[mu_s.term]:
                    tau = tau[:start] + mu_rho_2.term + tau[stop:]
                tau = Term(tau)

                # Report
                report += "\t" * indent + f"\t\t\t==> τ = {tau}\n"
                report += "\t" * indent + f"\t\t\tAdding pair (τ, μ(ρ1)): ({tau}, {mu_rho_1}) to Critical Pairs\n\n"

                v = tau.variables | mu_lambda_1.variables
                v = sorted(list(v))
                unifying_substitution = {_v: sorted(pool)[i] for i, _v in enumerate(v)}
                pair = (tau.apply_substitution(unifying_substitution),
                        mu_rho_1.apply_substitution(unifying_substitution))
                inverse_pair = (pair[1], pair[0])
                if pair == inverse_pair:
                    report += "\t" * indent + "\t\t\trule is trivial ==> nothing happens\n"
                    continue
                if not pair in C and not inverse_pair in C:
                    C.add(pair)

    # Report
    report += "\n" + "\t" * indent + "Critical pairs: " + " | ".join([x[0].term + " = " + x[1].term for x in C]) + "\n"
    report += "\t" * indent + '#' * 60

    if generate_report:
        print(report)

    return C

def knuth_bendix(reduction_system: list, term_order: Callable = DefaultTermOrder(),
                 generate_report: bool = True):

    """
    Takes reduction system and a term_ordering function.

    Term-order must take two terms and return True if term_1 < term_2, False if term_1 > term_2 and None otherwise.
    """
    assert isinstance(reduction_system, Collection), f'reduction system must be Collection , not {type(reduction_system)}'
    assert len(reduction_system) > 0, 'reduction system cannot be empty!'
    red_sys, left_parts, right_parts = [], [], []
    for rule in reduction_system:
        assert not isinstance(rule, str) and isinstance(rule, Sequence), f'each rule must be a sequence , not {type(rule)}'
        assert len(rule) == 2, f'each rule must be a 2-sequence , not {len(rule)}-sequence'
        left, right = rule
        assert isinstance(left, Union[str, Term]), f'rule parts must be str / Term , not {type(left)}'
        assert isinstance(right, Union[str, Term]), f'rule parts must be str / Term , not {type(right)}'
        if isinstance(left, str):
            left = Term(left)
        else:
            left = left.copy()
        if isinstance(right, str):
            right = Term(right)
        else:
            right = right.copy()
        left_parts.append(left)
        right_parts.append(right)
        red_sys.append((left, right))
    reduction_system = red_sys

    # Report
    report = ""
    report +=  " KNUTH-BENDIX ".center(60, '#') + '\n'
    report += 'Reduction system (STEP 0): \n' + '\n'.join(rule[0].term + " -> " + rule[1].term for rule in red_sys) + '\n\n'
    

    R = reduction_system
    C = critical_pairs(R, indent = 1, generate_report=False)

    steps = 0
    while len(C) > 0:

        steps += 1

        # Report
        report += '\nCritical pairs were recomputed\n'
        report += f"Critical pairs left to check ({len(C)}):\n"
        report += '\n'.join(rule[0].term + " -> " + rule[1].term for rule in C) + '\n\n'

        critical_pair = C.pop()

        # Report
        report += f"Working with critical pair: {critical_pair[0].term + " -> " + critical_pair[1].term}\n"

        rho, _ = reduction_of_term(R, critical_pair[0], generate_report=False)
        rho_prime, _ = reduction_of_term(R, critical_pair[1], generate_report=False)

        report += f"Obtained: ρ = {rho} and ρ' = {rho_prime}\n"


        if rho != rho_prime:
            comparison = term_order(rho, rho_prime)
            if comparison is None:
                return "failed due to inconsistent term order"
            if comparison:
                rho, rho_prime = rho_prime, rho
            if (rho, rho_prime) not in R and (rho_prime, rho) not in R:
                R.append((rho, rho_prime))

                # Report
                report += f"Rule {rho} -> {rho_prime} is appended to reduction rules\n"

                updated_critical_pairs = critical_pairs(R, check_one_rule=len(R) - 1, generate_report=False)

                updated_critical_pairs -= {critical_pair, (critical_pair[1], critical_pair[0])}
                C |= updated_critical_pairs
        else:
            report += "Equal results obtained - no new rules deduced.\n\n"

    report += 'Reduction system (FINAL): \n' + '\n'.join(rule[0].term + " -> " + rule[1].term for rule in red_sys) + '\n'
    report += "#" * 60

    if generate_report:
        print(report)

    return R

