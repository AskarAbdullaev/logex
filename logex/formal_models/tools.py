import regex as re
import numpy as np

LTL_CONNECTIVES = ['G', 'F', '!', 'X', '&', 'U', '|', '→']
CTL_CONNECTIVES = ['AG', 'EG', 'AF', 'EF', '!', 'AX', 'EX', '&', 'EU', 'AU', '|', '→']
PA_CONNECTIVES = ['.', '+', '||']
PA_COMBINED = r'(\.|\+|\|\|)'
DICT = {'¬': '!', 'not': '!', 'or': '|', 'and': '&', '∧': '&', 'NOT': '!', 
        'IMPLIES': '→', 'implies': '→', 'OR': '|', 'AND': '&', '∨': '|',  
            '=>': '→', '->': '→','[': '(', ']': ')', '{': '(', '}': ')'}
LTL_COMBINED = r'(\!|&|\||→|G|F|X|U|A|E)'
CTL_COMBINED= r'(\!|&|\||→|AG|AF|AX|AU|EG|EF|EX|EU)'
EMPTY_STATE = '∅'
EMPTY_WORD = '𝜖'


def _clean_parenthesis(string: str, mode: str = 'LTL'):

    if mode == 'CTL':
        connectives = CTL_CONNECTIVES
    elif mode == 'LTL':
        connectives = LTL_CONNECTIVES
    else:
        raise AssertionError(f'mode can only be "CTL" or "LTL", not {mode}')

    cleaned_string = string[:]

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
            for symbol_ in cleaned_string[i+1:]:
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
        if not any(connective in enclosure for connective in connectives[1:4]):
            cleaned_string = cleaned_string.replace(enclosure, enclosure[1:-1])

    while 2 * '!' in cleaned_string:
        cleaned_string = cleaned_string.replace(2 * '!', '')

    if cleaned_string.count(')') == cleaned_string.count('(') == 1 and cleaned_string[-1] == ')' and cleaned_string[0] == '(':
        cleaned_string = cleaned_string[1:-1]

    return cleaned_string


def _get_outermost_connective(string: str, mode: str = 'LTL') -> tuple[str, tuple[str, str]]:

    """
    Takes a string, cleans it and finds the outermost logical connective with respect to the
    conventional precedence rules;
    Returns tuple of the form: 
        outermost_connective, (left-hand side, right-hand side)
    """
    if mode == 'CTL':
        connectives = CTL_CONNECTIVES
        combined = CTL_COMBINED
    elif mode == 'LTL':
        connectives = LTL_CONNECTIVES
        combined = LTL_COMBINED
    elif mode == 'PA':
        connectives = PA_CONNECTIVES
        combined = PA_COMBINED
    else:
        raise AssertionError(f'mode can only be "CTL", "LTL" or "PA", not {mode}')
    
    string = string.replace(' ', '')
    for symbol, replacement in DICT.items():
        string = string.replace(symbol, replacement)

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

    split_string = [x for x in re.split(combined, string) if x]  # split by logical connectives
    if {'AG', 'EG', 'AF', 'EF', 'AX', 'EX', 'AU', 'EU'} & set(split_string) and mode == 'LTL':
        raise ValueError(f'"A" or "E" are only applicable when using CTL, not LTL')
    # print(split_string)

    if not any(x in split_string for x in connectives):
        # If there are no logical connectives - return the atomic formula
        outermost_connective = ''
        left_part, right_part = ''.join(split_string), ''
    elif not any(x in split_string for x in set(connectives) - {'!'}) and split_string[0] == '!':
        # If the only connective is NOT in the beginning
        outermost_connective = '!'
        left_part, right_part = '', '(' + ''.join(split_string[1:]) + ')'
    else:
        # Else find the first connective by precedence and split the formula at its index
        if mode == 'LTL' and split_string[0] in {'G', 'F'}:
            outermost_connective = split_string[0]
            left_part, right_part = '', ''.join(split_string[1:])
        elif mode == 'CTL' and split_string[0] in {'AG', 'EG', 'AF', 'EF'}:
            outermost_connective = split_string[0]
            left_part, right_part = '', ''.join(split_string[1:])
        elif mode == 'PA':
            for j, connective in enumerate(connectives):
                if connective == '!':
                    continue
                for i, symbol in enumerate(split_string):
                    if symbol == connective:
                        outermost_connective = connective
                        break
            parts = ''.join(split_string).split(outermost_connective)
            
            # Back substitution of enclosures
            for i, part in enumerate(parts):
                for special_symbol, enclosure in replacement_dict.items():
                    if part == special_symbol:
                        parts[i] = parts[i].replace(special_symbol, enclosure)[1:-1]
                    else:
                        parts[i] = parts[i].replace(special_symbol, enclosure)
            if outermost_connective == '.':
                parts = [parts[0], '.'.join(parts[1:])]
            return outermost_connective, parts
        
        else:
            for j, connective in enumerate(connectives):
                if connective == '!':
                    continue
                for i, symbol in enumerate(split_string):
                    if symbol == connective:
                        outermost_connective = connective
                        left_part, right_part = ''.join(split_string[:i]), ''.join(split_string[i+1:])
                        break
                    
    # Back substitution of enclosures
    for special_symbol, enclosure in replacement_dict.items():
        if left_part == special_symbol:
            left_part = left_part.replace(special_symbol, enclosure)[1:-1]
        else:
            left_part = left_part.replace(special_symbol, enclosure)
        if right_part == special_symbol:
            right_part = right_part.replace(special_symbol, enclosure)[1:-1]
        else:
            right_part = right_part.replace(special_symbol, enclosure)
        
    return outermost_connective, (left_part, right_part)