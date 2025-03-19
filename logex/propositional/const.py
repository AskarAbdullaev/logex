OR, AND, IMPLIES, EQUIVALENT, NOT, XOR, TRUE, FALSE = '|', '&', '→', '⇔', '!', '⊕', '⊤', '⊥'

CONNECTIVES = [NOT, AND, OR, IMPLIES, EQUIVALENT, XOR]

COMBINED = r'(\!|&|\||→|⊕|⇔)'

COMBINED_EX = r'[^\!&\|→⊕⇔\(\)]*'

CONNECTIVES_DICT = {'|': OR,
                    '+': OR,
                    '!': NOT,
                    '&': AND,
                    '*': AND,
                    '¬': NOT,
                    'not': NOT,
                    'xor': XOR, 
                    'or': OR,
                    'and': AND,
                    '∧': AND,
                    'NOT': NOT,
                    'IMPLIES': IMPLIES, 
                    'TRUE': TRUE,
                    'FALSE': FALSE,
                    'XOR': XOR,
                    'OR': OR,
                    'AND': AND, 
                    '∨': OR,
                    '<->': EQUIVALENT,
                    '->': IMPLIES,
                    '<=>': EQUIVALENT, 
                    '=>': IMPLIES,
                    'EQUIVALENT': EQUIVALENT,
                    '⊤': TRUE,
                    '⊥': FALSE, 
                    '↔': EQUIVALENT,
                    '[': '(',
                    ']': ')',
                    '{': '(',
                    '}': ')'}

LIMBOOLE = {OR: '|',
            AND: '&',
            IMPLIES: '->',
            NOT: '!',
            EQUIVALENT: '<->'}

FANCY = {OR: '∨',
            AND: '∧',
            IMPLIES: '→',
            NOT: '¬',
            EQUIVALENT: '↔',
            XOR: '⊕',
            TRUE: '⊤',
            FALSE: '⊥'}

WORDS = {OR: ' OR ',
         AND: ' AND ',
         IMPLIES: ' IMPLIES ',
         NOT: ' NOT ',
         EQUIVALENT: ' EQUIVALENT ',
         XOR: ' XOR ',
         TRUE: ' TRUE ',
         FALSE: ' FALSE '}

DEFAULT = {OR: '|',
           AND: '&',
           IMPLIES: '→',
           NOT: '!',
           EQUIVALENT: '⇔',
           XOR: '⊕',
           TRUE: '⊤',
           FALSE: '⊥'}

OUTPUT_FILE = 'logical_report.txt'

INTERPRETATION = {OR: '+',
                  AND: '*',
                  IMPLIES: '^',
                  NOT: '-',
                  EQUIVALENT: '==',
                  XOR: '!=',
                  TRUE: ' boolean._boolean(1) ',
                  FALSE: ' boolean._boolean(0) ',
                  ' ': ''}

BACK = {'+': OR,
        '*': AND,
        '^': IMPLIES,
        '-': NOT,
        '==': EQUIVALENT,
        '!=': XOR,
        ' ': '',
        'boolean._boolean1': TRUE,
        'boolean._boolean0': FALSE,
        'boolean._boolean(1)': TRUE,
        'boolean._boolean(0)': FALSE,}

AST_NAMES = {'NotEq': XOR,
             'Eq': EQUIVALENT,
             'Mult': AND,
             'Add': OR,
             'USub': NOT,
             'BitXor': IMPLIES,
             ' ': ''}
