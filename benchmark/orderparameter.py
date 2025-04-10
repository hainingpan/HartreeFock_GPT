from sympy import *
from sympy.physics.quantum import *

### API
c={'up':Operator(r'c_{\uparrow}(k)'),'down':Operator(r'c_{\downarrow}(k)')}


normalorder = [x for x in c.values()]
normalorder_dag = [Dagger(x) for x in c.values()]

normalorder_rule = [(normalorder_dag[i]*normalorder_dag[j],-normalorder_dag[j]*normalorder_dag[i],) for i in range(len(normalorder_dag)) for j in range(i,len(normalorder_dag)) if j>i ] + [(normalorder[i]*normalorder[j],-normalorder[j]*normalorder[i])  for i in range(len(normalorder)) for j in range(i,len(normalorder)) if j>i] + [(normalorder[i]*normalorder_dag[j],-normalorder_dag[j]*normalorder[i]) if j!=i else (normalorder[i]*normalorder_dag[j],1-normalorder_dag[j]*normalorder[i]) for i in range(len(normalorder)) for j in range(len(normalorder))]

def recursiveapply(expr,subs):
    expr1=expr.subs(subs)
    counter=0
    while not expr1== expr:
        expr=expr1
        expr1=expr.subs(subs).expand()
        counter+=1
        # print(expr1)
    return expr1

### Parse latex
import re

def extract_latex_equations(text):
    # First extract double dollar equations to avoid overlapping with single dollar pattern
    double_dollar_pattern = r'\$\$(.*?)\$\$'
    double_dollar_matches = re.findall(double_dollar_pattern, text, re.DOTALL)
    
    # Replace double dollar equations with placeholders to avoid conflicts
    modified_text = text
    for i, match in enumerate(double_dollar_matches):
        placeholder = f"PLACEHOLDER_{i}"
        modified_text = modified_text.replace(f"$${match}$$", placeholder)
    
    # Now extract single dollar equations
    single_dollar_pattern = r'\$(.*?)\$'
    single_dollar_matches = re.findall(single_dollar_pattern, modified_text, re.DOTALL)
    
    # Combine all matches
    all_equations = double_dollar_matches + single_dollar_matches
    
    # Raise error if no equations found
    if not all_equations:
        raise ValueError("not a valid latex equation")
    
    return all_equations

def extract_bracket_content(equation):
    # Define bracket pairs to check
    bracket_pairs = [
        ('<', '>'),
        ('\\langle', '\\rangle'),
        ('(', ')'),
        ('[', ']'),
        ('\\{', '\\}'),
        ('\\lfloor', '\\rfloor'),
        ('\\lceil', '\\rceil'),
        ('\\|', '\\|'),
        ('|', '|')
    ]
    
    equation = equation.strip()
    
    for start_bracket, end_bracket in bracket_pairs:
        # Check if equation starts with start_bracket and ends with end_bracket
        if equation.startswith(start_bracket) and equation.endswith(end_bracket):
            # Extract content between brackets
            content_start = len(start_bracket)
            content_end = len(equation) - len(end_bracket)
            
            # Make sure there's actual content between the brackets
            if content_end > content_start:
                return equation[content_start:content_end].strip()
    
    # If we get here, no proper brackets were found
    raise ValueError("This is not a proper order parameter")

def extract_operators(latex_expr):
    """
    Extract quantum operators and arithmetic operators from a LaTeX expression.
    
    Args:
        latex_expr: String containing LaTeX expression.
        
    Returns:
        List of dictionaries with operator and arithmetic information.
    """
    import re
    
    # Define pattern for quantum operators with various dagger positions
    operator_pattern = r'([a-zA-Z]+)(?:\^\\(?:dagger|dag))?(?:_\{([^}]*)\})?(?:\^\\(?:dagger|dag))?(?:\(([^)]*)\))?(?:\^\\(?:dagger|dag))?'
    
    # Find all potential operator matches
    operators = []
    for match in re.finditer(operator_pattern, latex_expr):
        op_text = match.group(0)
        
        # Only consider it a valid operator if it has both subscript and momentum
        if '_' in op_text and '(' in op_text:
            # Extract subscript
            subscript_match = re.search(r'_\{([^}]*)\}', op_text)
            subscript = subscript_match.group(1) if subscript_match else None
            
            # Extract momentum
            momentum_match = re.search(r'\(([^)]*)\)', op_text)
            momentum = momentum_match.group(1) if momentum_match else None
            
            operators.append({
                'text': op_text,
                'start': match.start(),
                'end': match.end(),
                'symbol': match.group(1),
                'has_dagger': '\\dagger' in op_text or '\\dag' in op_text,
                'subscript': subscript,
                'momentum': momentum
            })
    
    # Sort operators by position
    operators.sort(key=lambda x: x['start'])
    
    # Process operators and find arithmetic operators between them
    results = []
    
    # Add the first operator if there are any
    if operators:
        first_op = operators[0]
        results.append({
            'type': 'operator',
            'symbol': first_op['symbol'],
            'dagger': first_op['has_dagger'],
            'subscript': first_op['subscript'],
            'momentum': first_op['momentum'],
            'full_match': first_op['text']
        })
    
    # Process pairs of consecutive operators
    for i in range(1, len(operators)):
        prev_op = operators[i-1]
        curr_op = operators[i]
        
        # Check what's between these operators
        between = latex_expr[prev_op['end']:curr_op['start']].strip()
        
        # Look for arithmetic operators (LaTeX and standard)
        arithmetic_match = re.search(r'\\times|\\cdot|[+\-*/]', between)
        if arithmetic_match:
            results.append({
                'type': 'arithmetic',
                'symbol': arithmetic_match.group(0)
            })
        else:
            # If there's nothing or no recognized arithmetic operator, add an empty one
            results.append({
                'type': 'arithmetic',
                'symbol': ''
            })
        
        # Add the current operator
        results.append({
            'type': 'operator',
            'symbol': curr_op['symbol'],
            'dagger': curr_op['has_dagger'],
            'subscript': curr_op['subscript'],
            'momentum': curr_op['momentum'],
            'full_match': curr_op['text']
        })
    
    return results

def construct_operator(op):
    assert op['type']=='operator'
    operator = Operator(f"c_{{{op['subscript']}}}({op['momentum']})")
    if op['dagger']: 
        operator = Dagger(operator)
    return operator
def arithematic(parent,child1, child2):
    assert parent['type']=='arithmetic'
    symbol=parent['symbol'].strip()
    if symbol=='+':
        return child1+child2
    if symbol=='-':
        return child1-child2
    if symbol=='*' or symbol=='' or symbol == r'\times':
        return child1*child2
    if symbol=='/':
        return child1/child2
    raise ValueError(r"Invalid latex expression: Unrecognize operator {symbol}")

def latex_to_expr(parse_latex_list):
    expr = None
    parent = None
    for x in parse_latex_list:
        if x['type'] == 'operator':
            op_x=construct_operator(x)
            if expr is None:
                expr = op_x
            else:
                expr = arithematic(parent, expr, op_x)
                parent = None
        if x['type'] == 'arithmetic':
            if parent is None:
                parent = x
            else:
                raise ValueError("Invalid latex expression: Two successive operators")
    return expr

def list_latex_to_expr(expr):
    equations = extract_latex_equations(expr)
    expr_set = set({})
    for equation in equations:
        equation_ = extract_bracket_content(equation)
        equation_list = extract_operators(equation_)
        expr = latex_to_expr(equation_list)
        expr_set.add(expr)
    return expr_set
    
def iscorrect(expr,ground_truth):
    expr_set = list_latex_to_expr(expr)
    expr_set_normalorder = {recursiveapply(expr,normalorder_rule) for expr in expr_set}
    ground_truth_normalorder = {recursiveapply(expr,normalorder_rule) for expr in ground_truth}
    return expr_set_normalorder == ground_truth_normalorder
