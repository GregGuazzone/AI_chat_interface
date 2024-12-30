import pandas as pd

def get_available_commands():           #Currently only support two commands
    return {'FILTER': ['column', 'condition', 'value'], 'SELECT': ['columns'], 'SORT_LIMIT': ['column', 'direction', 'n']}

def filter(df, column, condition, value):
    # Get the column's dtype
    col_dtype = df[column].dtype
    # Convert value based on dtype
    if pd.api.types.is_numeric_dtype(col_dtype):
        value = float(value)
    elif pd.api.types.is_datetime64_any_dtype(col_dtype):
        value = pd.to_datetime(value)
    elif pd.api.types.is_bool_dtype(col_dtype):
        value = bool(value)
    # Add more type conversions as needed
    print(column, condition, value)
    if condition == '==':
        df = df[df[column] == value]
    elif condition == '!=':
        df = df[df[column] != value]
    elif condition == '>':
        df = df[df[column] > value]
    elif condition == '<':
        df = df[df[column] < value]
    elif condition == '>=':
        df = df[df[column] >= value]
    elif condition == '<=':
        df = df[df[column] <= value]
    return df

def sort_limit(df, column, direction='DESC', n=1):
    if direction.upper() == 'ASC':
        return df.sort_values(by=column, ascending=True).head(n)
    return df.sort_values(by=column, ascending=False).head(n)

def select(df, columns):
    return df[columns]

def apply_command(df, command):
    if command['command'].upper() == 'FILTER':
        return filter(df, command['column'], command['condition'], command['value'])
    elif command['command'].upper() == 'SORT_LIMIT':
        return sort_limit(df, command['column'], command['direction'], int(command['n']))
    elif command['command'].upper() == 'SELECT':
        return select(df, command['columns'])
    return df
    
def apply_transformations(df, transformations):
    print(transformations)
    for transformation in transformations:
        df = apply_command(df, transformation)
    return df


