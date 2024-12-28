import pandas as pd

def get_available_commands():           #Currently only support two commands
    return ['FILTER', 'SELECT']

def apply_command(df, command):
    if command['command'].upper() == 'FILTER':
        column = command['column']
        condition = command['condition']
        value = command['value']
        
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
    elif command['command'].upper() == 'SELECT':
        columns = command['columns']
        df = df[columns]
    return df
    
def apply_transformations(df, transformations):
    print(transformations)
    for transformation in transformations:
        df = apply_command(df, transformation)
    return df


