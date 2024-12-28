import pandas as pd
from update import apply_transformations
from ai_chat import AIChat

def main():
    df = pd.read_csv('test.csv')
    print("I am an AI assistant, here is your DataFrame:")
    print(df)
    ai = AIChat(df)
    while True:
        user_input = input("How would you like to query this dataframe?\n")
        transformations = ai.chat(user_input)
        new_df = apply_transformations(df, transformations)
        print(new_df)
        #Optionally update the original dataframe
        #df = new_df

if __name__ == "__main__":
    main()