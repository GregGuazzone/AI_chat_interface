from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
from update import get_available_commands
import torch
from accelerate import disk_offload
import ast
import re

#Edit this modify the implementation of the llm model as needed
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",     #https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    device_map=device,
    torch_dtype=torch.bfloat16,
)
model = torch.compile(model)
disk_offload(model, offload_dir="offload")

class AIChat:
    """
    A class representing an AI chatbot which generates transformations.
    """
    def __init__(self, df):
        """
        Initialize the AIChat with an LLM and a DataFrame schema.
        """
        #Edit this to use the model you want
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        #

        self.schema = df.dtypes.to_dict()   #Schema needed to generate appropriate prompts
        #Message history which ensures the model has context
        self.messages = [{
                            "role": "system",
                            "content": f"""
                            You are given a dataframe with the following schema:
                            {self.schema}.
                            You will answer questions by providing only a JSON object with a 'transformations' key and the necessary parameters.
                            Each transformation should include a 'command' (one of the following: {get_available_commands()}) and the necessary parameters, ensure there is a logical ordering of the commands that limits computation. 
                            The output should STRICTLY be in JSON and only contain the JSON object with the 'transformations' key, here is a example: 'transformations': [{{\'command\': \"FILTER\", \"column\": \"col\", \'condition\': \"==\", \"value\": \"X\"}}, {{\'command\': \"SORT_LIMIT\", \'column\': \"col1\", \'direction\': \"ASC\", \"n\": \"5\"}} {{\'command\': \"SELECT\", \'columns\': [\"col1\", \"col2\"]}}]
                            """
                        }]  #Original prompt to setup the context
        

    def parse_transformations(self, text):
        """
        Extracts the JSON object with 'transformations' from the input text.
        """
        outputs_dict = ast.literal_eval(text)
        output = json.loads(outputs_dict['content'])
        return output.get("transformations", [])

    def chat(self, user_input):
        """
        Process a user query and return the AI's response.
        """
        self.messages.append({"role": "user", "content": user_input})
        outputs = self.pipeline(self.messages, max_new_tokens=256)[0]["generated_text"][-1]
        self.messages.append(outputs)
        outputs = str(outputs)
        transformations = self.parse_transformations(outputs)
        return transformations