from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
from update import apply_command
from update import get_available_commands
import torch
import transformers
from accelerate import disk_offload
import ast
import warnings

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'
device = get_device_map()  # 'cpu'

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map=device,
    torch_dtype=torch.bfloat16,
)

model = torch.compile(model)
disk_offload(model, offload_dir="offload")

class AIChat:
    def __init__(self, df):
        """
        Initialize the AIChat with an LLM and a DataFrame schema.
        """
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.schema = df.dtypes.to_dict()  # Schema needed to generate appropriate prompts
        self.messages = [{"role": "system", "content": f"Here is the schema of the DataFrame you will be working with: {self.schema}. Your task is to generate an ordered sequence of data transformations based on user input. Provide a JSON object with a 'transformations' key. Each transformation should include a 'command' (one of the following: {get_available_commands()}) and the necessary parameters, ensure that there is a logical ordering of the commands that limits computation. The output should STRICTLY be in JSON and have the following format: \"transformations\": [{{\"command\": \"FILTER\", \"column\": \"col\", \"condition\": \"==\", \"value\": \"X\"}}, {{\"command\": \"SELECT\", \"columns\": [\"col1\", \"col2\"]}}]"}]
        self.df = df

    def parse_transformations(self, text):
        """
        Expects the model to return JSON with a 'transformations' key.
        """
        try:
            outputs_dict = ast.literal_eval(text)
            output = json.loads(outputs_dict['content'])
            return output.get("transformations", [])
        except:
            return []

    def chat(self, user_input):
        """
        Process a user query and return the AI's response.
        """
        self.messages.append({"role": "user", "content": user_input})
        outputs = self.pipeline(self.messages, max_new_tokens=256)[0]["generated_text"][-1]
        self.messages.append(outputs)
        transformations = self.parse_transformations(outputs)
        return transformations