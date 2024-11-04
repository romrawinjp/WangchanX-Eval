import os
import pathlib
import transformers
import torch
import pandas as pd
from tqdm.auto import tqdm
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


root_dir=pathlib.Path(__file__).parent.resolve()

eval_df = pd.read_csv(os.path.join(root_dir,"eval.csv"))


parser = argparse.ArgumentParser()
parser.add_argument("-m","--model", help="HuggingFace model id")
args = parser.parse_args()


model_path = args.model
# model_path = "../models/__"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
print(model)

# load tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
EOS_TOKEN = tokenizer.eos_token

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    return_full_text=False
)

terminators = [
   pipeline.tokenizer.eos_token_id,
   pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

predictions=[]

for reference,context,question in tqdm(list(zip(eval_df.references,
                                                eval_df.context,
                                                eval_df.question))):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Read the context and answer the question."},
        {"role": "user", "content": f"Context : {context}\nQuestion: {question}"},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipeline(prompt, 
                       eos_token_id = terminators,
                       max_new_tokens=512,
                       do_sample=False)  # Don't change do_sample=False
    predictions.append(outputs[0]["generated_text"])

eval_df["predictions"]=predictions

eval_df.to_csv(model_path.split("/")[-1]+".csv",index=False)