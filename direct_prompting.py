import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Access token and model details
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the translation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device = 'cpu'
)

def generate_response(en):
    messages = [
        {"role": "system", "content":"""
    You are an expert translator specializing in English-to-Vietnamese translation. Your task is to translate the following document sentence-by-sentence, ensuring that each translation builds on the context provided by the previously translated sentences.
    Key Guidelines:
    1.	Maintain Lexical and Terminological Consistency: Use consistent terminology, especially for technical terms, proper nouns, and recurring phrases.
    2.	Pronoun Accuracy: Ensure correct usage of pronouns that aligns with the context and grammatical rules of Vietnamese.
    3.	Deixis and Context Awareness: Resolve references such as "this," "that," or "it" appropriately to preserve meaning in Vietnamese.
    4.	Coherence and Cohesion: Translate each sentence in a way that connects logically and stylistically with the preceding translations, ensuring a smooth flow.
    5.	Tone Appropriateness: Maintain the tone and register of the original document, whether formal, technical, or conversational.
    The user's input would be in this format:
    [Sentence 1]
    [Sentence 2]
    [Sentence 3]
    ...
    Output Requirements:
    Provide the translations in this format:
    Sentence 1
    Sentence 2
    Sentence 3
    ...

    Do not add any additional information such as labels or numbering.
    Ensure that each translation reflects the above guidelines.
    """},
        {"role": "user", "content": en},
    ]
    outputs = pipe(
        messages,
        max_new_tokens = 1024
    )
    return outputs[0]["generated_text"][-1]['content']

prompt = """
Alice is a high school girl.
She is majoring in Mathematical.
She has a younger brother named Anthony.
"""

print(generate_response(prompt))