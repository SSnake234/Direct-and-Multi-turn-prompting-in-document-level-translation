import os
import torch
from transformers import pipeline

# Access token and model details
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Define the translation pipeline
pipe = pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.bfloat16,
    device_map = 'auto'
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

# Function to process a single file
def process_file(file_path, pipeline):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.readlines()
    response = generate_response("".join(content))
    return response

# Function to process all .txt files in a folder
def process_folder(source_folder_path, target_folder_path, pipeline):
    for file_name in os.listdir(source_folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(source_folder_path, file_name)
            print(f"Processing file: {file_name}")
            try:
                translation = process_file(file_path, pipeline)
                output_file = os.path.join(target_folder_path, f"translated_{file_name}")
                with open(output_file, "w", encoding="utf-8") as out_file:
                    out_file.write(translation)
                print(f"Saved translation to {output_file}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

source_folder_path = ""
target_folder_path = ""
process_folder(source_folder_path, target_folder_path, pipeline)