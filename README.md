# **Document-level Machine Translation (DocMT)**

**Document-level Machine Translation (DocMT)** involves translating entire documents instead of using the traditional sentence-by-sentence approach. This technique leverages the broader context of a document, ensuring better linguistic coherence and consistency throughout the translation.

---

## **Project Overview**  
This repository provides code to compare **direct prompting** and **multi-turn prompting** approaches for DocMT.  
- Both approaches use the [Llama 3.1-8B Instruct model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).  
- Evaluation is performed using the **BLEU score** to measure translation quality.  
- The **multi-turn prompting** approach is inspired by the insights from [this paper](https://arxiv.org/pdf/2409.06790).  

---

## **Getting Started**  

### **Dependencies**  
Ensure you have the following Python libraries installed:  
- `transformers`  
- `torch`  
- `sacrebleu`  

Install them using:  
```bash
pip install transformers torch sacrebleu
```

### **Usage**  

#### 1. Direct Prompting  
- Update the following variables in `direct_prompting.py` to match your file structure:  
  - `source_folder_path`: Path to the folder containing the source text files.  
  - `target_folder_path`: Path to the folder where the translated files will be saved.  

- Run the script:  
  ```bash
  python direct_prompting.py
  ```

#### . Multi-turn Prompting
- Update the following variables in `multi_turn.py` to match your file structure:
  - `source_folder_path`: Path to the folder containing the source text files.  
  - `target_folder_path`: Path to the folder where the translated files will be saved.  

- Run the script:  
  ```bash
  python multi_turn.py
  ```
