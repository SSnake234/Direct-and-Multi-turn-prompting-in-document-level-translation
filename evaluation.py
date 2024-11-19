from sacrebleu.metrics import BLEU
import os

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def calculate_bleu(references, candidates, bleu):
    return bleu.corpus_score(candidates, [references])

def calculate_average_bleu(reference_dir, candidate_dir, bleu):
    reference_files = sorted(os.listdir(reference_dir))
    candidate_files = sorted(os.listdir(candidate_dir))
    
    if len(reference_files) != len(candidate_files):
        raise ValueError("Mismatch in the number of reference and candidate files.")
    
    refs = []
    cands = []
    for ref_file, cand_file in zip(reference_files, candidate_files):
        ref_path = os.path.join(reference_dir, ref_file)
        cand_path = os.path.join(candidate_dir, cand_file)
        reference = read_file(ref_path)
        candidate = read_file(cand_path)
        refs.append(reference)
        cands.append(candidate)
    print(refs)
    return calculate_bleu(refs, cands, bleu)

bleu = BLEU()
reference_dir = ""
candidate_dir = ""
bleu_score = calculate_average_bleu(reference_dir, candidate_dir, bleu)
print(bleu_score)
