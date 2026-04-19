import sys
import csv
import ast
import multiprocessing as mp

def parse_skills(raw_str):
    if not isinstance(raw_str, str) or not raw_str.strip():
        return []
    try:
        skills = ast.literal_eval(raw_str)
        if isinstance(skills, list):
            return [str(s).strip() for s in skills if str(s).strip()]
        return []
    except (ValueError, SyntaxError, TypeError):
        return []

def process_score_chunk(chunk_rows, relatedness):
    scores = []
    for row in chunk_rows:
        raw = row.get('raw_skills', '')
        skills_list = parse_skills(raw)
        skills_set = set(skills_list)
        if not skills_set:
            scores.append(0.0)
            continue
        sum_score = sum(relatedness.get(s, 0.0) for s in skills_set)
        score = sum_score / len(skills_set)
        scores.append(score)
    return scores

if __name__ == "__main__":
    input_csv = sys.argv[1]
    skills_csv = sys.argv[2]
    output_csv = sys.argv[3]
    relatedness = {}
    with open(skills_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            relatedness[row['skill']] = float(row['score'])
    with open(input_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        all_rows = list(reader)
    num_processes = mp.cpu_count()
    chunk_size = len(all_rows) // num_processes + 1
    chunks = [all_rows[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
    with mp.Pool(num_processes) as pool:
        score_lists = pool.starmap(process_score_chunk, [(chunk, relatedness) for chunk in chunks])
    all_scores = []
    for sublist in score_lists:
        all_scores.extend(sublist)
    for row, score in zip(all_rows, all_scores):
        row['ai_relatedness'] = score
    new_headers = list(headers) + ['ai_relatedness']
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_headers)
        writer.writeheader()
        writer.writerows(all_rows)