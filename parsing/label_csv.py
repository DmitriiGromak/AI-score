import sys
import csv
import ast
import re
import multiprocessing as mp
from collections import defaultdict

CORE_AI_TERMS = [
    "artificial intelligence", "machine learning", "natural language processing", "computer vision",
    "искусственный интеллект", "машинное обучение", "обработка естественного языка", "компьютерное зрение",
    "ML", "AI", "NLP", "CV"
]

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

def get_core_helpers(cores):
    core_lowers = [c.lower() for c in cores]
    patterns = []
    for c in core_lowers:
        escaped = re.escape(c)
        pat_str = r'\b' + escaped.replace(r'\ ', r'\s+') + r'\b'
        patterns.append(re.compile(pat_str))
    return core_lowers, patterns

def has_core_ai(title, skills_list, core_lowers, patterns):
    skills_lowers = [s.lower() for s in skills_list]
    if any(core == sl for core in core_lowers for sl in skills_lowers):
        return True
    title_lower = str(title).lower()
    for pat in patterns:
        if pat.search(title_lower):
            return True
    return False

def process_chunk(chunk_rows, core_lowers, patterns):
    total = defaultdict(int)
    core_count = defaultdict(int)
    for row in chunk_rows:
        title = row.get('name', '')
        raw = row.get('raw_skills', '')
        skills_list = parse_skills(raw)
        skills_set = set(skills_list)
        has_core = has_core_ai(title, skills_list, core_lowers, patterns)
        for s in skills_set:
            total[s] += 1
            if has_core:
                core_count[s] += 1
    return dict(total), dict(core_count)

if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    with open(input_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    core_lowers, patterns = get_core_helpers(CORE_AI_TERMS)
    num_processes = mp.cpu_count()
    chunk_size = len(all_rows) // num_processes + 1
    chunks = [all_rows[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
    with mp.Pool(num_processes) as pool:
        results = pool.starmap(process_chunk, [(chunk, core_lowers, patterns) for chunk in chunks])
    total_global = defaultdict(int)
    core_global = defaultdict(int)
    for t, c in results:
        for s in t:
            total_global[s] += t[s]
            core_global[s] += c.get(s, 0)
    skill_list = []
    for s in total_global:
        tot = total_global[s]
        score = core_global[s] / tot
        skill_list.append([s, score])
    skill_list.sort(key=lambda x: x[1], reverse=True)
    with open(output_csv, 'w', newline='', encoding='utf-8', errors='replace') as f:
        writer = csv.writer(f)
        writer.writerow(['skill', 'score'])
        writer.writerows(skill_list)