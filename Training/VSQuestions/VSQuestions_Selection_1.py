#Training question selection based on Muennighoff, N, Yang, Z, Shi, W, Li, XL, Fei-Fei, L, Hajishirzi, H, Zettlemoyer, L, Liang, P, Candès, E & Hashimoto, T 2025, s1: Simple test-time scaling. Available from: https://github.com/simplescaling/s1

import json
import re
import unicodedata

Input = "Training/VSQuestions/VSQuestions.json"
ESGenius = "Benchmark/ESGenius/ESGenius_1136q.json"
VSBenchmark = "Benchmark/VSBenchmark/VSBenchmark.json"

Result = ("Training/VSQuestions/VSQuestions_Selection_1.json")

NGRAM_length = 8
Benchmarks = (
    (ESGenius, "question"),
    (VSBenchmark, "problem"),
)

#Defines a unique ID for each question 
def build_question_id(prefix, source_id):
    formatted_id = f"{int(source_id):04d}"
    return f"{prefix}:{formatted_id}"


#Loads QA pairs from JSON as a list and assigns the unique IDs
def load_qa_pairs():
    questions = []
    source_counts = {"VSME": 0, "VS Standard": 0}

    with open(Input, encoding="utf-8") as file:
        source_data = json.load(file)

    for qa_pair in source_data:
        if "VSME Module" in qa_pair:
            source_name = "VSME"
            id_prefix = "VSME"
            report = qa_pair["Report"]
            module = qa_pair["VSME Module"]
        elif "VS Module" in qa_pair:
            source_name = "VS Standard"
            id_prefix = "VS"
            report = ""
            module = qa_pair["VS Module"]

        source_id = qa_pair["ID"]
        question = qa_pair["Problem"]
        answer = qa_pair["Solution"]

        source_counts[source_name] += 1
        questions.append(
            {
                "id": build_question_id(id_prefix, source_id),
                "problem": question.strip(),
                "solution": answer.strip(),
                "question_type": "non_factoid",
                "category": "VS/VSME",
                "source_dataset": source_name,
                "source_id": source_id,
                "report": report,
                "module": module,
            }
        )

    return questions, source_counts

#Exclusion of dublicates
#Checks if columns (Problems) have dublicates
def is_unique(elem, column, memory):
    if elem[column] in memory:
        return False
    memory.add(elem[column])
    return True

#Removes duplicate questions and counts them
def deduplicate_qa_pairs(questions):
    memory = set()
    dublicates = 0
    retained_questions = []

    for question_data in questions:
        if is_unique(question_data, "problem", memory):
            retained_questions.append(question_data)
        else:
            dublicates += 1

    return retained_questions, dublicates

#Exclusion of 8-gram overlaps with benchmarks
#Normalization of text towards unicode, only lowercase, and only single spaces
def normalize_string(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

#Build Ngrams of 8 words from normalized text
def word_ngrams(text, size=NGRAM_length):
    words = normalize_string(text).split()
    return (
        " ".join(words[index : index + size])
        for index in range(len(words) - size + 1)
    )

#Loads benchmarks and builds a set containing all ngrams for the benchmark questions
def load_benchmark_ngram_lookup():
    lookup = set()
    for file_path, question_field in Benchmarks:
        with open(file_path, encoding="utf-8") as file:
            questions = json.load(file)
        for example in questions:
            lookup.update(word_ngrams(example[question_field]))
    return lookup

#Excludes training questions that have an ngram overlap with benchmarking questions
def exclude_contaminated_questions(questions, reference_lookup):
    exclusions = 0
    retained_questions = []

    for question_data in questions:
        if any(
            ngram in reference_lookup
            for ngram in word_ngrams(question_data["problem"])
        ):
            exclusions += 1
        else:
            retained_questions.append(question_data)

    return retained_questions, exclusions

#Identifies strings with formatting problems like markdown, latex expressions or excluded references
def find_formatting_issues(text):
    issues = []
    if "[asy]" in text:
        issues.append("[asy]")
    if "![Image]" in text:
        issues.append("![Image]")
    if re.search(r"\bp[1-9]\d{0,2}\.", text):
        issues.append("page reference")
    if "**" in text:
        issues.append("**")
    if "\\" in text:
        issues.append("\\")
    if "$" in text:
        issues.append("$")
    return issues

#Checks training data problems and solutions for formatting issues and excludes them
def exclude_qa_pairs_with_formatting_issues(questions):
    exclusions = 0
    retained_questions = []

    for question_data in questions:
        question_issues = find_formatting_issues(question_data["problem"])
        answer_issues = find_formatting_issues(question_data["solution"])
        if question_issues or answer_issues:
            exclusions += 1
        else:
            retained_questions.append(question_data)

    return retained_questions, exclusions

#Saves QA pairs that passed dedublication, contamination and formatting check
def write_result_json(result):
    with open(Result, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main():
    questions, source_counts = load_qa_pairs()
    input_count = len(questions)

    questions, duplicate_count = deduplicate_qa_pairs(questions)

    questions, exclusions = exclude_contaminated_questions(
        questions,
        load_benchmark_ngram_lookup(),
    )

    questions, formatting_exclusions = (
        exclude_qa_pairs_with_formatting_issues(questions)
    )

    result = {"selected_questions": questions}
    write_result_json(result)

    print(f"Number of QA-pairs in input: {input_count}")
    print(f"Excluded dublicates: {duplicate_count}")
    print(f"Excluded benchmark overlaps: {exclusions}")
    print("Excluded due to formatting issues: "f"{formatting_exclusions}")
    print(f"Number of resulting QA-pairs: {len(questions)}")

if __name__ == "__main__":
    main()
