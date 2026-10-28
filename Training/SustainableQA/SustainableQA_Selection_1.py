#Training question selection based on and modified from Muennighoff, N, Yang, Z, Shi, W, Li, XL, Fei-Fei, L, Hajishirzi, H, Zettlemoyer, L, Liang, P, Candès, E & Hashimoto, T 2025, s1: Simple test-time scaling. Available from: https://github.com/simplescaling/s1 [19. September 2026] under Apache License Version 2.0, January 2004 found under /Third_Party_Licenses

import json
import re
import unicodedata
from collections import defaultdict
from fractions import Fraction


Input = {"train": "Training/SustainableQA/train.json","validation": "Training/SustainableQA/validation.json","test": "Training/SustainableQA/test.json",}
ESGenius = "Benchmark/ESGenius/ESGenius_1136q.json"
VSBenchmark = "Benchmark/VSBenchmark/VSBenchmark.json"
Result = ("Training/SustainableQA/SustainableQA_Selection_1.json")

#SustainableQA Structure
QA_Container = ("qa_pairs_factoid", "qa_pairs_non_factoid")
QA_types_by_container = {"qa_pairs_factoid": "factoid", "qa_pairs_non_factoid": "non_factoid", "qa_pairs_tabular": "tabular",}
Difficulty_types = {"factoid": "F", "non_factoid": "NF", "tabular": "T",}
Categories = ("EU Taxonomy", "ESG", "Sustainability")
Question_types = ("factoid", "non_factoid", "tabular")
Diverity_quotas = {
    "EU Taxonomy": {
        "factoid": 261,
        "non_factoid": 303,
        "tabular": 36,
    },
    "ESG": {
        "factoid": 279,
        "non_factoid": 321,
        "tabular": 0,
    },
    "Sustainability": {
        "factoid": 279,
        "non_factoid": 321,
        "tabular": 0,
    },
}

#F1 Scores to identify questions difficulty based on their question word, as laid out by Bahak, H, Taheri, F, Zojaji, Z & Kazemi, A 2024, 'Evaluating ChatGPT as a Question Answering System: A Comprehensive Analysis and Comparison with Existing Models', Journal of Computing and Security, vol. 11, no. 2, pp. 97–116. Available from: https://jcomsec.ui.ac.ir/article_28768.html.
Difficulty_Scores_Question_Words = { "why": 665, "what": 740, "how": 760, "who": 765, "where": 780, "when": 785, "which": 800,}
F1_min = min(Difficulty_Scores_Question_Words.values())
F1_max = max(Difficulty_Scores_Question_Words.values())
Identify_question_word = re.compile(r"\b(" + "|".join(Difficulty_Scores_Question_Words) + r")\b",flags=re.IGNORECASE,)

NGRAM_length = 8
Benchmarks = (
    (ESGenius, "question"),
    (VSBenchmark, "problem"),
)

#Load Training-, Test-, an d Validation Split from SustainableQA as one dictionary
def load_sustainableqa():
    sustainableqa = {}
    for split, file_path in Input.items():
        with open(file_path, encoding="utf-8") as file:
            sustainableqa[split] = json.load(file)
    return sustainableqa

#Loops through train, test and validation and all_chunk_qas and all_table_qas and returns dictionaries of containing the qa_pairs 
def iterate_qa_containers(sustainableqa):
    for split, split_data in sustainableqa.items(): #split_data contains train.json, test.json and validation.json
        for chunk_index, chunk in enumerate(
            split_data["all_chunks_qas"],
            start=1,
        ):
            for container_name in QA_Container:
                container = chunk[container_name]
                yield {
                    "split": split,
                    "source_index": chunk_index,
                    "container_name": container_name,
                    "question_type": QA_types_by_container[container_name],
                    "source": chunk,
                    "container": container, #contains actual qa-pairs
                }

        for table_index, table in enumerate(
            split_data["all_table_qas"],
            start=1,
        ):
            yield {
                "split": split,
                "source_index": table_index,
                "container_name": "qa_pairs_tabular",
                "question_type": "tabular",
                "source": table,
                "container": table,
            }

#Defines and assigns a unique ID for each question 
def build_question_id(sustainableqa):
    for container_info in iterate_qa_containers(sustainableqa):
        questions = container_info["container"]["qa_pairs"]
        for question_index, question_data in enumerate(questions, start=1):
            question_data["question_id"] = (
                f"{container_info['split']}:"
                f"{container_info['source_index']:06d}:"
                f"{container_info['container_name']}:"
                f"{question_index:04d}"
            )

#Exclusion of dublicates
#Checks if columns (Problems) have dublicates
def is_unique(elem, column, memory):
    if elem[column] in memory:
        return False
    memory.add(elem[column])
    return True

#Iterates over sustainableqa chunks and identifies dublicate questions.  
def deduplicate_qa_pairs(sustainableqa):
    memory = set()
    dublicates = 0

    for container_info in iterate_qa_containers(sustainableqa):
        container = container_info["container"]
        questions = container["qa_pairs"]
        retained_questions = []

        for question_data in questions:
            if is_unique(question_data, "question", memory):
                retained_questions.append(question_data)
            else:
                dublicates += 1

        container["qa_pairs"] = retained_questions

    return dublicates

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

#Loads benchmarks and builds a set containing all ngrams for the benchmarks questions
def load_benchmark_ngram_lookup():
    lookup = set()
    for file_path, question_field in Benchmarks:
        with open(file_path, encoding="utf-8") as file:
            questions = json.load(file)
        for example in questions:
            lookup.update(word_ngrams(example[question_field]))
    return lookup

#Excludes training questions that have an ngram overlap with benchmarking questions
def exclude_contaminated_questions(sustainableqa, reference_lookup):
    exclusions = 0

    for container_info in iterate_qa_containers(sustainableqa):
        container = container_info["container"]
        questions = container["qa_pairs"]
        retained_questions = []

        for question_data in questions:
            if any(
                ngram in reference_lookup
                for ngram in word_ngrams(question_data["question"])
            ):
                exclusions += 1
            else:
                retained_questions.append(question_data)

        container["qa_pairs"] = retained_questions

    return exclusions

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
def exclude_qa_pairs_with_formatting_issues(sustainableqa):
    exclusions = 0

    for container_info in iterate_qa_containers(sustainableqa):
        container = container_info["container"]
        questions = container["qa_pairs"]
        retained_questions = []

        for question_data in questions:
            question_issues = find_formatting_issues(question_data["question"])
            answer_issues = find_formatting_issues(question_data["answer"])
            if question_issues or answer_issues:
                exclusions += 1
            else:
                retained_questions.append(question_data)

        container["qa_pairs"] = retained_questions

    return exclusions

#Identifies the question words in a question and returns the one with the lowest F1-score (=most difficult)
def find_hardest_question_word(question):
    matches = {
        match.group(0).casefold()
        for match in Identify_question_word.finditer(question)
    }
    if not matches:
        return None
    return min(matches, key=Difficulty_Scores_Question_Words.__getitem__)

#Excludes non-factoid questions that dont contain any question word
def exclude_questions_without_ranked_question_word(sustainableqa):
    """Non-Factoid- und Tabellenfragen ohne definiertes Fragewort entfernen."""
    exclusions = 0

    for container_info in iterate_qa_containers(sustainableqa):
        if container_info["question_type"] == "factoid":
            continue

        container = container_info["container"]
        questions = container["qa_pairs"]
        retained_questions = []
        for question_data in questions:
            if find_hardest_question_word(question_data["question"]) is None:
                exclusions += 1
            else:
                retained_questions.append(question_data)
        container["qa_pairs"] = retained_questions

    return exclusions

#For factoid questions the number of spans relevant for an answer is counted
def count_relevant_text_spans(question_data, chunk):
    normalized_answer = normalize_string(question_data["answer"])
    spans = {
        normalize_string(span)
        for group in chunk["spans"]["groups"]
        for span in group["spans"]
    }
    return sum(
        bool(normalized_span) and normalized_span in normalized_answer
        for normalized_span in spans
    )

#Assigns IDs to questions based on their difficulty separately by question type. Factoid questions are ranked based on number of spans and non-factoid and tabular based on question word and question length
def assign_difficulty_ids(sustainableqa):
    records_by_type = {question_type: [] for question_type in Difficulty_types}

#Saves number of relevant text spans per factoid question as difficulty_value and hardest question words and their F1 scores per non-factoid and tabular questions
    for container_info in iterate_qa_containers(sustainableqa):
        question_type = container_info["question_type"]
        for question_data in container_info["container"]["qa_pairs"]:
            if question_type == "factoid":
                difficulty_value = count_relevant_text_spans(question_data, container_info["source"],)
                record = {"question_data": question_data, "difficulty_value": difficulty_value,}
            else:
                question_word = find_hardest_question_word(question_data["question"])
                record = {"question_data": question_data, "question_word": question_word, "question_word_f1": Difficulty_Scores_Question_Words[question_word],"question_length": len(question_data["question"]),}
            records_by_type[question_type].append(record)

#
    for question_type, records in records_by_type.items():
        prefix = Difficulty_types[question_type]

#Calculates maximum question length range for all non-factoid and tabular questions
        if question_type != "factoid":
            lengths = [record["question_length"] for record in records]
            length_min = min(lengths)
            length_max = max(lengths)
            length_range = length_max - length_min

#Calculates and saves length and F1-Scores for non-factoid and tabular questions 
            for record in records:
                difficulty_f1 = Fraction(F1_max - record["question_word_f1"],F1_max - F1_min,)
                difficulty_length = Fraction(record["question_length"] - length_min, length_range,)
                difficulty_value = (difficulty_f1 + difficulty_length) / 2
                record["difficulty_value"] = difficulty_value

                question_data = record["question_data"]
                question_data["difficulty_question_word"] = record["question_word"]
                question_data["question_word_f1"] = (record["question_word_f1"] / 1000)
                question_data["difficulty_f1"] = round(float(difficulty_f1),12,)
                question_data["difficulty_length"] = round(float(difficulty_length),12,)
                question_data["difficulty_score"] = round(float(difficulty_value),12,)

#Assigns difficulty IDs per question type
        values = sorted({record["difficulty_value"] for record in records},reverse=True,)
        difficulty_ids = {
            value: f"{prefix}-{rank:06d}"
            for rank, value in enumerate(values, start=1)
        }
        for record in records:
            value = record["difficulty_value"]
            question_data = record["question_data"]
            question_data["difficulty_id"] = difficulty_ids[value]


def rename_category(category):
    categories = {
        "eu taxonomy": "EU Taxonomy",
        "esg": "ESG",
        "sustainability": "Sustainability",
    }
    return categories[normalize_string(category)]

#Retrieves difficulty rank of a question
def difficulty_rank(question_data):
    return int(question_data["difficulty_id"].rsplit("-", maxsplit=1)[1])

#Selects the most difficult questions respecting the category distribution
def select_preselection(sustainableqa):
    candidates = defaultdict(list)

#Chunks are assigned a unique source_sequence_id to select only one QA-pair per chunk
    for container_info in iterate_qa_containers(sustainableqa):
        source = container_info["source"]
        category = rename_category(source["metadata"]["classification"])
        question_type = container_info["question_type"]
        source_kind = "table" if question_type == "tabular" else "text"
        source_sequence_id = (
            f"{container_info['split']}:"
            f"{source_kind}:"
            f"{container_info['source_index']:06d}"
        )

#Candidates are grouped by all question_type category combinations
        for question_data in container_info["container"]["qa_pairs"]:
            candidates[(category, question_type)].append(
                {
                    "question_data": question_data,
                    "source": source,
                    "split": container_info["split"],
                    "source_kind": source_kind,
                    "source_sequence_id": source_sequence_id,
                }
            )

#Selection of most difficult questions respecting the category ad question_type quotas
    selected_questions = []
    selected_counts = {}
    for category in Categories:
        selected_counts[category] = {}
        for question_type in Question_types:
            quota = Diverity_quotas[category][question_type]
            group_candidates = sorted(
                candidates[(category, question_type)],
                key=lambda item: (
                    difficulty_rank(item["question_data"]),
                    item["question_data"]["question_id"],
                ),
            )
            used_sequences = set()
            group_selection = []

            if quota:
                for candidate in group_candidates:
                    source_sequence_id = candidate["source_sequence_id"]
                    if source_sequence_id in used_sequences:
                        continue
                    used_sequences.add(source_sequence_id)
                    group_selection.append(candidate)
                    if len(group_selection) == quota:
                        break

            selected_counts[category][question_type] = len(group_selection)
            for group_rank, candidate in enumerate(group_selection, start=1):
                source = candidate["source"]
                selected_item = {
                    "category": category,
                    "question_type": question_type,
                    "selection_rank_in_group": group_rank,
                    "split": candidate["split"],
                    "source_kind": candidate["source_kind"],
                    "source_sequence_id": candidate["source_sequence_id"],
                    "metadata": source["metadata"],
                }
                if "spans" in source:
                    selected_item["spans"] = source["spans"]
                selected_item.update(candidate["question_data"])
                selected_questions.append(selected_item)

    return {
        "selection_summary": {
            "total_questions": len(selected_questions),
            "category_order": list(Categories),
            "question_type_order": list(Question_types),
            "quotas": Diverity_quotas,
            "selected_counts": selected_counts,
        },
        "selected_questions": selected_questions,
    }

#Safe the selected questions
def write_preselection(selected_questions):
    with open(Result, "w", encoding="utf-8") as file:
        json.dump(selected_questions, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main():
    sustainableqa = load_sustainableqa()
    build_question_id(sustainableqa)
    dublicates = deduplicate_qa_pairs(sustainableqa)
    exclusions = exclude_contaminated_questions(
        sustainableqa,
        load_benchmark_ngram_lookup(),
    )
    formatting_exclusions = exclude_qa_pairs_with_formatting_issues(
        sustainableqa
    )
    question_word_exclusion = (
        exclude_questions_without_ranked_question_word(sustainableqa)
    ) 
    assign_difficulty_ids(sustainableqa)
    selected_questions = select_preselection(sustainableqa)
    write_preselection(selected_questions)

    for split, data in sustainableqa.items():

        print(f"Number of QA-pairs in input: {split}: {len(data['all_chunks_qas'])}")

    print(f"Excluded dublicates: {dublicates}")
    print(f"Excluded benchmark overlaps: {exclusions}")
    print("Excluded due to fromatting issues: "f"{formatting_exclusions}")
    print("Excluded due to missing question word: "f"{question_word_exclusion}")
    print(f"Number of resulting QA-pairs "f"{selected_questions['selection_summary']['total_questions']}")

if __name__ == "__main__":
    main()
