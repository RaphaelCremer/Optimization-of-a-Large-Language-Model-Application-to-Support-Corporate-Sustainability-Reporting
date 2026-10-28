#Final selection of 900 SustainableQA questions
import json
import random

Input = "Training/SustainableQA/SustainableQA_Attempt_1_Grading.json"
Output = "Training/SustainableQA/SustainableQA_Selection_2.json"

#Filters qa_pairs by category and question type
def select_qa_pairs(qa_pairs, category, question_type):
    return [
        qa_pair
        for qa_pair in qa_pairs
        if qa_pair.get("category", "") == category
        and qa_pair.get("question_type", "") == question_type
    ]

def main():
    with open(Input, encoding="utf-8") as file:
        data = json.load(file)

#Only keep questions with false attempts
    false_attempts = []
    for elements in data["selected_questions"]:
        grading = elements.get("grading", "")

        if grading == "Yes": 
            continue
        else:
            false_attempts.append(elements)

#Select random QA-pairs of false attempts with respect to the diversity criterion
    selection_esg_factoid = random.sample(select_qa_pairs(false_attempts, "ESG", "factoid"), 194)
    selection_esg_non_factoid = random.sample(select_qa_pairs(false_attempts, "ESG", "non_factoid"), 106)
    selection_esg_tabular = random.sample(select_qa_pairs(false_attempts, "ESG", "tabular"), 0)

    selection_EU_Taxonomy_factoid = random.sample(select_qa_pairs(false_attempts, "EU Taxonomy", "factoid"), 118)
    selection_EU_Taxonomy_non_factoid = random.sample(select_qa_pairs(false_attempts, "EU Taxonomy", "non_factoid"), 164)
    selection_EU_Taxonomy_tabular = random.sample(select_qa_pairs(false_attempts, "EU Taxonomy", "tabular"), 18)

    selection_sustainability_factoid = random.sample(select_qa_pairs(false_attempts, "Sustainability", "factoid"), 215)
    selection_sustainability_non_factoid = random.sample(select_qa_pairs(false_attempts, "Sustainability", "non_factoid"), 85)
    selection_sustainability_tabular = random.sample(select_qa_pairs(false_attempts, "Sustainability", "tabular"), 0)

    with open(Output, "w", encoding="utf-8") as file:
        json.dump(
            {
                "selected_questions": (
                    selection_esg_factoid
                    + selection_esg_non_factoid
                    + selection_esg_tabular
                    + selection_EU_Taxonomy_factoid
                    + selection_EU_Taxonomy_non_factoid
                    + selection_EU_Taxonomy_tabular
                    + selection_sustainability_factoid
                    + selection_sustainability_non_factoid
                    + selection_sustainability_tabular
                )
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

if __name__ == "__main__":
    main()
