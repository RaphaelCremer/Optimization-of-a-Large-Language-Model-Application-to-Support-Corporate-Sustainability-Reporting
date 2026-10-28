#Final selection of 100 VSQuestions
import json
import random

Input = "Training/VSQuestions/VSQuestions_Attempt_1_Grading.json"
Output = "Training/VSQuestions/VSQuestions_Selection_2.json"

with open(Input, "r", encoding="utf-8") as file:
    data = json.load(file)

#Only keep questions with false attempts
false_attempts = []
for element in data:
    if element["grading"] == "No":
        false_attempts.append(element)

#Select 100 wrong answered questions at random
selected_questions = random.sample(false_attempts, 100)

with open(Output, "w", encoding="utf-8") as file:
    json.dump(selected_questions, file, ensure_ascii=False, indent=2)

