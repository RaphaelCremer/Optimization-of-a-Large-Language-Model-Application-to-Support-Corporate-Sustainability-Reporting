#gpt5.6-luna judges the attempts on SustainableQA
import openai 
import getpass
import os
import json

Input = "Training/SustainableQA/SustainableQA_Attempt_1.json"
Output = "Training/SustainableQA/SustainableQA_Attempt_1_Grading.json"

def main():
    OPENAI_API_KEY = os.getenv("OpenAI token")
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = getpass.getpass("Enter OpenAI token: ")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    with open( 
       Input,
        encoding="utf-8"
    ) as file:
        data = json.load(file)

#Problem, Attempt and Solution is passed to gpt-5.6-luna, which generates Yes if attempt = solution and No if not
    for Question_Number, element in enumerate(
        data["selected_questions"],
        start=1,
    ):

        completion = client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=[
                {
                    "role": "system",
                    "content": """
                                You are an AI assistant for grading an ESG problem. The user provided you the question itself, an attempt made by
                                a student and the correct answer to the problem. Your job is to judge whether the attempt is correct by comparing it with the
                                correct answer. If the final answer in the attempt conveys the same essential factual meaning as the correct answer you should respond with Yes, even if the wording differs, the attempt is more detailed than the solution or the attempt contains additional correct information. 
                                If it is not the same, you should respond with "No".
                                The user provided the attempt and the correct answer in the following format:
                                ##Problem
                                {problem}
                                ##Attempt
                                {attempt}
                                ##Correct answer
                                {solution}
                                Give as your response only "Yes" when correct or "No" when incorrect (without quotes)
                                """
                        
                },
                {
                    "role": "user",
                    "content": f"""
                    ## Problem
                    {element['problem']}
                    ## Attempt
                    {element['attempt']}
                    ## Correct answer
                    {element['solution']}"""

                }
            ],
            max_completion_tokens=1000,
            reasoning_effort="low",
        )
        element["grading"] = completion.choices[0].message.content
        print(f"Question number {Question_Number} \n total token cost: {completion.usage.total_tokens} tokens")

  #Saving Grading Results
    with open(
        Output,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
