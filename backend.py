from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
OPENAI_API = os.getenv("OPENAI_API")
client = OpenAI(api_key=OPENAI_API)

#feedback on current state
def query_feedback(mission:str, dir: str, state: list, noise: int) -> str:
    prompt = f'You are providing feedback to help an agent complete the mission: {mission}. \
                The agent is facing {dir} and sees: {state}. \
                You should give helpful and relevant feedback.  \
                However, with probability {noise}%, introduce noise by making the feedback irrelevant, misleading, or vague. \
                Output only one short feedback sentence.'
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

#evaluate current feedback using LLM
def query_evaluation(mission:str, feedback: str) -> str:
    prompt = f'Evaluate the following feedback:\
                {feedback} \
                Use these criteria: \
                1. Clarity (1 - 5) \
                2. Relevance to mission: {mission}(1 - 5) \
                3. Overall score: +1 (good state) or -1 (bad state) \
                Respond strictly in this format: \
                    Clarity: <score> \
                    Relevance: <score> \
                    Overall: <+1/-1>'
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content