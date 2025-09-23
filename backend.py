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
                Provide feedback to help the agent however,   \
                there is a probability {noise} that you add conflicting and random information into the feedback\
                The final feedback should at most be 40 words. \
                An example of a conflicting feedback: go up, maybe dont go up, maybe left?'
                    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

#evaluate current feedback using LLM
def query_evaluation(state:list, mission:str, feedback: str) -> float:
    prompt = f'Evaluate the following feedback:\
                {feedback} \
                Use these criteria: \
                1. Clarity (1 - 5): how detailed is it and is the feedback contradictory?  \
                2. Relevance (1 - 5): given this state: {state} and mission: {mission}, is the feedback relevant?  \
                3. Overall score: +1 (good state) or -1 (bad state) \
                low clarity: go the blue door, actually go to the red door instead \
                high clarity: go up 1 square, then go left 1 square and open the door \
                Respond strictly in this format: \
                    Clarity: <score> \
                    Relevance: <score> \
                    Overall: <+1/-1>'
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    feedback = [] #clarity, relevance, sentiment
    for line in  response.choices[0].message.content.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            feedback.append(int(v.strip().replace("+", "")))

    return (feedback[0] / 5) * ((feedback[1]) / 5) * feedback[2]