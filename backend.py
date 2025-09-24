import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API = os.getenv("OPENAI_API")
client = OpenAI(api_key=OPENAI_API)

#feedback on current state
def query_feedback(mission:str, dir: str, state: list, noise: int) -> str:
    prompt = f'You are providing feedback to help an agent complete the mission: {mission}. The agent is facing {dir} and sees: \n {state}. \
\nProvide feedback to help the agent. There is a probability {noise} that you add conflicting and random information into the feedback. \
The final feedback should at most be 40 words. An example of a conflicting feedback: go up, maybe dont go up, maybe left?'
                    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

#evaluate current feedback using LLM
def query_evaluation(state:list, mission:str, feedback: str) -> float:
    prompt = f'Evaluate the following feedback: \n {feedback} \
\nUse these criteria: \
\n1. Clarity (1 - 5): how detailed is it and is the feedback contradictory? \
\n2. Relevance (1 - 5): is the feedback relevant given this mission: {mission} and state: \n {state} \
\n3. Overall score: +1 (good state) or -1 (bad state) \n \
\n\nlow clarity: go the blue door, actually go to the red door instead \
\nhigh clarity: go up 1 square, then go left 1 square and open the door \
\nRespond strictly in this format: Clarity: <score>, Relevance: <score>, Overall: <+1/-1>'
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    feedback = [] #clarity, relevance, sentiment
    for response in  response.choices[0].message.content.split(", "):
        k, v = response.split(":", 1)
        feedback.append(int(v.strip().replace("+", "")))
    return (feedback[0] / 5) * ((feedback[1]) / 5) * feedback[2]