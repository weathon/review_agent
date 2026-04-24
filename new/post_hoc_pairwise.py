import asyncio
import os
from claude_agent_sdk import query, ClaudeAgentOptions

prompt = """You will be given two peer reviews, each for a different paper. Your task is to determine which of the two papers is of higher quality based on the content of the reviews.

Important:
- You are judging the papers, not the reviews. Do not prefer a paper simply because its review is longer, better-written, or more detailed.
- Base your judgment on what the reviews reveal about each paper's contributions, soundness, and weaknesses.
- Ignore stylistic differences between the two reviews.

Provide your answer as:
- explanation (str): a brief justification for your decision
- answer (str): the reviewer's name, it has to be exact match of the full name
"""

from openai import OpenAI
import dotenv
import random
from pydantic import BaseModel

dotenv.load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ["OPENROUTER_API_KEY"]
)

import numpy as np
from scipy.optimize import minimize_scalar

def estimate_score(anchor_scores: list[float], wins: list[int]) -> float:
    """
    anchor_scores: anchor的已知分数
    wins: 对每个anchor的结果，1表示target赢（target更好），0表示target输
    返回target的估计分数
    """
    anchors = np.array(anchor_scores)
    outcomes = np.array(wins)

    def neg_log_likelihood(target_score):
        # Bradley-Terry: P(target beats anchor) = sigmoid(target - anchor)
        logits = target_score - anchors
        log_p_win = -np.logaddexp(0, -logits)
        log_p_loss = -np.logaddexp(0, logits)
        return -np.sum(outcomes * log_p_win + (1 - outcomes) * log_p_loss)

    lo, hi = min(anchors) - 5, max(anchors) + 5
    result = minimize_scalar(neg_log_likelihood, bounds=(lo, hi), method='bounded')
    return result.x


class ExtractedReview(BaseModel):
    review: str
    anchors: list[str]

def parse(review):
    response = client.chat.completions.parse(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": "You will be given a review, extract information and remove 'removed points' and calibration logic and scoring, keep only Summary, Strengths, and Weaknesses. Normalize the tone to be more neutral and concise. Out put the normalized review and also a list of filenames (no path no extension, like DvU8ijSn1p) for the anchor used. If no anchor is used, output an empty list."
            },
            {
                "role": "user",
                "content": review
            }
            ],
        response_format=ExtractedReview
    )
    response = response.choices[0].message.parsed
    review = response.review.strip()
    anchors = response.anchors
    if len(anchors) == 0:
        raise ValueError("No anchor found in the review.")
    return review, anchors


class Res(BaseModel):
    reasoning: str
    answer: str

import json
def compare_reviews(reviewer1_string, reviewer2_string):

    reviewer1_name = "".join(random.choices("QWERTYUIOPPASDFGHJKLZXCVBNM1234567890", k=5))
    reviewer2_name = "".join(random.choices("QWERTYUIOPPASDFGHJKLZXCVBNM1234567890", k=5))

    user_message = f"""Here are the two reviews:
    {reviewer1_name}
    {reviewer1_string}

    {reviewer2_name}
    {reviewer2_string}"""

    # First API call with reasoning
    response = client.chat.completions.parse(
        model="gpt-5.4",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": user_message
            }
            ],
        extra_body={"reasoning": {"enabled": True}},
        response_format=Res
    )

    response = response.choices[0].message.parsed
    answer = response.answer.strip()
    if answer == reviewer1_name:
        return 1
    elif answer == reviewer2_name:
        return 0
    else:
        print(f"Unexpected answer: {answer}")
    
import pickle
with open("/home/wg25r/review_agent/new/human_review_score_index.pkl", "rb") as f:
    human_review_score_index = pickle.load(f)

from concurrent.futures import ThreadPoolExecutor
import pandas as pd

rating = pd.read_csv("../iclr2025/ratings.csv")

reviews = os.listdir("bench_reviews")
with open("pair_wise.csv", "w") as f:
    f.write("paper_id,gt_score,pred_score\n")

for review_name in reviews:
    print(review_name)
    if review_name.split(".")[0] not in rating["paper_id"].values:
        continue

    with open(f"bench_reviews/{review_name}", "r") as f:
        try:
            review, anchors = parse(f.read())
        except ValueError:
            print(f"Skipping {review_name} because no anchor was found.")
            continue

    if not all(os.path.exists(f"../human_reviews/{anchor}.md") for anchor in anchors):
        print(f"Skipping {review_name} because not all anchors have corresponding human reviews.")
        continue

    print(anchors)
    reviewer1_string = review
    with ThreadPoolExecutor(max_workers=len(anchors)) as executor:
        futures = []
        scores = []
        for anchor in anchors:
            with open(f"../human_reviews/{anchor}.md", "r") as f:
                review_content = f.read().split("## Human Reviews")[1]
            scores.append(human_review_score_index[f"{anchor}.md"])
            futures.append(executor.submit(compare_reviews, reviewer1_string, review_content))
        
    pred_score = estimate_score(scores, [f.result() for f in futures])
    gt_score = rating[rating["paper_id"] == review_name.split(".")[0]]["avg_score"].item()
    print(f"Predicted score: {pred_score}, Ground truth score: {gt_score}")
    with open("pair_wise.csv", "a") as f:
        f.write(f"{review_name.split('.')[0]},{gt_score},{pred_score}\n")
