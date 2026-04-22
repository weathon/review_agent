# Do Stop Me Now: Detecting Boilerplate Responses with a Single Iteration

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 0, 4, 2

## Abstract
Large Language Models (LLMs) often expend significant computational resources generating boilerplate responses, such as refusals, simple acknowledgements and casual greetings, which adds unnecessary cost and latency. To address this inefficiency, we propose a simple yet highly effective method for detecting such responses after only a single generation step. We demonstrate that the log-probability distribution of the first generated token serves as a powerful signal for classifying the nature of the entire subsequent response. Our experiments, conducted across a diverse range of small, large, and reasoning-specialized models, show that the first-token log-probability vectors form distinctly separable clusters for different response types. Using a lightweight k-NN classifier, we achieve high accuracy in predicting whether a response will be a substantive answer or a form of boilerplate response, including user-specified refusals. The primary implication is a practical, computationally trivial technique, optimizing LLM inference by enabling early termination or redirection to a smaller model, thereby yielding significant savings in computational cost. This work presents a direct path toward more efficient and sustainable LLM deployment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a simple approach to predict the response type of a large language model (LLM) using the log-probability of the first generated token. The goal is to prevent LLMs from producing boilerplate or low-information responses, thereby saving computational resources.

### Strengths
1. The problem of avoiding boilerplate responses is a valid and practically relevant one, as it relates to the efficiency and quality of LLM outputs.
2. The authors release a dataset, which may be useful for further exploration of response-type prediction tasks.

### Weaknesses
1. **Presentation quality:** The paper’s presentation requires substantial improvement. There are noticeable formatting issues (e.g., excessive white space in Figures 2–4), and the absence of a main figure illustrating the overall method makes it difficult for readers to grasp the approach at a glance. Improving the paper’s structure and visual clarity would make it more accessible.
2. **Experimental design and analysis:** The experiments are insufficiently detailed and lack comprehensive analysis. The evaluation setup is minimal, with limited discussion of baselines, ablation studies, or error cases. As it stands, the submission reads more like a preliminary project report than a mature research contribution.
3. **Limited novelty and depth:** The proposed method is conceptually simple and appears to be a straightforward heuristic. Without stronger theoretical grounding, comparative baselines, or empirical justification, the contribution seems too limited for acceptance at a top-tier venue like ICLR.

Overall, the work feels incomplete and underdeveloped. The core idea is relevant but not explored in sufficient technical or empirical depth to warrant publication at this stage.

### Questions
1. How does the proposed method compare with simple baselines such as classifying response types directly from the first few generated tokens, rather than relying solely on its log-probability?
2. The dataset used contains prompts with widely varying characteristics and frequencies. Could the authors justify why this dataset serves as a valid benchmark, and how it reflects real-world distributions of LLM interactions?
3. Have the authors considered extending the method to use information from the first few tokens instead of just the first one? This might provide a more stable and reliable signal for response-type prediction.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes using first-token log-probability distributions to classify LLM responses as either substantive ("Chat") or boilerplate (refusals, greetings, acknowledgements), with the goal of early termination or routing to smaller models to save costs.

### Strengths
* Addresses a practical problem of reducing inference costs for predictable responses

### Weaknesses
* **Fundamentally unsound task definition**: The paper groups refusals, greetings, and acknowledgements together as "boilerplate responses" that can be handled uniformly. This is deeply problematic. Refusals are safety-critical responses that embody the model's alignment training, not "boilerplate waste" to be optimized away. Replacing careful safety mechanisms with a k-NN classifier trained on 3k synthetic examples is inappropriate and potentially dangerous. The proposed solution of routing refusals to smaller models requires futher justification - do you really want safety decisions made by weaker models with degraded alignment? The paper doesn't explain what you would do after detecting a refusal.
* **Why use machine learning for trivial patterns?** For simple responses like "Hello", "Hi", or "You're welcome", why not just use string matching or simple heuristics? These patterns are deterministic and don't require log-probability analysis or k-NN classification. Using complex ML methods for trivial pattern matching is over-engineering.
* **Severe data imbalance makes results unreliable** The dataset has only ~30 "Hello" examples and ~250 "Thanks" examples out of ~3k total samples. The "Hello" class represents only 1.4% of the dataset, yet the paper reports 100% F1 scores on it. These numbers are statistically meaningless with so few samples. The reported 99%+ accuracies may overlooked the minority label.

### Questions
see weakness above

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the computational inefficiency of LLMs generating boilerplate responses, such as refusals or greetings. The proposed method is simple and effective: classifying the entire response type by analyzing the log-probability distribution of only the first generated token using a k-NN classifier. Experiments across a few models, demonstrate that these first-token vectors form distinct clusters for categories like 'Chat', 'Refusal', 'Thanks', and 'Hello', achieving high classification accuracy.

### Strengths
The proposed method is extremely simple and computationally lightweight, as it only requires a single forward pass to get the first token's probabilities and a fast k-NN lookup.

It uses the entire log-probability vector with a k-NN classifier rather than a manually selected subset of tokens, and extends this classification from just "Refusal" to also include "Thanks" and "Hello"

### Weaknesses
The primary weakness of this paper is its limited novelty and contribution. The core idea that the first token's probabilities can predict the subsequent response, especially for refusals, is not new. The authors themselves cite related work (Arditi et al., 2024) which already derived a "refusal metric" by summing probabilities of "refusal tokens" at the first token position.

The reliance on a k-NN classifier is sensitive to the training data. It's unclear how this approach would generalize to new, unseen types of boilerplate or how it would cope with model updates, which could shift the entire log-probability space and render the saved vectors obsolete.

### Questions
N.A.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles the compute waste caused by LLMs producing boilerplate text. The key idea is strikingly simple: use only the log-probability vector of the very first generated token to predict the eventual response type, then early-stop or re-route generation to save cost and latency. The authors show that common response types (Refusal/Thanks/Hello/Chat) form separable clusters in the first-token log-prob space, and that a lightweight k-NN classifier suffices to achieve high accuracy across several model scenarios.

### Strengths
- Direct Approach. Reading a single first-token log-probability vector and classifying with k-NN delivers useful discrimination among boilerplate types.
- Clear Motivation. Framing the work around early stopping/routing aligns with real-world needs.
- Experimental Transparency. The paper describes data construction, fixed k in k-NN, and cross-validation, which helps readers reproduce the general setup.

### Weaknesses
- Adversarial Mixed Intents and False Positive. The dataset design around hello, refusal and thanks is reasonable, but it overlooks adversarial or mixed-intent prompts that can blur class boundaries. For example, a user input like “Hello, nice to meet you. How’s the weather today?” will likely elicit a first token such as “Hello,” followed only then by substantive content about weather. A first-token classifier is severely stressed in such cases, and the paper does not analyze this reliability gap. Similarly, larger models sometimes begin with a refusal and then pivot to supportive guidance (e.g., for self-harm queries). Such trajectories can produce ambiguous logprob signals that the method may misread.

- Missing Comparisons. The Related Work section lists adjacent lines of research, but the paper offers no head-to-head comparisons.

- Efficiency in Practice. Beyond inference, efficiency trade-offs are underexplored. Because tokenizers and vocabularies differ across model families, each model (or family) needs its own k-NN built on its own token space, which raises training and maintenance cost. Moreover, the targeted classes (refusals, greetings, thanks) are typically very short replies; in realistic deployments, the end-to-end cost of extracting first-token logprobs and running the k-NN router can approach the cost of just letting the model emit the short reply. Without concrete latency numbers, it’s unclear that the method consistently yields net wins.

### Questions
Please refer to Weaknesses for points requiring further clarification.

### Soundness
2

### Presentation
3

### Contribution
2
