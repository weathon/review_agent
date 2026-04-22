# The Challenge of Reliable Vision–Language Model Responses in Driving

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
A reliable driving assistant should provide consistent responses and reasoning based on observed information. In this work, we investigate whether Vision-Language Models (VLMs), when applied as driving assistants, can response consistantly and genuinely understand how present observations shape future outcomes, or whether their outputs merely reflect patterns memorized during training without grounded temporal reasoning. While recent efforts have integrated VLMs into autonomous driving, prior studies typically emphasize scene understanding and instruction generation, implicitly assuming that strong visual interpretation naturally enables consistant future reasoning and thus ensures reliable decision-making, a claim we critically examine.
We focus on two major challenges limiting VLM reliability in this setting: response inconsistency, where minor input perturbations yield different answers or, in some cases, responses degenerate toward near-random guessing, and limited temporal reasoning, in which models fail to reason and align sequential events from current observations, often resulting in incorrect or even contradictory responses. Moreover, we find that models with strong visual understanding do not necessarily perform best on tasks requiring temporal reasoning, indicating a tendency to over-rely on pretrained patterns rather than modeling temporal dynamics.
To address these issues, we adopt existing evaluation methods and introduce FutureVQA, a human-annotated benchmark dataset specifically designed to assess future scene reasoning. In addition, we propose a simple yet effective self-supervised tuning approach with chain-of-thought reasoning that improves both consistency and temporal reasoning without requiring temporal labels.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper probes whether VLMs used as driving assistants truly perform temporally grounded reasoning, finding two reliability failures—response inconsistency under semantics-preserving perturbations and weak temporal reasoning.

In addition, It introduces FutureVQA and an evaluation protocol (self-aligned future descriptions + multi-trial consistency) to test future-scene reasoning over 1–12-second horizons.

### Strengths
* The paper formalizes reliable temporal reasoning with explicit alignment between past-only and future-conditioned predictions and gives concrete measures under semantics-preserving perturbations.
* The paper propose a well-constructed dataset that targets future reasonin. Human-annotated FutureVQA focus on the time-specific prediction with diverse, naturally phrased questions and a multi-trial protocol.

### Weaknesses
* The main concern is the scale and context limitations. The benchmark contains 2.7k human-annotated QA and each input provides only a 5-second history while evaluating up to 12 s, which may underrepresent longer-horizon dynamics and diverse real-world conditions. 
* Evaluation may be judge-biased. Future caption quality is partly scored by a single model-based judge (GPT-4o), and text similarity metrics (BLEU/ROUGE/CIDEr) are used—both may poorly capture safety-critical temporal reasoning and can introduce evaluator bias.

### Questions
The primary concern is the limitation in scale and context. Providing additional clarification here would be helpful.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates VLM reliability in driving, finding models struggle with response inconsistency and temporal reasoning. It introduces FutureVQA, a human-annotated benchmark for future prediction. It also proposes FutureAgent, a self-supervised method that trains the model to predict pseudo-descriptions of future frames, improving temporal consistency.

### Strengths
This paper shifts focus from simple accuracy to the critical issues of reliability and temporal reasoning in driving. The analysis of response inconsistency using option shuffling is a simple and effective diagnostic. The introduction of FutureVQA provides a valuable, human-annotated resource for the field. The paper is clearly written, and the proposed FutureAgent method is an intuitive self-supervised approach that demonstrates improved performance. This work provides a useful framework for evaluating VLM foresight.

### Weaknesses
A key limitation lies in the problem's formulation. The FutureAgent task trains the model to predict the single, recorded future from the dataset. This setup treats the future as a passive, deterministic event. However, for a reliable driving assistant, the future is conditional on the ego-vehicle's own actions (e.g., braking vs. accelerating). The current method trains for passive prediction of what did happen, not for the action-conditional foresight of what might happen given different choices. This overlooks the agent's own influence on the environment.

### Questions
1. The FutureAgent task trains the model to passively predict a single, recorded future. Do the authors agree this is a limitation? How might the proposed method be extended to learn action-conditional future reasoning (e.g., "What will happen if I brake now?" vs. "...if I continue at this speed?")?
2. Table 1 shows that FutureAgent reduces the accuracy drop (the "S-M" column) compared to its baseline. Could you elaborate on why you believe this specific self-supervised task improves this measure of consistency?
3. The exponential decay weighting prioritizes short-term predictions. Have you experimented with other weighting functions, such as one that gives more weight to challenging long-term predictions?
4. How do you interpret the performance of FutureAgent compared to models explicitly trained on video dataset?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the reliability of VLMs when applied as driving assistants, focusing on their ability to perform temporal reasoning and generate consistent responses. It identifies two critical limitations of current VLMs: response inconsistency and limited temporal reasoning. To address these issues, this paper introduce FutureVQA, a human-annotated benchmark dataset designed to evaluate future scene reasoning capabilities of VLMs, and propose a self-supervised tuning approach (FutureAgent) that enhances temporal consistency and reasoning without requiring explicit temporal labels. Experiments on multiple open-source and commercial VLMs demonstrate that the proposed method effectively improves response consistency and future scene prediction performance.

### Strengths
This paper analyzes and highlights key reliability issues (response inconsistency and poor temporal reasoning) of VLMs in safety-critical driving scenarios.

FutureVQA provides a valuable human-annotated dataset tailored for evaluating future scene reasoning in driving, addressing the gap in existing benchmarks that lack focus on temporal dynamics.

### Weaknesses
Inference speed and suitability for real-time driving applications are not discussed

The cite format is incorrect, it seems the authors used 'cite' rather than 'citep' required in the template.

FutureVQA focuses on basic future scene questions; it does not fully cover complex driving scenarios (e.g., emergency situations, multi-agent interactions), raising concerns about the benchmark’s ecological validity.

This paper identifies response inconsistency but does not deeply analyze its underlying causes (e.g., model architecture, training data biases, or prompt sensitivity mechanisms), limiting targeted improvements.

Whether there are other specialized temporal reasoning models, it is hard to assess the technical contribution and relative advantage of this paper in the driving domain.

### Questions
Have you explored why VLMs exhibit response inconsistency (e.g., internal randomness, prompt phrasing sensitivity, or knowledge gaps)? How can these specific causes be mitigated beyond the proposed self-supervised tuning?

How does the inference speed of FutureAgent? Can this method be applied for real-time deployment in autonomous driving systems?

Can FutureVQA include more complex driving scenarios (e.g., adverse weather, traffic accidents) and diverse question types (e.g., causal reasoning about collisions)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper systematically evaluates the temporal reasoning and future scene prediction capabilities of VLMs in the context of autonomous driving. The authors introduce the FutureVQA benchmark, a challenging human-annotated dataset specifically designed for future scene understanding. They further propose a self-supervised fine-tuning approach that improves models’ temporal consistency and reasoning ability without requiring explicit temporal annotations. Experimental results demonstrate the limitations of existing VLMs and show that the proposed method provides significant gains in both accuracy and temporal alignment.

### Strengths
1. The paper addresses an important and underexplored problem of reliable temporal reasoning for VLMs in safety-critical driving scenarios.
2. The introduction of the FutureVQA benchmark fills a gap in the evaluation of future scene understanding, featuring diverse, human-annotated, and time-specific questions.
3. The proposed self-supervised fine-tuning method is practical, annotation-efficient, and yields clear improvements without requiring additional temporal data labels.

### Weaknesses
1. The current experiments are conducted on general-purpose VLMs and do not include domain-specific models pre-trained for autonomous driving. Since the proposed self-supervised fine-tuning method relies on the quality of pseudo-labels generated by the baseline model, it would be interesting to see whether using models with driving-specific knowledge would lead to different performance improvements.

2. The evaluations mainly focus on quantitative metrics and lack more intuitive case studies. What are the concrete improvements before and after applying the self-supervised fine-tuning approach? It would be helpful to include representative qualitative examples to support the quantitative results, which could provide clearer evidence of the method’s effectiveness.

### Questions
1. The current experiments are conducted on general-purpose VLMs and do not include domain-specific models pre-trained for autonomous driving. Since the proposed self-supervised fine-tuning method relies on the quality of pseudo-labels generated by the baseline model, it would be interesting to see whether using models with driving-specific knowledge would lead to different performance improvements.

2. The evaluations mainly focus on quantitative metrics and lack more intuitive case studies. What are the concrete improvements before and after applying the self-supervised fine-tuning approach? It would be helpful to include representative qualitative examples to support the quantitative results, which could provide clearer evidence of the method’s effectiveness.

### Soundness
3

### Presentation
3

### Contribution
2
