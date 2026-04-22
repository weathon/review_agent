# Scaling Open-Ended Reasoning to Predict the Future

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
While language models now show remarkable capabilities on fully specified exam-style problems, most real-world decisions involve reasoning under uncertainty. In this work, we train language models to make predictions on open-ended questions about the future. To scale up training data, we continually synthesise novel forecasting questions from global events reported in daily news, using a fully automated, careful curation recipe. We train the Qwen3 thinking models on our dataset, OpenForesight.  To prevent leakage of future information during training and evaluation, we use an offline news corpus, both for data generation and retrieval in our forecasting system. Guided by a small validation set, we show the benefits of retrieval, a supervised finetuning phase, and an improved reward function for reinforcement learning (RL). Once we obtain our final forecasting system, we perform held-out testing between May to August 2025. Our specialized model, OpenForecaster-8B, matches much larger proprietary models, with our training improving the accuracy, calibration, and consistency of predictions. We find calibration improvements from forecasting training generalize across popular benchmarks. We will open-source our models, code, and data to make LLM based forecasting research broadly accessible.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a training framework for open-ended future event forecasting using LLMs. However, despite this ambition, the degree to which the system advances open-ended reasoning for future prediction is not fully demonstrated by the current experiments.

### Strengths
- The motivation is clear and timely, targeting the critical but underexplored problem of evaluating models’ ability to forecast real future events.
- The paper contributes a valuable news-based dataset, enabling scalable research on forecasting with LLMs.
- The authors explicitly mitigate data leakage by using offline news, detecting answer leakage in prompts, and rewriting or filtering problematic questions.

### Weaknesses
- The SFT + RL training configuration is presented as essential. Still, the experiments do not isolate their respective contributions to reasoning improvements. 
- While the task is framed as “open-ended,” most answers are short named entities, and evaluation relies on semantic matching rather than deeper causal understanding. 
- Long-term predictive claims depend only on proxy consistency metrics, lacking results on resolved events that truly challenge foresight over extended horizons.

### Questions
1. Can the authors provide more fine-grained evidence of reasoning improvement, such as causal inference quality, evidence attribution, or handling of low-signal future scenarios, rather than aggregate accuracy and calibration alone?

2. What distinct roles do SFT and RL play in improving forecasting behavior? Ablations or qualitative analyses would help clarify whether models genuinely anticipate future outcomes rather than react to available information.

3. How do the authors justify the “open-ended” nature of the task, given that answers are primarily short entity names? Could evaluation incorporate more structurally complex future events?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper targets open-ended, probability-bearing forecasting (free-form short answer + confidence) and trains/evaluates language models for future events under strict anti-leakage constraints. The authors propose an automated pipeline, OpenForesight, which continuously synthesizes forecasting questions with explicit resolution dates and evaluation criteria from time-stamped offline news; a multi-stage filtering process yields a high-quality training set.

### Strengths
Practical problem: Open-ended prediction with probability output is of clear real-world value.

Scalable data synthesis: From hundreds of thousands of articles to ~60k high-precision samples via multi-stage filtering (validity → best-candidate selection → leakage cleaning → answer-type control), maintaining quality and breadth.

Training recipe that balances goals: SFT warm-start expands solution diversity/ceiling; the joint reward avoids the “accuracy-only hurts calibration / calibration-only suppresses exploration” trade-off.

Time-aware retrieval design (only pre-resolution evidence).

Comprehensive evaluation coverage.

Clear figures and pipeline presentation.

### Weaknesses
Compute/cost transparency is limited: missing token counts, steps, GPU-hours for SFT/RL, and end-to-end latency (including retrieval).

No human/market baselines: comparisons to aggregated human forecasters or prediction-market probabilities are absent.

Source/language bias risk: test set from five English outlets—please assess topic/region balance and consider multilingual evaluation.

Long-horizon behavior unclear: what happens for resolutions ≥6/12 months?

Safety: unclear whether sensitive domains (policy/health) receive safety review or filtering.

Open-sourcing: currently a promise without links to released artifacts.

Error analysis: add failure-mode attribution and difficulty stratification of the dataset.

Live evaluation: while I appreciate the anti-leakage design, live evaluation [1] would better reflect deployment and compare with stronger model/agent baselines with no-leak settings.

[1] FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction

### Questions
Please check weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a system for training language models to be better at open-ended forecasting. The authors first create OpenForesight, a large-scale training dataset of over 60,000 open-ended questions. This dataset is generated automatically from a static, offline news corpus to prevent data leakage, and then put through a careful, multi-stage curation pipeline to ensure data quality and remove noisy signals. The authors then train Qwen3 models (4B and 8B) using a combination of retrieval, a supervised fine-tuning (SFT) warm-up, and reinforcement learning (RL). A key part of their method is a novel RL reward function that combines both accuracy and the Brier score. The paper demonstrates through ablations that both their data curation recipe and their specific reward function are essential for success. Their final 8B model, OpenForecaster, is shown to match or exceed the performance (specifically in Brier score) of much larger proprietary models on a held-out test set of future events.

### Strengths
The paper tackles a valuable and difficult problem: scalable, open-ended forecasting.

It presents a complete, end-to-end system, meticulously validating each component choice, from data curation to the final RL reward.

The data generation and curation pipeline is fully-automated and scalable, and it thoughtfully designed to avoid common pitfalls like data leakage (by using an offline corpus) and self-preference bias (by using different models for generation and filtering).

The ablation studies are a significant strength. The paper empirically proves that its specific data curation recipe is necessary and that its novel 'Accuracy + Brier' reward function is superior to simpler, more common rewards .

The evaluation is rigorous. It uses a held-out test set derived from different news sources than the training set, testing for generalization.

The final result is interesting a specialized, small, open-weight model is shown to be competitive with much larger, general-purpose proprietary models on this specific task.

The authors also demonstrate that the calibration improvements from their training generalize to other standard downstream benchmarks, which strengthens their claims.

Overall I like this paper's focus on forecasting task as I am interested in this shift from fix-known knowledge evaluation to forecasting. I think this paper is a timely paper.

### Weaknesses
The paper's novelty is primarily in the systematic combination of existing techniques which makes it hard to pinpoint the exact contribution

The experiments on specialized models are confined to a single model family, Qwen3. This limited scope makes it unclear how dependent the results are on the base Qwen3 model architecture and whether the pipeline and 'Accuracy + Brier' reward would be equally effective if applied to other popular open-weight models. How dependent are the results on the base Qwen3 model architecture? Would the pipeline and 'Accuracy + Brier' reward be equally effective if applied to a Llama or Mistral-family model?

The paper's methodology makes it difficult to disentangle the performance contributions of its two main training stages. It is unclear how much of the final boost comes from the SFT distillation of Grok-3-mini's reasoning versus the new forecasting skill learned during the RL phase on the OpenForesight dataset. Do you have any experiments/results on this?

The reliance on LLMs for the entire curation pipeline might introduce its own set of systematic, unmeasured biases. It is possible that this automated pipeline is systematically filtering out harder, more ambiguous, or more complex forecasting questions, thereby creating a "clean" but simplified version of the real-world task. Basically, the data curation recipe is shown to be effective, but is it possible that this automated pipeline is systematically filtering out harder, more ambiguous, or more complex forecasting questions, thereby creating a "clean" but simplified version of the real-world task?

The training and evaluation setup may obscure the true generalizability of the model. The training set is heavily dominated by two news sources, while the test set uses five completely different ones. This domain mismatch isn't analyzed, making it hard to understand what generalizable skills were learned. This is compounded by the dataset's inherent biases, such as being sourced only from news and the potential for "late reporting" bias.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces OpenForesight, a dataset of 60,000 forecasting questions automatically generated from global news articles using the CommonCrawl News corpus and OpenForecaster models, which are Qwen3-4B and 8B trained using SFT on Grok-3-mini reasoning traces, followed by GRPO using accuracy + Brier score as reward. The OpenForecaster 8B model significantly outperforms its base model and matches the Brier score of much larger models, such as gpt-oss-120B, on a held-out test set. The paper also demonstrates that the calibration improvements generalize to standard benchmarks, such as MMLU-Pro and GPQA.

### Strengths
1. OpenForesight generation is a scalable, automated pipeline for generating open-ended forecasting questions.
2. The ablation studies provide a thorough validation of design choices like filtering, reward, and retrieval.

### Weaknesses
1. The data and models can be heavily skewed towards topics news media covers. Might perform better on domains like politics, etc, but not on long-term cultural shifts, scientific breakthroughs, etc. 
2. The filtering pipeline removes ~90% of generated questions. This makes the generation process very expensive and requires further analysis on why the generation model fails so frequently. Will in-context learning help?
3. The paper doesn't justify why numeric answers are filtered out beyond avoiding "vague" responses.
4. The paper shows that training improves forecasting, but provides limited insight into reasoning traces after GRPO and how they hey evolve during training.

### Questions
1. Have you considered improving the initial generation prompt or using few-shot examples to increase the acceptance rate of data generation?
2. Do you have any measure of question difficulty? 
3. What are the most common failure modes? Are there systematic biases (e.g., toward countries, types of events, recency bias)?
4. Have you tested the system on live forecasting platforms like Metaculus?

### Soundness
3

### Presentation
3

### Contribution
3
