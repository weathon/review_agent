# PlugGuard: A Streaming Safeguard for Large Models via Latent Dynamics-Guided Risk Detection

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Large models (LMs) are powerful content generators, yet their open‑ended nature can also introduce potential risks, such as generating harmful or biased content.
Existing guardrails mostly perform post-hoc detection that may expose unsafe content before it is caught, 
and the latency constraints further push them toward lightweight models, limiting detection accuracy.
In this work, we propose PlugGuard, a novel plug-in framework that enables streaming risk detection within the LM generation pipeline. 
PlugGuard leverages intermediate LM hidden states through a Streaming Latent Dynamics Head (SLD), which models the temporal evolution of risk across the generated sequence for more accurate real-time risk detection.
To ensure reliable streaming moderation in real applications, we introduce an Anchored Temporal Consistency (ATC) loss to enforce monotonic harm predictions by embedding a benign-then-harmful temporal prior. Besides, for a rigorous evaluation of streaming guardrails, we also present StreamGuardBench—a model-grounded benchmark featuring on-the-fly responses from each protected model, reflecting real-world streaming scenarios in both text and vision–language tasks.
Across diverse models and datasets, PlugGuard consistently outperforms state-of-the-art post-hoc guardrails and prior plug-in probes (15.61% higher average F1), while using only 20M parameters and adding less than 0.5 ms of per-token latency.
The code and StreamGuardBench are released at **PlugGuard** to facilitate research on streaming guardrails.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents PlugGuard which is a streaming safeguard for content moderation with the goal of improving latency by moderating on latent space. The novelties of the approach lies in two aspects: 1) a streaming latent dynamics head for modeling the temporal evloution of risks, 2) anchored temporal consistency loss to enforce monotonic harm predictions. In addition, this work also introduces a streaming-style content moderation benchmark. The experimental results show that the proposed approach outpeforms multiple baselines on multiple benchmarks. And the proposed approach is also the most efficient one.

### Strengths
1. This work introduces a streaming content moderation benchmark, which doesn't exist before.
2. The proposed approach per the experimental results achieves the best performance in terms of moderation accuracy and efficiency.

### Weaknesses
1. One major concern is how the data is collected, which can impact the fairness of the experiments. The prompts are collected from WildGuard, S-Eval, MMSafetyBench, and FigStep. The model responses are generated based on those prompts for both model training and evaluation. This doesn't seem fair, as the other baselines doesn't train on the same domain of data. This may be why the proposed approach has the best performance, while the other baselines even much larger model doesn't perform so well.
2. The proposed benchmark is model-specific, which limits its application range.

### Questions
See weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents PlugGuard which is a streaming safety moderation framework. PlugGuard is different from post-hoc moderation guardrails as it aims to detect unsafe content during model's decoding process. To do so, authors introduce Streaming Latent Dynamics Head (SLD) and Anchored Temporal Consistency (ATC) loss. The authors also present StreamGuardBench which is a model-grounded benchmark reflecting real-world streaming scenarios in both text and vision–language tasks. Authors perform experiments and showcase effectiveness of PlugGuard against SOTA guardrail mechanisms.

### Strengths
1. The paper studies an important and timely problem.
2. The paper has technical rigor and introduces innovative components (e.g., introduction of Streaming Latent Dynamics Head (SLD) and Anchored Temporal Consistency (ATC) loss)
3. The StreamGuardBench benchmark can be beneficial to the community.
4. The authors performed various experiments and showcases the effectiveness of the approach over SOTA approaches.

### Weaknesses
1. In section 2.1, under Annotation protocol more details are needed for the human evaluations. 
2. StreamGuardBench is created based on existing datasets (S-Eval, WildGuard, MMSafetyBench) so there is a possibility of train-test contamination in the experiments. Some clarity on this could be beneficial in the paper.
3. While authors discuss latency concerns, I am wondering what is the overhead for long benign generations for use-cases where the benign generations dominate harmful ones. A good thing about post-hoc guardrails is that they can be chosen to be used or not used depending on the use-case, so I am wondering what is the trade-off for use-cases with majority benign generations. 
4. In addition, PlugGuard seems more costly cause for each model we need to train it vs for safety classifiers you can train one model and attach it to any model. 
5. There were a few minor grammatical issues: lines 142-143 "thus can enables faithful ..." -> thus can enable faithful ...

### Questions
1. StreamGuardBench is created based on existing datasets (S-Eval, WildGuard, MMSafetyBench) so is there a possibility of train-test contamination in the experiments? 
2. While authors discuss latency concerns, I am wondering what is the overhead for long benign generations for use-cases where the benign generations dominate harmful ones. A good thing about post-hoc guardrails is that they can be chosen to be used or not used depending on the use-case, so I am wondering what is the trade-off for use-cases with majority benign generations.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces PlugGuard, a real-time risk detection framework for large language models during generation. Unlike traditional post-processing methods, PlugGuard evaluates the safety of each token as it is generated, intervening immediately when harmful content is detected. By leveraging intermediate hidden states of the model and introducing a Streaming Latent Dynamics Head (SLD), PlugGuard ensures efficient safety detection with minimal latency. Experimental results show that PlugGuard outperforms existing methods across multiple datasets. The paper also introduces a new evaluation benchmark, StreamGuardBench, and uses Anchored Temporal Consistency (ATC) loss to enhance real-time detection stability. PlugGuard offers a promising approach for safer, real-time content generation in large-scale models.

### Strengths
1. The proposed method is low-cost and more efficient, as demonstrated in Table 8 of the appendix, where PlugGuard adds less than 0.5 milliseconds of latency for most tokens.
2. The introduction of StreamGuardBench, a benchmark specifically designed to evaluate streaming safety protection methods.
3. Experiments validate the effectiveness of the proposed approach.

### Weaknesses
1. PlugGuard still relies on specific datasets, which raises concerns about its generalization capability.
2. PlugGuard depends on the internal states and generation process of the target model, which may limit its applicability across different models.
3. Although the latency per token is very low, current inference models may process a large number of tokens. The accumulation of latency could potentially affect performance in practical applications.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
