# What makes the preferred thinking direction for LLM in Multi-choice Questions?

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Language models usually use left-to-right (L2R) autoregressive factorization.
However, L2R factorization may not always be the best inductive bias.
Therefore, we investigate whether alternative factorizations of the text distribution could be beneficial in some tasks.
We investigate right-to-left (R2L) training as a compelling alternative, focusing on multiple-choice questions (MCQs) as a test bed for knowledge extraction and reasoning. Through extensive experiments across various model sizes (2B-8B parameters) and training datasets, we find that L2R is not always preferred over R2L models on MCQ benchmarks, especially on logical reasoning, commonsense understanding, and truthfulness assessment tasks. Our analysis reveals that this performance difference may be fundamentally linked to multiple factors including calibration, computability, and directional conditional entropy.
We ablate the impact of these factors through controlled simulation studies using arithmetic tasks, where the impacting factors can be better disentangled. 
Our work demonstrates that exploring alternative factorizations of the text distribution can lead to improvements in LLM capabilities and provides theoretical insights into optimal factorization towards approximating human language distribution, and when each reasoning order might be more advantageous.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates alternative autoregressive factorizations for large language models (LLMs), focusing on right-to-left (R2L) modeling as a counterpart to the conventional left-to-right (L2R) approach. Using multiple-choice question (MCQ) benchmarks as a controlled setting for evaluating reasoning and knowledge extraction, the authors train R2L models (2B–8B parameters) from scratch with 350B fineweb-EDU dataset and compare their performance with L2R counterparts across various tasks, including logical reasoning, commonsense understanding, and truthfulness assessment.
The authors find that R2L models can outperform L2R ones in certain reasoning-oriented MCQ tasks, particularly TruthfulQA. They further hypothesize three potential contributing factors—Calibration, Computability, and Conditional Entropy (3C)—and conduct ablation and simulation studies (e.g., 4-digit multiplication experiments) to analyze why different reasoning directions might be preferred under different conditions.

### Strengths
• Interesting question and Model orientation.
The paper revisits a fundamental modeling assumption (L2R factorization) and provides a systematic exploration of R2L modeling, which is conceptually simple but rarely studied at large scale.

• Well-designed pretraining setup. 
The authors pretrain R2L models of substantial scale (2B–8B) for a fair comparison, extending beyond prior small-scale or synthetic studies. This large-scale empirical effort is valuable.

• Clear motivation from reasoning evaluation.
Using MCQs as a reasoning testbed is well-motivated, as the task structure naturally fits the “reverse thinking” paradigm where the model can evaluate ( P(q|a) ).

• Interesting analytical framing (3C hypotheses).
The “Calibration–Computability–Conditional Entropy” guessing provides an interpretable structure for analyzing why certain directions might be more effective, helping connect intuitive reasoning behavior to probabilistic modeling.

• Observation on truthfulness tasks.
The link between reverse thinking and improved truthfulness assessment is intriguing and may open new perspectives for evaluating model honesty or factuality.

### Weaknesses
• Lack of strong causal evidence.
While the paper proposes three hypotheses (3C), the experiments results do not show strong evidence to support or reject their hypotheses.

• Insufficient analysis of the scoring paradigm ( s_i ).
Much of the MCQ performance difference seems to originate from how ( s_i ) (the relevance score) is defined—e.g., whether ( s_i = \log p(a_i|q) ) or ( s_i = \log p(q|a_i) ). However, this crucial aspect is under-discussed, and the forward/reverse paradigms are not symmetrically mirrored.

• Notation and explanations are sometimes unclear.
The definitions of ( p_{R2L}(x_t|x_{>t}) ) and its relationship to the MCQ scoring paradigms are not clearly stated. It is sometimes ambiguous whether “reverse thinking” means token-level reversal or question–answer reversal.

• Conditional entropy analysis is underdeveloped.
The Monte Carlo estimation of conditional entropy uses a single-sample rollout and yields mixed results (e.g., Table 3), which weakens the claim that lower entropy strongly correlates with better reasoning direction.

• Limited generalization beyond MCQs.
The improvements mainly appear on TruthfulQA, with mixed results elsewhere. The scope of benefit from R2L modeling remains narrow and may not generalize to open-ended or non-MCQ tasks. Experiments around some curated toy case or other datasets have similar characteristics with TruthfulQA is needed to strengthen their findings.

• Presentation issues.
Some sections (e.g., R2L(m,n) vs R2L(m)) are too brief and ambiguous. Mathematical notation and factorization explanations could be more explicit to help readers reproduce or interpret the findings.

### Questions
• Could you clarify whether R2L modeling reverses token order during both training and inference, or only during the scoring phase for MCQs?

• Why not mirror the “reverse thinking” scoring formulation for L2R (i.e., include ( P(q|a) ) under an L2R prior) to control for the prior effect and ensure a fair comparison?

• In the conditional entropy analysis, did you measure entropy over token sequences or aggregated answer-level probabilities? Could multi-sample rollouts yield more stable results?

• Can the authors elaborate on how the 4-digit multiplication simulation maps to real-world reasoning tasks? What insight does it provide for scaling to larger models?

• Is the tokenizer or positional embedding also reversed for R2L models, or only the data order?

• Since R2L training leads to 2–2.5% higher loss, could this gap be due to optimization dynamics rather than inherent asymmetry?

• How might R2L modeling affect efficiency or applicability in open-ended generation tasks where outputs must be human-readable?

### Soundness
3

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
This paper investigates whether the standard left-to-right (L2R) autoregressive factorization is always the optimal choice for language models, particularly for knowledge extraction and reasoning tasks. The authors specifically explore a right-to-left (R2L) alternative in the context of multiple-choice questions (MCQs). They compare the standard "forward thinking" approach, where an L2R model scores the probability of an answer given the question ($p(a_i|q)$), with a "reverse thinking" method that uses an R2L-trained model to score the probability of the question given the answer ($p(q|a_i)$). The authors find that R2L models, despite incurring a slightly higher pretraining loss, outperform their L2R counterparts on 4 benchmarks (out of 11)

The paper's primary contribution is a framework for understanding these directional preferences, proposing three factors: calibration, computability, and conditional entropy. Authors argue that reverse thinking inherently mitigates calibration issues like "surface form competition" that can affect the forward-thinking approach. Their hypothesis, is that the reasoning direction with the lower conditional entropy for a given task distribution will be learned more effectively and yield better performance. This theory is supported by empirically linking lower conditional entropy to higher accuracy on the evaluated benchmarks. The authors validate this concept in a controlled arithmetic simulation designed to isolate the conditional entropy factor from calibration effects.

### Strengths
This paper presents a surprising and valuable empirical result, finding that right-to-left (R2L) models can outperform standard left-to-right (L2R) factorization on several important reasoning and commonsense benchmarks. This observation challenges a fundamental and often unstated assumption in language modeling.

The work's primary strength lies in its principled and thorough investigation of this phenomenon. Instead of merely reporting the empirical result, the authors propose a clear theoretical framework built on three potential factors: calibration, computability, and conditional entropy. The hypothesis linking the preferred reasoning direction to the direction of lower conditional entropy is particularly insightful. This claim is impressively substantiated through a clean and well-designed controlled simulation using arithmetic tasks. This simulation successfully isolates the proposed factors, lending strong support to the authors' analysis and providing a robust foundation for this new line of inquiry.

### Weaknesses
While the paper introduces a compelling hypothesis, its primary weakness lies in the empirical validation of its central claim regarding conditional entropy. The authors acknowledge that estimating the conditional entropy is intractable and resort to a single-sample Monte Carlo rollout as a proxy. This is an extremely high-variance, and likely unreliable, estimator. This measurement weakness may be why the paper's own results show that two of the four tasks where R2L wins (OpenbookQA and CommonsenseQA) are "outliers" that *contradict* the main hypothesis. The paper dismisses this contradiction by referencing "other confounding factors," which weakens the explanatory power of the proposed framework.

Furthermore, the main experiment confounds two distinct variables: the model's factorization direction (L2R vs. R2L) and the scoring method ("forward" $p(a_i|q)$ vs. "reverse" $p(q|a_i)$). The paper compares the standard L2R approach against a specific R2L heuristic (paradigm 3) that was chosen empirically over other Bayesian-derived options. It is plausible that the observed performance gains are an artifact of the "auto-normalizing" properties of the $p(q|a_i)$ scoring method rather than a benefit of the R2L pretraining itself. A crucial missing ablation would be to evaluate the R2L model using a "forward" $p(a_i|q)$ objective to properly isolate the effect of the pretraining direction.

Finally, while the controlled arithmetic simulation is a clean method for separating computability and entropy, its external validity is not well established. The simulation clearly demonstrates that models excel at the low-entropy, many-to-one direction (multiplication). However, the paper fails to draw a convincing link as to *why* real-world language tasks like TruthfulQA or HellaSwag should map onto this "many-to-one" or "one-to-many" structure. Without this connection, the simulation's findings remain isolated from the main linguistic claims.

### Questions
1.  The paper's main hypothesis relies on the conditional entropy of the task distribution, but its estimation is a significant challenge. The authors use a single-sample Monte Carlo rollout as a proxy, which is noted to be a high-variance estimator. This weak estimation may be the reason why the paper's own results show that two of the four tasks where R2L performs better (OpenbookQA and CommonsenseQA) are described as "outliers" that *contradict* the hypothesis. Could the authors provide a more robust estimation of the conditional entropy, perhaps using more samples? Alternatively, could they offer a more detailed analysis of these outliers beyond "other confounding factors"? Clarifying this point is critical, as these outliers currently undermine the main theoretical claim.

2.  The main experiment compares an L2R-trained model using "forward thinking" ($p(a_i|q)$) against an R2L-trained model using "reverse thinking" ($p(q|a_i)$). This experimental design confounds two separate variables: the model's pretraining direction (L2R vs. R2L) and the scoring method (forward vs. reverse). The observed gains could simply be an artifact of the reverse scoring method, which, as the authors note, cleverly mitigates "surface form competition". To isolate the benefit of the R2L *pretraining*, a crucial ablation is missing. Could the authors evaluate their R2L-trained model using the standard *forward* scoring method ($p(a_i|q)$)? This result would reveal whether the R2L pretraining provides any independent benefit on these tasks.

3.  The controlled arithmetic simulation is a clean and insightful part of the paper. It clearly separates computability from conditional entropy in a many-to-one (multiplication) versus one-to-many (factorization) setting. However, the paper does not establish the external validity of this simulation for its linguistic claims. Could the authors elaborate on *why* a real-world task like TruthfulQA should be viewed as an analogous "one-to-many" problem (low reverse entropy) or why HellaSwag should be seen as "many-to-one" (low forward entropy)? A more explicit justification of this mapping is needed to connect the simulation's findings back to the main claims about language.

4.  The study finds that R2L pretraining consistently results in a higher training loss compared to L2R pretraining on the same data. The evaluation is also restricted to MCQ tasks. This raises a question about the trade-offs. Does this more difficult R2L pretraining harm the model's general-purpose capabilities? For example, how does the R2L-trained model perform on standard L2R *generative* tasks? Understanding if the R2L model is a specialized expert or if it retains robust generative abilities is important for assessing the practical implications of this work.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper compares left-to-right (L2R) vs. right-to-left (R2L) autoregressive factorizations for multiple-choice questions (MCQs). The authors report that R2L can beat L2R on a subset of MCQ benchmarks and attribute the differences to three factors, including calibration, computability, and directional conditional entropy, supported by a controlled 4-digit arithmetic simulation.

### Strengths
* Clear formulation of how to score MCQs under L2R ($\log p(a\mid q)$) vs. R2L ($\log p(q\mid a)$), including three R2L paradigms.
* The “3C” perspective a useful way to reason about when direction might matter.
* The arithmetic simulation is cleanly designed to isolate directional effects and helps illustrate the conditional-entropy story.

### Weaknesses
* Evidence is limited to MCQs, and R2L wins on only a minority of datasets. There is no validation on generative reasoning or long-form tasks where factorization may more directly affect decoding. This weakens the conclusion that L2R “may not be the best inductive bias.”
* There should be some established L2R-only calibration models (contextual calibration) that address the same artifacts without retraining an R2L model. The absence of baselines makes it unclear whether R2L training is necessary versus better L2R scoring.
* Pretrain–inference mismatch is underspecified. The R2L model is trained by reversing all tokens, yet evaluation prompts are shown in human-readable forward order as Figure 4. It is unclear why a reversed-token R2L model can reliably parse forward human-readable MCQs.
* For several benchmarks, the LM-eval templates are changed to present full answer choices rather than labels. This can alter length and surface-form statistics, which are central to the paper’s calibration narrative. A sensitivity analysis (original vs. modified templates) is needed to confirm that the direction effect is not template-dependent.

### Questions
Please refer to the previous Weaknesses.

### Soundness
3

### Presentation
3

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
This paper investigates the impact of autoregressive factorization direction (L2R vs. R2L) on LLM performance in multiple-choice reasoning tasks. The authors train models from scratch and evaluate them on MCQ benchmarks. Results demonstrate that R2L can outperform L2R on specific reasoning categories (e.g., logical, commonsense, and truthfulness tasks). The authors propose calibration, computability, and conditional entropy as key factors explaining directional preferences, supported by a controlled arithmetic simulation.

### Strengths
1. New Insight: Reverse factorization can outperform L2R in reasoning tasks, which challenges a standard assumption.
2. The experimental setup is clear and ensures fair comparisons between models.
3. The theoretical claims are supported by empirical evidence, particularly through the analysis of conditional entropy.

### Weaknesses
1. The study focuses solely on MCQs, and it remains unclear how well the findings generalize to broader language tasks.
2. Conditional entropy estimation relies on approximate sampling, which introduces noise in the interpretation.
3. The observed advantages appear task-specific rather than universal, making the broader impact less definitive.

### Questions
1. R2L outperforms L2R on 4 out of 11 MCQ benchmarks, but the paper generalizes directional preference as a broader phenomenon. Can the authors clarify whether the effect is domain-specific rather than indicative of a general modeling principle?
2. The R2L advantage heavily depends on using an unnormalized scoring of p(q|a), while normalized scoring reduces or eliminates gains. To what extent are the improvements attributable to the scoring formulation rather than the factorization direction?
3. The paper argues that lower conditional entropy correlates with better directional reasoning, but entropy is estimated via single-sample Monte Carlo and shows notable exceptions (e.g., CommonsenseQA, OpenBookQA). How robust is this conclusion when the metric itself is noisy and not strictly predictive across benchmarks?

### Soundness
2

### Presentation
3

### Contribution
2
