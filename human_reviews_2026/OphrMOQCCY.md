# Characterizing and Mitigating Reasoning Drift in Large Language Models

- Decision: Accept (Poster)
- Scores: 8, 4, 2, 6

## Abstract
While chain-of-thought prompting enables powerful multi-step reasoning in Large Language Models (LLMs), the stochastic nature of the generation process undermines its reliability. In this work, we first analyze thousands of reasoning paths to identify Reasoning Drift, a key failure mode where models get locked into flawed reasoning patterns. We reveal that the manifestation of drift is a complex interplay between universal functional tendencies and unique, model-specific signatures. Based on the diagnosis, we propose Reasoning-Aware Activation Steering, a novel inference-time intervention method to gently nudge the model's activations away from pathological patterns. We pre-compute a library of vectors from contrastive functional transitions and apply them dynamically. Experiments show that our method effectively mitigates the drift problem and boosts accuracy. Additionally, it generalizes to out-of-distribution tasks, demonstrating a deeper capture of valid reasoning principles.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper first characterizes the failure modes of LLM reasoning patterns, and then proposes to steer the generation with pattern-specific steering vectors.

To characterize the failure patterns, the paper trains a classifier to label each reasoning step into one of the 9 categories (e.g. "initial question". "Problem setup", "Uncertainty management" etc). It identifies model-specific failure modes, e.g. Llama tends to get stuck in self-loops, and Qwen tends to make erractic jumps.
- Specifically, the classifier is a DistillBERT trained on the reasoning trajectories from the Math-Rollout datasets and the category labels therein.

Based on these findings, the paper constructs steering vectors by taking the difference in activations between a desired step and an undesired step. At inference time, the model computes per-step steering vector adaptively and adds the steering vector to the activation.
- The layer to intervene is chosen to be the one that incurs the most change in the activation.
- The adaptive steering vector is a weighted sum of all steering vectors, where the weights are based on the similarity the current activation and the steering vectors.

The experiments are based on R1-distilled Llama and Qwen, using 4 math datasets (i.e. GSM8k, AIME2024, AIME2025, GPQA-Diamond). They show that:
- The steering method effectively changes the transition probabilities among the 9 categories.
- Steering improves math reasoning performance over baselines (vanilla CoT, self-consistency, and PiCSAR).
- The steering method is applicable to Llama and Qwen without distillation, though the improvement is much smaller.

### Strengths
- The analysis on characterizing per-step patterns can be useful for diagnosing models, which helps inform intervention which the paper subsequently explores. 
- The proposed inference-time intervention is lightweight and can be practically useful. Results on the 4 math datasets are promising: the intervention achieves noticeably better accuracy without requiring resampling / parallel sampling.
  - Notably, the steering vectors computed on Math-Rollout are applicable to 4 datasets which can be considered OOD.
- The writing is easy to follow.

### Weaknesses
- Sec 4.4 / Table 2: The effectiveness of steering seems to decrease significantly with non-distilled models.
- The improvement on AIME2025 (i.e. the newest benchmark) is much more moderate than other datasets. This might suggest that the method might be less effective when the applicable is truly OOD -- which is understandable though worth noting.
- The intervention requires white-box access to the activations, which can be strong for certain practical considerations.

### Questions
- To clarify, is it correct that: the trained DistilBERT classifier is for annotating rollouts that are not in Math-Rollouts, and the classifier is used only for analysis and not during inference.
- line 297: should it be $S^+$ has the label $i$?
- Sec 4.4 / Table 2: are the steering vectors extracted per-model? What do you think makes the performance gain much weaker now (e.g. is this related to whether the models have gone through distillation)? What are the performance of SC and PiCSAR?
- How to improve the effect of steering in later stage? For example, is this because one token matters less when the attention is put on more tokens compared to at the beginning? Would additionally biasing the attention weights be helpful?

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
3

### Summary
This paper introduces "Reasoning Drift," a failure mode in LLM chain-of-thought processing where models adopt flawed functional patterns. To address this, the authors propose Reasoning-Aware Activation Steering (RAAS), an inference-time method that uses a pre-computed library of steering vectors to guide model activations away from pathological states. The method is evaluated on several mathematical reasoning benchmarks and demonstrates significant accuracy improvements.

### Strengths
1) The paper provides a rigorous, data-driven characterization of "Reasoning Drift." The analysis of functional state transitions offers a valuable new lens for understanding LLM reasoning failures that moves beyond surface-level error analysis.
2) The proposed RAAS intervention is elegantly designed and directly motivated by the preceding diagnostic analysis. As a lightweight, interpretable, inference-time technique, it presents a practical alternative to computationally expensive methods like large-scale sampling.
3) The method achieves consistent and substantial accuracy gains across multiple challenging mathematical reasoning benchmarks, including complex, out-of-distribution datasets like AIME.

### Weaknesses
1) The core methodology relies on a strong and unproven assumption. The diagnostic analysis and steering vectors are derived from sentence-level representations (using mean-pooled activations; see Eq. 2), yet the intervention is applied at the token level (see Eq. 4, 6). The paper provides no evidence to validate that a single token's activation can be meaningfully compared to the averaged activation of an entire sentence, calling the intervention's foundational principle into question.
2) The paper demonstrates a correlation between the RAAS intervention and improved performance but fails to establish causation. The accuracy gains could be a byproduct of an uncontrolled factor, such as a regularization effect from the activation shifts, rather than the specific correction of functional transitions as claimed. The lack of critical control experiments (e.g., applying randomly selected steering vectors) leaves the mechanistic claims unverified. Maybe adopting significance tests (such as bootstrap) can make results more convincing.
3) The paper's claim of generalization to "out-of-distribution tasks" may be misleading. The steering vectors are extracted from a mathematics dataset (Math-Rollout), and all evaluation benchmarks (GSM8K, AIME, GPQA) are also within the mathematical reasoning domain (see Sec 4.1). This constitutes within-domain generalization at best, and there is no evidence to suggest the framework would be effective in other domains such as programming.

### Questions
The questions refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper identifies the issue of "Reasoning Drift" (when LLMs use unsuccessful reasoning patterns) and propose Reasoning-Aware Activation Steering (RAAS), an approach based on steering vectors to mitigate this issue. They claim that current methods: 1. require lots of model inferences to implement some kind of best-of-N like mechanism, and 2. don't use LLM internals so we don't understand where they go wrong (although this method doesn't help too much with this). The analysis in the paper is done using the Math-Rollout dataset, a dataset of CoT reasoning traces of a DeepSeek R1 distilled Qwen model on common math datasets, where reasoning sentences are labeled to one of 8 different reasoning categories, e.g., problem setup or plan generation, and also sentences are swapped out in various parts of the CoT to study counterfactual generations. The authors use this dataset to train a DistillBERT model to classify the reasoning category of sentences, and then use this to determine which reasoning category transitions on a sentence by sentence level are useful and not useful, e.g., perhaps self checking after problem setup is not very useful. The RAAS algorithm uses these transitions to design steering vectors and then adds a weighted average of these vectors when the reasoning category transition is expected to not be useful. Unrelatedly, they identify a "funneling effect", where on this math reasoning dataset, changing sentences early in the CoT have more influence over final answers than later sentences.

### Strengths
- The analysis of the Math-Rollout dataset to identify the reasoning drift is conceptually interesting. The "funneling effect" and the distinction between model-specific failure signatures (Inertial vs. Chaotic Drift) are also interesting.

- Use of activation steering to the process of reasoning (RAAS) at test-time is a creative idea. This makes the method computationally efficient.

### Weaknesses
- The most significant weakness is the paper's heavy reliance on the Math-Rollout dataset and limited evaluation. This dataset is specific to the math domain and derived from only two specific distilled models (R1-Distill-Llama/Qwen). It is unclear if the identified drift patterns, the 8-category taxonomy, or the derived steering vectors generalize beyond this domain and beyond these models. The gains on non-distilled models are too small to justify generalizability.

- The paper suffers from significant clarity issues, making it difficult to follow, and the writing style often lacks precision and lots of loose terminology. Key terms such as "universal functional tendencies" and "functional shifts" are vague and not formally defined. The introduction frames the problem confusingly. For example, it describes how changing sentences in a CoT can lead to "entirely different outcomes" as if this is inherently problematic, whereas this is obvious and expected in a reasoning process. The specific "negative impact of stochasticity" they aim to address is not clearly articulated. The figures are confusing and poorly presented.

- Experimental details for the DistilBERT classifier, a critical component of RAAS, are largely missing from the main paper and sparse even in Appendix D.

### Questions
- How dependent are the identified functional transitions and drift patterns on the specific characteristics of the Math-Rollout dataset and the DeepSeek distillation process? Can it extend to other model families and tasks?

- In Figure 3, what do the neon blue and purple outlines around certain rectangles signify? The caption is incomplete and the text does not clarify this.

- In the introduction, the authors seem to treat the fact that changing sentences in a CoT leads to different outcomes as a problem stemming from stochasticity. Can the authors clarify why this is viewed as a "negative impact"? Isn't it expected that different reasoning steps lead to different conclusions?

- Given the very small performance gains on the base models (Table 2), does this suggest that the steering vectors are primarily capturing behaviors specific to the distillation process or the specific dataset, rather than generalizable reasoning principles?

- Writing needs massive improvement. There is a lot of jargon phrases like 'reasoning drift', 'funneling effect', 'universal functional tendencies' and 'functional shifts' to name a few. They don't really mean much and are never formally defined. Also the intro seems to use a lot of such repetitive language but doesn't convey the main results. Also, RAAS uses Reasoning-Aware and Reasoning-Adaptive in different parts.

### Soundness
2

### Presentation
1

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
This paper exploits a rich dataset of math reasoning rollouts to trace reasoinng trajectories that cause failure or success. Through defining corrective and erroneous rollouts, it identifies the promising and unpromising steps and studies the transitions of the model between step categories. This gives insight into which transitions are useful and which ones might hurt performance. Interestingly, this reveals different failure patterns in the Qwen and Llama models. Moreover, the paper proposes a method for detecting the undesired drift in reasoning and learns steering vectors that prove to be effective to correct reasoning trajectories in inference time.

### Strengths
The paper provides a granular study of the frequent reasoning patterns, which is helpful in studying the failure modes of reasoning in large language models. Through using a rich dataset, it is able to identify model-specific patterns. Then, it extends its study to a practical method that detect drifts in reasoning and corrects them at inference time with interventions with steering vectors. Overall, the work both brings clarity to understanding reasoning behavior of language models and provides practical solutions.

### Weaknesses
The proposed methods, while being effective in experiments, do not seem very well-motivated. It is not clear why defining the DriftScore in the way it is defined is plausible. Moreover, it seems that v_{i->j} can be computed from a_{i, j}^+ and a_{i, j}^-. Finally, the application of the method to other methods RQ3 could be explored more thoroughly to understand the gap between the results.

### Questions
1. Could you please elaborate on why the improvements for other models (RQ3) is less significant? Are the transition patterns different? 
2. Could you explain the reason behind the definition of DriftScore? Is it just a heuristic or it is well-motivated?

### Soundness
3

### Presentation
4

### Contribution
3
