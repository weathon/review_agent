# DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Recent reasoning Large Language Models (LLMs) demonstrate remarkable problem-solving abilities but often generate long thinking traces whose utility is unclear. We conduct a systematic analysis across models and datasets and discover a U-shaped entropy pattern: high entropy on simple problems despite high accuracy, low entropy on medium difficulty, and high entropy on hard problems reflecting uncertainty. The 22--25\% entropy reduction from simple to optimal regions reveals a fundamental inefficiency—an \emph{overthinking} phenomenon on easy instances. Building on these insights, we introduce \textbf{DiffAdapt}, a lightweight, deployment-ready framework that predicts problem difficulty from hidden states and selects among Easy/Normal/Hard reasoning strategies to allocate computation adaptively. DiffAdapt requires no retraining of the base LLM and is compatible with common inference optimizations. Across five models and eight benchmarks, DiffAdapt achieves comparable or improved accuracy while reducing token usage by up to 22.4\%, establishing a practical path toward compute-efficient reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the inefficiency of reasoning LLM, which often generate excessively long reasoning traces regardless of task difficulty. Through a systematic empirical analysis across multiple models and datasets, the authors discover a U-shaped entropy pattern on simple problems, low entropy on medium difficulty, and again high entropy on hard problems. This reveals a phenomenon termed overthinking, where models over-allocate computational resources to easy problems. Building on this observation, the paper proposes DiffAdapt, a lightweight, training-free framework that predicts problem difficulty from the model’s hidden states and dynamically selects among three reasoning strategies to adapt computational budget during inference. DiffAdapt requires no retraining of the base LLM and is compatible with existing inference systems.

### Strengths
The proposed DiffAdapt is a simple yet effective solution that does not require retraining or finetuning the base LLM. By attaching a small hidden state probe to predict difficulty, the method enables difficulty-aware computation allocation at inference time, making it highly practical for real world deployment. The paper conducts extensive experiments across five reasoning models and eight benchmarks, providing convincing evidence for the generality and robustness of DiffAdapt across architectures, scales, and domains. It can be seamlessly combined with reinforcement-learning-based length control methods such as ThinkPrune and LC-R1, as well as deployed under common inference frameworks, which underscores its engineering value and scalability.

### Weaknesses
1. Limited evaluation of reasoning-chain integrity under tight computational budgets. The paper primarily reports accuracy, token reduction, and speed-up metrics but does not evaluate the integrity or consistency of reasoning traces (e.g., whether truncated reasoning affects logical completeness). This omission makes it difficult to judge whether DiffAdapt preserves coherent chain-of-thought reasoning when aggressive token reduction is applied.

### Questions
1. Appendix D.2 states that thresholds for distinguishing Easy/Normal/Hard are heuristically chosen per model based on entropy, with only a small sanity check. Since these thresholds depend on model family, temperature, and domain, does this design undermine generalization and zero-shot ability? Would each new deployment require manual calibration?
2. Could the authors show how often misclassification (e.g., Hard -> Easy) causes early truncation or logical failure, and evaluate the real effectiveness of this fallback mechanism during deployment?

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
The paper introduces DiffAdapt, a lightweight, training-free framework for difficulty-adaptive reasoning in large language models (LLMs). The authors identify a U-shaped entropy pattern across problem difficulty levels, indicating an “overthinking” phenomenon where models exhibit high uncertainty on easy problems despite high accuracy. To address this, DiffAdapt uses a small probe trained on the model’s hidden states to predict problem difficulty and select among Easy/Normal/Hard inference strategies dynamically, without retraining the base model. Experiments on five LLMs and eight benchmarks show that DiffAdapt maintains or improves accuracy while reducing token usage by up to 22.4%, and decreases end-to-end inference time by up to 6×. The framework is compatible with existing inference systems and complements length-control RL methods.

### Strengths
1. The work identifies and systematically analyzes a previously underexplored phenomenon—overthinking in reasoning LLMs—via a novel U-shaped entropy pattern analysis. This empirical finding is insightful and establishes a theoretical foundation for adaptive reasoning.
 
2. The proposed DiffAdapt framework is conceptually elegant: predicting difficulty from hidden states and dynamically adjusting reasoning strategy without any retraining of the base model.

3. The experimental section is comprehensive, spanning five models and eight benchmarks (both in-domain and out-of-domain). The Oracle analysis provides solid upper bounds that justify the design choices.

4. The contribution is practically important: it offers a lightweight, deployment-ready framework. DiffAdapt requires no retraining of the base LLM and is compatible with common inference optimizations.

### Weaknesses
1. The evaluation is heavily focused on mathematical and scientific reasoning. It remains unclear whether the proposed three-regime (Easy/Normal/Hard) framework generalizes to other domains, such as commonsense reasoning, dialogue, or creative writing, where difficulty is harder to quantify.

2. The difficulty predictor leverages only prefill hidden states, which simplifies deployment but may overlook valuable cues from generation dynamics. Discussion or experimentation on this trade-off would be helpful.

3. The paper does not provide ablations on the probe’s architecture or its sensitivity to training data size. Such results would clarify how much of DiffAdapt’s performance depends on probe complexity or data scale.

### Questions
1. How sensitive is DiffAdapt to the chosen thresholds (α, β, γ)? 

2. Could DiffAdapt extend beyond reasoning tasks—for example, to summarization or dialogue where difficulty varies dynamically?

3. How does probe misclassification (wrong strategy selection) affect performance or stability?

4. In the Hard strategy, generating only a "method outline" may fail on problems requiring detailed computation. Could you discuss potential limitations of this approach—particularly for problems that may still benefit from detailed computation?

### Soundness
3

### Presentation
3

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
The paper observes a U‑shaped entropy–difficulty pattern (high entropy for easy/hard, low for medium) and posits “overthinking” on easy items. DiffAdapt trains a light probe on prefill hidden states to predict Easy/Normal/Hard and selects a matching reasoning strategy at inference. Across five model families and eight benchmarks (including OOD), DiffAdapt maintains or improves accuracy while reducing tokens (up to 22.4%). The approach is orthogonal to Length‑Control RL and integrates with mainstream serving stacks.

### Strengths
- Strategy selection (not just length/temperature) based on a learned difficulty proxy. Notably, the paper’s identification of a U-shaped entropy–difficulty curve is an interesting empirical finding. Previous work mainly reported monotonic increases of entropy with problem difficulty, but not this symmetric “overthinking” pattern on easy items. This observation provides a concrete diagnostic for inefficiencies in reasoning-token allocation.
- Consistent savings across models/benchmarks; OOD and LC-RL results broaden applicability.
- Oracle and ablation studies (vs. fixed strategies, DEER) clarify where gains come from.

### Weaknesses
- Heuristic thresholds (α,β,γ) are set per model from scatterplots; it’s unclear how stable they are across multi‑task mixtures and domain shifts, or how often the probe/thresholds need re‑tuning.
- Stage‑1 data generation (multiple long samples per item) is expensive for new domains; low‑budget or few‑shot variants are not discussed.
- Missing head‑to‑head with “when‑to‑think” switchers (e.g., Thinkless/AdaCoT); this would position the magnitude of gains.

### Questions
- Multi‑task generalization: In a realistic mixed‑task setting, how robust are the probe and (α,β,γ) without re‑tuning? Can you show cross‑domain transfer or per‑task calibration drift?
- Sensitivity of results to (α,β,γ); can you show heatmaps or robust ranges?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the overthinking phenomenon in reasoning LLMs, where models exhibit high uncertainty and unnecessarily long reasoning traces even on simple problems. Through systematic entropy analysis across models and datasets, the authors reveal a consistent U-shaped entropy pattern across difficulty levels, indicating computational inefficiency. Building on this finding, the authors propose DiffAdapt, a difficulty-adaptive inference framework that predicts problem difficulty using a lightweight probe on hidden states and dynamically selects from Easy/Normal/Hard reasoning strategies. The approach requires no retraining of the base LLM and can be deployed with existing inference frameworks. Empirical results on five models and eight benchmarks show that DiffAdapt can reduce token usage by up to 22.4% while maintaining or improving accuracy.

### Strengths
- The proposed DiffAdapt framework only requires training a small external probe, yet it effectively improves reasoning efficiency and model performance without modifying or fine-tuning the LLM itself.
- The identification of a consistent U-shaped entropy–difficulty pattern is novel and insightful. It sheds light on the overthinking behavior of reasoning models and provides valuable guidance for future “long-to-short” reasoning research.
- The framework is complementary to reinforcement learning–based length control methods (e.g., ThinkPrune). This suggests DiffAdapt can be combined with those training-based long-to-short approaches for even greater efficiency gains.

### Weaknesses
- Model-specific probe requirement: The main weakness lies in the need to train a separate probe for each LLM. This introduces (1) additional training overhead and (2) potential generalization issues — the probe’s limited transferability could restrict DiffAdapt’s applicability across different models or domains.
- Lack of probe analysis experiments: The paper would be stronger if it included an analysis of the probe’s generalization ability. For example: (a) How well does a probe trained on the DeepMath dataset transfer to other reasoning tasks or domains? (b) Can probes be transferred between models of similar architecture or scale (e.g., between Qwen-3/4B and LLama-3B)?
- Ad-hoc reasoning strategy design: The selection of parameters for the Easy/Normal/Hard strategies (e.g., temperature = 0.5/0.8/0.4, token ratio = 0.4×/1.0×/0.5×) feels somewhat heuristic. The paper lacks experimental justification or ablation to support why these specific hyperparameters are optimal.

### Questions
- Have the authors tested whether a probe trained on DeepMath generalizes to other domains or reasoning benchmarks?
- How transferable is the probe across models of similar size or architecture?
- What motivates the specific parameter choices for the three reasoning strategies, and have alternative configurations been compared experimentally?

### Soundness
2

### Presentation
3

### Contribution
3
