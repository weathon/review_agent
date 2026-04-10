## Summary
This paper investigates whether large language models (LLMs) fully utilize training data for mathematical reasoning. Through experiments on models like Llama3 and Gemma3 with supervised fine-tuning and reinforcement learning on datasets including GSM8K and MATH, the authors find that adding more training data causes a significant portion (10–15%) of previously correct test answers to become incorrect. They attribute this to high predictive multiplicity (Rashomon effect), where models trained on the same data with different random seeds learn divergent functions, each solving only a disjoint subset of the test set.

## Strengths
- **Identifies and robustly demonstrates a counter-intuitive phenomenon:** The paper clearly shows that increasing training data can lead to forgetting of correct answers in math reasoning tasks, challenging standard scaling assumptions. This is evidenced across multiple model families (Llama3, Gemma3, Qwen2.5), training methods (SFT, RL), datasets (MAWPS, GSM8K, MATH), and inference techniques (greedy decoding, majority voting).
- **Comprehensive empirical coverage:** The experiments establish generality within the studied domain, with consistent results across varied settings, including tests without parameter-efficient fine-tuning (Appendix A.2) and with different model capacities (Appendix A.1).
- **Effective theoretical framing:** The paper connects empirical observations to established concepts of predictive multiplicity and Rashomon sets, providing a plausible conceptual explanation beyond mere correlation (Sections 4.1–4.2).

## Weaknesses
### Major:
- **Insufficient statistical support for variability claims:** The paper’s central argument about predictive multiplicity and seed-dependent differences relies on a very small number of random seeds—3 for supervised fine-tuning and only 1 for reinforcement learning experiments. This undermines the reliability of conclusions regarding the diversity of learned functions and the intersection of correct answers (e.g., Figures 5, 6, 7). For findings that hinge on randomness, more seeds are essential to ensure statistical significance.
- **Weak causal linkage between data addition and predictive multiplicity:** While the paper documents both forgetting with added data (Section 3) and high multiplicity in fixed-set training (Section 4.1), the direct mechanistic link between these phenomena is asserted rather than proven. The theory in Section 4.2 explains why large Rashomon sets exist but does not model why adding data should trigger shifts within this set, leaving open alternative explanations like optimization dynamics or catastrophic interference.

### Minor
- **Simplified theoretical analysis:** The combinatorial framework for Rashomon sets (Section 4.2) relies on strong assumptions (e.g., independence of per-sample strategies) and is not empirically validated beyond strategy counts. A more rigorous analysis of the loss landscape or model agreement would strengthen the contribution.
- **Superficial analysis of “strategies”:** The paper defines strategies as sequences of operations and reports counts, but lacks deeper investigation into what makes strategies different (e.g., semantic vs. syntactic variations) or how strategy choice correlates with correctness flips. This limits insight into the root causes of multiplicity.
- **Limited exploration of mitigation and practical implications:** The paper diagnoses a significant problem but offers no concrete solutions or experiments on how to mitigate it (e.g., via ensembling, regularization, or data curation). While the core contribution is diagnostic, addressing mitigation would enhance impact.

### Trivial
- **Narrow model scale range:** Appendix A.1 shows the effect persists up to 12B parameters, but frontier models are much larger. However, within the paper’s scope of studying the phenomenon, the models used are sufficient for initial demonstration.

## Nice-to-Haves
- A deeper qualitative analysis of examples where answers flip from correct to incorrect, including reasoning traces, to provide intuitive understanding.
- Experiments with more advanced test-time scaling techniques (e.g., verifier-based ranking) to further confirm the robustness of the phenomenon.
- Ablation studies on hyperparameters like learning rate or batch size to assess sensitivity.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Criticisms about model/dataset availability:** Any doubt about the existence or release status of cited models (Llama3, Gemma3, Qwen2.5) or datasets (GSM8K, MAWPS, MATH) is removed, as they are assumed available per hard rules.
- **Reproducibility nitpicks:** Requests for more detailed hyperparameters or implementation details beyond those provided in Sections 3.2.1 and 3.2.2 are removed as trivial per hard rules.
- **Unfair comparisons:** No such weaknesses were present.
- **Missing related works:** Suggestions to add more references are omitted per hard rules.
- **Formatting/style comments:** Any minor writing or presentation issues are excluded.
- **Strawman weaknesses:** Claims that the paper does not address multiplicity or linking are removed, as the paper explicitly discusses these in Sections 4.1–4.2.

## Suggestions
- Increase the number of random seeds to at least 5–10 for all experiments to bolster statistical claims about variability and predictive multiplicity.
- Conduct a controlled experiment where a single model is trained sequentially on data subsets to disentangle catastrophic forgetting from Rashomon effect-driven variability.
- Enhance the strategy analysis by clustering reasoning traces beyond operation sequences and examining how strategy distributions shift with added data or across seeds.