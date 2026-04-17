# Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Test-time compute (TTC) strategies have emerged as a lightweight approach to boost reasoning in large language models, but their applicability to vision–language models (VLMs) remains unclear.
We present a systematic study of TTC for visual reasoning across seven open-source VLMs and six benchmarks, revisiting two paradigms: (i) feature-based scoring of chain-of-thought (CoT) traces and (ii) confidence-based aggregation via majority voting (MV).
In the single-model setting, feature cues (e.g., length, pivot words) fail to improve accuracy, while MV yields only modest, CoT-dependent gains.
To explain this limitation, we theoretically show that the voting method's effectiveness depends on prediction diversity: when outputs are highly correlated, the benefit of voting vanishes. 
In contrast, multi-model ensembles introduce stronger diversity through architectural differences, training data, and scale, making them both more realistic and more promising for TTC. 
However, MV treats all models equally, leaving it vulnerable to correlated errors from weaker models. 
To address this, we propose Entropy-based TTC, which selects the most confident prediction based on predictive entropy. 
Our method reduces to MV in the single-model case but, in ensembles, leverages confidence disparities to prioritize stronger models. 
We prove that our method theoretically outperforms MV under mild dependence assumptions, and empirically show that it consistently surpasses both MV and the best individual model across diverse visual reasoning benchmarks. 
This demonstrates that smaller models can enhance, rather than hinder, larger ones when combined appropriately, unlocking synergistic gains not achievable with existing TTC strategies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper systematically studies Test-Time Compute (TTC) strategies for visual reasoning with Vision-Language Models (VLMs), extending methods originally designed for LLMs. It finds that feature-based heuristics (e.g., reasoning length, pivot words) fail, and majority voting (MV) yields only modest gains due to limited prediction diversity. The authors provide a theoretical explanation linking MV's effectiveness to output dependency and introduce Entropy-based TTC (ETTC), which selects the most confident prediction via predictive entropy. ETTC generalizes MV, performs better under dependency, and is effective in multi-model ensembles, even allowing smaller models to enhance larger ones. Empirical results across seven VLMs and six benchmarks show that ETTC consistently outperforms both MV and the strongest single model without extra training.

### Strengths
1. The paper systematically examines TTC across multiple VLMs and benchmarks, bridging an important gap between LLM and VLM reasoning.
2. It provides a clear analytical link between prediction correlation and ensemble effectiveness, offering principled insight into why traditional voting fails.
3. ETTC is elegant, computationally light, and backward compatible with existing TTC setups—making it practical and generalizable.
4. The discovery that smaller models can enhance larger ones when weighted by confidence sounds interesting.

### Weaknesses
1. This work compares mainly to majority voting and other heuristics, but omits several recent, high-performing TTC methods that explicitly address confidence calibration and adaptive compute, such as: 
Efficient Test-Time Scaling via Self-Calibration (Huang et al., 2025)
https://arxiv.org/pdf/2503.00031
COME: Test-Time Adaptation by Conservatively Minimizing Entropy (Zhang et al., 2025)
https://openreview.net/forum?id=506BjJ1ziZ
2. Why restrict evaluation to multimodal models (MLLMs/VLMs) rather than also testing on pure LLM reasoning benchmarks? Was there something specific about the vision-language setting, such as multimodal uncertainty, input grounding, or modality-induced diversity, that made TTC particularly interesting or challenging here?
3. The paper does not quantify inference overhead or efficiency trade-offs of the proposed method vs the baselines.

### Questions
1. Did you control for dataset bias or label imbalance that might affect entropy distributions or voting behavior?
2. How sensitive is ETTC to the number and composition of models in the ensemble? Does performance plateau or degrade beyond a certain number of weaker models?
3. In the same-family experiments (Qwen 3B–72B), what explains cases where smaller models overrode larger ones? Could you add some examples?

### Soundness
3

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
- This work presents a systematic study of test-time compute for reasoning of vision-language models, evaluating seven VLMs using six benchmarks.
- The authors examine two paradigms: feature-based scoring of chain-of-thought (CoT) traces and confidence-based aggregation via majority voting (MV).
- The authors first studied TTC in a single-model setting and concluded that MV does not perform well there due to a lack of diversity (even with CoT prompting).
- The authors introduced Entropy-based Test-time Consistency (ETTC), which selects the prediction with the lowest entropy (i.e., most confident), exploiting confidence gaps between models.

### Strengths
- Multiple benchmarks were used to validate their TTC strategy, covering math reasoning, diagram understanding, and general visual reasoning.
- The authors presented their motivation and the background for this work well. The analysis of single-model MV shows only small gains, which motivates this work further.
- The authors verified ETTC with two ensemble configurations: 1) diverse models of different families, 2) scaled models developed by the same model developers. It's informative that the authors checked both cross-family diversity and same-family scaling.

### Weaknesses
- It would be nice to see results on frontier VLMs. This framework can be further validated with thinking models, but was mainly validated with non-frontier VLMs using just chain-of-thought prompting (i.e., the model is explicitly asked to think step-by-step). It's hard to say how much better MV or even their entropy-based approach with an ensemble of models will perform compared to just a single, actually top-tier frontier model alone, which should be confident in its answers when evaluated on these standard VLM benchmarks.
- The scope of this work is limited to or dedicated to VLMs. However, why is it limited to VLMs? More concretely, I would like more empirical evidence that backs the author's claim in the introduction and section 3, where this problem "...is further exacerbated in VLMs due to the perception bottleneck, visual content must first be interpreted before any meaningful variation can emerge." Please back this with experiments (e.g., text-only LLM MCQ using ETTC, and visual corruption evaluations where perception really matters).
- This work examines test-time compute for VLMs, which appears to be a general proposal to improve TTC for VLMs overall. However, results are only on MCQ task--does this framework hold for open-ended QA?

### Questions
- Should there be more ablations on the number of models with a mix of strong + weak models? It would be nice if the authors showed diminishing returns as the number of models increases and related gains to diversity and dependency measures. It's hard to fully trust the main results, given that ETTC was mainly run on two model ensemble configurations.
- Ensembling is a compute trade-off. Should there be more analysis of the trade-off between compute cost and accuracy, and between the TTC budget and accuracy?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the efficacy of the Test Time compute techniques in VLMs. The authors show that in the single model setting, feature based TTC methods fail to improve accuracy in VLMs, and confidence based majority voting yield 2-4% average gain. The main insight from the paper is that effectiveness of majority voting is monotonically decresing in prediction dependency and therefore the paper introduces Entropy based TTC, which consistently surpasses individual model and majority voting.

### Strengths
- The paper is well written and easy to follow.
- The paper introduces ETTC, which is a novel technique to ensemble aggregation of VLMs and it addresses the problem of correlated errors in majority voting. This enables smaller models to enhance larger models.
- The evaluation in the paper is comprehensive using VLMs such as Qwen, Llama, Gemma, Pixtra across various reasoning domains such as MathVista, TQA, MMMU.
- The theoretical analysis (Theorem 1) grounds the empirical observations about majority voting limitations.

### Weaknesses
- The key idea of ETTC relies on the assumption that predictive entropy can be used as a proxy for correctness. But poorly calibrated models can make this method less effective. Though authors mention this, I would encourage authors to analyze the robustness of their method under such scenarios.
- ETTC requires access to full predictive distribution of the VLMs which may not be accessible for black box models available under API calls.
- The paper focuses on the multiple choice visual question answering tasks and it is unclear how the proposed method would perform on the open ended generation tasks.

### Questions
- How would the performance gap between ETTC and Majority voting change if the VLMs are intentionally miscalibrated?
- What is the computational overhead of ETTC. Both ETTC and MV require generating U samples, how much cost is for the post processing step?
- Since the goal is compute efficiency, is the proposed method robust to weak small models?

### Soundness
3

### Presentation
3

### Contribution
3
