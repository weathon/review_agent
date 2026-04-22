# FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 6

## Abstract
The integration of new modalities enhances the capabilities of multimodal large language models (MLLMs) but also introduces additional vulnerabilities.
In particular, simple visual jailbreaking attacks can manipulate open-source MLLMs more readily than sophisticated textual attacks.
However, these underdeveloped attacks exhibit extremely limited cross-model transferability, failing to reliably identify vulnerabilities in closed-source MLLMs.
In this work, we analyse the loss landscape of these jailbreaking attacks and find that the generated attacks tend to reside in high-sharpness regions, whose effectiveness is highly sensitive to even minor parameter changes during transfer.
To further explain the high-sharpness localisations, we analyse their feature representations in both the intermediate layers and the spectral domain, revealing an improper reliance on narrow layer representations and semantically poor frequency components.
Building on this, we propose a Feature Over-Reliance CorrEction (FORCE) method, which guides the attack to explore broader feasible regions across layer features and rescales the influence of frequency features according to their semantic content.
By eliminating non-generalizable reliance on both layer and spectral features, our method discovers flattened feasible regions for visual jailbreaking attacks, thereby improving cross-model transferability.
Extensive experiments demonstrate that our approach effectively facilitates visual red-teaming evaluations against closed-source MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper focuses on transferable visual jailbreaking attacks. It provides a detailed analysis from the perspectives of loss landscape, feature representation, and frequency-domain features to explain why current optimization-based attacks suffer from poor transferability. To address this issue, the authors propose a new approach, Feature Over-Reliance Correction (FORCE), which enhances the transferability of adversarial attacks.

### Strengths
1. The focus on transferable visual jailbreaking addresses an important topic in the safety community.


2. The paper provides a thorough analysis of why optimization-based attacks exhibit poor transferability and generalization.


3. The paper is clearly written and easy to follow.


4. The proposed method improves transferability, with especially strong gains on adapter-based MLLMs.

### Weaknesses
1. The proposed method shows clear improvement on adapter-based MLLMs, but for early-fusion and commercial MLLMs, although it performs slightly better than the baseline, the results are still far from being practically useful.


2. There are no experiments on defenses, which makes it difficult to assess the robustness of the proposed attack.


3. Transferable adversarial attacks have been well-studied in traditional CV and NLP domains. However, the discussion of these prior approaches, especially regarding their applicability to MLLMs, is missing from this paper.

### Questions
1. As the authors analyzed, current adversarial attacks are highly sensitive to weight perturbations. A straightforward idea would be to introduce weight perturbations during the optimization process. Do the authors have any insights on this?


2. In the proposed method, the authors choose to enlarge the feature representation region, which I think I understand the motivation behind. However, an opposite idea, finding adversarial examples whose neighboring regions in the feature space are more similar, might also reduce sensitivity. I’m curious how the authors view this opposite alternative.


Overall, the paper is well-motivated. If the authors can adequately address the concerns mentioned above, I would be inclined to raise my rating.

### Soundness
3

### Presentation
4

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
This paper investigates transferable adversarial attacks on MLLMs, first examining their effectiveness in representation space and then proposing a solution.

### Strengths
1. The paper focuses on an important research question
2. The paper proposes an intriguing method and validates its performance on different benchmarks.

### Weaknesses
1. Clarity of writing (Section 3.1 Motivation).
The exposition is unclear and significantly hinders comprehension. In Section 3.1 (motivation), please clarify:
(a) What is the purpose of adding random noise?
(b) Why does adding adversarial noise increase the optimization loss? If the objective is to elicit “Sure, here it is,” this seems contradictory.
(c) In the bottom figure, what is the purpose of perturbing the model weights? In addition, the claim that “the attack is trapped in a local optimum of the source MLLM” is not evident from the visualization. The figures should be redrawn and more clearly explained to support the stated motivation. This ambiguity also propagates to later sections (e.g., Sec. 3.2).

2. Interpretation of Figure 3.
The explanation provided is insufficient. In Figure 3, layers 1, 6, 16, and 21 exhibit almost identical feature sensitivity, i.e., the required interpolation rate is around 0.2 for all. This observation does not convincingly support the claim that lower layers have worse generalization. Please reconcile this discrepancy or provide additional evidence/analysis.

3. Experimental setup and baselines.
(a) All noises are generated using LLaVA-1.5-7B, lacking cross-model validation across different open-source MLLMs.
(b) The method is not compared against other transferable MLLM adversarial attacks. such as [1,2], which is necessary to contextualize performance and transferability.

[1] A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1 NeurIPS 2025

[2] M-Attack-V2: Pushing the Frontier of Black-Box LVLM Attacks via Fine-Grained Detail Targeting. Arxiv 2025

### Questions
Please first refer to the weakness section.

### Soundness
2

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
This paper proposes the **Feature Over-Reliance CorrEction (FORCE)** method to enhance the cross-model transferability of visual jailbreaking attacks on multimodal large language models (MLLMs). The method addresses the issue of attacks relying on model-specific features by incorporating layer-aware regularization and spectral rescaling. Extensive experiments demonstrate that FORCE significantly improves the transferability and efficiency of visual attacks across various MLLM architectures, including commercial models.

### Strengths
1. The paper offers a clear and well-supported analysis of the underlying reasons for poor transferability in visual jailbreaking attacks. By examining the loss landscape of these attacks, the authors identify that attacks often reside in high-sharpness regions, making them highly sensitive to small changes in model parameters.

2. FORCE effectively addresses the problem of over-reliance on model-specific features in visual jailbreaking attacks. The method combines layer-aware regularization and spectral rescaling to improve the generalizability of the attacks across different models.

3. The experiments conducted demonstrate the effectiveness of FORCE in improving attack transferability across various MLLM architectures. The method is shown to outperform baseline approaches, achieving higher attack success rates and reducing the number of queries required for successful attacks.

### Weaknesses
1. The paper compares FORCE mainly with standard PGD and a few basic methods, but it does not evaluate the proposed method against more recent state-of-the-art transferable adversarial attack techniques. 

2. The experiments do not assess how FORCE-generated attacks perform when models are equipped with defense mechanisms.

### Questions
1. The paper should provide concrete measurements of the computational cost and memory footprint for the proposed FORCE method, as the additional operations for layer regularization and spectral rescaling likely impose significant overhead.

2. The number of baselines compared in your work is relatively limited. Several existing white-box attack methods, including textual attacks like GCG, offer transferable attack strategies by learning a universal suffix for transferability. It would be beneficial to understand why your approach does not include comparisons with these methods.

3. It is important to evaluate how FORCE performs against defended models, particularly testing its effectiveness when target models employ common input-level defenses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the reason of optimization-based visual jailbreaking attacks on multimodal large language models (MLLMs) show poor cross-model transferability. Through detailed analyses of the loss landscape, layer-wise feature representations, and spectral dependencies, the authors find that existing attacks over-rely on model-specific shallow features and semantically weak high-frequency components, leading to sharp and non-transferable adversarial regions. Extensive experiments across both open-source and commercial MLLMs (e.g., LLaVA, InstructBLIP, Qwen, Claude, Gemini, GPT-5) demonstrate consistent and substantial improvements in cross-model transferability.

### Strengths
1. The paper provides a clear and multi-perspective diagnosis of the transferability issue, combining loss landscape visualization, layer interpolation, and Fourier analysis. And the proposed method seems practically effective in improving cross-model robustness of visual jailbreaks.

2. The method is tested on a wide variety of MLLMs (adapter-based, early-fusion, and commercial) and three benchmark datasets, showing consistent ASR gains (up to 20%) and reduced query costs. Component analysis, frequency influence plots, and feature interpolation experiments convincingly support the claimed mechanisms.

3. The paper is well-written and easy to follow.

### Weaknesses
1. The weighting rule w_m = \min(1, \ell_{m-1}/\ell_m) is somewhat ad-hoc and not derived from optimization principles; an adaptive or learning-based variant could be more convincing.

2. While the proposed FORCE demonstrates clear gains over PGD, the evaluation omits comparisons with other state-of-the-art transferable attack methods. Including these would strengthen the empirical evidence and contextualize the contribution.

3. “In this challenging setting, our method substantially improves transferability, achieving nearly a 100% increase over the baseline ASR”, corresponds to the ASR increase from 1% to 2%. I still think the performance can be limited. While I think the authors can briefly discuss why Early-Fusion MLLMs achieved limited attack performance.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
