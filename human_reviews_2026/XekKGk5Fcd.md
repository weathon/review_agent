# AMPS: Adaptive Modality Preference Steering via Functional Entropy

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Multimodal Large Language Models (MLLMs) often exhibit significant modal-
ity preference, which is a tendency to favor one modality over another. Prior
work has applied steering methods to adjust the modality preference of MLLMs.
However, these conventional approaches apply a uniform steering intensity to all
samples. This lack of adaptation is problematic because strong steering can dis-
rupt a model’s standard inference capabilities, leading to high error rates, while
weak steering may be ineffective. To address this limitation, a sample-wise diag-
nostic tool is required to measure MLLMs’ susceptibility to steering across differ-
ent multimodal samples. To reduce the disruption of strong steering to MLLMs’
inference capabilities, we first introduce a diagnostic metric that quantifies the in-
formation contribution ratio from each modality in MLLMs. This metric reveals
varying susceptibility to steering across different samples. Building on these di-
agnostic insights, we further propose a steering scaling strategy that applies lower
steering intensity for samples highly sensitive to steering, and design a learnable
steering module that automatically learns appropriate scaling patterns, enabling
context-aware adjustment of modality preference. Experimental results show that
our context-aware scaling method outperforms conventional steering strategies
in modulating modality preference, achieving effective adjustment while signif-
icantly reducing generation errors.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Adaptive Modality Preference Steering (AMPS) to tackle the significant modality preference problem in MLLMs. The authors propose a novel, sample-wise diagnostic metric, Modality Contribution Score (MCS), derived from functional entropy, which quantifies each modality's information contribution and reveals varying steering susceptibility. Building on MCS insights, AMPS employs a learnable module that adaptively adjusts steering intensity: applying weaker steering for highly sensitive samples to prevent errors, and stronger steering for robust ones to ensure an effective preference shift. Experimental results on the MC2 dataset demonstrate that AMPS significantly outperforms conventional strategies.

### Strengths
1. Novel MCS diagnostic, grounded in functional entropy, combined with an original adaptive, learnable steering framework (AMPS).
2. This is a high-quality research with sound methodology, comprehensive experimental validation across models and tasks, strong baselines, and clear ablation studies.
3. This paper is well-structured, clear articulation of problem, solution, and benefits. Effective use of figures and explanations.
4. Significantly advances MLLM control by providing both a powerful diagnostic and an effective adaptive steering mechanism.

### Weaknesses
1. The paper frequently mentions reducing generation errors. Could the authors provide a more detailed and quantitative definition of these errors and explain the evaluation methodology?
2. Figure 2 appears wrong expression “Tuxtual”.

### Questions
1. The paper's related work acknowledges modality bias in video QA. Why were no experiments conducted on video tasks, and what specific challenges would arise from such an application?
2. Why is the log-Sobolev bound valid in your setup, and what evidence support replacing the true KV-state distribution with a Gaussian?
3. While the theoretical foundation of MCS is rooted in functional entropy and Fisher information, please elaborate on the practical approximations made in Algorithm 1 and discuss their implications for the fidelity and robustness of the MCS measurement.
4. Without online MCS estimation when inference, isn’t the “adaptive” scaling effectively an offline fit? How do you validate robustness and OOD generalization?
5. Do the paper include results in other general MLLM understanding or reasoning benchmarks? Like VQA, OCR and multi-round QA.
6. Can the authors include results versus more relevant adaptive controllers (AutoSteer, CausalMM, decoding-time reweighting) and clarify how metric fairness is maintained when comparing to prompt-based methods?

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
This paper focuses on adaptive modality preference steering in multimodal large language models (MLLMs), which can be prone to modality preference conflicts during inference. The authors identify the challenge of using a fixed steering strength that can either over- or under-adjust the model’s behavior depending on the sample. 

To resolve this issue, they propose a new methodology called AMPS (Adaptive Modality Preference Steering). It uses a sample-level diagnostic metric, Modality Contribution Score (MCS), to measure the sensitivity of the model to modality preference shifts, which is based on functional entropy. AMPS adaptively adjusts the strength of modality preference steering based on the MCS. Experimental results demonstrate that AMPS significantly improves modality preference transfer compared to traditional static steering approaches, while reducing the generation errors, particularly in more sensitive tasks.

Overall, the idea of the proposed method is reasonable and somewhat interesting. However, the paper writing can be further improved, where the intuition of modules is not very clearly clarified and may impede readability.

### Strengths
1.	This research addresses the valuable and practical task of mitigating modality preference bias in multi-modal large language models (MLLMs), which directly impacts real-world performance and application versatility.
2.	The proposed method introduces a novel modality contribution score (MCS) mechanism for adaptive steering, effectively resolving limitations of uniform steering strength through sample-specific sensitivity analysis. The functional entropy is interesting to measure the sensitivity of modalities. The data-adaptive steering is also reasonable.
3.	Experiments demonstrate that the AMPS framework significantly improves modality preference shifting while reducing task errors, providing empirical validation for the approach.

### Weaknesses
1.	While the use of modality contribution score (MCS) is innovative and interesting, the detailed intuition and theoretical justification can be further enhanced. This lack of background knowledge (especially in Eq. 3-5) may confuse broader readers. Besides, it would be better to provide a more detailed theoretical analysis or evidence supporting MCS.
2.	The paper compares AMPS with static steering methods but lacks a comparison to recent approaches in modality preference steering. The baselines are introduced unclearly. It would be better to include comparisons with state-of-the-art methods to better highlight the advantages of AMPS.
3.	The experiments primarily use a limited set of datasets (e.g., MC2, Qwen-VL, LLaVA), which may not fully represent the diversity of tasks where modality preference steering is important. It would be better to evaluate the method on a broader range of datasets and include examples that stress-test the method under various real-world conditions. For instance, incorporating datasets that involve noisy or ambiguous inputs could provide further insight into the robustness of AMPS.
4.	The paper lacks ablation studies to assess the impact of individual components of AMPS. It would be helpful to include ablation experiments to understand the contribution of each part of the framework.
5.	The current experimental results fail to sufficiently rule out the possibility of overfitting. It is suggested that supplementary validations be conducted across heterogeneous datasets and multi-scale model architectures to ensure performance improvements are generalizable rather than contingent upon specific training data or model configurations.

Typos：
There are several typos. For example: 
1. Line 185: We-> we
2. Line 198: f -> $f$
3. Line 204: We -> we
4. Line 265: “Previous studies ()” no citations.
By the way, some equations do not have punctuation marks at the end.

### Questions
Please see strengths and weaknesses. Besides:

1.	How does MCS perform on more complex multi-modal tasks?

2.	Why does Eq.(11) apply the scaling factor (1+γ) when integrating Eq.(9) and Eq.(10) to formulate the steering module’s prediction target?

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
2

### Summary
This paper addresses the modality preference in Multimodal LLMs, where models may over-rely on text or visual inputs irrespective of user intention or task requirements. Existing steering methods often apply uniform intervention strengths, which can degrade performance. To address this, the authors propose the Modality Contribution Score (MCS), a diagnostic metric that evaluates the contribution of each modality in a context-sensitive manner. Building on MCS, they introduce AMPS, an adaptive steering framework that dynamically adjusts intervention strength on a per-sample basis via a learnable module. The approach is supported by theoretical grounding (functional entropy, Sobolev inequality), clear algorithmic implementation, and extensive experiments showing improved preference alignment and reduced errors compared to baselines.

### Strengths
1. The paper is well-motivated: The proposal of the Modality Contribution Score (MCS) based on functional entropy and Fisher information is rigorous and well-motivated.
2. Extensive empirical results: The paper provides comprehensive empirical analysis—including comparisons with prompt-based, static steering, and prior adaptive approaches—across multiple model families (LLaVA, Qwen-VL) and sizes. In Table 1 and Table 2, AMPS shows consistently superior performance for controlling preference while minimizing error rates.

### Weaknesses
1. Experiments on more benchmarks are needed: the experiments are executed with the $M C^{2}$ dataset only, and lack evaluation on broader, more real-world multimodal tasks, such as MME, MM-Vet, LLaVA Bench, and MMstar.
2. To avoid a tendency on one modality, the easiest way is to move the tokens or replace the tokens with pad tokens. Have you tried this strategy?
3. Different models and evaluation benchmarks are mixed across Tables 1 and 3, creating confusion and undermining the interpretability of results. 

If possible, the author can reorganize the experiments, employing more advanced benchmarks for evaluation.
If you can provide convincing clarification, I would be open to increasing the score.

### Questions
My main concern is the evaluation strategy; the current evaluation is limited, can not demonstrate the effectiveness of this paper.

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
This paper introduces AMPS, an adaptive steering framework for controlling modality preference in multimodal large language models (MLLMs). The authors first propose the Modality Contribution Score (MCS) as a diagnostic metric with functional entropy and Fisher information to quantify the per-sample reliance on each modality. Leveraging this insight, they design a context-aware, learnable module to apply sample-specific steering intensities, aiming to shift modality preference more effectively than traditional uniform (static) steering. Experimental results demonstrate improved preference adjustment and robustness.

### Strengths
1.	The paper introduces the Modality Contribution Score (MCS), grounded in functional entropy and Fisher information, to provide a rigorous quantification of modality contribution at the sample level.
2.	Instead of the uniform steering of traditional methods, the authors propose the sample-adaptive steering via a scaling coefficient, justified by the diagnostic metric. The inclusion of the learnable module further enables context-sensitive adjustment.
3.	The paper provides a comprehensive evaluation across 2 model families and scales, and the results show consistent improvements over previous baselines.

### Weaknesses
1.	The context-aware scaling factor $\gamma$ (Equation 9, Page 6) is constructed as a linear deviation from an anchor ratio, modulated by $\beta$. It seems somewhat heuristic. A more detailed justification for why this specific formula is the right way to quantify "severity of preference" would strengthen the method.
2.	The MCS measurement requires multiple forward passes with KV-cache perturbations for a single input. The computational cost of this diagnostic process is not discussed.
3.	It’s better for the authors to take more recent and highly relevant works and benchmarks on modality preference steering into comparison.

### Questions
1.	The MCS metric is central to the method. Did the authors explore alternative ways to quantify modality contribution or susceptibility to steering?
2.	Will the authors release code, data splits, and detailed hyperparameter configuration to facilitate reproducibility?

### Soundness
3

### Presentation
2

### Contribution
3
