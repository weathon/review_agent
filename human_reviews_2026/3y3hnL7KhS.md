# AEGIS: Adversarial Target-Guided Retention-Data-Free Robust Concept Erasure from Diffusion Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4, 4

## Abstract
Concept erasure helps stop diffusion models (DMs) from generating harmful content; but current methods face robustness-retention trade-off. **Robustness** means the model fine-tuned by concept erasure methods resists reactivation of erased concepts, even under semantically related prompts. **Retention** means unrelated concepts are preserved so the model’s overall utility stays intact. Both are critical for concept erasure in practice, yet addressing them simultaneously is challenging, as existing works typically improve one factor while sacrificing the other. Prior work typically strengthens one while degrading the other—e.g., mapping a single erased prompt to a fixed safe target leaves class-level remnants exploitable by prompt attacks, whereas retention-oriented schemes underperform against adaptive adversaries.  This paper introduces Adversarial Erasure with Gradient-Informed Synergy (AEGIS), a retention-data-free framework that advances both robustness and retention. First, AEGIS replaces handpicked targets with an Adversarial Erasure Target (AET) optimized to approximate the semantic center of the erased concept class. By aligning the model’s prediction on the erased prompt to an AET-derived target in the shared text–image space, AEGIS increases predicted-noise distances not just for the instance but for semantically related variants, substantially hardening the DMs against state-of-the-art adversarial prompt attacks. Second, AEGIS preserves utility without auxiliary data via Gradient Regularization Projection (GRP), a conflict-aware gradient rectification that selectively projects away the destructive component of the retention update only when it opposes the erasure direction. This directional, data-free projection mitigates interference between erasure and retention, avoiding dataset bias and accidental relearning. Extensive experiments show that AEGIS markedly reduces attack success rates across various concepts while maintaining or improving FID/CLIP versus advanced baselines, effectively pushing beyond the prevailing robustness–retention trade-off. The source code is in https://github.com/Feng-peng-Li/AEGIS.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
Concept erasure helps stop diffusion models (DMs) from generating harmful content; but current methods face robustness–retention trade-off. This paper introduces a novel method, Adversarial Erasure with Gradient-Informed Synergy (AEGIS), aimed at enhancing the robustness of concept erasure under adversarial prompt attacks (APAs). Extensive experiments show that AEGIS markedly reduces attack success rates across various concepts while maintaining or improving FID/CLIP versus advanced baselines.

### Strengths
1. The paper conducts extensive experiments and demonstrates significant improvements over existing methods.
2. The paper provides both empirical and theoretical support for the proposed method.

### Weaknesses
1. Placing the related work section in the main body of the paper would improve readability. In addition, it is difficult to understand some details when reading only the main text.
2. Some fine-tuning details are missing, such as the amount of data used and the required memory.
3. Are all the results in Table 10 obtained using 8 × 80GB H800 GPUs?
4. Typographical error: “¿” in line 1432.


Overall, this paper is of good quality. However, since I am not very familiar with this research field, I set my confidence level to 1.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed retention-data-free framework for the robust concept erasure of diffusion models. To be more specific, the adversarial erasure target (AET) is obtained through optimization instead of handpicking and retention data is avoided by finetuning with gradient regularization projection (GRP) to achieve better trade-off between robustness and retention.

### Strengths
1. Well-formulated motivation: This paper identifies a key weakness in existing concept erasure methods, which overrely on the single-instance erasure targets.
2. Retention-data-free: It is necessary to avoid the usage of additional datasets to maintain the model utility which might involve hidden bias.
3. Extensive experiments are convincing: This paper evaluates multiple baselines through three adversarial prompt attacks.
4. Good ablation study to isolate the effect of AET, PR and DGR.

### Weaknesses
1. The integration of AER and GRP is novel, however, both ideas are drawn from well-known paradigms: adversarial target optimization and gradient surgery. It is good to further clarify the differences compared to existing paradigms.
2. Insufficient analysis on computational overhead. Computation reduction brought by AET is mentioned in the paper, however, the paper lacks detailed computation 
3. Evaluation scope somewhat narrow. Only three representative concepts (nudity, Van Gogh, Church) are analyzed.
4. Limited interpretation of semantic centers. Visualization or embedding analysis of AET would be beneficial.

### Questions
1. How stable is the AET optimization process? 
2. Scalability to large models (e.g., SDXL): The experiments focus on SD v1.4/v2.1. Have the authors attempted to extend AEGIS to higher-capacity models?

### Soundness
3

### Presentation
3

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
This paper AEGIS proposes a robust defense algorithm against adversarial attack on concept-erased diffusion models. It shows the trade-off between erasure robustness and quality preservation, by Adversarial Erasure Target and Gradient Regularization Projection separately. AET approximates the semantic center of the concept to be erased, enables class-level removal instead of single prompt. GRP preserves the quality using parameter regularization and a novel gradient surgery technique that selectively projects away retention gradients conflicting with the erasure objective. Author's experiments show AEGIS significantly reduces attack success rates by 5.31~24% across various concepts and state-of-the-art attacks like P4D and UnlearnDiffAtk, while maintaining or improving image quality over baselines.

### Strengths
1. Instead of defending a single prompt, AEGIS dynamically optimizes a target prompt to approximate the semantic center of the concept being erased. With both AEGIS and GRP, it claims to achieve better tradeoff and supported by experiment results.
2. Experiment is thorough - it validates its method across multiple concept types (object, style, nudity), model versions (SD v1.4, v2.1), and against a suite of strong adversarial attacks (P4D, UnlearnDiffAtk), proving its generalizability and robustness.

### Weaknesses
1. it looks like it's sensitive to hyper-parameters such as w in 5.3 ablation study. how to pick the best value for unlearning a new concept?
2. how to scale if the model needs to unlearn many concepts or objects?

### Questions
1. what about attack in the input image or even embeddings?

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
The paper addresses the problem of robust concept erasure in diffusion models. The paper proposes Adversarial Erasure with Gradient-Informed Synergy (AEGIS), improving erasure robustness and retain performance after unlearning.

### Strengths
The robustness of diffusion model unlearning is a highly important problem. The idea of adversarial erasure target (AET) is novel and well-motivated, and the authors provide detailed and solid explanations for their proposed methods.

The robustness of AEGIS is validated on multiple attacks. The authors also compare AEGIS with multiple baselines.

The figures and illustrations are of good quality.

### Weaknesses
1. The paper lacks comprehensive evaluations on the retain performance. Currently, FID and the CLIP score are used. However, common DM unlearning benchmarks such as UnlearnCanvas [1] include evaluation metrics such as in-domain retain accuracy (IRA) and cross-domain retain accuracy (CRA). Since the authors claim AEGIS has great robustness–retention trade-off, a more comprehensive retention evaluation is needed.

2. The motivation of Parameter Regularization (PR) and Directional Gradient Rectification (DGR) seems unclear. It seems that AET alone can achieve concept erasure while retaining model utility, and PR and DGR can be added upon any unlearning methods, and they are not specifically related to AET. Possibly, AET seriously degrades the model's utility, so PR and DGR are employed to balance retention performance. However, this motivation still seems weak: why not try other methods that do not incur additional computation costs, such as tuning the retain loss coefficient, or adjusting the learning rate?

3. For the baselines, the paper did not include more recent methods on robust concept erasure, such as STEREO [2]. This CVPR 2025 paper addresses the same problem as yours, and I think it should be mentioned and compared.

4. The paper lacks run-time and GPU memory comparison between different methods. I am concerned about the increased computation cost brought by AEGIS, particularly the DGR part. Could the authors discuss this possible trade-off?

[1] UNLEARNCANVAS: A Stylized Image Dataset for Enhanced Machine Unlearning Evaluation in Diffusion Models

[2] STEREO: A Two-Stage Framework for Adversarially Robust Concept Erasing from Text-to-Image Diffusion Models

### Questions
1. Can the authors explain why they employ the CLIP score to evaluate retention? As shown in Table 2-3, the CLIP score has very little changes before and after unlearning. Besides, all the unlearning methods have similar CLIP scores (ranging from 0.29 to 0.31). This gives me the impression that the CLIP score is slightly affected by the unlearning process. In this case, how can it serve to faithfully evaluate the retention performance of different methods?

2. Why and how does Directional Gradient Rectification (DGR) contribute to robustness? In Table 6, 'AEGIS w/o DGR' has significantly higher ASR compared to AEGIS. This result is somehow confusing to me: in theory, DGR serves to improve retention performance by resolving the confliction between forget gradient and retain gradient. However, the authors did not explain how it contributes to the robustness gain.

3. For DGR, have the authors tried using the moving average of gradients in the gradient projection process? This might yield better performances, according to [1].

[1] GRU: Mitigating the Trade-off between Unlearning and Retention for LLMs

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a framework called Adversarial Erasure with Gradient-Informed Synergy (AEGIS) to improve the robustness of concept erasure in diffusion models against adversarial prompt attacks (APAs).

The paper demonstrates that the vulnerability of concept erasure is closely tied to the choice of the target concept (i.e., the concept that the to-be-erased concept is mapped to). If the target is semantically close to the to-be-erased concept, the erasure performance degrades.
To address this, the paper proposes the AEGIS framework with two main components:

•	Adversarial Erasure Target (AET): Guides the erasure by selecting a target concept that is semantically close to the original, while also maximizing the output difference between the old and new models on the same concept.

•	Gradient Projection: Mitigates gradient conflict between the erasing and preserving tasks

### Strengths
-	The problem of machine unlearning is an emerging and important area in the machine learning community.
-	The paper focuses on an important sub-problem — robustness in unlearning, which is gaining increasing attention.
-	The experimental setup appears comprehensive, and the results are promising

### Weaknesses
There are several concerns about the paper’s novelty. More specifically:

•	The first contribution—"the vulnerability of concept erasure stems from an inappropriately chosen learning target. In particular, if the target lies too close to the semantic center – formed by words semantically related to the erased concept – the concept information cannot be fully removed"—has already been studied in prior work [AGE, 1]. Specifically, AGE (Section 4) showed that the choice of the target concept significantly affects both erasing and retaining performance. AGE further suggests that a good target should be semantically related to, but not similar to, the to-be-erased concept—an insight that is more general and comprehensive than what is presented in this paper.

•	The proposed min-max optimization in Equation 7 is very similar to that in AGE, with the only difference being the retention loss (regularization). In AGE, the preservation loss measures the output difference between the new and old models on the same input concept. While this paper propose to minimize the change of model parameter

•	The idea of using gradient projections to mitigate conflict between erasing and preserving tasks has already been proposed in several works [2, 3, 4]. Yet, the paper lacks any discussion or comparison with these related methods.

1: Fantastic Targets for Concept Erasure in Diffusion Models and Where To Find Them

2: Erasediff: Erasing data influence in diffusion models

3: Scissorhands: Scrub Data Influence via Connection Sensitivity in Networks

4: GDR-GMA: Machine Unlearning via Direction-Rectified and Magnitude-Adjusted Gradients

### Questions
•  Could the authors provide a discussion on how their proposed method differs from previous works such as [1, 2]?

•  Given that the core claim of the paper is improved robustness against adversarial or recovery attacks, could the authors include experiments using the recent Random Probe recovery attack proposed in [4], which perturbs the text encoder to confuse generation and recover unlearned concepts

[5] Lu, Kevin, et al. "When Are Concepts Erased From Diffusion Models?." NeurIPS 2025

### Soundness
2

### Presentation
2

### Contribution
2
