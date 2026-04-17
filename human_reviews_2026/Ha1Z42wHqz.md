# Uncovering Activation Keys in the Dark: Revealing Learned Concepts in LoRA Text-To-Image Models

- Decision: Reject
- Scores: 6, 4, 4, 2, 4

## Abstract
Low-Rank Adaptation (LoRA) has become a widely adopted technique for customizing large diffusion models, enabling users to inject new styles, characters, or identities into text-to-image generation with minimal computational cost. While this flexibility fuels creative expression, it also opens the door for injecting sensitive or potentially harmful content, such as political figures’ faces, copyrighted characters, or explicit imagery, into generative models. These LoRA adapters are often distributed without documentation, making it difficult to identify the concepts they encode or understand how they are triggered. This lack of transparency poses serious challenges for moderation, accountability, and large-scale content auditing in open-source model ecosystems. To address this risk, we adopt the role of a model investigator and introduce the LoRA ``activation key'' discovery problem: given a suspect LoRA and its base model, identify a text embedding that reliably activates behaviors unique to the LoRA. This activation key serves as a forensic probe to reveal hidden concepts introduced during fine-tuning. To achieve so, we propose a two-stage optimization framework. We first perform an evolutionary search in the token space to identify promising candidate prompts, followed by gradient-based refinement in the embedding space. Our objective encourages the LoRA model to generate concentrated outputs while maximizing divergence from the base model, resulting in an embedding that reveals distinct LoRA-specific behaviors. Experiments on six public LoRA adapters show that our method recovers ground-truth concepts in both white-box and black-box settings. Our work demonstrates the feasibility of LoRA forensics and highlights the need for auditing tools in open-source model ecosystems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the problem of uncovering hidden concepts in LoRA-fine-tuned diffusion models. It introduces a contrastive objective with three loss terms—intra-LoRA consistency, intra-base diversity, and inter-model dissimilarity—and a two-stage optimization framework to recover activation keys. Stage 1 conducts a black-box evolutionary search over discrete prompts, while Stage 2 refines the discovered embedding through gradient ascent. Experiments on several public LoRA adapters with Stable Diffusion 1.5 / SDXL show that the method performs effectively within the tested settings.

### Strengths
I genuinely like the paper’s methodological design — it is well-motivated, logically structured, and technically sound.

1. The paper is clearly written and easy to follow, with well-organized sections and informative visualizations.

2. The three-term objective (intra-LoRA consistency, intra-base diversity, and inter-model dissimilarity) is intuitive, elegant, and well grounded in the goal of contrasting LoRA and base behaviors.

3. The two-stage design—evolutionary search followed by gradient-based refinement—is a smart and practical solution to the difficulty of optimizing discrete prompts.

### Weaknesses
While the paper is well written and conceptually solid, the evaluation part feels somewhat limited.

1. The experimental scale is relatively small—only six LoRA adapters and two base models (SD 1.5 and SDXL). Expanding to more diverse concepts or datasets, such as the DreamBooth dataset (might need to incorporate some automatic captioning process) or larger community LoRA collections, would better demonstrate robustness and generality.

2. The current comparison baseline (a random or heuristic prompt) is weak. Incorporating stronger baselines such as prompt inversion or optimization methods  would make the empirical claims more convincing.

3. The three loss components are central to the method, yet no ablation study or sensitivity analysis is provided. Evaluating the effect of removing or reweighting each term would clarify their relative importance.

4. It would be valuable to test the approach on newer diffusion architectures (e.g., SD 3 or Flux) to assess whether the mechanism generalizes.

Overall, the proposed method is insightful, and if its with stronger empirical validation, I would lean toward acceptance.

### Questions
Apart from the weaknesses mentioned above, I also wonder:


Could the proposed framework be extended to more general fine-tuning settings, such as full-parameter DreamBooth?

### Soundness
4

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
5

### Summary
This paper introduces the **LoRA activation key discovery problem**, a novel forensic approach for identifying hidden or undocumented concepts within T2I LoRA. Motivated by concerns over moderation and accountability in open-source generative ecosystems, the authors propose a **two-stage optimization framework** consisting of (1) **evolutionary search** in token space and (2) **gradient-based refinement** in embedding space. The objective maximizes behavioral divergence between the LoRA and base models using CLIP-based inter/intra-model similarity measures. Experiments on six public LoRA adapters show that the method effectively recovers ground-truth triggers and reveals distinct LoRA-specific behaviors, verified quantitatively (TrigSim, CapSim, CMMD) and semantically via VLM analysis.

### Strengths
- **Novelty:** Defines the previously unexplored area of activation key discovery in LoRA models.

- **Technical Depth:** Well-founded objective and two-stage optimization pipeline, combining discrete and continuous strategies.

- **Empirical Breadth:** Evaluation on multiple LoRAs with robust metrics and semantic validation.

- **Relevance:** Addresses growing safety, transparency, and forensic auditing challenges in T2I ecosystem.

- **Presentation:** Strong writing, clear figures, and reproducible experiments.

### Weaknesses
- **Scalability:** The procedure’s computational cost and time may hinder large-scale deployment.

- **Limited Soundness:** The author present that "Our approach is model-agnostic and applicable in both white-box and black-box settings" while actually the proposed method is a two-stage method that relies on the initialization of a black-box token-level optimization for a white-box optimization, which is a white-box method. Results in Fig.5 and Fig.6 are not enough to demonstrate the proposed method is effective in white-box and black-box scenarios (Fig.6 should be organized as only stage1, only stage2 with random initialzed embedding and stage1+stage2 to demonstrate the improvement). 

- **Confusing Objectives:** The objective function of "Consistency within the LoRA", "Diversity within the base model" and "Discrepancy across models" is confusing without any rationale. First, some LoRAs are highly semantic-align with their corresponding trigger words which means that the LoRAs can actually replace by a prompt suffix, disaligning with the objective "Diversity within the base model". Second, why should there be "Discrepancy across models" if the LoRAs themselves do not introduce large semantic change to the base LoRA. There are all kinds of LoRAs, some of which might only be used for fine-grained adjustments. Third, "Consistency within the LoRA" is also not grounded, since for style-based LoRAs, CLIP might not capable to extract these features. I recommand the author clarify the audit scope (what types of LoRA) of their method.

- **Ablation Study:** There are no ablation studies for the proposed three objectives and I wonder if they really contribute to the optimization for the above mentioned scenarios.

- **Lack of Multi-task Discovering:** An adaptive attacker can easy design a multi-task LoRAs evasion attack by hidden a malicous task with complex trigger words into a benign task with simple trigger words. The benign task acts as a trapdoor to hijack you optimization.

### Questions
Refer to **weakness**. The proposed scenario for LoRA auditing is promising and if the author can address some of my concerns above, I am willing to raise my rating anytime.

### Soundness
2

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
4

### Summary
This paper addresses LoRA adapter auditing by discovering "activation keys", a concept defined for text embeddings that expose hidden concepts in fine-tuned diffusion models. A two-stage framework is proposed, combining evolutionary search over discrete tokens (stage 1) with gradient-based refinement in continuous embedding space (stage 2). The objective function balances intra-model dispersion (LoRA outputs should be consistent) against inter-model similarity (LoRA should differ from base model). Experiments on six publicly available LoRAs are solid, demonstrating successful concept recovery, though with notable computational costs and ~22% failure rate.
This work aligns well with ICLR's core themes. Model auditing and interpretability are increasingly critical as generative models proliferate across the fields. The paper combines optimization theory, computer vision, and ML security in ways that should interest the ICLR community. The focus on LoRA adapters is particularly timely given their widespread deployment as a powerful PEFT method.

### Strengths
1. Important Problem: Auditing undocumented LoRA adapters matters. The community shares countless fine-tuned models, and yet many are without proper documentation. Having systematic ways to discover what they encode is genuinely useful.
2. Clean Problem Formulation: The activation key concept is well-defined in a straightforward manner. The objective function combining intra-model dispersion and inter-model similarity is intuitive and principled.
3. Practical Two-Stage Design: Starting with evolutionary search for coarse exploration, then refining with gradients makes sense. Stage 1 works in black-box settings; Stage 2 works with white-box access when available. This flexibility could be valuable.
4. Solid Empirical Validation: Testing on both stylistic and identity-based LoRAs shows generality. The quantitative metrics (CMMD, CLIP similarity) combined with qualitative VLM analysis provide multiple perspectives.

### Weaknesses
1. Objective Function Lacks Theoretical Grounding: Why should maximizing intra-LoRA dispersion while minimizing inter-model similarity necessarily recover the true concept? The paper doesn't provide a thorough theoretical justification. What guarantees exist that this objective aligns with finding semantically meaningful triggers rather than adversarial perturbations?
2. Limited Baseline Comparisons: The random prompt baseline is not a very strong case to be based upon. Why not compare against existing LoRA auditing methods mentioned in related work (Yao 2024's weight leakage, membership inference approaches)? Without stronger baselines, it's hard to assess whether the complexity of the two-stage framework is justified.
3. Computational Cost Is Prohibitive: Each experiment requires generating hundreds to thousands of images. On an A100, Stage 1 needs roughly 10×n×G images, Stage 2 adds about 1500 more. For practical auditing at scale, this cost seems problematic. The authors mention parallelization could help, but don't provide concrete timing comparisons or discuss computational efficiency as a design consideration.
4. Failure Rate Concerns: Combined ~22% failure rate across stages is perhaps not an ignorable number. The explanation attributes this to "initial random seeds in Stage 1," but this seems fixable. Why not run multiple Stage 1 initializations in parallel and select the best? The dependence on good initialization suggests the objective landscape could be better understood/designed.
5. Limited Discussion of False Positives: Can this method be fooled? What if someone deliberately creates a LoRA that appears benign under this auditing approach? The adversarial robustness of the framework isn't explored.
6. Evaluation Metrics Could Be Stronger: CMMD and CLIP similarity are reasonable but indirect. For identity-based LoRAs, why not consider adding face verification type of scores? For style LoRAs, perceptual metrics like LPIPS might be more appropriate. The VLM captioning is interesting, but perhaps can only serve as a qualitative (subjective) assistant judge.

### Questions
1. Can you provide a theoretical analysis showing your objective function provably recovers ground-truth concepts under reasonable assumptions?
2. What happens if you run multiple Stage 1 initializations? Does this reduce the failure rate proportionally, or are some LoRAs fundamentally harder to audit?
3. Have you considered testing adversarial scenarios where someone actively tries to evade your auditing method?
4. The objective function feels somewhat ad hoc. The authors don't justify the specific formulation of these terms beyond intuition. Why is Euclidean distance in CLIP space the right metric? Have you tried other divergence measures? Furthermore, the weighting parameters α and β are mentioned, but their values aren't specified. How sensitive are results to these choices? Ablation studies would strengthen this section. In addition, can you characterize when your objective succeeds? Even a toy model showing why dispersion+dissimilarity recovers concepts would help.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper focuses on the auditing challenges of undocumented LoRA adapters in text-to-image diffusion models, which can be used to inject sensitive or harmful concepts without disclosure. The authors introduce Activation Key, which is generated through a two-stage search framework, to reveal the concept only embedded in the LoRA adapter. Experiments on six public LoRA adapters show that the method effectively recovers the hidden concepts. Besides, the authors visualize the optimization trajectory with t-SNE, demonstrating how their method progressively creates a clear separation between the LoRA and base diffusion model.

### Strengths
+ Important research motivation.
+ Reasonable design of the auditing method.

### Weaknesses
- Poor writing and unclear methodology introduction.
- White-box method design is inconsistent with contributions.
- Insufficient experiment settings and evaluation.

### Questions
1. According to “Related Work”, one of your contributions is that the method is applicable in both white-box and black-box settings, while the stage-2 needs white-box access as demonstrated in subsec 6.2.2.

2. Many details of your method are difficult to understand and lack thorough introductions. What is the target to maintain the diversity within the base model in the training objective? How to achieve the token-level score $s_t$ of the prompt text based on $f(p)$, that is generated by image embedding?

3. The writing employs some non-standard terms and contains several incomplete sentences. For example, you use “spread” to denote $S_M$ without a detailed explanation; The second sentence in subsect 5.2 ends suddenly and unnaturally.

4. The experimental design lacks persuasiveness. You implement experiments only in six LoRA adapters without the large-scale datasets evaluation. Besides, the baselines are too limited and the effectiveness of “random prompt” is too weak as the baseline.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the critical and underexplored problem of auditing undocumented Low-Rank Adaptation (LoRA) models. The authors formalize this as the "LoRA activation key discovery" task, where the goal is to find a text embedding that reliably triggers a LoRA's specific, fine-tuned behavior while remaining inert for the base model. To solve this, they propose a two-stage optimization framework: a black-box evolutionary search to find a promising initial prompt, followed by a white-box, gradient-based refinement of its embedding. The search is guided by a novel objective function designed to maximize the behavioral divergence between the LoRA and its base model. Experiments on six publicly available LoRA models show that the method can successfully recover the intended concepts.

### Strengths
1. Important Problem Formulation: The paper identifies and formalizes a critical, real-world problem concerning the safety, accountability, and auditing of community-shared generative models. This is a significant contribution to the responsible AI ecosystem.

2. Elegant Objective Function: The objective function, based on maximizing intra-LoRA consistency while minimizing inter-model similarity, is a principled and intelligent way to define the desired characteristics of an "activation key."

3. Principled Hybrid Search: The two-stage framework, combining evolutionary search for broad exploration and gradient-based methods for fine-tuning, is a strong and logical approach to the complex, hybrid search space.

### Weaknesses
1. Crucial Mismatch in Experimental Validation: The most significant weakness is that the experiments do not validate the method's utility for its stated purpose. The paper claims to uncover concepts "in the dark," but it is only tested on LoRAs with publicly known, non-adversarial triggers. This fails to demonstrate that the method can handle intentionally obfuscated, non-semantic, or compositional triggers that would be used in malicious LoRAs.

2. Insufficient Experimental Scale: The evaluation is conducted on only six LoRA models. While diverse in type, this small sample size is insufficient to make strong claims about the method's generalizability across the vast and heterogeneous landscape of community-trained LoRAs.

3. Practicality and Scalability Concerns: The proposed method, particularly the evolutionary search stage, is computationally expensive, requiring thousands of model inferences to audit a single LoRA. This raises serious questions about its feasibility for deployment at the scale required by model-sharing platforms.

### Questions
1. The core claim of the paper is to uncover hidden concepts, but the experiments were performed on LoRAs with public, semantically meaningful triggers. Could the authors provide evidence of their method's performance on a LoRA trained with a deliberately obfuscated or non-semantic trigger (e.g., a random string) to better support the central thesis?

2. Given the high computational cost, how do the authors envision this framework being practically deployed for large-scale auditing of thousands of models? Are there opportunities to significantly reduce the cost of the Stage 1 search?

3. The optimal performance relies on Stage 2, which requires white-box access. In a more realistic black-box (API-only) auditing scenario, what is the performance of the Stage 1 evolutionary search alone, and is it sufficient for reliable concept discovery?

### Soundness
2

### Presentation
3

### Contribution
2
