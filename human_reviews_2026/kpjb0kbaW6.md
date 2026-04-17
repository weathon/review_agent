# Explicit Representation Alignment via Subspace Elimination for Robust LLM Unlearning

- Decision: Reject
- Scores: 4, 8, 6, 4

## Abstract
Large Language Models often retain sensitive or hazardous knowledge that must be suppressed without compromising their general linguistic abilities. However, existing unlearning methods are often unstable, sensitive to hyperparameter choices, and fail to generalize across knowledge types. 
We introduce ERASER, a principled framework for targeted unlearning. It combines a subspace-based target construction with an auxiliary ranking objective that enforces separation between forget and retain domains, thereby achieving stable and effective unlearning.
Beyond existing evaluations, we conduct thorough experiments as follows: 
(i) ERASER achieves state-of-the-art unlearning effectiveness while preserving general knowledge on existing benchmark datasets, 
(ii) it removes knowledge not only at the surface level but also at deeper semantic and compositional levels using the Fictional Knowledge dataset, 
and (iii) it demonstrates strong robustness against adversarial threats, including jailbreak, membership inference, and relearning attacks. 
These results establish ERASER as a practical framework for safe LLM unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ERASER (Explicit Representation Alignment via Subspace Elimination for Robust LLM Unlearning), a principled framework for stable and effective unlearning in large language models. ERASER identifies harmful directions in the representation space and nullifies them via a Subspace-based Target Vector, eliminating the need for heuristic tuning. It further introduces a Disentangle-Head, a ranking-based auxiliary module that separates forget and retain representations, improving convergence and stability.

### Strengths
1.  Compared with RMU’s random target vector, the Subspace-based Target Vector offers a clear advantage: It replaces heuristic randomness with a data-driven forgetting subspace, enabling more stable and interpretable unlearning across diverse datasets and models.

2.  Experiments on multiple LLMs show state-of-the-art forgetting performance with minimal loss in general ability and strong robustness against jailbreak, membership inference, and relearning attacks.

### Weaknesses
1.  A limitation of the Disentangle-Head is that it relies on both forget and retain samples, which may constrain its applicability when clear domain separation or labeled retain data is unavailable.

2.  The proposed method heavily depends on the RMU framework and is difficult to generalize to other unlearning paradigms such as gradient ascent or negative preference optimization.

3.  The evaluation is limited to the WMDP benchmark, lacking validation on other unlearning datasets such as TOFU, MUSE, and RWKU, which raises concerns about the generality of the results.

### Questions
Please see the above Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents ERASER, a representation-based LLM unlearning framework that identifies and neutralizes harmful directions in forget representations, building on RMU (Li et al., 2024). Two main ideas are introduced: first, by creating a shared forgetting subspace using domain-specific principal components and projecting forget-sample activations away from this subspace, ERASER stabilizes unlearning without relying on heuristics. Second, it introduces a Disentangle-Head auxiliary ranking loss that speeds up the separation between forget and retain domains. Extensive experiments demonstrate state-of-the-art forgetting on benchmarks, more thorough semantic erasure of fictional knowledge, and improved robustness against jailbreak, membership inference, and relearning attacks.

### Strengths
- Two novel ideas (shared forgetting subspace and Disentangle-Head loss) are introduced to enhance representation-based LLM unlearning. These ideas are well-motivated both empirically and logically.
- Extensive experiments, including comparisons with existing methods using four types of LLMs on standard datasets in LLM unlearning, along with ablation studies, demonstrate the effectiveness of ERASER across various benchmarks and attack scenarios. The ablation studies convincingly show the contribution of two components to the overall performance.

### Weaknesses
- One natural variation of the proposed method could be explored. Basically, the paper's idea is to push representations away from certain directions (forgetting subspace), which is estimated from forget samples from all domains. Another natural idea is to leverage a domain-wise forgetting subspace, i.e., estimating a forgetting subspace for each domain separately (use $\mathbf{Z}_j$ directly?) and pushing representations away from the corresponding domain-specific forgetting subspace. It would be interesting to see how this variation performs compared to the proposed method. If the reviewer understands correctly, the number of forgotten domains in the experiments is only two (because it has WMDP-Bio and WMDP-Cyber), so this variation would be feasible (and theoretically, it should work better?).
- While RMU collapses all the forget representations into a single (random) target vector, ERASER projects representations away from a forgetting subspace, which means that forget representations will be dispersed in the representation space. This creates a significant difference in the unlearning objective compared to RMU, and it would strengthen the paper if the authors could provide more analysis of this difference, both theoretically and empirically, to better understand the effectiveness (and possible downside) of the proposed approach.
- Some might view this work as incremental since it builds on RMU and adjusts target vectors. However, the reviewer believes the contributions are substantial because there has been no principled method to determine target vectors, which is a key limitation of RMU. Overcoming this limitation is a significant advancement.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ERASER, a representation-guided unlearning framework for large language models (LLMs), designed to effectively and stably remove harmful knowledge from models while preserving their general language capabilities. The core innovations of ERASER include:1.A Subspace-based Target Vector construction, which identifies and nullifies harmful directions in the representation space.2.An auxiliary module, the Disentangle-Head, which promotes the separation of representations between the forget and retain domains via a ranking loss.
The authors validate ERASER's effectiveness on multiple base models (Zephyr, Qwen, Llama, Yi) and conduct a comprehensive evaluation covering unlearning efficacy, semantic-level forgetting, and robustness against adversarial attacks. The results demonstrate that ERASER achieves the best balance between targeted forgetting and knowledge retention.

### Strengths
1.The proposed data-driven approach for constructing the target vector via subspaces avoids the limitations of existing methods that rely on heuristic targets, significantly enhancing the stability and generalizability of unlearning.
2. Beyond traditional unlearning assessments, the paper provides the first systematic evaluation of model robustness at semantic/compositional levels and against various adversarial threats (e.g., Jailbreak, Membership Inference, Relearning attacks), validating the method's practical utility.
3.The Disentangle-Head is demonstrated to be a general-purpose plugin that can improve the performance of other unlearning methods (e.g., RMU), enhancing the method's versatility and practical value.

### Weaknesses
1.While the Fictional Knowledge dataset is useful for evaluating deep semantic forgetting, there might be a gap between it and the complex, implicit harmful knowledge in the real world. The method's generalization needs further verification in more realistic and diverse harmful knowledge scenarios.
2. Although the parameter efficiency advantage of ERASER is mentioned, the computational cost of the offline subspace construction phase is not analyzed in detail, especially regarding its scalability in large-scale, multi-domain forgetting scenarios.
3.Although the method performs excellently in experiments, there is a lack of theoretical or interpretability analysis regarding "to what extent forgetting is considered successful," making it difficult to assess the risk of over-forgetting or under-forgetting.

### Questions
1.The construction of ERASER's shared forgetting subspace relies on a predefined number of forget domains N and principal components k. In practical applications where forget samples come from unknown or dynamically changing domains, how can the method adaptively determine these hyperparameters?
2.In the adversarial evaluation, ERASER shows robustness against Relearning Attacks. However, if an adversary uses a larger-scale or more representative subset of the forget data for retraining, would the model's unlearning effectiveness degrade significantly?

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
4

### Summary
The paper proposes ERASER, a representation-guided unlearning method for LLMs. It (i) computes a Shared Forgetting Subspace by aggregating domain-wise principal directions of forget sets and (ii) performs projection–nullification during training to remove those components from hidden states. An auxiliary Disentangle-Head adds a pairwise ranking loss to separate retain/forget representations and speed convergence. Empirically, ERASER improves ΔAcc over baselines across several base models and reports robustness to jailbreak, membership inference, and relearning attacks.

### Strengths
- **Principled target construction.** The subspace approach replaces heuristic/random target vectors, reducing sensitivity to hand-tuned coefficients.
- **Clear mechanics & reproducibility.** The three-phase pipeline and training objective are explicit, with algorithmic detail in the appendix.
- **Broad empirical scope.** Multi-model comparisons and robustness evaluations (GCG, embedding-space attacks, MIAs, relearning).

### Weaknesses
**W1. Baseline Reproducibility Issue (Adaptive RMU).** The reported Adaptive RMU[1]  performance diverges from the numbers in its original paper. The authors acknowledge variability but fail to reconcile the cause or show matched configurations. This undermines fairness in comparison and raises questions about the experimental setup consistency.

**W2. Missing optimization based Baselines.** The study omits comparisons to recent finetuning-style unlearning methods (e.g., Sim**NPO[2]**, **Gradient Ascent**). These are strong, widely-used baselines for LLM unlearning; excluding them weakens the empirical claim of superiority.

**W3. Limited Utility Evaluation.** The paper reports mainly **MMLU** results, lacking broader coverage of reasoning tasks (e.g., **GSM8K**, **MATH**, **BBH**) or multi-turn dialogue evaluations such as **MT-Bench**. Since unlearning often impacts reasoning ability, broader utility tests are necessary.

**W4. Insufficient Hyperparameter Sensitivity Analysis.** The paper reports fixed values (e.g., α = 1200) without explaining sensitivity. The wide gap between α for ERASER and for RMU (1–5) demands justification. Also, I did not find the definition of β.

**W5. Limited Exploration of Alternative Subspace Methods.** The method relies solely on PCA/SVD to define harmful directions. Variants like weighted PCA, robust PCA, or LEACE-style concept erasure are not tested, limiting generality.

**W6.  Ambiguous Relation to Existing Paradigms.** Conceptually, I see ERASER resembles “null-space optimization + DPO-style ranking”, which is actually common in knowledge editing [3]. The paper should clarify conceptual distinctions and empirical advantages over these established approaches. Especially explaining why do you need the auxilary loss.



[1] On Effects of Steering Latent Representation for Large Language Model Unlearning

[2]  Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning

[3] AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models

### Questions
**Q1. Unexplained Data Efficiency.** Why ERASER achieves similar performance using only 10 % of the forget set as with 100 %. This strong claim lacks analysis. It remains unclear whether the model genuinely generalizes from a small forget sample or simply overfits shared latent directions?

**Q2. Potential Over-forgetting due to Subspace Overlap.** If retain and forget domains overlap, does projection–nullification inadvertently erase useful features? Can you quantify retention degradation under controlled overlap?

**Q3. Attack Surface.** Could the shared subspace concentration make ERASER more predictable or vulnerable? Have you visualized the principal directions to verify domain diversity? Are jailbreak defenses mainly due to parameter updates or orthogonalization? Could adaptive attackers exploit complementary subspaces?

**Q4. Concept-level and Large-scale Forgetting.** Would ERASER remain effective on **book-scale (MUSE)** or **concept-level (RWKU)** unlearning tasks where harmful content spans broader contexts?

**Q5. Optimization Claim for joint forget/retain optimization.** You state that joint forget/retain optimization stability is an open problem `L106-L107`, yet you still jointly optimize. What empirical evidence (e.g., convergence variance, loss curvature) supports ERASER’s improved stability?

**Q6. The Ablation Study Discrepancies.** Why do Table 11 and 12 ablation results differ from the main experiments? Are the datasets or hyperparameters identical?

**Q7. Utility Drop Explanation In Appendix C3.** Why does full fine-tuning reduce general capabilities so drastically? Can you link this to your subspace framework (e.g., low-rank vs. full-rank drift)? It would be interesting if you have explanation about this phenomenon.

**Q8. Alternatives for the disentangle-head and subspace construction.** 

- Would a simpler contrastive or orthogonal-loss baseline achieve similar disentanglement without the Disentangle-Head?
- If PCA were replaced by a weighted or robust variant, or if the subspace were built using mutual-information weighting instead of SVD, how would performance change?

### Soundness
2

### Presentation
3

### Contribution
2
