# GUARD: Gold-Unchanged Anchored Distillation for Defending LLMs Against Membership Inference Attacks

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large language models (LLMs) are widely fine-tuned for many domain-specific tasks that often contain sensitive and private data. This heightens the risk of membership inference attacks (MIAs), which aim to infer whether a particular sample appeared in training. Prior work has developed increasingly strong MIAs for fine-tuned LLMs, but practical and effective defenses remain significantly limited. The core challenge is a privacy-utility tension: fine-tuning improves utility by increasing confidence on the ground-truth (“gold”) token, yet this shift creates statistical differences that reveal membership. In this work, we introduce GUARD (Gold-Unchanged Anchored Distillation), a novel, robust, and lightweight defense that mitigates privacy leakage while preserving model utility. GUARD first fine-tunes a teacher model on downstream data to capture generalization and memorization capabilities. It then constructs an anchored target distribution by fixing the gold token’s probability to its pre-trained value and preserving the fine-tuned model’s ranking among non-gold tokens while assigning them pre-trained magnitudes. A student is distilled to match this target. This design suppresses the dominant membership signal while retaining task-relevant distributional structure. Across diverse model families and benchmarks, GUARD demonstrates state-of-the-art downstream utility, enhanced robustness against membership inference attacks, improved design efficiency, and strong scalability across tasks. Code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes GUARD (Gold-Unchanged Anchored Distillation), a new defense framework designed to protect fine-tuned large language models (LLMs) against membership inference attacks (MIAs).
The core insight is that fine-tuning increases the model’s confidence on the gold token, which amplifies the statistical gap between members and non-members — the key signal exploited by MIA methods.
GUARD mitigates this by anchoring the gold token’s probability to its pre-trained value while preserving the relative ranking of other tokens from the fine-tuned model.
A student model is then distilled to match this “anchored” target distribution.
This approach aims to suppress membership signals without degrading task performance.

### Strengths
Anchoring the gold token probability while distilling non-gold ranking is a fresh, conceptually elegant idea that directly attacks the statistical signal used by most MIAs.
Experiments span multiple model architectures and datasets, comparing GUARD to the SOTA defense SOFT, with GUARD achieving near-random MIA success rates while maintaining comparable ROUGE-L and GPT-4o scores.
The transition from theoretical O(ε) analysis to empirical verification is convincing and well-executed.

### Weaknesses
Although the paper provides thorough theoretical and empirical analyses showing that rearranging the gold token distribution does not affect model performance, I have a major concern: which component is actually more important — the gold token anchoring or the distillation process itself?
Knowledge distillation alone can already reduce the membership signal and make MIAs ineffective^1. If plain distillation is sufficient to achieve defense, then why is it necessary to reassign the gold token’s probability?
Without proving that anchoring contributes additional protection beyond what distillation inherently offers, the gold token reassigning becomes somewhat meaningless, and the paper’s claimed innovation would be significantly weakened.

1. Li M, Ye Z, Li Y, et al. Membership inference attack should move on to distributional statistics for distilled generative models[J]. arXiv preprint arXiv:2502.02970, 2025.

### Questions
Would it be possible to include an ablation study to demonstrate that both steps of the proposed method are crucial?
If the reallocation of the gold token probability plays a decisive role in improving the defense, I believe this would make the work very meaningful and would raise the score.

### Soundness
2

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
The paper introduces GUARD, a framework for defending fine-tuned large language models (LLMs) against membership inference attacks (MIAs). GUARD utilizes a technique of Gold-Unchanged Anchored Distillation, which anchors the gold token's probability to its pre-trained value while transferring distributional knowledge from the fine-tuned model through knowledge distillation. The paper claims to achieve state-of-the-art defense performance while preserving the utility of the model.

### Strengths
+ The approach presents a new defense mechanism for MIA, which tackles the privacy risks in fine-tuned models without requiring extensive retraining or significant changes to model architecture.

+ The authors evaluate GUARD on a diverse set of models (GPT-Neo, Qwen, LLaMA) and across multiple datasets (PileCC, Wikipedia, PubMed, etc.), showing its robustness against a wide range of MIAs, both reference-based and reference-free. The empirical results are strong and demonstrate the effectiveness of the defense.

+ The proposed method balances privacy protection and model utility, with minimal performance degradation across different tasks.

### Weaknesses
- Limited baseline comparisons.
- Theoretical analysis issues.
- Potential vulnerabilities when the pre-trained model is accessible.
- Similarities with existing methods.

### Questions
- Limited Baseline Comparisons. The paper compares GUARD only with SOFT. While SOFT is a strong baseline, it would be more comprehensive to include additional baselines such as Differential Privacy (DP) based ones and classic knowledge distillation methods like DMP [1] and KCP[2]. Some of these methods are mentioned in the related works, but are not tested.

- The theoretical analysis of the bound on the probability distribution shift seems problematic. Specifically, the paper uses $\sum \Delta(y)$ (net sum) instead of $\sum |\Delta(y)|$ (sum of absolute values, or $L_1$ norm) in line 782, which leads to a bound that is too small. Using the triangle inequality or similar established inequalities, the correct bound for $|\Delta(\cdot \mid x)|_1$ should be $2$ (this can also be understood as two vectors not overlapping, which can be achieved), implying that the convergence order is $O(1)$. Thus, the approach to deriving bounds based on the loss difference may not be a solid starting point, unless a more compact bound or stricter assumptions can be provided. For clarity, a counterexample can be constructed: let $a = [0.6, 0.4, 0.2]$ and $b = [0.5, 0.1, 0.4]$ with the first position representing the gold token. Here $\delta(y^* \mid x) = 0.6 - 0.5 = 0.1$, but $\sum{y \neq y^*} \delta(y \mid x)$, resulting in $2\epsilon \le (|a_2 - b_2| + |a_3 - b_3|) = ((0.4 - 0.1) + (0.4 - 0.2)) = 0.5$. This makes the theoretical justification weaker.

- The method aligns the token probability distribution of the fine-tuned model with that of the pre-trained model. Although this design mitigates membership inference attacks (MIA), it may weaken robustness when the pre-trained model is accessible or open-sourced. In such cases, an attacker could leverage the output of the defended (GUARD-trained) model to identify its corresponding pre-trained model and, by analyzing the pre-trained token distribution, infer which token is likely to be the "golden" one. This coupling between defense and pre-training weakens adaptability and reduces robustness in practical scenarios.

- The Decoupled Knowledge Distillation (DKD) [3] method shares similarities with GUARD's approach, particularly in the handling of target and non-target tokens. DKD splits the loss into two parts: one focuses on minimizing the logits difference for the correct token, and the other calculates the KL divergence between the student and teacher models over non-target tokens, which aligns with GUARD’s idea of preserving token rankings and capturing “dark knowledge.” 

[1] Membership privacy for machine learning models through knowledge transfer

[2] Knowledge Cross-Distillation for Membership Privacy

[3] Decoupled knowledge distillation

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
This paper introduces GUARD (Gold-Unchanged Anchored Distillation), a novel lightweight and robust framework designed to mitigate privacy leakage in fine-tuned large language models (LLMs) while preserving task performance. The authors address the limitations of existing privacy defense methods (machine unlearning, differential privacy, data obfuscation), which often entail sharp trade-offs in scalability, utility, efficiency, and efficacy. GUARD operates by anchoring the gold token's probability to its pre-trained value, ensuring it remains lower than the typically elevated value assigned by the fine-tuned model. This approach minimally perturbs the fine-tuned model's output distribution, as theoretically proven in the paper, and effectively prevents privacy leakage without significant utility degradation. The empirical results demonstrate that GUARD maintains high utility while providing strong privacy protection, as shown through comparisons with fine-tuned and pre-trained models on various datasets.

### Strengths
S1: The paper identifies a significant gap in privacy defense for fine-tuned LLMs and proposes a theoretically grounded solution that addresses the limitations of existing methods.

S2: The GUARD framework is lightweight and robust, with a clear three-stage process that effectively balances privacy protection and utility preservation.

S3: The theoretical analysis (Theorem 4.5) rigorously demonstrates that anchoring the gold token probability perturbs the soft-label KD objective only marginally (O(ε)), providing strong justification for the method's effectiveness.

S4: The empirical evaluation (Table 1) clearly demonstrates GUARD's effectiveness in maintaining high utility (top-1 rate and top-50 overlap) while preventing privacy leakage, with results consistently showing the fine-tuned model's output distribution remains close to the pre-trained model's.

S5: The paper provides a clear explanation of why GUARD works, showing that anchoring the gold token's probability ensures it remains lower than the typically elevated value assigned by the fine-tuned model, which is crucial for privacy protection.

### Weaknesses
W1: The paper lacks comprehensive evaluation across a wide range of LLM architectures and tasks, focusing primarily on GPT-Neo and Qwen models on specific datasets (PileCC and Wiki), which limits the generalizability of the findings.

W2: While the theoretical analysis is strong, the paper doesn't provide sufficient empirical evidence of GUARD's effectiveness against specific privacy attacks (e.g., membership inference attacks), which would strengthen the claim of its privacy protection capabilities.

W3: The paper doesn't discuss potential limitations of GUARD, such as its performance on extremely sensitive datasets or its robustness against sophisticated extraction attacks.

W4: The practical implementation details of GUARD are not fully explained, making it difficult for practitioners to implement the method in real-world scenarios.

W5: The paper doesn't compare GUARD with other state-of-the-art privacy defense methods beyond the brief mention of existing approaches, making it challenging to fully assess its relative performance.

### Questions
Q1: Could you provide a more comprehensive evaluation of GUARD across a wider range of LLM architectures (e.g., different transformer variants, sizes) and tasks (e.g., different NLP tasks), to better establish the generalizability of your approach?

Q2: How does GUARD perform against specific privacy attacks, such as membership inference attacks (MIAs) or data extraction attacks? A direct comparison with attack performance before and after applying GUARD would strengthen the privacy claims.

Q3: Could you provide more detailed implementation guidelines for GUARD, including any specific hyperparameters or tuning required for different models and datasets?

Q4: How does GUARD compare to other privacy defense methods (e.g., differential privacy, data obfuscation, machine unlearning) in terms of privacy protection level, utility preservation, and computational efficiency? A direct comparison would help position your contribution.

Q5: The paper mentions "GUARD proceeds in three interlocked stages" but doesn't clearly explain how these stages are implemented in practice. Could you provide a more detailed description of the implementation process?

Q6: The paper focuses on preserving the gold token's probability but doesn't discuss how GUARD handles cases where the gold token might not be the most probable token in the pre-trained model. How does GUARD handle such scenarios?

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
This paper introduces GUARD (Gold-Unchanged Anchored Distillation), a defense technique for fine-tuned large language models (LLMs) against membership inference attacks (MIAs). GUARD operates by anchoring the gold token’s output probability to its pre-trained value while distilling distributional knowledge from a fine-tuned teacher. The method is evaluated on multiple LLM architectures, datasets, and MIA variants, consistently demonstrating strong defense with minimal loss of model utility. The approach is benchmarked against the strongest prior defenses, and both empirical and theoretical analyses are provided.

### Strengths
- Clear and Well-Motivated Problem Formulation: The authors identify and articulate the privacy-utility tension in fine-tuned LLMs, focusing on the heightened risk of MIAs due to increased gold token probabilities after fine-tuning.
- Methodological Innovation: GUARD’s key idea—anchoring the gold token probability to the pre-trained model while preserving the relative ranking of non-gold tokens from the fine-tuned model—is creative and efficiently operationalized through a principled knowledge distillation framework (see Algorithm 1, Page 5). This approach goes beyond prior strategies that indiscriminately obfuscate or degrade all output probabilities.
- Empirical Rigor & Breadth: Experiments span several model families (LLaMA, GPT-Neo, Qwen), six public datasets (PileCC, Wikipedia, HackerNews, PubMed, Arxiv, GitHub), and a wide spectrum of MIA variants, including both reference-based and reference-free attacks (Tables 2–5, Pages 7–8, 18). Results consistently favor GUARD, with defense effectiveness approaching random guessing for MIAs without degrading downstream performance.
- Multi-faceted Benchmarking: The analysis is not limited to privacy metrics. It includes model utility scores using both ROUGE-L and the LLM-as-a-Judge framework with ChatGPT-4o and qualitative examination (Figures 2, 9). Figure 1d provides an informative radar plot comparing MIA defense, extraction defense, efficiency, scalability, and downstream utility across methods.
- Mathematical Justification: The theoretical analysis (Page 5–6 and Appendix A.5) clarifies why the anchored modification produces only small, bounded perturbations on the knowledge distillation loss, guaranteeing negligible impact on student model generalization or risk. The bounds are stated formally, with necessary assumptions and order-of-magnitude reasoning.
- Interpretability via Visualizations: Figure 1a–c provides clear visual evidence of how fine-tuning inflates gold token probabilities (enabling MIAs), and how GUARD restores these values while preserving the distribution structure on alternatives.
- Careful Consideration of Alternative Defenses: The paper benchmarks against SOFT (data obfuscation) and analyzes unlearning and DP-based approaches (Tables 13, Figure 6), discussing their downsides in efficiency and/or utility (Page 19).
- Transparency and Reproducibility: Complete experiment details, hyperparameters, prompts, and ablation studies are provided for reproducibility (Appendix A.4, A.3, A.6, A.7).

### Weaknesses
1. Positioning vs. the Latest MIA Evaluations and Defenses: The related work currently omits several directly relevant and contemporary papers that provide frameworks, taxonomies, or empirical findings on MIAs, especially those evaluating the latest attack and defense strategies in LLMs. This has a twofold effect: (i) the significance and novelty of GUARD are not fully contextualized relative to the most comprehensive current understanding, and (ii) the empirical pipeline may miss certain edge attacks or metrics proposed in these works, which could stress-test GUARD further.
2. Empirical Breadth in Baseline Selection: While SOFT is used as the primary comparison, Tables 2–5 reveal that only SOFT, pre-trained, and fine-tuned models are reported as baselines. Alternative defense families discussed in related work (machine unlearning, DP-based training, e.g., DP-LoRA) are only sparsely evaluated or buried in the appendix. This prevents a fully transparent, side-by-side comparison of GUARD vs. the complete suite of practical baselines proposed in recent work.
3. Attack Diversity: The selected MIAs are principled and fairly implemented, but the evaluation is still limited to attacks referenced in SOFT and a handful from the literature. Recent works (see Potentially Missing Related Work) have introduced/or systematized new MIA attack strategies; it remains uncertain whether GUARD would hold up under such attacks without at least discussion/positioning.
4. Exposition – Notation, Mathematics, and Algorithm Details: While the mathematics is generally sound, certain aspects could benefit from greater formal clarity or explicitness in the main text:
- Equation (Page 5) and Algorithm 1 use terms like $\tau$ and top-$K$ selection, but practical details (e.g., how temperature interacts with probability anchoring, how top-$K$ is chosen for different models) are mostly deferred to the appendix. For example, the handling of rare tokens in large vocabularies (>150k) is mentioned as “top-1000,” but the impact on low-probability alternatives is unclear. Readers seeking to reproduce or extend GUARD would benefit from a more explicit discussion in the main section.
- The loss term $\mathcal{L}_{\text{final}}$ could benefit from an explicit normalization scheme or a more principled proof of generalization, particularly regarding $\lambda$’s sensitivity or how non-gold-mass redistribution is handled mathematically (is it permutation-invariant?).
- While the theoretical bound is clear, the main text glosses over possible degenerate cases (e.g., where the gold token is neither in the top-$K$ pre-trained nor fine-tuned predictions), and there is no detailed analysis of failure/misalignment modes.
5. Interpretability and Qualitative Assessment: Figure 1 provides an intuitive visualization of GUARD’s impact on token probabilities and membership signals, and Figure 2 shows qualitative response evaluations. However, the paper would benefit from a more thorough qualitative error analysis in Sections 4.3 and Appendix A.7.6. For instance, is there a consistent type of answer (e.g., factual recall, definition, paraphrase) for which GUARD consistently underperforms or fails subtly compared to the fully fine-tuned model? The two examples in Figure 2 are not enough to uncover systematic trade-offs.
6. Ablation Study Limitations: The ablation study in Table 8 (Appendix A.3) presents only a simplified variation (“reordering only”) compared to full GUARD. More granular ablations—e.g., gold token anchoring only, diverse ranking strategies among non-gold tokens, variable top-$K$ thresholds, or temperature scaling effects—would offer sharper insights into which elements contribute most significantly to defense or utility.
7. Potential Exploitation of Non-Gold Distribution Rank: In Appendix A.2 the authors acknowledge the possibility that non-gold token ranking overlap between fine-tuned and pre-trained models may itself provide a membership leakage channel (possibly for an adaptive adversary). The presented evidence is that such overlap patterns vary randomly and the signal is weaker than the gold token, but this remains an empirical claim. Additional experiments (e.g., adversarial MIA variants that specifically test overlap exploitation) or further theoretical justification would strengthen the defense story.
8. Clarity in Naming and Consistency: Some notation and dataset/model naming is inconsistent or unclear. For example, Table 4 uses column headers like “Acuis” and “Gidush,” which seem like typos or confused references (likely “Arxiv” and “Github”). Such errors, though minor, can impede reproducibility and undermine confidence.

Required Figure and Table Engagement:
1. Referencing Figure 1 (Page 2): This figure is central; it visually illustrates the core premise that fine-tuning inflates the gold token probability (observable through the sharply heightened yellow bar) and therefore exposes the model to MIAs. GUARD by contrast restores the gold bar to its pre-trained level while preserving structure in the alternatives—a direct graphical argument for why attack success is neutralized. Furthermore, the radar plot (Figure 1d) concretely demonstrates GUARD’s strongly balanced performance across privacy, utility, scalability, and efficiency axes compared to other approaches, charting a clear visual superiority.
2. Referencing Table 2 and Table 6 (Pages 7 and 9): Table 2 shows AUC-ROC scores across datasets/attacks, consistently demonstrating GUARD’s ability to drive MIA success to near-random (≈0.5). Table 6, meanwhile, provides utility scores (GPT-4o and ROUGE-L), which—while slightly lower than full fine-tuning—still approach or surpass the utility of the best prior defenses, giving a quantitative measure of the privacy-utility trade-off.
3. Referencing Algorithm 1 and Equation in Section 3.2 (Page 5): This is where the technical heart of GUARD is laid out. While the loss is clearly defined, the implementation specifics (especially for top-$K$ retention, regularization strength $\lambda$, and distribution normalization under extreme class imbalance) warrant greater transparency for robust adoption.

### Questions
1. Can the authors clarify how the choice of top-$K$ for non-gold token retention impacts both the privacy leakage and model utility trade-off, especially for very large vocabularies ($>150$K tokens)? Would retaining more (or fewer) alternatives shift the defense-utility balance, and are there diminishing returns?
2. In Algorithm 1 and the main loss formulation, how sensitive is GUARD to the gold anchoring weight $\lambda$? Is there empirical/theoretical justification for the chosen default, and what guidance would you offer practitioners to tune this without heavy validation?
3. Have the authors considered or experimented with adaptive MIAs that attempt to exploit top-$K$ overlap or distributional rank instead of just the gold token? If so, please report findings and/or elaborate on how GUARD would defend against such adversaries.
4. Appendix A.2 gives some evidence that overlap-based signals are weak, but could you provide additional experiments (either adaptive MIAs or ablation studies) to confirm these results are robust to dataset/model scale and do not emerge as a secondary avenue for membership leakage?
5. Can the authors extend the ablation study to more granular variants (e.g., gold-only anchoring, alternative labeling of non-gold ranking, variable reweighting schemes, or temperature scaling effects) to better isolate the critical ingredients of GUARD’s empirical success?

### Soundness
3

### Presentation
3

### Contribution
3
