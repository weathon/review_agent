# SUIT: Knowledge Editing with Subspace-Aware Key-Value Mappings

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Knowledge editing aims to efficiently correct factual errors in language models. Widely used locate-then-edit methods update an MLP layer by adjusting its weights to change the mapping between the layer’s input vector (key) and output vector (value), thereby editing the model’s knowledge. 
As this update is driven by key and value vectors, obtaining these vectors without careful constraints causes significant model perturbations beyond the targeted edit, a common issue in many prior knowledge editing methods.
To address this, we propose Subspace Knowledge Edit (SUIT), which computes key and value vectors only within the subspace of critical features relevant to the edit. Our empirical results on LLaMA3, GPT-J, and Qwen2.5 models show that SUIT dramatically improves knowledge preservation over strong baselines while maintaining high editing performance. These results support the claim that SUIT successfully identifies the critical subspace for the edit. 
Beyond quantitative gains, our analyses show that SUIT reduces unintended perturbations in hidden states while confining updates to directions that are more effective for editing.
Taken together, these findings establish edit-critical subspace identification as a key principle for reliable, low-perturbation knowledge editing.
Our code is available at https://github.com/holi-lab/SUIT.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SUIT (Subspace Knowledge Edit), a new knowledge editing method aimed at solving the problem of poor "specificity" (i.e., corrupting unrelated knowledge) that is common in locate-then-edit approaches. The method claims to achieve its goal via two core constraints: 1) When computing the "key vector" $k$, it identifies and removes an "entity-agnostic" subspace derived from 10,000 samples, purportedly isolating the edit to "entity-specific" features. 2) When computing the "residual vector" $\delta$, it forcibly constrains the optimization to a fixed two-dimensional subspace, defined by two vectors $w_1$ and $w_2$ intended to promote the new knowledge and suppress the old. Experimental results show the method improves specificity on the COUNTERFACT dataset for models like LLaMA-3.

### Strengths
1. This paper identifies the challenge of low specificity in knowledge editing.
2. The method demonstrates a significant specificity boost on the COUNTERFACT dataset based on a generation-based criterion.
3. The paper provides mechanistic evidence (e.g., token-level perturbations in residual streams) to validate the reduced perturbation.

### Weaknesses
1. The paper **assumes** the residual vector (the knowledge update) can be constrained to a two-dimensional subspace which lacks theoretical justification.
2. The subspace identification relies on an energy threshold hyperparameter $\tau_{energy}$ to which the model performance is **sensitive** as shown in Figure 5. The optimal $\tau_{energy}$ value obtained on COUNTERFACT may not be applicable to other datasets which adds an extra tuning burden for the method's application.
3. The paper selects $N=10,000$ subjects for the SVD pre-computation step which has no justification for this specific number nor any sensitivity analysis. It is unclear how robust the identified "entity-agnostic" subspace is; a different $N$ could potentially yield a different subspace. Furthermore, the computational cost of this SVD step scales directly with $N$, yet the authors fail to demonstrate that $N=10,000$ represents a reasonable trade-off between computational cost and the stability of the resulting subspace.
4. The analysis in $\S 6.2.1$ does not independently substantiate the semantic claim that $K_s^{\perp}$ is “entity-agnostic.” The subspace is defined via an SVD criterion, and the subsequent variance comparison merely reflects that construction rather than testing semantic invariance.
5. The second $\lambda_{KL}$ that appears in Appendix B should be weight decay $\lambda_{WD}$, which is likely a typo.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel knowledge editing method, SUIT (Subspace Knowledge Edit), which constrains updates to task-relevant subspaces by decomposing the key vector 𝑘 into entity-specific features and restricting the residual vector 𝛿 to the feature directions most relevant to the new object. By operating only within these targeted subspaces, SUIT enables precise and localized modification of factual knowledge.

### Strengths
1. The performance of SUIT is good and surpasses other baselines across datasets.

2. The idea of making a finer-grained distinction in the editing subspace is novel and also reminds me of SAE..

3. The paper's presentation is clear and well-organized.

### Weaknesses
1. It would be valuable to further evaluate the "ripple effects" metric of knowledge editing, which is mentioned in [1].

2. As with other locate-then-edit approaches, it would be good to explore whether this method can generalize beyond the triple-based question.

3. It would strengthen the work if experiments were extended to larger-scale models, such as 14B-parameter models.

4. On the ZSRE dataset, the average performance (S score) on two models is still lower than that of AlphaEdit.

5. SUIT generally underperforms AlphaEdit on the Generalization metric; it would be helpful to investigate the underlying reasons for this gap.

6. It would also be insightful to compare SUIT with SAE-based knowledge editing approaches and examine performance differences.

If the above weaknesses are addressed well, I will reconsider my rating.

---
**References**:

[1] Evaluating the Ripple Effects of Knowledge Editing in Language Models

### Questions
Please see the weaknesses list above.

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
4

### Summary
This paper studies the unintended side effects of LLM knowledge editing and aims to improve specificity. Building on the locate-then-edit paradigm that treats the MLP down-projection as a linear associative memory W mapping keys to values, the authors propose Subspace Knowledge Edit (SUIT). SUIT (i) removes entity-agnostic components from the key vector by projecting onto and subtracting a data-driven “common” subspace obtained via SVD over many subject keys, and (ii) restricts the residual/value update to a low-dimensional subspace spanned by two optimized directions that respectively promote the new object and suppress the old one.

### Strengths
- Subspace isolation via data-driven SVD. The paper presents a clean and principled subspace isolation method: collect key vectors for a large set of subjects, perform SVD, and define the entity-agnostic subspace as the span of top singular vectors up to an energy threshold \tau; subtracting this projection yields an entity-specific key k. This is simple, effective, and empirically validated (variance analysis in Table 2; reduced interaction with the common subspace in Table 3)
- The experimental results suggest that the proposed method is comparable to SOTA baselines.

### Weaknesses
- Sensitivity to the energy threshold τenergy and selection cost. The core localization step relies on a manually chosen τenergy. Fig. 5 shows a nontrivial trade-off: as τenergy increases, generation tends to decrease while specificity increases, with a “sweet spot” around 0.3–0.4. Although efficacy is fairly stable, the turning point is not sharply defined and may require cross-validation that is data-dependent and potentially costly. Please (a) provide a principled selection rule/heuristic (e.g., knee detection on cumulative energy, variance ratio targets, or stability-based criteria), (b) report sensitivity across datasets/models when fixing \tau energy without extra tuning (you currently fix \tau=0.4 for all models ), and (c) discuss compute overhead for building K subject and running SVD (N=10k subjects) and how often this needs to be repeated per model/layer.
- Not uniformly state-of-the-art across all settings. While SUIT is very strong overall, it is not consistently the best on every model/dataset. For example, on ZSRE, SUIT’s harmonic mean S trails AlphaEdit slightly on GPT-J and Qwen (e.g., AlphaEdit S≈96.9 vs. SUIT S≈95.9 on GPT-J; AlphaEdit S≈89.6 vs. SUIT S≈88.2 on Qwen).
- AlphaEdit’s results are slightly lower than their original paper.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents SUIT (Subspace Knowledge Edit), a new framework for editing factual knowledge in large language models while minimizing disruption to unrelated information. Building on the Linear Representation Hypothesis, the authors propose subspace-aware editing that confines changes to semantic subspaces relevant to the target entity and relation. SUIT improves upon prior methods like ROME, MEMIT, and AlphaEdit by refining the computation of the key and residual vectors through subspace decomposition. This design enhances specificity and stability during edits. Experiments on LLaMA-3-8B, GPT-J-6B, and Qwen2.5-7B show large gains in specificity (e.g., +43.2 points on LLaMA-3-8B) while maintaining strong edit accuracy.

### Strengths
This paper makes a solid and coherent contribution to the field of knowledge editing in large language models. It stands out in three main aspects.
(1) It provides a clear theoretical framing grounded in the Linear Representation Hypothesis, establishing a principled link between semantic subspaces and factual updates. This theoretical foundation not only motivates the method design but also enhances interpretability and conceptual rigor.
(2) The paper offers comprehensive and convincing analyses, including ablation studies, perturbation comparisons, and examinations of key components such as subspace isolation and feature decomposition. These analyses provide both empirical depth and transparency, showing that the improvements are consistent and well-supported.
(3) The overall structure and presentation are highly polished. The logical flow—from motivation to methodology, experiments, and analysis—is clear and well-balanced, and the exposition makes complex ideas accessible without oversimplifying them. Collectively, these qualities make the work both theoretically meaningful and practically persuasive.

### Weaknesses
While the paper presents a clear and well-executed framework, several aspects limit its novelty and clarity.
(1) The core technical innovation is not particularly strong. Although the subspace-based approach is conceptually sound, its methodological route appears closely aligned with the null-space perturbation ideas introduced in AlphaEdit. The distinction between the proposed subspace isolation and existing null-space constraints is not sufficiently emphasized, making the contribution seem more incremental than foundational.
(2) There are issues in the main experiments that reduce confidence in the reported improvements. The primary evaluation table introduces the “S” metric as the harmonic mean of three scores, yet the performance gain seems largely driven by unusually high specificity, while the other two components remain modest. Moreover, the specificity scores for baseline methods such as MEMIT and AlphaEdit appear unexpectedly low compared to prior results. Additionally, the table includes a “GA” metric that lacks explanation—it is likely intended to be “GC,” but this ambiguity should be clarified.
(3) Finally, although the appendix provides numerous additional analyses, their purpose and logical role within the overall argument are unclear. The paper does not clearly justify why these analyses are necessary or how they support the main claims. Some component analyses could also be presented more transparently to better highlight their individual contributions to the overall performance gains.

### Questions
(1) The proposed subspace-based approach seems conceptually similar to the null-space technique used in AlphaEdit. Could the authors elaborate on the fundamental distinction between the proposed subspace isolation and the existing null-space constraint formulation? In particular, what new insights or advantages does SUIT provide beyond current research’s framework?
(2) Regarding the main experimental results, the “S” metric—described as the harmonic mean of three sub-metrics—appears to be driven primarily by the high specificity score. Could the authors provide a detailed breakdown of the three components and explain whether the improvements are consistent across all of them? Additionally, the baseline scores for MEMIT and AlphaEdit seem unexpectedly low compared to prior reports. Were there any differences in implementation or evaluation setup that could explain this? Finally, please clarify the meaning of the “GA” metric in the main table; should this instead refer to “GC”?
(3) The additional analyses in the appendix are extensive, but their motivation is not entirely clear. Could the authors better articulate how each analysis (e.g., perturbation visualizations, subspace decomposition tests, or ablations) contributes to the main argument? For instance, which specific results directly support the claimed benefits of subspace isolation? More explicit linking between these analyses and the core findings would strengthen the logical flow of the paper.

### Soundness
2

### Presentation
3

### Contribution
2
