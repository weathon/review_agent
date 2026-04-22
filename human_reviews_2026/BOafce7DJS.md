# UPCORE: Utility-Preserving Coreset Selection for Balanced Unlearning

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
User specifications or legal frameworks often require information to be removed from pretrained models, including large language models (LLMs). This requires deleting or "forgetting" a set of data points from an already-trained model, which typically degrades its performance on other data points. Thus, a balance must be struck between removing information and keeping the model's other abilities intact, with a failure to balance this trade-off leading to poor deletion or an unusable model. To this end, we propose UPCORE (Utility-Preserving Coreset Selection), a method-agnostic data selection framework for mitigating collateral damage during unlearning. Finding that the model damage is correlated with the variance of the model's representations on the forget set, we selectively prune the forget set to remove outliers, thereby minimizing model degradation after unlearning. Across three standard unlearning methods, UPCORE consistently achieves a superior balance between the competing objectives of deletion efficacy and model preservation. To better evaluate this trade-off, we introduce a new metric, measuring the area-under-the-curve (AUC) across standard metrics. Our results show that UPCORE improves both standard metrics and AUC, benefiting from positive transfer between the coreset and pruned points while reducing negative transfer from the forget set to points outside of it.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses machine unlearning in LLMs by proposing UPCORE, a data selection framework that reduces collateral damage during the forgetting process. The core insight is that hidden state variance (HSV) in the forget set correlates with model utility degradation after unlearning. UPCORE uses Isolation Forest to identify and prune high-variance outliers, creating a lower-variance coreset for unlearning. The authors evaluate their approach across three unlearning methods (Gradient Ascent, Refusal, NPO) and multiple datasets, introducing an AUC-based metric to better capture the deletion-utility tradeoff. Results show consistent improvements in balancing forget effectiveness with model preservation.

### Strengths
Clear problem formulation and strong motivation: Machine unlearning for privacy compliance is increasingly important, and the deletion-utility tradeoff is well-articulated. The focus on mitigating collateral damage addresses a real pain point.

Method-agnostic and practical: UPCORE works across different unlearning algorithms (GA, Refusal, NPO) without modification, making it broadly applicable. The use of Isolation Forest is computationally efficient and doesn't require expensive hyperparameter tuning.

Comprehensive experiments: The evaluation is thorough - three unlearning methods, multiple datasets (Counterfact, TriviaQA), multiple baselines (Random, D2-pruning, RUM), and robustness tests (paraphrase, jailbreak variants). Statistical significance is properly reported (Table 12).

Consistent improvements: UPCORE achieves 3-7 AUC point gains across all settings with high statistical significance (p < 10^-15). The gains hold even in multi-topic scenarios (Table 5), suggesting good generalizability.

Novel evaluation perspective: The AUC metric provides a more holistic view of the deletion-utility tradeoff over time, which is conceptually valuable even if its practical importance is debatable.

### Weaknesses
Weak theoretical justification: The paper relies heavily on empirical correlation between variance and collateral damage, but doesn't provide rigorous theoretical analysis of why this relationship holds. Section 2.1's problem formulation is informal, and Appendix B (per my understanding from the main text) appears to offer intuitive discussion rather than formal proofs. What are the sufficient/necessary conditions for UPCORE to work? When might it fail?

Positive transfer mechanism unclear: Table 4 shows that pruned points get "forgotten" (ROUGE drops to ~0.05), but the random baseline shows similar effects. What's unique about UPCORE's pruning strategy compared to random sampling of the same size? The paper doesn't adequately explain the mechanism or provide direct evidence beyond ROUGE. Are the pruned points actually unlearned (e.g., via knowledge probes, adversarial queries) or just showing surface-level metric correlation?

Limited scope of applicability: The method assumes topic-level unlearning with semantically coherent clusters. While Section 4.4 shows multi-topic results, real-world forget requests are often heterogeneous and sparse (e.g., random user deletion requests under GDPR). How does UPCORE perform when the forget set lacks clear structure? The paper mentions "if sparse or heterogeneous, DC = DF" but doesn't empirically validate this fallback strategy or discuss when clustering might fail.

### Questions
Validation of pruned point forgetting: Table 4 only shows ROUGE scores dropping. Can you provide more direct evidence that pruned points are actually forgotten? For instance, test with specific knowledge probes (e.g., "Who is X?" questions for pruned facts) or adversarial extraction attempts. This is crucial for validating the core claim about positive transfer.

Hyperparameter sensitivity and selection principles: You prune 10% by default and vary it in scaling studies. But how should practitioners choose this percentage in real applications? Is there a principled way to set the threshold τ based on properties of the forget set? What about Isolation Forest parameters (number of trees, subsample size)? A sensitivity analysis in the main text would be valuable.

Computational overhead breakdown: How much time does hidden state extraction and Isolation Forest training add compared to the unlearning process itself? For large-scale LLMs and bigger forget sets, is this overhead acceptable? A concrete runtime comparison with baselines would help assess practical feasibility.

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
5

### Summary
This paper proposes a dataset sampling strategy designed to mitigate the problem of catastrophic forgetting that occurs when existing unlearning methods utilize the entire unlearning dataset. In this approach, the core set is determined prior to the unlearning process based on the variance of data representations in the model’s final-layer feature space. The paper centers its discussion on how different sampling strategies can facilitate more effective unlearning and includes comparisons among several baseline sampling methods. Experimental results show that the proposed UPCORE method improves the efficiency of unlearning and mitigates the problem of catastrophic forgetting.

Although the paper presents a novel idea and achieves some methodological success, many important details and practical aspects of unlearning remain insufficiently explored. Several experimental setups also leave room for improvement. These limitations are crucial to determining whether the proposed unlearning approach is truly effective. Please refer to the Weaknesses section for more detailed comments.

### Strengths
1. The motivation of this paper is both clever and timely. It addresses the problem of knowledge redundancy within unlearning datasets and proposes a sampling-based approach to alleviate the inefficiency and excessive forgetting caused by such redundancy.

2. The proposed UPCORE method is simple yet effective. It employs a basic statistical approach to identify dense unlearning samples while discarding peripheral ones, making the overall idea intuitive and easy to understand.

3. The extensive experimental results clearly illustrate the effectiveness and superiority of UPCORE compared to existing unlearning approaches.

4. The paper is well written and clearly organized, making the ideas easy to follow and understand.

### Weaknesses
1. UPCORE is a sampling-based unlearning method that optimizes the model using unlearn samples from dense regions of the feature space. However, relying solely on core unlearn samples may result in suboptimal unlearning performance on non-core samples. Since these non-core samples are farther away in the feature space, they may be less influenced by the UPCORE optimization process. It would be helpful for the authors to clarify or empirically verify whether the method maintains consistent unlearning effectiveness across both core and non-core samples.

2. The paper claims that the UPCORE method effectively preserves model utility. However, it remains unclear how the unlearned model behaves on the forgotten samples. Does the model produce meaningless or incoherent outputs, or does it still generate fluent but semantically altered responses? This distinction is critical, as it determines whether the model truly performs appropriate unlearning or simply collapses on the unlearned data. Providing concrete qualitative examples of model outputs on forget samples would significantly improve the clarity and credibility of the paper.

3. As a sampling-based approach, the proportion of data retained after sampling is a critical factor. Although Appendix D presents the performance of UPCORE under different sampling ratios, the paper lacks a fair comparison across different sampling methods under the several same sampling rates. Such a comparison would provide a more complete and convincing demonstration of UPCORE's advantages over alternative methods.

4. The generality of UpCore should be further validated across a wider range of unlearning methods. For example, it would be valuable to test its effectiveness when applied to approaches such as FLAT [1], RMU [2], DPO [3], WGA [4], SatImp [5], and others. Such experiments would provide stronger evidence that UpCore can serve as a broadly applicable sampling strategy rather than one tied to a specific framework.

[1] Wang, Yaxuan, et al. "Llm unlearning via loss adjustment with only forget data." arXiv preprint arXiv:2410.11143 (2024).

[2] Li, Nathaniel, et al. "The wmdp benchmark: Measuring and reducing malicious use with unlearning." arXiv preprint arXiv:2403.03218 (2024).

[3] Rafailov, Rafael, et al. "Direct preference optimization: Your language model is secretly a reward model." Advances in neural information processing systems 36 (2023): 53728-53741.

[4] Wang, Qizhou, et al. "Rethinking llm unlearning objectives: A gradient perspective and go beyond." arXiv preprint arXiv:2502.19301 (2025).

[5] Yang, Puning, et al. "Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning." arXiv preprint arXiv:2505.11953 (2025).

### Questions
Please refer to the weakness.

I would be willing to increase my score if the authors can adequately address the concerns and clarify the issues discussed above.

### Soundness
2

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
This paper introduces UPCORE, a method for selecting a lower-variance “core forget set” to minimize the utility loss during unlearning in LLMs. The authors demonstrate that the variance in the representation space of the forget set correlates strongly with post-unlearning damage, and apply Isolation Forests to prune outliers before performing unlearning. Experimental results show that UPCORE outperforms several baselines in preserving model utility while maintaining high forgetting efficacy.

### Strengths
1. The paper, for the first time, attempts to mitigate the impact of model unlearning on general performance by identifying high-variance “bad points” in the forgetting dataset. This finding and its supporting evidence are novel.
2. The experiments are relatively thorough and, across multiple datasets, demonstrate that the method can preserve unlearning performance while maintaining the original performance of the model.
3. The writing is clear and easy to follow, with no obvious typographical or formatting errors.
4. The authors also provide the source code.

### Weaknesses
1. There is a lack of deeper theoretical guarantees about how much variance reduction one should expect to translate to concrete utility improvement, or about optimality of the Isolation Forest-based selection. For example,  the equation (in Section 2.1) frames the coreset selection as an optimization, but in practice this is relaxed to a heuristic on variance. There seems to be little analysis of approximation gap or the behavior under adversarial/practically pathological distributions.
2. The explanation for why Isolation Forest consistently outperforms other baselines (Table 8) is shallow: there is little insight into failure cases or when denser/sparser baselines might be preferable.
3. It would be preferable for the authors to number the equations in the paper for ease of reference.

### Questions
1. Could the authors provide a more explicit enumeration of limitations, potential failure modes or related discussions.
2. Did the authors experiment with more sophisticated representations or with influence scores instead of raw variance? Is there an upper bound on what variance-based pruning can achieve?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies machine unlearning in LLMs, removing specific topics or data while keeping the rest of the model useful. The authors observe that higher variance in the model’s hidden representations of the forget set correlates with more "collateral damage" to unrelated capabilities after unlearning. They propose UPCORE, a simple data-selection step (Isolation Forest–based outlier pruning) that forms a lower-variance coreset of the forget set before applying any standard unlearning algorithm. They also introduce an AUC-style metric that evaluates the trajectory of the deletion–utility trade-off across checkpoints rather than at a single step. Across Counterfact and TriviaQA with three unlearning methods (Gradient Ascent, Refusal, NPO), UPCORE generally improves the deletion–utility frontier and shows some positive transfer (the pruned, untrained points are often forgotten anyway), while reducing negative transfer to unrelated data.

### Strengths
1. Coreset pruning via Isolation Forest is simple, plug-and-play, and can sit atop existing unlearning procedures. 
2. The paper documents a strong correlation between hidden-state variance of the forget set and post-unlearning utility degradation, which is actionable and intuitive.
3.The paper distinguishes beneficial transfer (pruned points still forgotten) from harmful spillover (unrelated capability loss), and shows reductions in the latter.
4. The AUC trade-off metric across training steps addresses comparability across methods/checkpoints and is a sensible complement to point estimates.
5. Results span multiple datasets/methods and include robustness (paraphrases, jailbreaks), with consistent AUC improvements.

### Weaknesses
1. While the paper shows that forget-set variance correlates with collateral damage, it does not establish a causal mechanism linking variance to the learning dynamics of unlearning (e.g., how gradient updates induced by high-variance points propagate damage to neighborhoods). Strengthening this with controlled interventions (e.g., synthetic variance manipulations at fixed difficulty/semantic content) or influence-function analyses would make the claim more compelling. (Author text frames the finding as correlation; causal linkage remains under-developed.)

2. The pruning rule relies on Isolation Forest anomaly scores over hidden states, thus helping identifying outliers and pruning them out, thus minimizing the variance overall; however, additional analyses disentangling variance from difficulty/memorization/entanglement are limited. The paper notes checks on other attributes but a more thorough ablation (e.g., control for answer length, lexical diversity, topic dispersion, gradient curvature) would clarify whether variance is the primary driver.

3. The appendix (appendix B) sketches influence-style intuition but lacks concrete assumptions, lemmas, or citations grounding the claimed propagation of deletion to pruned neighbors and the conditions under which pruning is utility-optimal. This section should substantially expand citations to prior work on representation-space neighborhoods and gradient influence to justify the claims.

### Questions
1. How does UPCORE compare to difficulty- or memorization-aware selection when matched for coreset size? Are the gains additive if you combine criteria?

2. Is there a way to quantify the distance (in hidden space) over which positive transfer reliably holds?

### Soundness
3

### Presentation
4

### Contribution
2
