# On the Fragility of Latent Knowledge: Layer-wise Influence under Unlearning in Large Language Model

- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Large language model (LLM) unlearning has emerged as an essential post-training mechanism for erasing specific knowledge or undesirable behaviors. However, forgetting target data often causes an unintended degradation in overall model utility. Although various advanced methods have explored different learning objectives to mitigate the trade-off, it remains unclear how the highly entangled internal representations in LLMs contribute to unlearning. In this work, we introduce the notion of latent knowledge fragility to explore the vulnerability of retained knowledge to unlearning. We develop a unified analytical approach via component-wise parameter patching that isolates and quantifies fragility in terms of different transformer blocks. We observe that the LLM encodes different levels of abstraction, from surface syntax in shallow layers to complex semantics in deeper layers, which align with different degrees of representation disruption and utility degradation. Based on the insights, we propose a lightweight framework called Component-wise Replacement Unlearning (CRU) that restores fragile layers (also extendable to other components) from the original model based on post-hoc validation, which allows us to obtain a hybrid model without additional training. Extensive experiments on various aspects verify that our method generally improves the trade-off between removal and retention. Our analysis highlights the non-uniform influence of different LLM layers and provides a new possibility of surgical unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces the notion of "latent knowledge fragility" to examine how LLM unlearning affects the latent knowledge encoded in LLMs and develops a unified analytical approach using component-wise parameter patching to isolate and quantify the fragility in terms of transformer blocks. Through layer-wise patching experiments, the paper observes a semantic abstraction hierarchy explanation, which aligns with different degrees of representation disruption and utility degradation. The paper proposes Component-wise Replacement Unlearning, a lightweight, post-hoc framework that selectively restores fragile layers from the original model to improve the forget quality vs. model utility trade-off.

### Strengths
+ The paper studies how unlearning with GA, NPO, and variants affects the latent representations of the model via Centered Kernel Alignment, which is an interesting aspect of LLM unlearning. 
+ CRU is method-agnostic, requires no additional training, making it applicable across various unlearning settings and model scales.

### Weaknesses
# Major Issues and Concerns
1. **Overstated “U-shape” claim**: Section 3.1, Fig. 3 presents the forget quality (FQ) and model utility (MU) via layer-wise patching from the unlearned LLM on the original LLM. The paper claims that "the middle ones generally causes the significant utility degradation", which seems to be **overstated and potentially unsupported** by the reported results. In Fig. 3a (Llama 3.2 1B with GA), the MU ranges from *approximately* 0.586 (worst, at layer 10) to 0.600 (best, layer 0 or 1), representing only a 0.01 (or 0.02) absolute decrease, which is relatively small and could plausibly fall within random variation. MU is **not significantly decreased**. Similar small variations appear across other subfigures (Fig.3 b and Fig. 9 Appendix). The evidence for a strong "U-shaped" degradation pattern appears *weak*, questioning both the strength of this claim and the subsequent motivation built upon it. 
2. **Limited generalizability**: The paper conducts a fine-grained component-wise replacement analysis on attention heads, MLPs, and input/post normalization layers (Appendix D.1). However, the results indicate that the "U-shape" phenomenon **does not consistently hold across these components**, suggesting limited generalizability of the proposed CRU method. Although the paper claims that CRU can be extended to such fine-grained components, the presented evidence **does not substantiate this claim**.
3. **Weak theoretical justification for the "U-shape"**: The paper interprets the observed "U-shape" via CKA similarity. However, the paper lacks a rigorous theoretical explanation for why this pattern emerges universally. The semantic abstraction hierarchy explanation (shallow encodes local syntax, middle presents abstract, high-level concepts, deep is involved in output fluency) is intuitive and seems well-known and not formally established in the context of LLM unlearning.
4. **Costly and heuristic-driven**: The key idea of the proposed CRU is to post hoc quantify the importance of components by the change in validation performance, which is costly and time-consuming, and strongly depends on the top-k selection heuristic and the chosen validation set. Further, this seems not novel enough for this venue, to my taste.
5. **Limited architectural generalization**: All experiments use decoder-only transformers ($e.g.,$ Llama, Zephyr). While this is a representative setup, the effectiveness of the proposed method on other architectures ($e.g.,$ Mixture-of-Experts, State-Space Models), larger models, remains unverified.
# Minor Issues
+ Line 261:  $f^{\phi =l}$ should be $f^{\phi =[l]}$
+ Proposition 1 Eq. 2: Use clearer notation for squared terms: $\sigma_{c}^{orig2} \to (\sigma_{c}^{orig})^{2}$
+ Appendix C: Incorrect reference, should be Proposition 1, not Proposition 2.
+ Line 249: "formal proof are provided" $\to$ "the formal proof is provided"
+ Figure 2 caption: "Highlighted are distinction from original answer." $\to$ "Highlighted are distinctions from the original answer."
+ Appendix E. 4: The paper states that "RMU pursues two parts of the objective, the first is to encourage the hidden
representation of forget target to be orthogonal to the original latent space,...". The objective of RMU is to steer the latent representations of the forget targets to *a predetermined random vector*, rather than orthogonalizing them with respect to the original latent subspace.

### Questions
+ How is the proposed component-wise patching different from "Attribution Patching"?
+ How sensitive is component selection to validation set size? What is the minimum validation set size needed for reliable component selection? In cases, the validation data is unrepresentative. What is the effect?
+ When the U-shape doesn't hold, the effectiveness of the proposed method remains questionable.
+ The bound in Eq. 2 of Proposition 1 depends on $||W_c||^2$ and concept-subspace projections $P_c$, but the paper does not explain how $P_c$ is chosen or estimated.

### Soundness
2

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
4

### Summary
This paper explores the idea of localzied unlearning where the activation of some layers within a model to achieve more surgical, and more effective unlearning. Authors observe that some layers might have more task-related effect, that is, when they are replaced with that of the unlearned model, forgetting happens while also they have lower leakage, i.e., lower degradation in model utility.

They observe that this method of patching is effective, and achieve competitive peformance, across different models and benchmarks.

### Strengths
I find the idea of localizing where unlearning happens compelling: identifying the most important layers and modifying only their weights can yield more targeted, surgical unlearning with fewer side effects. The experiments and analysis explaining why this works are convincing and well supported. The comparison between structural finetuning and post-hoc edits is also valuable and clearly executed.

### Weaknesses
I don’t see major issues with this work, aside from the choice of $k$, which may be task- and model-dependent. Insight into how to choose k and an analysis of robustness to $k$ would be valuable. Relatedly, the validation depends on the chosen metrics; different validation scores may yield different outcomes. For example, TOFU shouldn’t be assessed solely by FQ and MU. This is a minor point, though, as the paper also demonstrates generalization across other metrics.

Improvements are minor, which is acceptable since I view this work as more interpretability-oriented than method-oriented. The finding that different layers have different effects on unlearning is interesting, though related ideas have been explored in [1–4].

It would also be informative to evaluate on the RESTOR benchmark [5], which focuses on data-level unlearning and restorative ability. Many methods perform poorly there; I’m curious whether the proposed approach can actually improve those results, rather than merely offering more surgical edits.


------

[1] Meng, Kevin, et al. "Mass-editing memory in a transformer." arXiv preprint arXiv:2210.07229 (2022).

[2] Gupta, Akshat, Anurag Rao, and Gopala Anumanchipalli. "Model editing at scale leads to gradual and catastrophic forgetting." arXiv preprint arXiv:2401.07453 (2024).

[3] Basu, Samyadeep, et al. "On mechanistic knowledge localization in text-to-image generative models." Forty-first International Conference on Machine Learning. 2024.

[4] Zarei, Arman, et al. "Localizing Knowledge in Diffusion Transformers." arXiv preprint arXiv:2505.18832 (2025).

[5] Rezaei, Keivan, et al. "RESTOR: Knowledge Recovery in Machine Unlearning." arXiv preprint arXiv:2411.00204 (2024).

### Questions
They are discussed in Weakness section.

### Soundness
3

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
3

### Summary
This paper investigates a fundamental yet underexplored question in the field of machine unlearning for large language models (LLMs), how internal representations are structurally affected during unlearning. The authors introduce the concept of latent knowledge fragility, which captures how certain layers in LLMs are disproportionately damaged when removing specific knowledge. Through rigorous layer-wise analyses, they reveal that middle layers are especially fragile, leading to overall performance degradation.
To mitigate this problem, the paper proposes Component-wise Replacement Unlearning (CRU), a post-hoc, training-free method that selectively restores fragile components of the model from the original parameters. By identifying critical layers based on Model Utility (MU) and Forget Quality (FQ) scores, CRU reassembles a “patched” model that maintains the unlearning effect while recovering general capabilities.

### Strengths
1. The introduction of latent knowledge fragility provides a new theoretical lens to understand how unlearning interacts with internal representation geometry. Theoretical results (e.g., Proposition 1) give a principled basis for quantifying representational drift via CKA similarity.

2. The study systematically quantifies per-layer influence on model retention and forgetting, offering concrete insights into how unlearning propagates across the transformer hierarchy.

3. CRU requires no retraining or gradient computation, operating entirely post-hoc. Despite its simplicity, it achieves a superior trade-off between FQ and MU across various benchmarks (TOFU, MUSE, WMDP) and model scales (llama-3.2 1B & 3B, llama-2 7B, phi-3.5-mini, zephyr-7B).

### Weaknesses
1. Although evaluated empirically, restoring original parameters from the base model always entails some risk of reviving forgotten knowledge, especially if the targeted information is localized within the restored layers. While each layer inherently contains a mixture of heterogeneous knowledge, the proposed method restores certain layers as a whole, overlooking the fact that diverse types of information are entangled within a single layer. Although the method demonstrates improved performance over baselines on several unlearning benchmarks, it remains uncertain whether such advantages would generalize to more comprehensive or heterogeneous datasets.

2. A deeper comparison with the more straightforward and practical "structural freezing" approach is necessary. In Lines 314–317, the paper discusses the comparison between replacement and structural freezing. However, it is unclear which layers were frozen during the structural freezing experiments. If only the middle layers were frozen, further analysis would be necessary. The observation that middle layers exhibit larger representation differences was derived after unlearning; thus, freezing those same layers during unlearning may not be a valid comparison, as it is an independent analysis. A more comprehensive analysis (e.g., freezing up to varying depth levels, or freezing varying depth layers) would strengthen the validity of the comparison.

3. The formal analysis (Proposition 1) characterizes lower bounds of representational drift but does not model causal interactions between layers or nonlinear dependencies within the transformer stack. The paper also introduces several formal definitions (e.g., patched model, component-wise partitioner), which establish the theoretical foundation of the proposed method. However, these formulations mainly serve as notational formalities rather than substantial theoretical contributions.

### Questions
See weaknesses

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
This paper investigates how different layers of LLMs are affected during unlearning. The authors introduce the concept of latent knowledge fragility, which measures the sensitivity of internal representations under unlearning updates. Using layer-wise model patching, they analyze how shallow, middle, and deep transformer layers behave when unlearned. They observe that middle layers are particularly fragile and play a central role in balancing knowledge removal and model utility.

Based on this insight, the authors propose Component-wise Replacement Unlearning (CRU) — a lightweight, post-hoc framework that selectively restores fragile layers from the original model (without additional fine-tuning). They claim that CRU achieves a better trade-off between forgetting and retention compared with gradient-based methods, across benchmarks like TOFU, MUSE, and WMDP.

### Strengths
1. The paper focuses on analyzing unlearning effects from the viewpoint of internal representations, which is less explored in existing LLM unlearning literature.
2. CRU avoids extra training and introduces a modular, interpretable post-hoc procedure that could be useful for debugging or analysis.
3. Results are reported on multiple models and datasets, showing stable improvement in utility-forgetting trade-off.

### Weaknesses
1. The selection process of fragile layers (via top-k scores) is heuristic. It’s unclear why summing ranks of FQ and MU is an optimal or principled criterion. No ablation is presented to show sensitivity to k, nor justification for why layer replacement (discrete switch) is better than, say, soft interpolation or fine-tuning (though mentioned briefly, not analyzed rigorously).
2. The work claims novelty in “component-wise patching,” yet similar ideas exist in interchange interventions, representation surgery, and parameter-efficient editing. The novelty should be further clarified. 
3. Reported improvements (Table 1–3) are sometimes numerically small and may not be statistically significant. This raises the concern of the proposed method.
4. The observation that middle layers are “fragile” may correlate with unlearning sensitivity but doesn’t prove causality, e.g., deeper layers might also appear fragile under different architectures or datasets. The U-shaped pattern (Figure 3) could simply reflect gradient distribution rather than intrinsic semantic hierarchy.

### Questions
1. How exactly is “latent knowledge fragility” quantitatively defined? Is it directly proportional to the observed drop in model utility, or is there a separate latent-space measure?
2. Why sum the rank of MU and FQ to get the patching score? Would a weighted or Pareto-based method yield different results?
3. How do the authors ensure that observed middle-layer fragility is not a byproduct of model scaling or optimizer dynamics rather than inherent semantic abstraction?
4. Have the authors tested CRU on models larger than 7B (e.g., 13B or 70B)? Does the same “middle-layer fragility” pattern persist across scales?
5. Why are methods like MEMIT[1], or KnowledgeEditor[2] not included as baselines, given their similarity to modular patching?
6. CRU is said to be “lightweight,” yet it involves multiple forward validations per layer. What is the computational overhead compared to single-pass fine-tuning methods?
7. Some Tables and Figures are dense; figure captions do not clearly indicate what metrics or settings are used (Figure 4).

[1] Mass-Editing Memory in a Transformer
[2] Editing Factual Knowledge in Language Models

### Soundness
3

### Presentation
2

### Contribution
3
