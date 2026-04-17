# Graft: Integrating the Domain Knowledge via Efficient Parameter Synergy for LLMs

- Decision: Reject
- Scores: 2, 6, 6, 6, 4

## Abstract
Large Language Models (LLMs) have achieved success across various domains. However, their applicability tends to degrade when confronted with different types of data inputs, especially for LLMs that have been fine-tuned for specific tasks. Despite its importance, the study of knowledge sharing among domain-specific LLMs—such as those trained for mathematics or code—remains largely underexplored. To address the fragmentation of knowledge across domain-specialized LLMs, we propose a unified parameter integration framework that enables modular composition of expert capabilities. Our method is grounded in a novel Compatibility-Aware Parameter Splicing (CAPS) strategy, which leverages both local functional attribution and global information-theoretic signals to guide selective parameter fusion. By extending this mechanism to the low-rank adaptation layer granularity, we ensure efficient integration with minimal inference overhead. Furthermore, we introduce a domain compatibility scoring mechanism that quantifies inter-expert alignment at the activation level and correlates with downstream task utility. This principled fusion protocol allows the final model to synergize heterogeneous expertise while preserving structural modularity. Extensive evaluations across diverse language and reasoning benchmarks validate the effectiveness of our framework, offering a scalable path toward compositional, domain-adaptive LLMs. Our project is available at https://anonymous.4open.science/r/Graft-8213.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Graft, a dual‑gate parameter fusion method that mixes weights from domain experts using a local channel‑wise gate and a global entropy‑based gate. The authors propose a compatibility score derived from activations helps select expert pairs, and a semantic subsampling pipeline builds representative datasets for fusion. Experiments on Qwen2/phi‑3 LLMs and Qwen2‑VL MLLMs show the effectiveness of Graft.

### Strengths
1. The dual-gate design is considered and is applicable to full fine-tuning and LoRA by fusing both attention and MLP layers.
2. This paper porvides experimental results on both LLMs and MLLMs, demonstrating the effectiveness of the proposed method.
3. The introduced activation‑based compatibility metric is practical and it shows how this metric correlates with improvements across domains.

### Weaknesses
1. The use of weight‑entropy is mostly heuristic and lacks theoretical support, and alternative signals such as spectral norms are not compared. This makes it hard to attribute gains to the specific signal design.
2. The proposed method relies on small datasets for gate learning and compatibility estimation, so calling it "data‑free"  or "training free" is not accurate.
3. The experiments are mainly conducted on small LLMs/MLLMs, lacking results on 7B level models. It is unclear if the method scales smoothly to larger models. Also, it is suggested to add missing results of baseline methods in Table 2 by running baselines in the same settings to ensure fair comparison instead of simply refering.
4. When implement the method on larger models, it would be good to see results and analysis on merging efficiency.
5. I'm also curious about if it is possible to compute the compatibility score at inference time to enable dynamic gating rather than a static pre‑merge decision.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
I'm not confident enough to provide technical assessment to this paper.

### Strengths
n/a

### Weaknesses
n/a

### Questions
n/a

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
The paper proposed the method Graft, aiming to integrate the parameters from multiple fine-tuned models efficiently so that the base model is competitive in the corresponding tasks. 

The framework is a combination of the model fusion and data exploitation methods. In terms of the model fusion, Graft calculates the local and global weights according to the difference between the base model and the graft (target) model at the channel and global levels, and then combines the parameters of the two models with the weighted (as a function of the global and local weights) average. However, not all models are good fits for fusion. Thus, the method proposes to do a dataset compatibility analysis according to the activation pattern of the model w.r.t the dataset. If the model is compatible with the target dataset, it would be eligible for the fusion.

In terms of data exploitation, the method does a representative subsampling for the samples close to the cluster means.

### Strengths
1. The paper explores model fusion across domains, particularly integrating models specialized in mathematics and code—an area that remains largely underexplored.

2. The proposed Graft method demonstrates strong and consistent performance across multiple datasets, often outperforming or matching domain-specific models.

3. By combining local and global adjustments, the method achieves fine-grained control over the fusion process, leading to improved overall performance.

4. Furthermore, Graft incorporates a compatibility analysis mechanism to assess the alignment between the source model and the target datasets, ensuring successful and meaningful model fusion.

### Weaknesses
1. Lack of design intuition and theoretical grounding:
The proposed methods appear somewhat rough, or at least not well-explained. The paper does not provide sufficient intuition behind the design of the fusion strategy, particularly regarding Eqs. (5), (6), and (12–14). Moreover, the connections to existing model fusion techniques (mathematical formulations) are not established, making these formulations seem unsupported. A more comprehensive discussion of related work and the rationale behind the design choices would greatly improve clarity and credibility.

2. Missing computational complexity analysis:
Although the authors claim that the Graft method is computationally efficient, no comparison or quantitative analysis of its complexity is provided. Including such results would help substantiate the efficiency claims.

3. Absence of ablation and component analysis:
Given that the framework involves a multi-stage pipeline with several intermediate components, an ablation study is essential to demonstrate the contribution of each step. For instance, evaluating the effect of the representative data subsampling method would offer valuable insight into the method’s internal dynamics.

4. Unclear parameter selection and lack of sensitivity analysis:
The procedure for determining key parameters—such as the value of $k$ in representative subsampling or the threshold used in compatibility analysis—is not explained. Moreover, a sensitivity analysis is missing, leaving readers uncertain about how robust the results are to these hyperparameter choices.

### Questions
Please refer to the weaknesses section.

Besides, in the ablation study on gating components (Table 5), the MME with only local is the best, but the authors claim the combination of local and global is leading. Is there a typo in the table?

### Soundness
2

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
2

### Summary
The paper proposes a dual-gated parameter fusion framework named Graft for integrating domain-specialized models. At the local scale, a learnable gate assigns channel-wise fusion weights based on parameter differences; at the global scale, an entropy-based score modulates a single fusion weight to mitigate cross-domain conflicts. The paper also introduces an activation-driven compatibility score to predict whether two experts will fuse well, and a representative data subsampling pipeline to keep costs manageable. Experiments on LLMs/MLLMs show improvements over Task Arithmetic, TIES and DARE across Math, Code, and several multimodal benchmarks, with ablations indicating the dual-gate design outperforms single gates.

### Strengths
1. Addressing domain fragmentation in practice (especially with LoRA experts) is timely and relevant; evidence suggests the framework scales beyond pairwise fusion without severe catastrophic forgetting in the tested settings.
2. The dual-gate idea—combining channel-wise (local) gating with an entropy-based (global) gate—offers a principled way to balance complementarity vs. interference, going beyond element-wise or sign-based heuristics used in prior merging methods. 
3. The activation-driven compatibility metric is a practical contribution for select-then-fuse, reducing trial-and-error when pairing experts.

### Weaknesses
1. While the paper includes multimodal evaluations (MathVista, MMMU, MME) and some multi-domain fusions (adding Finance/Medical adapters), the core LLM story remains Math+Code-centric. Evaluations on more domains are encouraged.
2. The approach relies on representative data subsampling (embeddings→K-Means→centroids), which somewhat INTERVENES the training stage. IMO, a good merging algorithm shall outperform baselines on any model groups (trained with or without data subsampling). Could the author provide the comparison results without the data subsampling?

### Questions
Pls see the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Graft, a novel parameter fusion framework for integrating multiple domain-specialized Large Language Models (LLMs) or LoRA-adapted models into a unified model without retraining. The method introduces a dual-gate fusion mechanism that combines: 1. **Local weight adjustment**: a channel-wise gating network that quantifies parameter differences to emphasize locally important features; 2. **Global weight adjustment**: an entropy-based signal capturing distributional information content for global parameter alignment. 

To ensure reliable fusion, the authors propose a dataset compatibility analysis based on activation statistics (magnitude, sparsity, variance) and a representative data subsampling approach using K-Means clustering for semantic diversity.  

Empirically, Graft outperforms several baselines (Task Arithmetic, TIES-Merging, and DARE) across diverse LLMs and multimodal models (e.g., Qwen2, Phi-3, Qwen2-VL), achieving superior results on both domain-specific (Math, Code) and general benchmarks (MMLU, TruthfulQA). The framework scales effectively to multi-domain fusion while maintaining performance stability.

### Strengths
**Originality**:
1. The dual-gate mechanism elegantly combines channel-level and entropy-based fusion for adaptive parameter integration.
2. The activation-based compatibility metric provides a principled criterion for selecting which domain experts to fuse.
3.  The semantic-aware data subsampling step is an innovative procedural contribution for efficiency and data balance.

**Clarity**:
1. The paper is clearly written and visually well-organized.
2. Figures effectively illustrate the pipeline and mechanism (e.g., Fig.1–2), and Algorithm 1 concisely summarizes the method.

**Significance**:

1. Addresses a timely and practical challenge in efficient model merging and domain adaptation for LLMs.
2. Empirically strong, achieving notable gains (up to $+8$–$10$ points) on specialized benchmarks without degrading general performance.
3.  Offers a scalable, modular paradigm for compositional LLM construction.

### Weaknesses
**Theoretical justification**: The link between entropy and representational richness remains heuristic; additional theoretical or empirical validation would strengthen the argument.

**Efficiency analysis**: The computational cost of training or applying the gating network is not reported; an explicit runtime comparison would improve transparency.

**Baselines**: Some baselines (e.g., DARE, TIES-Merging) may not have been fully optimized for large-scale settings, potentially affecting fairness.

**Interpretability**: The work lacks qualitative visualization of how the dual gates behave across domains or layers.

**Ablation completeness**: While the compatibility metric is correlated with performance, a random or naive pairing control would clarify its contribution.

### Questions
1. How sensitive is the global weighting performance to the constants $a$ and $c$ in Eq.~(4)? Could the authors provide an intuition or sensitivity analysis?
2.  What is the computational overhead of training the gating network $\phi(\cdot)$ relative to the base model size?
3.  In multi-domain fusion (Table~4), performance gains plateau. How does Graft handle conflicts when merging more than four experts?

### Soundness
2

### Presentation
3

### Contribution
2
