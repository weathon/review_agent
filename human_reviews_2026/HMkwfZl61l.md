# Data Selection for Fine-tuning Vision Language Models via Cross Modal Alignment Trajectories

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Data-efficient learning aims to eliminate redundancy in large training datasets by train-
ing models on smaller subsets of the most informative examples. While data selection
has been extensively explored for vision models and large language models (LLMs), it
remains underexplored for Large Vision-Language Models (LVLMs). Notably, none of
existing methods can outperform random selection at different subset sizes. In this work,
we propose the first principled method for data-efficient instruction tuning of LVLMs. We
prove that examples with similar cross-modal attention matrices during instruction tun-
ing have similar gradients. Thus, they influence model parameters in a similar manner
and convey the same information to the model during training. Building on this insight,
we propose XMAS, which clusters examples based on the trajectories of the top singu-
lar values of their attention matrices obtained from fine-tuning a small proxy LVLM. By
sampling a balanced subset from these clusters, XMAS effectively removes redundancy in
large-scale LVLM training data. Extensive experiments show that XMAS can discard 50%
of the LLaVA-665k dataset and 85% of the Vision-Flan dataset while fully preserving per-
formance of LLaVA-1.5-7B on 10 downstream benchmarks and speeding up its training
by 1.2×. This is 30% more data reduction compared to the best baseline for LLaVA-665k.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces XMAS, a framework for data selection in fine-tuning large vision-language models (LVLMs). The key idea is that examples with similar cross-modal attention matrices during instruction tuning also have similar gradients, and hence, are redundant for training. XMAS fine-tunes a small proxy LVLM to compute cross-modal alignment trajectories (the top singular values of attention matrices across training checkpoints), clusters examples with similar trajectories, and samples a balanced subset emphasizing stable trajectories.

### Strengths
**Theoretical Grounding:** The idea of clustering cross-modal alignment trajectories derived from a proxy LVLM is original and supported by theoretical justification.

**Strong Empirical Results:** Comprehensive experiments on two large datasets demonstrate clear improvements in data reduction efficiency (up to 30% better than the best baseline) with minimal performance loss.

### Weaknesses
**Incremental Technical Novelty:** While the theoretical analysis is elegant, the practical pipeline (proxy training + clustering + balanced sampling) remains conceptually close to COINCIDE (activation-based clustering). The novelty is incremental, largely reinterpreting feature-space redundancy through attention SVD trajectories.

**Dense Presentation:** The paper is well-organized but mathematically dense, with long proofs occupying many pages. Some key intuitions—such as why top singular values reflect gradient dynamics—could be better highlighted through figures or simplified derivations.

**Limited Validation:** The derivations (Theorems 4.1–4.2) make restrictive assumptions—single-layer transformers, small RMS gains, bounded curvature—that are not empirically verified in large-scale LVLMs. The link between these theoretical results and the practical proxy implementation is assumed rather than demonstrated.

### Questions
Please refer to the weakness section.

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
This paper proposes Cross-Modal Alignment SVD (XMAS), a data selection method that identifies and removes redundant training examples in LVLM by clustering samples based on their cross-modal attention trajectories. To achieve this, XMAS fine-tunes a small proxy model and tracks the evolution of  alignment scores for each sample across checkpoints. By sampling a balanced subset of stable samples, XMAS effectively eliminates redundancy without compromising performance.  Experiment results on LLaVA-665K and Vision-Flan datasets demonstrate the effectiveness of the proposed approach.

### Strengths
1. This work provides substantial theoretical proofs to elucidate the limitations of existing methods and justify the necessity of the proposed XMAS, which is highly convincing.
2. The paper conducts extensive experiments and comparisons against a wide range of baseline methods, making its findings particularly solid.
3. Experimental results demonstrate that XMAS can achieve nearly identical performance at a reduced training cost, underscoring its effectiveness.

### Weaknesses
1. In lines 52-54, the authors state that "the part of the gradient corresponding to multimodal alignment dominates the parts corresponding to individual modalities." Is this claim supported by any prior work or empirical experiments?
2. The writing could be improved, as several informal terms are used (e.g., "big training data").
3. Additional evaluations on diverse Multimodal Large Language Models (MLLMs) could be conducted to further substantiate the generalizability of XMAS.

### Questions
Please refer to weakness above.

### Soundness
3

### Presentation
2

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
This paper addresses the data inefficiency of instruction tuning for Large Vision-Language Models, noting that existing methods often fail to outperform random selection. To solve this, the paper proposes XMAS, a method centered on the idea that data redundancy should be defined by gradient similarity during training. As direct gradient computation is infeasible , XMAS instead trains a small proxy LVLM and tracks each data point's "cross-modal alignment trajectory", defined as the evolution of the top singular values of its cross-modal attention matrix across training checkpoints. By clustering these trajectories and performing a balanced sampling of the most stable examples from these clusters, XMAS effectively removes data redundancy. Experiments demonstrate that this method can discard 50% of the LLaVA-665k dataset and 85% of the Vision-Flan dataset while maintaining full performance , achieving a 1.2x training speedup and significantly outperforming existing baselines.

### Strengths
1. XMAS is the only method in the experiment that consistently outperforms random selection across both LLaVA-665k and Vision-Flan datasets, at all tested data budgets (5% to 50%).
2. Achieving full-data performance while dropping 50% of LLaVA-665k and 85% of Vision-Flan  is a remarkable achievement.
3. The method not only improves data efficiency but also delivers a 1.2x end-to-end training speedup, including the selection overhead, making it a genuinely practical tool.

### Weaknesses
1. This is the paper's primary weakness. Its theoretical justification relies on an extremely simplified single-layer, single-head, L2-loss model. The authors fail to provide sufficient argument for why this theory should extrapolate to the practical deep, multi-head, cross-entropy-loss LLaVA models. This makes the theoretical section seem more like "post-hoc decoration" than a solid foundation for the method's success.
2. The XMAS method itself introduces new, and seemingly sensitive, hyperparameters: the number of clusters K and the number of checkpoints T. The ablation studies show that the choice of K and T significantly impacts performance. This creates a new problem: users may need to tune K and T before using XMAS, adding to the method's cost and complexity.
3. The method assumes the alignment trajectories of the proxy model (TinyLLaVA-2B) and target model (LLaVA-1.5-7B)  are similar. While Figure 4 provides evidence for this , the experiment is limited to two scales of the same model family (TinyLLaVA). It is unclear if XMAS would remain effective if the proxy and target models had significant architectural differences (e.g., different LLM backbones or Vision Encoders).

### Questions
Please respond to the weaknesses I mentioned above.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles the problem that LVLMs need huge multimodal instruction-tuning datasets, but a lot of the data is redundant, and existing heuristic data-selection methods (CLIP score, influence, uncertainty, de-dup, activation clustering) don’t reliably beat random selection. The authors argue that redundancy should be defined in terms of gradient similarity—examples that push the model in almost the same direction are redundant—but computing gradient similarity for LVLMs is infeasible. They therefore analyze a (simplified) transformer and show that the distance between two examples’ cross-modal attention matrices upper-bounds their gradient distance, and this relation remains stable across nearby checkpoints (when the Hessian is small, as in instruction tuning). Based on this, they propose XMAS (Cross Modal Alignment SVD): fine-tune a small proxy VLM, track the leading singular values/trajectories of cross-modal attention for each example over a few checkpoints, cluster examples with similar “attention trajectories,” and sample balanced subsets from clusters. XMAS can drop 50% of LLaVA-665k and 85% of Vision-Flan while keeping LLaVA-1.5-7B performance on 10 benchmarks and even speeding training by 1.2×.

### Strengths
1.Instead of ad-hoc scores, the paper gives a principled route from “we want to keep gradient-diverse examples” to “we can approximate gradient distance via cross-modal attention distance”. XMAS is designed to be workable: it uses a smaller proxy model and only a few checkpoints, then clusters, so it’s not “compute gradients for every example at every step” unrealistic.

2.Empirical results show that you can drop 50% (LLaVA-665k) or even 85% (Vision-Flan) and match downstream performance — and beat prior data-selection baselines — is a compelling empirical story.

### Weaknesses
1.the major contribution of this paper is to use the computation of attention similarity to help select high-quality training examples. As VLm contains a lot of attention layers and attention head, compared with gradient-based data selection strategy, the extraction and computation of attention scores are also memory- and time-consuming. More detailed memory use and time cost for the feature extraction stage should be shown in this work, to highlight the efficiency contribution. 

2.In experiment part, the efficiency and effectiveness comparison with baselines like LESS and TIVE, have not been exhibited. In these two papers, they also use LoRA to reduce the memory and computation cost. It might have a lower cost than attention similarity computation in practice.

### Questions
Please refer to the questions in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
