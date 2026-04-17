# Hybrid ACNs: Unifying Auto-Compressing and Residual Architectures

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
We propose **Hybrid Auto-Compressing Networks (H-ACNs)**, unifying ACNs and ResNets under a single mathematical formulation controlled by trainable scalar residual weighting parameters per layer. Through theoretical analysis, we show that both architectures represent points on a continuous spectrum, with traditional ACNs and ResNets as special cases. Our key contribution is demonstrating that H-ACNs, when initialized close to ACNs, match ResNets training efficiency while preserving ACN-like robustness and compression capabilities. Experiments across vision transformers, MLP-mixers, and GPT-2 architectures show that H-ACNs achieve training convergence on par with ResNets,  while maintaining ACNs superior noise robustness and generalization. Furthermore, we discover that learned residual weights exhibit distinct connectivity patterns across tasks, namely, vision tasks favor local connectivity patterns resembling early visual cortex processing, while language tasks converge to modular hierarchical inter-layer structures similar to hierarchical language processing regions. We also examine how initialization impacts performance and connectivity, challenging the universality of the common ResNet-like initialization of residual weights. Overall, our results establish Hybrid ACNs as a practical framework for efficiently balancing training speed and representation quality, while revealing principles of how functional connectivity patterns should vary across domains, modalities, and tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Hybrid Auto-Compressing Networks (H-ACNs), a simple scalar-gated formulation that interpolates between ACNs and ResNets. A per-layer trainable scalar controls the residual/long-skip mixture and induces a direct inter-layer connectivity matrix C used to analyze learned connectivity. Across MLP-Mixer/CIFAR-10, ViT/ImageNet-1k, and GPT-2-style pretraining, H-ACNs are claimed to match ResNet training efficiency while retaining ACN-like robustness/“auto-compression.” The paper further visualizes distinct connectivity patterns for vision vs. language and studies initialization sensitivity.

### Strengths
- Unified & lightweight parameterization. The interpolation adds only O(L) scalars and yields a tractable C to probe connectivity; implementation appears straightforward.
- Multi-domain evaluation. Results reported on vision and language.
- Connectivity analysis. Clear visualizations of the learned C and its training evolution; differences between modalities are interesting and potentially useful for architecture design.

### Weaknesses
- Despite repeatedly claiming “ResNet-like training efficiency,” there are no FLOPs, wall-clock, throughput, or peak-memory metrics, and no cross-scale profiling (CIFAR→ImageNet→GPT-2). This undercuts a central claim.
- Paper includes some small-scale comparisons (Mixer/CIFAR-10) to dense/concat-style variants, but omits such baselines at ImageNet and GPT-2 scales, where they matter most.
- Beyond defining C and showing empirical patterns, there are no formal results on gradient flow, expressivity, or stability (no theorems/lemmas/bounds).

### Questions
- Have you evaluated H-ACNs on canonical ResNet architectures (e.g., ResNet-18/34/50) with BatchNorm? If so, how does performance and training stability compare under matched compute (FLOPs, wall-clock, peak memory)? If not, could you comment on expected interactions with BatchNorm
- Have you explored per-channel or per-head trainable gates instead of a single scalar per layer? If yes, how do accuracy, convergence, and efficiency (FLOPs/throughput/memory) compare? If not, could you discuss whether finer-grained gating might offer benefits and what trade-offs you anticipate?

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
3

### Summary
The paper proposes Hybrid auto-compressing Networks (H-ACNs). These are a combination of the recently proposed ACN, and the standard residual architecture. The authors argue that H-ACNs achieve both the benefits of ResNets in terms of fast training, and ACNs in terms of robustness and compression capabilities.

Giving a 4 but would give a 5 if there was the option.

### Strengths
1. The paper proposes H-ACNs which bridge the gap between ACNs and ResNets, and appears to combine benefits of both. I found table 1 quite nice here (improved robustness of H-ACN)
2. In doing some, there is a nice framework for bridging between these extremes, by looking at the cumulative multiplicative interactions (eq 4).

### Weaknesses
1. Novelty: I think there are a lot of existing papers about scaled residual architectures with trainable scales (https://arxiv.org/abs/2002.10444, https://arxiv.org/abs/2003.04887, https://arxiv.org/abs/2010.12859), which are conceptually very similar to equation 2. This reduces the novelty of the proposed method and also is a missing baseline in the experiments. As I understand it, the only difference is that the output y is the sum of the f_i in HACN with coefficient 1, which I am not convinced is a big change. If it is an important change, then it would be good to demonstrate empirically. On that note, in figure 9 it appears like the 1 coefficients on the final column are not constant across layers after training, are these coefficients trainable?
2. Relatedly, one benefit of the ResNet is that you can write it such that each layer only depends on the previous layer and not on all previous layers (by the way the coefficients are constructed). Are there any computational considerations/downsides to H-ACN as you have to keep around each layers f_i in order to construct the output, particularly for the backward pass?
3. Some claims are overstated e.g. the "stronger generalisation capabilities of H-ACNs" appears to be a small difference in table 4c of 41.2 vs 41.8, without multiple seeds? Likewise, I am not convinced that looking at train loss in Figure 2 showing training speed is so insightful given that it is very easy to overfit on cifar10. Test loss would be better.

### Questions
- In Figure 4a/b, what is the motivation behind looking at intermediate layer perplexity?
- x axis is epoch in figure 2b.

### Soundness
2

### Presentation
2

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
The authors propose several inter-layer relationship metrics based on H-ACNs and draw analogies between trained weights and brain mechanisms. It is interesting.  But for the method part,  the authors appear to use the "unified framework of ACNs and Resnets" to compensate for shortcomings in innovation magnitude, experimental depth, and performance improvement. Unfortunately, since ACNs themselves are only a preprint without broad recognition or peer validation, this unified framework lacks persuasiveness.

### Strengths
1.	The authors propose several inter-layer relationship metrics based on H-ACNs and draw analogies between trained weights and brain mechanisms. In Chapter 4, they conduct experiments to explore this phenomenon; while no definitive explanation is provided, the results are indeed intriguing and worthy of further discussion.
2.	The paper presents improvements based on ACNs, claiming that ACNs possess strong representational capabilities and framing the theoretical unification of ACNs and Resnets as a key innovation. 
3.	The paper is written in clear, coherent English, ensuring good readability for readers.

### Weaknesses
1.	The authors do not elaborate on how the representational capabilities of ACNs specifically contribute to model performance. They cite Densenet to contextualize the importance of representational capabilities—yet Densenet outperformed Resnets with much fewer parameters at the time of its release. The authors, however, do not include sufficient information of Densenet in their experimental comparisons, nor do they show the clear advantage of parameter counts of H-ACNs. 
2.	The paper’s core innovation lies in introducing L trainable weights. Notably, it does not modify the inter-layer connection logic of ACNs, relying solely on these trainable weights to achieve unification with Resnets. This results in limited novelty in the proposed improvements.
3.	ACNs require caching outputs from all layers, and their inter-layer connection scheme appears to cause quadratic memory growth during forward propagation. The authors claim that H-ACNs solve this memory consumption issue, but there is a concern that their solution may significantly increase forward pass latency—essentially trading time for space. Please provide a detailed comparison of forward/backward pass latency and FLOPs against the ResNet and original ACN baselines to prove this solution is efficient in practice
4.	The authors only compare H-ACNs with Resnets. To better demonstrate H-ACNs’ effectiveness, more recent state-of-the-art methods should be included in comparisons. 
5.	The μ parameter has a significant impact on model performance. This raises doubts about the generalizability of the proposed method across different modalities.
6.	On GPT-2, the Perplexity (PPL) of H-ACNs is nearly identical to that of Resnets, with extremely limited improvement. No error analysis or calculation of average performance across multiple runs is provided, making it possible that the marginal improvement observed stems from randomness rather than the proposed architecture’s inherent advantages.
7.	The authors claim that H-ACNs can achieve lower PPL using only outputs from earlier layers, yet the PPL values across all layers are reported as similar. I don’t know if this feature makes a sense.
8.	There are several formula errors in the paper. For example, in Equation (2), the index i may incorrectly take a value of -1 (likely a typo), which could confuse readers and undermine the paper’s rigor.

### Questions
1.	Why does the number of paths in H-ACNs change with task complexity? Based on the paper, paths should exist as long as the residual weight ai != 0. In practical training, however, it is highly unlikely for ai to be trained to exactly 0. The authors should provide a more detailed explanation of how task complexity modulates path count.
2.	Please elaborate on the specific solution used by H-ACNs to address ACNs’ memory consumption issue. In particular, does this solution introduce new latency overhead during forward/backward propagation? Quantitative analysis (e.g., latency comparisons) should be provided to validate its efficiency.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Hybrid Auto-Compressing Networks (H-ACNs), a unified architecture that bridges Auto-Compressing Networks (ACNs) and Residual Networks (ResNets) through trainable scalar residual weights per layer. This formulation places ACNs and ResNets as two endpoints of a continuous spectrum, allowing H-ACNs to combine the fast, stable training of ResNets with the robustness, compression, and generalization strengths of ACNs.

### Strengths
1. The analysis of learned residual weights provides novel insights into task-specific connectivity, revealing biologically inspired patterns.
2. The paper unifies two architectures—ACNs and ResNets—under a single mathematical formulation.

### Weaknesses
1. The paper currently lacks a motivation. The main architectural change—introducing a trainable scalar on the residual connection—is presented without a clear explanation of why this interpolation is needed, in what scenarios ACN and ResNet are insufficient, or what concrete problem the proposed hybrid is intended to solve.

2. The method is not theoretically justified. Claims such as “preserving ACN-like robustness,” “better generalization,” or “auto-compression” are made, but no formal analysis is provided to explain why introducing layerwise residual scalars should lead to improved noise robustness or better inductive bias.

3. The experimental section is underpowered. For instance, the zero-shot evaluation covers only three tasks, which is unusually few compared to prior literatures and makes it difficult to assess the generality of the approach.

4. Even under the reported metrics, the improvements of H-ACN over ResNet are marginal, raising the question of whether the added architectural complexity is actually warranted.

5. The comparison to related methods is too narrow. Most experiments only compare against ResNet and ACN, and the only learnable-residual-weight baseline (DenseFormer) appears in the appendix and on CIFAR-10, which is not very convincing. There are several closely related approaches on learnable residual scaling (e.g., LAUREL [1], ReZero [2]) that are not compared or even discussed.

6. Minor issue: Line 351 — “Fig. 1” should be “Table 1.”

[1] Menghani, Gaurav, Ravi Kumar, and Sanjiv Kumar. "LAuReL: Learned Augmented Residual Layer." Forty-second International Conference on Machine Learning.
[2] Bachlechner, Thomas, et al. "Rezero is all you need: Fast convergence at large depth." Uncertainty in Artificial Intelligence. PMLR, 2021.

### Questions
Check above.

### Soundness
2

### Presentation
2

### Contribution
1
