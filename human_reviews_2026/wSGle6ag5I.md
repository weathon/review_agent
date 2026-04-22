# Improving Diffusion Models for Class-imbalanced Training Data via Capacity Manipulation

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 6, 6, 6, 6

## Abstract
While diffusion models have achieved remarkable performance in image generation, they often struggle with the imbalanced datasets frequently encountered in real-world applications, resulting in significant performance degradation on minority classes. In this paper, we identify model capacity allocation as a key and previously underexplored factor contributing to this issue, providing a perspective that is orthogonal to existing research. Our empirical experiments and theoretical analysis reveal that majority classes monopolize an unnecessarily large portion of the model's capacity, thereby restricting the representation of minority classes. To address this, we propose Capacity Manipulation (CM), which explicitly reserves model capacity for minority classes. Our approach leverages a low-rank decomposition of model parameters and introduces a capacity manipulation loss to allocate appropriate capacity for capturing minority knowledge, thus enhancing minority class representation. Extensive experiments demonstrate that CM consistently and significantly improves the robustness of diffusion models on imbalanced datasets, and when combined with existing methods, further boosts overall performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work investigates the challenge of generative modeling for imbalanced datasets. The hypothesis is that the poor generation quality for minority classes is primarily caused by an imbalance in "model capacity," where the model's learning resources are disproportionately occupied by the head (majority) classes. To address this, the paper introduces a novel technique named Capacity Manipulation (CM), which explicitly reallocates and reserves model capacity for the tail (minority) classes. The proposed method employs a low-rank decomposition of the model's parameters, enabling fine-grained control over capacity allocation. A bespoke capacity manipulation loss function is introduced to ensure sufficient capacity is dedicated to learning the features of minority classes, leading to a significant enhancement in their generative representation. The claims are substantiated by comprehensive experimental results, and the overall methodology is presented with clear and coherent logic.

### Strengths
1. I find this approach remarkably novel in how it attributes the class imbalance problem to "uneven model capacity allocation." It represents a significant departure from traditional paradigms like data resampling or loss re-weighting, introducing a fresh perspective by intervening directly at the model parameter level. By the way, I'm also curious if the author's method could be applied to long-tail recognition tasks (e.g., with ResNeXt-50 on CIFAR). No detailed explanation is needed if the implementation is complex—I'm simply wondering about its potential.
2. The design of  loss function is exceptionally clear in its objective. By creating a "push-pull" dynamic between 'consistency' and 'diversity', it effectively channels distinct knowledge into separate parameter subspace.
3. The paper doesn't just rest on solid experimental results; it also provides theoretical analysis (Theorems 2.1 and 3.1) to substantiate its core thesis: that majority classes indeed dominate parameter updates and that low-rank decomposition can effectively mitigate this dominance.
4. The experimental validation is remarkably comprehensive. It covers a wide range of datasets (from simple to complex, low-res to high-res), various imbalance ratios, and multiple evaluation metrics, all benchmarked against strong baseline methods.

### Weaknesses
1. I'm also curious if the author's method could be applied to long-tail recognition tasks (e.g., with ResNeXt-50 on CIFAR). No detailed explanation is needed if the implementation is complex—I'm simply wondering about its potential.
2. My main question is about the capacity 'reservation.' The structure of the parameter decomposition seems to be fixed. This makes me wonder: is this 'hard partitioning' approach truly optimal? Could there be a way for the model to dynamically and adaptively decide how much capacity to allocate to each component during training, rather than relying on a predefined split?
3. Are there any toy experiments that can visually illustrate this? For instance, using the two-class example you mentioned, could you show how the majority class ends up occupying most of the model's parameter capacity?
4. I think this assumption has some limitations, especially with varying balance ratios like in ImageNet-LT. For example, in an extreme case with one head class and 999 tail classes, is a single, the setting of rank still appropriate? Or does the rank itself need to be adjusted based on class frequency?
5. To empirically validate the hypothesis, is it possible to visually demonstrate that the class-specific parameters specialize in learning features unique to minority classes, while the class-agnostic parameters focus on capturing generic features dominated by the majority (head) classes? We propose achieving this through visualization techniques.
6. My last question is about the diversity within the tail itself. Can you visualize them?

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the poor minority-class performance of diffusion models trained on long-tailed data, arguing that a key culprit is capacity misallocation—majority classes dominate parameter updates and monopolize representational space. 
To tackle this, it proposes Capacity Manipulation (CM): each weight matrix is decomposed into a general/majority component and a reserved low-rank minority component, and training employs a capacity-manipulation loss that enforces consistency for majority classes while promoting diversity for minority classes. 
At inference, parameters are merged, introducing no additional latency. 
Across imbalanced CIFAR-10/100, CelebA-HQ, ImageNet-LT, iNaturalist, and ArtBench-10 (including Stable Diffusion fine-tuning), CM improves FID/KID and delivers especially strong gains on Medium/Few splits over strong baselines (e.g., CBDM, OC), while remaining orthogonal and complementary to them.

### Strengths
The paper offers a clear and original lens—capacity allocation—and introduces a simple, effective mechanism that reserves low-rank capacity for minority classes, moving beyond reweighting or oversampling.
Method quality is strong: the parameter split plus a consistency/diversity loss is minimally invasive, theoretically motivated by gradient/representation analyses, and incurs no inference overhead due to weight merging.
Empirically, results are broad and convincing across multiple datasets/backbones (including SD fine-tuning), with especially large gains on Medium/Few splits and stable ablations over ranks and loss weights.
The approach is practical and orthogonal to existing long-tail remedies (e.g., CBDM, OC), making it easy to adopt and combine for further improvements.

### Weaknesses
1. The paper should more clearly distinguish CM from class-balanced objectives, reweighting/oversampling, class-specific adapters/LoRA, and Mixture-of-Experts. Add a side-by-side comparison and reproduce at least one adapter/MoE-style baseline under matched compute.

2. The analysis explains majority gradient dominance and motivates reserving rank, but does not specify conditions ensuring no loss of global likelihood or bounds on interference.

3. Most results are class-conditional image benchmarks.
For instance, text-to-image (multi-attribute, compositional) and multi-label long-tails are underexplored.

### Questions
1. How is the minority/majority split determined, and how sensitive are results to this choice under dataset drift or rebalancing?

2. When merging weights at inference, how do the authors prevent cross-talk between the general and minority subspaces?

3. Does reserving capacity degrade majority-class fidelity or diversity in any regimes?

4. What are the exact training overheads introduced by the extra low-rank factors and CM loss? Do gains persist under tight compute budgets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Capacity Manipulation (CM), a method to improve diffusion models trained on class-imbalanced data. It identifies that majority classes dominate model capacity, limiting minority representation. CM explicitly reserves capacity for minority classes through low-rank parameter decomposition and a capacity manipulation loss that balances consistency and diversity. Experiments on multiple benchmarks show that CM consistently enhances minority-class generation quality and overall robustness.

### Strengths
1. The paper is clearly written, well-structured, and easy to follow.
2. The proposed Capacity Manipulation (CM) method is conceptually simple yet effective, relying on low-rank decomposition and a targeted regularization loss to reserve model capacity for minority expertise.
3. Theoretical analyses provide solid intuition about how imbalance affects parameter updates and how CM mitigates this effect.
4. Extensive experiments across small- and large-scale datasets convincingly demonstrate that CM improves minority-class quality without degrading majority-class performance.

### Weaknesses
1. The calculation of loss change in figure 1(b) is not explained.
2. Although the authors evaluate CM across multiple datasets, there is limited discussion of failure cases or sensitivity to extreme imbalance ratios beyond 100:1.
3. Some comparisons (e.g., with Overlap Optimization) are only mentioned in passing, a direct experimental comparison would strengthen claims of superiority.

### Questions
N/A

### Soundness
3

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
4

### Summary
This paper addresses the problem of class imbalance in diffusion models, which leads to poor generation performance on minority classes. The authors identify model capacity allocation as a key overlooked factor, where majority classes dominate model parameters, leaving insufficient capacity for minorities. To mitigate this, they propose Capacity Manipulation (CM), a method that reserves model capacity for minority classes via low-rank decomposition of parameters and a novel capacity manipulation loss. The method is orthogonal to existing approaches and does not increase inference cost. Extensive experiments on multiple datasets demonstrate consistent improvements in minority-class generation without sacrificing majority-class performance.

### Strengths
(1) The method is well-motivated, supported by both empirical observations (e.g., pruning sensitivity) and theoretical analysis (Theorems 2.1 & 3.1). The experimental setup is rigorous, covering multiple datasets, architectures, and metrics.

(2) The paper offers an orthogonal viewpoint on class imbalance in diffusion models by focusing on model capacity allocation, diverging from prior works that primarily emphasize loss reweighting or knowledge transfer (e.g., CBDM and OC). The integration of low-rank decomposition with a tailored loss function represents a creative combination of ideas from parameter-efficient fine-tuning and imbalanced learning for targeted capacity reservation.

### Weaknesses
(1) The term "capacity" is not clearly defined. Is it the number of parameters, the magnitude (e.g., L1-norm) of the weights or something else? The pruning experiment suggests a link to weight magnitude, but this connection is not explicitly made or theoretically grounded. Therefore, " capacity" remains a somewhat vague concept.

(2) The capacity manipulation loss is designed to force minority-specific knowledge into the low-rank adapter. A potential risk is that this adapter becomes too specialized, failing to leverage the shared, general features learned by the main model. This could limit its ability to generate diverse minority samples that still rely on common underlying features (e.g., a "rare breed of dog" should still benefit from general "dog" features). The paper does not discuss or analyze this potential limitation.

### Questions
(1) The method proposed in the paper primarily focuses on the context of known classes. A natural follow-up question is how Capacity Manipulation would perform in scenarios involving more compositional and fine-grained concepts. For example, in a dataset imbalanced towards "photos of cats" vs. "paintings of dogs," how would the model reserve capacity for the minority concept of "painting" style, which is orthogonal to the object "dog"? Does this framework extend to reserving capacity for concepts rather than just classes? 

(2) The method's architecture—using a LoRA-like adapter—makes a strong, implicit assumption: that "minority expertise" is inherently low-rank. What is the theoretical or empirical justification for this? One could easily argue the opposite: minority classes might be more complex and have a higher intrinsic dimensionality (e.g., "impressionist painting" vs. "female face") but are simply under-sampled. If the minority knowledge is, in fact, high-rank, then the fixed low-rank of the adapter would become the primary performance bottleneck, ironically limiting the minority class's capacity more than a standard full-rank model. How does CM cope with this potential issue? 

(3) The current formulation appears to use a single $\theta^e$ to capture the expertise for all minority classes collectively. On datasets with highly heterogeneous minority classes (e.g., the "Few" split in Imb. CIFAR-100 or ImageNet-LT, which can contain wildly different concepts), is it plausible that a single low-rank subspace can effectively represent this diverse and multimodal knowledge? Does this not create a new "capacity collapse" problem within the minority adapter itself? Have the authors considered a more flexible architecture, such as a Mixture-of-Experts (MoE) model for $\theta^e$, where different "experts" (adapters) are dynamically allocated to different minority clusters?

(4) The paper should discuss and cite relevant literature on reweighting or balancing techniques for generative models [1-5].

Reference:

[1] Xie et al. Doremi: Optimizing data mixtures speeds up language model pretraining. NeurIPS, 2023.

[2] Fan et al. DoGE: Domain Reweighting with Generalization Estimation. ICML, 2024.

[3] Kim et al. Training unbiased diffusion models from biased datase. ICLR, 2024.

[4] Li et al. Pruning then Reweighting: Towards Data-Efficient Training of Diffusion Models. ICASSP, 2025.

[5] Liu et al. RegMix: Data Mixture as Regression for Language Model Pre-training. ICLR, 2025.

### Soundness
3

### Presentation
3

### Contribution
3
