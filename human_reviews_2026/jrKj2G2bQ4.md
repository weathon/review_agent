# MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Vision-language alignment in multi-modal large language models (MLLMs) relies on supervised fine-tuning (SFT) or reinforcement learning (RL). 
To align multi-modal large language models (MLLMs) in the post-training stage, supervised fine-tuning (SFT) is a stable choice but requires human annotations and lacks task generalizations, while Reinforcement Learning (RL) searches for better answers from reward signals but suffers from computational overhead and instability.
To achieve balance among scalability, efficiency, and alignment generalizations, we propose MergeMix, a unified paradigm that bridges SFT and RL with an efficient Token Merge based Mixup augmentation. As for the Mixup policy, we generate contextual aligned mixed images with the corresponding labels according to the merged attention maps with cluster regions. Then, we enhance the preference-driven paradigm for MLLMs by building preference pairs with raw images and MergeMix-generated ones and optimizing the soft preference margin with the mixed SimPO loss.
Extensive experiments demonstrate that MergeMix not only achieves dominant classification accuracy as an augmentation method but also improves generalization abilities and alignment of MLLMs, providing a new learning paradigm for preference alignment with training efficiency and stability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes MergeMix, a unified augmentation framework for visual and multimodal learning. It combines token merging and mixup to create attention-aware mixed images, which are then used to build preference pairs for training. The model treats clean samples as “winners” and mixed ones as “losers,” optimized with the SimPO loss to bridge supervised fine-tuning (SFT) and reinforcement learning (RL). Experiments on image classification (CIFAR100, ImageNet-1K, Stanford Cars) and MLLM benchmarks (LLaVA, Qwen2.5-VL) show that MergeMix improves accuracy, calibration, and training efficiency compared with existing mixup and alignment methods.

### Strengths
- The paper proposes a unified augmentation framework that connects supervised fine-tuning and preference optimization.

- The idea of using token merging to generate mixed samples is a reasonable extension of existing mixup methods and leverages attention information effectively.

- Experiments cover both image classification and multimodal benchmarks, showing consistent  improvements over baselines such as CutMix, MixPro, and SeVa.

- The method achieves competitive performance with lower computational cost and good training efficiency, suggesting potential for practical use.

### Weaknesses
- Incremental Overlap with Prior Work: While token merging and mixup are cleverly combined, the conceptual novelty may be perceived as moderate since both techniques are pre-existing; the innovation mainly lies in their integration.

- Marginal Multimodal Gains: Improvements on LLaVA benchmarks (0.8%) and Qwen2.5-VL (2.9%) are positive but not substantial, raising questions about statistical significance.

### Questions
- 1.Comparison with RL-based Preference Methods: Since MergeMix aims to bridge SFT and RL, it would be helpful to include direct comparisons with RL-based methods such as PPO or GRPO on MLLM benchmarks to quantify the alignment improvement.

- 2.Statistical Significance of Gains: Many reported improvements (e.g., +0.8% on LLaVA) are relatively small. Are these consistent across multiple random seeds? Please report standard deviations or significance tests to confirm reliability.

- 3.Generality Beyond ViT-based Models: MergeMix is mainly evaluated with ViT and DeiT architectures. Could the authors test it on convolutional backbones or hybrid models to verify broader applicability?

- 4.Computational Overhead of Token Merge: Although the paper claims better efficiency, the token merging and reconstruction steps may introduce overhead. Can the authors provide a detailed runtime breakdown to clarify the trade-offs?

- 5.Visualization and Qualitative Analysis: Including visual examples of merged attention maps or mixed images could help readers understand what semantic information MergeMix preserves compared to conventional mixup.

I will adjust the score based on the authors’ response.

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
2

### Summary
The paper proposes MergeMix, a unified augmentation paradigm for visual and multi-modal understanding. MergeMix builds on the concept of image mixing, specifically leveraging token merge strategies to generate mixed images and preference pairs for model alignment. The approach integrates ideas from supervised fine-tuning and preference-based optimization, aiming to improve both efficiency and robustness in multi-modal large language models (MLLMs) and image classification tasks. Experimental results show competitive performance across several benchmarks.

### Strengths
1. Quality: The experimental setup is rigorous, with thorough benchmarking and ablation studies.
2. Clarity: The methodology and results are clearly explained.
3. Significance: The approach demonstrates practical improvements in accuracy, calibration, and efficiency.
4. Applicability: MergeMix is shown to be effective across both image classification and multi-modal tasks.

### Weaknesses
1. Limited impact on multi-modal tasks: The gains for MLLMs are marginal, suggesting the method’s strengths are domain-specific.
2. Outdated baselines: Most compared methods in classification tasks are from two years ago, which may not represent the latest advances.
3. Scope of contribution: The paper could better clarify its impact boundaries.
4. Discussion of limitations: More explicit discussion of why the method is less effective for multi-modal tasks and why recent baselines were not included.

### Questions
1. Recency of baselines: Why were recent state-of-the-art methods not included in the classification comparisons? Can the authors provide results against more current baselines?
2. Domain-specific effectiveness: Why is MergeMix more effective for classification than for multi-modal tasks?
3. Future directions: What modifications might enhance MergeMix for multi-modal models?
4. Broader applicability: Are there other domains (audio, video) where MergeMix could be tested?
5. Limitations: Please discuss scenarios where MergeMix may not be suitable or where its benefits are minimal.

### Soundness
2

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
5

### Summary
This paper proposes a novel image-data augmentation method, **MergeMix**, which first produces clustered attention maps by Token-Merge and then mixes the important visual tokens rather than raw pixels, yielding semantically smoother synthetic images. 
The authors embed this augmentation into two tasks:
Image classification: MergeMix is used to train ViT/DeiT models 
  Results: +2.5 % accuracy vs. the best prior mixup, **15 % higher throughput**, **–0.68 G FLOPs**, and lowest calibration error (ECE) under severe occlusion.
MLLM alignment: clean images are treated as winners and MergeMix images as *losers*; a SimPO ranking loss is added to SFT, without any reward model**.
  Results: LLaVA-7B gains +0.83 % on nine VQA/understanding benchmarks; Qwen2.5-VL-7B gains +2.88 % on reasoning sets, while vision tokens can be reduced to 25 % without performance drop.
Overall, MergeMix provides a unified, reward-free training paradigm that simultaneously improves accuracy, efficiency and calibration for both pure-vision and multi-modal models.

### Strengths
* A novel image mixing augmentation method is proposed, which demonstrates significant improvements across multiple datasets.

* The mixed images are directly used as the "loser" in a pairwise ranking setup via SimPO, eliminating the cost and potential bias of training a separate Reward Model (RM) and simplifying the pipeline.

* Extensive experiments are conducted, providing multi-faceted validation of the method's effectiveness.

### Weaknesses
* The application of MergeMix to image classification and MLLM alignment tasks shows some innovation, but the degree of novelty is limited.

* The assumption that attention-based merged images are inherently of lower quality than original images lacks substantiating evidence. 
While the paper discusses the method from an MLLM perspective, validation is only conducted in the visual modality, leaving its efficacy in other modalities unexplored.

* The performance drop on MMBench and MathVista after MergeMix compared to SFT results suggests that the visual enhancement method is not universally effective.

* Although MergeMix employs token compression, it does not enhance model inference efficiency. Yet, the paper incorrectly claims this as a merit of the method.

### Questions
* What is the specific mechanistic link between the ViT-enhancing visual mixing method and the broader objective of preference alignment, as discussed in the paper?

* What evidence can substantiate that the attention-based mixed regions genuinely correspond to semantically "more important" objects within the images?

* How can it be guaranteed that training the vision encoder of VLM with MergeMix does not adversely impact the model's capabilities in other tasks or modalities?

### Soundness
2

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
The paper proposes MergeMix, a training strategy combining token merging and mixup to bridge supervised fine-tuning (SFT) and reinforcement learning (RL) for both image classification and multi-modal large language. The key idea is to merge semantically similar vision tokens before applying mixup, thereby reducing redundant visual tokens while preserving spatial semantics, and then form preference pairs (winner/loser) for preference optimization via the SimPO loss, enabling preference alignment without reward models.  The authors demonstrate improvements on CIFAR100, ImageNet-1K, and multiple MLLM benchmarks (LLaVA, Qwen2.5-VL), with notable efficiency and calibration benefits.

### Strengths
1.The introduction of a token merge–based mixing mechanism and attention recovery via bipartite soft matching is novel within the mixup literature. Compared to heuristic or random masking, the method provides a more structured way to preserve salient regions during interpolation.
2.The token-merge + mixup combination is reasonable, and the design (Top-K attention selection, λ re-scaling, ranking loss) can be integrated into existing ViT and MLLM frameworks with minimal modifications.The authors demonstrate that MergeMix can reduce FLOPs and slightly improve throughput, confirming that the design is at least practically implementable.

### Weaknesses
1.The paper’s organization hinders readability. The introduction directly dives into technical detail without motivating the gap, and the related work section is largely enumerative rather than analytical. Notation is inconsistent and transitions are abrupt, making it difficult to follow the method’s rationale.
2.The paper mixes several technical ideas—token merging, mixup, λ re-scaling, and ranking loss—without a clear unifying formulation.
It is unclear how the policy P(·,·) determines masks, how λ̂ interacts with attention, or how the ranking loss relates to reward modeling.The method section reads as a collection of components rather than a coherent algorithmic framework.
3.The paper’s central claim is that MergeMix bridges SFT and RL paradigms, but the experiments do not provide direct or conceptual evidence of this.There are no comparisons with preference optimization methods such as DPO, PPO, or SimPO; no analysis of reward-like behavior; and no ablation isolating the contribution of the ranking loss.Consequently, the connection to RL remains metaphorical rather than empirical.

### Questions
1.Since the paper claims to bridge SFT and RL through preference-style optimization, have you compared MergeMix with established preference optimization methods such as DPO, PPO, or SimPO? Such a comparison would be essential to substantiate the claimed “bridge” between SFT and RLHF paradigms.
2.The paper defines λ̂ as derived from a policy 𝑃, but the implementation details remain vague. Is λ̂ obtained from attention scores, gradients, or optimized independently? How sensitive are results to this design choice?
3.How exactly does the ranking loss applied to synthetic mixed pairs emulate human preference learning? Does the “preference” in this context reflect semantic quality, output diversity, or another measurable signal?
4.Could you report standard deviations across multiple runs and clarify whether all baselines share identical training setups (e.g., frozen encoders, same number of epochs)?

### Soundness
2

### Presentation
2

### Contribution
2
