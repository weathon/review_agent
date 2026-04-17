# S4NN: Scalable Self-Supervised Spiking Neural Networks

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Spiking Neural Networks (SNNs) offer a promising alternative to traditional artificial neural networks by leveraging sparse, event-driven computation that closely mimics biological neurons. When deployed on neuromorphic hardware, SNNs enable substantial energy savings due to their temporal and asynchronous processing. However, training SNNs remains a major challenge because of the non-differentiable nature of spike generation. In this work, we introduce the first fully self-supervised learning (SSL) framework for SNNs that scales to large-scale visual tasks without requiring labeled fine-tuning. Our method leverages intrinsic spike-time dynamics by aligning representations across time steps and augmented views. To address gradient mismatch during surrogate training, we propose the MixedLIF neuron, which combines a spiking path with an antiderivative-based surrogate path during training to stabilize optimization, while retaining a fully spiking and energy-efficient architecture at inference. We also introduce two temporal objectives, Cross Temporal Loss and Boundary Temporal Loss, that align multi-time-step outputs to improve learning efficiency. Our approach achieves strong results across ResNet and Vision Transformer-based SNNs on CIFAR-10, CIFAR10-DVS, and ImageNet-1K. Our approach further generalizes through transfer learning from ImageNet-1K to downstream tasks, including image classification, as well as COCO object detection and instance segmentation. Notably, our self-supervised SNNs match or exceed the performance of some non-spiking SSL models, demonstrating both representational strength and energy efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Deep SNNs trained with surrogate gradients struggle to reach optimal performance, and the coarse gradient signals also hinder self-supervised learning. This paper proposes a self-supervised method named S4NN to address these issues. Specifically, it introduces the MixedLIF neuron that possesses two forward paths: the standard LIF path and a second path derived by integrating the surrogate gradient function. The second path supplies dense gradients and is used only during training. MixedLIF is embedded in a contrastive-learning framework where losses are computed from the outputs of both paths. The approach is evaluated on classification, object detection, semi-supervised learning, and transfer-learning tasks; feasibility of deployment on neuromorphic chips is further verified with the LAVA-DL framework.

### Strengths
- The two paths of MixedLIF naturally correspond to the two outputs required by contrastive learning—a clever design. In effect, a full-precision SNN guides the training of a standard binary SNN.
- Validation on LAVA-DL demonstrates the method’s practicality for neuromorphic hardware deployment.

### Weaknesses
- The term “Self-Supervised” in the title is overly broad; the paper actually focuses on contrastive learning.
- Contrastive methods are expected to outperform direct supervised training or distillation, yet the ImageNet results fall short of recent SNNs published at top AI venues. The claimed superiority remains unconfirmed.

### Questions
The authors release their source code, but does it include the LAVA-DL deployment part?

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
4

### Summary
This paper proposes a scalable self-supervised learning (SSL) framework for SNNs, enabling ImageNet-level pretraining and transfer to downstream tasks such as COCO object detection and semantic segmentation. The core contributions include:

1. MixedLIF dual-path architecture: combining spiking (A-path) and continuous surrogate (B-path) activations during training to stabilize optimization.
2. Two temporal alignment objectives:

   * Cross Temporal Loss: full-time step consistency alignment
   * Boundary Temporal Loss: low-cost alignment on early and final time steps

### Strengths
* Hardware deployment evidence: Loihi execution results show high deployability and minimal accuracy drop.
* Promising transferability: Works on dense prediction tasks beyond classification.

### Weaknesses
1. Training-time energy consumption not measured.

   * Although inference energy efficiency is well-discussed, training energy is never quantified.
   * The claim that SSL enables energy-efficient SNNs **across the full lifecycle** remains speculative.

2. No fair comparison to supervised-pretrained SNNs. Without showing that SSL achieves comparable representation quality to labeled pretraining, the benefit of SSL itself is uncertain.

3. Lack of theoretical justification for Boundary Temporal Loss. The boundary sampling strategy seems heuristic. It remains unclear why only first/last timesteps sufficiently capture temporal structure for all SNN architectures.

4. Temporal augmentations are weak/inconsistent. It’s unclear whether performance gains actually come from *temporal* modeling rather than spatial SSL.

5. Projection head during inference contradicts “full spiking” claims. The MLP head is non-spiking, introducing hidden compute and power overhead.

### Questions
1. Why is the hardware validation compared against ANN models? Does this comparison sufficiently demonstrate the energy efficiency advantages claimed in the paper? Or, how does the current hardware evaluation reflect the contribution of the proposed SSL framework to overall energy savings?

2. Where exactly is the training-time energy saved? Please provide **measured** training energy vs supervised SNN and ANN-SSL baselines.

3. Why do first and final time steps suffice?

4. Where do the claimed representation gains come from? Spatial contrast? Temporal features? Dual-path gradient diversity?

## Additional Minor Suggestions

- Provide convergence curves to support “more stable optimization.”

- Include more meaningful temporal augmentations for static datasets.

- Improve notation consistency in sec. 3.1–3.3.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors, focusing on self-supervised learning in Spiking Neural Networks (SNNs), proposed the MixedLIF neuron, which outputs both spike signals and analog values within a fixed range, thereby stabilizing the gradients in traditional SNNs trained with surrogate functions.
They further introduced Cross Temporal Loss and Boundary Temporal Loss to exploit the temporal dynamics of SNNs for self-supervised representation learning.
Finally, the proposed method was validated on classification, detection, and segmentation tasks, demonstrating its effectiveness and general applicability.

### Strengths
1. The authors designed a general framework for self-supervised learning in SNN and conducted experiments across multiple tasks.

2. The descriptions of experimental settings and procedures are clear and detailed, ensuring strong reproducibility of the experiments.

### Weaknesses
1. The motivation of the paper is not very clear. The authors repeatedly mention in the abstract and throughout the paper that the non-differentiability of spikes poses a major challenge for training SNNs. I agree that spike signals are inherently non-differentiable; however, numerous studies have already addressed this issue through surrogate gradient methods, achieving excellent performance on datasets such as ImageNet, comparable to that of ANNs. Moreover, training efficiency has been greatly improved, and the gap between SNNs and differentiable ANNs has become increasingly small. Therefore, I find it difficult to agree that non-differentiability remains a major problem that SNNs need to overcome at present.

2. In comparing self-supervised learning with traditional surrogate-gradient-based supervised methods, the authors also fail to provide sufficient analysis of why it is necessary to study self-supervised learning in SNNs, especially given that the performance of self-supervised learning across various tasks remains significantly lower than that of conventional supervised approaches.

3. The concepts are not clearly defined. In the abstract and the methods section, the authors state that training SNNs is challenging due to the non-differentiability of spike signals, whereas in Section 2.2, they claim that the non-differentiability of spikes is the fundamental reason hindering the development of SSL in SNNs. I believe these two concepts are not equivalent, yet the authors seem to conflate them in the motivation, particularly in the statement “As discussed above, ...” within the Methods section.

4. In Section 2.1, the authors devote nearly half a page to introducing the background of SNNs. However, this section reads more like a general overview or popular science introduction, and its relevance to the specific research problem addressed in this paper appears limited.

5. The related work section is insufficient, and the literature review is not comprehensive. The most recent references cited for surrogate gradient optimization in SNNs only go up to 2018, while for self-supervised learning, the latest cited work is from 2021, and it pertains to the ANN domain. Moreover, no SNN-specific self-supervised learning studies are included in the experimental comparisons, which raises concerns about the validity and fairness of the proposed method’s evaluation.

### Questions
1. What exactly does the term “scalable” in the title refer to? The paper does not explicitly define or clarify this point.

2. What do the spatial augmentation and temporal augmentation mentioned in Section 3.1 specifically refer to?

3. In Section 3.1, the authors claim that soft reset causes distributional shift, which ultimately affects SSL performance. Is there any evidence or justification provided for this claim?

4. In Section 3.1, the authors state that they conducted experiments based on the Spikeformer framework and mention that GELU was replaced with MixedLIF, LayerNorm was replaced with BatchNorm, and softmax attention was removed. However, Spikeformer itself does not use GELU, LayerNorm, or softmax operations. This raises the question of whether the authors have carefully read the original Spikeformer paper.

5. In Table 7, the authors compare energy consumption and state that the energy cost of a MAC and an AC operation is 3.1 pJ and 0.1 pJ, respectively, claiming that this results in a 31× energy advantage for SNNs. However, in the original reference, these energy values were obtained based on 32-bit integer operations. Did the authors maintain integer precision in their network architecture?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Scalable Self-Supervised Spiking Neural Networks, the first fully self-supervised learning (SSL) framework designed to train SNNs for large-scale visual tasks without requiring labeled fine-tuning. The authors tackle the challenge of training SNNs, which stems from the non-differentiable nature of spike generation, by proposing two main innovations. First is the MixedLIF neuron, a dual-path module that uses a standard spiking path for energy-efficient inference and a parallel, antiderivative-based surrogate path during training to stabilize optimization and reduce gradient mismatch. Second, to leverage the natural temporal dynamics of SNNs, they introduce two novel training objectives: the Cross Temporal Loss, which aligns representations across all time steps, and the more computationally efficient Boundary Temporal Loss, which aligns only the initial and final time steps. The framework demonstrates strong performance across both ResNet and Vision Transformer-based SNNs on datasets like CIFAR-10, CIFAR10-DVS, and ImageNet-1K, achieving results that match or exceed some non-spiking SSL models while preserving the inherent energy efficiency of SNNs at inference.

### Strengths
This paper presents an exceptional study with significant strengths in originality, quality, clarity, and significance. Its originality is outstanding, introducing the first fully self-supervised learning (SSL) framework for SNNs that scales to large-scale tasks like ImageNet without labeled fine-tuning. The proposed MixedLIF neuron, a novel dual-path design to stabilize training, and the SNN-specific temporal loss objectives are highly creative contributions that exploit intrinsic spike dynamics. The research quality is excellent, supported by comprehensive experiments across multiple architectures (CNN and ViT) and datasets (static and neuromorphic) where the method matches or surpasses strong non-spiking baselines. Rigorous ablation studies validate each component's effectiveness, and successful deployment on Loihi hardware underscores its practical viability . The paper is written with remarkable clarity, logically presenting the problem of gradient mismatch and its solution, aided by effective diagrams and algorithms . Finally, its significance is substantial; by removing the reliance on labeled data, it breaks down a key barrier for SNN adoption and shows they can achieve high-level representational power, paving the way for scalable, energy-efficient, brain-inspired AI.

### Weaknesses
While inference efficiency is a clear strength, the proposed method requires two forward passes and gradient aggregation, increasing training complexity. A direct comparison of total training time and memory against the primary ANN baseline would offer a more complete picture of the trade-offs involved.

### Questions
See the Weaknesses

### Soundness
2

### Presentation
2

### Contribution
3
