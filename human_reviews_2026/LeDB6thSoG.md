# UniGP: Taming Diffusion Transformer for Prior Preserved Unified Generation and Perception

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 6

## Abstract
Recent advances in diffusion models have shown impressive performance in controllable image generation and dense prediction tasks (e.g., depth and normal estimation). However, existing approaches typically treat diffusion-based controllable generation and dense prediction as separate tasks, overlooking the potential benefits of jointly modeling the different distributions. In this work, we introduce UNIGP, a framework built upon MMDiT, which unifies controllable generation and dense prediction through simple joint training, without the need for complex task-specific designs or losses, while preserving the backbone’s versatile priors. By learning controllable generation and prediction under different conditions, our model effectively captures the joint distribution of image-geometry pairs. UNIGP is capable of versatile controllable generation (as ControlNet), dense prediction (as Marigold) and joint generation (as JointNet). Specifically, the proposed UNIGP consists of DUGP and a unified dataset training strategy. The former, following the principle of Occam’s razor, uses only a copied image branch of MMDiT to model dense distributions beyond RGB, while the latter integrates different types of datasets into a unified training framework to jointly model generation and perception tasks. Extensive experiments demonstrate that our unified model surpasses prior unified approaches and comparable with specialized methods. Furthermore, we show that through multi-task joint training, the performance of controllable generation and dense prediction can mutually enhance each other.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents UNIGP (Unified Generation and Perception), a Diffusion Transformer-based framework designed to simultaneously model both RGB image distribution and dense perception distributions, such as depth and surface normals, thereby unifying generation and perception tasks. The core of the work is the DUGP (Disentangled Unified Generation and Perception) branch, which follows the principle of Occam's razor by selectively copying and reusing only the image branch of the pre-trained MMDiT backbone to model additional visual distributions with fewer parameters, while preserving the backbone's versatile generative priors. By employing a unified dataset and training strategy, UNIGP jointly optimizes both generation and perception tasks.

### Strengths
1. The method demonstrates competitive quantitative performance across multiple unified, generation, and perception tasks, achieving results that are on par with or surpass many specialized generative baselines (Tab. 1, 2, 3). 
2. The paper is well-written, clearly organized, and the proposed framework (DUGP) is logically presented, aiding in the comprehension of the overall architecture.

### Weaknesses
I acknowledge the strong quantitative results, though they may be primarily attributed to the powerful priors of the SD3 backbone. The most significant weakness is the overarching lack of novelty. 
1. Limited Architectural Contribution: Compared to JointNet-stype, this paper merely copies the weights of the image branch. This does not constitute a significant architectural contribution.

2. Training Strategy: The proposed "unified training strategy" is common in multi-task learning.

3. Task Novelty: While the unified generation and perception task is interesting, similar joint-modeling capabilities have already been demonstrated by existing method JoDi.

If the authors can address my concern regrading novelty, I am willing to raise my score.

### Questions
1. Why do methods that establish a joint distribution of multiple tasks, such as UniCon, OneDiffusion, and JoDi, ignore the potential connection between perception and generation tasks?
2. In the MMDiT block, the image and text branches have a shared Self-Attention, and two separate Gate and FFN layers. In the DUGP Branch, is the Attention Projector equivalent to the Self-Attention in the MMDiT block? What percentage of the original MMDiT's parameters does the DUGP Branch comprise?

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
This paper proposes a unified diffusion transformer for dense prediction and controllable generation. Building upon the Multi-Modal Diffusion Transformer (MMDiT), the authors introduce DUGP, a model designed to accept additional modalities such as depth, normal, and edge. The proposed DUGP consists of three main components: conditional patchers, control layers, and perception layers. The control and perception layers are respectively designed to control the Stable Diffusion 3.5 backbone and to predict depth and normal maps. The training objective is defined as a weighted sum of image generation, depth estimation, and normal estimation’s denoising losses, where the weighting dynamically varies depending on whether the input data is intended for generation or perception. The proposed method is evaluated on COCO-5k and several depth and normal benchmarks, showing performance comparable to existing competitors and achieving superior results among joint generation approaches.

### Strengths
1. DUGP is initialized by copying only the image branch of MMDiT. This design choice is practical, as it enables the model to handle additional modalities (depth, normal, and other conditional signals) with only half the number of parameters compared to the original model. The decision to copy only the image branch is also reasonable.
2. Among the joint generation methods evaluated in this paper, DUGP achieves superior performance in depth and normal estimation. The authors also justify their design choices through an ablation study analyzing the ratio between control and perception layers, validating the effective ratio.

### Weaknesses
1. Missing Experimental Results for Tasks Presented in the Paper
- Among the various tasks introduced in Figure 1, the quantitative results for sketch-to-image and pose-to-image are not provided in either the main paper or the supplementary material. Only a few qualitative examples for these tasks could be found. In addition, there are no experiments or discussions related to image restoration (deblurring) anywhere in the paper.

2. Lack of Clear Explanation on Why Previous Works Lack Connection Between Perception and Generation Tasks
- In Lines 64–69, the authors mention that previous works lack potential connections between perception and generation tasks, but the paper does not clearly explain why this is the case.
For instance, UniCon enabled the integration of perception and generation without separate training through separable noise-level training for each modality. Furthermore, Table 3 of UniCon has shown the mutual synergy of perception and generation tasks. The ICCV 2025 work JointDiT [a1] also demonstrated strong performance on perception tasks using the separable noise-level training, and Orchid [a2] achieved high performance by defining a joint latent space. It would be better if the authors could clarify the specific reasons why previous works are considered to have insufficient connections between perception and generation tasks.

[a1] Byung-Ki et al., “JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers”, ICCV 2025.

[a2] Krishnan et al., “Orchid: Image Latent Diffusion for Joint Appearance and Geometry Generation”, ICCV 2025.

3. Marginal Improvement in Mutual Synergy Between Perception and Generation Training
- In Table 4, the proposed method shows only marginal improvement compared to the w/o perception training and w/o generation training variants, which makes it difficult to strongly support the claimed mutual synergy between perception and generation training (one of the main contribution). It would be beneficial to include experiments on other datasets (e.g., ImageNet and etc.) to further validate this claim.

4. The comparison with Geowizard’s depth and normal outputs provides the indirect evaluation of joint generation performance. Therefore, it is difficult to assess whether the mutual connection between image and geometry generation is effectively established. It would strengthen the paper if the authors could demonstrate 3D lifting results from the jointly generated image and depth pairs to provide more direct evidence of mutual connection.

5.  From Equations (1) and (4), the proposed architecture appears to significantly increase the FLOPs required for attention operations compared to the original backbone. It would be helpful to report the additional computational cost, such as FLOPs and GPU memory usage.

### Questions
1. In Table 4, when training the Marigold-style and JointNet-style baselines, was the text branch frozen during training?
2. I found that the normal estimation performance of some competitors (e.g., Marigold, Geowizard, etc.) differs from the values reported in their original papers. Could the authors clarify how these numbers were obtained, for example, whether they were reproduced?

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
4

### Summary
This paper proposes a new framework, UNIGP, which aims to unify controllable image generation (e.g., ControlNet) and dense perception (e.g., depth and normal estimation) within a single model. Based on the MMDiT (SD3) architecture, its core contributions include: 1) Proposing the DUGP architecture, which follows Occam's razor by copying only the image branch of MMDiT (rather than the entire backbone) to efficiently model visual distributions beyond RGB; 2) Proposing a unified dataset and training strategy that mixes generation and perception datasets and performs joint training via specific loss weights. Experiments show that the model can handle multiple tasks within a single framework—including text-to-image, depth/normal estimation, and multi-condition generation—and reveals that joint training of generation and perception tasks can mutually enhance performance.

### Strengths
- Novel and Efficient Architecture: The core idea of this paper (DUGP) is excellent. Compared to copying the entire backbone (JointNet-style) or fine-tuning the backbone and causing catastrophic forgetting (Marigold-style), UNIGP only copies the MMDiT image branch to handle new visual modalities. This design is more parameter-efficient, conceptually simple, and well-justified.
- "Mutual Benefit" Synergy Discovery: A key contribution of the paper is the convincing demonstration that joint training of generation and perception tasks can mutually enhance each other. The ablation study strongly supports this: perception training helps the generation task adhere more accurately to geometric conditions (lower RMSE), while generation training helps the perception task capture finer details.
- Strong Performance: As a unified model, UNIGP demonstrates performance comparable to or even better than SOTA (state-of-the-art) specialized models on several tasks. This is particularly true for surface normal estimation, where it outperforms existing generative and discriminative methods on all metrics. In controllable generation, it also excels in control fidelity (RMSE).
- Data Efficiency in Perception: Although it lags behind the SOTA discriminative model (DepthAnything) in depth estimation, UNIGP uses only 59K perception samples, far fewer than the 62.6M used by DepthAnything (a difference of over 1000x). This highlights the significant advantage and data efficiency of the UNIGP framework in leveraging pre-trained generative priors.

### Weaknesses
- Performance Gap in Depth Estimation: The paper claims to be "on par with specialized methods", but this is not entirely accurate for the depth estimation task. As shown in the experimental results, there is a significant gap between UNIGP and the SOTA discriminative model, DepthAnything. While the authors correctly identify the data volume (59K vs 62.6M) as the primary reason, this does weaken the "on par" claim. UNIGP is the best generative unified model, but not the best specialized depth estimation model.
- Lack of Support for Image Restoration Task: Figure 1d lists "Image Restoration (Deblurring)" as one of the model's capabilities. However, this task is completely ignored in the main experimental evaluation—it appears neither in the qualitative comparisons nor in the quantitative comparisons. Although the training data section mentions a "blur" annotation, the lack of evaluation leaves this functional claim unproven.
- Unclear Motivation for C/P Split in DUGP: The DUGP branch is divided into "Stacked Control Layers (C)" and "Stacked Perception Layers (P)", and the ablation study shows that a balanced C=12, P=12 setting is optimal. However, the paper does not adequately explain why this split is necessary. What is the functional design motivation behind the control and perception layers? Why can't all 24 DUGP blocks use the same design (e.g., all using or all bypassing certain connections)?
- Ambiguity in Unified Training Dataset: How is the 1M sample set for generation tasks constructed? Does each of the 1M samples possess all 6 annotations (depth, normal, canny, sketch, etc.) simultaneously, or is 1M the total count, with different subsets corresponding to different annotations? This is very important for understanding the actual amount of data the model was trained on for multi-condition control.

### Questions
1. Regarding the Depth Estimation Gap: Do the authors believe the performance gap between UNIGP and DepthAnything in depth estimation is entirely attributable to the 59K vs 62.6M data difference? Or is it possible that the generative prior of this unified framework (while helpful for generalization) somehow limits the model from achieving the SOTA metrics on specific benchmarks that a purely discriminative model can reach?
2. Regarding the Image Restoration Function: Could you clarify the "Image Restoration (Deblurring)" function shown in Figure 1d? How was the model trained for this task (was the "blur" annotation from MultiGen-20M used)? Why were the results for this task omitted from the experimental evaluation in Section 4?
3. Regarding the C/P Split in DUGP: Could you elaborate on the design motivation for splitting DUGP into Stacked Control Layers (C) and Stacked Perception Layers (P)? Why does C=P=12 perform best? What is the theoretical or empirical basis for this division? What are the intended functional differences between the perception layers (which bypass certain connections) and the control layers (which use them)?
4. 4S. Regarding the Details of Mutual Enhancement: The "mutual enhancement" is a very interesting finding. Looking at the ablation study, depth estimation is already quite good without generation training, and the improvement after adding it is limited. However, the improvement for normal estimation is very significant. Why does generation training help normal estimation so much more than it helps depth estimation?

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
This paper introduces UNIGP, a novel framework for unifying generation and perception by jointly modeling multiple distributions within a single Diffusion Transformer (DiT) model. The core of the proposed method is built upon the Multi-Modal DiT (MMDiT) framework, which is extended with a parameter-efficient "Disentangled Unified Generation and Perception" (DUGP) branch, which only copies the image-processing components of the frozen, pre-trained backbone, thereby preserving its powerful generative priors. The authors propose a unified training strategy that combines generation and perception datasets and uses an adaptive loss to train the model end-to-end.

The experiments are extensive and rigorous, demonstrating that not only UNIGP surpasses prior unified models but also performs on par with or even exceeds specialized, task-specific models in some cases. The ablation studies provide strong evidence for the key claims, particularly the concept of cross-task synergy.

### Strengths
1. Elegant and Well-Motivated Method: The core architectural idea of UNIGP is both simple and highly effective. Instead of duplicating the entire backbone (as in "JointNet-style") or fine-tuning it directly and risking catastrophic forgetting (as in "Marigold-style"), the authors copy only the necessary image branch. The design is parameter-efficient and leverages the strengths of pre-trained generative models effectively.
2. Strong and Comprehensive Experimental Validation: The model is tested across a diverse set of tasks, and is compared against a comprehensive set of state-of-the-art models, including specialized generative models, specialized perception models, and other unified approaches. Quantitatively, UNIGP shows outstanding performance.
3. Compelling Evidence for Mutual Enhancement: The most important contribution of this paper is that it empirically demonstrates a mutually beneficial relationship between generative and perceptual tasks. The ablation studies in Section 4.3 and Table 4 provide clear, quantitative evidence that removing the training process or reducing the data for one task negatively impacts the performance.

### Weaknesses
1. Lack of Motivation for Layer Design: In Section 3.2.1, the paper describes "Stacked Control Layers" and "Stacked Perception Layers." A key difference is that the feature addition from Equation (3) is bypassed in the perception layers. The paper states the goal is to "extract features," but the high-level intuition for why this bypass is beneficial for perception is not fully elaborated. A more explicit explanation of this design choice would be helpful for the reader's understanding.
2. Significant Inference and Multi-Condition Overhead: The supplementary material A.2 reveals that the model's parameter count increases from 2B to 3B, and inference time nearly doubles from 5.07s to 9.54s (on a V100 with 20 steps). This presents a challenge for increasing the number of parameters and for any real-time application. Furthermore, Appendix A.1.3 states that handling multiple conditions is achieved by "duplicating DUGP for each condition," which implies that parameter and computational costs could grow linearly with the number of simultaneous control signals. This scaling strategy is not efficient and could be a major limitation.
3. Performance Gap with State-of-the-Art Discriminative Models: While UNIGP is the top-performing generative method for dense prediction, it still lags behind the leading discriminative model (such as DepthAnything V2 and other State-of-the-Art methods) in depth estimation. The authors attribute this to the vast difference in training data scale (59K for UNIGP's perception tasks vs. 62.6M for DepthAnything). However, the paper does not discuss potential strategies to close this gap, such as leveraging knowledge distillation from a pre-trained specialist model or using it to generate pseudo-labels to enrich the perception dataset.

### Questions
1. Could you provide more intuition on the design choice to bypass the feature addition in the "Stacked Perception Layers"? Does this prevent the perception task's gradients from overly influencing the primary generative pathway of the backbone, thereby helping to preserve the priors?
2. Given the performance gap with DepthAnything, which is largely due to data scale, have you considered methods like knowledge distillation from the specialized model or using it to generate pseudo-labels to augment your perception dataset and potentially close this gap?
3. The increase in parameter count and inference time is significant. Have you considered designing a more lightweight variant of DUGP, for example, by sharing some Transformer modules between the perception and control pathways, instead of using completely separate layers?

### Soundness
3

### Presentation
3

### Contribution
3
