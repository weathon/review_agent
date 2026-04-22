# VLOD-TTA: Test-Time Adaptation of Vision-Language Object Detectors

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Vision–language object detectors (VLODs) such as YOLO-World and Grounding DINO achieve impressive zero-shot recognition by aligning region proposals with text representations. However, their performance often degrades under domain shift. We introduce VLOD-TTA, a test-time adaptation (TTA) framework for VLODs that leverages dense proposal overlap and image-conditioned prompt scores. First, an IoU-weighted entropy objective is proposed that concentrates adaptation on spatially coherent proposal clusters and reduces confirmation bias from isolated boxes. Second, image-conditioned prompt selection is introduced, which ranks prompts by image-level compatibility and fuses the most informative prompts with the detector logits. Our benchmarking across diverse distribution shifts -- including stylized domains, driving scenes, low-light conditions, and common corruptions -- shows the effectiveness of our method on two state-of-the-art VLODs, YOLO-World and Grounding DINO, with consistent improvements over the zero-shot and TTA baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes VLOD-TTA, the first test-time adaptation (TTA) framework specifically designed for vision-language object detectors (VLODs). Its core goal is to address the significant performance degradation of VLODs when faced with new environments that differ from the training data distribution (e.g., "domain drift" due to changes in style, lighting, and weather).

### Strengths
1.The paper establishes a comprehensive benchmark, validating the method’s effectiveness across up to 96 different test scenarios. The results show that VLOD-TTA consistently improves model performance under various domain shifts, including artistic styles, real-world driving scenes, and image corruptions.
2.The paper concerns DA problem by tta, which is easy to follow

### Weaknesses
[1] The IWE mechanism relies on clusters of candidate boxes with dense overlaps. Therefore, when dealing with a large number of tiny, sparse, and low-overlap objects—such as in scenes from the Cityscapes dataset—its effectiveness may be reduced, as forming sufficiently “high-density” regions to guide optimization becomes challenging.
[2] The construction of this prompt pool (generated using GPT in the paper) is itself a labor-intensive step. Moreover, for highly specialized domains, the existing prompt pool may be insufficiently comprehensive, which could limit the effectiveness of IPS.

### Questions
[1] TTA introduces significant latency. How can we balance performance gains with computational cost? In the future, it may be worth exploring a **“selective adaptation”** strategy, where the model first evaluates the degree of domain shift in the current input and only triggers the TTA process when the shift exceeds a certain threshold, thereby maintaining fast inference in most cases.
[2] VLOD-TTA demonstrates effectiveness under various domain shifts. However, is there a “limit” to its adaptation capability?

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
3

### Summary
This paper proposes VLOD-TTA, a test-time adaptation framework for vision-language object detectors such as YOLO-World and Grounding DINO. The approach combines IoU-weighted entropy minimization to focus adaptation on spatially coherent proposal clusters and image-conditioned prompt selection to fuse only the most relevant prompts with detector outputs. Only lightweight adapters are optimized during test time. Experiments across several domain shifts indicate consistent improvements over zero-shot and baseline TTA strategies.

### Strengths
- **Relevant problem**.

    Test-time adaptation for open-vocabulary object detection remains underexplored and has strong relevance for real-world robustness.


- **Solid empirical results**.

    The study covers multiple datasets and two state-of-the-art VLODs, showing consistent gains.

### Weaknesses
- **Architecture-dependent adaptation**.

    Different parameters are adapted for YOLO-World and Grounding DINO, which reduces generality and complicates baseline comparisons.

- **Unclear adaptation protocol**.

    It is not specified whether the model is reset after each image or adapts continuously, which can lead to very different behavior and raises reproducibility concerns.

- **Unusual adaptation target**.

    The method updates lightweight adapters rather than normalization parameters, which differs from standard TTA practice. The rationale for this choice should be clarified.

- **Significance of the contributions**.

    The proposed image-conditioned prompt selection and IoU-weighted entropy minimization modules seem reasonable but somewhat limited in scope. While they contribute to performance it is not clear that these represent sufficiently substantial innovations to justify a top-tier venue.

### Questions
1.	Are the model parameters reset after every image, or does adaptation accumulate across the evaluation set? Please clarify the protocol and its practical justification.
2.	Could the IoU-weighted entropy component be used alone without depending on adapters, or could adapters be placed uniformly at shallow layers to avoid architecture-specific tuning?
3.	Did you evaluate or consider updating BN or LayerNorm statistics? How would a normalization-only adaptation compare in terms of accuracy and compute?

### Soundness
3

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
4

### Summary
This paper introduces an approach to perform Test-Time Adaptation (TTA) for the task of Object Detection with Vision Language Models (VLM). The method is based on two main contributions: a weighting of the visual proposals contribution based on their Intersection over Union (IoU) weighting, and a prompt selection conditioned over the image. Extensive evaluations were performed on multiple datasets and corruptions.

### Strengths
The paper tackles TTA for object detection using VLM, which is an original and interesting approach.  The mIoU weighting of the entropy is simple yet original and well in phase is already existing TTA approaches.  

The results show consistent gains in mAP on all datasets, which evaluate different kinds of simulated and natural corruption/domain shift.

### Weaknesses
TTA is generally conducted by optimizing batch norm parameters. Here, the method relies on adapters and learnable residual prompts. This parameter overhead might be enough to explain the gains of the proposed approach. Furthermore, if adaptation requires back-prop or large memory, it may not be viable for streaming/inference environments. This point should be discussed, and the number of learnable parameters and complexity should be shown for the method and the baselines. 

 The paper lacks an ablation to disentangle the contribution of the mIoU weighting scheme and the prompt selection. Furthermore, more description would be required to understand how the prompts were generated using a GPT model, as this step appears to have a great impact.

### Questions
Which GPT model was used to generate the prompts?

Which prompts were given to the GPT model to construct the textual prompts?


It is unclear how the method deals with the different samples in the batch and how this parameter should have an impact on the TTA task.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes VLOD-TTA, the first test-time adaptation (TTA) framework tailored specifically for vision-language object detectors (VLODs) such as YOLO-World and Grounding DINO, both of which have demonstrated strong zero-shot generalization. The main technical contributions are twofold: (1) IoU-weighted entropy minimization (IWE), which emphasizes adaptation over spatially coherent clusters of object proposals to address confirmation bias and localization uncertainty, and (2) image-conditioned prompt selection (IPS), which selects and fuses the most relevant textual prompts for each image to improve detection robustness. The approach is empirically validated on a comprehensive benchmark encompassing diverse domain shifts, showing consistent improvements over both zero-shot and existing TTA baselines.

### Strengths
1. The paper proposes a well-motivated VLOD-TTA framework, which is interesting and inspiring. 	

2. Clear motivation and problem setup for TTA in VLODs. Figures 1 and 2 (Pages 2) concretely illustrate failure modes of standard entropy and uniform prompt averaging, and how the proposed IWE and IPS address them.

3. Comprehensive experimental evaluation and Clear presentation and informative figures/tables.

### Weaknesses
1. Despite comprehensive benchmarks, the paper mostly highlights consistent positive gains. However, as briefly noted in the conclusion and Section 4.4, IWE can underperform in scenes with numerous small, scattered objects (e.g., Cityscapes); yet the depth of analysis is minimal. It would significantly strengthen the paper to have a more granular breakdown for such problematic cases, including visualizations and quantification of failure cases. 

2. As shown in Figure 5, the placement of adapters differs between YOLO-World (vision backbone+neck) and Grounding DINO (text encoder), with minimal theoretical motivation provided. This is largely left to empirical ablation, when some architectural analysis or interpretation might clarify practical trade-offs or inform future practitioners.

3. While overall clarity is reasonable, the related work section (Section 2) needs to be more explicit about methodological distinctions versus related TTA for VLM and OD approaches

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
