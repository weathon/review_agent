# EraseLoRA: MLLM-Driven Foreground Exclusion and Background Subtype Aggregation for Dataset-Free Object Removal

- Avg Score: 4.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4

## Abstract
Object removal requires more than erasing a target—it must reconstruct the missing region with high structural fidelity while preserving diverse background context. Existing diffusion-based dataset-free approaches attempt to redirect self-attention away from the masked target but fail in two critical ways: (1) non-target foregrounds are often misinterpreted as background, causing unintended object regeneration, and (2) disruption of short-range activations degrades fine details and prevents coherent integration of multiple background cues. We introduce EraseLoRA, a dataset-free object-removal framework that leverages the visual reasoning power of multimodal large-language models (MLLMs) to exclude foreground distractions and assemble rich background content. The first stage, BRF (Background Reconstruction with Foreground Exclusion), isolates and removes non-target objects through MLLM-guided reasoning on a single image–mask pair, producing clean background candidates without ground-truth supervision. The second stage, Background Subtype Aggregation (BSA), restores the masked region by treating each inferred background subtype as a puzzle piece, enforcing their consistent integration to preserve both local detail and global context. EraseLoRA achieves state-of-the-art object-removal performance across diverse diffusion backbones without any additional training data or ground-truth background, demonstrating that MLLM reasoning—applied here for structural reconstruction rather than object generation—can directly guide diffusion models to rebuild complex scenes from a single image with unprecedented structural and contextual coherence.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
he paper proposes EraseLoRA, a dataset-free object-removal framework that plugs into modern text-to-image diffusion models and uses an MLLM to (i) explicitly exclude non-target foregrounds and (ii) aggregate multiple background “subtypes” to fill the hole more coherently.

### Strengths
1. The motivation of this submission is clear and the design aligns well with the motavation.
2. The paper is well-written and easy to follow.
3. The qualitative results demonstrated in this submission are impressive.

### Weaknesses
1. Evaluation metrics. On OpenImages there is no ground-truth removed image; so they use F-DINO/F-CLIP and B-DINO/B-CLIP. That’s reasonable, but the community may want human or more visual realism metrics. Currently the strongest claims rest on these proxy numbers.
2. Novelty is partly compositional. The proposed method is kind of good engineering on top of existing tools rather than a brand-new learning principle which might not be suitable for ICLR.
3. Robustness experiments on errors from tools such as MLLM mistakes could strengthen this submission.

### Questions
My concerns are listed in the weaknesses part.

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
The paper proposes EraseLoRA, a dataset-free object removal framework that addresses the issues of unintended non-target foreground regeneration and fine-detail loss caused by disrupted short-range attention in existing dataset-free object removal frameworks.

This framework consists of two stages during Test Time Adaption (TTA):
1. The first stage is Background Reconstruction with Foreground Exclusion (BRF), which distinguishes between foreground and background using an MLLM (Multimodal Large-Language Model) and preserves the background via a background reconstruction loss.
2. The second stage is Background Subtype Aggregation (BSA), which guides the generation of the target region by controlling the cross-attention map of the background through an alignment loss, while avoiding the generation of only a single type of background via a diversity loss.

### Strengths
1. The proposed EraseLoRA specifically addresses the issues of unintended non-target foreground regeneration and fine-detail loss caused by disrupted short-range attention in dataset-free object removal through BRF (Background Reconstruction with Foreground Exclusion) and BSA (Background Subtype Aggregation).
2. Compared with directly constraining the attention map, the Test Time Adaption method is more robust.

### Weaknesses
1. **Methodological Limitation on Occluded Non-Target Foregrounds.** While suppressing non-target foregrounds effectively avoids their interference during target-region background generation, it also creates a limitation: when non-target foregrounds are occluded by the target foreground, the method cannot reconstruct those occluded parts. This flaw is clearly visible in the provided result figures.
2. **High Computational Overhead from MLLM Dependency.** Invoking an MLLM for each inference significantly increases both computational load and latency, which may limit practical use in resource-constrained scenarios.
3. **Dataset Ambiguity.** Please provide a formal citation for the used dataset. Additionally, clarify if the "ROAD dataset" refers to the existing "RORD dataset".
4. **Inadequate Result Presentation.** The presentation of comparisons with previous SOTA methods is insufficient. Additionally, Figure 7 in the Appendix also lacks the display of input images and masks.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a dataset free object removal framework using multimodal LLMs to differentiate between foreground from background. It also incorporates a test-time adaptation method for guiding the diffusion. It broadly consists of two stages, namely background reconstruction with foreground exclusion and background subtype aggregation. Experiments on two datasets show improvement over the existing methods.

### Strengths
1. Use of MLLMs for foreground-background separation in object removal is novel. 

2. Motivation seems convincing. 

3. The method is agnostic to the base segmentation models. 

4. Strong evaluation protocol.

### Weaknesses
1. Gains achieved on the SOTA are only marginal (Tab 2) and at times, loses to dataset-driven methods. 

2. Given these marginal gains, statistical significance isn't given. 

3. Very large computational cost - Tab 3 shows a significant increase in VRAM. 

4. The method seems to be critically dependent on MLLMs. There is no clarity on what happens if and when MLLMs hallucinate objects?

5. Datasets (ROAD) are large enough, and metrics aren't representative enough. 

6. There are a lot of missing details in the methodological description. 

7. There are no formal theoretical arguments on why losses should impose "puzzle-like" reconstruction.

### Questions
1. Report wall-time clock time per image and VRAM for all methods.

2. How does TTA time scale with image resolution? 

3. What percentage of test samples have incorrect foreground detection? 

4. B-DINO/B-CLIP uses your own MLLM-defined background, which is circular. Provide metrics that don't depend on your methods' intermediate outputs. 

5. Statistical significance analysis needs to be done. 

6. There is no ablation on the choice of MLLM, Authors should consider doing it. 

7. Compare against a simple baseline like masking MLLM detected baselines. 

8. Give a detailed failure analysis by showing the cases where MLLM misclassifies, loss doesn't converge, and when there is conflict between background subtypes. 

9. Provide details on hyperparamter sensitivity, and number of TTA iterations needed and so on. 

10. Given the 2.4× VRAM cost, why not use SmartEraser with lesser params?

11. How does your method perform on cases where multiple objects needs to be removed, where there are unclear boundaries?

### Soundness
2

### Presentation
2

### Contribution
2
