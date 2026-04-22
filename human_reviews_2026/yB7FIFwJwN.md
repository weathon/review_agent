# I-DRUID: Layout to image generation via instance-disentangled representation and unpaired data

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Layout-to-Image (L2I) generation, aiming at coherently generating multiple instances conditioned on the given layouts and instance captions, has raised substantial attention in the recent research. The primary challenges of L2I stem from 1) attribute leakage due to the entangled instance features within attention and 2) limited generalization to novel scenes caused by insufficient image-text paired data. To address these issues, we propose I-DRUID, a novel framework that leverages instance-disentanglement representations (IDR) and unpaired data (UID) to improve L2I generation. IDR are extracted with our instance disentanglement modules, which utilizes information among instances to obtain semantic-related features while suppressing spurious parts. To facilitate disentangling, we require semantic-related features to trigger more accurate attention maps than spurious ones, formulating the instance-disentangled constraint to avoid attribute leakage. Moreover, to improve L2I generalization, we adapt L2I with unpaired, prompt-only data (UID) to novel scenes via reinforcement learning. Specifically, we enforce L2I model to learn from unpaired, prompt-only data by encouraging / rejecting the rational / implausible generation trajectories based on AI feedback, avoiding the need for paired data collection. Finally, our empirical observations show that IDM and RL cooperate synergistically to further enhance L2I accuracies. Extensive experiments  demonstrate the efficacy of our method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper targets for solving two issues in L2I generation, including instance overlapping and poor generation to new scenes. The method it proposed has two components, 1/ the instance-disentanglement module to disentangle semantic-related features and spurious features; 2/ RL module to generate unpaired data for model training. The evaluation on COCO_MIG and LayoutSAM-eval demonstrates that the proposed method achieves the SOTA performance compared with prior work.

### Strengths
- The way of incorporating RL on unpaired data into L2I model training is impressive, which indeed solves the generation limitation of current L2I approaches. The conversion to SDE solves an emerging pain point of exploring flow-matching models.

- The ablation and sensitivity analyses make the paper solid, especially the illustration of module effectiveness.

- The visualization is also good to follow with both good cases and failure cases provided.

### Weaknesses
- IDC is heuristic and does not stand out compared to previous attention ideas. The novelty in CAS-based Softplus inequality is incremental to existing attention regularization in L2I, such as region-aware attention, for example, Rethinking The Training And Evaluation of Rich-Context Layout-to-Image Generation (NeuIPS 2024), which should also be compared in the result tables.

- The reward design (GDINO IoU plus confidence) mainly measures the spatial grounding, ignoring the attribute binding such as color and texture, and text fidelity. This should be improved in the future.

- Comparison fairness: the results mix SD-1.5 and SD-3 while some baselines are UNET-based only. It would be better to use stronger base model such as SD-3 for the baselines if possible.

- The contents are too long. The introduction of experimental settings should be included in the original paper rather than appendix.

### Questions
- Why SD-1.5 has better performance than SD-3 on COCO-MIG and is it because IDM helps one backbone more than the other?

- Is there any overlap or duplicate risk for testing on COCO-MIG since RL uses COCO-2014-MIG prompts for generating unpaired data?

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
This work proposes to use the feature disentanglement and reinforcement learning to improve the performance of layout-to-image generation task. The author designs an instance disentanglement module to separate the image feature and encourages the model to concentrate semantic-related information into R+ while “spurious parts” into R-. Afterwards, this work uses the reinforcement learning with PPO to train the model. The final model is optimized jointly with the diffusion loss, disentanglement loss and RL loss.

### Strengths
* The motivation to separate the semantic-related features and spurious features is interesting.

### Weaknesses
* From the abstract, it seems that the purpose of feature disentanglement is for avoiding attribute leakage, is there any evidence that support that attribute leakage has been mitigated after using the feature disentanglement?
* The reviewer would like the author to clarify  the novelty in the RL part. From the current manuscript, it seems that the method just adopts the PPO and uses a new reward for layout-to-image task. Also I would like the author to quantitatively justify their choice of reward, e.g., why use IoU + confidence, what if change the importance ratio between them. 
* The reviewer has several questions for instance disentanglement part.
    * ln 196, what does “global-prompt-enhanced features” mean?
    * if the reviewer understands correctly, the purpose of the disentanglement is to concentrate the semantic-related features in cross-attention region, i.e., increasing the response in R+ while decreasing the response in R-. While from the illustration in Fig 2, it seems that in both building and background regions, the R- has higher response. The reviewer wonders if this observation suggests that the disentanglement is not work as expected.
    * the reviewer wonders the design rational of Eq 3. the (1-M) part is understandable as noted in ln 218 (evaluates the degree of attention value beyond instance i’s scope.) but why use R+ - AVG(R+)? What is the purpose of computing the difference between each instance response and average response? And what is the point to have the average response?
    * ln 225-227 mentioned that “Minimizing the above disentanglement loss Ldis enables IDM to accurately separate each instance and thus avoid attribute leakage explicitly.”. why minimizing Eq 4 can lead to instance separation and avoid attribute leakage?

### Questions
Please see my weaknesses

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
4

### Summary
The paper proposes I-DRUID, a framework for layout-to-image (L2I) generation that combines reinforcement learning and instance disentanglement learning to enhance generalization and spatial consistency. The proposed method eliminates attribute leakage between instances by introducing the Instance Disentanglement Module (IDM) and Instance Disentanglement Constraint (IDC), which separate semantically relevant and irrelevant features. A Reinforcement Learning (RL) module is applied on top of a diffusion-based generator, trained using PPO optimization on unpaired prompt-only data and guided by feedback from an external grounding model (Grounding-DINO).

### Strengths
+ Original combination of disentanglement and RL in diffusion-based L2I generation.
+ Experimental results demonstrated improved spatial alignment and instance fidelity.
+ The use of unpaired data is practical and reduces annotation costs.
+ Ablation studies show evidence of reduced attribute leakage.

### Weaknesses
- It is questionable to employ Grounding-DINO as a reward evaluator. Although it excels at object detection, it was trained for spatial grounding on real images rather than perceptual or stylistic quality. As a result, the reward may be biased toward positional correctness, unstable on synthetic generations, and insensitive to visual realism, leading to inconsistent RL behavior across datasets.
- No quantitative analysis of computational cost, which is critical given the RL overhead.
- Evaluation metrics mostly focus on geometric accuracy (mIoU, ISR); perceptual or semantic quality is not deeply discussed.

### Questions
1. Inconsistency between SD-1.5 and SD-3 results across Tables 1 and 2. The reported results are not consistent between the two backbones. On COCO-MIG (Table 1), SD-1.5 clearly outperforms SD-3 with higher Instance Success Rate (69.13 vs 62.75) and mIoU (68.18 vs 55.60). However, on LayoutSAM-eval (Table 2), the trend reverses, SD-3 achieves markedly better scores (Spatial 93.14 vs 86.95, Color 75.37 vs 70.49, Texture 78.35 vs 73.56, Shape 77.20 vs 72.30, FID 17.21 vs 22.92, IS 22.45 vs 17.95). Could the authors clarify why?
2. Grounding-DINO is primarily a detection model trained on real images, focusing on spatial grounding rather than perceptual or stylistic quality. When used as a reward function for diffusion RL, its outputs may be biased toward positional correctness while ignoring visual realism and may also be unreliable on synthetic generations. How do the authors mitigate these biases or ensure that the reward signal remains stable and meaningful during RL training?
3. Could the authors provide runtime or FLOPs comparison to quantify efficiency versus baselines?

### Soundness
3

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
5

### Summary
This paper introduces I-DRUID, a novel framework for Layout-to-Image (L2I) generation that addresses the critical challenges of attribute leakage and limited generalization. To prevent attribute leakage between instances, it proposes an Instance Disentanglement Module (IDM) that isolates semantic features for more precise attention control. To enhance generalization to novel scenes, the framework uniquely leverages reinforcement learning (RL) with unpaired, prompt-only data, enabling the model to learn from diverse AI feedback without requiring expensive paired image-text datasets. The method demonstrates state-of-the-art performance across multiple benchmarks and shows high flexibility by successfully adapting to both UNet and MM-DiT architectures.

### Strengths
1. The paper introduces a creative combination of instance-level disentanglement and reinforcement learning (RL) to address challenges in layout-to-image (L2I) generation.
2. The paper is logically structured and the language is fluent.
3. A comprehensive performance comparison was conducted against several recent SOTA methods, achieving leading results on multiple metrics and fully demonstrating the superiority of the proposed approach.

### Weaknesses
1. Using Grounding-DINO as a fixed reward model introduces bias and limits generalization. The paper does not examine robustness when GDINO fails to detect objects or misinterprets captions.
2. The framework is too complex and maks this work hard to follow. The I-DRUID framework integrates multiple complex components. While its performance is superior, its high complexity may pose challenges for researchers attempting to reproduce and build upon this work.
3. While synergy is claimed, the mechanism of mutual benefit (how disentangled features improve RL stability and vice versa) is discussed qualitatively but not empirically isolated. Table 3 does not show results with only RL applied (without IDM). It's better to provide these results to support the qualitatively analysis.

4. Minor issue: Some notations (e.g., R+, R−, CAS) could be made clearer.

### Questions
See Weakness.

### Soundness
4

### Presentation
4

### Contribution
3
