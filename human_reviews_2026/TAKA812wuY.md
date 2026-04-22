# Beyond Overconfidence: Rethinking Calibration in Large-Scale Vision Models

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Reliable uncertainty calibration is crucial for the safe deployment of deep neural networks in high-stakes settings. While these networks are known to exhibit systematic overconfidence, especially under distribution shifts, the calibration of large-scale vision models, such as ConvNeXt, EVA, and BEiT, has remained underexplored. We comprehensively examine their calibration behavior, uncovering evidence that challenges well-established assumptions. We find that these models are underconfident on in-distribution data, which results in increased calibration error, yet exhibit improved calibration under distribution shifts. This phenomenon is primarily driven by modern training techniques, including massive pretraining and sophisticated regularization and augmentation methods, rather than architectural innovations alone. We also demonstrate that these large-scale models are highly responsive to post-hoc calibration techniques in the in-distribution setting, enabling practitioners to mitigate underconfidence bias effectively. However, these methods become progressively less reliable under severe distribution shifts and can occasionally produce counterproductive effects. Our findings highlight the complex, non-monotonic effects of architectural and training innovations on calibration, challenging established narratives of continuous improvement.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper evaluates several state-of-the-art vision models on ImageNet-1k and its distribution-shifted variants (ImageNet-C, ImageNet-A, ImageNet-V2), assessing both calibration—measured via top-label Expected Calibration Error (ECE)—and accuracy. The six models considered are ResNet, ViT, Swin, BEiT, EVA, and ConvNeXt. The results show that **ConvNeXt, EVA, and BEiT tend to be under-confident** in-distribution but exhibit improved calibration under distribution shifts, whereas ResNet, ViT, and Swin display the opposite pattern, being better calibrated in-distribution but at the cost of lower accuracy.

The authors then investigate two factors that may influence this calibration behavior. First, for a fixed ViT architecture, they show that switching the pretraining objective from supervised to contrastive learning improves accuracy but leads to under-confidence.
Second, for a fixed ViT or ResNet architecture, they find that applying advanced regularization and data augmentation techniques similarly improves accuracy while creating under-confidence.

Finally, the authors demonstrate that this miscalibration can be effectively mitigated by standard post-hoc calibration methods, although these techniques remain less effective under severe distribution shifts.

### Strengths
* The paper is clearly written and easy to follow.

* I have seen results showing that base LLMs tend to be under-confident (e.g., [Cruz2024]), while their instruction-tuned versions become over-confident. However, I am not aware of prior work reporting that models such as ConvNeXt, EVA, and BEiT are under-confident in-distribution and better calibrated under distribution shifts. To my knowledge, this observation is therefore novel.

[Cruz 2024] Cruz et al, Evaluating language models as risk scores

### Weaknesses
I think the significance and potential impact of the paper are limited by the relatively narrow scope of the experimental study, which evaluated the ECE of 6 models + 4 variants.


In line 462, the authors mention that the paper focuses on diagnostics (e.g., identifying under-confidence in certain models) rather than uncovering its causes. However, given ICLR’s standards, it would be reasonable to expect a broader experimental exploration in this direction, across architectures, model sizes, pretraining objectives, regularization strategies, fine-tuning, etc (or at least a subset of these.) I acknowledge that Section 4.3 provides preliminary insights, and they are very valuable. But again for ICLR standards, I would expect deeper insights in this direction.

### Questions
- Did the authors train any of the evaluated models, or are they all available on Hugginface? If yes, could the authors provide the links to the models used?
- In section 4.2, the authors state that large-scale models exhibit systematic in-distribution underconfidence, encompassing ConvNext, BEiT & EVA in the large scale models. However, why is the ViT not considered as large scale, since it is also pretrained on ImageNet-21k, and is (I would say) not smaller than the other models ?

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
This paper systematically evaluates the uncertainty calibration performance of three large-scale vision models—ConvNeXt, EVA, and BEiT—and reveals findings that contradict prior research conclusions. The study shows that these models generally exhibit systematic underconfidence on in-distribution data, instead of the commonly reported overconfidence of tranditional models, leading to higher calibration errors. What's more, their calibration error decreases under out-of-distribution conditions. Furthermore, the authors demonstrate that these models respond well to post-hoc calibration methods (e.g., Temperature Scaling) in in-distribution scenarios. Nonetheless, the effectiveness of such methods degrades significantly under severe distribution shifts.

### Strengths
1. The overall structure of the paper is clear and well-organized.

2. The related work section clearly articulates how this paper differs from previous studies, allowing readers to quickly grasp the central contributions.

3. The experiments are diverse and conducted across multiple architectures and datasets. Figures 4(a) and 6 effectively correspond to and support the main conclusions.

### Weaknesses
1. There are several typos and grammatical errors throughout the paper. The authors should carefully proofread the manuscript to meet publication standards. For example, there is a mistake "Eq. ??" in line 135.

2. Some of the paper’s conclusions are not clearly stated. For instance, Section 4.4 only provides descriptive commentary on figures without drawing any conclusions. If the intended point is that “Figure 3 shows architecture-specific differences in effectiveness,” this claim is neither visually evident from the figure nor supported by textual explanation. Similarly, while Figure 4 is later discussed, the meaning of the solid and dashed markers and the significance of their connecting lines are not made clear from the figure itself, although in the latter context.

3. Certain conclusions lack sufficient evidence or persuasiveness. For example, the findings in Section 4.3 are only validated on a single model, which limits their generality. In Section 5, the final key insight labeled as “best” actually refers only to being the best among the three evaluated models, rather than a universal best.

### Questions
Why did the authors choose to evaluate these three models rather than using larger-scale models such as LLaVA or Qwen-VL?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates how miscalibration differs under various distribution shifts, showing that models tend to be *underconfident* for in-distribution data but become *overconfident* under out-of-distribution (OOD) shifts.

### Strengths
**Pros**

* It is valuable to examine calibration performance across different data groups, such as in-distribution and out-of-distribution scenarios.
* The writing is clear to follow.

### Weaknesses
**Cons**
* Missing important related works with similar findings [1] or relevant methods for reducing miscalibration across different groups [1][2].
* Lacks theoretical analysis or insights explaining the observed calibration behaviors.
* The evaluated models are still limited in scope compared to prior comprehensive studies [1][3].
* The overall contribution is limited. Most findings have already been identified in prior works, such as miscalibration under group shifts [1] and the accuracy–calibration trade-off [3]. 


---

**References**

[1] Xiong, Miao, et al. *"Proximity-informed calibration for deep neural networks."* NeurIPS 2023

[2] Perez-Lebel, Alexandre, Marine Le Morvan, and Gaël Varoquaux. *"Beyond calibration: estimating the grouping loss of modern neural networks."* ICLR 2023

[3] Minderer, Matthias, et al. *"Revisiting the calibration of modern neural networks."* NeurIPS 2021

### Questions
Please see **Weaknesses**.

### Soundness
2

### Presentation
2

### Contribution
2
