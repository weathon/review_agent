# UnLoc: Leveraging Depth Uncertainties for Floorplan Localization

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 4, 4, 6

## Abstract
We propose UnLoc, an efficient data-driven solution for sequential camera localization within floorplans. Floorplan data is readily available, long-term persistent, and robust to changes in visual appearance. We address key limitations of recent methods, such as the lack of uncertainty modeling in depth predictions and the necessity for custom depth networks trained for each environment. We introduce a novel probabilistic model that incorporates uncertainty estimation, modeling depth predictions as explicit probability distributions. By leveraging off-the-shelf pre-trained monocular depth models, we eliminate the need to rely on per-environment-trained depth networks, enhancing generalization to unseen spaces. We evaluate UnLoc on large-scale synthetic and real-world datasets, demonstrating significant improvements over existing methods in terms of accuracy and robustness. Notably, we achieve $2.7$ times higher localization recall on long sequences (100 frames) and $42.2$ times higher on short ones (15 frames) than the state of the art on the challenging LaMAR HGE dataset.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes combining pre-trained depth models with uncertainty modeling to improve the accuracy of floor-plan localization. Experiments are conducted on multiple datasets, including both real-world and synthetic scenes.

### Strengths
1. The paper is generally well written and easy to follow.

2. It makes sense to leverage uncertainty information for localization.

3. The proposed method achieves state-of-the-art performance on two floor-plan localization benchmarks.

### Weaknesses
1. The paper mainly integrates existing methods and applies them to a relatively narrow task—floor-plan localization. While the topic is relevant, the contribution appears incremental rather than conceptually novel. This paper does not introduce any new contribution in terms of problem formulation; it focuses on a narrow task with clearly defined inputs and outputs.

2. As shown in Table 6, F3Loc + Depth Anything V2 already achieves higher accuracy than UnLoc on several datasets. This suggests that much of the performance gain may stem from the use of off-the-shelf pre-trained depth foundation models rather than from the new proposed method in this paper. The paper, therefore, reads more as an application of pre-trained depth models to a specific downstream task than as a novel algorithmic/theory contribution. 

3. The discussion on uncertainty is underdeveloped. Uncertainty in neural network predictions can generally be categorized into aleatoric and epistemic uncertainty [1], yet this paper lacks a theoretical explanation or analysis of which type is modeled or how it contributes to performance improvements.

4. The data preprocessing and benchmark setup are not sufficiently transparent. The paper appears to introduce newly processed data and custom split strategies from public datasets, which raises concerns about reproducibility. It is unclear whether the test set division used in this work aligns with the official benchmark split and whether the authors may have chosen a split that yields the best performance. Providing clear details on dataset handling and evaluation would significantly improve the credibility of the results. This paper does not provide the recommended Reproducibility statement and the Use of Large Language Models (LLMs) statement. 


[1] Alex Kendall et al., What uncertainties do we need in Bayesian deep learning for computer vision?, NeurIPS 2017.

### Questions
I am particularly curious about how this work is evaluated on the LaMAR dataset. To my knowledge, LaMAR does not include publicly available floor plans. Could the authors clarify where the floor plan data used in this paper originated? The description in the Appendix is quite brief and would benefit from a more detailed explanation. 

Moreover, LaMAR does not provide ground-truth poses for the test set, and the paper does not specify the session divisions used. Since the evaluation is based on estimating 2D poses on floor plans, it remains unclear how these poses were submitted or compared within the LaMAR benchmark to produce the results shown in Table 2. Additionally, even if the official floor plans were obtained, how were the 6Dof camera poses from LaMAR aligned with the 2D pose in the floor plans? If you use custom's align algorithm to calculate the ground-truth label of 2D poses, how can you ensure that this ground-truth itself is reliable?
 
A detailed explanation of the data processing pipeline and evaluation protocol would be very helpful. 

I also noted that F3Loc did not include such evaluation details—it only released the link to the Gibson Floorplan Localization Dataset. This raises further concerns about the fairness and transparency of comparing F3Loc and the proposed method on the LaMAR HGE/CAB benchmark. Overall, this area still lacks openly available datasets and standardized baselines, making reproducibility a critical issue at this stage.

If the authors could provide a more comprehensive description of the evaluation methodology and experimental setup on LaMAR, I would consider improving my overall rating. Furthermore, I strongly encourage the authors to release the processed version of the LaMAR data used in this work. Since the paper involves substantial preprocessing of the raw LaMAR dataset, referring to it as “widely used” without sharing those processed resources seems somewhat inappropriate and limits the community’s ability to validate the claimed improvements in accuracy and reproducibility.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents UnLoc (Uncertainty-aware Localization), a method for sequential camera localization within 2D floorplans. The input of the method is: 
1/ a sequence of t RGB images,
2/ relative poses between these images,
3/ gravity direction 
4/ camera intrinsics, and
5/ geometry layout of the floorplan

It aims to address two key shortcomings of previous state-of-the-art methods: the lack of uncertainty modeling in depth predictions and the need for custom, environment-specific depth networks for each environment (e.g. in $F^3Loc$). The paper reports strong performance compared to previous methods on LaMAR HGE dataset.

### Strengths
(+) Strong performance shown in the experiment section, although the paper mainly focuses on LaMAR HGE dataset

(+) Compared to the previous method, such as $F^3Loc$, the proposed method is more general, in the sense that it does not rely on per-scene depth estimator, which significantly improves the usability of such methods

(+) the writing is the paper is good -- it is relatively easy to the reviewer to follow

(+) Using a depth encoder seems new to me and makes a lot of sense

### Weaknesses
(-) Biggest concern from me is the lack of novelty and suitability in the sense of the ICLR (learning representation) community. The paper is a 3D computer vision problem in pose estimation in a room. The main contributions / problems to improve from previous methods are 1/ the lack of uncertainty modeling in depth predictions and 2/ the need for custom, environment-specific depth networks for each environment. These are very specific novelties for this particular task, instead of any major novelty for the learning representation community

(-) The proposed method feels like using / combining a few existing tools/engineering techniques (DepthAnything, Rotate/Translate the images, Floorplan matching, and heavy post-processing etc) and made it work for a particular task. This feels more suitable for submission to a 3DV conference, CVPR or robotics conference, instead of a machine learning conference

(-) The performance of the method will be limited by DepthAnything's accuracy, so this paper essentially trades accuracy ceiling with generalizability

(-) the experiments focus heavily on a small dataset

### Questions
n/a

### Soundness
3

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
The paper proposes a method for visual floorplan localization that integrates uncertainty-aware monocular depth estimation. Specifically, the authors leverage large-scale pre-trained monocular depth models (e.g., Depth Anything) to (1) enhance generalization to unseen environments, and further (2) incorporate weighting on depth predictions based on their estimated confidence/reliability. Experimental results show substantial gains in both accuracy and robustness over existing methods.

### Strengths
- Simple yet sound idea that leads to performance improvements.
- This presentation is clear and easy to follow, with figures that are mostly intuitive and effectively support the explanations.
- The evaluation section is thorough – Table 1 and 2 showing sequence localization on Gibson and LaMAR HGE datasets, respectively. The author also reports the runtime which is crucial for localization tasks. Furthermore, the paper also ablates different depth estimation models and validates the effectiveness of the proposed uncertainty estimation.

### Weaknesses
- The overall technical contribution is somewhat limited, as the main novelty lies in replacing the F3Loc depth networks with SOTA monocular depth estimation models. Moreover, uncertainty modeling in depth estimation has already been explored in many prior works [1, 2].
- Lack of discussion about the uncertainty in the related works or preliminary. The author simply throws the equation at Sec3.5 without sufficient explanation of its derivation or references to prior studies that motivated it.

[1] Kendall and Garl, What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

[2] Poggi et al., On the uncertainty of self-supervised monocular depth estimation.

### Questions
- Is this depth encoder being fine-tuned during training? If it is, wouldn’t this make the approach similar to F3Loc, which requires model training for each environment?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents a method for floor plan localisation based on a sequence of input images. Its main contribution lies in introducing an uncertainty-aware depth estimation framework and replacing customised depth models with modern monocular depth networks. As a result, the proposed approach outperforms prior methods on two datasets, LaMAR and Gibson.

Strength: The idea of incorporating uncertainty into the depth matching process is novel and technically well-motivated. The experimental results convincingly support the method’s effectiveness.

Weakness: The evaluation lacks results on the Structured3D dataset, which is used by the closely related baseline F3Loc. Moreover, replacing a customised depth network with a modern monocular model, while beneficial, is a relatively straightforward modification.

### Strengths
1. The idea of introducing uncertainty into the depth matching process is novel and technically sound.
2. The experiments are comprehensive and largely convincing.
3. The paper is well written and easy to follow.

### Weaknesses
1. The results on the Structured3D dataset are missing, which is used in the closely related baseline F3Loc.
2. Replacing the customized depth networks with modern monocular depth networks is relatively straightforward.

### Questions
1. Could you include results on the Structured3D dataset?
2. Are there any challenges in replacing the customized depth network with monocular depth networks?

### Soundness
3

### Presentation
3

### Contribution
2
