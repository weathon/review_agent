# Better Together: Leveraging Unpaired Multimodal Data for Stronger Unimodal Models

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 6, 2

## Abstract
Traditional multimodal learners find unified representations for tasks like visual question answering, but rely heavily on large paired datasets. However, an overlooked yet potentially powerful question is: can one leverage auxiliary $\textit{unpaired}$ multimodal data to directly enhance representation learning in a $\textit{target}$ modality? We introduce $\textbf{UML}$: $\textbf{U}$npaired $\textbf{M}$ultimodal $\textbf{L}$earner, a modality-agnostic training paradigm in which a single model alternately processes inputs from different modalities while sharing parameters across them. This design exploits the assumption that different modalities are projections of a shared underlying reality, allowing the model to benefit from cross-modal structure without requiring explicit pairs. Theoretically, under linear data-generating assumptions, we show that unpaired auxiliary data can yield representations strictly more informative about the world than unimodal training. Empirically, we show that incorporating unpaired data that share underlying semantic information from auxiliary modalities—such as text, audio, or images—consistently improves downstream performance across diverse unimodal targets such as image and audio. Our project page: https://unpaired-multimodal.github.io/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose the Unpaired Multimodal Learner (UML), which leverages auxiliary unpaired multimodal data to enhance representation learning in a target modality. Specifically, the authors present two frameworks, one for self-supervised learning and one for supervised learning, both of which utilize multiple encoders and a shared network. Theoretical and empirical results indicate that adding unpaired data of modality Y can lead to better reconstruction in modality X compared to adding additional data from X itself.

### Strengths
1. The topic addressed in the paper is interesting.
2. The paper is clearly written and easy to follow.
3. The theoretical analysis is sound, and the proposed method is simple yet effective.

### Weaknesses
1. The primary concern lies in the motivation: unpaired multimodal data is relatively uncommon, whereas missing-modality data is more frequently encountered. If the data is only unpaired, why not use existing models (e.g., CLIP) to align them and create paired data? Therefore, a more practical setting might be using images from one dataset alongside text from another dataset.
2. It would strengthen the paper to include experiments using various encoders and various shared networks to validate the general effectiveness of the method.

### Questions
1. The experiments do not include comparisons with related works. Do the authors think it is necessary to compare against other approaches mentioned in the Further Related Works section?

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
4

### Summary
This paper proposes a novel method, **UML** (**U**npaired **M**ultimodal **L**earner), to improve the performance of unimodal models by leveraging abundant, unpaired data from other modalities.
The core idea is that even without direct pairings (like an image and its specific caption), data from an auxiliary modality (e.g., text) can provide complementary information to enhance a model focused on a target modality (e.g., images).
The **UML** method works by having a single model with shared parameters (weight sharing) alternately process inputs from the different modalities. This design allows the model to capture shared underlying concepts and structures from both datasets, even though they are not explicitly linked.
Moreover, the paper theoretically demonstrates that this approach strictly increases the Fisher information under linear assumption of generating data, resulting in more informative and robust representations for the target modality than training on that modality alone.

### Strengths
- This paper presents strong theoretical approaches demonstrating how unpaired multimodal datasets, or datasets with missing modalities, can be effectively synergized with existing datasets. It shows that these datasets can be linearly combined, leading to an increase in Fisher Information.
- The paper provides extensive experiments on various benchmark datasets that support its theoretical framework, validating the proposed methods and concepts.
- The paper is well written, particularly in the Introduction and the section explaining the main concept.

### Weaknesses
**Major**

The paper’s effort to ground its empirical findings in theoretical analysis is commendable; however, the theoretical assumptions appear too restrictive to be fully explanatory. The authors provide valuable intuition for UML by presenting theorems (Sec. 3.1) derived under a linear data-generating process.
Nonetheless, this represents a significant simplification of the highly non-linear dynamics of the large Transformer-based models (e.g., DINOv2, OpenLLaMA) used in the experiments. While the idea that shared weights act as a practical analogue of *Fisher information linear combination* is intriguing, it remains more of an intuitive analogy than a formal justification.
In my view, this creates a potential gap between theory and practice. The empirical results are undeniably strong and well-presented, yet they seem to be motivated by the theoretical framework rather than explained by it.

**Minor**

- Although the theoretical motivation is clearly written, the derivations of the theorems are somewhat verbose and occasionally redundant, which reduces readability. A more concise and streamlined presentation could enhance the clarity of the theoretical section.
- While the experimental validation is extensive, its presentation in Section 4 and the Appendix seem overly dense. For example, much of the dataset and model information is relegated to the Appendix and only briefly mentioned in Section 4.1, where it hinders readability. Additionally, the color scheme used in Table 1 and Table 2 (blue and pink) is identical, even though the categories differ (Table 1: datasets; Table 2: settings).
As a suggestion (not a requirement), reorganizing the experimental sections might improve clarity. (e.g., focusing on the supervised setting in Section 4.1 and moving the self-supervised setting to Section 4.2, and removing the color scheme from Tables 1 and 2 to avoid confusion)

### Questions
- What is the main difference from prior works, and what constitutes the key novelty of this paper? In my view, as the authors mention in Section 3 and Appendix A.1, UML might seem slightly incremental, as it combines concepts from previous studies, such as shared model parameters (e.g., [1]), and the use of unpaired datasets (e.g.,[2]).
- What happens if the text or data (modality $\text{Y}$) are randomly generated? For instance, what if the text description is entirely unrelated to the image (e.g., the image depicts a dog, but the text describes playing sports)? Would such mismatched modalities disrupt the increase in Fisher Information?
- Yet the authors note in the limitations section that most experiments are conducted on classification tasks, I still raise a concern regarding the applicability of the proposed method to other tasks, such as image–text retrieval.


[1]  Chada, et al. "Momo: A shared encoder model for text, image and multi-modal representations." arXiv preprint 2023\
[2] Lee, Jae-Jun, and Sung Whan Yoon. "Can One Modality Model Synergize Training of Other Modality Models?." ICLR 2025

=======================================================

**Note**: I acknowledge that I may have partially misunderstood certain aspects of the paper. Therefore, I am willing to raise my rating score if these questions and concerns are adequately addressed.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose to augment unimodal models with unpaired training data from other modalities. Their approach shares weights across all modalities while optimizing for the downstream task on a joint, unpaired, multimodal training dataset. They evaluate their approach on several supervised, self-supervised, and transfer learning settings.

### Strengths
1. The paper is very well written - ideas and motivations are expressed clearly and in easy-to-understand terms; experimental results are stated and discussed in a well-organized manner.

2. The experiment in Section 4.4 on marginal rate-of-substitution between modalities is quite interesting and the results are insightful.

3. The authors empirically show that in the scenarios considered, it is indeed possible to achieve some practical performance improvement through joint/pre-training with data from modalities other than the target.

### Weaknesses
1. The idea that data from multiple modalities can be used for unified pretraining without any special consideration for the nature of the modalities being integrated is a fundamentally flawed premise. It is well known in literature that modalities can often have conflicting information, and even multimodal models, which often operate on explicitly paired data and are trained with the objective of aligning the modalities, struggle with this integration. There are plenty of works dedicated to addressing specifically this problem [a, b]. At the low level, for instance, modalities can often have different convergence rates [c] and provide conflicting gradients to the model [d], none of which can be thought of as being helpful from a training perspective without special treatment [d]. In fact, [c] specifically established the general impossibility of what the authors propose in this work.

2. Although the experiments show performance some improvements, the generality with which the paper is presented is misleading. Attempting to perform joint multimodal training using the proposed approach in the settings and datasets used in [a, b, c, d], is likely to falsify the findings reported, since special adjustments to cope with the challenges of integrating multiple modalities is needed to deal with those settings.

3. No ablation or analytical studies have been reported for the proposed approach, which makes it difficult to evaluate the contribution of the various design choices, for instance, the proposed UML objectives, the contribution of unpaired samples from a different modality, convergence in the uni-modal vs multi-modal setting, etc.

4. Finally, neither the idea of sharing model weights, nor doing pretraining with data from multiple modalities can be regarded as novel, since they have been around in the multimodal learning community for a long time [e, f, g]. In fact, this is also the problem that works on domain generalization attempt to solve, albeit framed in a different manner [h].

Minor:\
Line 449: "Further results on" -> "For further results on"

References:\
[a] Zhang et al., "Robust Multimodal Large Language Models Against Modality Conflict", ICML 2025.\
[b] Ma et al., "Improving Multimodal Learning Balance and Sufficiency through Data Remixing", ICML 2025.\
[c] Wang et al., "What makes training multi-modal classification networks hard?", CVPR 2020.\
[d] Javaloy et al., "Mitigating Modality Collapse in Multimodal VAEs via Impartial Optimization", ICML 2022.\
[e] Ngiam et al., "Multimodal Deep Learning", ICML 2011.\
[f] Hu et al., "Towards Unsupervised Sketch-based Image Retrieval", BMVC 2022.\
[g] Rastegar et al., "MDL-CW: A Multimodal Deep Learning Framework with Cross Weights", CVPR 2016.\
[h] Gulrajani et al., "In Search of Lost Domain Generalization", ICLR 2021.

### Questions
Please refer to the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
