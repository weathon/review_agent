# Decoupled Classifier-Free Guidance for Counterfactual Diffusion Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 6

## Abstract
Counterfactual generation aims to simulate realistic hypothetical outcomes under causal interventions. Diffusion models have emerged as a powerful tool for this task, combining DDIM inversion with conditional generation and classifier-free guidance (CFG). In this work, we identify a key limitation of CFG for counterfactual generation: it prescribes a global guidance scale for all attributes, leading to significant spurious changes in inferred counterfactuals. To mitigate this, we propose \textit{Decoupled Classifier-Free Guidance} (DCFG), a flexible and model-agnostic guidance technique that enables attribute-wise control following a causal graph. DCFG is implemented via a simple attribute-split embedding strategy that disentangles semantic inputs, enabling selective guidance on user-defined attribute groups. Our experiments demonstrate that DCFG significantly improves the axiomatic soundness of inferred counterfactuals on challenging medical imaging data, mitigating spurious amplification effects, and enhancing counterfactual reversibility.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose Decoupled Classifier-Free Guidance, a counterfactual image generation method based on classifier-free guidance. The difference to prior work is that the authors cluster attributes into different groups and apply different guidance weights to them. THe method is empirically evaluated on CelebA-HQ, MIMIC-CXR and EMBED.

### Strengths
1. The paper is well-written. I appreciate how the approach is contrasted with the existing method by [1]. It becomes immediately clear what is done differently in this work.

2. The experimental evaluation is quite strong. The method is demonstrated on many different data sets with many examples. I am not a radiologist, so I cannot judge the mammograms and chest x-ray images. However, it is clear that this is an important application domain.

3. The proposed method is simple.

[1] Sanchez, Pedro, and Sotirios A. Tsaftaris. "Diffusion causal models for counterfactual estimation." Conference on Causal Learning and Reasoning (2022).

### Weaknesses
1. My main concern with this method is that it is somewhat trivial. The only novelty seems to lie in a grouping of attribute variables and then applying different guidance strengths to each group. Given the lack of contribution, I am afraid that this work is far from being publishable.

2. The soundness of this method is also not so clear to me. From a mathematical perspective, the weights $w_i$ should actually all be $1$. Also, the authors propose to have one weight for the "affected" variables and an additional weight for "unaffected" variables and it is not clear why this is a reasonable grouping. Generally, I do not see any deeper grounding to the argument of why we should group attributes and apply different guidance weights to them and why the groups are chosen the way that they are.

3. Some notation could be improved. For instance the $\mathrm{pa}$ notation: In case the authors do not know it, "$\mathrm{pa}$" means "parents", in the sense of "causal parents". I would therefore not use $\mathrm{pa}^{(m)}$ to denote attribute groupings, because it sounds as if we are grouping variables by their causal mechanisms (which is apparently not the case).

### Questions
* I highly suggest to remove "proposition 1". This factorization is trivial and there is nothing to be proved, as far as I can see. Maybe I am missing something?

* I do not understand why the method is based on classifier-free guidance. It seems to me that the method could also be implemented with classifier guidance.

* Why is the method called "Decoupled Classifier-Free Guidance"? What is decoupled? It seems more that it is grouped, so should be called something like "Grouped Classifier-Free Guidance".

* In appendix D, it says that the anti-correlation between "young" and "male" stems from data set bias. What is "data set bias"? To me it seems more like selection bias: Given that someone is a celebrity, being young and female are dependent.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper identifies a key limitation in the conventional Classifier-Free Guidance (CFG) approach for counterfactual generation: applying a global guidance weight $\omega$ to the entire counterfactual embedding can violate causal relations and unintentionally alter attributes that should remain unchanged. To address this issue, the author proposes Decoupled Classifier-Free Guidance (DCFG), which employs multiple MLPs, each corresponding to a specific attribute, to generate independent semantic attribute vectors and provide more targeted counterfactual generation guidance. The DCFG framework is evaluated on three datasets, demonstrating its effectiveness over conventional CFG.

### Strengths
1. The discussion on how traditional CFG can violate causal relations is interesting and can be inspirational for future research in counterfactual generation.

2. DCFG shows promising performance across all case studies.

3. The paper presents a clear and thorough explanation of the technical background and motivation.

### Weaknesses
1. The proposed solution, which uses a separate MLP for each attribute, is not scalable. For instance, while the CelebA-HQ experiments involve only three attributes, the CelebA dataset includes 40 attributes. Applying DCFG to all attributes would require 40 MLPs, which raises a serious concern on the scalability.

2. Related areas such as disentangled representation learning and debiasing for protected attributes have extensively explored methods for isolating and manipulating specific attribute features without affecting others. Although the paper claims to focus on counterfactual inference and structural causal models, its discussion of causality remains limited. Beyond the abduction–action–prediction procedure, there is little discussion to causal graphs or explicit modeling of causal relations.

3. The experiment setup appears closer to studying disentangled representation learning rather than causality, as the attributes used in the case studies (e.g., Figure A.1) are independent with each other rather than causally related. Consequently, the experiments may not sufficiently demonstrate DCFG’s ability to leverage causal relations for counterfactual generation.

### Questions
Please refer to the Weaknesses.

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
3

### Summary
In this work, the authors address the issue that classifier-free guidance for counterfactual generation can lead to attribution amplification, referring to unwanted correlations between attributes. To mitigate this problem, they propose a new method, decoupled classifier free guidance, which leverages a causal graph. The proposed approach is evaluated on three datasets, including CelebA, mammography, and chest X-rays, and demonstrates convincing results.

### Strengths
1. The motivation of this work is solid and well-justified. Avoiding spurious correlations in generative models is a challenging task that is worth pursuing.
2. This work provides the necessary prerequisites for understanding the paper in Section 2. The structure of the paper is clear and easy to follow, and the writing is overall clear and coherent.
3. The visualizations of the results are quite convincing.
The evaluation of the generated images in terms of effectiveness and reversibility also makes sense for counterfactual generation tasks.
4. I also appreciate the effort of including medical data, as it is valuable to incorporate datasets that are closer to real-world settings.

### Weaknesses
1. The paper lacks prior literature discussing the issue of attribute amplification. I believe a paragraph in sec 2 dedicated to attribute amplification is needed, clarifying how it is defined and summarizing previous studies that have encountered this issue, especially in the context of counterfactual generation.

2. There is also a lack of details on how the CFG model was trained. I am somewhat confused about the training setup: did the authors use only the target attributes for supervision, or were other attribute annotations included as well?

3. There are no numerical results presented in tables; all results are shown in figures, which makes it difficult to assess quantitative values. This could be considered a minor weakness.

4. Regarding reproducibility, it does not appear that the experiments were conducted with multiple random seeds.

### Questions
1. I am a bit unsure about why it needs to be a CFG for counterfault generation. As I believe for generating counterfactual, you have a target in mind, for example, change the disease label from 0 to 1, it makes more sense to have a classifier guidance to me. Can you explain the intuition of having CFG here? 
Following this question, the biassed results only happen when w is bigger, whether more and more CFM is introduced. If it is purely classifier guided, will there still be such an issue?

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
4

### Summary
The paper proposes Decoupled Classifier-Free Guidance (DCFG), a simple yet effective modification of standard classifier-free guidance (CFG) for counterfactual diffusion models. Standard CFG suffers from the problem of *attribute amplification*, where increasing the guidance strength for one intervention unintentionally alters correlated attributes (e.g., changing “Young” decreases “Male”). DCFG addresses this by splitting attributes into disjoint groups (intervened vs. invariant) and assigning separate guidance weights for each group. This decoupling enables more disentangled, causally faithful counterfactuals without retraining or architectural changes. Experiments on CelebA-HQ, EMBED (mammography), and MIMIC-CXR show that DCFG reduces amplification and improves reversibility compared to standard CFG.

### Strengths
- **Practical and elegant idea:** The attribute-split conditioning mechanism is extremely simple and requires minimal changes to existing diffusion pipelines while producing clear improvements.  
- **Generality:** The approach is model-agnostic and could extend to other settings.
- **Strong empirical validation:** Results are shown across both natural and medical datasets with quantitative metrics and convincing qualitative examples. The inclusion of reversibility and cross-attribute correlation analysis is helpful and demonstrates the problem concretely.

### Weaknesses
* **Clarity of motivation and intuition.**
  The paper tackles an important issue (attribute amplification in classifier-free guidance) but why this happens is not very clearly explained. The reviewer had to re-read [1] and infer the connection independently. Including a brief intuitive explanation, figure or toy example would make the motivation easier to grasp.

* **Simplicity and missing contextualization.**
  The proposed fix of splitting attributes into intervened and invariant groups and applying separate guidance weights is extremely simple and easy to adopt, which is a strength. However, the paper does not discuss whether similar disentangling or conditional-guidance approaches have been explored before, or how this method compares to more sophisticated alternatives for representation disentanglement. A short discussion or empirical comparison would clarify novelty.

* **Missing baselines and ablations.**
  The experiments compare only to standard CFG. It would strengthen the work to include other diffusion-based counterfactual explanation or editing baselines (e.g., [2], [3]).

* **Incomplete evaluation metrics.**
  The paper does not report realism measures such as FID or sFID, nor composition or minimality metrics commonly used in counterfactual image generation. The authors explain the omission of composition in the Appendix, mentioning this in the main text would help. Measuring also minimality through a VLM or a user study, or a brief note on why it is difficult to quantify would make the evaluation more comprehensive.

* **Unaddressed observations.**
  In Figure 1, increasing guidance for *do(Young)* appears to reduce *Male*, likely due to a dataset bias that biases the classifier. This should be discussed and attributed to this bias or another factor. Figure 5 shows similar unexplained behavior (*do(circle)* affecting density AUROC).

* **Minor presentation issues.**
  - Figure 5 would benefit from same interventions between subfigures.
  - Figure 3 is difficult to read and should be split.
  - Equation (5) seems to omit $\tilde{x}$
  - Equation (12) may need a $(1-\omega)$ term for completeness
  - line 284 should reference Appendix C.

[1] Tian Xia, Mélanie Roschewitz, Fabio De Sousa Ribeiro, Charles Jones, and Ben Glocker. Mitigating
attribute amplification in counterfactual image generation. In International Conference on Medical
Image Computing and Computer-Assisted Intervention, pp. 546–556. Springer, 2024.

[2] Guillaume Jeanneret, Loı̈c Simon, and Frédéric Jurie. Diffusion models for counterfactual explana-
tions. In Proceedings of the Asian conference on computer vision, pp. 858–876, 2022.

[3] Preechakul, K., Chatthee, N., Wizadwongsa, S., & Suwajanakorn, S. (2022). Diffusion Autoencoders: Toward a Meaningful and Decodable Representation. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

### Questions
1. Can you provide a clearer intuition, illustrative example or figure for why CFG leads to attribute amplification?
2. Have you tried a version that removes invariant attributes from conditioning entirely?
3. How does DCFG compare with other diffusion-based counterfactual or editing methods such as DiME, or diffusion autoencoders?
4. Can you report or discuss realism metrics (FID/sFID) and briefly justify the exclusion of composition/minimality metrics?
5. Could you clarify the observed cross-attribute effects in Figures 1 and 5 and verify the potential inconsistencies in Equations (5) and (12)?

### Soundness
3

### Presentation
3

### Contribution
2
