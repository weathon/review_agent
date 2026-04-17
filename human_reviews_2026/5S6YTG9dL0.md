# Disrupting Hierarchical Reasoning: Adversarial Protection for Geographic Privacy in Multimodal Reasoning Models

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Multi-modal large reasoning models (MLRMs) pose significant privacy risks by inferring precise geographic locations from personal images through hierarchical chain-of-thought reasoning. Existing privacy protection techniques, primarily designed for perception-based models, prove ineffective against MLRMs' sophisticated multi-step reasoning processes that analyze environmental cues. We introduce **ReasonBreak**, a novel adversarial framework specifically designed to disrupt hierarchical reasoning in MLRMs through concept-aware perturbations. Our approach is founded on the key insight that effective disruption of geographic reasoning requires perturbations aligned with conceptual hierarchies rather than uniform noise. ReasonBreak strategically targets critical conceptual dependencies within reasoning chains, generating perturbations that invalidate specific inference steps and cascade through subsequent reasoning stages. To facilitate this approach, we contribute **GeoPrivacy-6K**, a comprehensive dataset comprising 6,341 ultra-high-resolution images ($\geq$2K) with hierarchical concept annotations. Extensive evaluation across seven state-of-the-art MLRMs (including GPT-o3, GPT-5, Gemini 2.5 Pro) demonstrates ReasonBreak's superior effectiveness, achieving a 14.4\% improvement in tract-level protection (33.8\% vs 19.4\%) and nearly doubling block-level protection (33.5\% vs 16.8\%). This work establishes a new paradigm for privacy protection against reasoning-based threats.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a defense method that aims to disrupt geographic reasoning in large multimodal reasoning models by adding perturbation noise. The authors conduct experiments across multiple models and release the GeoPrivacy-6K dataset, which contains high-resolution, spatially localized images annotated with bounding boxes. The study addresses an important privacy issue and contributes a new dataset and adversarial defense framework.

### Strengths
The work targets an emerging privacy threat related to geographic inference from images by large reasoning models.

The authors propose an adversarial approach that effectively disrupts hierarchical reasoning through concept-aware perturbations.

The new GeoPrivacy-6K dataset is a valuable contribution, containing ultra-high-resolution and spatially localized images.

### Weaknesses
Dataset Size and Analysis. It would strengthen the paper to analyze which types of images are more vulnerable to attacks and which are more resistant. Please summarize the visual or semantic characteristics of easily attacked versus robust images.

Computational Performance. Include a comparison of running time and GPU computation cost between ReasonBreak and other defense baselines.
Resolution Impact. The paper uses ultra-high-resolution images. Please explain why this resolution level was chosen and analyze whether image resolution affects attack or defense effectiveness.

Dataset Relationships. Clarify the relationship between GeoPrivacy-6K and DOXBENCH. Are they complementary or overlapping?

Defense Comparison. Compare the performance of your defense method (ReasonBreak) against differential privacy (DP) methods implemented in DOXBENCH or similar benchmarks.

Appendix Suggestion. It would be helpful to include sample images from GeoPrivacy-6K in the appendix to illustrate the dataset’s diversity, annotation quality, and spatial localization.

### Questions
see Weakness

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
This paper proposes a novel attack approach to break the mulitmodal large reasoning model (MLRM) for performing geographical inference. To support this, the authors introduce a new dataset called GeoPrivacy-6K, which contains 6,000 more images and concept annotations for training. Experiments on DOXBENCH demonstrate that the proposed method achieves superior results compared to previous approaches on both open- and closed-source models.

### Strengths
* The motivation of the paper is intriguing and highly relevant, especially for protecting users’ geographical information in line with existing privacy regulations.
* The proposed idea is supported by reasonable theoretical analysis, making the design both robust and practical.
* The experiments are thorough, and the results are impressive. In particular, the introduction of the protected success rate metric, instead of the usual attack success rate, is a valuable addition for future evaluations.

### Weaknesses
* While the paper is well-motivated, the methodology is somewhat difficult to follow. For example, in Equation (3), the meaning of defining $(m^\*, n^\*)$ is unclear. Do the authors mean that all $(m^\*, n^\*)$ follow the stated condition? Also, terms like "concept subset" and "spatial overlap analysis" are mentioned without sufficient explanation. It would help to clearly define these terms or include a figure illustrating the training pipeline.
* The proposed min-max target selection involves finding a "null point" from the database. This might be a limitation because if an image is dissimilar or unrelated to the database, the results could be negatively affected. Moreover, the reasoning behind selecting $\min \{ \mathbf{e} \in \mathcal{E} \}$ is not fully explained and would benefit from further clarification.
* The choice of $N_{\text{max}}$ is not entirely convincing. As shown in Figure 4, when $N_{\text{max}} > 16$, the PRR of the "metro" class drops to zero. Although other classes improve, this choice does not appear optimal. It would be helpful to explain why the "metro" class is considered less important or why this trade-off is acceptable.

---

**Overall**: The paper addresses an important and timely topic with clear motivation and solid contributions. The results are promising and provide meaningful insights for the community. However, the methodological descriptions are sometimes unclear and would benefit from additional clarification. My current score is 4, and I would be happy to raise it once these issues are addressed.

### Questions
The questions are mainly about the weaknesses.

1.	What are the "concepts" used for training?
2.	Does using the database as an anchor point affect the results for images that fall outside the database distribution?
3.	Is the inclusion of $\min \{ \mathbf{e} \in \mathcal{E} \}$ intended to prevent the reconstructed error from deviating too far from the original images?
4.	Could the authors elaborate further on the selection of $N_{\text{max}}$ and its effect on different classes?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents ReasonBreak, an innovative adversarial framework that effectively counters geographic privacy threats posed by Multi-modal Large Reasoning Models (MLRMs) by disrupting their hierarchical reasoning processes through concept-aware perturbations. Unlike conventional methods that apply uniform noise, our approach strategically targets critical conceptual dependencies within reasoning chains, causing cascading failures in location inference. The framework is enabled by GeoPrivacy-6K, a comprehensive dataset of 6,341 ultra-high-resolution images with hierarchical annotations. Extensive evaluation across seven state-of-the-art MLRMs demonstrates remarkable effectiveness, achieving 33.8% tract-level and 33.5% block-level protection rates - nearly doubling baseline performance and establishing a new paradigm for reasoning-aware privacy defense against sophisticated AI threats.

### Strengths
1.The paper introduces the first concept-aware adversarial framework, ReasonBreak, specifically designed to target the hierarchical reasoning chains of multimodal large models. It achieves a paradigm shift in defense from the perception layer to the reasoning layer through a minimax target selection strategy.
2.The black-box transfer capability of this framework has been successfully validated on commercial APIs, including state-of-the-art models such as GPT-5 and Gemini 2.5 Pro, demonstrating its practical effectiveness in real-world scenarios.

### Weaknesses
1.The scenario of the paper is very clear. However, it does not clearly point out the technical difficulties of the problems in this scenario. It is hoped that the problems under this scenario will be analyzed.
2.It is recommended to additionally supplement a module diagram in the framework overview of Section 4.3 on page 5 to visually display the ReasonBreak framework.
3.The author only conducted training on the datasets they proposed and performed black-box testing on open-source datasets. It is recommended that the author provide white-box experimental results.
4.The author states that the training set used has a high resolution. What would be the impact if a low-resolution dataset were used?

### Questions
1.The scenario of the paper is very clear. However, it does not clearly point out the technical difficulties of the problems in this scenario. It is hoped that the problems under this scenario will be analyzed.
2.It is recommended to additionally supplement a module diagram in the framework overview of Section 4.3 on page 5 to visually display the ReasonBreak framework.
3.The author only conducted training on the datasets they proposed and performed black-box testing on open-source datasets. It is recommended that the author provide white-box experimental results.
4.The author states that the training set used has a high resolution. What would be the impact if a low-resolution dataset were used?

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
2

### Summary
This paper studies the problem of protecting the geographic privacy of images due to the ability of multimodal large reasoning models (MLRMs) to perform image geolocation. The authors first create a new GeoPrivacy-6K dataset consisting of high-resolution images of locations and bounding-box-based concept or feature annotations that are used during the defense strategy. The proposed defense ReasonBreak is motivated by the supposed step-by-step geolocation reasoning of reasoning models in inferring geolocation. If perturbations can affect any step in this reasoning process, the disruption can cascade to cause incorrect geolocation outputs from MLLMs. ReasonBreak works by identifying relevant concepts in each block of the image, and then finding a suitable perturbation that pushes the embedding of the perturbed block far from the original concepts it contained. The authors conduct an analysis of the effectiveness of their method across numerous open and closed source models and note its defensive effectiveness against baselines. Finally, the authors also perform ablations on the perturbation budget, block size, and the use of their target selection algorithm.

### Strengths
1. The paper introduces a novel method to disrupt the geolocation reasoning of MLRMs. To my knowledge, this is the first paper that explicitly targets the reasoning ability of MLRMs, by focusing on especially important concepts used in the reasoning process. It seems like this general paradigm would be extended outside of the domain of image geolocation to other intensive MLLM reasoning tasks. 

2. ReasonBreak seems to significantly outperform the more general-purpose attack baselines, while remaining imperceptible, which is important for image geolocation, especially, i.e., many real-world attack use-cases may use social media images.

3. ReasonBreak does not require model weights to create defenses. This is a significant benefit, given that the most powerful and accessible MLRMs are closed source.

### Weaknesses
1. The theoretical motivation is a bit confusing. In lines 208-209, you mention that the model "progressively refines its location," and in lines 235 - 236, you mention that "an error introduced at an early state ... does not remain localized". While there have been methods that induce this sort of least to most reasoning for image geolocation through prompting [1], generally, R1-style reasoning models can backtrack and revisit conclusions made earlier in their reasoning trace.

2. The dataset construction, especially the scene annotation phase, is missing quite a few details. Did a model perform the annotation? If so, what model and what was the prompt for the model? How are bounding boxes identified by a model if it is used? It seems like the method requires similar concept annotations on any inference image, is this true? If so, how expensive are these image annotations for each image? Do you have an ablation on the model used for concept annotation with overall performance i.e., it would be great if this works well even with an open source VLM annotator?

3. Some parts of the method are unclear / could use an improvement in their description. For instance, the definition of N_max has to be inferred from the partitioning in equation 1. It would be good to define this clearly since there is a later ablation on this variable. The concept assignment section is also a bit confusing. What is the default concept? Does it have a specifically chosen embedding? How much overlap does there need to be between a concept and a block for the concept to be assigned?

4. The training set images are high resolution. Why was the choice made? It seems many social media images may be highly compressed, and these are often the images that adversaries may want to geolocate for malicious purposes. It would be necessary to verify that the method also works on images that are not high resolution in other geolocation datasets like IM2GPS.

5. JPEG compression is a known workaround for adversarial attacks on vision models. Does your defense hold up to JPEG compression?

6. While the current approach works well for MLRMs, it would be interesting to see if it can also work for specifically designed image geolocation models like [2].

[1] Mendes, Ethan, Yang Chen, James Hays, Sauvik Das, Wei Xu, and Alan Ritter. "Granular privacy control for geolocation with vision language models." arXiv preprint arXiv:2407.04952 (2024).

[2] Haas, Lukas, Michal Skreta, Silas Alberti, and Chelsea Finn. "Pigeon: Predicting image geolocations." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12893-12902. 2024.

### Questions
1. If you do not use the ensemble training, how diminished are the results on various MLRMs? 

2. Is there a correlation between protection with ReasonBreak and how famous or well-known an input image is? For instance, can you successfully protect the location of a famous landmark like the Eiffel Tower?

3. Prior work found that universally transferable image jailbreaks are hard to find [1]. How do you think your work fits in with these claims?


[1] Schaeffer, Rylan, Dan Valentine, Luke Bailey, James Chua, Cristobal Eyzaguirre, Zane Durante, Joe Benton et al. "Failures to find transferable image jailbreaks between vision-language models." arXiv preprint arXiv:2407.15211 (2024).

### Soundness
2

### Presentation
3

### Contribution
3
