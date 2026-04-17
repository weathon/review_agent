# Dynamic Reflections: Probing Video Representations with Text Alignment

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
The alignment of representations from different modalities has recently been shown to provide insights on the structural similarities and downstream capabilities of different encoders across diverse data types.
While significant progress has been made in aligning images with text, the temporal nature of _video_ data remains largely unexplored in this context. 
In this work, we conduct the first comprehensive study of video-text representation alignment, probing the capabilities of modern video and language encoders. 
Our findings reveal several key insights. 
First, we demonstrate that cross-modal alignment highly depends on the richness of both visual (static images vs. multi-frame videos) and text (single caption vs. a collection) data _provided at test time_, especially when using state-of-the-art video encoders. 
We propose parametric test-time scaling laws that capture this behavior and show remarkable predictive power against empirical observations.
Secondly, we investigate the correlation between semantic alignment and performance on both semantic and non-semantic downstream tasks, providing initial evidence that strong alignment against text encoders may be linked to _general-purpose_ video representation and understanding.
Finally, we correlate temporal reasoning with cross-modal alignment providing a challenging test-bed for vision and language models. 
Overall, our work introduces video-text alignment as an informative zero-shot way to probe the representation power of different encoders for spatio-temporal data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper extends the static cross-modal alignment evaluation scheme to the dynamic temporal domain via the Mutual k-NN metric. It explores the laws of video-text alignment using the VATEX and PVD datasets; validates the correlation between alignment and downstream tasks using Kinetics-400 and SSv2; and verifies the sensitivity of alignment to temporal information using VideoComp. Its core contributions are: 1. Proposes an alignment evaluation scheme tailored for the temporal domain; 2. Confirms that multiple video frames and multiple captions during testing can enhance alignment performance; 3. Proposes the Test-time Scaling Laws formulation, which can guide data collection and model capability comparison; 4. Preliminarily verifies that video-text alignment can serve as a zero-shot metric, enabling replacement of expensive cross-modal decoder-based evaluation without additional training; 5. Verifies the temporal sensitivity of alignment using the Test of Time and VideoComp datasets.

### Strengths
1.For the first time, static alignment evaluation strategies are extended to the dynamic temporal domain, enabling alignment assessment between video and text.
2.It clarifies the law that "multiple frames/multiple captions can improve alignment but exhibit a saturation effect," providing a directly reusable quantitative basis for "data collection scale optimization" and "rapid model selection" in practical applications, thus avoiding redundant resource investment.
3.Covering multiple downstream tasks, the zero-shot evaluation metric features a low-cost advantage: it enables the assessment of model representation quality without the need to train additional expensive cross-modal decoders, significantly simplifying the evaluation process and reducing computational and annotation costs, which adapts to the demand for efficient model screening in production.
4.It empirically verifies temporal sensitivity, pointing out directions for the optimization of video models.

### Weaknesses
1.The core theory relies solely on experimental observations and lacks solid support from theoretical derivation.
2.The Test-time Scaling Laws are essentially derived from data fitting, lack theoretical support, and their generalizability across scenarios remains limited.
3.The downstream tasks covered in the preliminary exploration of the zero-shot metric are incomplete. In practical scenarios, more focus is placed on the actual performance of downstream tasks, and relying solely on relative scores may not meet the practical needs of video model selection.
4.The temporal analysis lacks validation on large datasets, casting doubt on the applicability of its conclusions.

### Questions
Reference Weaknesses.

### Soundness
2

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
This paper presents a large-scale empirical study on the emergent alignment between unimodal video and text representations, extending the Platonic Representation Hypothesis to the temporal domain. The authors' central thesis is that the alignment is not a static property but is highly dependent on the amount of information provided at test-time. This observation is formalized through a proposed "test-time scaling law," a parametric model that achieves a high degree of fit with the empirical data. Finally, the authors show that this alignment score strongly correlates with downstream performance on a variety of video understanding tasks, proposing it as a computationally efficient, zero-shot proxy for evaluating video model capabilities.

### Strengths
1.	Large-Scale Empirical Study: The primary strength of this work lies in its experimental rigor. The authors conduct a comprehensive analysis across a vast suite of 63 vision models and 30 language models on multiple datasets.

2.	Critical Baseline: The paper's systematic use of a powerful image-encoder-plus-frame-averaging baseline is a significant contribution. The fact that this simple baseline outperforms many purpose-built video models is a critical finding for the community.

3.	Novel Phenomenon: The introduction of the test-time scaling law (Eq. 2) is a valuable contribution. The high coefficient of determination (R² > 0.98) indicates its descriptive power, and the interpretability of its parameters provides a principled way to compare how different architectures leverage temporal information.

### Weaknesses
1.	The Scaling Law: I am concerning about the scaling law proposed in the paper. What is the true significance of scaling data to boost alignment scores? Isn't the high alignment achieved this way just an artificial way to minimize error? Fundamentally, isn't this just about providing more information to reduce the randomness of the MkNN metric and make the metric itself more robust? But if a model is incapable of producing a robust, comprehensive, and unambiguous representation from a single piece of data, then it naturally deserves a lower alignment score. Isn't that precisely what's supposed to happen? Why do we need to artificially intervene to change this score at all?

2.	Insufficient Control for Confounding Variables: The paper's core claim is that video-text alignment is a predictive proxy for downstream performance. However, the analysis does not control for obvious confounding variables like model scale, pre-training data volume, and architecture. It is unclear whether alignment is a true predictive cause or simply another correlated effect of a "better model." The study would be significantly stronger if it could demonstrate this correlation holds even when controlling for these factors (e.g., within a single model family of varying sizes).

3.	The statement of "General-Purpose" Applicability: The claim that the alignment metric serves as a probe for "general-purpose video representation and understanding" is not fully substantiated. The reported weak correlation with the point tracking task is a direct counterexample, suggesting the metric's predictive power may be limited to a specific class of tasks. 

4.	Lacking theoretical analysis on Scaling Law: The proposed scaling law is presented as an empirical fit. While the fit is excellent, the paper offers no theoretical justification for its specific mathematical form. Without a basis in information theory, learning theory, or another principled framework, the law remains an observation specific to this experimental setup, and its generalizability to other datasets, modalities, or alignment metrics is not guaranteed.

### Questions
1.	What is the true significance of scaling data to boost alignment scores?

2.	Isn't the high alignment achieved this way just an artificial way to minimize error? Fundamentally, isn't this just about providing more information to reduce the randomness of the MkNN metric and make the metric itself more robust? 

3.	If a model is incapable of producing a robust, comprehensive, and unambiguous representation from a single piece of data, then it naturally deserves a lower alignment score. Isn't that precisely what's supposed to happen? Why do we need to artificially intervene to change this score at all?

4.	Whether alignment is a true predictive cause or simply another correlated effect of a "better model."? 

5.	Lacking theoretical analysis on Scaling Law: The is presented as an empirical fit. While the fit is excellent, Is there any theoretical justification/guarantee for its specific mathematical form of the proposed scaling law. 

Other questions please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates video-text representation alignment, focusing on modern video and language encoders. It demonstrates how cross-modal alignment depends on the richness of visual and textual data provided at test time. The study introduces test-time scaling laws, showing how adjustments to the number of frames and captions can improve alignment scores. The paper further explores the relationship between alignment quality and downstream task performance, including temporal reasoning and general video understanding. The findings suggest that alignment could serve as a valuable zero-shot metric for evaluating video models.

### Strengths
**Comprehensive Approach**: The paper provides the first comprehensive study of video-text representation alignment, extending the Platonic Representation Hypothesis to the temporal domain, making it a significant contribution.

**Correlation with Downstream Tasks**: The correlation between alignment scores and performance on semantic and non-semantic tasks demonstrates the practical value of alignment as a metric.

### Weaknesses
The idea of probing visual representation with video-text alignment is not such convincing. This evaluation is fair for models proposed on cross-modal tasks, but visual ability is not only cross-modal alignment.   
For example, in tasks such as video object detection and video object tracking, the vision model only need to detect pixel-level difference in the picture, without the need to be aware of textual semantics.   
The DINO-series [1], SAM-seris [2], I-JEPA [3] and V-JEPA [4] are some evidence that model can excel in visual tasks without the need of textual semantic. 

[1] Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P. and Joulin, A., 2021. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 9650-9660).    
[2] Kirillov, Alexander, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao et al. "Segment anything." In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4015-4026. 2023.  
[3] Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y. and Ballas, N., 2023. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 15619-15629).   
[4] Assran, Mido, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi et al. "V-jepa 2: Self-supervised video models enable understanding, prediction and planning." arXiv preprint arXiv:2506.09985 (2025).

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work conducts a study of video-text representation alignment, probing the capabilities of modern video and language encoders.The authors propose a systematic probing framework using mutual k-NN alignment, extending previous static image–text  to videos. The study is comprehensive especially in video-text alignment study.

### Strengths
1, The study of the video-text alignment is meaningful and important.
2, Provides comprehensive experiments, which requires a lot of hardwork.
3, the idea of test-time scaling seems sound
4, Obervation provided in L182-186,L484-485 is informative.

### Weaknesses
1, Limited novelty in methodology.
The approach seems  a empirical report, might not  meet ICLR’s innovation threshold.
2, abstract is different from the paper, could be misleading.
3, the improvement over previous image based methods seems limited.

### Questions
1, What's the main difference or challenge in adapting mutual k-NN from  Huh et al. (2024) to the video domain? It seems the novelty is incremental. 
2, L484-485, what could the potential reason? This question may help with future directions

### Soundness
3

### Presentation
3

### Contribution
3
