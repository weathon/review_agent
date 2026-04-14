# GROD: Enhancing Generalization of Transformer with Out-of-Distribution Detection

- Decision: Reject
- Scores: 5, 6, 6, 5, 3

## Abstract
Transformer networks excel in natural language processing (NLP) and computer vision (CV) tasks. However, they face challenges in generalizing to Out-of-Distribution (OOD) datasets, that is, data whose distribution differs from that seen during training. The OOD detection aims to distinguish data that deviates from the expected distribution, while maintaining optimal performance on in-distribution (ID) data. This paper introduces a novel approach based on OOD detection, termed the Generate Rounded OOD Data (GROD) algorithm, which significantly bolsters the generalization performance of transformer networks across various tasks. GROD is motivated by our new OOD detection Probably Approximately Correct (PAC) Theory for transformer. The transformer has learnability in terms of OOD detection that is, when the data is sufficient the outlier can be well represented. By penalizing the misclassification of OOD data within the loss function and generating synthetic outliers, GROD guarantees learnability and refines the decision boundaries between inlier and outlier. This strategy demonstrates robust adaptability and general applicability across different data types. Evaluated across diverse OOD detection tasks in NLP and CV, GROD achieves SOTA regardless of data format. The code is available at https://anonymous.4open.science/r/GROD-OOD-Detection-with-transformers-B70F.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper establishes the OOD detection learnability of the transformer model via PAC learning theory. The GROD is proposed to enhance the detection performance of transformer-based models for both CV and NLP tasks, which generate virtual OOD samples for fine-tuning. GROD first identifies the boundary ID samples by PCA and LDA and synthesizes the fake OOD by Gaussian mixtures.

### Strengths
1. This paper establishes the OOD detection learnability of the transformer model.
2. This paper considered both NLP and CV scenarios.

### Weaknesses
1. Why do the maximum and minimum values projected by PCA and LDA are considered boundary points? Further analysis of the intrinsic mechanism of PCA and LDA is needed.
2. As claimed in line 149, LDA is selected to guarantee the robustness of generated OOD, but it is only utilized when the number of ID classes is small as defined in Equation (4). Does this mean that the generated OOD samples are not robust with large-scale ID datasets?
3. The baseline NPOS adopts a similar OOD synthesis pipeline, which first identifies boundary ID samples and then generates OOD samples via Gaussian sampling. The superiority of GROD against NPOS should be explicitly stated and the generated OOD samples of the two methods can be statistically compared to further distinguish GROD.
4. The notations are confusing, e.g., line 144 indicates the feature space is $\mathbb{R}^{n\times s}$, however, line 168 defines another $n$.
5. The experiments are insufficient to prove that GROD achieves SOTA performance. Since the authors leverage the OpenOOD benchmark, more far-OOD datasets, such as Textures, Places-365, and MNIST, can be tested to validate GROD's performance.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The GROD paper introduces Generate Rounded OoD Data (GROD), designed to improve the generalization of transformer models in Out-of-Distribution (OoD) detection. This method leverages synthetic OOD data generated using PCA and LDA projections to refine decision boundaries during training, aiming to enhance performance on both in-distribution (ID) and OoD data. GROD is supported by a PAC learning framework and validated through experimental results demonstrating its state-of-the-art performance in OoD tasks.

### Strengths
**1. PAC Learning Framework:** The paper is based on theory, and makes sufficient contribution to the theory of OoD and PAC learnability of OoD however only for transformer architectures.

**2. Computational Efficiency:** Despite the complexity of GROD, the overhead is during training and inference cost is relatively inexpensive.

**3. Ablation Studies:** Ablation studies provide insights into key parameters that control the performance on GROD, which could help guide future work in OOD detection using transformers.

### Weaknesses
**1 . Baseline Comparison with Outlier Exposure (OE):** While the authors propose a synthetic OoD generation approach, they do not include a comparison to Outlier Exposure (OE) methods. A comparison, especially with traditional OE using Gaussian noise, would be valuable in demonstrating GROD’s necessity and superiority. OE [1] proposed using additional OoD data which is used to train/finetune a model to better OoD detection performance, a similar to the idea presented in this paper.

Citations of Related OE Work: The paper does not cite several relevant studies in the Outlier Exposure space, which is a significant oversight given that GROD’s fundamental methodology aligns closely with existing OE methods that use OoD data during fine-tuning. See citations [1, 2].

**2. Evaluation of Synthetic OoD Data:** While GROD’s OoD data generation is sophisticated, it is unclear if the benefits of PCA and LDA projections over simpler alternatives like Gaussian noise have been adequately evaluated. Including a comparison experiment would strengthen claims about the effectiveness of GROD’s approach in OoD data synthesis.

[1] Hendrycks, Dan, Mantas Mazeika, and Thomas Dietterich. 2019. "Deep Anomaly Detection with Outlier Exposure." In Proceedings of the International Conference on Learning Representations. https://openreview.net/forum?id=HyxCxhRcY7.

[2] Zhu, Jianing, Yu Geng, Jiangchao Yao, Tongliang Liu, Gang Niu, Masashi Sugiyama, and Bo Han. 2023. "Diversified Outlier Exposure for Out-of-Distribution Detection via Informative Extrapolation." In Proceedings of Advances in Neural Information Processing Systems, vol. 36, 22702–22734.

### Questions
1. Comparison with Standard Outlier Exposure (OE) Baseline:
How does GROD’s approach to OoD data generation compare to traditional Outlier Exposure (OE) methods, particularly when using standard Gaussian noise or other simple forms of synthetic OoD data?

   *Suggested experiment*: Implement a baseline comparison between GROD and OE methods (e.g., Gaussian noise or straightforward OE with diverse datasets). This experiment could involve evaluating performance differences in OoD performance and computational efficiency.

2. Impact of GROD’s Data Generation Methodology:
Does GROD’s use of PCA and LDA projections for generating synthetic OoD data significantly outperform simpler methods?

    *Suggested experiment:* Conduct an ablation study comparing GROD’s synthetic OoD data generation method to simpler techniques like Gaussian noise or uniformly random OoD data. Evaluate performance of using the proposed technique but instead of the proposed data generation use Gaussian noise at the input.

If the authors can provide sufficient evidence for the benefits of the proposed OoD data generation method over using simpler techniques and provide result on comparison with OE I am willing to increase my score.

### Soundness
2

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
The paper presents a novel algorithm, GROD, aimed at enhancing OOD detection in transformer networks, which is a timely and innovative addition to current research. By combining PCA and Linear Discriminant Analysis (LDA) projections for OOD data synthesis, it proposes an original approach to address limitations in existing OOD detection methods.

### Strengths
- The theoretical foundation provided, especially the PAC learning framework for OOD detection, is a noteworthy contribution that bridges theoretical gaps in understanding transformers learnability for OOD tasks.
- The paper provides thorough experimental validation across multiple OOD detection tasks for both NLP and CV, showing GROD’s adaptability to various data formats, which is admirable.
- Key terms and concepts, such as the PAC learning framework and the GROD algorithm, are introduced clearly, though some technical sections might benefit from additional explanation for accessibility.
- The theoretical insights, especially the derived conditions and error bounds for learnability in transformers, could pave the way for future advancements in OOD detection frameworks for transformers, making it a valuable reference for ongoing research.

### Weaknesses
+ The GROD algorithm involves several hyperparameters (e.g., the scaling parameter for Mahalanobis distance, LDA cluster dimensions) that require fine-tuning. 
+ While GROD achieves a balance between computational efficiency and performance, its iterative processes, including OOD data synthesis and Mahalanobis distance calculation, may not scale well with significantly larger datasets (e.g., ImageNet) or models. This limitation could restrict its deployment in real-time applications where processing speed is crucial.
+ There are missing citations in the manuscript. For example, the paper introduces generative models, but generative-based methods [1, 2, 3] are missed without corresponding details in the bibliography.   

[1]: Kong, Shu, and Deva Ramanan. "Opengan: Open-set recognition via open data generation." In ICCV. 2021.

[2]: Wang, Qizhou, et al. "Out-of-distribution detection with implicit outlier transformation." In ICLR, 2023.

[3]: Zheng, Haotian, et al. "Out-of-distribution Detection Learning with Unreliable Out-of-distribution Sources." In NeurIPS, 2023.

### Questions
+ "Learnability" is repeatedly used without a concise definition in layman’s terms, which could be clarified for a broader audience.
+ How do the authors interpret the performance of LDA-based inter-class OOD generation in enriching OOD representation? More specifically, what are the primary limitations observed when using PCA-only projections, and how might these affect model robustness?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a PAC learning framework for OOD detection of transformer networks. And it also propose a novel approach GROD to improve the OOD detection performance,  including a loss function that penalizes the misclassification of OOD data, and a method for generating synthetic outliers. The GROD algorithm is evaluated across various OOD detection tasks in NLP and CV, demonstrating state-of-the-art performance regardless of data format.

### Strengths
1. The paper establishes a PAC learning framework for OOD detection applied to transformers, providing necessary and sufficient conditions for learnability and error boundary estimates. The approach of generating synthetic outliers using PCA and LDA projections is innovative and contributes to the robustness of the model.

2. GROD enhances the generalization capabilities of transformers, leading to improved performance on both ID and OOD data across different tasks and data types. The algorithm achieves SOTA results in OOD detection for both NLP and CV tasks, outperforming other prevalent methods.

3. The paper includes extensive experiments and ablation studies that validate the effectiveness of GROD and provide insights into hyperparameter tuning.

### Weaknesses
1. The pre-training of GROD is conducted on the ImageNet-1K dataset, whereas OOD detection is evaluated using the CIFAR dataset. Some categories overlap, such as dogs and cats, which seems unreasonable.
2. In line 147, "Feat() represents extracting CLS tokens," which implies that the GROD algorithm utilizes the CLS token for feature extraction. While it is true that many transformer-based models do not necessarily require a CLS token, reducing the generality of the algorithm.
3. How is the scalability of GROD algorithm? If it work well on other transformer-based pretrained backbone?
4. Can the GROD algorithm be adapted to other types of deep learning architectures beyond transformers (e.g. , ResNet)? It seems to be only related to the input feature.

### Questions
See above weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper introduces GROD, an approach to enhance transformers' OOD detection performance by incorporating synthesized OOD data. GROD leverages a Probably Approximately Correct (PAC) theory framework, proposing a learnable criterion for transformers that improves their ability to recognize OOD instances. By integrating OOD misclassification penalties into the loss function and generating synthetic outliers through PCA and LDA projections, GROD establishes a more robust boundary between in-distribution and OOD data.

### Strengths
* The experimental results cover multiple modalities, including both text and image data.

* The study provides both theoretical analysis and experimental validation to support the proposed pipeline.

### Weaknesses
* (A) The authors should explore additional architectures, including MLP-based and CNN-based models, and explain how their method would apply to these. While the study clarifies that it focuses on transformers, it should explicitly address the pipeline’s compatibility with different architectures and provide a discussion on potential adaptations.

* (B) The study includes only a limited number of transformer-based architectures, specifically ViT-B16 and BERT.

* (C) The datasets used in this study are relatively small (e.g., CIFAR vs. SVHN). Larger and higher-resolution benchmarks (e.g., ImageNet, Texture) should be considered to show the contribution.

* (D) Several studies have incorporated synthetic sampling strategies for OOD detection [1,2,3,4,5,6], but there is a lack of comparison with these methods.

* (E) The primary idea of the pipeline shows similarities with [7].


* (F) Using a large pretrained model, such as ViT-B16, for the relatively small Tiny ImageNet dataset raises the issue that the pipeline may rely on extra information seen by the backbone during pretraining rather than on the proposed pipeline itself.


[1] Lee et al., "Training Confidence-Calibrated Classifiers for Detecting Out-of-Distribution Samples," ICLR 2018.

[2] Kirchheim et al., "On Outlier Exposure with Generative Models," NeurIPS ML Safety Workshop, 2022.

[3] Du et al., "VOS: Learning What You Don’t Know by Virtual Outlier Synthesis," ICLR 2022.

[4] Tao et al., "Non-Parametric Outlier Synthesis," ICLR 2023.

[5] Du et al., "Dream the Impossible: Outlier Imagination with Diffusion Models," NeurIPS 2023.

[6] Chen et al., "ATOM: Robustifying Out-of-Distribution Detection Using Outlier Mining."

[7] "Fake It Till You Make It: Towards Accurate Near-Distribution Novelty Detection."

[8] Deep Hybrid Models for Out-of-Distribution Detection

### Questions
* The main motivation for this study is unclear, given that existing methods already achieve strong results on the OOD detection benchmarks considered. For example, [8] achieves competitive performance on CIFAR10 vs. CIFAR100 without using additional information (i.e., without pre-trained models). Thus, the necessity of the proposed pipeline remains uncertain.

* The authors should clearly outline their contributions over similar works [1-7], detailing the limitations of previous approaches and supporting these claims with comprehensive experiments.

* The authors are encouraged to explore a broader range of architectures and models rather than focusing solely on ViT-B16.


* Tiny ImageNet has overlap with both CIFAR-10 and CIFAR-100. How do the authors justify considering these datasets as ID and OOD?

### Soundness
1

### Presentation
2

### Contribution
1
