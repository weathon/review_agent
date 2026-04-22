# Looking Alike From Far to Near: Enhancing Cross-Resolution Re-Identification via Feature Vector Panning

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 6, 2

## Abstract
In surveillance scenarios, varying camera distances cause significant differences among pedestrian image resolutions, making it hard to match low-resolution (LR) images with high-resolution (HR) counterparts, limiting the performance of Re-Identification (ReID) tasks. Most existing Cross-Resolution ReID (CR-ReID) methods rely on super-resolution (SR) or joint learning for feature compensation, which increases training and inference complexity and has reached a performance bottleneck in recent studies. Inspired by semantic directions in the word embedding space, we empirically discover that semantic directions implying resolution differences also emerge in the feature space of ReID, and we substantiate this finding from a statistical perspective using Canonical Correlation Analysis and Pearson Correlation Analysis. Based on this interesting finding, we propose a lightweight and effective Vector Panning Feature Alignment (VPFA) framework, which conducts CR-ReID from a novel perspective of modeling the resolution-specific feature discrepancy. Extensive experimental results on multiple CR-ReID benchmarks show that our method significantly outperforms previous state-of-the-art baseline models while obtaining higher efficiency, demonstrating the effectiveness and superiority of our model based on the new finding in this paper.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the cross-resolution person re-identification (CR-ReID) problem, where image resolution discrepancies caused by varying camera distances significantly degrade ReID performance. The authors empirically discover that feature representations in CR-ReID exhibit a consistent semantic direction corresponding to resolution differences, analogous to semantic offsets in word embedding spaces. Based on this observation, they propose a lightweight Vector Panning Feature Alignment (VPFA) framework that models and compensates for resolution-induced discrepancies at the feature level, without retraining or modifying the backbone network. The proposed method introduces a Vector Panning (VP) module—a residual MLP transformation—and a Vector Panning Loss (VPL) to align both direction and magnitude between low- and high-resolution features. Experiments on multiple CR-ReID benchmarks demonstrate significant improvements over prior state-of-the-art methods, with superior efficiency and generalization ability.

### Strengths
1. The discovery of a semantic direction of resolution in ReID feature space is an insightful and novel perspective on the CR-ReID problem. 
2. The proposed VPFA framework is technically sound and well-motivated.
3. The paper is clearly written and well-structured, with intuitive figures and detailed algorithmic descriptions. The step-by-step exposition—from observation to modeling, training, and analysis—is coherent and easy to follow.
4. The proposed feature-level postprocessing approach is both lightweight and generalizable. It achieves substantial performance gains across multiple datasets while requiring minimal computational overhead.

### Weaknesses
1. While the CCA and Pearson analyses provide empirical justification, the theoretical foundation for the “semantic direction of resolution” remains somewhat shallow. A more formal connection between this direction and the learned embedding manifold could strengthen the paper’s conceptual rigor.
2. Since VPFA relies on fixed pretrained ReID backbones, its performance may vary significantly with different feature extractors. The paper would benefit from a deeper analysis of how backbone quality and feature dimensionality affect the learned vector transformations.
3. Although the paper compares with super-resolution and resolution-invariant methods, it omits comparison with more general feature adaptation or residual tuning approaches (e.g., MLP-based alignment in other domains). This could help situate VPFA more precisely within the broader literature on post-hoc feature correction.

### Questions
1. Is there any visualization or analysis showing what semantic components (e.g., texture, shape, scale) the VP vectors actually modify? 
2. The initialization strategy (near-identity mapping) is mentioned as critical. How sensitive is the training to this setup? Does instability occur when training from random initialization or with higher learning rates?

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
2

### Summary
This paper proposes a novel cross-resolution re-identification framework to address the challenge of significant resolution discrepancies among pedestrian images. Interestingly, the authors discover and verify a semantic direction between high- and low-resolution features in the embedding space, analogous to the well-known relational properties observed in word embeddings. Based on this finding, the resolution difference is modeled through a lightweight Feature Vector Panning (FVP) mechanism, enabling effective feature alignment across different resolutions. Extensive experiments demonstrate the effectiveness and the generalization of the proposed model. More detailed comments are provided as follows.

### Strengths
- The paper is well-written and easy to follow.
- The idea of drawing an analogy between the resolution differences in ReID features and the semantic differences in word embeddings is novel and interesting. Through comprehensive Canonical Correlation Analysis and Pearson Correlation Analysis, the authors convincingly demonstrate that the resolution variations in pedestrian ReID features exhibit a consistent semantic direction. 
- The proposed Feature Vector Panning (FVP) module provides a simple yet effective mechanism for modeling feature resolution variations, and can be implemented as a lightweight post-processing module with high efficiency and scalability.
- Extensive experimental results demonstrate the effectiveness and superiority of the proposed model, as evidenced by its higher accuracy and efficiency compared with several state-of-the-art baselines.

### Weaknesses
- Improve cross-modality reproducibility: Suggesting describe how cross-modal training pairs are constructed in RegDB and RSTPReid, whether separate VPs are trained per retrieval direction (visible-to-infrared and infrared-to-visible), feature dimensions/normalization, and whether the mapping is trained per dataset or jointly.
- Consider cross-dataset generalization: Train VP on one dataset (e.g., Market-1501) and test on another (e.g., CUHK03), to evaluate robustness of the proposed VPFA.
- The t-SNE visualizations presented in the paper are not sufficiently intuitive. Although the effectiveness can be inferred from the feature distance tables in the supplementary materials, more direct and informative t-SNE plots in the main text would better illustrate the results.

### Questions
- Improve cross-modality reproducibility: Suggesting describe how cross-modal training pairs are constructed in RegDB and RSTPReid, whether separate VPs are trained per retrieval direction (visible-to-infrared and infrared-to-visible), feature dimensions/normalization, and whether the mapping is trained per dataset or jointly.
- Consider cross-dataset generalization: Train VP on one dataset (e.g., Market-1501) and test on another (e.g., CUHK03), to evaluate robustness of the proposed VPFA.
- The t-SNE visualizations presented in the paper are not sufficiently intuitive. Although the effectiveness can be inferred from the feature distance tables in the supplementary materials, more direct and informative t-SNE plots in the main text would better illustrate the results.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper tackles the performance bottleneck of Cross-Resolution Re-Identification (CR-ReID) caused by camera distance-induced resolution gaps, authors propose Vector Panning–based Feature Alignment (VPFA), a post-processing method for cross-resolution person ReID, which works purely at the feature level and integrates into existing ReID models without retraining. Experiments on four benchmarks demonstrate the effectiveness of VPFA.

### Strengths
1. Authors offer a new way to address CR-ReID beyond traditional SR or resolution-invariant methods.
2. VPFA keeps lightweight efficiency, which works as a post-processing module, requiring no ReID backbone modification/retraining, with only 24.14M parameters and high inference speed (4.4M samples/sec).
3. experiments outperform SOTA models on 4 CR-ReID benchmarks and extend effectively to cross-modality tasks.
4. More comprehensive experiments, such as statistical tests, ablation studies, cross-dataset transfer, are shown.

### Weaknesses
1. Regarding the novelty, the novelty of the paper is limited. The lightweight module designed by the authors is nothing more than traditional backpropagation in the latent space, which seems to have no essential difference from existing visual network-based methods Reviewers are difficult to extract the useful contributions.
2. The authors only show the standard ReID metrics and did not report its confidence interval and indicators for "resolution robustness". It is recommended that the authors report the experiment under 2x, 3x, and 4x downsampling conditions separately.
3. The paper only verifies the existence of this direction through statistical correlation analyses (CCA and Pearson correlation) (Section 3.2) and fails to provide a fundamental theoretical explanation for its rationality and inevitability.
4. The ablation studies only cover "residual connections, hidden layer dimensions, and the number of VP blocks" (Table 4) and fail to verify the necessity of VPFA’s core components such as VPL loss, LayerNorm, Tanh Gate.
5. The author emphasizes the lightweight nature of the VPFA framework, but the experiment did not include a comparison of computational complexity and parameter analysis with existing methods. It is suggested that the author provide comparative results of FLOPs and Parameters under various methods.

### Questions
1. Regarding the novelty, the novelty of the paper is limited. The lightweight module designed by the authors is nothing more than traditional backpropagation in the latent space, which seems to have no essential difference from existing visual network-based methods Reviewers are difficult to extract the useful contributions.
2. The authors only show the standard ReID metrics and did not report its confidence interval and indicators for "resolution robustness". It is recommended that the authors report the experiment under 2x, 3x, and 4x downsampling conditions separately.
3. The paper only verifies the existence of this direction through statistical correlation analyses (CCA and Pearson correlation) (Section 3.2) and fails to provide a fundamental theoretical explanation for its rationality and inevitability.
4. The ablation studies only cover "residual connections, hidden layer dimensions, and the number of VP blocks" (Table 4) and fail to verify the necessity of VPFA’s core components such as VPL loss, LayerNorm, Tanh Gate.
5. The author emphasizes the lightweight nature of the VPFA framework, but the experiment did not include a comparison of computational complexity and parameter analysis with existing methods. It is suggested that the author provide comparative results of FLOPs and Parameters under various methods.

### Soundness
2

### Presentation
2

### Contribution
2
