# Instance Data Condensation for Image Super Resolution

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Deep learning based image Super-Resolution (ISR) relies on large training datasets to optimize model generalization; this requires substantial computational and storage resources during training. While dataset condensation has shown potential in improving data efficiency and privacy for high-level computer vision tasks, it has not yet been fully exploited for ISR. In this paper, we propose a novel Instance Data Condensation (IDC) framework specifically for ISR, which achieves instance-level data condensation through Random Local Fourier Feature Extraction and Multi-level Feature Distribution Matching. This aims to optimize feature distributions at both global and local levels and obtain high-quality synthesized training content with fine detail. This framework has been utilized to condense the most commonly used training dataset for ISR, DIV2K, with a 10\% condensation rate. The resulting synthetic dataset offers comparable or (in certain cases) even superior performance compared to the original full dataset and excellent training stability when used to train various popular ISR models. To the best of our knowledge, this is the first time that a condensed/synthetic dataset (with a 10\% data volume) has demonstrated such performance. The associated code and synthetic dataset are available here.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Instance Data Condensation (IDC), a novel framework tailored for image super-resolution (ISR). The method introduces two main components: Random Local Fourier Feature Extraction, which preserves high-frequency local details crucial for ISR, and Multi-level Feature Distribution Matching, which aligns feature distributions at both instance and group levels to maintain diversity and fidelity in the synthesized data. Experiments conducted on the DIV2K dataset with a 10% condensation rate show that the synthetic dataset achieves comparable or even superior performance to the full dataset when training state-of-the-art ISR models such as EDSR, SwinIR, and MambaIRv2. The work represents the first successful application of dataset condensation to low-level vision, demonstrating efficient data compression without loss of model quality or training stability.

### Strengths
This paper proposes a new data condensation framework, Instance Data Condensation, to the best of my knowledge, is the first to apply the concept of dataset condensation to image super-resolution (ISR). While numerous condensation methods have been explored in high-level vision tasks such as classification, detection, and segmentation, similar attempts have not been made for low-level vision problems. Therefore, the proposed method shows a degree of novelty and exploratory value in extending data condensation to the SR domain.

### Weaknesses
1. The paper’s presentation is quite poor, making it difficult to follow the main ideas. The Related Work and Methodology sections are intermixed, with prior studies and the proposed approach discussed together without clear separation. This confuses the reader and obscures the novelty of the work.
2. Beyond presentation issues, the paper also suffers from conceptual ambiguity in several analyses. For example, in the discussion of Figure 1 (left), the authors claim that the DCSR method suffers from a bias toward complex textures, yet the selected “snow mountain” region actually corresponds to a structured area rather than a purely textured one. Moreover, the figure does not convincingly show that the proposed IDC method avoids such bias. Similarly, the claim that transforming features into the Fourier domain leads “to a more informative representation” (line 259) is unsubstantiated and conceptually weak—Fourier transformation changes the representation domain but does not inherently increase information content. These unclear or overstated interpretations undermine the analytical rigor of the paper and should be supported by clearer quantitative evidence or theoretical reasoning.
3. Another concern arises from the ablation study (Table 2). The results for variants V5–V7 show larger performance drops compared with V4, even though each variant removes different components of the proposed framework (e.g., Unfolding, Local Feature, or Instance/Group Losses). This trend appears inconsistent with the claim that these components are beneficial, since removing them does not lead to clearly distinguishable or interpretable degradations.

### Questions
The paper claims that using a 10% condensed dataset significantly improves training efficiency. However, no quantitative evidence is provided. Could the authors clarify how the training time, number of iterations, and computational cost compare between training on the condensed dataset and the full (“Whole”) dataset?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel framework termed Instance Data Condensation (IDC) for image super-resolution. IDC addresses the challenge of reducing training data volume while maintaining or even enhancing model performance. The framework leverages Random Local Fourier Features (RLFF) and Multilevel Feature Distribution Matching to condense training datasets at the instance level, eliminating the need for class labels common in high-level vision tasks. Extensive experiments and ablation studies validate the effectiveness and robustness of the proposed method.

### Strengths
1. This paper is well-motivated and easy to follow.
2. The proposed framework achieves better performance with only 10% synthetic crops.

### Weaknesses
1. The condensation process is computationally intensive.
2. Although the instance-level paradigm is promising, its effectiveness across diverse tasks remains to be validated.
3. The scalability of the IDC framework across datasets of different volumes lacks empirical validation.

### Questions
1. Does the IDC data distillation method affect the generalization performance of super-resolution models? Please provide relevant experimental results to illustrate.
2. What is the memory footprint of RLFF?  
3. When the condensation ratio falls below 10%, is any modal collapse observed? Or is the model overfitting?

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
This paper proposes a new dataset condensation framework for image super-resolution, by designing a Multi-level Feature Distribution Matching approach and Random Local Fourier Features. The conducted experiments show that the condensed datasets give promising performance.

### Strengths
1. The motivation is clear. This paper shows a new dataset condensation method for image super-resolution.
2. The writing of this paper is fluent, and the content is easy to follow.
3. The designs of the two approaches make sense to some degree.
4. The conducted ablation experiments are detailed and well-designed.

### Weaknesses
1. More large-scale datasets should be condensed to show the promising performance of the proposed method. The related datasets in the paper are DIV2K and Flickr2K, which are not very large in real scenarios. I believe that the value of condensation is more evident on large-scale datasets than in experiments with specific case studies.
2. The performance improvement is not very obvious, as shown in Table 1, and the condensation burden comparison should be provided to give more analysis.
3. Can you give some theoretical analysis or insights about your designs, such as random local Fourier features?
4. Can this method extend to other low-level missions, such as deblur and denoise? Please give some discussions.

### Questions
Please refer to "Weaknesses".

### Soundness
3

### Presentation
3

### Contribution
2
