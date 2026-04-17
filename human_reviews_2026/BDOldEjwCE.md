# TrajFlow: Nation-wide Pseudo GPS Trajectory Generation with Flow Matching Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 10

## Abstract
The importance of mobile phone GPS trajectory data is widely recognized across many fields, yet the use of real data is often hindered by privacy concerns, limited accessibility, and high acquisition costs. As a result, generating pseudo–GPS trajectory data has become an active area of research. Recent diffusion-based approaches have achieved strong fidelity but remain limited in spatial scale (small urban areas), transportation-mode diversity, and efficiency (requiring numerous sampling steps). To address these challenges, we introduce TrajFlow, which to the best of our knowledge is the first flow-matching-based generative model for GPS trajectory generation. TrajFlow leverages the flow-matching paradigm to improve robustness and efficiency across multiple geospatial scales, and incorporates a trajectory harmonization \& reconstruction strategy to jointly address scalability, diversity, and efficiency. Using a nationwide mobile phone GPS dataset with millions of trajectories across Japan, we show that TrajFlow or its variants consistently outperform diffusion-based and deep generative baselines at urban, metropolitan, and nationwide levels. As the first nationwide, multi-scale GPS trajectory generation model, TrajFlow demonstrates strong potential to support inter-region urban planning, traffic management, and disaster response, thereby advancing the resilience and intelligence of future mobility systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces TrajFlow, a flow-matching-based generative model for synthesizing pseudo-GPS trajectories at a nationwide scale. Addressing key limitations of diffusion-based methods—namely poor scalability, limited transportation-mode diversity, and high sampling cost—TrajFlow integrates a trajectory harmonization and reconstruction pipeline with conditional flow matching. It is evaluated on a large-scale mobile GPS dataset from Japan and demonstrated superior performance.

### Strengths
S1: It is the first application of flow matching to trajectory generation; novel integration of RDP-based harmonization and OD-conditioned normalization.

S2: Strong empirical performance across multiple scales and modes; comprehensive ablation studies validate design choices.

S3: Well-organized structure; clear problem motivation and contribution statement.

### Weaknesses
W1: The model is trained and evaluated only on Japanese data; it is unclear how well it generalizes to other countries with different urban structures or mobility patterns.

W2: While the paper claims efficiency gains, runtime and memory usage are not compared in detail with baselines (e.g., training time, GPU hours).

W3: Although privacy is discussed, the paper does not explore potential misuse scenarios (e.g., synthetic data used to infer real user behavior or re-identification risks).

W4: The evaluation focuses on Tokyo for mode diversity; nationwide mode-specific performance is not as thoroughly analyzed.

### Questions
Q1: How does TrajFlow perform on non-Japanese datasets, particularly in countries with less structured transportation networks?

Q2: Could the authors provide a more detailed comparison of training/inference time and memory usage versus diffusion-based models?

Q3: What are the potential risks of model misuse, and have any safeguards been considered to prevent re-identification or surveillance applications?

Q4: Why was RDP chosen over other curve simplification methods (e.g., spline fitting) in the final model, given that some alternatives performed similarly in the appendix?

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
5

### Summary
This paper presents a novel high-fidelity GPS trajectory generation model, called TrajFlow, which aims to address key challenges in current research. Although real GPS data is highly valuable, its application is hindered by privacy concerns, high costs, and access restrictions. Existing generation methods based on diffusion models have high fidelity but suffer from three major limitations: they are limited to small urban areas, lack diversity in multi-traffic patterns, and are inefficient in terms of training and inference. The TrajFlow applies the flow-matching paradigm to GPS trajectory generation. The core of the approach is to address the signal-to-noise ratio collapse problem encountered by diffusion models when scaled to large scales. It achieves this goal through a trajectory coordination and reconstruction strategy: the trajectories are first compressed using the RDP algorithm and then normalized to a uniform feature space for training. Experiments on a nationwide GPS dataset covering the entire country of Japan demonstrate that TrajFlow outperforms baselines, such as diffusion models, at city, metropolitan area, and nationwide scales, while maintaining the diversity of traffic patterns and being highly efficient.

### Strengths
1. This paper presents the first application of the flow-matching paradigm to the task of GPS trajectory generation, which addresses the problem that the performance of existing models, especially diffusion models, degrades dramatically when scaling from small urban scales to regional or national scales.
2. TrajFlow is much more efficient than the computationally expensive diffusion model, which requires a large number of sampling steps. TrajFlow achieves high fidelity in generation with only about 10 ODE steps.
3. TrajFlow overcomes the limitations of previous studies that are mainly limited to cab trajectory data, and is able to generate trajectories for multiple modes of transportation, including trains, cars, bicycles, and walking.

### Weaknesses
1. This paper lacks some details about the reproducibility of the models and algorithms, such as model architecture, parameter settings, code, algorithm process, etc.
2. This paper uses trajectory data with multiple travel modes and a national scale. However, it's not publicly available, and the authors do not present many details about the dataset. This limits the reader's ability to review the technical performance of this paper in depth.
3. The main evaluation metrics of the paper are biased towards space rather than time, such as DTW and Fréchet distance. These metrics are well-suited to measure the geometric similarity of two curves, but are limited in their ability to assess the temporal fidelity of trajectories.

### Questions
1. How does this paper deal with the problem of variable-length generation of data?
2. If the RDP algorithm is used to extract keypoints, it would hold the spatial characterization. So what is the difference between a trajectory generated using TrajFlow and one we interpolate directly based on keypoints?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of pseudo-GPS trajectory generation, which is challenged by issues of spatial scalability, multi-modal transportation diversity, and generation efficiency. To tackle these limitations, the authors propose TrajFlow, a novel flow-matching-based generative framework that incorporates trajectory harmonization and reconstruction within a conditional generative paradigm. Experimental results demonstrate the effectiveness of the proposed method.

### Strengths
S1. This paper studies the problem of pseudo GPS trajectory generation, which seems interesting. 

S2. The paper presents the first flow-matching-based generative framework.

S3. Experiments show that the proposed TrajFlow outperforms the existing baselines.

### Weaknesses
W1. Novelty: The paper would benefit from a deeper discussion clarifying the differences between the proposed approach and more recent baselines.

W2. Datasets: Experiments are conducted on only one dataset, which limits the generalizability of the conclusions. It is recommended to include additional commonly used datasets such as Chengdu and Xi’an to strengthen the empirical validation.

W3.Baseline: The baselines used for comparison (from 2020, 2021, and 2023) are relatively outdated. The paper should include more recent baselines mentioned in the related work section and existing work such as Diffusion-TS to ensure a fair and comprehensive comparison.
[1] Interpretable Diffusion for General Time Series Generation, ICLR 2024.

W4. Lack of Complexity Analysis: The paper does not provide a theoretical analysis of time and space complexity. Including such analysis would help readers better understand the computational efficiency and scalability of the proposed framework.

W5. Reproducibility: The paper lacks the codes, which may hinder reproducibility.

### Questions
Q1: Missing Figure and Table References: Some figures (e.g., Figures 1 and 3) and tables (e.g., Table 1) are not referenced, which affects readability. 

Q2: ODE is missing citation.

Q3: The methodology section is somewhat difficult to follow. For instance, it is unclear where is  Figure 3 referred and how it aligns with the overall model design. It is better to give more discussions.

Q4: Why are these evaluation metrics chosen? What is the rationale for using P10/P90 to describe central accuracy and dispersion?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
The paper proposes a new generative model for GPS trajectories of human mobility. The architecture is based on flow-matching. The GPS trajectories are first normalized and then simplified to both help with efficiency and training stability. The model is shown to outperform state-of-the-art models across both trajectory-level and aggregate-level evaluation measures.

### Strengths
1. Very strong performance for trajectory generation at nation-level scale
2. The ablation study not only shows the importance of each part, but also discusses some of the inherent limitations of using a global coordinate frame when generating trajectories, which is that it introduces the risk of small details being lost when different trajectories have different scales.
3. Provides new insights into what is important for generating trajectories: trajectory simplification, normalization, and flow matching are all critical components.

### Weaknesses
1. All results are based on a single dataset, which is not publicly accessible.
2. Auxiliary data is required to sample from the model: departure times, OD pairs, and transportation modes. 
3. Limited discussion around the risk of memorization. The DTW measure shows the average DTW distance to the closest real trajectory. While achieving a low score might seem positive, it could also indicate that the model has learned to copy the training set, potentially increasing the risk of leaking private information. This can also lead to inflated evaluation scores, as copying training data can improve evaluation metrics without the model genuinely learning to generate novel trajectories.

### Questions
1. How are departure times, OD pairs, and transportation modes obtained in practice?
2. Could you elaborate on why TrajFlow-w/o RDB & OD shows strong performance on the Central Tokyo region?
3. Is there a risk that the model has memorized the training set? The reported DTW is the average across multiple generated trajectories, but what is the smallest value you have observed for individual trajectories?

### Soundness
4

### Presentation
4

### Contribution
4
