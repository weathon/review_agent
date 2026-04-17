# WaMo: Wavelet-Enhanced Multi-Frequency Trajectory Analysis for Fine-Grained Text-Motion Retrieval

- Decision: Reject
- Scores: 8, 4, 4, 8

## Abstract
Text-Motion Retrieval (TMR) aims to retrieve 3D motion sequences semantically relevant to text descriptions. However, matching 3D motions with text remains highly challenging, primarily due to the intricate structure of human body and its spatial-temporal dynamics. Existing approaches often overlook these complexities, relying on general encoding methods that fail to distinguish different body parts and their dynamics, limiting precise semantic alignment. To address this, we propose WaMo, a new wavelet-based multi-frequency feature extraction framework. It fully captures joint-specific and time-varying motion details across multiple resolutions on body joints, extracting discriminative motion features to achieve fine-grained alignment with texts. WaMo has three key components: (1) Trajectory Wavelet Decomposition decomposes motion signals into frequency components that preserve both local kinematic details and global motion semantics. (2) Trajectory Wavelet Reconstruction uses learnable inverse wavelet transforms to reconstruct original joint trajectories from extracted features, ensuring the preservation of essential spatial-temporal information. (3) Disordered Motion Sequence Prediction reorders shuffled motion sequences to improve learning of inherent temporal coherence, enhancing motion-text alignment. Extensive experiments demonstrate WaMo's superiority, achieving 17.0\% and 18.2\% improvements in $Rsum$ on HumanML3D and KIT-ML datasets, respectively, outperforming existing state-of-the-art (SOTA) methods. Code will be open-sourced upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a wavelet-based motion representation that incorporates multi-frequency modeling to capture both global structure and local details in human motion. The use of an inverse wavelet reconstruction loss and a temporal reordering classification loss effectively regularizes the representation, encouraging stronger spatiotemporal awareness. Extensive experiments show significant improvements in text-motion retrieval over existing baselines, and comprehensive ablation studies confirm the contribution of each proposed component.

### Strengths
- The paper is clearly written and well-organized. The idea of integrating wavelet decomposition for motion representation is intuitive and well-grounded. The technical design is elegant and has been shown to be lightweight in supplemental experiments.

- The multi-scale modeling effectively captures both long-term structure and fine motion details, addressing a key weakness of prior models.

- The experimental evaluation is very comprehensive, with multiple baselines, ablations, and qualitative visualizations that support the main claims.

### Weaknesses
- The authors note that FFT-based methods struggle to capture local motion details. However, some prior works have explored multi-level or part-level feature modeling within FFT frameworks, demonstrating benefits for motion representation. A deeper discussion of these approaches and, if possible, an ablation comparing against such variants would further strengthen the paper’s analysis. 
    - Starke, Sebastian, et al. "Local motion phases for learning multi-contact character movements." ACM Transactions on Graphics 39.4 (2020).
    - Starke, Sebastian, Ian Mason, and Taku Komura. "Deepphase: Periodic autoencoders for learning motion phase manifolds." ACM Transactions on Graphics (ToG) 41.4 (2022): 1-13.
    - Wan, Weilin, et al. "Diffusionphase: Motion diffusion in frequency domain." arXiv preprint arXiv:2312.04036 (2023).
- Additional visualizations similar to Figure 1 would be helpful for interpreting the effectiveness of the wavelet formulation for modeling fine-grained details. Moreover, videos may be clearer than figures, for example in Figure 1, the two-hand interaction is difficult to see in the human motion image.

### Questions
- What value is used for J? Does it correspond to the total number of joints in the dataset, or is it based on a part-level grouping? Figure 2 appears to suggest ( J = 5 ), where “l-arm,” “r-arm,” etc., denote body parts that each aggregate multiple joints. If so, how is the trajectory of such a grouped body part defined?

- How are the temporal segments divided for DMSP? Are they evenly split or determined by motion characteristics?

- What is the reconstruction quality of TWR across different levels of wavelet decomposition? It would be interesting to quantify the degree of information loss and discuss whether this loss meaningfully affects retrieval performance.

- The t-SNE visualization in Figure 8 (b) seems to be sparser (i.e., fewer dense clusters). Could the authors provide insights into why this occurs? It would also be interesting to visualize how the t-SNE distribution changes across different levels of wavelet decomposition.

### Soundness
4

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
5

### Summary
This paper proposes WaMo, a framework for text-to-motion retrieval that models human motion at multiple temporal scales. The key idea is to decompose joint trajectories into frequency bands using wavelets, reconstruct them, and also predict the correct temporal order of shuffled motion segments. This design aims to capture both slow, global trends and fast, local movements, improving alignment between motion and text descriptions. Experiments on HumanML3D and KIT-ML show improved retrieval performance compared with prior methods.

### Strengths
•	Uses multi-frequency decomposition to better represent different motion dynamics across time scales.

•	The DMSP task encourages the model to understand temporal structure, which may help fine-grained text-motion alignment.

•	Empirical gains on standard datasets demonstrate practical effectiveness.

### Weaknesses
1.	**Motivation needs further clarification.**
The motivation presented in Figure 1 is not very representative. From the figure, the three frequency scales show rather flat trajectories between frames 50–100, followed by sharp changes around 150. However, it remains unclear what these low- and high-frequency components actually correspond to in terms of specific human motions. The paper would benefit from deeper analysis and interpretation of the relationship between motion dynamics and their frequency-domain characteristics.
2.	**Limited novelty in core ideas.**
The concepts of wavelet and frequency decomposition have been explored in several prior works related to human motion (e.g., WaveAR, MotionWavelet, HumanMAC, LearnTrajDep). The authors should better articulate what is unique about applying such ideas to text-to-motion retrieval (TMR). What task-specific challenges or opportunities make the wavelet-based approach particularly suitable here?
3.	**Misleading claim about prior work.**
The paper states that existing methods “indiscriminately process human motion across different parts and moments.” This claim is inaccurate. For example, MotionPatch (CVPR 2024) already incorporates patch-based motion modeling that captures local spatiotemporal structure and part-wise motion patterns. The discussion should acknowledge such advances and clarify how WaMo differs beyond this level.
4.	**Common idea for Disordered Motion Sequence Prediction (DMSP).**
The idea of shuffling and reconstructing temporal order is widely used in other domains such as video and image tasks (e.g., Jigsaw Puzzle). The authors should explain what is special or necessary about applying this strategy to motion data. Without a clearer task-specific motivation, DMSP appears to be a direct adaptation of a generic self-supervised idea.
5.	**Use of MoMask for evaluation.**
MoMask is primarily designed for text-to-motion generation. It is unclear how it is used here to evaluate text-to-motion retrieval performance. More explanation is needed on how the retrieval results are computed or adapted from the generation framework.
6.	**Computational overhead unaddressed.**
Frequency-based modeling introduces additional computation, yet the paper provides no discussion or quantitative benchmark of training/inference speed or memory usage. A comparison with existing lightweight TMR models would strengthen the empirical evaluation.
7.	**Limited dataset diversity.**
All experiments are conducted on HumanML3D and KIT-ML, which are relatively small, clean, and constrained. The generalization ability to larger datasets (e.g., Motion-X) remains unverified. Broader evaluation would be necessary to demonstrate robustness.
8.	**Unexpectedly poor LaMP results.**
The reported LaMP performance is substantially lower than what was shown in the original paper, where it slightly outperformed baseline TMR methods. Please double-check the evaluation protocol or implementation to ensure fairness and reproducibility.
9.	**Strong dependency on wavelet basis choice.**
Table 10 shows that performance is sensitive to the wavelet initialization (e.g., db1 achieves the best result). The paper should discuss why the Haar wavelet (db1) is particularly effective for human motion modeling and whether this suggests dataset-specific bias or inherent alignment between Haar filters and certain motion patterns.

### Questions
See Weaknesses

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
5

### Summary
This paper introduces WaMo, a method for fine-grained text-to-motion retrieval. To better capture the complex temporal dynamics of human motion, WaMo decomposes joint trajectories into multiple frequency bands using wavelets, reconstructs them to preserve spatial-temporal structure, and employs a disordered sequence prediction task to encourage understanding of temporal order. By combining multi-frequency motion analysis with temporal reasoning, the model aims to align motion sequences more accurately with textual descriptions. Experiments on HumanML3D and KIT-ML demonstrate consistent improvements over prior approaches.

### Strengths
•	Introduces wavelet-based multi-frequency analysis for fine-grained motion modeling.

•	Focuses on part-specific and temporal dynamics, improving text-motion alignment.

•	Achieves clear empirical gains on standard benchmarks.

•	Provides a coherent, modular pipeline (TWD, TWR, DMSP) from decomposition to retrieval embedding.

### Weaknesses
1. **Conceptual novelty and task-specific motivation are unclear.**
WaMo combines multi-frequency decomposition and disordered motion sequence prediction (DMSP), but each component has clear precedents. Frequency-based representations have been used in motion-related tasks (e.g., WaveletMotion, WaveAR, HumanMAC), local/part-level modeling is already incorporated in methods like MoPa（MotionPatch，CVPR2024）, and shuffling/reordering sequences resembles common self-supervised tasks in video and image domains. It remains unclear what is genuinely novel for text-to-motion retrieval. The authors should clarify: which aspects of multi-frequency decomposition are retrieval-specific, and whether DMSP includes any motion-specific adaptations.

2.	**Ambiguity in evaluation using MoMask.**
Since MoMask was originally designed for text-to-motion generation, it is unclear how it is adapted for retrieval. More details on the evaluation procedure are needed to ensure fairness and reproducibility.

3.	**Computational overhead not discussed.**
Frequency decomposition adds processing cost, but the paper does not report memory usage, latency, or parameter increase. Including such analysis would help assess scalability and practical feasibility.

4.	**Limited dataset diversity.**
Experiments are confined to HumanML3D and KIT-ML, which are relatively small and clean. Evaluation on larger or more diverse datasets (e.g., Motion-X) would better demonstrate robustness and generalization.

5.	**Unexpectedly low LaMP results.**
Reported LaMP performance is lower than in the original paper, where it slightly outperformed TMR baselines. The evaluation protocol or implementation should be double-checked to ensure comparability.

6. **Sensitivity to wavelet basis choice.** 
Table 10 shows performance is strongly dependent on the wavelet initialization (e.g., db1 performs best). The authors should discuss why Haar (db1) is particularly effective and whether this reflects dataset-specific bias or intrinsic alignment with certain motion patterns.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a new wavelet-based multi-frequency feature extraction framework (thus a representation) that contributes to significantly advancing the SoTA on the Text-Motion Retrieval task.

The three modules proposed (Trajectory Wavelet Decomposition, Reconstruction, and Disordered Motion Sequence Prediction appear to significantly improve the learning of the temporal structures.

Results are impressive. The quality of the presentation is high.

The experimental settings and discussion are trustworthy and complete.

The appendix contains many details and is useful.

The only limitation is that the proposed approach is specific to the given task. There is no discussion regarding other possible applications of the proposed approach to different tasks and domains.

### Strengths
- impressive results
- trustable experimental settings
- comprehensive ablations
- qualitative evidence
- theoretical grounding

### Weaknesses
- the proposed approach is very specific for the given task (3D motion retrieval). This could limit the interest for the overall CLR community
- the contribution is rather architectural than conceptual

### Questions
Could you discuss how you expect the approach to perform on longer or noiser motion sequences (i.e., beyond the datasets tested that are the ones available in the literature)? Just you opinion...

Would it be possible to interpret the learned wavelet filters visually?

### Soundness
4

### Presentation
4

### Contribution
3
