# ARFlow: Auto-regressive Optical Flow Estimation for Arbitrary-Length Videos via Progressive Next-Frame Forecasting

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Optical flow estimation is a fundamental computer vision task that predicts per-pixel displacements from consecutive images. Recent works attempt to exploit temporal cues to improve the estimation performance. However, their temporal modeling is restricted to short video sequences due to the unaffordable computational burden, thereby suffering from restricted temporal receptive fields. Moreover, their group-wise paradigm in one forward pass undermines inter-group information exchange, leading to modest performance improvement. To address these problems, we propose a novel multi-frame optical flow network based on an auto-regressive paradigm, named ARFlow. Unlike previous multi-frame methods, our method can be scalable to arbitrary-length videos with marginal computational overhead. Specifically, we design an Auto-regressive Flow Initialization (AFI) module and an Auto-regressive Multi-stride Flow Refinement (AMFR) module to forecast the next-frame flow based on multi-stride history observations. Our ARFlow achieves state-of-the-art performance, ranking 1st on both KITTI-2015 and Spring official benchmarks and 2nd on the MPI-Sintel (Final) benchmark among all open-sourced methods. Furthermore, due to the auto-regressive nature, our method can generalize to arbitrary video length with a constant GPU memory usage of 2.1GB.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ARFlow, a novel framework for multi-frame optical flow estimation, which leverages an auto-regressive paradigm to predict optical flow in arbitrary-length videos with marginal computational overhead. ARFlow adopts a progressive next-frame forecasting strategy, with two key components: the Auto-regressive Flow Initialization (AFI) module, which predicts the next-frame initial flow using multiple history flow estimates from the memory bank, and the Auto-regressive Multi-stride Flow Refinement (AMFR) module, which iteratively refines the flow using multi-stride temporal information. The experiment results show that ARFlow achieves state-of-the-art performance on benchmarks like KITTI-2015, Sintel, and Spring and handles long sequences efficiently with consistent memory usage.

### Strengths
- The paper proposes using auto-regressive prediction for optical flow estimation, which is novel. By progressively forecasting the next-frame flow based on historical estimates, ARFlow can handle long sequences more efficiently, significantly improving the scalability of optical flow estimation.
- The paper presents good benchmark results, with ARFlow achieving state-of-the-art performance on popular datasets like KITTI-2015, Sintel, and Spring.
- The paper promises to open-source the code after publication, which is a positive development for the open-source community. ARFlow can also serve as a baseline method for subsequent improvements and comparisons.

### Weaknesses
- In Line 047, the paper points out that existing multi-frame methods fail to exploit sufficient temporal cues because of Limited temporal receptive fields. However, the temporal intervals used in this paper are similar to those of mainstream methods in both the AFI (T=6) and AMFR (max stride=4) modules. Furthermore, I would question the validity of long-distance information for optical flow estimation tasks. The experiments in this paper also indicate that the benefits diminish over excessively long distances.
- Although the introduction of auto-regressive estimation has reduced the computational overhead of ARFlow, it still falls short of real-time applications.

### Questions
- The paper treats optical flow estimation as a streaming problem, which aligns with the inherent properties of the auto-regressive paradigm. However, in most cases, a complete video sequence is already available, so it's natural to anticipate information from later frames. Actually, many existing methods do this. I'm curious why the paper uses unidirectional auto-regression, sacrificing bidirectional information that may further improve accuracy.
- How does ARFlow handle large motions and sudden occlusions? The auto-regressive paradigm relies on historical information, but how does ARFlow handle corner cases when historical information becomes invalid?

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
5

### Summary
This work proposes a novel multi-frame optical flow estimation network in a so-called "Auto-Regressive" manner, which features flexible and efficient processing of an arbitrary number of input frames and achieves certain accuracy improvements compared to previous methods. This Auto-Regressive approach is not reflected in the internal architecture of the network but rather in the overall pipeline. Nevertheless, it represents a relatively innovative approach in the methodology of multi-frame optical flow estimation.

### Strengths
1. This work proposes a novel approach for multi-frame optical flow processing.  
2. The method can handle videos of arbitrary length with significantly reduced memory usage and optimized speed.  
3. While improving efficiency, the work enhances model performance through effective temporal modeling techniques.

### Weaknesses
1. The term "Auto-Regressive" is somewhat nuanced here. Unlike traditional methods like LLMs that perform next-token prediction in discrete spaces, this approach implements autoregression through modifications to the pipeline. Therefore, comparing it with LLMs in related work is less meaningful. It would be more relevant to discuss autoregressive video generation works, as they also incorporate autoregressive changes in their pipelines.

2. Table 1: The font size is too small.

3. Table 2: Should include more Spring metrics. In the Non-finetune section, additional methods like SEA-RAFT could be added.

4. Recommend including more metric evaluations on the Sintel test set.

5. Are there other cases for Figure 6? The hair strand difference in the bottom-left corner is not clearly visible.

### Questions
Please refer to the weaknesses. If necessary, I will update the score based on the response and further discussions.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents ARFlow, a novel approach to multi-frame optical flow estimation. The proposed method includes two key components: the Auto-regressive Flow Initialization (AFI) and the Auto-regressive Multi-stride Flow Refinement (AMFR), both of which significantly enhance the flow prediction accuracy and computational efficiency. Extensive experiments on various benchmark datasets (KITTI-2015, MPI-Sintel, Spring) demonstrate that ARFlow achieves state-of-the-art performance while maintaining a constant memory usage of 2.1 GB, making it highly efficient.

### Strengths
Innovative Approach:The paper introduces a unique solution to multi-frame optical flow estimation by utilizing an auto-regressive model. 
State-of-the-Art Performance: ARFlow achieves top-tier results on standard optical flow benchmarks, including KITTI-2015, MPI-Sintel, and Spring.

### Weaknesses
1.  The paper does not provide the detailed architecture of its Temporal Transformer. How this transformer process the flow input? Does it employ spatial-temporal attention directly?
2. What’s the inference time per-frame and # model parameter when compared to previous method such as MemFlow?
3. I am also curious about the accuracy of the predicted flow with Temporal Transformer (stride 1). How it performs under the flow prediction setting of MemFlow-P?
4. For the title, I find the phrase of ‘Arbitrary-length videos’ seems to be some wise over-claimed or confused. As previous methods like

### Questions
MemFlow typically maintains a memory length of 3 and can also be applied for Arbitrary-length videos. And I cannot tell any other difference with previous methods in terms of videos length from the paper. As for figure 2, I think there may be some minor implementations problems for MemFlow and leads to memory usage increased (, better engineering can solve this problem). 
Overall, this is a nice work with solid experiments, I would like to give a rating of accept.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
It presents ARFlow, an autoregressive multi-frame optical flow estimation method designed to address key limitations of existing approaches—limited temporal receptive fields, high computational/memory overhead, and insufficient multi-stride temporal modeling. By introducing an Auto-regressive Flow Initialization (AFI) module and an Auto-regressive Multi-stride Flow Refinement (AMFR) module, it achieves good performance and efficiency.

### Strengths
1. Innovative Autoregressive Paradigm: The autoregressive design naturally captures long-range temporal dependencies and supports arbitrary video lengths, which is a good advantage for real-world applications.
2. Efficient Module Design: The AFI module and AMFR effectively modeling both short-term subtle motions and long-term occlusions and achieves good efficiency. 
3. Exceptional Performance and Efficiency Balance: According to the paper, ARFlow delivers leading results across three standard benchmarks, outperforming recent strong baselines (e.g., MEMFOF, StreamFlow) while maintaining constant memory usage.

### Weaknesses
1. The fonts in Figure 4 and Figure 5 are not uniform.

2. The speed of StreamFlow seems to be inaccurate. If the SIM pipe is not enabled, it will cause many redundant frame calculations and a lot of time waste, please the author check this point again.

3. Considering that this is a brand new structure, it is recommended that the author include more detailed base network structures in the final version and open source them. I am also wondering whether the author has verified that if replacing some pre-trained Transformer methods will bring improvements?

4. More indicators and methods on Spring can be reported.

### Questions
Please refer to the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
