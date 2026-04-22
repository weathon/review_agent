# From Sparse to Dense: Spatio-Temporal Fusion for Multi-View 3D Human Pose Estimation with DenseWarper

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
In multi-view 3D human pose estimation, models typically rely on images captured simultaneously from different camera views to predict a pose at a specific moment. While providing accurate spatial information, this traditional approach often overlooks the rich temporal dependencies between adjacent frames. We propose a novel 3D human pose estimation input method: the sparse interleaved input to address this. This method leverages images captured from different camera views at various time points (e.g., View 1 at time $t$ and View 2 at time $t+\delta$), allowing our model to capture rich spatio-temporal information and effectively boost performance. More importantly, this approach offers two key advantages: First, it can theoretically increase the output pose frame rate by N times with N cameras, thereby breaking through single-view frame rate limitations and enhancing the temporal resolution of the production. Second, using a sparse subset of available frames, our method can reduce data redundancy and simultaneously achieve better performance. We introduce the DenseWarper model, which leverages epipolar geometry for efficient spatio-temporal heatmap exchange. We conducted extensive experiments on the Human3.6M and MPI-INF-3DHP datasets. Results demonstrate that our method, utilizing only sparse interleaved images as input, outperforms traditional dense multi-view input approaches and achieves state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a novel 3d human pose estimation framework which leverages images captured from different camera views at various time points to capture rich spatio-temporal information and effectively boost performance. Their approach theorectically increase the output post frame rate by N times with N cameras and enhance the temperal resolution of the production. In addition, using a spare subset of available frames, their method can reduce data redundancy while simultaneously achieve better performance.
They also introduce DenseWarper model which leverages epipolor geometry for efficient spatio-temporal heatmap exchange. Extensive evaluations using Human3.6M and MPI-INF-3DHP show that their method outperforms SOTA methods.
However, there are certain parts that need some clarifications and authors should discuss how their method can be extended to analyze videos involving multi-persons.

### Strengths
•	They are the first that propose 3d pose estimation task based on sparse interleaved multi-view input
•	They design DenseWarper to convert sparse interleaved inputs into dense pose outputs with high spatio-temporal consistency.
•	They conduct rigorous experiments to demonstrate that their proposed technique achieves better performance.

### Weaknesses
•	There are some parts in current writeup that need clarification When the Spatial Fusion module (Eq. 8) runs on the expanded set H at n_th frame,  it is fusing the real heatmap H_n at view1 with the replicated heatmap H'_n at view 2(which is actually the heatmap from n+∆). This means the module is still applying epipolar constraints to heatmaps of a person at two different frames (n and n+∆). The epipolar geometry is invalid for a moving object at different times. It seems like this paper implicitly assumes that this step can still refine approximate feature correspondences, even if they are not geometrically perfect.
•	Authors have only evaluated their method using datasets involving single person performing an action in each video. They should comment on how to extend their method to videos involving multiple persons. In this case, simply doing spatial fusion using their method will not work. There may be multiple heatmaps (one for each person) and if the people are close by, it may be hard to differentiate how to fuse nearby points correctly.

### Questions
•	Motivate why CPN is used as 2D detector. It seems to perform poorer than simplebaselines
•	Discuss how your method can be extended to handle multi-person videos.
* Perhaps run one set of experiments using multi-person activity dataset such as CMU Panoptic to find out how your method performs compared to existing Multiview 3D pose estimation methods such as MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation, a CVPR 2025 paper.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a novel input paradigm for 3D human pose estimation: sparse interleaved multi-view input.

### Strengths
The "sparse interleaving paradigm" sounds okay.

### Weaknesses
1. I don’t quite understand the significance of this work. Is there still a need to study multi-view markerless human motion capture? In the past four years, there have been many works that directly use the results of multi-view markerless human motion capture as GT. Just like EasyMocap[1]. It can produce the 3D skeleton mentioned in this article, as well as the SMPL human skin template and very high-quality visualization. There is no research significance in this field anymore. Even if this work proposing DenseWarper is published, I don’t think I will use this algorithm. Just like the results in Tables 1 and 2, when MPJPE reaches 20 to 30, I think the visualization effects of these methods are no different to the naked eye.

2. Continuing from the previous point. From the perspective of ICLR academic papers, the experiments also seem to lack a lot of content. First, in terms of visualization. The most important thing for a human pose estimation paper is the visualization presentation, but after reading the entire article, only Figures 6 and 7 have comparisons between the current method and GT. As a reviewer, I am very concerned about qualitative comparative experiments with other methods. But I am disappointed that there is no such content.

3. Human3.6M and MPI-INF-3DHP are really too old. I think there is no need to study the datasets from more than ten years ago. A few years ago, the MPJPE of Human3.6M was already in the twenties[2]. After five years, it has dropped to single digits. Is it still necessary to conduct such experiments?

-----

[1] Dong J, Fang Q, Jiang W, et al. Fast and robust multi-person 3d pose estimation and tracking from multiple views[J]. IEEE transactions on pattern analysis and machine intelligence, 2021, 44(10): 6981-6992.

[2] Zhang Z, Wang C, Qiu W, et al. Adafuse: Adaptive multiview fusion for accurate human pose estimation in the wild[J]. International Journal of Computer Vision, 2021, 129(3): 703-718.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles an interesting and practical problem, proposing a solution that is logically sound and clearly explained. The experimental results demonstrate the method's effectiveness and advantages.

### Strengths
*   **Problem Significance:** The problem addressed in this paper is very interesting and corresponds to a genuine practical need.
*   **Methodology:** The proposed solution is overall logically reasonable and clearly articulated.
*   **Experimental Validation:** The experimental results substantiate the method's effectiveness and advantages.

### Weaknesses
*   **Insufficient Experimental Details and Analysis:**
    *   The impact of the sampling interval $\delta$ on performance within the defined scenario is not thoroughly analyzed. Although a heatmap is mentioned in Appendix Figure 5, corresponding experimental results, particularly quantitative findings, are missing.
*   **Inadequate Citations:**
    *   The methodology sections (Sections 2 and 3) contain very few references. References should also be added to the main text of the experimental section to help readers understand related work.
*   **Formatting and Presentation Issues:**
    *   **Layout:** The placement of several tables does not correspond well with the relevant textual discussions, hindering readability and understanding.
    *   **Table Formatting:** Specific formatting issues exist: Table 3 is missing a bottom horizontal line, and Table 4 has an extra vertical line on the far right.
    *   **Nomenclature Consistency:** The notation for the heatmap (`H` or possibly `**H**`) is not used consistently throughout the text. The variable name `rH` is also somewhat unconventional.
    *   **Writing Quality Suspicions:** Specific lines, such as "the input fps f" (line 439) appearing abruptly, inconsistent citation of method names within parentheses, and the phrase "As shown in Table 5." (line 447) seeming erroneous or out of context, raise concerns. Based on my experience, the entire "Model Efficiency Analysis" section reads as if it might have been generated by an LLM, lacking the flow of human-written academic prose.
*   **Potential Obfuscation in Reporting:**
    *   Only performance efficiency (e.g., MPJPE/mm per MB) is reported, omitting the absolute model size. This gives the impression of potentially skewing the complexity presentation. Reporting the absolute model size is recommended for clearer understanding of the actual complexity.

### Questions
1.  **Sampling Interval ($\delta$) Analysis:** Could you provide a detailed quantitative analysis of how the sampling interval $\delta$ affects performance, based on the heatmap in Appendix Figure 5? What are the specific quantitative results?
2.  **Handling Non-Uniform Intervals:** In practice, the intervals between different views might be non-uniform. Is the algorithm designed to adapt to this? How does its performance hold under non-uniform sampling, or what modifications would be necessary to handle it effectively?
3.  **Clarification on Reporting Metrics:** Could you please also report the absolute model size alongside the performance efficiency metrics to provide a complete picture of the model's complexity?
4.  **Writing and Coherence Clarification:**
    *   Can the authors clarify the abrupt phrasing in lines 439 and 447, and ensure methodological names are cited consistently throughout the text?
    *   Could the authors confirm the provenance and carefully review the "Model Efficiency Analysis" section for coherence and accuracy?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a novel input design for multi-view, temporal 3D human pose estimation, called sparse interleaved input, which reduces the computational cost compared to methods that use all frames across all camera views. The proposed method uses a diagonal approach to select the input frames (e.g., view 1 at frame 1, view 2 at frame 2, view 3 at frame 3, view 4 at frame 4) to predict the 3D pose of the sequence (3D pose at frame 1, 2, 3, and 4). The paper argues that doing so theoretically can lead to an increased frame rate of N, for N cameras in a system. To process this input, the paper introduces a model called DenseWarper, which first replicates the selected frames across all time-steps to create a dense input space. Then, it uses epipolar geometry to fuse 2D heatmaps across different views. Next, several parallel temporal fusion networks aggregate the information across time and output the 3D human pose. By evaluating the proposed method on two popular benchmarks, the paper shows the effectiveness of the method.

### Strengths
1. The sparse interleaved input idea is a simple, yet effective and novel way to reduce the computational load of multi-view, temporal human pose estimation. The method is somewhat counterintuitive (e.g., using less information leads to better performance), but experimental results support its effectiveness.
2. The computational cost of multi-view, temporal models is an ongoing problem in the pose estimation research, which this paper has addressed.
3. The proposed architecture is sound, and the presentation is clear. The paper is also well-written and uses clear terms to convey its points.

### Weaknesses
1. The paper reports state-of-the-art performance, but the conclusion is based on a comparison with baseline results that have been replicated from the original works. As a result, the majority of the results in the tables do not match the original works. While this is understandable for methods that have been replicated, it contradicts the results that use the original model weights (e.g., AdaFuse). What is the reason for this discrepancy? Was a different evaluation protocol used in this paper?
2. I assume that camera parameters have been used in this paper. In that case, some crucial and highly cited references (e.g., [1] and [2]) are missing from the results.
3. While the core idea is interesting, the paper does not position its performance within the existing literature. 

References:
1. Iskakov, Karim, et al. "Learnable triangulation of human pose." Proceedings of the IEEE/CVF international conference on computer vision. 2019.
2. Remelli, Edoardo, et al. "Lightweight multi-view 3D pose estimation through camera-disentangled representation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

### Questions
1. I would appreciate if the authors can address the points I raised in the Weaknesses section (e.g., missing references & results discrepancies) 
2. The spatial fusion described in the paper bears a strong resemblance to the method in AdaFuse. Could you please clarify what the methodological novelty of the proposed approach is compared to AdaFuse?

### Soundness
2

### Presentation
3

### Contribution
3
