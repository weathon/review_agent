# Epipolar Prompt: A Simple Baseline for Motion Segmentation

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Reconstructing dynamic 3D/4D scenes from uncalibrated videos remains challenging due to moving objects violating the static assumptions required for multi-view consistency. While motion segmentation can resolve this, existing methods struggle to generalize across datasets and ignore 3D geometry cues. To this end, we propose **Epipolar Prompt**, a zero-shot framework that synergizes epipolar geometry with foundation segmentation models (e.g., SAM) to achieve robust motion segmentation. Our approach first computes epipolar error maps from optical flow correspondences to localize regions that violate static scene assumptions. These error maps then guide an iterative prompt selection strategy to generate precise segmentation from SAM. Surprisingly, our simple yet effective prompt-based method outperforms both supervised and unsupervised approaches on standard benchmarks (e.g., +9.3 IoU over DAVIS2017) and demonstrates strong generalization to in-the-wild videos. Furthermore, we show that our motion masks serve as a plug-and-play enhancement for existing dynamic 4D reconstruction methods, leading to improved performance. View results at: https://anonymous-for.github.io/ICLR-4426

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Epipolar Prompt, a training-free network for robust motion segmentation between pairwise images. The authors leverage a pretrained optical flow model to predict correspondences, which are then used to estimate the relative camera pose and compute epipolar error maps. The epipolar error is then used as a cue to prompt the foundation segmentation model SAM to achieve precise motion segmentation. The authors show that the proposed pipeline outperforms both supervised and unsupervised approaches on DAVIS2017 and in-the-wild videos. It can also be used with 4D reconstruction models for better camera tracking.

### Strengths
- The paper proposes a simple and training-free solution for motion segmentation, which leverages pretrained optical flow and segmentation models SAM.
- The authors use an epipolar-guided error map to prompt the SAM model, providing useful dynamic cues by estimating camera pose and further improving segmentation through the foundation model.
- The authors demonstrate state-of-the-art performance on multiple benchmarks, outperforming both supervised and unsupervised methods.

### Weaknesses
- Limited novelty. As the authors mention in the introduction, the idea of using epipolar maps has also been explored in RoMo (Goli et al., 2024) and other previous works. The processing steps are similar: (1) use optical flow to obtain correspondences; (2) use correspondences to recover the fundamental matrix and compute epipolar error. This paper uses the epipolar error map to prompt the SAM model, while RoMo uses the error map to train a classifier. 
- Reliability of the epipolar error map in dynamic scenes. The authors propose recovering the fundamental matrix via RANSAC. However, on challenging dynamic scenes (e.g., Sintel), RANSAC often fails to capture static correspondences correctly, which leads to inaccurate camera poses. As a result, the epipolar error map may not provide useful cues for these scenes.
- Comparison to MonST3R masks and runtime. As shown in Table 3, the proposed mask performs similarly to the MonST3R mask, which doesn’t require a foundation segmentation model. Also, as mentioned in the supplementary, the runtime with VLM-based filtering takes about 2–3 seconds per pair, which is slow compared to other methods like the MonST3R mask.

### Questions
- It would be helpful if the authors compared their approach with RoMo (Goli et al., 2024) and clarified the specific contributions. In addition, the authors should include a comparison with RoMo in Table 2.
- Since RANSAC can handle only limited amounts of dynamic motion, how robust is the proposed method for predicting motion segmentation if RANSAC fails to obtain the static correspondences?
- In Table 3, the authors show that incorporating the motion segmentation mask improves depth estimation. The authors should further explain how the mask improves depth performance in DUSt3R/MonST3R, as this is not trivial.
- The proposed method relies on pairwise optical flow and camera pose estimation for motion segmentation. How does the method extend to video sequences? Can it produce temporally consistent segmentation of the same object?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Epipolar Prompt, a zero-shot motion segmentation framework that integrates epipolar geometry with the Segment Anything Model (SAM). The key idea is that violations of epipolar geometry in optical flow correspondences can identify moving regions in a scene. These regions are then used to prompt SAM to produce motion segmentation masks without training or fine-tuning.

### Strengths
1. The paper smartly combines epipolar geometry—a classical geometric constraint—with the Segment Anything Model (SAM). 
2. The method does not require training or fine-tuning, making it lightweight and adaptable. 
3. The results show the effectiveness of the proposed method.

### Weaknesses
1. Dependence on heuristic thresholds: The approach relies heavily on manually tuned parameters such as the epipolar error threshold and confidence/stability scores. This dependence can make performance sensitive to dataset variations and limit robustness in uncontrolled environments.
2. Limited performance on older benchmarks: Although strong overall, the method performs slightly worse than FlowP-SAM on SegTrack v2 and FBMS59, suggesting that it may struggle with low-quality or complex motion data.
3. Vulnerability to optical flow errors: Since the method builds on optical flow correspondences, inaccuracies from flow estimation (due to blur, occlusion, or lighting changes) can directly affect segmentation quality, producing false motion regions.
4. Engineering-heavy, limited theoretical novelty: The contribution lies mainly in the integration of existing tools (epipolar geometry, SAM, heuristic filtering) rather than a fundamentally new theoretical framework, which may reduce its perceived novelty for top-tier conferences.
5. Although the paper provides quantitative benchmarks and qualitative frame examples, the absence of supplementary video results (or at least a more extensive collection of visual examples in the appendix) makes it difficult to fully assess the temporal consistency and perceptual quality of the motion segmentation. Providing such results would strengthen the empirical evidence and transparency of the work.

### Questions
1. The authors mostly follow the evaluation datasets and metrics used in [1]. However, it is unclear why the MOCA dataset [2] was not included in the evaluation. Could the authors explain the reason for excluding this dataset?
2. The combined FlowP-SAM + FlowI-SAM model in [1] demonstrates higher performance than the individual models. It would strengthen the comparison if the authors included the results of FlowP-SAM + FlowI-SAM in Table 2. Since this combined model is likely larger, comparing the number of parameters would further highlight the efficiency advantage of the proposed method.
3. Could the authors also report the model size and inference time to better illustrate the computational efficiency of the proposed approach?
4. How sensitive (or robust) the performance is with respect to the thresholds (confidence, stability, IoU)?


References
[1] Xie, Junyu, et al. "Moving Object Segmentation: All You Need is SAM (and Flow)." Asian Conference on Computer Vision. Singapore: Springer Nature Singapore, 2024.
[2] Lamdouar, Hala, et al. "Betrayed by Motion: Camouflaged Object Discovery via Motion Segmentation." Proceedings of the Asian Conference on Computer Vision, 2020.

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
4

### Summary
The paper introduces a learning-free approach to moving object segmentation via so-called epipolar prompt. Given a pretrained SAM, VLM, and optical flow model, the method first calculate epipolar error map and detect regions that violates the epipolar constraints (ie, moving part). Then it samples few points from the region, prompts SAM using those points, and gets moving object mask. It iteratively updates the mask while being validated via VLM. The method achieves the best/competitive accuracy on the benchmark.

---

The proposed method sounds great with good accuracy. However, there are concerns on the novelty, paper fit, and overstating of 'no training needed' while using more advantageous setups than others, which can mislead. Thus the recommendation is **4: marginally below the acceptance threshold**. However, any thoughts and justifications regarding these concerns would be appreciated!

### Strengths
* **Clarity**

  The paper is written clearly and read really well. It includes necessary technical details to fully understand the method. It further includes in depth analyses of the method, such as ablation study, hyperparameter choices, limitations, justifications on the design choices, etc.

* **Good results**

  The methods achieves the competitive/best accuracy among other methods in the benchmark. In the downstream task (camera pose evaluation and depth estimation), the setup with the proposed methods achieves the best accuracy over other setups using other methods.

### Weaknesses
* **Novelty concern / paper fit**

  Though the paper shows good accuracy on the benchmark, I cannot erase an impression that the method is likely a well-composed pipeline of established methods using heuristics (one could interpret it as a contribution of the method though, ie simple composition of off-the-shelf methods beat learning-based algorithms). The idea is neat indeed. However, I am not so sure if there can be any new novel finding or learning representation learned from the paper. 

  Call of papers in the ICLR webpages says that it accepts applications in vision, but the paper seems quite at the end of the application side (close to WACV, for example). I was wondering if the paper would fit to the ICLR's interests. 


* **Argument on 'no training needed'**

  In Table 2, the paper classifies its method as `No training needed`, which is true. Though the proposed technique itself is `No training needed`, the whole pipeline is based on a more advantageous setup using several off-the-shelf methods including RAFT, SAM, and VLM (with 7M parameters). This argument might mislead others. 

  Also, without the usage of VLM, how much does the accuracy drop on each dataset in Table 2? (Table 8 shows only partial results)

  Also in Table 2, some numbers (on SegTrack v2 and FBMS59) underperform FlowP-SAM. It's fine because the method still outperforms on the other benchmarks and downstream tasks as well. However, given its advantageous setup, it weakens the strength of the paper.

### Questions
* **Minor**

  In Table 4, the number on Sintel, `DUSt3R w/ FlowP-SAM mask`, I think the RMSE 0.5111 might be a typo. 


* **Accuracy of the epipolar map**

  I was wondering if there are any analyses on the accuracy of the epipolar error map and curious how accurate/reliable it is. (it may not be necessary but good to evaluate the accuracy on a dataset with GT camera pose)

* **Extreme scenario**

  I was also wondering if there is any limitation coming from the epipolar map. For example, it's a rare case that other methods can fail as well, but let's assume an image pair where a foreground object dominate most of the image part, and the epipolar error map highlights the background region. Then how will the method behave?

* **What's the runtime of the method?**

  Especially it's curious how much time does each stage take.

### Soundness
3

### Presentation
3

### Contribution
2
