# SpikeStereoNet: A Brain-Inspired Framework for Stereo Depth Estimation from Spike Streams

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Conventional frame-based cameras often struggle with stereo depth estimation in rapidly changing scenes. In contrast, bio-inspired spike cameras emit asynchronous events at microsecond-level resolution, providing an alternative sensing modality. However, existing methods lack specialized stereo algorithms and benchmarks tailored to the spike data. To address this gap, we propose SpikeStereoNet, a brain-inspired framework to estimate stereo depth directly from raw spike streams. The model fuses raw spike streams from two viewpoints and iteratively refines depth estimation through a recurrent spiking neural network (RSNN) update module. To benchmark our approach, we introduce a large-scale synthetic spike stream dataset and a real-world stereo spike dataset with dense depth annotations. SpikeStereoNet outperforms existing methods on both datasets by leveraging spike streams' ability to capture subtle edges and intensity shifts in challenging regions such as textureless surfaces and extreme lighting conditions. Furthermore, our framework exhibits strong data efficiency, maintaining high accuracy even with substantially reduced training data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the limited availability of methods and benchmarks for stereo depth estimation from raw spike camera streams, a sensor modality distinct from both frame-based and event cameras. To this end, the authors contribute:
- Benchmarks: A synthetic dataset with dense ground truth and a real-world dataset captured using dual spike cameras and a Kinect.
- SpikeStereoNet: An end-to-end framework that adapts iterative refinement (similar to RAFT-Stereo) by replacing GRU updates with a Recurrent Spiking Neural Network (RSNN) using adaptive Leaky Integrate-and-Fire neurons for native spike data processing.

### Strengths
The paper is clearly written, with detailed appendices that support reproducibility. The benchmark contributions are substantial, and the integration of spiking dynamics into a refinement framework is well aligned with the characteristics of spike data. The analysis is thorough, covering dynamical stability, theoretical convergence, and ablations that validate the design.

### Weaknesses
The paper's primary weakness is the disconnect between its claims and its evidence. It claims to be a solution for "rapidly changing scenes" and "highly dynamic scenarios," but the new datasets appear to be limited to simple, object-centric tabletop scenes with non-complex motion (e.g., smooth camera panning/hovering, "camera shake"). This dataset is insufficient to validate the central claims of the paper.

Following from previous point, the dataset seems to lack challenging, realistic scenarios. The scenes in Fig. 4, 5, 8, and 9, together with the supplemented video, all show static objects on a planar surface. The paper gives no evidence of testing on scenes with:

Fast egomotion (e.g., a car or drone navigating).
Multiple independently moving objects.
Complex, non-planar 3D environments (e.g., outdoor scenes, complex indoor rooms). This makes the benchmark's "challenge" questionable.
The adaptive LIF model (Eq. 3, Appendix A.2.4), with learned parameters, is a core component of the RSNN updater. However, the ablation study (Table~2) does not isolate its contribution, leaving it unclear how much adaptivity improves performance over a simpler, fixed LIF variant.

While the paper outlines detailed calibration procedures (Appendix A.4), the use of a Kinect for real-world depth ground truth introduces noise. This is a known limitation and, despite the challenges of data collection, deserves acknowledgment.

The quantitative results for the real-world dataset (Table 4) are central to validating sim-to-real transfer, yet they are placed in the appendix. Given their importance, they should be included in the main paper, arguably more so than the extended qualitative results.

### Questions
Can you please justify the claim that your method is designed for "highly dynamic scenarios" when the provided datasets appear to only contain tabletop scenes with simple, non-dynamic motion?

Are there any scenes in your synthetic or real-world test sets that feature (a) fast egomotion, (b) multiple independently moving objects, or (c) complex non-planar environments? If not, how can you be sure the method generalizes beyond static tabletop/floor scenes?

Given that the real-world video is not available, can you provide more details on the "object motion" and "camera shake" mentioned in the appendix? Are these motions fast and challenging, or slow and simple?

The adaptive LIF model (Eq. 3, A.2.4) is a key component. How much does this adaptivity (i.e., learning) contribute to performance versus using a simpler, non-adaptive LIF with fixed (but tuned) hyperparameters?

You state that all baselines were "re-trained using the same settings" for fairness, which is excellent. Could you please clarify if this included a full hyperparameter search for each baseline on this new spike dataset, or were the default hyperparameters from their original papers used? Ensuring baselines are optimally tuned for this new data modality is crucial.

Your work focuses purely on spike streams. However, hybrid sensor systems are common. Have you considered fusing the spike stream with either a reconstructed TFI image (from the spikes themselves) or with synchronous RGB frames (like those from the Kinect) to see if this further improves robustness? How might the network change to accommodate this?

I strongly recommend bringing the quantitative results for the real dataset (Table 4) into the main paper. To make space for Table 4, perhaps the qualitative results (Fig. 4/5) could be slightly condensed.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SpikeStereoNet, a biologically inspired framework for stereo depth estimation directly from raw spike streams. The method integrates a recurrent spiking neural network (RSNN) into an iterative refinement structure and introduces two benchmark datasets, including a large-scale synthetic dataset and a real-world spike stereo dataset. The approach achieves state-of-the-art performance compared with both event-based and frame-based stereo methods and demonstrates strong generalization and data efficiency.

### Strengths
1. Novel and timely problem setting in neuromorphic stereo vision. The work explores stereo depth estimation directly from spike streams, an under-explored but promising topic.

2. Valuable dataset contribution. The synthetic and real spike stereo datasets fill an existing gap and will likely benefit future research in this area.

3. The proposed RSNN-based iterative update operator is biologically motivated and well integrated into the overall pipeline. The adaptive leaky integrate-and-fire neuron design with context-dependent parameters ($\alpha$, $\beta$, $\gamma$) is innovative.

4. The experimental design is comprehensive. Comparisons include both frame-based baselines such as RAFT-Stereo and Selective-Stereo, and event-based methods, such as ZEST. Results consistently show clear advantages of the proposed method.

5. The data efficiency experiments demonstrate strong generalization under limited supervision, which is an important property for spike-based systems.

6. Theoretical analysis of convergence and temporal stability adds rigor to the model design.

### Weaknesses
1. The real-world dataset remains small and limited to indoor scenes. Quantitative results for real data are not clearly presented in the main paper, which weakens the validation of real-world applicability.

2. The claimed biological plausibility is conceptually appealing, but its computational or representational benefits compared with conventional recurrent architectures are not fully quantified. Some biological architecture works lack discussion [1, 2].
 
3. The ablation study covers some components but does not isolate the contribution of the adaptive neuron model or explore the sensitivity to regularization weights.

[1] ClearSight: Human Vision-Inspired Solutions for Event-Based Motion Deblurring[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025: 7462-7471.

[2] SABV-Depth: A biologically inspired deep learning network for monocular depth estimation. Knowledge-based systems, 263, 110301.

### Questions
1. How sensitive is SpikeStereoNet to the spike threshold and the temporal quantization step used during spike generation?

2. Does the proposed method provide measurable computational or energy efficiency benefits?

3. Does the adaptive LIF with input-dependent propose the deployment challenge? because all the parameters will change under different timesteps.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes SpikeStereoNet, a framework for stereo depth estimation directly from spiking camera data. Using a Recurrent Spiking Neural Network (RSNN), the method iteratively refines depth estimation. It provide synthetic and real-world datasets with depth labels for evaluation. Experiments show that SpikeStereoNet outperforms some existing methods on both datasets.

### Strengths
1. The paper introduces large-scale synthetic and real-world spiking stereo datasets, filling an important gap in spike-based depth estimation.

2. It proposes a biologically inspired RSNN-based architecture (SpikeStereoNet) for stereo depth estimation from spiking data.

3. It provides a theoretical analysis of the RSNN’s iterative dynamics, demonstrating the model’s stability and convergence.

### Weaknesses
1. The paper overlooks prior studies and incorrectly claims to be the first to estimate stereo depth from spiking data. Actually, “Learning Stereo Depth Estimation with Bio-Inspired Spike Cameras” has already investigated the same problem. Consequently, the novelty and originality of this work are debatable.

2. The paper does not clearly explain the model’s novelty. The proposed framework is similar to existing stereo methods, differing mainly in using an RSNN as the update module. However, the paper does not explain why an RSNN is necessary compared to existing update modules. And it also does not explain what makes SpikeStereoNet particularly suited for spiking data.

3. The paper lacks a systematic organization and analysis of the experimental results. Although it demonstrates that the proposed method outperforms both event-based and image-based approaches, it does not analyze the source of this performance advantage. Moreover, the paper does not clearly justify the choice of comparison methods, explain their relevance, or indicate whether they represent state-of-the-art baselines.

### Questions
1. For the real-world dataset, it is unclear how temporal synchronization between different cameras is achieved. The paper does not describe any synchronization circuitry or mechanism.

2. It is unclear why event-based stereo methods are included for comparison. If such comparisons are necessary, why are only ZEST and StereoSpike selected, while other representative methods such as DDES[1] and Se-cff[2] are not considered?

[1] Yeongwoo Nam, Mohammad Mostafavi, Kuk-Jin Yoon, and Jonghyun Choi. Stereo depth from events cameras: Concentrate and focus on the future. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6114–6123, 2022.

[2] Stepan Tulyakov, Francois Fleuret, Martin Kiefel, Peter Gehler, and Michael Hirsch. Learning an event sequence embedding for dense event-based deep stereo. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1527–1537, 2019.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes SpikeStereoNet, a brain-inspired framework and the first to estimate stereo depth directly from raw spike streams. Moreover, this work introduces a large-scale synthetic spike stream dataset and a real-world stereo spike dataset with dense depth annotations. Results demonstrate that SpikeStereoNet outperforms existing methods.

### Strengths
This work is clearly written, makes significant contributions, and proposes a new dataset and an SNN-based processing method. I believe it will be beneficial to the neuromorphic vision community.

### Weaknesses
1. This work used a spiking neural network, but it was not mentioned in the related work. What is the difference between a recurrent spiking layer and a raw spiking layer? I suggest the authors supplement their research on SNNs.

2. In the experimental section, there is a lack of ablation experiments related to RSNN construction, which is puzzling as to why the proposed method is effective.

3. What are the differences between adaptive LIF neurons and vanilla LIF neurons? If vanilla LIF is used, will there be a performance loss?

### Questions
see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4
