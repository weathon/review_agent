# Learning Flow-Guided Registration for RGB–Event Semantic Segmentation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Event cameras capture microsecond-level motion cues that complement RGB sensors. However, the prevailing paradigm of treating RGB-Event perception as a fusion problem is ill-posed, as it ignores the intrinsic (i) Spatiotemporal and (ii) Modal Misalignment, unlike other RGB-X sensing domains. To tackle these limitations, we recast RGB-Event segmentation from fusion to registration. We propose BRENet, a novel flow-guided bidirectional framework that adaptively matches correspondence between the asymmetric modalities. Specifically, it leverages temporally aligned optical flows as a coarse-grained guide, along with fine-grained event temporal features, to generate precise forward and backward pixel pairings for registration. This pairing mechanism converts the inherent motion lag into terms governed by flow estimation error, bridging modality gaps. Moreover, we introduce Motion-Enhanced Event Tensor (MET), a new representation that transforms sparse event streams into a dense, temporally coherent form. Extensive experiments on four large-scale datasets validate our approach, establishing flow-guided registration as a promising direction for RGB–Event segmentation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work proposes a method for semantic segmentation with RGB + event camera inputs. It first aligns the feature of event data and rgb images using optical flow on the event samples, then use the aligned feature to assist the final rgb image segmentation. Experiment shows improvements on finetuned datasets over previous RGB + event camera work.

### Strengths
The idea of using optical flow to align event data with RGB data is interesting. The authors claim that it works better than previous fusion-based methods, where the proposed one finds better pixel-level correspondences.

The proposed method effectively aligns event data with natural images, leading to better representation and performance improvement over previous RGB+event methods on several datasets.

### Weaknesses
1. It is unclear from both method description and figure explanation why the proposed method is not a "fusion-based" method. I understand it performs alignment step, but ultimately it is still a way to fuse 2 modalities. How does existing methods fuse the information should be presented clearer in fig 1 (a).
2. The RGB-based baselines in table 2 are out-dated, there are many more recent methods that perform (much) better, which should be compared against. This also raises a serious concern: though theoretically the method is complimentary with RGB-based strategies, is adding event data for RGB segmentation really a good choice comparing to simply scaling up the training data in the RGB side, which is arguably much easier and should provide much better zero-shot generalization capabilities. The complex design of the proposed method seems to be only marginally better than RGB-based counter-parts on some metrics (accuracy) and datasets (DSEC).

### Questions
Are the numbers of the RGB methods your own implementation or the results reported in the original paper?

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
3

### Summary
This article aims to solve the semantic segmentation task of combining RGB images and event camera data. The author points out that the existing fusion-centric methods have inherent flaws as they ignore the inherent (i) Spatiotemporal Misalignment and (ii) Modal Misalignment issues between the two modalities. To address this issue, this article reconstructs the problem from "fusion" to "registration" and proposes a design principle of “registers first, then fuses.” Its core contribution is a novel flow-guided bidirectional registration framework called BRENet. The key component is the Motion-enhanced Event Tensor (MET) proposed by the author, which is a new event representation method that utilizes coarse-grained optical flow (as a guide) and fine-grained event temporal features to transform sparse event streams into dense representations. This framework uses MET representation in the Bidirectional Registration Module (BRM) and Temporal Fusion Module (TFM) to align and fuse features. Experiments on four large datasets (DDD17, DSEC, DELIVER, and M3ED) have shown that the proposed method achieves state-of-the-art performance.

### Strengths
1. The main advantage is the shift in the problem from the concept of "fusion" to "registration". This reconstruction directly solves a well-known fundamental challenge in RGB–Event perception (Spatiotemporal and Modal Misalignment), which many previous works have overlooked. The idea of converting inherent motion lag into optimizable optical flow estimation error is insightful.

2. The Motion-enhanced Event Tensor (MET) proposed by the author has clear motivation. It cleverly combines coarse-grained global motion (from optical flow) with fine-grained local temporal details (from Temporal Conv Module) to create a dense and easily registered representation. The ablation experiment (Table 5) confirmed its superiority over other representations such as voxel grid and AET.

3.  The proposed method demonstrated significant and consistent performance improvements compared to the previous SOTA model in four different and challenging benchmark tests. 

4. The author provided a detailed ablation study that validated the contribution of each key component. The study clearly demonstrates the advantages of MET representation, the importance of coarse-grained optical flow and fine-grained temporal features, and the effectiveness of BRM and TFM. The plug-and-play verification of MET is also a bonus point.

### Weaknesses
1.  A major issue is that the framework relies on a pre-trained optical flow encoder (E-RAFT). The appendix mentions that the encoder was pre trained on the DSEC dataset. This raises a key question about comparative fairness, especially in DSEC benchmark testing. It is currently unclear to what extent the performance improvement is attributed to powerful, pre-trained prior knowledge of optical flow, rather than the BRENet architecture itself. The ablation experiments in Table 7 demonstrate robustness to different encoders, but they are all complex pre-trained models. The paper needs to clarify (a) how the performance would be if the optical flow encoder and segmentation task were trained together from scratch; (b) if the E-RAFT encoder has indeed been pre trained on DSEC, how does this not constitute data leakage for DSEC evaluation. 
2.: The complete framework is very complex, involving multiple encoders (image, event temporal, optical flow), a CFE (Coarse-to-Fine Estimator), a bidirectional BRM, and a TFM. The use of bidirectional propagation and spatial warping increases the "significant computational cost" that cannot be ignored. Table 4 claims that MACs are "equivalent" to EISNet, but does not explicitly state whether the MACs of the optical flow encoder (E-RAFT) are included in this calculation. This negligence may be misleading. The paper needs to provide a more transparent delay analysis (such as FPS) and a clear breakdown of computational costs (including optical flow estimators). 
3.  The design of the core modules (BRM and TFM) is an adaptation of existing work. The paper explicitly states that BRM is an extension of the Frequency-aware Cross-modal Feature Enhancement (FCFE) module in FEVD (Kim et al., 2024). TFM uses standard deformable convolutions for alignment. The main architectural innovation seems to be focused on the design of CFE/MET, rather than the registration and fusion modules themselves. 
4. Failure case analysis (Section E) shows that in extremely dark areas, optical flow will "significantly degrade", leading to "blurry" predictions. This confirms that the performance of the model is closely related to the quality of the optical flow estimator. This is a direct and inherent weakness of the "flow guidance" method.

[1] Frequency-aware event-based video deblurring for real-world motion blur.CVPR2024

### Questions
Please seek weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper reframes RGB–event segmentation from fusion to registration: estimate forward/backward optical flow on the event stream, build a Motion-enhanced Event Tensor (MET) that mixes coarse (flow) and fine (temporal event) cues, register events to the RGB frame in both directions, then fuse temporally (TFM). Results on DDD17, DSEC, DELIVER, and M3ED are consistently strong, with SOTA performances on all four datasets.

### Strengths
1. The idea of leveraging optical flow to help event-frame registration is interesting.
2. The performances of the method on the frame-event datasets are very compelling.
3. The motivation of the method is clear.

### Weaknesses
Major:
1. Backward “flow” is obtained by reversing the order of the event frames to compute $O^b$. That means inference uses future events inside each window, which limits true online deployment and complicates latency guarantees.
2. The authors claim “comparable processing latency” but do not report FPS or wall-clock latency, only Params/MACs. This is not convincing enough.
3. It's not clear whether the rise of feature similarity in Tab. 1 results from the optical flow itself. It could also be attributed to the dense property of optical flow compared to the voxel grid.

Minor:

The contraction argument around the flow refinement (Eq. 5) is heuristic and not tied to specific network conditions. There’s no theorem that bounds segmentation error as a function of measurable flow error on these datasets.

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces BRENet, a flow-guided, registration-centric framework for RGB–Event semantic segmentation. Unlike previous fusion-based methods that assume perfect alignment between RGB and event data, BRENet explicitly addresses spatiotemporal and modal misalignment by reformulating the task as a registration problem. The model uses optical flow as a bridge between modalities and proposes three key components: Motion-Enhanced Event Tensor (MET) for dense temporal representation, Bidirectional Registration Module (BRM) for forward/backward alignment, and Temporal Fusion Module (TFM) for temporally coherent fusion. Experiments on four benchmarks show consistent improvements over state-of-the-art baselines.

### Strengths
(1) Clear motivation and paradigm shift. The paper provides a well-articulated motivation for redefining RGB–Event perception from a fusion problem to a registration problem. The analysis of spatiotemporal and modal misalignment is well grounded, and the use of optical flow as a registration bridge is novel and conceptually elegant. 

(2) Comprehensive and well-validated framework. BRENet integrates several complementary modules (MET, BRM, TFM) in a coherent design. The experimental validation is extensive, covering multiple benchmarks and ablation studies.

### Weaknesses
(1) Model complexity and runtime considerations. BRENet involves multiple heavy components—bidirectional flow estimation, frequency-domain operations, and deformable convolutions—making the pipeline complex. Although parameter counts and MACs are reported, there is no analysis of inference speed or runtime efficiency, which is critical for real-time event-based perception. It is necessary to provide relevant data to evaluate the complexity of the model.

(2) Lack of analysis experiments on architectural or hyperparameter settings. Although the model includes several tunable components, the paper lacks sensitivity or analysis studies. For instance, it would strengthen the work to analyze: Bin size (B) and its impact on event density and segmentation quality; Number of refinement iterations (J) in the Coarse-to-Fine Estimator; Frequency-domain fusion settings (e.g., FFT resolution or real/imaginary channel ratios). Such experiments would clarify the model’s stability and provide deeper insights into why each module design choice matters.

(3) The limitation section in the appendix is brief and does not thoroughly discuss the dependence on pre-trained flow encoders, potential propagation of flow errors, or scalability to other modalities (e.g., depth, LiDAR).

### Questions
See the weakness.

### Soundness
2

### Presentation
3

### Contribution
2
