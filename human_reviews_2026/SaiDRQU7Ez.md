# StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams

- Decision: Accept (Poster)
- Scores: 8, 8, 4

## Abstract
Real-time reconstruction of dynamic 3D scenes from uncalibrated video streams demands robust online methods that recover scene dynamics from sparse observations under strict latency and memory constraints. Yet most dynamic reconstruction methods rely on hours of per-scene optimization under full-sequence access, limiting practical deployment.
In this work, we introduce **StreamSplat**, a fully feed-forward framework that instantly transforms uncalibrated video streams of arbitrary length into dynamic 3D Gaussian Splatting (3DGS) representations in an online manner. 
It is achieved via three key technical innovations: 1) a probabilistic sampling mechanism that robustly predicts 3D Gaussians from uncalibrated inputs; 2) a bidirectional deformation field that yields reliable associations across frames and mitigates long-term error accumulation; 3) an adaptive Gaussian fusion operation that propagates persistent Gaussians while handling emerging and vanishing ones.
Extensive experiments on standard dynamic and static benchmarks demonstrate that StreamSplat achieves state-of-the-art reconstruction quality and dynamic scene modeling. Uniquely, our method supports the online reconstruction of arbitrarily long video streams with a $1200\times$ speedup over optimization-based methods. 
Our code and models are available at https://streamsplat3d.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes **StreamSplat**, a **feed-forward framework for online dynamic 3D reconstruction from uncalibrated video streams**.  
It aims to replace conventional optimization-based dynamic 3DGS pipelines with a real-time, fully feed-forward alternative.

**Key contributions include:**
1. **First feed-forward dynamic 3D reconstruction framework** under an *uncalibrated* camera setting.  
2. **Effective technical components:**
   - *Probabilistic position sampling* for robust Gaussian position inference under geometric uncertainty.  
   - *Bidirectional deformation fields* that jointly model forward and backward motion to ensure temporal coherence.  
3. **High-quality results:** Both quantitative tables and qualitative visualizations show consistent improvements and strong temporal coherence across datasets.

### Strengths
- **Impressive empirical results:** The method achieves substantial gains over prior 3DGS and NeRF-based approaches, setting a new benchmark for *online dynamic reconstruction from uncalibrated video streams*.  
- **Reasonable and clear pipeline:** The proposed two-stage static/dynamic training scheme is well-motivated and reproducible. The combination of a strong image encoder and a dynamic decoder makes architectural sense.  
- **Excellent efficiency:** StreamSplat attains orders-of-magnitude speedup (seconds vs. hours) compared to optimization-based methods.  
- **Comprehensive evaluation:** The paper compares against strong baselines on multiple datasets, including both static (RE10K, CO3Dv2) and dynamic (DAVIS, YouTube-VOS) settings.

### Weaknesses
Major Weaknesses

**W1.** The formulation in *line 157* appears problematic: $(u,v)$ represents pixel coordinates while the offset $o_i$ is in unit space. Their direct addition may be incorrect if the coordinate system is rectilinear. A clarification of this coordinate transformation is needed.  

**W2.** Algorithm 2’s *aggregation and fusion* step is ambiguous. The operation `UPDATE` in line 228 is not clearly defined—does it simply replace $\tilde{\mathcal{G}}$ with$ \mathcal{G}_{k-1}^+ $, or else?  
Additionally:
- The notation $Î_{t_{k−1}→t_k}$ is unclear—what is the meaning of the arrow?  
- The definition of $π$ (projection parameters) is missing.  
- The term **“cached”** in line 219 feels implementation-specific and should be formalized or rephrased.  
Finally, “rendered frames” in line 221 conflicts conceptually with “dynamic scenes” in line 292—please clarify the intended distinction.

Minor Weaknesses

**W3.** The frame indices in *Figure 2* appear inconsistent. Section 3.2 (line 209) describes deformation between $t=0$ and $t=1$; thus, the figure should use either $(t_n, t_{n-1})$ or $(t_0, t_1)$ for consistency.

### Questions
**Q1.** Could you provide a visualization of *multi-point tracking*? Figures 3 and the supplementary videos only illustrate a single tracked point. From my understanding, a per-pixel feed-forward network may struggle with precise point correspondence. Showing multiple tracked points (e.g., ~30) in a local neighborhood would give a fairer sense of temporal coherence.  

**Q2.** Please include a **runtime breakdown**—how much time is spent in the encoder, deformation module, and rasterization, respectively? This would help assess real-time feasibility.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper first points out the challenge of achieving real-time dynamic 3D reconstruction from uncalibrated video streams while maintaining the quality and functionality of offline methods. To address this, the paper proposes an online dynamic 3D reconstruction framework called StreamSplat, which integrates probabilistic 3D Gaussian encoding, a bidirectional deformation field, and adaptive Gaussian fusion for temporally coherent and efficient scene modeling. The method aims to enable scalable, feed-forward, and fully online reconstruction of arbitrarily long video streams with state-of-the-art quality and significant speed improvement over optimization-based approaches.

### Strengths
* The paper tackles an underexplored yet practically important problem: real-time dynamic 3D reconstruction from uncalibrated video streams, which existing 3DGS and NeRF-based methods generally overlook due to their offline and per-scene optimization nature.

* Unlike prior optimization-based dynamic 3DGS methods, StreamSplat introduces a fully feed-forward pipeline that supports online inference without requiring camera calibration or pre-computed poses, making it highly suitable for real-world deployment scenarios.

* The experiments cover static and dynamic benchmarks, interpolation tasks, and zero-shot evaluation, providing convincing evidence of the model’s robustness, scalability, and temporal coherence.

### Weaknesses
* It would be helpful if the authors could clarify whether their framework is capable of predicting or estimating camera poses, given that StreamSplat operates under uncalibrated input conditions. If not, discussing potential extensions in this direction would strengthen the paper's completeness.

* The paper would benefit from additional discussion or experiments on highly dynamic scenes with significant topological changes (e.g., frequent object entries and exits from the field of view). It remains unclear how robust the proposed method is under such conditions.

### Questions
See weaknesses.

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
4

### Summary
StreamSplat introduced a feed-forward framework for reconstructing dynamic 3D scenes from uncalibrated video streams. 

Streamsplat work simliarly as current Dust3R style model and predict the 3D gaussian parameters with a vision transformer.

StreamSplat also introduce three key technical innovation: 1. a probabilistic sampling mechanism 2. a bidirectional deformation field 3. an adaptive Gaussian fusion operation

### Strengths
Strength:
1. The paper is well-written and easy to follow.
2. StreamSplat is feedforward and increase the speed of reconstruction.
3. In the figure 4, StreamSpalt shows persistent gaussians across frames, which shows the potential of long-term modeling.

### Weaknesses
Major Weakness:

1. Lack of Rigorous Evaluation Protocols: The evaluation does not follow established dynamic reconstruction benchmarks such as DyCheck or NVIDIA Dynamic Scene Dataset. The chosen datasets (DAVIS and YouTube-VOS) are more typical for video segmentation or interpolation, not 4D reconstruction.

2. Limited Training Dataset: The paper uses a mix of static (CO3Dv2, RealEstate10K) and limited dynamic (DAVIS, YouTube-VOS) datasets for training. However, DAVIS contains only a few short clips, this raises concerns about the model’s generalization to complex  motion and real-world application.

3. Dynamic reconstruction typically involves multi-view or multi-camera setups to test geometric consistency (e.g., DyCheck). StreamSplat, in contrast, evaluates on DAVIS using middle-frame interpolation and trivializes the problem.

Minor Weakness:

1.In Figure 1, “curent” should be corrected to “current.”

### Questions
Please see the major weakness above. 

1. In the section on the Bidirectional Deformation Field, the authors describe how forward and backward deformations enhance temporal consistency. However, given that StreamSplat is evaluated on 5-frame and 8-frame interval tasks, it’s unclear why the model does not increase the number of active Gaussians to better represent accumulated motion over longer intervals.

2. Since the input frames are two, why not use the pretrain model from Croco (e.g. Dust3R, Cut3R)

### Soundness
2

### Presentation
3

### Contribution
3
