# Flow-Matching Guided Deep Unfolding for Hyperspectral Image Reconstruction

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 4, 0

## Abstract
Hyperspectral imaging (HSI) provides rich spatial–spectral information but remains costly to acquire due to hardware limitations and the difficulty of reconstructing three-dimensional data from compressed measurements. Although compressive sensing systems such as CASSI improve efficiency, accurate reconstruction is still challenged by severe degradation and loss of fine spectral details. We propose the \textit{Flow-Matching-guided Unfolding network} (FMU), which, to our knowledge, is the first to integrate flow matching into HSI reconstruction by embedding its generative prior within a deep unfolding framework. To further strengthen the learned dynamics, we introduce a mean velocity loss that enforces global consistency of the flow, leading to a more robust and accurate reconstruction. This hybrid design leverages the interpretability of optimization-based methods and the generative capacity of flow matching. Extensive experiments on both simulated and real datasets show that FMU significantly outperforms existing approaches in reconstruction quality. Code and models will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript proposes a flow-matching-guided unfolding network (FMU) for hyperspectral image reconstruction. The key idea is to introduce flow-matching generative priors into the deep unfolding framework for the first time, combined with a two-stage training scheme to embed learned priors and a mean-velocity constraint to enhance training stability. Experiments on an optical-filter-based HSI system demonstrate leading performance. While the method shows clear potential for integrating flow-matching with computational imaging, the technical novelty is incremental and mainly lies in component-level improvements over an existing unfolding architecture.

### Strengths
1. This is the first attempt to incorporate flow-matching generative priors into a deep unfolding network for HSI reconstruction, achieving notable improvements in reconstruction quality compared with latent diffusion priors.
2. The proposed mean-velocity constraint is a tailored regularization strategy that enhances global stability of flow-matching training for HSIs.
3. Comparisons on both synthetic and real data demonstrate clear quantitative advantages over prior methods, and the ablation studies are well-designed to support the effectiveness of major components.

### Weaknesses
1. The overall architecture—including the GAP unfolding scheme, Trident Transformer, and two-stage training pipeline—closely follows LADE-DUN (Wu et al., 2024). The main novelty lies in replacing latent diffusion with flow matching, which is a component-level modification rather than a substantial architectural innovation. Given that LADE-DUN itself is very recent, the incremental novelty here is limited.
2. The mean-velocity constraint, while useful, is essentially a simple regularizer with limited theoretical depth. The manuscript also lacks a deeper analysis of how the generative prior quality correlates with reconstruction performance, and why flow matching provides intrinsic advantages over diffusion in this context.
3. The main quantitative results (Table 1) are reported only on the optical-filter-based HSI system. Since CASSI is the dominant benchmark in the community, the lack of quantitative CASSI comparisons weakens the general SOTA claim. The qualitative CASSI results (Fig. 6) are insufficient to fill this gap.
4. Reproducibility is affected by missing key details: sampling steps for the flow prior, epoch numbers for the two training stages, and comparison of inference speed with baseline methods.
5. The manuscript claims SOTA performance on an optical-filter-based HSI system, whose forward model should produce measurements with no spatial shift and the same spatial size as the RGB image. However, the measurement in Fig. 5 clearly exhibits the spatial shift artifacts that are unique to CASSI. This contradiction raises serious concerns about the correctness of the experimental setup and the reliability of all related conclusions.

### Questions
1. Motivation and Efficiency: Please provide a direct comparison with LADE-DUN (or other diffusion-prior methods), showing how inference time and reconstruction quality vary under different sampling budgets. This is essential to justify the claimed efficiency advantage of flow matching.
2. Benchmark Completeness: Since CASSI is the mainstream benchmark, can you provide full quantitative CASSI results to support the claimed SOTA performance?
3. Mechanism Understanding: In what aspects are flow-matching priors superior to diffusion priors? Can you provide feature visualizations, numerical stability evidence, or statistical analysis, especially under heavy degradation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Reconstructing accurate 3D hyperspectral images from compressed measurements remains a significant challenge due to degradation and loss of spectral details. This paper for the first time integrates flow matching and its generative prior into hyperspectral image reconstruction and proposes a flow-matching-guided unfolding network with interpretability. Besides the generative capacity of flow matching, it also introduces a mean velocity loss that enforces global consistency of the flow.

### Strengths
- It is the first time that physics-driven deep unfolding with flow matching is applied in HSI reconstruction. Besides, the global consistency of predicted flow is further boosted by a mean velocity loss.
- It leverages both the external prior knowledge derived from clean HSIs and the strong generative capabilities of flow matching to boost reconstruction quality.
- The quantitative results validate the superiority and efficiency of the proposed method. The paper is well written and easy to follow.

### Weaknesses
- The most vital contribution of this paper is the first application of flow matching to HIS reconstruction. However, the principle of flow matching that conforms to the data or task characteristics is not specifically explained. The third and fourth contributions are both related to the performance validation. 
- The derivation of some formulations is unclear. The reason why it is necessary to modify some formulations and the underlying physical or mathematical principles are not clearly explained.
- It is difficult to observe the improvement of the proposed method compared to the competitors in the qualitative results shown in Fig. 5.

### Questions
- How to infer Eq. (7) from the optimization problem of Eq. (5) as there is no definition on $v$ expect defining it as an auxiliary parameter?
- What does “DSC” in Eq. (8) denote?
- Why the latent feature extracted from $(y_nor, x)$ can be assumed to be the high-quality prior as there is the ground-truth HSIs in the input?

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
This paper proposes Flow-Matching-guided Unfolding (FMU), a novel framework for hyperspectral image (HSI) reconstruction that integrates flow matching generative priors into a deep unfolding network (DUN). The method addresses the challenges of reconstructing 3D hyperspectral data from compressed CASSI or optical filter–based measurements, where high spectral detail is often lost.
FMU introduces a mean velocity loss to enhance the stability and global consistency of the learned flow, improving spectral fidelity and robustness under heavy degradation. Experiments on both simulated (CAVE, KAIST) and real CASSI datasets show that FMU achieves state-of-the-art performance with 42.13 dB PSNR / 0.9900 SSIM, outperforming strong baselines like LADE-DUN and DAUHST while maintaining moderate computational cost. The authors claim FMU’s hybrid design effectively combines interpretability from optimization and strong generative priors from flow matching, suggesting practical value for compact, chip-integrated HSI systems.

### Strengths
1. Novel integration of flow matching and deep unfolding: the first to embed flow-matching priors into a physics-based HSI reconstruction framework.

2. Strong empirical results: consistently outperforms all SOTA baselines on both simulated and real-world datasets with clear gains in PSNR/SSIM.

3. Comprehensive analysis: includes ablation studies on prior types, loss weights, and denoiser architectures, demonstrating robustness and interpretability.

4. Practical implications: This paper validated on multiple HSI systems (CASSI and optical filter–based), showing promise for miniaturized imaging hardware.

### Weaknesses
1. Limited theoretical justification: the integration of flow matching into the unfolding process is empirically motivated, with minimal theoretical analysis of convergence or stability.

2. Two-phase training complexity: The proposed method requires pretraining an encoder and a flow module, increasing training cost and implementation difficulty.

3. Comparative fairness concerns: The paper's evaluation mainly focuses on PSNR/SSIM; real-world generalization and robustness across noise types are less explored.

4. Insufficient qualitative diversity: visual comparisons emphasize limited spectral bands; cross-scene or dynamic spectral variation tests are lacking.

5. Insufficient dataset: This paper only conducts experiments on some simulation datasets and 5 real scene, which cannot prove the methods' effectiveness and generalization.

### Questions
Please provide results on more hyperspectral datasets and provide some insights about converting flow matching with deep unfolding.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes a flow-matching-guided unfolding (FMU) for hyperspectral image (HSI) reconstruction from compressed measurements—specifically in CASSI and optical filter–based systems. The whole pipeline is highly related on LADE-DUN published in ECCV 2024. The only technical difference is to relace the latent diffusion model (LDM) of LADE-DUN with flow matching. A mean velocity constraint is proposed for flow matching but such contribution is very limitted. Moreover, the testing results are not on the same standard, making the effectiveness unclear.

### Strengths
- Following the whole pipeline of LADE-DUN published in ECCV 2024, flow matching is first used in compressed hyperspectral image (HSI) reconstruction task.
- In the same simulated optical filter-based HSI datset, proposed FMU (4.09 M, 98.84 GDLPS) outperforms previous SPECAT (0.29 M, 12.4 GLOPS) by a large margin (1.76 dB).

### Weaknesses
- The whole method is highly similar with previous LADE-DUN in terms of unfolding architecture, denoising network, two-stage trainning, etc. The only difference is to replace latent diffusion with flow matching. For flow matching, proposed mean velocity constraint is incremental for previous methods (eg. Rectified Flow, FM-OT). Thus, the novelty is very limitted.
- As for simulated testing, almost all compared methods (excpet for SPECAT) are designed for CASSI system instead of optical filter-based system. Two different systems correspond to two different image formation models, leading to two different reconstruction tasks. In the core Table 1, the testing results are not the same standard. The real effectiveness is unknow.
- As for real testing, visualized results are overexposed. The reconstructed details are hard to evaluate.

### Questions
- Compared to LADE-DUN, what are the core contributions?
- As for Table 1, how to test compared methods designed for CASSI data on optical filter-based data?

### Soundness
2

### Presentation
3

### Contribution
1
