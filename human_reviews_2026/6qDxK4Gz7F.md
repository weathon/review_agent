# Gradient-Direction-Aware Density Control for 3D Gaussian Splatting

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
The emergence of 3D Gaussian Splatting (3DGS) has significantly advanced Novel View Synthesis (NVS) through explicit scene representation, enabling real-time photorealistic rendering. However, existing approaches manifest two critical limitations in complex scenarios: (1) Over-reconstruction occurs when persistent large Gaussians cannot meet adaptive splitting thresholds during density control. This is exacerbated by conflicting gradient directions that prevent effective splitting of these Gaussians; (2) Over-densification of Gaussians occurs in regions with aligned gradient aggregation, leading to redundant component proliferation. This redundancy significantly increases memory overhead due to unnecessary data retention. We present Gradient-Direction-Aware Gaussian Splatting (GDAGS) to address these challenges. Our key innovations: the Gradient Coherence Ratio (GCR), computed through normalized gradient vector norms, which explicitly discriminates Gaussians with concordant versus conflicting gradient directions; and a nonlinear dynamic weighting mechanism leverages the GCR to enable gradient-direction-aware density control. Specifically, GDAGS prioritizes conflicting-gradient Gaussians during splitting operations to enhance geometric details while suppressing redundant concordant-direction Gaussians. Conversely, in cloning processes, GDAGS promotes concordant-direction Gaussian densification for structural completion while preventing conflicting-direction Gaussian overpopulation. Comprehensive evaluations across diverse real-world benchmarks demonstrate that GDAGS achieves superior rendering quality while effectively mitigating over-reconstruction, suppressing over-densification, and constructing compact scene representations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a fundamental flaw in the "densification" process of 3D Gaussian Splatting (3DGS)—the mechanism that decides where to add more detail. The original 3DGS method relies solely on the magnitude (or norm) of the position gradient (which can be seen as the "rendering error signal") while completely ignoring its direction. To resolve this fundamental issue, the paper proposes a Gradient-Direction-Aware density control framework (GDAGS). Its core idea is to quantify and leverage gradient directional information to make smarter decisions. The method consists of two main components: The Gradient Coherence Ratio (GCR) and The Nonlinear Dynamic Weighting System. The method not only resolves the blurring issue and improves rendering quality but, more importantly, it reduces redundancy, producing compact 3D models than AbsGS.

### Strengths
1. The paper is well-written and easy to follow. 
2. The work's originality lies in its more complete diagnosis of a known issue. While prior work like AbsGS had already identified the problem of gradient cancellation causing blur, this paper correctly points out that this is only half of the story; gradient amplification in aligned regions is an equally important problem that leads to model bloat. The proposed Gradient Coherence Ratio (GCR) is a novel and intuitive metric to directly measure this directional consistency. Using this metric to create a dual-purpose control system—one that simultaneously encourages splits in chaotic regions and suppresses them in stable ones—is a clever and well-motivated approach.

### Weaknesses
1. The authors correctly identify in their limitations section that the GCR metric may be unreliable in very sparse regions with little gradient information. However, this critical failure mode is only mentioned briefly and not explored empirically.
2. The method introduces new hyperparameters for its weighting function. Although a sensitivity analysis is provided in the appendix, the paper offers little intuition or practical guidance on how these should be set. This lack of guidance could make the method difficult for others to apply effectively to new and different scenes.

### Questions
1. To better understand the method's limitations, the authors should include a targeted experiment on a scene with known sparse viewpoints or textureless surfaces. A qualitative analysis showing how GDAGS behaves in these low-information areas compared to the baseline would be very instructive. Does it fail gracefully by simply not adding primitives, or does it make poor decisions? A discussion of potential fallback strategies in such cases would also strengthen the work.
2. The authors should report end-to-end training times and compare them against the key baselines. This is essential for a fair assessment of the trade-offs. It would also be helpful to include a brief profiling analysis that quantifies the specific overhead of the GCR computation step.

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
4

### Summary
This paper proposes GDAGS, a framework that addresses the over-densification, over-reconstruction, and memory inefficiency issues caused by the ill-posed densification mechanism in the original 3DGS, leveraging the directional consistency method and online dynamic weighting schemes. Experimental results show the efficacy of the proposed method.

### Strengths
1. The paper is well writen and easy to follow.
2. This paper targets a general and key component in the 3DGS pipeline and provides a simple yet effective solution.
3. The experimental results show promising performance improvements.

### Weaknesses
1. The proposed GDAGS introduces additional computational overhead during the optimization process. I believe it is necessary to report a comparison of training and testing times to better characterize the efficiency of the proposed method.

2. In Eq. (1), the variable \( i \) is not clearly defined. Does it refer to the Gaussian kernel? In addition, how is the number of views \( V \) determined in the experiments? Has the effect of different \( V \) values on performance been examined?

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
3DGS introduced an adaptive density control algorithm to grow the number of Gaussians to fit the underlying scene structure. However, it is relatively suboptimal in properly splitting and cloning Gaussians, an issue that even prior work such as AbsGS could not fully address. This paper identifies that this issue arises from: 1) gradient cancellation due to diverging sub-gradient directions, and 2) exaggerated gradients caused by the simple aggregation of absolute values of diverging sub-gradients in local regions. To capture the directional consistency of sub-gradients, the authors define the Gradient Coherence Ratio (GCR; Equation 5) and modulate gradients using a nonlinear weighting function defined in Equation 6. Using this criterion, they not only promote the splitting of large Gaussians with diverging sub-gradients but also suppress the cloning of small Gaussians with diverging sub-gradients. Experimental results demonstrate state-of-the-art scene reconstruction quality while maintaining lower memory consumption on standard benchmarking datasets.

### Strengths
This paper properly identifies the shortcomings of the prior method (AbsGS) and achieves the best control of the number of Gaussians during training by addressing this problem.

### Weaknesses
Please see Questions section for my major concerns.

Presentation issues:
- Math error in Equation 3. The expansion of $T_k$ is incorrect. It must be $\prod_{j=1}^{k-1} (1-\alpha_jG'_j(\textbf{x}'))$.
- Typo in line 273: $(1 − C_i)^a$ → $(1 − C_i)^p$.
- Clarify the unit of x-axis in Figure 5. It seems k (thousand).

### Questions
* Please provide the training duration in the experiments. This will help readers understand the training efficiency of the method.
* In Figure 3, some images show that GDAGS fails to reconstruct particular areas compared to vanilla 3DGS, which weakens the authors’ argument that GDAGS avoids over-densification and over-reconstruction issues. Do the authors have an explanation for why GDAGS renders these artifacts? For example, GDAGS produces noisy artifacts on the crown molding (top-left of the first-row image) and blobby artifacts in the area between trees and the sky (top-left of the second-row image).
* One major limitation is that the newly introduced hyperparameters $\alpha, \beta, p$ are tuned specifically for different scenes. They need to be searched heuristically to find the best trade-off between quality and efficiency (VRAM usage), which is cumbersome. Is there a learnable approach for these parameters?
* In Figure 5, the authors show the dynamics of the number of Gaussians associated with clone/split operations. According to the graphs, AbsGS tends to split Gaussians far more frequently than GDAGS, yet AbsGS retains poorer reconstruction quality than GDAGS according to Table 1. This seems counter-intuitive because generally, propagating more Gaussians should improve the reconstruction of complex structures. What is the authors’ explanation for why AbsGS has lower PSNR despite splitting Gaussians much more than GDAGS? In other words, what is the key factor that allows GDAGS to achieve the highest rendering performance without splitting as many Gaussians?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work trying to achieve a balance between over-reconstruction and over-densification which raise from gradient based adaptive control in classical 3DGS. The authors propose Gradient-Direction-Aware Gaussian Splatting (GDAGS), which introduces the Gradient Coherence Ratio (GCR) to quantify the directional consistency of view-space gradients and a nonlinear dynamic weighting mechanism that regulates Gaussian splitting and cloning based on gradient alignment.

### Strengths
1.	The proposed method achieves a good balance between performance and storage, aligns well with intuition, and enhances the overall usability of Gaussian Splatting models.
2.	The authors conduct experiments across three benchmark datasets and compare against a wide range of strong baselines (NeRF, 3DGS, AbsGS, Pixel-GS, etc.). The inclusion of ablation studies and sensitivity analyses provides convincing evidence for the method’s robustness and interpretability.
3.	The motivation of this paper is well articulated, and the proposed solution is intuitive. In particular, Figure 1 clearly illustrates the problems of over-reconstruction and over-densification that occur in 3D reconstruction under two extreme scenarios.

### Weaknesses
1.	The overall performance(especially the LPIPS metric) is highly influenced by the Hyper parameters(α、β、p) which raises concerns about the generalization of the method.
2.	Still about generalization ability. In section 4.3.4 the authors evaluate the proposed module combine with MCMC-3DGS and Compact-3DGS. However, noticeable performance changes appear mainly in the LPIPS metric and the SSIM metric on the Deep Blending dataset. Therefore, a qualitative analysis corresponding to these metric variations should be presented. 
3.	Although experiments show that GDAGS is superior to AbsGS and Pixel-GS but the analysis of the reasons for their performance differences in the paper is relatively insufficient. 
4.	While the GCR is well-defined, the paper lacks theoretical insights into how it affects optimization dynamics or convergence. The justification for its effectiveness is mostly empirical.

### Questions
1.	Why is the directionality measurement of GCR superior to the Pixel weighting mechanism of Pixel-GS?
2.	The nonlinear dynamic weighting model proposed in this paper is intuitive and effective. But why adopt the current form instead of the exponential weighting function among numerous nonlinear models? This point requires sufficient explanation and clarification.
3.	The comparison methods cited in the main text are up to conference works from 2024.  Since over-reconstruction and over-densification are widely discussed topics in the 3DGS community, have there been related works published in 2025?  If so, what are the advantages of this paper compared to those?

### Soundness
3

### Presentation
3

### Contribution
3
