# MEGS^{2}: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning

- Decision: Accept (Poster)
- Scores: 2, 4, 8, 6, 2

## Abstract
3D Gaussian Splatting (3DGS) has emerged as a dominant novel-view synthesis technique, but its high memory consumption severely limits its applicability on edge devices. A growing number of 3DGS compression methods have been proposed to make 3DGS more efficient, yet most only focus on storage compression and fail to address the critical bottleneck of rendering memory. To address this problem, we introduce MEGS², a novel memory-efficient framework that tackles this challenge by jointly optimizing two key factors: the total primitive number and the parameters per primitive, achieving unprecedented memory compression. Specifically, we fully replace the memory-intensive Spherical Harmonics with lightweight, arbitrarily oriented and prunable Spherical Gaussian lobes as our color representations. More importantly, we propose a unified soft pruning framework that models primitive-number and lobe-number pruning as a single constrained optimization problem. Experiments show that MEGS² achieves a 50% static VRAM reduction and a 40% rendering VRAM reduction compared to existing methods, while maintaining comparable rendering quality.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a memory-efficient Gaussian Splatting method for edge devices. It uses spherical Gaussians to replace the original spherical harmonics for better efficiency. It also involves a pruning process to eliminate the redundant Gaussian points. Experimental results show the proposed method can reduce VRAM compared to the original 3DGS.

### Strengths
1. This paper is well-organized and easy to follow. Language is good.
2. The proposed method can render comparable results to 3DGS.
3. The presentation is good. The demonstrated effect is impressive.

### Weaknesses
1. In the abstract, "Experiments show that MEGS2 achieves a 50% static VRAM reduction and a 40% rendering VRAM reduction ...", this sentence confuses me. What's static VRAM and rendering VRAM? There's no clear claim.

2. In lines 60-62, the authors claim, "However, these methods require decoding the full Gaussian parameters from a compressed
state before rendering, even resulting in a larger rendering memory than the 3DGS methods without compression.".  I am wondering if the proposed method doesn't need to decode the compressed Gaussian parameters for rendering? I did not find a clear description.

3. The authors claim the proposed spherical Gaussian is their contribution. However, the spherical Gaussian was already proposed by [1]. The authors do not present a clear discussion of it. I did not find a detailed description for arbitrarily-oriented SG. How to achieve it? Why is it better than orthogonal? Is there any theoretical analysis?

4. The proposed memory-constraint optimization is too straightforward. Please elaborate on the contribution. The formulation did not claim the relation between opacity $o$ and sharpness $s$.

5. To the proposed post-processing procedure, the authors claim they have a three-step strategy. However, I only find two: (1) removal according to low opacity, (2) by sharpness. Moreover, I do not think it is novel. It is also straightforward. There's no specific design in it.

6. As for experiments, the authors do not compare with state-of-the-art GS compression methods, like LocoGS[2], speedy-splat[3], LightGaussian[4], maskGaussian[5].


7. There's no visual comparison for the ablation study. I think it would be better to present it. There are no Gaussian numer comparisons. 
It would be better to show the training costs and storage memory. There are no detailed VRAM analyses for each Gaussian attribute.






[1] Wang Y, Chen S, Yi R. Sg-splatting: Accelerating 3d gaussian splatting with spherical gaussians[J]. arXiv preprint arXiv:2501.00342, 2024.

[2] Shin S, Park J, Cho S. Locality-aware gaussian compression for fast and high-quality rendering[J]. arXiv preprint arXiv:2501.05757, 2025.

[3] Hanson A, Tu A, Lin G, et al. Speedy-splat: Fast 3d gaussian splatting with sparse pixels and sparse primitives[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 21537-21546.


[4] Fan Z, Wang K, Wen K, et al. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps[J]. Advances in neural information processing systems, 2024, 37: 140138-140158.

[5] Liu Y, Zhong Z, Zhan Y, et al. Maskgaussian: Adaptive 3d gaussian representation from probabilistic masks[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 681-690.

### Questions
No code attached. I think it would be better if the authors released the code. 

Have the authors try 4DGS? In my opinion, it is quite easy to adapt the proposed method for 4DGS.

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
This paper presents a memory-efficient Gaussian Splatting framework that replaces high-dimensional spherical harmonics with lightweight spherical Gaussian lobes. It also introduces a unified pruning strategy for both Gaussian primitives and spherical Gaussians, resulting in improved pruning performance due to its regularization effect. Experimental results demonstrate that the proposed framework achieves substantial reductions in both static and dynamic VRAM usage while providing superior rendering quality.

### Strengths
- The paper effectively analyzes the key factors of large VRAM consumption and mitigates them using the proposed techniques.

- The use of spherical Gaussian verifies to be an efficient alternative for color representation, achieving meaningful parameter reduction while maintaining a comparable representation capacity to spherical harmonics.

- Experimental results show that the proposed method outperforms various 3DGS baselines in rendering quality while requiring only minimal VRAM consumption.

### Weaknesses
- The lack of technical novelty needs to be enhanced. The main performance gain for the memory reduction originates from the replacement of spherical harmonics with spherical Gaussian lobes. However, spherical Gaussians have already been applied to 3DGS in prior work, such as SG-Splatting. 

- Pruning solely based on the opacity value may lead to performance limitations, since the contribution of Gaussians is also influenced by other attributes, such as scales. 

- Despite the low VRAM consumption, this representation requires a relatively high storage demand compared to compact Gaussian approaches.

### Questions
- Could you compare the proposed pruning method with other pruning metrics from existing approaches, such as CompactGaussian [CVPR'24], EAGLES [ECCV'24], and LightGaussian [NeurIPS'24]?

-  Could a lower degree of spherical harmonics be more efficient than spherical Gaussian lobes? It could be helpful to compare the diverse SH degrees against the proposed representation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents MEGS2, a memory-efficient framework for 3D Gaussian Splatting (3DGS) that targets the rendering VRAM bottleneck rather than only the storage footprint. The authors propose two key ideas: (1) replacing high-order Spherical Harmonics (SH) color modeling with Spherical Gaussians (SG) for compact, local view-dependent representation, and (2) a unified soft pruning framework that simultaneously prunes redundant Gaussian primitives and per-primitive SG lobes under a total memory constraint, formulated as a constrained optimization problem solved by an ADMM-inspired method. Experiments show the effectiveness of the proposed method. The approach is further validated on both desktop and mobile devices, showing strong frame rates in a WebGL viewer.

### Strengths
- Replacing SH with arbitrarily oriented Spherical Gaussians is an effective solution to reduce per-primitive parameters. The formulation is physically intuitive and well integrated into the 3DGS framework.
- The paper provides detailed quantitative and qualitative results across multiple datasets and baselines.
- The WebGL-based real-time rendering results demonstrate real usability on mobile platforms. 
- The method’s ability to generalize across devices and maintain reasonable quality is a clear step toward deployable Gaussian-based rendering.

### Weaknesses
- Several important parameters are not explicitly described or justified. The choice of these hyperparameters may significantly affect performance and reproducibility.
- Scenes with highly specular highlights or dense transparency are acknowledged as future work but should be illustrated. The method’s performance on dynamic or large-scale scenes remains unknown.
- While the authors promise code release, the current text lacks explicit training schedules, optimization hyperparameters, and hardware runtimes, which hinders replication.
- The convergence and optimality properties of the ADMM-inspired procedure are not formally analyzed.

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

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
The paper presents MEGS^2, a 3D Gaussian Splatting framework that cuts both primitive count and per-primitive parameters to fix the rendering VRAM bottleneck. 
The paper replaces SH colors with lightweight, arbitrarily oriented spherical Gaussians (SG) so most Gaussians use only a few lobes. 
The paper’s method formulates primitive and lobe pruning as one memory-constrained optimization, reaching ~8× static and ~6× rendering VRAM compression over vanilla 3DGS, still close to SOTA quality. 

The paper backs its claims with extensive experiments on standard datasets (MipNeRF 360, etc), with qualitative evaluations. 
Ablation studies show different pruning scheme is outperformed by the method presented by the paper, unified soft pruning.

### Strengths
The paper provides following strenghts:
- Unlike storage-only methods, it reduces static and rendering VRAM (50% static, 40% rendering vs SOTA), so it runs on edge/mobile. 
- Usage of SG color model is prunable. SGs keep high-freq/view-dependent details with about half the params of SH and allow lobe-wise pruning.
- Unified soft pruning works better than sequential. ADMM-style optimization over opacity and sharpness finds a better quality–memory trade-off than “prune primitives then prune SH.”

### Weaknesses
The paper suffers from following weaknesses:
- Optimization pipeline is complex. Needs proxy variables, proximal projection, then post-processing + fine-tuning, so it’s harder to integrate than simple importance-pruning. 
- As stated in the limitation section, the paper mostly focuses on renderer-agnostic static memory. Paper explicitly leaves renderer-specific dynamic VRAM optimization “for future work,” so some engines may still hit a ceiling.
- As authors noted, performance on “highly complex highlights” needs more study, so scenes with extreme view-dependent effects may degrade first.

### Questions
- The authors argue SGs are “more amenable to pruning.” Can the authors quantify how many lobes are actually used per Gaussian at convergence, per scene?
- The optimization looks more complex than sequential pruning: how sensitive is it to the pruning budget? Do small changes lead to big quality swings?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes memory efficient gaussian splatting method by combining Spherical Gaussian and ADMM-inspired unified pruning. Noticed from related paper SG-Splatting, they found fixed orthogonal axes of SG lobes limit rendering performance, and they make it arbitrary oriented. Inspired from GuassianSpa, they optimize proxy of opacity and sharpness parameters simulatenously to sparsify the values. At the end of optimization of proxies, they prune primitives based on significance of the proxies. They found sequential pruning (prune gaussians first then prune parameters) shows poorer trade-off between primitive count and primitive parameters than unified pruning (prune gaussians and parameters together). To recover rendering quality degradation caused by pruning, they further finetune diffuse color based on color loss. Their experiment results show comparable rendering quality of their method against state-of-the-art method but significant save in VRAM, tested on standard benchmarking datasets.

### Strengths
This paper addresses the limited color representation capacity of SG-Splatting and further introduces opacity- and sharpness-based pruning, along with energy recovery of removed lobes, to reduce VRAM usage with maintaining high reconstruction quality.

### Weaknesses
I think the major issue is the inaccurate contribution claim of this paper. As discussed in Questions 1 and 5, the first contribution claim seems exaggerated, and their findings are not well supported by ablation experiments. Similarly, the second contribution claim is also inaccurate, as discussed in Question 2. Moreover, poor presentation of the method, such as missing information on model choices, makes it difficult for readers to fully understand their approach. Please refer to my Questions for detailed comments.

Presentation issue:
* I could not find the authors’ conclusion regarding model selection.
  1. In lines 306–311, the authors propose two distinct operators: **sharpness-based selection** and **range-based selection**. Which one is ultimately adopted in the final model and why? I could not find a statement clarifying this choice.
  2. Similarly, in lines 785–787, which criterion was finally chosen and why?: **magnitude-based selection** or **importance-based selection**?
* I could not identify the setting difference between the last two rows in Table 1. Please provide distinct method names (e.g., ours-setting_A and ours-setting_B) and explain them clearly in the appropriate section of the main text.
* Please include a citation for ADMM at line 234.

### Questions
* The first contribution claim in lines 97–99 makes it sound as if the authors are the first to propose substituting Gaussian Splats with Spherical Gaussians, which is not true, as this was already introduced by SG-Splatting. The authors should instead highlight the distinction between **fixed axes** and **arbitrarily oriented axes** in Spherical Gaussians. Emphasizing the reduction of per-primitive parameter count and VRAM savings is also inappropriate, since these advantages originate from SG-Splatting.
* The second contribution claim in lines 100–102 also seems inaccurate. The memory-constrained problem and its solution were originally introduced by GaussianSpa, and the authors simply extend that method by incorporating the sharpness of Gaussians to retain those with better expressive capacity. The authors should revise the statement of contribution 2 to accurately reflect their original findings.
* The VRAM reduction of the proposed method is primarily attributed to the smaller number of parameters used for encoding view-specific color and pruning. To provide supporting evidence for the pruning aspect, I suggest that the authors present both VRAM consumption and the number of Gaussians in Table 1. This will help readers better understand the pruning performance compared to pruning-only baselines such as LP-3DGS, Mini-Splatting, and GaussianSpa.
* The qualitative results in Figure 3 do not fully support the authors’ claim that "MEGS² delivers a cleaner and more complete reconstruction in Playroom" (line 415). When examining the ceiling area in the top center of the rendered images, MEGS² produces a somewhat speckled texture, whereas GaussianSpa and 3DGS yield cleaner textures. Additionally, the light switch in the bottom-left area appears blurrier in MEGS² compared to GaussianSpa.
* I am wondering about the effect of using arbitrarily oriented Spherical Gaussians. Please include an ablation study to analyze this effect. For example, the authors could test:
  1. The proposed method with **Spherical Harmonics (SH)**
  2. The proposed method with **fixed-axis Spherical Gaussians**
* Furthermore, I am curious why **GaussianSpa (SH → SG)** shows a quality drop (27.01dB in Table 2) compared to the original GaussianSpa (27.56dB in Table 1). This raises doubts about the justification for using Spherical Gaussians.

### Soundness
3

### Presentation
1

### Contribution
1
