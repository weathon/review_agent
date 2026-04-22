# Frequency-aware Dynamic Gaussian Splatting

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
We present \textbf{Frequency-Aware Dynamic Gaussian Splatting (FAGS)}, a novel approach to mitigating motion blur in 4D reconstruction, particularly under novel viewpoints. This blur stems from a fundamental spectral conflict in existing methods, which struggle to \textbf{balance  high-frequency rendering details with high-frequency motion.}
FAGS addresses this challenge with two key innovations. First, we introduce a frequency-differentiated Gaussian kernel that refines the alpha-blending process of 3D Gaussian Splatting. By adaptively classifying Gaussians into two types—a slowly varying kernel for smooth, low-frequency regions and a sharp-transitioning kernel for high-frequency boundaries—our method explicitly separates representation responsibilities, preserving fine details without sacrificing continuity.
Second, we propose a Fourier-Deformation Network that enhances motion expressiveness. This network employs high-frequency Fourier embeddings to capture diverse motion patterns by learning amplitudes across frequency components. To further improve accuracy, we integrate a frequency-aware gate in fusion module, which predicts and regulates the relative deformation of each Gaussian.
Extensive experiments on both synthetic and real-world 4D benchmarks demonstrate that FAGS significantly reduces motion blur and enhances structural details, achieving state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper points out the crucial limitations of previous 4D-reconstruction methods: Most of the previous literatures struggle to balance high-frequency rendering details with high-frequency motion, hence resulting in blurred details. This is mainly due to the limited expressiveness of the 3DGS in the dynamic scenes where the standard alpha-blending of 3DGS fails to express modified geometry along the time steps.

To this end, this paper suggests the novel technique from a frequency perspective where they provide additional learnable parameters into 4DGS pipeline such as $\lambda$, $\beta$ and *frequency-aware gate*, enhancing the expressiveness of 3DGS and successfully distinguishing the low-frequency gaussians and high-frequency gaussians.

Proposed technique contributes to performance improvement to some extent.

### Strengths
This paper addresses the meaningful problem where the limited expressiveness of 3DGS fails to recover the fine-grained details across the time steps due to the lack of clear separation between low-frequency gaussians and high-frequency gaussians. This statement is well introduced in the second paragraph of *Introduction part* which makes their motivation clear.

The solution suggested by the paper has novelty inside where they improve the expressiveness of 3DGS by introducing the additional learnable parameters from a frequency perspective, clearly separating the gaussians into high-frequency GS and low-frequency GS. Furthermore, ablation studies support the effectiveness of each part where Tab. 4 shows the quantitative improvements and Fig. 7 demonstrates the clear separation of the gaussians into low-frequency and high-frequency components.

### Weaknesses
1. **Marginal performance improvement** 

* It seems like the performance improvement in SSIM and LPIPIS is quite marginal (mostly +0.001 in both metrics) compared to the Grid4D while performance in PSNR is relatively clear. Furthermore, in *Trex* scene of Tab. 1, it seems that SC-GS achieves the best result in LPIPS metric while the paper indicates 4DGS and their own method achieves the best scores. 
* Why is LPIPS metric not reported in Neu3D dataset? Also, please report the average scores in Neu3D dataset.
* In Fig. 6, I failed to find the clear improvement  between Grid4D vs paper's methods, especially in 2nd and 3rd rows. In 2nd row, I think the both of Grid4D and paper's method fail to reconstruct the clear details.
* The combination of 3 rendering matrices compared to GT (PSNR, SSIM, LPIPS) is sometimes not sufficient to show the effectiveness of the method. It would be better to adopt additional metrics such as MUSIQ [a] and CLIP-IQA [b] to estimate the naturalness of the images since the paper points out the blurred reconstructed details from the previous methods.

2. **Room of improvement in ablation study**

* In Sec. 3.2, paper introduces two learnable parameters, $\lambda$ and $\beta$, in the alpha-blending part of the 3DGS to improve the expressiveness of the 3DGS. $\lambda$ controls the transparency gradient from the projection center, effectively separating the high-frequency GS and low-frequency GS, while $\beta$ additionally controls the differentiated regions. What will be happened if we remove the $\beta$? It would be better if the paper reports the performance when they only use one of them ($\lambda$ and $\beta$).  
* Tab. 4 presents the ablation study for each component of FAGS. However, more diverse combinations should be explored.

Considering the contributions and weaknesses of this paper, I would like to recommend the score between 6 to 8 which is around 7.

---

**References**

[a] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021.

[b] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2555–2563, 2023

### Questions
**Primary area**: Why is the primary area of this paper indicated as *generative model*?

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
4

### Summary
The authors propose Frequency-Aware Dynamic Gaussian Splatting, or FAGS. FAGS composes of several components: it first intrudes FDGK, which plays on modifying the Gaussian distribution of a GS primitive. FDGK divides the Gaussian distribution into three regions, and modulate the scale and bias of each region, such that the distribution is more peaky or flatter based on optimization. It then introduces additional encoding scheme for the temporal domain, i.e. encoding time into higher dimensions/frequency bands compared to the spatial domain, such that the MLP can disentangle motion better. Frequency gating and a loss in frequency domain is introduced, though their effectiveness is not very evident. Overall, FAGS outperforms previous approach in terms of standard metrics; however, it is unclear to me if this is due to better disentanglement of high frequency detail and motion, or simply a better Gaussian representation capacity.

### Strengths
1. FAGS outperforms existing methods on standard 4D reconstruction datasets, particularly on synthetic datasets. Real dataset performances have a smaller improvement, but that seems reasonable given potential noise in real datasets.
2. The FDGK formulation does seem to improve Gaussian representation capacity and leads to better fitting.

### Weaknesses
I have several issues with this paper, specifically concerning the writing (difficult to read, bold claims without evidence, etc.) and the necessity of multiple parts. I think overall the paper can be more focused on FDGK and better distinguish that contribution. 

1. It is hard to follow this work, multiple variables are not defined. E.g. in Eq. (2), g is not defined. D is not defined in Eq. (8), multiple abuse of notation, e.g., where "MLP" indicates multiple neural networks with different parameters. The math is unclear, especially Eq. (2). How is this equation related to "numerical fluctuations"? While the first part of the Equation makes sense, the second part, where r is said to be able to go to infinity, does not seem mathematically correct. We may be able to scale with r in limited region of the distribution, by certainly not the entire distribution, otherwise opacity can be scaled to infinity. Proper constrains should be stated. I have to reference prior work, e.g., DRK to double-check my understanding. The details on frequency-aware gate score is very sparse, it is unclear to me why this score can do the things that is claimed. All that is conveyed is that this score seems to come out of a neural network. 

2. Multiple claims seem to be overboard/abusing concepts that have similar words but not really related. The main claim: "FAGS can disentangle high freq. detail and high freq. motion" does not feel properly addressed by FDGK. The FDN introduces higher dimensional embedding to the temporal axis with Fourier basis - I am not sure if this is necessarily related to motion frequency rather than higher dimensional embedding leads to better performance. Similarly, it is unclear to me how the computing the amplitude spectra loss over an image leads to better frequency disentanglement; both FG and L_fre seem to contribute very little based on ablation. 

3. It is unclear to me why DRK is not a baseline at least in the ablation study. It seems like this work is closely related to DRK. Despite that DRK does not explicitly address 4D reconstruction, it should be straightforward to add DRK as a kernel representation without affecting the motion estimation network too much. Without such experiments, it is unclear to me if FDGK is a simpler variant of DRK with pre-defined regions of Gaussians to optimize. 

I overall think this paper has interesting contributions, and would give a borderline accept if FDGK/high frequency detail and motion disentanglement can be more grounded, which may require datasets that have some notion of what "high/low frequency motions" are. Given the current state of writing, and the (seemingly) unnecessary components, I vote for borderline reject and think the authors should significantly modify the writing.

### Questions
1. WRT the main claim: other than visual quality, is there anything else that proves the predicted motion is now "higher frequency" and more accurate, e.g., what is the "correct" motion frequency? My impression is that FDGK does improve the representation capacity of individual Gaussians and does improve visual performance; however, its relationship to motion is unclear. I.e., if FDGK is applied to standard 3D reconstruction, will it not improve performance?
2. Higher dimensional embedding to the temporal axis with Fourier basis: is this related to anything specific about Fourier space, or alternative embedding scheme, e.g. higher dimension hash embedding other implicit neural representation embedding, would achieve similar results?
3. Are FG and L_fre necessary? Why can frequency-aware gate score behave in the manner that is described? Based on ablation, FG and L_fre contributes very minimally.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Frequency-Aware Dynamic Gaussian Splatting, a 4D Gaussian-splatting framework aimed at reducing motion blur in dynamic novel-view synthesis. The authors argue that prior deformation-field-based dynamic Gaussian methods struggle with a *spectral conflict*: they must both (i) stack many soft Gaussians to reproduce high-frequency spatial details, and (ii) drive those same Gaussians along accurate, potentially fast and highly non-smooth temporal trajectories. As a result, the deformation network tends to bias toward globally smooth, low-frequency motion, which causes blurred boundaries and smeared fast-moving parts under novel viewpoints.

To address this, the paper introduces two technical components.

- Frequency-Differentiated Gaussian Kernel (FDGK): Instead of using a fixed isotropic alpha falloff per Gaussian, each Gaussian gets a learnable piecewise alpha modulation ψ(g) with parameters λ (controls sharpness vs smoothness of the opacity profile) and β (controls the spatial extent / boundary of that sharpness). Gaussians can thus specialize as high-frequency (sharp boundaries, for edges) or low-frequency (smooth support for broad regions), reducing the need to stack many Gaussians to recover detail and easing pressure on the deformation network.

- Fourier-Deformation Network (FDN): The deformation network fuses low-frequency spatiotemporal hash encodings with high-frequency temporal Fourier embeddings, whose amplitudes are predicted per Gaussian. A frequency-aware gate η modulates how strongly each Gaussian is allowed to move at each time step, suppressing motion in static/background regions while permitting rapid, localized motion where needed. A Fourier-domain loss on amplitude spectra further emphasizes fitting high-frequency components.

Experiments on D-NeRF, HyperNeRF, and Neu3D show consistent improvements in PSNR / SSIM / LPIPS / MS-SSIM over strong baselines such as 4D-GS, DeformGS, SC-GS, and Grid4D, especially around fast-moving fine structures (fingers, tools, flames). Visualizations of per-Gaussian motion trajectories and ablations suggest that FDGK + FDN reduce drifting / smearing and yield sharper reconstruction of both spatial detail and rapid motion.

### Strengths
- Clear motivation (motion blur in dynamic GS): The paper focuses on a highly visible failure mode in existing dynamic Gaussian splatting pipelines: smeared or drifting fine details (e.g., fingertips, blades, flames) in novel-view renderings. This is both practically important and easy to judge qualitatively.
- Theoretically solid contributions: Frequency-Differentiated Gaussian Kernel (FDGK) allows each Gaussian to learn its own alpha profile via λ (sharp vs smooth) and β (boundary span), which lets some Gaussians specialize in high-frequency edges without needing dense stacking, while others handle smooth regions. Fourier-Deformation Network injects explicit high-frequency temporal bases and modulates each Gaussian’s deformation strength with a learned gate η, which is a reasonable mechanism to retain fast, local, non-smooth motion for moving parts while not injecting jitter into the static background.
- Analysis of parameter distributions: The supplementary shows how λ splits Gaussians into high-frequency vs low-frequency groups during training, and how β adapts boundary spans. The gate η distribution highlights that not all Gaussians become fully dynamic. This supports the claim that the method self-organizes a frequency-aware decomposition.

### Weaknesses
- Core causal story is qualitative, not rigorously established.
The paper’s central narrative is that existing methods blur because they cannot jointly model high-frequency spatial detail and high-frequency motion, and that FDGK+FDN resolves this spectral conflict. While the visual evidence is compelling, there is no direct quantitative analysis of temporal frequency content (e.g., Fourier spectrum of per-Gaussian trajectories before/after, or a measure of per-point temporal curvature). Maybe Fig.10 tries to display this point, but there is no sufficient explanation of Fig.10. The argument remains largely anecdotally supported via qualitative trajectory plots and better metrics.
- Definition of “high-frequency motion” is not clear.
The method heavily leans on the dichotomy of “low-frequency global motion” vs “high-frequency local motion,” but never formalizes it (e.g., in terms of time derivatives or spectral energy bands), even though it motivates the design of FDGK, the Fourier embedding, and the gating mechanism. A clearer mathematical definition would make the framing more than just intuitive storytelling.
- Fourier-Deformation Network assumes a quasi-periodic temporal structure.
Modeling trajectories as Fourier bases with learned amplitudes implicitly assumes motions can be expressed as superpositions of (possibly periodic) sinusoids. From the videos in supplementary materials, most the the shown case can be figured out certain periodic motion pattern. Real-world motion like sudden stops, impacts, or non-repetitive gestures may violate this assumption. The paper does not explore failure cases or quantify when Fourier modes become insufficient.
- Clarity / naming / polish.
The paper alternates between **FAGS** and **FDGS**. Fig.10 lacks an explanation. The gating equation (Formula 8) does not have a sufficient explanation. These presentation issues hurt readability.

### Questions
These are questions where a rebuttal could meaningfully improve my score:

- Can you quantitatively define/measure high-frequency motion?
For example, could you report, for baseline vs. your method, the temporal power spectrum or second-order temporal finite-difference energy of a set of tracked Gaussians on fast-moving parts (e.g., fingertips in JumpingJacks)? Right now, high-frequency motion is central but informal.

- Can you give a more concrete analysis supporting the “spectral conflict” claim?
In Fig.12, you show that baseline Gaussians drift to other regions to satisfy both detail fitting and motion alignment, which you argue leads to blur. Could you quantify how often this cross-region drift happens, and how it correlates with blur artifacts in novel views? This would turn the nice qualitative story into measurable evidence.

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
The authors attribute motion blur in prior dynamic Gaussian splatting methods to the inherent difficulty of simultaneously modeling high-frequency rendering details and high-frequency motion. To address this issue, they propose a deformation-field-driven framework that explicitly separates Gaussians into low- and high-frequency components, allowing each group to specialize in reconstructing its respective part before merging. Central to this framework is a Fourier-Deformation Network, which leverages Fourier embeddings to effectively capture fine-grained high-frequency local deformations while suppressing redundant motion in low-frequency regions. In addition, the method introduces a frequency-differentiated Gaussian kernel that adaptively sharpens or smooths Gaussians via learnable parameters, enabling clear separation between high- and low-frequency behavior. Extensive experiments on dynamic scene datasets demonstrate the effectiveness of the approach, supported by thorough ablation studies that validate each proposed component.

### Strengths
1. The paper proposes a deformation-field-driven method that is compatible with most existing dynamic 3D Gaussian Splatting approaches and it has a good theoretical explanation of solving motion blur.
2. The introduction of a frequency-aware deformation field provides a clear and effective strategy for mitigating motion blur in dynamic scenes, especially under novel views.
3. Experiments demonstrate that the proposed method reduces motion blur compared to prior dynamic Gaussian Splatting methods.

### Weaknesses
1. The paper lacks a clear analysis of the computational overhead introduced by the frequency-aware module, such as runtime or memory usage, especially in comparison to existing baseline methods (In your pipeline, it has 4 MLPs. I think it's necessary to show the training time and evaluation time in your paper).
2. Although the approach shows that it reduces motion blur with normal input, the paper does not include comparisons with existing approaches that mainly address deblurring. Moreover, it does not evaluate on datasets with motion blur to validate its effectiveness under such conditions.
3. The description of the formulas are not so clear, e.g. you should clarify the 'g' in expression(2)~(4).

### Questions
1. Could you please offer me the training time and evaluation time of the baselines in paper? And could you please tell me how many layers in your MLPs? 
2. Belowing current papers are focusing on dynamic, could you please evaluate your model on blurring dataset to see whether it can reduce heavy motion blur?
3. Could you select one baseline from the following list of papers and then compare both time metrics and result metrics between that baseline and your approach?


[1]. BARD-GS: Blur-Aware Reconstruction of Dynamic Scenes via Gaussian Splatting (Deblurring Method + Blurring Dataset)

[2]. Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video (Deblurring Method)

[3]. MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video (Deblurring Method)

### Soundness
3

### Presentation
2

### Contribution
3
