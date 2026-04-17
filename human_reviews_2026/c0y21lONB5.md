# $\texttt{GEM}$: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction

- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Cryo-electron microscopy (cryo-EM) has become a central tool for high-resolution structural biology, yet the massive scale of datasets (often exceeding 100k particle images) renders 3D reconstruction both computationally expensive and memory intensive. Traditional Fourier-space methods are efficient but lose fidelity due to repeated transforms, while recent real-space approaches based on neural radiance fields (NeRFs) improve accuracy but incur cubic memory and computation overhead. We introduce $\texttt{GEM}$, a novel cryo-EM reconstruction framework built on 3D Gaussian Splatting (3DGS) that operates directly in real-space while maintaining high efficiency. Instead of modeling the entire density volume, $\texttt{GEM}$ represents proteins with hundreds of thousands of compact 3D Gaussians, each parameterized by only 11 values. An efficient implementation further restricts gradient computation to 3D Gaussians that contribute to each pixel, substantially reducing both memory footprint and training cost. On standard cryo-EM benchmarks, $\texttt{GEM}$ achieves up to $48\times$ faster training and $12\times$ lower memory usage compared to state-of-the-art methods, while improving local resolution by as much as $38.8\\%$. These results establish $\texttt{GEM}$ as a practical and scalable paradigm for cryo-EM reconstruction, unifying speed, efficiency, and high-resolution accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper adapts 3D Gaussian Splatting to real-space homogeneous reconstruction in cryo-EM. GEM generally exceeds the local and global resolution (measured via Fourier Shell Correlation) of CryoSPARC and CryoDRGN, on representative proteins.

### Strengths
The method is a natural combination of 3D Gaussian Splatting, which works well in computer vision, to cryo-EM homogeneous reconstruction. Quantitative results are encouraging compared to strong baselines, and the implementation is computationally efficient.

### Weaknesses
Most of these weaknesses are presentation related; they are written roughly in order of appearance in the paper:
- The input to GEM is not entirely clear until the bottom of page 3, where it is specified that the CTFs, poses, and translations are assumed known. This context should be mentioned in the introduction.
- The description of how the Gaussians are initialized is lacking in details. The paper states that the Gaussians are initialized randomly, but doesn’t specify the sampling distribution for the Gaussian means, rotations, scales, and densities. As 3DGS is sensitive to initialization, this is important information to include.
- Line 100-101 claims that Fourier-space cryo-EM reconstruction methods incur information loss because they “involve repeated FFTs and interpolation steps and operate on band-limited representations.” What is the evidence for this claim of information loss? Since there is presumably limited resolution on the cryo-EM detector leading to an inherent bandlimit in the measurements themselves, it’s not obvious why FFTs with interpolation on a bandlimited representation would incur any further information loss. It would also be good to specify what the “bandlimited representation” is–does this refer to a voxel grid?
- Line 115 argues that a benefit of 3DGS as a representation is that the “gradients remain localized to the subset of primitives that actually contribute to each pixel.” While it is true that Gaussians satisfy this desirable property, a sparse voxel grid would have the same property. The same is true for the properties marked (i) and (ii) on lines 124-126.
Line 128 mentions joint optimization over density and “imaging latents” but doesn’t specify what these latents are. I would expect these might refer to CTF or pose, but line 161 says that both of these are assumed known so it’s not clear what remaining latents need to be jointly optimized over.
- Lines 162-167 explain the difference between Fourier-space and real-space reconstruction. Since this is part of the method section, it is important to specify that the proposed method is real-space; ideally the description of Fourier-space reconstruction would not be in the methods section as it could confuse a reader as to which method is used here.
- There are some typos, mostly repeated and missing words/phrases but also some typos at the letter level. See lines 191, 192, 195, 198, 271, 304, 360.
- Figure 3a is confusing/misleading. It makes it look like you start with a known protein density and use that to initialize the Gaussians. But in reality the Gaussians are initialized randomly and then optimized to fit the density.
- The paragraph surrounding equation 8 describes efficient gradient computation as part of the method. While this is important for computational and memory efficiency, it is not clear what (if any) aspects of this computation are novel to GEM versus reused from 3DGS. For example, equation 8 describes thresholding each Gaussian into an ellipsoid to enable skipping some Gaussians that do not meaningfully influence a given pixel projection; this is exactly the same thresholding that 3DGS does for the same purpose. The only difference I am aware of is that 3DGS uses perspective projection whereas cryo-EM uses orthographic projection, but that does not seem to be the focus of this section so I am left wondering what is the novelty in the efficient computation.
- Lines 415-417, and line 446, describe the Nyquist-Shannon resolution limit as based on the voxel size of the protein density map. I have two questions about this. (1) What voxels are being discussed? The proposed method does not use voxels, nor is there a ground truth voxel grid that is mentioned. (2) Line 446 claims that GEM’s average local resolution is better than the Nyquist-Shannon limit. How is this possible?
- Section 4.4 introduces FSLC, but doesn’t really explain how it differs from GSFSC. From the names I gather that the difference is comparison on Fourier slices instead of Fourier shells, but the reader shouldn’t need to infer.
- Section 4.5 describes an ablation study involving isotropic scaling, in which S_i = s_i I. However, instead of allowing s_i to be optimized over (which would still enforce isotropic scaling since s_i is scalar), this ablation study forces s_i to equal the distance between the i’th Gaussian and its nearest neighbor. I think this would be a more informative ablation study if s_i were allowed to optimize, while still being enforced to be isotropic.

### Questions
Please refer to questions embedded in weaknesses.

### Soundness
3

### Presentation
2

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
The paper presents GEM, a new framework for efficient and accurate 3D reconstruction in cryo-electron microscopy (cryo-EM) using 3D Gaussian Splatting (3DGS). Unlike traditional Fourier-based methods that lose information during repeated transformations, and neural field–based real-space methods that suffer from cubic memory and computation costs, GEM explicitly models protein density as a compact set of 3D Gaussians, each parameterized by 11 values. This representation enables localized gradient updates, parallel rasterization, and selective computation of only contributing Gaussians, dramatically reducing resource demands.

### Strengths
1. This paper presents an interesting and timely contribution, offering solid theoretical derivations for the application of 3D Gaussian Splatting (3DGS) to cryo-EM homogeneous reconstruction. The motivation for adopting 3DGS in this domain is sound: it provides an explicit and efficient real-space representation that avoids the information loss of Fourier-based methods and the heavy computational cost of neural-field models. This is a promising and original direction that meaningfully bridges computer graphics advances and structural biology, enabling high-quality protein reconstruction without sacrificing training speed.

2. The experimental protocol is comprehensive and carefully designed. The authors benchmark GEM across multiple standard cryo-EM datasets and report not only global resolution metrics (GSFSC) but also local resolution and Fourier Slice Correlation (FSLC) analyses. This multi-faceted evaluation demonstrates the method’s robustness, efficiency, and generalizability.

### Weaknesses
## Major Weaknesses

1. **Clarity and Writing Quality.** The paper’s writing is often rough, with numerous grammatical errors and typos that hinder readability and clarity. Several technical explanations are hard to follow, particularly the definitions of rays and the sentence spanning L204–L205, which require clearer formulation. A thorough language revision is essential to improve accessibility and ensure technical accuracy.
2. **Unclear Necessity of Gaussian Sorting.** The decision to sort 3D Gaussians along the projection axis (Sec. 3.2) is not well justified. In principle, sorting should not affect the final result of the projection integral, yet it introduces nontrivial computational overhead, especially in CUDA kernels. The authors should explain the rationale behind this step—e.g., whether it serves numerical stability, differentiability, or rasterization ordering—and ideally show an ablation demonstrating that removing sorting does not harm reconstruction quality.
3. **Inconsistent Quantitative and Qualitative Results.** The reported quantitative results (Tables 2 and 3) appear unusually strong and potentially inconsistent with the qualitative reconstructions shown in Figures 5 and 6. For example, the sharp improvement in GSFSC for EMPIAR-10076 near 4–5 Å seems atypical and lacks discussion. Meanwhile, the visual reconstructions in Figure 6 appear smoother and less detailed than those from CryoDRGN or CryoSPARC, especially in α-helix regions. This suggests that GEM may underfit the true density due to the strong structural prior imposed by 3DGS (insufficient number of 3DGS may cause this). A useful diagnostic would be to increase the number of Gaussians (e.g., from 1 k to 10 k) and analyze whether this reduces oversmoothing. Additionally, experimenting with more advanced loss functions—such as those used in RELION or other likelihood-based formulations—could help recover finer local features. Overall, the current performance claims are not fully convincing without further analysis of these discrepancies.
4. **Lack of Details on the Number and Impact of 3D Gaussians.** The paper does not specify how many 3D Gaussians are used in training or how this parameter affects speed, memory usage, and reconstruction quality. Since this is central to both the efficiency and fidelity of GEM, the authors should include an ablation or scaling study to clarify this relationship.

## Minor Weaknesses

1. L165: The slice operation in Fourier space still requires instantiating the entire 3D density volume because the slicing orientation is unknown a priori.

2. L192: “density-based, density-based” → should be “density-based.”

3. L195: The phrase “Explicit Integration for XXX” appears incomplete and should be clarified.

4. L204–L205: The sentence is difficult to interpret, and the definition of r is missing, despite being fundamental to the imaging model—please rewrite and define it formally.

5. L243: The phrase “CTF-corrected” should be replaced with “CTF-corrupted” or “CTF-applied,” as no correction occurs in this step.

6. L198: Typo — “3D 3D” → “3D.”

7. L721: Variables x^hat and p^hat are not defined in the Appendix and should be explicitly introduced, as they are used in the projection derivation.

### Questions
I have raised my questions in the main weakness sections.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduced a new way, using Gaussian splatting, to reconstruct 3D density maps from cryo-EM particles with given poses. The authors demonstrated better computational efficiency and better metrics (mostly GS-FSC) using 3DGS compared to other NeRF-based methods.

### Strengths
- The paper is well written and the figures are very clear and concise.
- The introduction of 3DGS into cryo-EM reconstruction is interesting and the improvement of the computational efficiency is promising.

### Weaknesses
- The major weakness of this paper is the evaluation. To evaluate and show that GEM can achieve better results compared to other methods, the paper heavily used metrics (including GSFSC, local resolution, FSLC) that assess the self consistency between the two half maps. This is wrong. Here the methods themselves, will lead to spurious correlation between the half maps due to the induction bias of the methods. Therefore, GSFSC, FSLC and local resolution are all hacked and unreliable. This can be reflected in the results by: 1) GSFSC curves never reaches zero and even increase by a lot at very high frequency (Fig. 5), 2) Final resolution (0.143 cutoff) almost reach Nyquist while the density maps do not support such high resolution, 3) Local resolutions are severely overestimated while the maps lack structural details compared to other methods (Fig. 6), 4) even when the particle poses are the same, FSLC are quite different (Fig. 7). All of these show that half maps based metrics are all over estimated.

- The author should also compare 3DGS with the traditional reconstruction method using (weighted) back-projection (WBP) in Fourier space. The comparison with cryoSPARC is unfair, since both ab initio and homogenous refinement in cryoSPARC refine particle poses, while other methods do not. The easiest way to compare is reconstruct only with given poses. In RELION, you can use the `relion_reconstruct` command. Since WBP is simply performing back-projection with accumulation, computation can be easily paralleled across particles and it is usually very fast even without a GPU.

- Given the previous points, the motivation of using 3DGS into cryo-EM, although interesting and insightful, seems very unclear to me. 

Minor: In Fig. 6, the threshold level for EMPIAR-10005 map is too low (probably also 10049 as well).

### Questions
- Just for clarification purpose, for both cryoDRGN and GEM experiments, the poses of the particles are given, correct? Where do they come from?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
GEM introduces a novel method for 3D cryo-EM reconstruction by adapting the 3D Gaussian Splatting (3DGS) technique. This explicit, point-based representation offers high accuracy while significantly improving efficiency and memory consumption compared to O(N^3) real-space methods like NeRF, making large-scale structural biology more feasible.

### Strengths
It proposes GEM, a novel application of 3D Gaussian Splatting (3DGS)—a technique recently popularized in computer graphics and vision for novel view synthesis—to the challenging domain of 3D cryo-EM reconstruction. Adapting this explicit, point-based rendering system to the parallel-ray projection geometry of cryo-EM is a highly creative methodological step. The significance is substantial: the paper directly addresses the major limitations of current state-of-the-art reconstruction methods. Traditional Fourier-space methods are fast but often sacrifice fidelity due to repeated transforms, while recent real-space methods (like those based on Neural Radiance Fields, NeRF) achieve high accuracy but incur cubic complexity (O(N^3)) in both memory and computation, making them impractical for the massive datasets common in cryo-EM. GEM promises to break this accuracy-efficiency deadlock. The quality of the method, particularly the adaptation of the 3DGS projection model to cryo-EM's parallel rays, appears sound, and the results on datasets like EMPIAR-10005 are promising. The clarity of the paper is good, clearly outlining the mathematical basis for the projection.

### Weaknesses
A substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals.

Limited Experimental Scope for Generalization: The primary weakness is the limited breadth and depth of the experimental evaluation. While results on a well-known dataset like TRPV1 (EMPIAR-10005) are shown, the evaluation must be expanded to demonstrate the generalizability of GEM. Specifically, the paper is missing results on:

Complexes with different symmetry types (e.g., helical, octahedral) beyond simple C-symmetry or no symmetry.

Datasets spanning a wider range of resolutions (from very low to near-atomic).

Very large complexes to definitively prove the claimed computational advantage over O(N^3) methods.

Explicit Representation Limitations: 3DGS is an explicit, discrete representation using a set of anisotropic Gaussians. The paper does not sufficiently discuss how this representation handles the continuous nature of the electron density map, especially at sub-nanometer resolutions where fine-grained features like side-chain density need to be accurately modeled. This discrete nature may introduce discretization artifacts or fail to capture the smoothness of the underlying density compared to implicit representations like NeRFs.

Comparison Rigor: The comparison against existing methods needs to be more robust. The paper should include detailed comparative metrics, not just for final resolution (FSC), but also for reconstruction speed (wall-clock time), memory footprint, and local resolution variability compared to both a leading Fourier-based method (e.g., RELION, cryoSPARC) and the state-of-the-art NeRF-based approaches.

### Questions
### Computational Complexity
Please provide a theoretical and empirical analysis of the time and memory complexity of GEM. Specifically, how does the number of Gaussians (M) scale with the size and resolution of the map, and how does the overall complexity compare to the O(N^3) NeRF methods and O(NlogN) Fourier methods?

### Gaussian Initialization 
How sensitive is the final reconstruction to the initialization of the 3D Gaussian set? What method is used to control the density (number of Gaussians) during optimization, and how is the over-splitting or under-splitting of Gaussians prevented? Showing a result of reconstruction quality vs. number of Gaussians would be highly informative.

### Handling of Anisotropy and Local Resolution 
Cryo-EM maps often exhibit anisotropic resolution. How does the adaptive nature of the anisotropic 3D Gaussians explicitly help in modeling regions with poorer local resolution (e.g., flexible domains) versus rigid, high-resolution core regions? Can GEM provide a per-Gaussian confidence metric analogous to local resolution maps?

### Contrast and Background Modeling 
How is the background solvent and noise modeled in this 3DGS representation? Since the Gaussians represent the object density, is there a risk that low-contrast regions of the molecule are poorly represented or pruned during optimization?

### Soundness
3

### Presentation
3

### Contribution
3
