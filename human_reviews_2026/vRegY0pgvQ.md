# Mobile-GS: Real-time Gaussian Splatting for Mobile Devices

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6, 6

## Abstract
3D Gaussian Splatting (3DGS) has emerged as a powerful representation for high-quality rendering across a wide range of applications.
    However, its high computational demands and large storage costs pose significant challenges for deployment on mobile devices. 
    In this work, we propose a mobile-tailored real-time Gaussian Splatting method, dubbed Mobile-GS, enabling efficient inference of Gaussian Splatting on edge devices.
    Specifically, we first identify alpha blending as the primary computational bottleneck, since it relies on the time-consuming Gaussian depth sorting process. 
    To solve this issue, we propose a depth-aware order-independent rendering scheme that eliminates the need for sorting, thereby substantially accelerating rendering.
    Although this order-independent rendering improves rendering speed, it may introduce transparency artifacts in regions with overlapping geometry due to the scarcity of rendering order. 
    To address this problem, we propose a neural view-dependent enhancement strategy, enabling more accurate modeling of view-dependent effects conditioned on viewing direction, 3D Gaussian geometry, and appearance attributes. 
    In this way, Mobile-GS can achieve both high-quality and real-time rendering.
        Furthermore, to facilitate deployment on memory-constrained mobile platforms, we propose first-degree spherical harmonics distillation, a neural vector quantization technique, and a contribution-based pruning strategy to reduce the number of Gaussian primitives and compress the 3D Gaussian representation with the assistance of neural networks. 
    Extensive experiments demonstrate that our proposed Mobile-GS achieves real-time rendering and compact model size while preserving high visual quality, making it well-suited for mobile applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to run 3D Gaussian Splatting representations on mobile devices. To this end, the authors first identified that sorting consumes a large portion of the computational cost. Therefore, the paper proposes a sort-free algorithm with lower computational costs compared to other existing sort-free methods. This is combined with view-dependent features to mitigate artifacts. To make the representation even more lightweight, the authors introduce color distillation, vector quantization, and pruning. The experiments cover both desktop GPUs and mobile devices, evaluated with both numerical metrics and user preference studies. With all these components, the results suggest that the proposed method increases FPS and decreases storage and peak memory, all while maintaining representation quality.

### Strengths
- Strong performance. The proposed method shows strong performance over baselines and is well-evaluated on mobile devices. In addition, it provides thorough ablation studies that help the reader understand the role and effect of each component.
- The paper provides a runtime analysis that identifies how expensive certain operations are, which operations are bottlenecks, and how these can be reduced for mobile devices. It also shows the resulting share of operations after removing sorting.
- Beyond numerical metrics, the inclusion of a user study strengthens the performance claims from a user's perspective.

### Weaknesses
- The sort-free method seems to be a core part of this paper's novelty. However, a critical equation for the weighting term (Eq. 3) is not supported by or provided with an ample description of its underlying idea or theoretical grounding. The paper would also benefit from a direct comparison of the equations from different sort-free methods. Although there are descriptive paragraphs in the supplementary, the relationship between existing methods is not clearly shown at the equation level.
- Although distillation is an important part of this method, the necessity of a teacher model is not fully justified, nor is the setup adequately explained. It's unclear why this method requires distillation unlike other methods. Since it requires much longer training, it would be helpful if the paper explained why the current comparison (directly comparing "from scratch" methods with the proposed method that additionally requires a teacher model) is fair.

### Questions
- **Weighting term (Eq. 3)**: Could the authors provide the theoretical motivation or intuition behind the weighting term in Eq. 3? Including comparisons at the equation level would help readers better understand the context and the novelty of this component.
- **Necessity of distillation**: Is the distillation step truly necessary for achieving strong performance? Since the proposed depth-aware, order-independent rendering pipeline could, in principle, be trained from scratch, it is unclear why the method depends on distillation rather than following the same training approach as the baselines.

If these points are adequately addressed, I would be willing to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel method called Mobile-GS, which enables the deployment of the Gaussian Splatting (GS) method on mobile devices and achieves real-time rendering performance. The authors analyze the limitations of GS rendering and identify alpha blending as the main bottleneck. To address this issue, they propose a depth-aware, order-independent rendering scheme that eliminates the need for sorting. In addition, they introduce a neural view-dependent enhancement strategy to mitigate rendering artifacts. Post-processing techniques such as distillation and quantization are also employed to achieve efficient rendering on mobile devices.

### Strengths
1. The authors provide deep insights into the rendering strategy of 3DGS and effectively analyze its bottlenecks.
2. The proposed depth-aware order-independent rendering approach is interesting and appears novel.
3. The use of quantization and pruning methods successfully enables the deployment of 3DGS on mobile devices.

### Weaknesses
1. Do the authors analyze the precision or deviation of the depth-aware rendering theoretically? It would be helpful if they could provide a mathematical analysis or proof, along with additional exploratory experiments.
2. The reviewer would like to know whether neural vector quantization and pruning affect rendering quality. Could the authors provide evaluation results rendered on mobile devices and compare them with other methods?
3. What is the time cost of the proposed approach? It would strengthen the paper if the authors could provide this metric and compare it with existing methods.

### Questions
Please refer to Weaknesses.

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
4

### Summary
Mobile-GS introduces the first real-time 3D Gaussian Splatting framework optimized for mobile GPUs by eliminating costly depth-sorting through a depth-aware, order-independent rendering scheme.
Compared with the prior sorting-free Gaussian representation (SortFreeGS), Mobile-GS models view-dependent opacity and rendering weights in an implicit and lightweight manner, reducing the per-Gaussian parameter footprint. It further integrates spherical-harmonics distillation, neural vector quantization, and contribution-based pruning to enhance compactness and efficiency.
The method achieves ~125 FPS on a Snapdragon 8 Gen 3 device with only 4–5 MB of storage, while preserving visual fidelity comparable to full 3DGS.

### Strengths
1. Real-time mobile performance: Demonstrates the first 3D Gaussian Splatting system achieving real-time rendering on mobile GPUs such as Snapdragon 8 Gen 3.
2. Order-independent efficiency: The proposed depth-aware order-independent rendering removes the costly sorting step, significantly improving runtime without significant quality loss.
3. Implicit view modeling: Replaces explicit per-Gaussian weights and opacity with a shared lightweight MLP, enabling stable training and reduced parameters compared with SortFreeGS.
4. Compact design: Through spherical-harmonics distillation, neural vector quantization, and contribution-based pruning, the model compresses storage from hundreds of MB to only ~4–5 MB while maintaining comparable visual fidelity to 3DGS.

### Weaknesses
1. Extended training time: The use of a pre-trained teacher model for spherical-harmonics distillation doubles the overall training iterations, increasing computational cost.

2. Complex weighting formulation: The depth-aware weighting term (Eq. 3) appears empirically designed and lacks clear theoretical justification or ablation on its components.

3. Missing key baseline: The paper does not include a direct quantitative comparison with SortFreeGS in Table 2, which should serve as the most relevant baseline for this work.

4. Incomplete dataset coverage (minor): Several Mip-NeRF 360 scenes are missing from the evaluation; including them would strengthen the completeness and reliability of the results.

### Questions
1. Pruning mechanism:
 This paper evaluates the contribution of each Gaussian primitive using scale and opacity, while prior works often rely on gradient-based importance or learnable pruning masks. Could the authors discuss the advantages and disadvantages of their design compared to these alternatives, particularly in terms of stability, computational efficiency, and adaptivity during training?
2. Choice of teacher model:
 The framework employs a pre-trained teacher model for spherical-harmonics distillation. Could the authors clarify the motivation behind this choice and whether alternative teacher configurations (e.g., vanilla 3DGS) would impact performance or training cost?
3. Potential use of the teacher model for initialization:
 Since the framework already depends on a teacher model, could this model also be leveraged to provide better Gaussian initialization or coverage, potentially reducing training time?
4. Initialization source:
 Are the initial Gaussian primitives derived from COLMAP point clouds or the teacher model’s reconstructed points?

### Soundness
3

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
4

### Summary
This paper proposes a mobile-friendly 3D Gaussian Splatting pipeline that removes depth sorting via depth-aware order-independent rendering, then recovers quality with a lightweight, view-dependent enhancement MLP. It further compresses the representation using first-degree SH distillation, neural vector quantization, and contribution-based pruning to cut storage to a few MB while keeping fidelity. Experiments report >100 FPS on Snapdragon 8 Gen 3 and >1k FPS on RTX 3090 with competitive quality versus 3DGS and recent lightweight baselines.

### Strengths
- This paper claims per-tile sorting as the dominant bottleneck and introduces a simple, parallelizable order-independent blending scheme to remove it.
- A small view-conditioned MLP effectively suppresses transparency/occlusion artifacts that arise from sorting-free compositing.
- The compression stack (first-degree SH distillation + neural vector quantization + contribution-based pruning) is complementary and yields strong storage reductions with limited quality loss.
- The evaluation is extensive, includes ablations/runtime breakdowns, and demonstrates impressive reported throughput on Snapdragon 8 Gen 3.

### Weaknesses
- The novelty relative to contemporary sorting-free methods (e.g., SortFreeGS, stochastic/OIT-style splatting) is incremental and would benefit from a deeper theoretical or empirical comparison.
-  Some hyperparameters, such as pruning thresholds/schedules, codebook sizes and SH-order trade-offs need further analysis.
- The related works on network design and pruning should be added.

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Mobile‑GS, a 3D Gaussian Splatting pipeline designed for real‑time rendering on mobile devices. The key contributions include: 

1) Depth‑aware order‑independent rendering (OIR) that removes near‑to‑far sorting by blending all Gaussians affecting a pixel with a depth/scale‑modulated weight; the weight includes an MLP‑predicted, view‑dependent factor ϕ. A small neural view‑dependent opacity/weighting module combats transparency artifacts that arise from dropping strict alpha compositing.

2) Compression for mobile: (i) first‑degree SH distillation from a teacher (Mini‑Splatting) to reduce color parameters, (ii) neural vector quantization using sub‑codebooks plus tiny decoders for diffuse/view‑dependent SH components, and (iii) contribution‑based pruning guided by opacity and maximum scale.

3) Implementation & results: a Vulkan implementation on a Snapdragon 8 Gen 3 device; the method achieves 116–127 FPS at mobile resolutions with ~4–5 MB per‑scene storage, and >1,100 FPS on an RTX 3090 with similar or better quality than lightweight baselines. The paper identifies sorting as the desktop bottleneck and provides runtime breakdowns showing the MLP overhead is modest.

### Strengths
++ The paper starts from a concrete performance study showing that near‑to‑far sorting dominates 3DGS inference time. It then replaces sorting with a depth‑aware, order‑independent blend: per‑pixel colors are computed by normalizing a weighted sum over all contributing Gaussians, where the weights increase with proximity and scale and are modulated by a small learned, view‑dependent factor. 

++ Order‑independent blending can cause depth‑ambiguity/“see‑through” artifacts; the authors respond with a tiny opacity/weighting MLP that conditions on Gaussian geometry, SH appearance and view direction to predict 𝜙 and a view‑conditioned opacity. The ablation in Table 3 shows that removing this module causes a notable quality drop (e.g., PSNR from 28.45 → 28.06 on Mip‑NeRF360) while the runtime overhead is small.

++ Three components—first‑degree SH distillation, neural vector quantization (NVQ) with sub‑codebooks and tiny decoders, and contribution‑based pruning—work together to shrink the footprint while keeping quality.

### Weaknesses
-- Eq. (2) uses a global transmittance 𝑇, then defines 𝑇, which is order‑dependent and index‑ambiguous under OIR; this needs a precise approximation/implementation 

-- Fig. 3 claims tile‑based rasterization is removed and “all Gaussians associated with a pixel” are blended, but the paper doesn’t detail how per‑pixel lists are built/cached on GPU (desktop or mobile).

-- The proposed weight 𝑤_i (Eq. (3), pp. 4–5) blends squared, inverse‑squared and exponential terms whose dynamic ranges can differ by orders of magnitude. The paper shows an ablation for turning OIR on/off (Table 3), but not a component‑level analysis, nor a discussion of clipping/normalization. Without this, it’s hard to assess numerical stability, generalization to thin structures, and the sensitivity to scene scale.

### Questions
1. Please clarify the exact computation of T in Eq. (2). If it is not the product in the definition (which requires sorting), what approximation is actually used and how is it implemented? Is it related to weighted blended OIT (e.g., a transmittance proxy from aggregated α)? A small derivation or pseudocode would help.

2. What data structure replaces tile binning? Are you using screen‑space bounding ellipses with per‑pixel lists, hierarchical culling, or compute‑shader binning? Please quantify the cost of building these lists, especially on mobile.

3. What ranges are enforced for 𝑑_𝑖 and 𝑠_max? Are terms clamped or normalized per‑tile/view? Could you share an ablation removing each term? 

4. What exact resolution(s) and camera path were used for Table 2? How long were runs and what was the steady‑state FPS after 5–10 minutes? Any power draw measurements?

### Soundness
3

### Presentation
3

### Contribution
3
