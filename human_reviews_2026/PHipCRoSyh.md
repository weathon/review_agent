# LazyDrag: Enabling Stable Drag-Based Editing on Multi-Modal Diffusion Transformers via Explicit Correspondence

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6, 6

## Abstract
The reliance on implicit point matching via attention has become a core bottleneck in drag-based editing, resulting in a fundamental compromise on weakened inversion strength and costly test-time optimization (TTO). 
This compromise severely limits the generative capabilities, suppressing high-fidelity inpainting and text-guided creation. In this paper, we introduce LazyDrag, the first drag-based image editing method for Multi-Modal Diffusion Transformers, which directly eliminates the reliance on implicit point matching. In concrete terms, our method generates an explicit correspondence map from user drag inputs as a reliable reference to boost the attention control. 
This reliable reference opens the potential for a stable full-strength inversion process, which is the first in the drag-based editing task. It obviates the necessity for TTO and unlocks the generative capability of models. Therefore, LazyDrag naturally unifies precise geometric control with text guidance, enabling complex edits that were previously out of reach: opening the mouth of a dog and inpainting its interior, generating new objects like a ``tennis ball'', or for ambiguous drags, making context-aware changes like moving hands into pockets. Moreover, LazyDrag supports multi-round edits with simultaneous move and scale operations. Evaluated on DragBench, our method outperforms baselines in drag accuracy and perceptual quality, as validated by mean distances, VIEScore and user studies. LazyDrag not only sets new state-of-the-art performance, but also paves a new way to editing paradigms. Here is the project website.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
LazyDrag is a training-free drag-based image editing method for Multi-Modal Diffusion Transformers. It replaces unstable attention-based point matching with an explicit correspondence map derived from user drags, which is very similar to Latent Warpage Function of FastDrag, enabling stable full-strength inversion without test-time optimization. This design preserves identity, maintains background consistency, and supports text-guided edits for complex transformations like movement or scaling. Evaluated on DragBench, LazyDrag achieves state-of-the-art accuracy and perceptual quality while remaining efficient and robust, demonstrating that explicit geometric correspondence can unify precise spatial control and semantic guidance in diffusion-based editing.

### Strengths
1 The method explicitly associates point matching with the user’s drag instructions following latent warpage function, removing the uncertainty of attention-based implicit matching. This design eliminates many stability and consistency issues seen in previous drag-editing approaches.


2 LazyDrag achieves state-of-the-art results on DragBench across all major metrics, demonstrating both higher drag accuracy and better perceptual quality without requiring test-time optimization.

### Weaknesses
1: Computational complexity not included. The paper does not provide quantitative analysis of runtime or memory costs. Since attention modification is applied across multiple layers and steps, its efficiency in large-scale or real-time scenarios is unclear.

2: Limited novelty beyond integration. While the explicit correspondence map is effective, serveral components build upon existing ideas from FastDrag in method section but not discussed in the intro sec. The contribution may be seen as a strong integration rather than a fundamentally new mechanism. 

Anyway, I still view it as a meaningful and well-executed improvement. So my suggetion is the authors could explicitly position their contribution as the first to replace implicit point matching with an explicit correspondence map, and visually emphasize this transition and its advantages in the Introduction (e.g., replace the current Figure 1 with a more conceptual figure illustrating this replacement and maybe with a table which shows the adavantages, and move the current example figure to the next page). Secondly, in the method section, a concise side-by-side illustration figure comparing FastDrag’s and LazyDrag’s formulations would also help readers immediately see where the improvements lie. A good example is the left part of Fig1 in https://arxiv.org/pdf/2503.20783.

3: Experiments are mainly conducted on FLUX-based MM-DiT and a single benchmark.

### Questions
The paper reports a user study with 20 expert participants evaluating 32 samples each. Could the authors clarify whether this sample size is sufficient to ensure statistical significance and generalization?

### Soundness
2

### Presentation
2

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
The paper introduces LazyDrag, a training-free, drag-based image editing method designed for Multi-Modal Diffusion Transformers (MM-DiTs). The core problem it addresses is the instability of prior methods, which rely on implicit point matching via attention, forcing them to use test-time optimization (TTO) or weakened inversion strength. LazyDrag's key contribution is to replace this fragile implicit matching with an explicit correspondence map derived directly from user drag instructions. This deterministic guidance enables stable editing under full-strength inversion without TTO, unlocking high-fidelity inpainting and robust text-guided control. The method is shown to outperform existing baselines on the DragBench benchmark in both accuracy and perceptual quality.

### Strengths
1.  **Novel Application and Success on DiT Architectures:** This is arguably the first work to successfully and robustly implement drag-based editing on the emerging MM-DiT architecture. As the field moves from U-Nets to Transformers, this paper provides a timely and valuable contribution, demonstrating a viable path for precise spatial control on these powerful new backbones.

2.  **Principled Solution to a Fundamental Problem:** The authors correctly identify a core bottleneck in previous drag-editing methods—the unreliability of implicit attention-based point matching. Their proposed solution, an explicit correspondence map, is a direct and elegant way to solve this issue. This allows them to eliminate the need for costly test-time optimization (TTO) and, more importantly, enables stable full-strength inversion, which is a significant step forward for the task.

3.  **Impressive Qualitative and Quantitative Results:** The method demonstrates state-of-the-art performance on the DragBench benchmark. The qualitative results are compelling, showcasing complex edits (e.g., opening a mouth and inpainting the interior, generating new objects with text guidance) that prior methods struggle with. The quantitative improvements in mean distance and VIEScore, despite requiring no TTO, strongly support the effectiveness of the proposed approach.

### Weaknesses
1.  **Confounded Comparison of Foundation Models:** The primary quantitative results claim state-of-the-art performance by comparing the proposed method, built on a powerful MM-DiT backbone (FLUX.1), against baselines developed for older U-Net architectures (e.g., Stable Diffusion v1.5). This comparison is confounded, as it's unclear how much of the performance improvement is due to the novel LazyDrag algorithm versus the superior generative capabilities of the underlying foundation model. A more rigorous evaluation would involve comparing the methods on the same backbone to isolate the algorithmic contribution.

2.  **Insufficient Differentiation from Prior Motion-Guided Methods:** The core idea of using an explicit motion field to guide a training-free editing process has precedents in recent literature (e.g., InstantDrag, LightningDrag). While the paper's specific implementation—transforming this field into a correspondence map to directly control attention is novel, the manuscript would be stronger if it more clearly articulated this distinction. A direct, mechanistic comparison of how LazyDrag *uses* the motion information versus how prior works use it (e.g., for latent warping or loss-based supervision) is needed to better situate the paper's unique contribution.

3.  **Sensitivity to Key Hyperparameters:** The method's effectiveness appears highly sensitive to the `activation timestep` hyperparameter, as revealed in the paper's own ablation study (Sec. 4.6). The results show a stark trade-off: lower values fail to move points accurately, while higher values introduce significant artifacts, especially with complex instructions. This implies that while the method avoids test-time optimization, it may require careful, per-edit manual tuning to achieve satisfactory results, which somewhat tempers the "lazy" and automated nature of the proposed workflow.

### Questions
Could you elaborate on the failure modes of the "winner-takes-all" strategy for displacement field calculation? For instance, how does it behave with a high density of conflicting drag points in a small region, and what are the computational implications?

### Soundness
3

### Presentation
2

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
The paper proposes LazyDrag, a training-free drag-based image editing method for MM-DiTs that replaces point matching with an explicit correspondence map derived from user drags. This design enables stable full-strength inversion without test-time optimization. LazyDrag supports multi-round operations (move/scale) and resolves ambiguous drags via prompt guidance, delivering higher drag accuracy and visual quality than prior baselines on DragBench and in user studies. From the paper, this is: (1) the first MM-DiT drag editor under full-strength inversion, (2) explicit correspondence-driven attention controls that remove TTO, and (3) SOTA results with text-guided, natural inpainting.

### Strengths
* From the experiments, this method achieves the state-of-the-art performance in DragBench (Table 1), as well as achieving solid empirical results with user preference. In addition, this paper also integrates editing, like scaling and movement. Although these operations are relatively common in latent editing, it's good to show such ability in one pipeline.
* Compared with optimization-based methods like Dragdiffusion and DragGAN, the proposed method provides faster inference speed.
* The overall pipeline is clearly illustrated and easy to understand.

### Weaknesses
* The core idea of explicit drag-to-correspondence plus attention control feels incremental relative to prior drag/edit pipelines. Specifically, the TTO-free idea has been used in many previous works (like SDE-Edit, RegionDrag), although the way of replacement may be different. The full-strength editing is also natural in the inversion-based editing. From this perspective, the technical novelty of this paper is not strong. The abstract and relative part needs to claim the core contribution more carefully and clearly. 
* Although the paper shows significant improvements in Table 1, the metrics' increase seems much more marginal in Table 5, meaning that the major improvement factor comes from changing to DiT. Since it makes such a big difference, it is helpful to remind of this detail in Table 1, which shows both cases (UNet and DiT).
* A detailed memory and time cost of the proposed method is missing, especially when the paper emphasizes the efficiency of TTO-free design.

### Questions
* The quantitative evaluation seems impressive, but why does the qualitative evaluation seem worse than the baselines? E.g., In Figure 8 last row, the man's face seems distorted compared with DragNoise.

### Soundness
2

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
4

### Summary
The paper introduces LazyDrag, a drag-based image editing method for DiT-family models (MM-DiTs). It performs full-strength inversion and avoids test-time optimization (TTO). The core idea is to learn an explicit correspondence map guided by text, combined with background and identity preservation to deliver high-quality, controllable drags. The paper provides broad comparisons with prior drag-editing approaches and reports strong qualitative and quantitative results.

### Strengths
1.	Modern backbone: Extends drag editing from U-Net diffusion (e.g., Stable Diffusion) to contemporary DiT architectures, which is timely and relevant.
2.	No test time optimization: The method avoids test-time optimization, which can reduce engineering complexity and potentially runtime.
3.	Instruction clarity: The correspondence map makes drag constraints explicit and easier to interpret.
4.	Compelling results: Both qualitative figures and reported metrics indicate performance gains over prior work.

### Weaknesses
1.	Runtime and efficiency claims need evidence
No TTO does not automatically imply faster end-to-end editing if additional modules (inversion, correspondence estimation, preservation) are heavy. 
Could the authors report end-to-end wall-clock time, GPU memory for LazyDrag vs. other methods? Include all components and required steps related to that method, both TTO free and non-free methods.

2.	The paper lists full-strength inversion as a main contribution. Why is full-strength inversion necessary for drag editing? In Fig. 2, full-strength inversion appears to improve LazyDrag’s results, but is this inherently required by the drag-editing setup? Also, higher inversion strength likely increases computation time.

3.	Please add experiments demonstrating that the performance gains come from the LazyDrag design rather than from the MM-DiT backbone. Prior methods use U-Net models (e.g., Stable Diffusion), while this paper uses FLUX.1 Krea-dev. How do you show that improvements are not dominated by the model shift? 

4.	The experiments report Semantic Consistency and Perceptual Quality. Could you add results to justify that these metrics are suitable for drag editing, for example, by also reporting other metrics used in related work and comparing against them?

5.	The stated contributions and the method description feel slightly misaligned. The methods section is divided into two parts, with the second devoted to preservation, but “preservation” is not clearly reflected among the main contributions. Meanwhile, the introduction emphasizes no test-time optimization and full-strength inversion. Clarifying and aligning the concepts and contributions across the introduction and methods would improve the paper.

### Questions
1.	This paper treats background and identity differently. How are background and objects detected and distinguished in practice? Are there any additional steps, or do you rely on other methods?
2.	The shift to DiT-based image models is natural, since newer models are generally more powerful than earlier ones. This could bring future benefits to the drag editing community.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents LazyDrag, a training-free drag-based image editing method tailored to multi-modal diffusion transformers. The core idea is to replace implicit attention-based point matching with an explicit correspondence map computed from user drag instructions. This map drives both latent initialization in uncovered regions and attention control via token replacement and gated blending, which allows edits under strong inversion while preserving image identity. Experiments on a standard drag-editing benchmark and several recent baselines show improved spatial accuracy, semantic consistency, and perceptual quality without test-time optimization. A user study further indicates that human raters prefer LazyDrag in most cases, suggesting practical relevance for interactive editing.

### Strengths
- Clear diagnosis of why attention-similarity–based matching fails under strong inversion, which directly motivates the shift to explicit correspondence maps and gives the method a coherent conceptual foundation.

- Elegant use of a single correspondence map to control both latent initialization and attention, enabling principled handling of background, inpainting, and transition regions and reducing artifacts from naïve displacement blending.

- Strong emprical results on a standard drag-editing benchmark, with consistent gains in spatial accuracy, semantic consistency, and perceptual quality over several recent drag-based baselines, and without test-time optimization.

- Qualitative and user-study evidence that the method handles complex and text-guided edits (e.g. ambiguous drags, newly revealed content) in a way that is preferred by human raters.

### Weaknesses
- Comparative results are confounded by differences in backbone and inversion quality, because LazyDrag runs on a strong MM-DiT while several baselines still use UNet models. Implementing LazyDrag on a shared U-Net backbone or porting at least one baseline to the same MM-DiT would clarify how much of the gain comes from the editing strategy itself.

- Efficiency claims remain qualitative, even though interactive editing hinges on latency and resource usage; reporting wall-clock runtime and memory consumption against speed-oriented methods would make the practical advantages more credible.

- Generalization beyond a single MM-DiT backbone and benchmark is not demonstrated.

- The text-guided resolution of ambiguous drags is mainly shown through qualitative examples. Maybe a small controlled benchmark or focused user study on this aspect would better quantify its advantage over purely drag-based methods.

### Questions
- Can you provide numbers for runtime and memory usage per edit (including inversion) compared to speed-oriented methods such as FastDrag-style or inpainting-based approaches on same hardware?

- Are there architectural assumptions that make LazyDrag significantly easier to implement on MM-DiTs than on U-Net backbones, or is it a design choice?

- How robust is the method to some poorly placed drag points, for instance when both handle and target lie slightly off the intended object or cross semantic boundaries?

### Soundness
3

### Presentation
3

### Contribution
3
