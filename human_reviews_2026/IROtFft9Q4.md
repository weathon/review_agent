# Depth Anything with Any Prior

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 8

## Abstract
This work presents Prior Depth Anything, a framework that combines incomplete but precise metric information in depth measurement with relative but complete geometric structures in depth prediction, generating accurate, dense, and detailed metric depth maps for any scene. To this end, we design a coarse-to-fine pipeline to progressively integrate the two complementary depth sources. First, we introduce pixel-level metric alignment and distance-aware weighting to pre-fill diverse metric priors by explicitly using depth prediction. It effectively narrows the domain gap between prior patterns, enhancing generalization across varying scenarios. Second, we develop a conditioned monocular depth estimation (MDE) model to refine the inherent noise of depth priors. By conditioning on the normalized pre-filled prior and prediction, the model further implicitly merges the two complementary depth sources. Our model showcases impressive zero-shot generalization across depth completion, super-resolution, and inpainting over 7 real-world datasets, matching or even surpassing previous task-specific methods. More importantly, it performs well on challenging, unseen mixed priors and enables test-time improvements by switching prediction models, providing a flexible accuracy-efficiency trade-off while evolving with advancements in MDE models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a metric-scale depth completion framework that can handle various types of sparse depth measurements, including LiDAR-like, SfM, masked, or range-out. The method combines affine-invariant depth estimation (referred to as geometric estimation in the paper) and metric measurement into a single pipeline to take advantage of both geometric consistency and metric accuracy. The framework is composed of two sequential stages: a monocular depth estimation (MDE) model and a conditional MDE model. The latter takes as input the RGB image, the output of the first MDE, and the filled metric measurements. Under the proposed evaluation setup, the model shows strong zero-shot performance in depth completion, super-resolution, and inpainting tasks. Ablation studies on how to fill metric measurements and how to configure the conditional MDE input further justify the design choices.

### Strengths
1. The proposed pipeline, which sequentially connects an MDE and a conditional MDE, may appear simple but is both reasonable and effective, as it integrates the strengths of geometric estimation and metric measurement within a unified framework.
2. Through ablation studies on pre-filled depth and the input configuration of the conditional MDE, the authors validate their design choices. The final model achieves consistently strong performance across diverse evaluation settings.

### Weaknesses
1. Experimental details
- The experimental section lacks sufficient detail regarding how different types of degraded depth data were generated. Specifically, it is unclear how sparse points (from SfM, LiDAR, or extremely sparse settings), low-resolution inputs (captured, ×8, ×16), and missing-area cases (range, shape, object) were obtained for each dataset. I would recommend providing explicit descriptions in the supplementary material on how these variations were constructed per dataset.
- The comparison table does not include citations for the baseline methods, and some references are missing from the main text as well. The visual emphasis in the table (e.g., bold, underline, gray shading) is inconsistent. What does “gray” indicate?

2. Figures and presentation clarity
- The visual presentation in Figures 3 and 4 is somewhat confusing. It is not entirely clear whether the error maps correspond to differences between output of proposed method and GT. It would be better to make the visualization and caption clearer to avoid misinterpretation.

3. Evaluation protocol and additional benchmarks
- While the proposed evaluation setup demonstrates strong performance, it would strengthen the paper to include additional experiments following the evaluation setups used in Marigold-DC or Zero-DC [a1], particularly on the VOID dataset. If feasible, including results on IBims would also help assess the outstanding performance of proposed method.

[a1] Hyoseok et al., "Zero-shot Depth Completion via Test-time Alignment with Affine-invariant Depth Prior", AAAI 2025.

### Questions
1. For the de-normalization process converting the predicted depth to metric scale, could the authors clarify whether the de-normalization process corresponds to applying the inverse of the normalization function used for metric measurements?
2. In Section 4.5 (Effectiveness of Metric and Geometry Condition and Table 7), was the training setup identical to the main experiments?
3. In Line 313, the paper mentions training with 8 GPUs. Which GPU model was used? and how long does it take?
4. In Section 4.5, is the conditional MDE trained using a single MDE backbone, or are multiple backbones used for the ablation study in Table 8?
5. There appears to be a typo in Line 455. It likely refers to Table 5 instead of Fig. 6.

### Soundness
3

### Presentation
4

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
This paper proposes Prior Depth Anything, a unified framework that fuses precise but incomplete metric depth priors (e.g., sparse points, low-resolution maps, masked regions) with complete but relative monocular depth predictions to produce dense and accurate metric depth. The pipeline is coarse-to-fine:(i) Pixel-wise scale–shift alignment with distance-aware weighting fills missing regions using a frozen monocular predictor, reducing gaps among different prior types;(ii) A conditioned monocular depth network refines results from the RGB image, the pre-filled prior, and the prediction (normalized), mitigating measurement noise and reconciling geometry.Experiments on seven datasets show strong zero-shot results in completion, super-resolution, and inpainting, with robustness to mixed priors. Test-time backbone swapping supports accuracy–efficiency trade-offs. Overall, the design provides a clear fusion recipe and broad, competitive performance across settings.

### Strengths
Broad applicability and novelty: One framework handles completion, upsampling, inpainting, and their combinations, covering common real-world inputs that previous methods often treat separately.

Strong empirical performance: Consistent zero-shot results across multiple datasets and tasks, often matching or surpassing task-specific baselines without per-task fine-tuning.

Robust to mixed priors: Maintains accuracy when prior types co-occur (e.g., sparse + low-res + holes), a challenging but practical scenario.

Clear integration pipeline: Pixel-level metric alignment plus a conditioned refiner is simple, well-motivated, and effective in practice; normalization enables flexible backbone choices.

### Weaknesses
Efficiency underreported: The two-stage design (predictor + kNN-style alignment + conditioned refiner) lacks detailed latency, memory, and component-wise cost, leaving deployability unclear.

Ablations and sensitivity limited: The impact of weaker/faster predictors, distance-aware weighting, neighborhood size, or removing coarse alignment is not fully quantified.

### Questions
Effect of predictor quality: How do metrics change with lighter/faster monocular predictors versus the default, and with different alignment settings? Please provide an accuracy–throughput curve and key ablations.

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
4

### Summary
This paper presents a framework that integrates incomplete but accurate metric depth measurements with complete yet relative geometric predictions from monocular depth estimation (MDE) models to generate detailed and metrically consistent depth maps. The motivation stems from the complementary nature of these inputs: MDE produces dense, fine-grained relative depth but lacks absolute scale, whereas metric measurements provide precise scaling information but are often sparse, low-resolution, or partially missing. The proposed method adopts a coarse-to-fine pipeline that first performs pixel-level metric alignment, followed by a conditioned MDE refinement stage. The framework demonstrates strong zero-shot generalization across depth completion, super-resolution, and inpainting tasks on seven real-world datasets.

### Strengths
1. The paper presents extensive experiments that effectively validate the proposed method and justify its performance against benchmark approaches.
2. The proposed framework achieves enhanced depth estimation quality through a simple and well-structured pipeline.

### Weaknesses
1. The proposed method shows limited novelty, as it primarily predicts per-pixel scale and shift values in MDE to align with the depth prior.
2. The paper states, “We highlight best and second-best results” in the quantitative results section; however, only a few columns in Tables 2–5 are correctly annotated. The remaining columns contain inconsistencies, including missing annotations, unclear labeling, or incorrect markings.
3. The reported quantitative and qualitative results do not convincingly demonstrate the proposed method’s superiority over baseline approaches. Specifically, the method fails to outperform in:
    + 14 out of 16 experiments in Table 4,
    + 4 out of 16 experiments in Table 5,
    + 5 out of 16 experiments in Table 3, and
    + 1 out of 3 comparisons in Figure 3 (the second experiment).

These results indicate that the proposed method underperforms in a significant portion of the evaluations.
Overall, the improvements reported do not sufficiently support the claimed performance gains.

### Questions
1. What specific problem does the proposed distance-aware weighting (line 229) aim to solve? The authors are encouraged to provide clearer intuition or theoretical justification for this formulation.
2. Why are recent baselines such as SharpDepth [1] not included in the comparison?
3. How were the baseline methods implemented, particularly with respect to the use of different types of depth priors? Additional implementation details would improve clarity and reproducibility.
4. It would strengthen the paper to include more qualitative comparisons with other benchmarks—ideally 10–20 diverse examples—to better illustrate the method’s generalization ability and visual performance.

[1] Pham, D.-H., Do, T., Nguyen, P., Hua, B.-S., Nguyen, K., & Rang, N. (2024). SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation. arXiv preprint arXiv:2411.18229.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a framework for prior-based monocular depth estimation, which accepts RGB images and depth priors (e.g. LiDAR or SfM in depth completion, low-resolution Time of Flight camera depth maps, incomplete depth maps, etc.) and outputs a dense metric depth map. Instead of focusing on any particular prior, this work proposes a unified method that can generate metric depth from any of these prior sources. The method consists of two stages. The first generates a coarse depth map based on depth priors and the predictions from a pre-trained monocular depth model (e.g. Depth Anything V2 is used in training experiments). This is done with the proposed "pixel-level metric alignment", where for each missing pixel, a set of $k$ nearest neighbors present in the prior are computed. These set of points are then used to compute a scale and shift parameter which best aligns (weighted by distance to the query pixel) these points with that predicted by the frozen MDE. The scale and shift parameters used to compute the missing pixel's values from the frozen MDE prediction. The second stage aims to improve robustness against noise in priors using another conditioned MDE. The input condition to this MDE is generated from RGB input, normalized frozen MDE prediction, and the normalized coarse prediction from the first stage via convolution layers. The output is then de-normalized to obtain the final predictions. The model is trained with synthetic datasets using several augmentations / corruptions to generate synthetic priors, and then evaluated zero-shot across multiple different real-world tasks with different sources of priors, across multiple datasets, and shown overall to perform comparably or better than existing methods.

### Strengths
- Extensive experiments are conducted to show that the method performs strongly across different sources of depth priors, which is shown to be a key strength of the model compared to existing works. The proposed method convincingly generalizes significantly better compared to existing state-of-the-art approaches.

- The pixel-level alignment method for in-painting missing depth regions seems novel, and shown to be significantly better than naive interpolation in Table 6 and 9. There is reason to believe that this insight can be generalized to improve other works in the field.

- Extensive ablations are also performed, ablating the different input conditions for the conditioned MDE model (Table 7), different frozen MDE models (Table 8), and in-painting strategy (Table 6 and 9).

### Weaknesses
- The method first requires using a heuristic-based approach for densifying a depth prior. This might not be effective or practical especially when the prior map is extremely sparse, from both a latency (since densification requires solving a least-square regression for each missing pixel) and performance (the nearest neighbors might be extremely far away from the query pixel) standpoint.

- Minor: I am not sure whether it is an issue with my PDF viewer, but the formatting of the paper seems off, especially in the first page. The hyperlinks seem to jump all over the place, and overlays / occludes existing text.

### Questions
- The authors provided an intuition for the two-stage approach, where the first seems to provide a coarse (possibly noisy) estimate, and the second refines this. How strongly is this observed is this in practice? In the extreme case, how robust is the method given a very noisy (or perhaps even randomly initialized) depth prior map, would it be possible to at least recover the performance of that from the pre-trained MDEs?

- The authors show that it is possible to stack multiple (2) MDEs to produce better depth predictions. I wonder if this is possible to improve results further at test-time by applying the conditional MDE in a recursive manner (i.e. repeatedly feeding in the output of the conditioned MDE in place of $\hat{D}_{prior}$ back into itself), which can allow using "adaptive" inference-time compute to produce outputs at various levels of refinement.

### Soundness
4

### Presentation
4

### Contribution
3
