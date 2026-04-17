# Uncertainty-Aware Gaussian Map for Vision-Language Navigation

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Vision-Language Navigation (VLN) requires an agent to navigate 3D environments following natural language instructions. During navigation, existing agents commonly encounter perceptual uncertainty, such as insufficient evidence for reliable grounding or ambiguity in interpreting spatial cues, yet they typically ignore such information when predicting actions. In this work, we explicitly model three forms of perceptual uncertainty (i.e., geometric, semantic, and appearance uncertainty) and integrate them into the agent’s observation space to enable informed decision-making. Concretely, our agent first constructs a Semantic Gaussian Map (SGM), composed of differentiable 3D Gaussian primitives initialized from panoramic observations, that encodes both the geometric structure and semantic content of the environment. On top of SGM, geometric uncertainty is estimated through variational perturbations of Gaussian position and scale to assess structural reliability; semantic uncertainty is captured by perturbing Gaussian semantic attributes to reveal ambiguous interpretations; and appearance uncertainty is characterized by Fisher Information, which measures the sensitivity of rendered observations to Gaussian-level variations. These uncertainties are incorporated into SGM, extending it into a unified 3D Value Map, which grounds them as affordances and constraints that support reliable navigation. Comprehensive evaluations across multiple VLN benchmarks show the effectiveness of our agent.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents an uncertainty-aware Vision-Language Navigation (VLN) framework that explicitly models and leverages perceptual uncertainty to improve the navigation decision making. The key contribution is constructing a Semantic Gaussian Map (SGM) using differentiable 3D Gaussian primitives from panoramic observations, then estimating three forms of uncertainty: geometric, semantic, and appearance uncertainty, which are then integrated into a unified 3D Value Map that encodes affordances and constraints, guiding more reliable navigation decisions. Empirically, the proposed method improves the performance across benchmarks including R2R, RxR, and REVERIE, demonstrating the method's effectiveness.

### Strengths
1. The motivation and problem identification is clear. The paper motivates why uncertainty matters in VLN through concrete examples (e.g., confusing similar doors, occlusion ambiguity) that existing agents struggle to handle.
2. The proposed framework is technically sound. The combination of 3D Gaussian Splatting with variational inference for uncertainty estimation is well-grounded. The mathematical formulation is rigorous and clearly presented. The uncertainty estimation addresses three types of uncertainty, providing a complete picture.
3. Experiments are well-conducted, validating the effectiveness of the proposed method. Comprehensive ablation study demonstrates each component's contribution. Runtime analysis shows acceptable overhead, eliminating the concern about  excessive computational resource consumption.

### Weaknesses
1. Limited hyperparameter sensitivity analysis. Only the pruning threshold $\tau_e$ and $\tau_{\alpha}$ are analyzed (in Table 5). Sensitivity of uncertainty-related hyperparameters like $\delta$ and $\eta$ for geometric uncertainty and $\epsilon$ for semantic uncertainty.
2. Theoretical grounding of uncertainty identification is weak. While Appendix E.1 briefly discusses epistemic vs. aleatoric uncertainty, the mapping of geometric/semantic to epistemic and appearance to aleatoric lacks rigorous justification. This categorization seems somewhat arbitrary.
3. While standard deviations are reported, no significance tests confirm whether improvements are statistically meaningful beyond variance.
4. It would be more persuasive if the authors could upload video demos.

### Questions
1. Could the authors provide a comprehensive hyperparameter sensitivity analysis?
2. Could the authors provide a stronger justification regarding the uncertainty categorization (mapping geometric, semantic, appearance to epistemic and aleatoric)?
3. Could the authors conduct significance tests to prove their improvements are statistically meaningful?
4. As mentioned in the Limitation and Future Work, since the major contributions of the work are made in simulation environments. Any thoughts on addressing expected transfer problems to the real world? Like potential uncertainty estimate degradation with real sensor noise, dynamic objects, or actuation errors.

### Soundness
3

### Presentation
2

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
The paper proposes an Uncertainty-Aware Gaussian Map (UAGM) framework for Vision-Language Navigation (VLN). The method represents the environment as a 3D Gaussian field and explicitly models three types of perceptual uncertainty—geometric, semantic, and appearance—via variational inference and Fisher-information-based estimation. These uncertainty estimates are aggregated into a 3D Value Map used by the navigation policy. The approach aims to improve robustness and interpretability by allowing the agent to reason about uncertain visual observations. Experiments on R2R, RxR, and REVERIE show consistent performance gains over prior VLN baselines, and qualitative visualizations demonstrate uncertainty-aware perception.

### Strengths
1. The paper introduces an explicit Gaussian-based 3D uncertainty representation, offering a principled way to quantify perception uncertainty in VLN.
2. The theoretical formulation (variational inference + Fisher information) is well motivated and mathematically sound.
3. Experiments are comprehensive, spanning three benchmarks with ablations and qualitative visualizations.
4. The paper is generally clear and technically solid, with detailed appendices.

### Weaknesses
1. The claim of being the first to introduce uncertainty modeling in VLN is somewhat overstated. Prior work such as LLM as Copilot for Coarse-grained Vision-Language Navigation (ECCV 2024)[1] already explores confidence or uncertainty-related reasoning. The main novelty here lies in adopting a Gaussian-based uncertainty representation, rather than introducing the uncertainty concept itself. Moreover, the estimated uncertainty is not explicitly incorporated into decision-making—it serves more as an auxiliary feature than a truly uncertainty-guided policy.
2. The paper employs ambitious terminology (e.g., 3D Value Map, Unified Uncertainty Grounding), which may slightly overstate conceptual novelty.
3. Although the appendix includes qualitative failure cases, the analysis remains superficial. There is no quantitative study demonstrating how uncertainty correlates with navigation errors or ambiguous observations.
4. The latest compared method is VER (CVPR 2024), which is relatively dated. It would strengthen the paper to include comparisons with more recent works such as [2][3][4].

References

[1]  LLM as Copilot for Coarse-grained Vision-Language Navigation, ECCV 2024.
[2]  Do Visual Imaginations Improve Vision-and-Language Navigation Agents?, CVPR 2025.
[3]  COSMO: Combination of Selective Memorization for Low-cost Vision-Language Navigation, ICCV 2025.
[4]  Bootstrapping Language-Guided Navigation Learning with Self-Refining Data Flywheel, ICLR 2025.

### Questions
Have the authors analyzed whether high predicted uncertainty can be used to predict or prevent navigation failure?

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
To address the perceptual uncertainty in robot navigation, this paper formalizes three forms of uncertainty, including geometric, semantic, and appearance uncertainty, enabling more informed decision-making. Specifically, a Semantic Gaussian Map (SGM) based on 3D Gaussian primitives is constructed to encode both the geometric structure and semantic content of environments. Then, three uncertainties are estimated: 1) geometric uncertainty is modeled through variational perturbations of Gaussian position and scale to assess structural reliability; 2) semantic uncertainty is estimated by perturbing the semantic attributes of Gaussians to reveal ambiguous interpretations; 3) appearance uncertainty is quantified by Fisher Information to measure the sensitivity of rendered observations to Gaussian-level variations. Then, a unified 3d Value Map is composed to ground these uncertainties as affordances and constraints, thereby guiding informed and more reliable trajectory planning. Extensive experiments on VLN benchmarks demonstrate the effectiveness of the proposed method.

### Strengths
1)	This paper focuses on a valuable but usually ignored research topic, i.e., the perceptual uncertainty in robot navigation, and systematically formalizes three forms of perceptual uncertainties, including geometric, semantic, and appearance uncertainty, which reveals a new improvement direction for robot navigation.
2)	Extensive experiments are constructed to demonstrate the effectiveness of the proposed uncertainties.
3)	This paper is well-written and easy to follow.

### Weaknesses
1）	Could the recognized perceptual uncertainty help to further lower the uncertainty? How do you improve the confidence/reliability? Will it help with robot navigation?
2）	In the action prediction of the 3D Value Map, all the representations of each Gaussian are projected into a feature vector, and then the aggregated representation is utilized to predict candidate nodes. So the uncertainties indirectly and aggregately impact action prediction, does it need to compute the uncertainty of each Gaussian? 
3）	According to the results in Tables 1 and 2, the improvement of the proposed method is marginal over the state-of-the-art, especially on the SR, SPL. 
4）	According to the results in Table 6, it seems the three uncertainties do improve the baseline method, but the baseline is obviously outperformed by the SOTA methods. Could the proposed method still work on a better baseline?

### Questions
Please try to address the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
