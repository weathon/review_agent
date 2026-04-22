# Benchmarking Physical Reasoning of Video Generative Models with Real Physical Experiments

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Recent advances in image and video generation raise hopes that these models possess world modeling capabilities—the ability to generate realistic, physically plausible videos. This could revolutionize applications in robotics, autonomous driving, and scientific simulation. However, before treating these models as world models, we must ask: Do they adhere to physical laws?  Current evaluation methods rely on subjective judgments or trajectory matching, limiting their usage for physical reasoning estimation, where many generations could be physically plausible. 
Thus, we introduce **Morpheus**, a new benchmark for evaluating video generation models on physical reasoning. It features 130 real-world videos capturing physical phenomena, guided by conservation laws. Using those as conditioning for video generation, we assess physical plausibility using physics-informed metrics evaluated with respect to infallible conservation laws known per physical setting, leveraging advances in physics-informed neural networks and vision-language foundation models.
% Since artificial generations lack ground truth, we assess physical plausibility using physics-informed metrics evaluated with respect to infallible conservation laws known per physical setting, leveraging advances in physics-informed neural networks and vision-language foundation models. 
Our findings reveal that even with advanced prompting and video conditioning, current models struggle to encode physical principles despite generating aesthetically pleasing videos.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Morpheus, a benchmark for evaluating the physical reasoning ability of video generative models (VGMs). Using 130 real-world physics experiments and physics-informed neural networks (PINNs), it measures whether generated videos obey Newtonian laws and conserve physical invariants like energy and momentum. The benchmark computes two metrics—Dynamical Score and Physical Invariance Score—to quantitatively assess physical plausibility. Experiments on major VGMs (e.g., COSMOS, Wan2.1, LTX, PyramidalFlow) show that, despite realistic appearances, current models fail to respect fundamental physical principles, highlighting the need for physically grounded video generation.

### Strengths
1. The paper introduces the first benchmark grounded in real physical experiments, offering a rigorous and reproducible setup beyond simulation or human judgment.

2. The use of PINN-based evaluation metrics to measure both dynamical consistency and physical invariants is technically sound and interpretable.

3. The analysis is comprehensive, covering multiple models, prompt settings (text, single/multi-frame), and experiments, providing broad insights into current limitations of VGMs.

### Weaknesses
1. The main limitation lies in scope and generality. The benchmark only covers basic Newtonian systems under controlled lab conditions, excluding non-linear, stochastic, or complex real-world settings (e.g., fluid, thermodynamic, or deformable-body dynamics). This makes it less reflective of broader physical reasoning in the wild.

2. The metrics rely heavily on accurate object tracking via SAM-2 and Depth Anything. Noise, occlusions, or tracking drift could propagate into PINN fitting, potentially biasing the scores. While the paper mentions smoothing and filtering, there is no sensitivity analysis showing metric robustness to visual noise or tracking failure.

3. While Morpheus claims objectivity, the design of invariants and ODE priors is manually specified. This limits scalability—each new physical process requires manual equation definition. Integrating data-driven symbolic discovery or learned physics priors could make the benchmark more generalizable.

4. The prompt engineering process (using upsamplers and ChatGLM-based expansions) introduces confounding factors. It is unclear whether performance differences arise from physical reasoning capability or prompt quality. The fairness of cross-model comparisons (especially between keyframe interpolation vs. open-ended generation) also remains questionable.

### Questions
1. How sensitive are the Morpheus scores to tracking or depth estimation errors in the trajectory extraction pipeline?

2. Are the PINN hyperparameters (e.g., λ for physics loss) fixed across all experiments, and how do they affect comparability?

3. Does Morpheus evaluate 3D consistency (e.g., parallax, occlusion) or only 2D motion projections?

4. Can this framework be extended to evaluate non-Newtonian or fluid-based systems?

### Soundness
3

### Presentation
2

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
This paper introduces Morpheus, a new benchmark designed to evaluate the physical reasoning capabilities of video generative models (VGMs). The authors argue that current evaluation methods, which rely on subjective human judgment or simple trajectory matching, are insufficient for rigorously assessing physical plausibility. Morpheus consists of a dataset of real-world videos capturing nine core physical phenomena governed by Newtonian mechanics (e.g., falling objects, projectile motion, pendulums). The proposed methodology extracts object trajectories from both real and generated videos and assesses their physical plausibility using two novel, physics-informed metrics: a "Dynamical Score," which measures adherence to the governing equations of motion via Physics-Informed Neural Networks (PINNs), and a "Physical Invariance Score," which quantifies the conservation of physical quantities like energy and momentum. The authors evaluate several state-of-the-art VGMs and find that, despite generating aesthetically pleasing videos, they struggle to adhere to these fundamental physical principles, highlighting a significant gap in their world modeling capabilities.

### Strengths
The main strength of this work is its shift from qualitative, subjective assessments to a quantitative, physics-informed evaluation framework. By grounding the metrics in fundamental conservation laws and equations of motion, the benchmark provides a more objective and rigorous way to measure the physical reasoning of VGMs. This is a significant step forward for the field. The creation of a new dataset based on controlled laboratory experiments is a valuable contribution. This controlled setting allows for the targeted evaluation of specific physical principles and ensures that the comparisons between models are fair and reproducible, which is often a challenge with in-the-wild video data.

### Weaknesses
1.Limited Scope of Physical Phenomena: The benchmark's reliance on trajectory extraction from rigid bodies inherently limits its scope. Many important real-world physical phenomena do not involve a clearly trackable rigid object, such as fluid dynamics (e.g., water flowing, smoke rising), thermodynamics (e.g., ice melting), non-rigid body dynamics, or events like explosions. While the focus on Newtonian mechanics is a reasonable starting point, the claim of "benchmarking physical reasoning" is very broad, whereas the method is confined to a relatively narrow fundamental subset of physics.
2.Simplicity of a Majority of the Physical Scenarios: Most of the experiments focus on the motion of a single simple object (e.g., a falling ball, a sliding book). While these are excellent for isolating variables, they may be too "trivial" for the rapid pace of VGM development. State-of-the-art models may quickly learn to master these simple scenarios, potentially limiting the long-term utility of the benchmark. The benchmark could be strengthened by including more complex multi-object interactions or scenarios where secondary physical effects (e.g., air resistance, complex friction) are more prominent.
3.Lack of Correlation with Human Judgment: The paper rightly critiques the subjectivity of human evaluation. However, it does not provide any analysis of how the Morpheus scores correlate with human perception of physical plausibility. A strong benchmark should ideally produce rankings that are consistent with human intuition. Without a correlation study, it is difficult to know if the proposed metrics might sometimes penalize generations that are perceptually plausible (e.g., due to unmodeled physics like friction) or reward generations that are mathematically sound but visually awkward. This comparison is crucial for validating that the metrics are capturing meaningful aspects of physical realism.
Constraints on Evaluation Modality: The proposed method is best suited for image-to-video or video-to-video generation, where the initial state (position, and sometimes velocity) of the object is clearly defined by the conditioning frame(s). This makes it more challenging to fairly evaluate pure text-to-video models, where the model generates the initial state from scratch, introducing ambiguity that the evaluation framework is not designed to handle. This limits the benchmark's applicability to a subset of VGMs.

### Questions
1.Could the authors elaborate on the decision to use Depth Anything V2 for checking depth consistency in videos? Were video-specific depth estimation models considered, and would they potentially offer more temporally consistent results that could benefit the analysis?
2.The discard rates for some models are notably high (e.g., 47% for PyramidalFlow in single-frame mode). This suggests a high rate of catastrophic failures before any fine-grained physics analysis can even be performed. Should this high rate of "unevaluable" generations itself be considered a primary metric for physical reasoning, perhaps weighted more heavily in the final assessment of a model's capabilities?
3.How robust is the evaluation pipeline to potential errors from the upstream object segmentation and tracking modules? Since the entire calculation of velocity and acceleration depends on the accuracy of the extracted centroids from SAM-2, even small tracking errors could be amplified into large errors in the final physics scores. Was any sensitivity analysis performed on this?
4.The paper states that the benchmark is restricted to Newtonian physics and assumes negligible friction and air resistance. However, these forces are always present in real-world videos. How does the framework handle a generated video that plausibly models friction (e.g., an object slowing down correctly), which would violate the ideal conservation of energy assumption? Could this lead to physically realistic videos being unfairly penalized?

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
Summary:   
This paper addresses the critical limitation of current video generative models (VGMs): their inability to adhere to fundamental physical laws despite producing visually realistic content. To solve this, the authors introduceMorpheus, a novel benchmark for evaluating VGMs’ physical reasoning capabilities using real-world physical experiments.  

Contributions:  
（1）Innovative quantitative evaluation framework: The combination of Dynamical Score (PINN-based equation fitting) and Physical Invariance Score (invariant conservation) provides objective, fine-grained metrics for physical plausibility, resolving the limitations of prior methods that fail to quantify subtle physical violations.  
（2）Extensive VGM evaluation and actionable insights: The large-scale evaluation (6 models, 9000 videos) identifies critical gaps and reveals that multi-frame conditioning and prompt enhancement modestly improve physical realism—guiding future VGM optimization.  
（3）Visually diverse conditioning for robust evaluation: Morpheus augments real-world videos with style transfers, ensuring evaluations are robust to visual variations in VGM training data.

### Strengths
Strengths:   
（1）Originality  
Morpheus, its proposed benchmark, is the first to use controlled real-world physics experiments for VGM physical reasoning evaluation. It designs a dual-metric (Dynamical + Physical Invariance Score) framework with PINNs for objective physical plausibility measurement. It also uses style transfers for diverse conditioning, ensuring robust evaluations.  
（2）Quality  
Its dataset comes from 9 controlled experiments (varied initial conditions) and metrics are validated with real videos. It evaluates 6 VGMs across 9000 videos, with systematic analyses of prompts and conditioning modes. Ablations isolate variable impacts to avoid cherry-picking.  
（3）Clarity  
It follows a clear “gap→solution→validation” flow, explaining complex concepts simply. Consistent terms and visual aids help readers follow design and results easily.  
（4）Significance  
As an open tool, it provides a shared platform for tracking VGM progress, guiding real-world uses. It shifts VGM evaluation to prioritize physical correctness and aligns with scientific integrity.

### Weaknesses
Weaknesses:    
（1）Narrow Physics Scope  
Morpheus only covers Newtonian physics in controlled lab settings, excluding domains like fluid dynamics or electromagnetism. It assumes negligible air resistance/friction, which may unfairly penalize plausible generations accounting for these real-world factors.  
（2）Uncalibrated 2D Trajectory Extraction  
It uses 2D pixel trajectories without camera calibration to real-world units, leading to ambiguous measurements. This risks inaccurate Dynamical/Physical Invariance Scores, especially for experiments needing precise distance/velocity data.  
（3）The comparison with the methods of predecessors is not sufficient.    
A comparison between the dataset and some previous related datasets, such as VideoREPA, WISA, NewtonGen, T2vphysbench, etc.?    
（4）Missing Closed-Source VGM Evaluation  
It excludes closed-source models (e.g., SORA) due to budget limits, omitting state-of-the-art performance. This weakens benchmark comprehensiveness, as readers can’t assess the full VGM spectrum.  
（5）Biased Time Window for Scores  
Physical Invariance Scores use different time window sizes for real (10% duration) and generated (25% duration) videos. No justification is given, and this inconsistency may artificially skew scores.

### Questions
（1）Physics Scope Limitation  
Your benchmark Morpheus only covers Newtonian physics in controlled laboratory settings and excludes domains like fluid dynamics or electromagnetism. Do you have plans to expand Morpheus to include these broader physical domains in future updates? Additionally, since you assume negligible air resistance and friction, how would you adjust the evaluation metrics to avoid penalizing physically plausible generations that account for these real-world factors?  
（2）2D Trajectory Extraction  
Morpheus uses 2D pixel trajectories without camera calibration to convert pixels into real-world physical units (e.g., meters, seconds). Have you tested whether adding camera calibration would reduce ambiguity in Dynamical and Physical Invariance Scores, especially for experiments requiring precise distance or velocity measurements? If not, what challenges prevent implementing such calibration?  
（3）Closed-Source VGM Inclusion  
You exclude closed-source VGMs (e.g., SORA) due to budget limits, which weakens the benchmark’s comprehensiveness. We suggest partnering with providers of closed-source models for limited access, or using publicly available generated videos from these models (if accessible) to include them in future evaluations. This would allow readers to assess how state-of-the-art closed-source models perform against Morpheus’ metrics.  
If the author can effectively solve my doubts, I will consider improving my score.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies the physics evaluation of video generation models. Current evaluation methods rely on subjective judgements or trajectory matching. This paper introduces a new benchmark on physical reasoning, which features 130 real world videos capturing physical phenomena guided by conservation laws. The paper shows that even with advanced prompting and video conditioning, the models still cannot encode physical principles very well

### Strengths
1. The paper introduces a benchmark to systemically evaluate physical reasoning of VGMs based on physical invariants using real world physical experiments.
2. The paper is clear and easy to read.

### Weaknesses
1. The benchmark only covers limited scenarios and has strong assumption of the environment. However, there are far more videos that might compose more complicated physics. 
2. The evaluation framework strongly relies on several other models like sam2, depth anything v2. I do not know in practice, if the framework can be generalized since each part can have some errors, which can be accumulated to influence the reliability of the final prediction of the morpheus score.
3. It does not include the latest SOTA VGMs like veo 3.1, sora 2, etc.

### Questions
1. How can this framework be generalized to other scenarios? We actually expect the model to generate very complicated physics dynamics now.
2. What is the performance on the latest close source VGMs?

### Soundness
2

### Presentation
3

### Contribution
2
