# ProJo4D: Progressive Joint Optimization for Sparse-View Inverse Physics Estimation

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Neural rendering has advanced in 3D reconstruction and novel view synthesis. With the integration with physics, it opens up new applications. The inverse problem of estimating physics from visual data, however, remains challenging, limiting its effectiveness for applications like physically accurate digital twin creation in robotics and XR. Existing methods that incorporate physics into neural rendering frameworks typically require dense multi-view videos as input, making them impractical for scalable, real-world use. Given sparse multi-view videos, the sequential optimization strategy used by existing approaches introduces significant error accumulation, e.g., poor initial 3D reconstruction leads to inaccurate material parameter estimation in subsequent stages. Instead of sequential optimization, simultaneous optimization of all parameters also fails due to the highly non-convex and often non-differentiable nature of the problem. We propose ProJo4D, a progressive joint optimization framework that gradually increases the set of jointly optimized parameters, leading to fully joint optimization over geometry, appearance, physical state, and material property. Evaluations on both synthetic and real-world datasets show that ProJo4D outperforms prior work in 4D future state prediction and physical parameter estimation, demonstrating its effectiveness in physically grounded 4D scene understanding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes ProJo4D, a progressive joint optimization framework for inverse physics estimation from sparse-view video. It identifies a key failure mode in prior sequential optimization methods: error accumulation from poor initial 3D reconstruction under sparse-views. ProJo4D avoids this by using a 3-stage curriculum: it first learns a 4D representation, then jointly optimizes for initial physical states (e.g., velocity), and finally conducts a full joint optimization including material properties. This progressive strategy is shown to dramatically outperform state-of-the-art methods like GIC and Spring-Gaus on standard sparse-view benchmarks.

### Strengths
The paper addresses a critical and practical bottleneck—the failure of physics estimation from sparse views. This is highly relevant for real-world robotics and XR applications. 

The experimental validation is exceptionally strong. The method demonstrates not just marginal but order-of-magnitude improvements (e.g., CD 16.11 $\rightarrow$ 1.60) over strong baselines on multiple datasets.

### Weaknesses
The abstract claims the optimization order is "guided by their sensitivity", but this concept is never defined, quantified, or justified in the paper. The chosen order (velocity, then material) appears to be an empirically-driven heuristic, not a formal principle.

The paper convincingly shows that ProJo4D works, but not how it avoids error accumulation. It lacks a crucial analysis (directly comparing the optimization trajectory of ProJo4D against the sequential (GIC) and full-joint (XASM ) methods.

### Questions
Could you please clarify the "sensitivity"  claim? Is the optimization order based on a formal analysis, or physical intuition?

Given the strong performance of the "XASM" ablation on simpler materials, would you agree that a more precise claim is that your progressive strategy acts as a robust regularizer for joint optimization, making it viable for complex materials where a naive joint approach fails?

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
The proposed ProJo4D framework aims to jointly estimate 3D Gaussian parameters, deformation models, initial velocity, and material properties in a progressive manner. Specifically, ProJo4D highlights its effectiveness under a sparse-view setting, where it outperforms other baselines in terms of future state prediction and physical parameter estimation.

### Strengths
1. The presentation is clear, and the background is investigated comprehensively. 

2. The optimization design of ProJo4D framework is neat overall.

### Weaknesses
1. The Deformation Network is only optimized at stage 0, which may not be correct. The later estimation for initial velocity and material parameters through physics simulation will rely on the predicted positions from the Deformation Network, which may not be reliable. 

2. It is unclear what the key is to the optimization strategy of ProJo4D. Is it the progressive estimation for different kinds of parameters? Or is it due to some parameters are optimized repeatedly across multiple stages? More ablations are needed.

3. Experiments can be fairer and more comprehensive: 

(1) It is suggested to compare the optimization cost (e.g., how many rounds of optimization are needed to estimate some parameters). 

(2) It is suggested to provide comparison results for dense views as well, and provide analysis on why ProJo4D is less sensitive to sparse view.

### Questions
1. For the comparison between different optimization strategies, can you make the total number of optimization stages comparable? For example, according to Table 1, position (X) in ProJo4D is optimized 4 times, while it is only optimized once in GIC. Experiments should be adapted accordingly to be fair. 

2. More ablations on the optimization strategy are needed. For example, instead of progressive optimization, what if parameters are estimated sequentially (X,A->S->M) but optimized for multiple rounds?

3. Is it necessary to include the Diff. Physics Simulation branch, given that the future state prediction is the ultimate goal as mentioned in line 442? Without the Physics Simulation branch, can you still estimate a reasonable future states from 3D Gaussian parameters and deformation models?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ProJo4D, a progressive joint optimization framework to learn physics parameters from sparse-view videos. In general, three stages are included in the training pipeline, progressively from learning a coarse 3D/4D representation to learning full set of physical parameters. Obvious performance gains are shown in both synthetic and real-world dataset.

### Strengths
1. The paper expresses its main target very clearly, making it easy to follow 
2. Although the method is simple, the improvement is obvious.

### Weaknesses
Although the method shows great performance in the experiments, there are still some conceptual or experimental weaknesses:
1. Since the stage 0 is purely learned with RGB supervision with no other constraints, it is possible that the learned deformation field does not obey the physical smoothness. Once the deformation field is learned, it will not be optimized further. So if there is issue in the first stage, the issue will accumulate to the later stages. The influence of enabling further refinement should be analyzed in later stages. 
2. The implementation details for the 4D representation learning stage is not well-elaborated. 
3. Ablation is not thorough enough. Performances after different stages should be reported.
4. Whether the performance gain is from the different ways in learning 4D representation or from the whole pipeline is not clear. The authors should report the reconstruction performance comparison between stage 0 and the 4D rep of GIC. 
5. It seems the ability to enble better sparse-view reconstruction is from the stage 0, which is not new. And the parameter estimation pipeline is not new either. So I doubt the novelty of this work. 

In general, where the performance gain comes from is not clearly demonstrated by the experiments in my opinion. So I can’t support the paper to be accepted. But I am willing to modify my ratings if the authors can address my concerns.

### Questions
1. Do ProJo4D and GIC share the same 4D reconstruction framework at the first stage?
2. Please refer to weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
ProJo4D innovatively proposes a "progressive joint optimization" paradigm, which addresses the core challenges of "error accumulation" and "non-convex optimization" in inverse physics estimation under sparse-view settings. Its "visual-physical bidirectional constraint" design provides a new perspective for the integration of neural rendering and physical simulation. The paper presents solid experimental results covering diverse scenarios and material types, and holds significant reference value for downstream applications such as robotic digital twins and XR physical interactions.

### Strengths
1.  Innovative Progressive Joint Optimization Paradigm Solves Core Sparse-View Challenges. ProJo4D addresses the two critical bottlenecks of inverse physics estimation under sparse views: error accumulation in sequential optimization and trapping in local minima in full joint optimization, by proposing a stage-wise variable expansion strategy. 
2. End-to-End Integration of 4D Dynamic Representation and Differentiable Physics Simulation. The framework tightly couples 3D Gaussian Splatting-based 4D dynamic representation with Material Point Method (MPM)-based differentiable physics simulation, forming bidirectional constraints between visual observation and physical laws.
3. Comprehensive and Rigorous Experimental Validation. Experiments cover diverse scenarios, material types, and view settings, ensuring convincing results.
4. Robustness to Complex Materials and Sparse Views. ProJo4D maintains stable performance across diverse material models.

### Weaknesses
1. The paper does not report key efficiency indicators such as per-frame optimization time or GPU memory usage.
2. It requires manual designation of material types and cannot handle unknown or mixed materials, limiting applicability to real-world scenes.
3. Experiments focus on 3-view settings, with no results for 2-view or single-view scenarios (common in real-world robotic monocular observation). It is unclear whether ProJo4D can maintain accuracy when view count drops further.
4. The "low-sensitivity first" optimization order is based on qualitative observation rather than quantitative analysis.
5. The weights of rendering loss and geometric loss are set empirically, with no analysis of how weights affect performance across materials/scenes.
6. The compared methods are relatively outdated, making it impossible to demonstrate the breakthrough of ProJo4D.

### Questions
1. The paper does not report key efficiency indicators such as per-frame optimization time or GPU memory usage.
2. It requires manual designation of material types and cannot handle unknown or mixed materials, limiting applicability to real-world scenes.
3. Experiments focus on 3-view settings, with no results for 2-view or single-view scenarios (common in real-world robotic monocular observation). It is unclear whether ProJo4D can maintain accuracy when view count drops further.
4. The "low-sensitivity first" optimization order is based on qualitative observation rather than quantitative analysis.
5. The weights of rendering loss and geometric loss are set empirically, with no analysis of how weights affect performance across materials/scenes.
6. The compared methods are relatively outdated, making it impossible to demonstrate the breakthrough of ProJo4D.

### Soundness
3

### Presentation
3

### Contribution
2
