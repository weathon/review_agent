# Velocity-Centric 4D Gaussian Splatting for Physical Realistic Dynamic Rendering

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Synthesizing novel views of dynamic scenes has long been a challenge in computer vision. While existing rendering methods have made progress with static scenes, they struggle to maintain temporal and spatial consistency, as well as physical plausibility, in dynamic scenes, often resulting in jerky motion and unrealistic physical effects. To address this, we propose Phys4DGS, a physically grounded framework that achieves high-fidelity and temporally coherent dynamic scene rendering. Phys4DGS introduces a velocity-aware physical consistency regularization that supervises motion across three complementary representations: intrinsic Gaussian motion attributes, geometric motion, and photometric motion. Furthermore, we introduce unit-time physical interval regularization, which stabilizes motion over time, ensuring continuous dynamics and temporal smoothness. Extensive experiments demonstrate that Phys4DGS outperforms leading methods on dynamic scene rendering, improving PSNR by 7.58 dB, reducing LPIPS by 80.00%, cutting training time by 72.22%, and increasing FPS by 175.48%, which ensures physically realistic, temporally consistent motion.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper first points out the core challenge of integrating physical consistency into efficient rendering frameworks to achieve smooth, realistic, and temporally coherent motion in dynamic scenes. To address this, the paper proposes a method called Phys4DGS, which introduces velocity-centric physical consistency regularization, multi-level motion alignment across intrinsic, geometric, and photometric domains, and a unit-time physical interval mechanism for temporal continuity. The method aims to ensure physically grounded and temporally consistent dynamic scene rendering with improved realism, smoothness, and efficiency.

### Strengths
* The paper introduces a comprehensive physical consistency framework that aligns velocity, displacement, and higher-order motion derivatives, effectively constraining Gaussian dynamics to obey physically plausible motion laws.

* The paper demonstrates strong generalization and robustness by evaluating Phys4DGS on multiple dynamic scene datasets, including both synthetic and real-world benchmarks such as the Plenoptic Video dataset. This comprehensive evaluation convincingly supports the method's scalability and applicability to a wide range of dynamic rendering scenarios.

### Weaknesses
* The experimental evaluation includes too few 3DGS-based baselines. Although the paper claims to solve the unrealistic motion problem in previous dynamic 3DGS methods, it lacks comparisons with recent 3DGS works, especially those using optical or scene flow constraints (e.g., MotionGS [1]). The limited baselines make it hard to judge the effectiveness and generality of the proposed method.

* The **Related Work** section discusses traditional, NeRF-based, and point-based methods in detail but lacks coverage of dynamic 3DGS works, which are central to this paper's focus. This gap raises doubts about the depth of understanding of related studies. Moreover, the discussion only includes works before 2024, missing recent progress. In addition, several NeRF-based methods are mixed into the Point-Based subsection, which should be reorganized for clarity.

* On the D-NeRF dataset, SCGS [2] achieves a PSNR of 43.31 and has been open-sourced (with the same 400×400 resolution), representing the current SOTA. Since SCGS performs significantly better than the proposed method, it should be included in the comparisons and discussed in the paper.

* In Figure 2, the highlighted static bottle in the first row raises some questions. Since the proposed method mainly focuses on improving motion modeling, it is unclear why the reconstruction quality on this static object is also significantly better than that of Deformable 3DGS. The authors highlight this result in the figure but provide no explanation or discussion in the text.

* The first part of the **Introduction** spends too much space discussing NeRF and static 3DGS. The authors should more quickly transition to the limitations of existing dynamic 3DGS methods and clearly introduce the specific problem this paper aims to solve.

* The paper's overall logic needs improvement. In Sec 3 **Method**, the first paragraph should clearly connect the main modules to give readers a quick overview of how the method is structured. Although Sections 3.1–3.3 describe three levels of consistency design, the logical links between them are weak. Each subsection starts with vague statements instead of explaining why a higher-level constraint is needed based on the previous one. For example, in line 233, "To complement velocity alignment" is too general, and the authors should explain why velocity consistency is not enough and why displacement consistency matters. The paper should not rely solely on ablation results to show effectiveness but also explain the methodological motivation behind each module.

[1] MotionGS: Exploring Explicit Motion Guidance for Deformable 3D Gaussian Splatting. NeurIPS 2024.

[2] SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes. CVPR 2024.

### Questions
See weaknesses.

### Soundness
3

### Presentation
1

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
This paper introduces Phys4DGS, a physically grounded framework for dynamic scene rendering that achieves both high fidelity and temporal coherence. The method employs a set of unit-time physical interval regularizations that jointly model geometric and photometric motions, including physical velocity consistency, displacement consistency, higher-order physical consistency, and temporal coherence constraints. This paper conducts comprehensive experiments that demonstrate the effectiveness of the proposed approach.

### Strengths
The unit-time physical interval regularizations derivations are mathematically sound and well-motivated. The theoretical formulation aligns closely with physical intuition, and the methodology is presented with clear definitions and supporting equations.

The experimental results are comprehensive and persuasive, providing solid evidence for the effectiveness of the proposed approach. 

The contribution is significant, as the proposed framework effectively addresses temporal coherence and fidelity in dynamic scene rendering for Gaussian Splatting in 4D spacetime.

### Weaknesses
Although the paper compares extensively with prior works and cites relevant literature, the experimental comparison lacks sufficient analysis regarding why baseline methods underperform relative to the proposed approach. The paper would benefit from a more detailed discussion about the specific limitations of existing methods in the context of dynamic scene rendering and temporal/fidelity constraints.

Since the central contribution of the paper is the formulation and use of unit-time physical interval regularizations, it would substantially strengthen the work to explicitly analyze and compare the regularization formulations of other baseline methods. Understanding the differences in regularization design would not only highlight the novelty of the proposed framework but also provide deeper insights into how these choices impact experimental results.

### Questions
1. **Minor typos:**

1) An additional ‘?’ is present before ‘Guo et al.’

2) In Figure 1, it should be ‘t+δt’ instead of ‘t,+δt.’

2. **Ablation study:** The qualitative result for ‘w/o 4D Temporal’ is presented in Figure 4, but is missing in Table 3. Is there any particular reason for this omission?

3. It would improve readability if the ablation studies were presented in a unified manner and followed the order described in the Method section.

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
3

### Summary
This paper proposes a new physically-grounded framework called Phys4DGS to achieve high-fidelity and temporally coherent dynamic scene rendering. Specifically, it aligns motion representations across multiple levels, ensuring that spatial trajectories and temporal variations remain coherent and grounded in geometric structure and observation data. A unit-time physical interval regularization, which enforces the consistency of dynamic physical features, i.e., velocity, across consecutive unit intervals, is proposed to preserve coherent motion trajectory. A regularization on high-order dynamics, i.e., acceleration and jerk, is proposed to ensure that rendered motion remains smooth and physically consistent across space and time. Experiments demonstrate that the proposed method achieves physically realistic and temporally consistent rendering in dynamic scenes with superior FPS and training efficiency.

### Strengths
1)	This paper aims to achieve physically consistent rendering in dynamic scenes, which is an important and interesting research topic.
2)	This paper proposes a new velocity-aware physically grounded framework called Phys4DGS for high-fidelity and temporally coherent dynamic scene rendering.
3)	A velocity-aware physically consistent regularization and a unit-time higher-order physical interval regularization are proposed in Phys4DGS, which ensure continuous dynamics and temporal smoothness.
4)	Experiments on public benchmark demonstrate the effectiveness of the proposed method.

### Weaknesses
1)	The logistic of Fig. 1 is not very clear, since the details of the corresponding regularization is not obviously denoted. Besides, new symbols/modules are not clearly introduced in the caption. It is difficult to understand the framework in Fig. 1.
2)	Line 371: Dataset -> dataset.
3)	The proposed framework introduces multi-level regularization, so why could it still achieve fast training and FPS? The time complexity of the proposed modules in the framework should be analyzed.
4)	Figure 4 is too small to recognize the obvious improvement of the proposed method.

### Questions
Please try to address the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a physically realistic and temporally consistent dynamic modeling framework based on 4D Gaussian Splatting (4DGS). The method introduces velocity-centric physical regularizations to enforce consistency across time, aiming to produce physically plausible motion.

### Strengths
The reported quantitative results are impressive — the method achieves significantly higher rendering quality (up to 7 dB improvement over existing approaches at the level of 30 PSNR), suggesting strong potential for dynamic scene representation.

### Weaknesses
It is not fully convincing that introducing physical regularizations — which are not directly optimized for rendering metrics — can improve both physical correctness and rendering quality simultaneously. Regularization typically imposes trade-offs (e.g., 2DGS often sacrifices rendering metric for geometric accuracy, and method other 3DGS for normals or depths model). Therefore, it remains unclear whether the performance improvement comes from physical consistency or other side effects.

If the paper claims physical consistency, it would be more convincing to include comparisons with existing physics-based dynamic modeling methods with ground truth data, such as differentiable physical simulation. They can provide ground-truth physical data label (e.g velocities, accelerations). Such comparisons would allow the correctness of the proposed model to be quantitatively evaluated using physical metrics, rather than relying solely on qualitative visualizations (e.g., optical flow plots) or rendering quality scores in the table.

The figure qualtiy looks ugly.

### Questions
Could you compare the same gaussian points trained at different runs to verify the physical consistency

For example, if we train the model with the same scene in two separate runs, we can compare the velocity and acceleration of the same Gaussian points to check whether they exhibit similar or physically consistent values.


Could you please provide more details about the FPS results reported for RealTime4DGS and Deformable4DGS in Table 1? Were these values measured by you under your experimental settings, or are they referenced directly from other papers?

### Soundness
2

### Presentation
2

### Contribution
3
