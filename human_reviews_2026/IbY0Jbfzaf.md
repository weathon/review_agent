# RiboPO: Preference Optimization for Structure- and Stability-Aware RNA Design

- Decision: Reject
- Scores: 4, 4, 8, 2

## Abstract
Designing RNA sequences that reliably adopt specified three-dimensional structures while maintaining thermodynamic stability remains challenging for synthetic biology and therapeutics. Current inverse folding approaches optimize for sequence recovery or single structural metrics, failing to simultaneously ensure global geometry, local accuracy, and ensemble stability—three interdependent requirements for functional RNA design. This gap becomes critical when designed sequences encounter dynamic biological environments.
We introduce **RiboPO**, a **Ribo**nucleic acid **P**reference **O**ptimization framework that addresses this multi-objective challenge through reinforcement learning from physical feedback (RLPF). RiboPO fine-tunes gRNAde by constructing preference pairs from composite physical criteria that couple global 3D fidelity and thermodynamic stability. Preferences are formed using structural gates, pLDDT geometry assessments, and thermostability proxies with variability-aware margins, and the policy is updated with Direct Preference Optimization (DPO). On RNA inverse folding benchmarks, RiboPO demonstrates a superior balance of structural accuracy and stability. Compared to the best non-overlap baselines, our multi-round model improves Minimum Free Energy (MFE) by **12.3%** and increases secondary-structure self-consistency (EternaFold scMCC) by **20%**, while maintaining competitive 3D quality and high sequence diversity. In sampling efficiency, RiboPO achieves **11% higher pass@64** than the gRNAde base under the conjunction of multiple requirements. A multi-round variant with preference-pair reconstruction delivers additional gains on unseen RNA structures. These results establish RLPF as an effective paradigm for structure-accurate and ensemble-robust RNA design, providing a foundation for extending to complex biological objectives.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces RiboPO, a framework for RNA inverse folding that applies Direct Preference Optimization (DPO), framed as Reinforcement Learning from Physical Feedback (RLPF), to RNA sequence generation. The method aims to jointly optimize structural accuracy and thermodynamic stability. RiboPO employs a multi round DPO process anchored by supervised fine tuning. The paper also proposes the SSTT Benchmark, an evaluation suite assessing quality across Sequence, Secondary Structure, Tertiary Structure, and Thermostability.

### Strengths
1. The paper systematically applies the DPO paradigm, borrowed from the protein design community, to the challenging RNA inverse folding task. This systematic transfer is useful, even if the underlying concept is not novel.

2. The authors construct the SSTT Benchmark, a necessary and valuable contribution. By assessing four complementary dimensions, it provides a more holistic performance evaluation than prior benchmarks focused only on single metrics like RMSD or sequence recovery.

3. RiboPO reports quantitative improvements compared to several baselines, suggesting that the DPO framework can be effectively implemented to balance multi objective constraints in RNA design.

### Weaknesses
1. The core RLPF/DPO framework is a direct, near trivial transfer of established techniques from protein design (or general sequence generation) to the RNA domain. The paper fails to introduce any significant RNA specific adaptations or novel mechanisms that justify the claim of an "innovative" framework. The contribution is thus fundamentally one of application, not invention.

2. The paper offers no theoretical justification for applying DPO (a preference learning objective) to optimize physical criteria (like MFE and pLDDT). This lack of explanation for why DPO is the appropriate objective for this specific physical optimization task is a major theoretical weakness, suggesting a mechanical application without deep understanding.

3. The central claim about the optimal performance arising from a "fixed reference multi round DPO with a margin curriculum" is purely empirical and anecdotal. Without any theoretical analysis or discussion on the behavior of the margin/curriculum in the context of the RNA energy landscape, this conclusion remains an unprincipled observation.

4. As noted, the explicit overlap between the test set and the training data of baseline models (RiboDiffusion, RDesign) is a critical flaw. This data leakage fundamentally compromises the validity of the reported performance improvements. Strictly de duplicated results are an absolute requirement.

5. Relying solely on the DAS dataset split and lacking validation on complex, real world, or experimentally characterized RNA structures raises severe doubts about RiboPO's robustness and generalization capability.

### Questions
The content in this section is identical to the points raised in Weaknesses. I would be very willing to increase my score if the authors successfully address these concerns.

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
This paper presents RiboPO, a reinforcement-learning–from–physical-feedback (RLPF) framework for RNA inverse folding. The method fine-tunes the gRNAde base model using Direct Preference Optimization (DPO) with composite physical criteria (geometry + thermodynamic stability). A multi-round curriculum strategy is employed to progressively refine the policy.
The authors also propose a comprehensive SSTT benchmark that evaluates sequence, secondary structure, tertiary structure, and thermostability.

Empirical results show notable improvements on secondary-structure self-consistency (scMCC +20%) and minimum free energy (MFE −12.3%) over baselines, though sequence recovery slightly declines.

### Strengths
* Conceptual novelty: Reformulating RNA inverse folding as multi-objective preference optimization is conceptually elegant and provides a unifying view linking structural and thermodynamic optimization.
* Well-analyzed framework: The round-wise preference construction and curriculum-based DPO training are systematically ablated, with clear evidence of trade-offs among objectives.
* Comprehensive evaluation: The SSTT benchmark covers geometric, energetic, and sequence-level properties in a single standardized framework, which can be valuable to the community.

### Weaknesses
## Lack of visual RNA design analysis:
The paper does not include any visual examples of designed RNA structures (e.g., 2D secondary structure plots or 3D conformational overlays).
In RNA design literature (e.g., RiboDiffusion, RDesign, RhoDesign), such visualizations are essential to demonstrate whether generated sequences structurally resemble the target folds.
The authors should provide a figure comparing RiboPO’s designs with ground truth (e.g., native vs. designed structure overlays) and discuss why the proposed method leads to more realistic or stable conformations.
## Decline in sequence recovery (Rec) metric:
Table 1 shows that RiboPO consistently underperforms gRNAde on the recovery metric (0.53 → 0.50).
Since recovery remains one of the most interpretable metrics in inverse folding, this drop raises concern:
* If RiboPO fine-tunes gRNAde, why does the sequence fidelity degrade?
* Does the model over-prioritize thermodynamic energy at the expense of biological plausibility?
A detailed analysis or visualization of where the recovery loss occurs (e.g., base-pair positions, local regions) would help clarify.
## Unclear benefit of multi-round optimization:
The paper emphasizes multi-round refinement, yet Table 1 shows that the second round achieves the best result while subsequent rounds (e.g., Round 4) show regression.
This pattern raises doubts about the necessity and stability of multi-round training.
A clearer justification—possibly with intermediate visualization of metric trajectories—should be provided to demonstrate that multi-round optimization is systematically beneficial rather than an overcomplication.
## Missing Pareto-front analysis of multi-objective trade-offs:
Since the paper explicitly frames the task as multi-objective optimization (balancing structural fidelity and thermostability), a Pareto analysis is expected.
For example, showing how models at different rounds or β-values lie along a Pareto front between recovery and MFE (or scMCC and RMSD) would concretely illustrate the claimed “balanced optimization.”
Currently, improvements in one dimension often coincide with regressions in another, making it unclear whether RiboPO achieves genuine Pareto superiority compared to single-objective baselines.
Without this, the claimed novelty in “multi-objective preference optimization” remains somewhat superficial.
## Lack of time and efficiency analysis:
As a fine-tuning framework intended for practical use, RiboPO should report runtime or sample-efficiency comparisons with baselines (e.g., gRNAde or RiboDiffusion).
Even brief statistics on training time, inference latency, or the computational cost of physical-feedback evaluation would enhance the paper’s practical credibility.
## Minor technical remarks:
* The choice of fixed reference policy is reasonable but could limit exploration; some discussion of adaptive or periodically updated references would be welcome.
* The method relies heavily on RhoFold+ and ViennaRNA outputs; this dependence could bias learning toward those models’ heuristics.

### Questions
## On missing visual RNA design evidence
* Could the authors provide qualitative visualizations (e.g., secondary structure diagrams, 3D overlays, or contact maps) comparing RiboPO-generated designs with ground truth backbones?
* How do these visualizations demonstrate that RiboPO’s sequences fold more stably or accurately than those from gRNAde or RiboDiffusion?
## On the drop in sequence recovery (Rec)
* The recovery metric drops notably from 0.53 to 0.50.
Can the authors analyze where this degradation occurs (e.g., specific structural regions or base-pairing positions)?
* Is the reduction a consequence of stronger thermodynamic regularization, and can it be mitigated without sacrificing stability?
## On the necessity and stability of multi-round optimization
* Table 1 shows that Round 2 yields the best results, while later rounds (e.g., Round 4) regress on several metrics.
Could the authors clarify whether multi-round training consistently improves performance, or if it risks over-optimization?
* Is there evidence (e.g., metric trajectories or intermediate checkpoints) that demonstrates the systematic benefit of the multi-round scheme?
## On the missing Pareto-front and multi-objective analysis
* Since RiboPO is framed as a multi-objective optimization balancing geometry and thermodynamics, can the authors provide a Pareto-front visualization showing trade-offs between key objectives (e.g., MFE vs. RMSD or Rec vs. scMCC)?
* How does RiboPO achieve Pareto superiority compared to single-objective baselines like gRNAde or RiboDiffusion?
## On runtime and practical applicability
* What is the computational cost per training round compared with baseline fine-tuning (e.g., gRNAde)?
* How many sequences or preference pairs are required to reach convergence?
* Given that RiboPO is presented as a practical fine-tuning framework, can the authors discuss its runtime efficiency and potential for integration into RNA design workflows?
## On reliance on surrogate predictors
* Since the feedback signals come from RhoFold+ and ViennaRNA, how robust are the results to replacing them with other predictors (e.g., NUPACK or RNAstructure)?
* Could the model be overfitting to the specific biases of these tools rather than learning transferable physical principles?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper tackles the problem of structural/ensemble property optimization in RNA molecule inverse design. The proposed methodology focusses on Reinforcement Learning and preference optimization algorithms tailored for 3D RNA structure. The experimental evaluation shows the benefits of RL-driven property optimization compared to baselines trained with supervised learning, from both a statistical as well as biological relevance.

### Strengths
- The problem being tackled is extremely significant, as RNA inverse design methods are becoming practically used. The paper identifies a key research gap and tackles the problem of RL-driven property optimization in RNA design very well. I particularly want to commend authors’ efforts to develop techniques that are not just copying what is done in the proteins-ML realm, but to do something that’s original and RNA-specific. To the best of my knowledge, this is the first work to tackle this important problem.

- I really enjoyed reading this paper. The exposition does a very good job at presenting technical/methodological ideas and motivating them with ideas from RNA biology, as well as interpreting the results not only from a statistical perspective but also from a biological lens. The use of bolded sentences really makes the manuscript easily graspable immediately.

- The evaluation and experimental setup is rigorous and described in great depth. The ablation studies rigorously analyse various components of the proposed methodology well, and quantify how much each contributes to overall performance.

- I found the results convincing. They support the main claims of the paper well. The analysis of the results goes into sufficient depth about the implications and findings.

- Overall, I believe this is a high quality paper tackling an important problem. I believe that the results and model (if open source) will be of considerable interest to both ML and RNA biology communities, especially as methods like gRNAde and RhoDesign have been validated in wet lab experiments.

### Weaknesses
Though not necessarily a weakness of this paper alone, the quality of RNA 3D structure prediction is pretty poor at the moment. Thus, its not surprising that the method does not yet lead to significant gains in terms of 3D metrics over gRNAde, as the structure predictor being used is not reliable. I believe that upcoming 3D structure predictors could push the state of the art further, and then further improve RiboPO as well.

Other than these, I do not see any major weaknesses worth highlighting with this paper. I think its in excellent shape, but I will watch out for other reviewers’ concerns.

### Questions
I don’t have major questions. 

I have some minor comments:
- I would be interested to see one or two case studies where the RiboPO model improves the ensemble/structural properties of some designs compared to the base model (gRNAde).
- Line 51: Ganser et al citation is for the wrong line, I believe.
- Consider adding some variances/error bars/standard deviations to all of the results reported in tables.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper incorporates structural and thermodynamic criteria into RNA design by applying Direct Preference Optimization (DPO) to fine-tune an existing RNA generative model.

### Strengths
1. Introducing multi-objective optimization into RNA design is a meaningful and timely direction.
2. RiboPO demonstrates improved secondary-structure consistency and thermodynamic metrics.

### Weaknesses
1. The paper introduces a highly complicated evaluation framework, SSTT (Section 3.4), which consists of 15 different metrics. However, most of these metrics are not actually used in the model optimization process. Preference construction relies only on pLDDT, RMSD, and MFE, and the reported results primarily focus on a small subset of about seven metrics. As a result, the necessity and practical value of introducing such a complex evaluation framework are unclear.
2. The central motivation of the paper is to use DPO to encourage the generation of thermodynamically stable RNA structures. However, the approach to “stability” is based solely on ViennaRNA’s minimum free energy (MFE) prediction. MFE is an inadequate surrogate for true thermodynamic stability, and the predicted MFE structure often does not correspond to the experimentally adopted conformation. Therefore, optimizing MFE does not necessarily imply improved thermodynamic robustness or biological stability of the designed RNAs.
3. The MFE structure is typically not the experimentally determined structure present in the dataset, nor is it the structure predicted by data-driven models such as RhoFold or EternaFold. The paper combines predictions from mutually inconsistent tools: RhoFold is used to predict tertiary structure, EternaFold is used to predict secondary structure, and ViennaRNA is used to derive the MFE structure for energy estimation. These three tools produce different structures, meaning that the optimization process is guided by multiple, incompatible structural targets. Using these conflicting structural definitions simultaneously raises concerns about the biological validity of the optimization objective.
4. The paper does not explicitly address the trade-off between multiple objectives. Although the preference construction incorporates both structural fidelity and energy, the relative priority between these two objectives is never clarified. It remains unclear how the method balances potentially conflicting goals, or how the optimization procedure avoids collapsing toward one objective at the expense of the other. Without a principled mechanism for multi-objective trade-off, it is unlikely that the method can reliably achieve a desirable balance.
5. In the ablation study, removing the SFT loss improves the energy objective while degrading structural metrics. This observation does not necessarily justify the inclusion of the SFT term; rather, it simply indicates that the model has moved to a different point on the Pareto frontier, possibly by chance. This further highlights the need for an explicit treatment of multi-objective trade-offs, which is currently lacking in the paper.
6. The paper states that “In Section 4.4.2 we show that these shifts correspond to movement into more designable basins” (line 112). However, the term “designable” is not clearly defined, nor is the notion of a “designable basin” formally introduced or supported. Moreover, I was unable to locate a Section 4.4.2 in the paper.
7. In Figure 1, the filter condition includes an MFE criterion, but this term does not appear in Equation (3).

### Questions
See the weaknesses part.

### Soundness
2

### Presentation
2

### Contribution
2
