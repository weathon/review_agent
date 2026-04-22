# GAMBIT: A Graph-structured and Decision-Aware Benchmark for MoBile GUI Tasks

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Mobile GUI agents powered by LMMs can perceive screens and follow instructions, yet existing benchmarks largely target short, linear workflows and step-level accuracy, offering limited insight into long-horizon planning and branching tasks. We present GAMBIT, a graph-structured, decision-aware benchmark comprising 830 task episodes and 11,345 actions across 35 applications on Android and iOS. Tasks are organized into Sequential, Conjunctive, Conditional, and Hierarchical workflows with dual-level annotations, capturing realistic multi-step and branching scenarios. To move beyond step metrics, we introduce weighted longest common subsequence for length-sensitive progress and decision accuracy for branch correctness. Evaluations on 7 diverse agents show that GAMBIT induces a substantial accuracy drop compared to prior datasets, with success rates falling below 5% on 6–8 step tasks and branch accuracy averaging 38%, underscoring weaknesses in conditional reasoning. By systematically exposing these failure modes, GAMBIT provides a challenging, diagnostic testbed for advancing decision-aware mobile GUI agents. Our code and dataset are available at: https://anonymous.4open.science/r/GAMBIT-40BB/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces GAMBIT, a new benchmark designed to evaluate mobile GUI agents on long-horizon, decision-aware tasks across Android and iOS apps. Unlike existing datasets focused on short, linear workflows, GAMBIT includes over 800 task episodes with branching structures and provides dual-level annotations. The authors also propose decision-sensitive evaluation metrics, such as weighted LCS and branch accuracy, and evaluate 7 models, finding sharp performance drops in complex settings. GAMBIT reveals significant challenges in current agents’ ability to reason, plan, and adapt in realistic mobile scenarios.

### Strengths
GAMBIT addresses a timely and underexplored challenge—evaluating mobile GUI agents on complex, decision-aware tasks—through the design of a well-constructed benchmark. GAMBIT is notable for its diversity (830 tasks across 35 apps), realistic graph-structured workflows, and dual-level annotations that support both fine-grained and high-level evaluation. The proposed metrics, particularly weighted LCS and decision accuracy, offer a more nuanced view of agent performance beyond traditional step-level success. The experimental evaluation is thorough, spanning seven competitive agents and revealing significant performance gaps that highlight the difficulty and diagnostic value of the benchmark.

### Weaknesses
* Although the paper claims that code and data are hosted on an anonymous site, the link was space.

* The experimental results show that all evaluated agents perform poorly on complex decision-aware tasks, with success rates dropping below 5% on longer or branching workflows. While this highlights the benchmark's difficulty, the paper stops short of analyzing *why* agents fail. A more detailed breakdown—e.g., by reasoning failures, perceptual grounding issues, or instruction misinterpretation—would clarify which capabilities current models lack. Additionally, exploring whether training on a subset of the benchmark improves performance on held-out complex tasks would help assess whether these challenges are surmountable with current architectures.

* Although the dataset includes a nontrivial portion of cross-app tasks (~12.5%), the paper does not explicitly analyze how agent performance varies between single-app and cross-app scenarios. Given the increased complexity of app-switching workflows, a focused breakdown would enhance understanding of where current models struggle and how to better design agents for multi-app interactions.

### Questions
* Given the extremely low success rates on longer or branching tasks, can the authors provide more detailed analysis of failure cases? Specifically:

   * Are these failures more often due to reasoning errors, perception/grounding problems, or instruction misunderstanding?
   * Could a few concrete examples be shared to illustrate common error modes?
   * Have you considered analyzing errors by task topology (e.g., are hierarchical tasks disproportionately prone to early failure)?

 * Have you considered training or fine-tuning any models on a subset of GAMBIT, such as shorter or less-branching tasks, and evaluating on held-out complex ones? This would help clarify whether current models can improve with exposure, or whether structural architectural innovations are required.

* While GAMBIT includes a nontrivial portion of cross-app tasks, there appears to be no separate evaluation or discussion of how agents perform on these compared to single-app workflows. Could the authors report metrics stratified by cross-app vs. single-app scenarios to better understand the specific difficulties introduced by cross-context transitions?

### Soundness
2

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
This paper introduces GAMBIT, a new benchmark for evaluating mobile GUI agents (with more then 800 episodes). Unlike existing benchmarks that primarily focus on simple, linear tasks, GAMBIT presents complex, graph-structured challenges that involve decision-making and branching scenarios to better reflect real-world usage.

### Strengths
1. The paper is well-structured and clearly written, making it easy to follow.
2. The contribution of a new, open-source benchmark dataset is a key strength and a valuable resource for the community.
3. The paper provides a thorough experimental evaluation, including comprehensive comparisons with existing agents.

### Weaknesses
1. The figures in the paper are of low resolution and are difficult to read.
2. The core innovation appears to be more complex high-level instructions (i.e., adding more conditions and logical judgments), which may not be a substantial enough contribution. The authors need to better justify this.

### Questions
1. Fundamental difference from existing datasets: For a "Conditional" or "Hierarchical" episode, are the underlying sequences of screenshots and step-level instructions still inherently sequential? If this is the case, the main distinction of GAMBIT seems to lie only in the complexity of the high-level instruction. Could a similar dataset be constructed by rewriting existing sequential datasets? For instance, by augmenting them with app-specific atomic instructions and constraints, one could transform sequential data into Conditional or Hierarchical tasks. The authors should add a discussion or experiments in the paper to address this point.
2. Unrealistic instructions: In real-world scenarios, users are unlikely to provide such long and detailed instructions. They tend to offer shorter constraints. The example in Figure 1 illustrates this well: a user is more likely to say, "Help me find the cheapest, non-smoking, pet-friendly accommodation in Amsterdam," rather than articulating the complex if-else logic shown in the blue box. The authors need to re-evaluate the plausibility of these high-level instructions. If they do not accurately reflect real user behavior, they should be considered for rewriting to be more naturalistic.
3. About atomic instructions: How many atomic instructions and constraints were generated for each application? This number directly impacts the diversity of the final dataset. If the variety and quantity of these building blocks are insufficient, the dataset's overall diversity will be limited. The authors should provide an analysis and present statistics on this.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GAMBIT, a new benchmark designed to evaluate mobile GUI agents in complex, graph-structured, and decision-aware task environments. The benchmark features 830 task episodes (totaling over 11,000 actions) across 35 Android and iOS apps, with a variety of task topologies including sequential, conjunctive, conditional, and hierarchical workflows. Tasks are annotated at both high and low levels, and the authors propose new evaluation metrics (weighted LCS and decision accuracy) to more effectively measure long-horizon progress and decision correctness. Extensive experiments on seven current agents demonstrate GAMBIT’s significant difficulty and diagnostic value, exposing key deficiencies in decision-aware and conditional reasoning under mobile GUI interaction.

### Strengths
1. GAMBIT advances the field by offering a benchmark that moves decisively beyond template-based and sequential task formulations, emphasizing branching, conditional, and hierarchical dependencies. The graph-based modeling of tasks expands coverage over prior GUI datasets.

2. The benchmark’s 830 graph-structured tasks, spanning multiple app categories, platforms (Android/iOS), and languages (English/Chinese), offer significantly more realistic complexity than previous datasets. Figure 1 and Figure 2 make the contrast between simple sequential templates and graph-structured tasks explicit, showing how real-world constraints and decisions are integrated.

### Weaknesses
1.  The methodology and evaluation sections do not discuss or compare against several directly relevant benchmarks from the broader graph learning literature. Notably, there is no mention of GSLB (Li et al., 2023) or GC-Bench (Sun et al., 2024), which set standards for graph-structured benchmarks and could inform both task modeling and metric choices in GAMBIT. Incorporating a principled comparison or at least discussion of these would sharpen the benchmark's positioning and theoretical underpinnings.

2. While the weighted LCS and decision accuracy metrics are described at a high level (Section 3.6, Equation W-LCS), critical formal and implementation details are missing. For example, the exact procedure for weighting within branching workflows (i.e., how weights are propagated on variable-length branches, how conflicting actions in parallel branches are resolved) is not fully specified. For decision accuracy, the criteria for identifying branch points and their correct traversal could be formalized, perhaps as a function $D(\hat{\mathcal{T}}, \mathcal{G})$ given the prediction $\hat{\mathcal{T}}$ and gold graph $\mathcal{G}$. This lack of mathematical rigor makes reproducibility and external comparison more difficult and reduces trust in the fairness of new metrics.

3. While the empirical pipeline is thorough, the paper does not attempt to theoretically ground the choice of task structures, guard conditions, or evaluation metrics. For instance, it is unclear whether the four chosen graph topologies (Figure 2(b)) cover the space of real-world app workflows (or could be unified under a broader formalism). There is also no formal complexity analysis (e.g., task branching factor distribution), nor a justification (proof or constructivist argument) of metric sensitivity or discriminative power over prior best practices (EM, SR, GP).

4. Insufficient comparison/enumeration of alternative metrics: Although GAMBIT introduces new metrics, the paper under-delivers on a rationale for why the proposed design is preferable to other sequence/graph alignment metrics (e.g., edit distance, graph isomorphism, tree edit distance), or whether decision accuracy is robust to minor label annotation inconsistencies. This leaves open whether the evaluation suite truly advances the measurement of agent competence.

5. Table 4 and appendix tables reveal stark differences in per-action model performance (e.g., on Long Press and Complete/Stop). Yet there's little explanation or modeling of possible action ambiguity (multiple equivalent ways to complete a task), nor an attempt to quantify inter-annotator agreement on low-level/stepwise execution.

6. For tasks involving decisions or fallback paths (hierarchical/conditional), the sampling procedure for branches (e.g., negative/”IMPOSSIBLE” cases) is only cursorily discussed. How are ambiguous, unreachable, or failed state paths annotated and incorporated in analysis? Are rejected paths equally weighted when scoring W-LCS or decision accuracy?

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes GAMBIT, a graph-structured and decision-aware benchmark designed for evaluating mobile GUI agents on long-horizon and complex tasks. It features diverse graph topologies, covers both Android and iOS platforms, and includes cross-application scenarios. It also proposes novel evaluation metrics beyond task success rate and step accuracy.

### Strengths
- The focus on long-horizon, decision-aware complex tasks is well-motivated and addresses a clear gap in existing mobile GUI benchmarks.

- Representing complex tasks using graph structures is innovative.

- The benchmark is comprehensive, covering cross-platform (Android and iOS) and cross-app scenarios.

### Weaknesses
- The benchmark remains offline and static. It is unclear how it handles evaluation scenarios where multiple valid action trajectories or paths exist for completing a task, which is common in real-world GUI interactions.

- The experimental evaluation lacks results from state-of-the-art closed-source models (e.g., Claude, Gemini), limiting the analysis of actual task difficulty

- Many descriptions in the paper are unclear and ambiguous. For instance, the "dual-layer quality control" is mentioned in the paper but not elaborated on subsequently.

### Questions
- How does the benchmark's graph architecture account for tasks where multiple valid trajectories can lead to successful completion? Are all valid paths considered in the ground truth or metrics?

- Could you please clarify what the "dual-layer" in "dual-layer quality control" means? 

- The benchmark is claimed to be representative of everyday usage but lacks justification, especially compared with other other benchmarks like AndroidWorld[1], SPA-bench[2].

[1] Rawles C, Clinckemaillie S, Chang Y, et al. Androidworld: A dynamic benchmarking environment for autonomous agents[J]. arXiv preprint arXiv:2405.14573, 2024.

[2] Chen J, Yuen D, Xie B, et al. Spa-bench: A comprehensive benchmark for smartphone agent evaluation[C]//NeurIPS 2024 Workshop on Open-World Agents. 2024.

### Soundness
3

### Presentation
2

### Contribution
2
