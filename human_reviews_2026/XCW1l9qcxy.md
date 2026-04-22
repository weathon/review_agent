# Transductive Visual Programming: Evolving Tool Libraries from Experience for Spatial Reasoning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Spatial reasoning in 3D scenes requires precise geometric calculations that challenge vision-language models. Visual programming addresses this by decomposing problems into steps calling specialized tools, yet existing methods rely on either fixed toolsets or speculative tool induction before solving problems, resulting in suboptimal programs and poor utilization of induced tools. We present Transductive Visual Programming (TVP), a novel framework that builds new tools from its own experience rather than speculation. TVP first solves problems using basic tools while accumulating experiential solutions into an Example Library, then abstracts recurring patterns from these programs into reusable higher-level tools for an evolving Tool Library. This allows TVP to tackle new problems with increasingly powerful tools learned from experience. On Omni3D-Bench, TVP achieves state-of-the-art performance, outperforming GPT-4o by 22% and the previous best visual programming system by 11%. Our transductively learned tools are used 5x more frequently as core program dependency than inductively created ones, demonstrating more effective tool discovery and reuse. The evolved tools also show strong generalization to unseen spatial tasks, achieving superior performance on benchmarks from SpatialScore-Hard collection without any testset-specific modification. Our work establishes experience-driven transductive tool creation as a powerful paradigm for building self-evolving visual programming agents that effectively tackle challenging spatial reasoning tasks. We release our code at https://transductive-visualprogram.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces an advanced visual programming method for 3D spatial understanding. The method starts with basic tools and automatically generates, stores, and optimizes useful tools (functions) while observing and solving problems. The reported quantitative results on 3D spatial reasoning benchmarks surpass all baselines, demonstrating the powerful potential of the method.

### Strengths
- The paper is well-written. Figures 1 and 3 clearly present the motivation and method.

- The self-evolving pipeline is carefully designed. The composition of spatial functions is novel in this area.

- The quantitative results on the sampled SpatialScore-Hard are promising.

- The case study and evolution curve effectively demonstrate the impact of evolution

### Weaknesses
Visual programming for spatial reasoning has also been used in 3D vision tasks, such as 3D visual grounding [1, 2], and [2] also employs a self-evolving process for spatial reasoning. The authors may consider discussing these works in Section 4.1.

- In lines 159–161, the LLM explores $m$ programs, but only one final answer is provided. Is the final answer selected by the VLMs?

- In line 169, there is mention of a quality threshold for functions, but the metric for quality is not provided.

- In Figure 4(c), why is the performance with new APIs lower?

- There is no analysis of the additional token costs for evolution. Is the cost worthwhile for the observed improvement in overall performance (3.4% in absolute terms, approximately 10% in relative terms)?

- In line 165, does the "final answer" refer to the ground truth answer?

- There is no analysis of the accuracy of the VLM judge.

- Does the ordering of the Omni3D-Bench data used for codebase construction affect the codebase? Would altering the data order drastically impact the performance of the resulting code?

References:

[1] Visual Programming for Zero-shot Open-Vocabulary 3D Visual Grounding, in CVPR, 2024.

[2] Language-to-Space Programming for Training-Free 3D Visual Grounding, in EMNLP, 2025

### Questions
See the weaknesses section. If I have any misunderstandings, I would greatly appreciate the authors' clarification.

### Soundness
4

### Presentation
3

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
This paper introduces Transductive Visual Programming (TVP), a novel framework that enables visual reasoning systems to evolve reusable tool libraries from problem-solving experience. TVP adopts a transductive approach: it first solves problems using basic vision tools, then abstracts recurring solution patterns into higher-level functions grounded in actual use. The architecture maintains a dual-library design: an Example Library storing verified program solutions and a Tool Library storing learned abstractions. Through iterative cycles of example accumulation, clustering, abstraction, validation, and merging, TVP progressively refines its tools and produces more efficient, accurate programs. On Omni3D-Bench, TVP achieves clear performance gains, surpassing GPT-4o and VADAR. The evolved tools exhibit strong zero-shot generalization to unseen spatial reasoning benchmarks, demonstrating robust transferability across domains.

### Strengths
- **Conceptual originality:** The paper introduces transductive tool evolution, which learns abstractions from experience rather than induction before use. This represents a genuine conceptual advance in visual programming and aligns well with human-like skill acquisition.  
- **Technical soundness:** The dual-library architecture and full algorithmic specification (program generation, clustering, abstraction, validation, and merging) are rigorous and clearly grounded. The validation mechanism ensures newly learned tools remain correct and reusable.  
- **Strong empirical results:** TVP achieves clear and consistent gains over prior visual programming systems and even large VLMs, particularly on complex 3D spatial reasoning and zero-shot transfer tasks.

### Weaknesses
- **Limited scope of evaluation:** While visual programming was originally designed for 2D visual reasoning and perception tasks, this paper evaluates TVP only on 3D spatial reasoning. It remains unclear whether the proposed transductive abstraction also benefits conventional 2D visual reasoning benchmarks (e.g., MME, MMMU).  
- **Heavy dependence on large proprietary models:** TVP’s components rely heavily on GPT-4o and its mini variants. It remains unclear how performance scales with smaller or open-source models, which may limit reproducibility and accessibility.  
- **Computational overhead:** The pipeline includes iterative clustering, abstraction, validation, and merging — likely computationally expensive. The paper reports no quantitative analysis of time or resource cost, which would be important for assessing practicality.

### Questions
1. Does TVP’s method also improve performance on conventional 2D visual reasoning tasks?  
2. How sensitive is TVP to the choice of backbone LLM? Would similar gains be observed when replacing GPT-4o with smaller or open-source models, and could the authors provide scaling trends or partial ablations in this direction?  
3. Could the authors provide a quantitative estimate of TVP’s computational and memory cost per iteration, and clarify whether any optimizations were applied to make the system practically deployable?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Transductive Visual Programming (TVP), a framework for visual reasoning that dynamically creates and refines its own library of tools. The core idea is to learn from problem-solving experience. TVP maintains a "dual-library" system:

- An Example Library that stores (question, program, solution) tuples for high-quality solutions it has found.

- A Tool Library that contains callable functions (tools).

When faced with a new query, TVP retrieves similar examples from the Example Library to use as in-context demonstrations for generating a solution program. Critically, TVP periodically analyzes its Example Library, clusters similar solutions, and "transductively abstracts" recurring programming patterns into new, higher-level tools, which are then added to the Tool Library. This allows the system to evolve from basic tools to more complex, specialized, and reusable functions. The paper shows that this approach achieves SOTA on the Omni3D-Bench for 3D spatial reasoning and that the learned tools generalize well to unseen spatial benchmarks.

### Strengths
- The core idea of "transductive abstraction" from a library of successful solutions is elegant and well-motivated. It ensures that created tools are practically useful and grounded in experience, which is a clear advantage over VADAR's more speculative, question-based induction (as clearly shown in Fig 2).

- The zero-shot generalization results on the SpatialScore-Hard collection (Table 2, Fig 5) are a key strength. Showing that tools learned only on Omni3D-Bench are effective on completely different datasets (3DSR-Bench, SpatialSense, VG-Bench) is a powerful demonstration that the system is learning robust and reusable reasoning patterns.

- The paper includes a strong set of analyses, such as the reduction in program cyclomatic complexity (Fig 4a), the performance boost from using new tools (Fig 4c), and the visualization of the library evolution over time (Fig 6).

### Weaknesses
- The TVP framework itself is extremely complex and computationally expensive. For each query, it makes multiple LLM/VLM calls (retrieve, generate m programs, execute m programs, judge m programs). It then has a heavy, periodic maintenance loop that involves more LLM calls for clustering, abstraction, validation, and merging. This "meta-cost" of running the TVP framework is not discussed but seems prohibitively high, likely many times more expensive than just running a baseline model.

- As mentioned, the field of tool-use and tool-creation is moving very fast. This paper proposes a new way to make tools. While this is a good contribution, it's an improvement on an existing line of work (VisProg -> ViperGPT -> VADAR -> TVP). It's not clear that this is a fundamentally new direction for the field, especially when compared to orthogonal approaches like differentiable soft-logic (e.g., NePTune).

- The entire framework (generation, judgment, abstraction, merging) is orchestrated by powerful, closed-source models (GPT-4o and 4o-mini). This makes the system dependent on SOTA models and raises questions about its robustness. Would the framework collapse if these "meta-LLMs" were replaced with less capable open-source models? The quality of the "judge" and "abstractor" seems critical.

### Questions
- Could the authors please comment on the computational cost of the TVP framework? Specifically, how many total LLM/VLM calls (and of what type, e.g., GPT-4o vs 4o-mini) are required, on average, to process a single query (including the amortized cost of library maintenance)? This "meta-cost" seems like a major factor in its practical utility.

- The paper's related work cites other tool-creation work (e.g., Skillweaver, ASI) which also learn from "trajectories" or "experience." Could you elaborate on the key differences between TVP's "transductive abstraction" and the skill-discovery methods used in these agentic works?

- How robust is the TVP framework to the choice of its "meta-LLMs"? The system relies heavily on GPT-4o for crucial steps like quality judgment and program generation. If a weaker, open-source model (e.g., Llama3-8B) were used for these orchestration tasks, would the system still be able to successfully identify, abstract, and validate high-quality tools?

### Soundness
3

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
This paper presents Transductive Visual Programming (TVP), an innovative framework that enables visual language models to evolve by learning reusable tools from their own problem-solving experience.

The method’s dual-library closed-loop design is conceptually clear, technically comprehensive, and contributes a structured approach to self-improving reasoning in LLM-based systems.

While the framework is well-presented and methodologically sound, key evaluation details (e.g., scoring criteria, abstraction prompts, complexity metrics, and computational cost) remain under-specified.

Empirically, the performance gains over ICL baselines are modest, and the claimed generalization largely reflects in-domain transfer rather than genuine out-of-distribution generalization.

Overall, TVP is a promising and well-presented system paper: its transductive framework and dual-library design are conceptually valuable and clearly executed. Despite under-specified evaluation details and modest gains, I lean weak accept, contingent on clarifications.

### Strengths
Innovative and Well-Structured Framework

+ The paper introduces Transductive Visual Programming (TVP), a novel and conceptually original framework that enables a model to iteratively learn reusable tools from its own problem-solving experience. Its dual-library closed-loop design (Example–Tool Library) is systematic and complete, effectively realizing a self-improving learning cycle.

Strong presentation quality
+ The paper is clearly written, logically organized, and well-illustrated with informative figures and detailed algorithms, making the methodology easy to follow.

### Weaknesses
Lack of Transparency in Evaluation Mechanisms

The evaluation procedures governing both the Example Library and the Tool Library are under-specified, which raises concerns about reproducibility and interpretability. Specifically:
+ Unclear criteria for Example Library admission.
Although the paper states that a VLM judge scores each generated program and admits examples whose quality exceeds a threshold of τq = 8.5, it does not define the concrete scoring dimensions—such as logical correctness, semantic relevance, visual consistency, or execution success. The basis for selecting τq (e.g., validation tuning versus heuristic choice) is also not explained. Moreover, no analysis is provided regarding how scores vary across task types or problem complexity, leaving the quality control process for examples largely opaque.
+ Opaque evaluation of tool abstraction potential.
The paper introduces an LLM-based cluster analyzer that outputs a textual “pattern” and a numeric “potential” score, using τpotential = 9.0 as a threshold for initiating tool abstraction. However, the work omits any description of how this score is computed, the intended scale, or the prompts used to elicit it. The rationale behind the chosen threshold is also absent. If this potential measure relies solely on LLM-as-judge scoring, it is likely susceptible to style bias and semantic drift, undermining objectivity.

Limited Empirical Gains and Undefined Complexity Metric

While the paper claims that TVP achieves improved performance and reduced program complexity through iterative transductive learning, the empirical evidence supporting these claims appears limited and partially confounded.
+ Marginal performance improvement over ICL baselines.
The main results show that the Example Library-only configuration (essentially an ICL baseline) already achieves 31.7% overall accuracy, while the full TVP framework after three iterations reaches 33.3%. This modest +1.6% gain raises doubts about whether the improvement truly stems from the proposed abstraction mechanism, or instead from the growing in-context example set providing stronger template guidance to the LLM. The observed benefits may therefore largely reflect in-context pattern imitation rather than genuine tool learning. Moreover, performance surpasses the ICL baseline only at iteration 3, when computational cost and LLM usage are substantially higher—yet the paper offers no analysis of cost-effectiveness or scaling trade-offs.
+ Undefined program complexity metric.
The paper reports that “program cyclomatic complexity decreases from 3.0 to 1.0,” using this as evidence that TVP learns simpler, higher-level abstractions. However, no formal definition or computation method for this complexity measure is provided. It is unclear whether this refers to classical McCabe complexity, the number of function calls, or another proxy metric. Without such clarification, the claimed reduction in complexity cannot be meaningfully interpreted or independently verified.

High Computational Cost and Unanalyzed Efficiency

+ TVP surpasses the ICL-only baseline only at iteration 3, implying that multiple costly iterations are required for modest gains (+1.6 pp). Yet the paper provides no analysis of runtime, token usage, or cost. Given that each iteration repeatedly invokes GPT-4o for program generation, judging, abstraction, and validation, the overall expense is likely high. Without quantitative efficiency reporting, it is unclear whether the observed improvement justifies the computational overhead or scales beyond small benchmarks.

### Questions
Evaluation standards for Example and Tool Libraries

+ Could the authors clarify the evaluation criteria for Example Library admission and Tool Library abstraction? Specifically, what dimensions does the VLM judge consider when scoring examples (e.g., logical correctness, semantic relevance, visual consistency)?
+ How was the threshold τq = 8.5 chosen—through tuning, validation, or heuristic selection? Similarly, for the Tool Library, how is abstraction potential measured, what prompts are used, and on what basis was τpotential = 9.0 determined?
+ If both evaluations rely solely on LLM-as-judge scoring, how do the authors control for potential bias or inconsistency across runs?

Empirical significance and complexity metric

+ The improvement over the ICL baseline is relatively small (+1.6 pp) and only appears after three iterations. Could the authors provide more evidence that the observed gains stem from true tool abstraction rather than ICL-style template learning?
+ Also, please clarify how program complexity is computed (e.g., McCabe complexity, function calls, or another proxy metric), and explain why its reduction should indicate better abstraction quality.

Computational efficiency

+ Since performance exceeds the ICL baseline only at iteration 3, what is the computational cost of running multiple iterations?
+ Please report runtime, token usage, or cost per iteration, and discuss whether the modest gain justifies the overall expense or scales to larger datasets.

### Soundness
2

### Presentation
2

### Contribution
2
