# JudgeFlow: Agentic Workflow Optimization via Block Judge

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Optimizing LLM-based agentic workflows is challenging for scaling AI capabilities, some current methods rely on coarse, end-to-end evaluation signals and lack fine-grained signals on where to refine, often resulting in inefficient or low-impact modifications. To address these limitations, we propose JudgeFlow, an Evaluation-Judge-Optimization-Update pipeline. We incorporate reusable and configurable logic blocks into agentic workflows, capturing fundamental forms of logic. On top of this abstraction, we design a dedicated Judge module that inspects execution traces, particularly failed runs, and assigns rank-based responsibility scores to problematic blocks. These fine-grained diagnostic signals are then leveraged by an LLM-based optimizer, which focuses modifications on the most problematic block in the workflow. Our approach improves sample efficiency, enhances interpretability through block-level diagnostics, and provides a scalable foundation for automating increasingly complex agentic workflows. We evaluate JudgeFlow on mathematical reasoning and code generation benchmarks, and the results demonstrate that JudgeFlow achieves superior performance and optimization efficiency compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces JudgeFlow, a framework for enhancing the optimization effect of agentic workflow by using llm-as-a-judge for attribution. The core contributions include (1) introducing reusable logic blocks as higher-level structural abstractions and attribution unit, (2) designing a Judge module that analyzes execution traces to assign rank-based responsibility scores to attribution unit, and (3) leveraging the attribution to guide targeted workflow optimization. With experiments on multiple benchmarks, JudgeFlow demonstrates improvements over existing methods including hand-crafted and autonomous multi-agent systems.

### Strengths
- **Sufficient Motivation**: The paper identifies an important gap in existing methods that rely only on end-to-end evaluation signals. The key insight is that failure attribution is critical for effective optimization. The Judge module provides block-level diagnostic signals by analyzing execution traces, enabling enables targeted optimization instead of blind exploration.
- **Comprehensive Experiments**: The paper evaluates on multiple benchmarks, including mathematical reasoning and code generation. Results show consistent improvements across all tasks. The ablation studies validate the importance of both logic blocks and the judge module.
- **Clear Presentation**: The paper is well-written with clear motivation and good use of figures. The case study effectively illustrates how the approach works in practice.

### Weaknesses
- **Limited Novelty in Attribution Strategy**: The Judge module is a straightforward application of llm-as-a-judge, which lacks technical depth and novelty. The rank-based scoring mechanism is simple and may not capture complex failure patterns. More sophisticated attribution methods could potentially provide better optimization signals.
- **Insufficient Analysis of Judge Module**: The paper acknowledges that llm-as-a-judge may be biased but provides no empirical validation. There is no ground-truth comparison to verify whether the judge correctly identifies problematic blocks. This is critical for trusting the optimization process.
- **Missing Cost Analysis**: The paper provides no analysis of optimization or execution cost. The Judge module requires additional LLM calls for every failed instance during evaluation. Combined with the optimization loop, the total number of LLM calls and token cost could be substantial. Without cost comparison against baselines, the practical value of the modest performance gains (1.4\% average) is unclear.

### Questions
- Could similar judge-based approaches work at the operator level? Is there empirical evidence that blocks are essential for effective judging?
- The paper fixes block abstraction at a specific level (between operators and full workflows). Is there an optimal granularity for attribution?

### Soundness
2

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
5

### Summary
This paper addresses a critical limitation in agentic workflow optimization: the lack of fine-grained diagnostic signals to guide the optimization process. By introducing a Judge module that performs block-level error attribution, JudgeFlow can identify problematic components within workflows and focus modifications accordingly. The experimental results demonstrate superior performance and sample efficiency compared to existing methods like AFlow. Overall, the paper is well-written with clear presentation and makes a solid contribution to automated workflow optimization.

### Strengths
S1. Identifies a key problem: The paper correctly identifies that prior workflow generation methods rely heavily on heuristic optimization with LLMs alone, lacking concrete optimization signals to guide the search process effectively.

S2. Improved optimization efficiency: The Judge-guided error attribution mechanism significantly enhances both efficiency and effectiveness of workflow optimization. As shown in Figure 4b, JudgeFlow achieves better performance than baseline AFlow with fewer optimization iterations.

S3. Novel evaluation-judge paradigm: The separation of evaluation and judgment provides actionable diagnostic information, enabling targeted modifications rather than blind search across the entire workflow space.

S4. Clear presentation: The paper is well-written with good organization. The methodology is clearly illustrated (e.g., Figure 3 effectively visualizes the evaluation-judge-optimization-update pipeline), making the contributions easy to follow.

### Weaknesses
S1. Computational efficiency: The optimization process requires evaluating all samples in the dataset at each iteration, which can be computationally expensive. A potential improvement would be to gradually reduce the validation set size during optimization to lower costs.

S2. Incremental contribution: While adding the Judge module is useful, the overall novelty appears limited. The paper primarily introduces optimization signals but lacks deeper insights or substantial architectural innovations beyond existing frameworks like AFlow and ADAS. Nevertheless, it remains a solid piece of work.

S3. Outdated benchmarks: The evaluation datasets (GSM8K, MATH, MBPP, HumanEval) are somewhat dated. Including more recent and challenging benchmarks such as LiveCodeBench or AIME would strengthen the paper's contribution and better demonstrate the method's capabilities on harder problems.

### Questions
Please refer to the weak points.

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
3

### Summary
This paper proposes **JUDGEFLOW**, an *Evaluation–Judge–Optimization–Update* pipeline for automated agentic workflow optimization. The system abstracts workflows into logic blocks (sequence/loop/condition), runs them on tasks, and applies an LLM-based Judge to assign responsibility scores to blocks in failed executions. The optimizer then modifies the weakest block via add/modify/delete operations. Experiments on GSM8K, MATH, MBPP, and HumanEval show consistent improvements over strong baselines, including AFlow and MermaidFlow, with ablations demonstrating the value of the judge and block abstraction.

### Strengths
1. Introduces block-level credit assignment for workflow refinement, moving beyond global end-to-end signals.

2. Experimental setup is clear and robust: multiple reasoning and coding benchmarks, consistent evaluation criteria, and ablations on logic blocks and judge components.

3. Well-written and easy to follow; modular structure and pseudocode clarify pipeline design. Case studies make the pipeline behavior interpretable.

4. Offers a more granular optimization signal for agentic workflows, improving interpretability and sample efficiency.

### Weaknesses
1. **Incremental novelty over AFlow**

The pipeline structure closely follows AFlow, raising concerns about incremental contribution. The primary change is shifting judgment from operator-level to block-level, while reusing the same operator abstraction and workflow-editing paradigm. The paper does not explain why this granularity shift should fundamentally improve workflow optimization, and no theoretical justification is provided.

Furthermore, the optimization remains benchmark-specific, without evidence of query-level generalization or transferable workflow patterns across tasks. As a result, the gains appear tied to engineering-level refinements rather than a general workflow-learning mechanism, limiting the conceptual novelty and broader research contribution.


2. **Limited cost–benefit analysis**

The paper does not provide a clear cost analysis. The method uses a similar amount of training data as prior work without reducing sample requirements, making it unclear whether the approach offers a meaningful efficiency improvement.

### Questions
1. How does the method scale with larger workflows (M≫3)? Does block-level scoring remain reliable as depth increases?

2. Is the system robust to noisy judge outputs? Any safeguards (e.g., majority voting, statistical filtering)?

3. Can the method reduce training cost compared to AFlow while maintaining performance? That would strengthen claims of efficiency.

4. Are there empirical results showing cross-benchmark generalization?

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
2

### Summary
Previous work in agentic workflow optimization agents rely on a three stage system (Evaluation, Optimization, Update) to optimize an agentic workflow. One key issue with this approach is that when multiple updates are needed, its hard for the optimization layer to prioritize the most problematic block. Consecutively,  the authors hypothesize that introducing an additional self-reflection layer, namely a Judge, after the evaluation stage but before the optimization layer can inform the decision of the LLM optimizer. Experiments on four mathematical reasoning and code generation domains show superior performance.

### Strengths
The paper is very well written and I commend the authors for putting in the extra work for including psuedocode and useful figures.

### Weaknesses
**Baselines:** I'm not convinced that the current results demonstrate the effectiveness of the method properly:
 - HumanEval and MBPP are widely regarded as saturated in code generation, partly because most frontier models have been trained on these datasets. Please show additional results on LiveCodeBench, AIME2025, SWE-Bench-Live, etc. which have more complex tasks.
 - The set of baselines do not seem reflective of the current state of the art either. How does this compare with OpenHands, SWE-Agent or other well-established agents' performance on the code generation benchmarks listed above?
 - For example, `o1-mini` archives `96.2%` pass@1 on HumanEval (which isn't mentioned in this paper). The datasets mentioned in this paper might be of interest as well as our understanding of how to evaluate methods on code geenration tasks has evolved [https://arxiv.org/pdf/2412.21199](https://arxiv.org/pdf/2412.21199).

**Overall:** I'm in favor of rejecting this work because I'm not sure if the current set of baselines and tasks is reflective of the best methods for code generation in the community. I urge the authors to find more competitive baselines as well as more competitive datasets to evaluate JudgeFlow.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
4

### Contribution
2
