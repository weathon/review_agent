# FeatureBench: Benchmarking Agentic Coding for Complex Feature Development

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Agents powered by large language models (LLMs) are increasingly adopted in the software industry, contributing code as collaborators or even autonomous developers. As their presence grows, it becomes important to assess the current boundaries of their coding abilities. Existing agentic coding benchmarks, however, cover a limited task scope, e.g., bug fixing within a single pull request (PR), and often rely on non-executable evaluations or lack an automated approach for continually updating the evaluation coverage.  To address such issues, we propose FeatureBench, a benchmark designed to evaluate agentic coding performance in end-to-end, feature-oriented software development. FeatureBench incorporates an execution-based evaluation protocol and a scalable test-driven method that automatically derives tasks from code repositories with minimal human effort. By tracing from unit tests along a dependency graph, our approach can identify feature-level coding tasks spanning multiple commits and PRs scattered across the development timeline, while ensuring the proper functioning of other features after the separation.  Using this framework, we curated 200 challenging evaluation tasks and 3825 executable environments from 24 open-source repositories in the first version of our benchmark. Empirical evaluation reveals that the state-of-the-art agentic model, such as Claude 4.5 Opus, which achieves a 74.4% resolved rate on SWE-bench, succeeds on only 11.0% of tasks, opening new opportunities for advancing agentic coding. Moreover, benefiting from our automated task collection toolkit, FeatureBench can be easily scaled and updated over time to mitigate data leakage. The inherent verifiability of constructed environments also makes our method potentially valuable for agent training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ACE-Bench, a new benchmark for evaluating LLM-based agents on feature-oriented software development tasks. Unlike existing benchmarks focusing primarily on bug-fixing within single PRs, ACE-Bench proposes more complete coding scenarios. The benchmark comprises 212 tasks and 889 executable environments from 16 open-source Python repositories.

Key contributions include:

(1) A feature-oriented evaluation framework with two difficulty levels (L1: extending existing codebases; L2: implementing from scratch)

(2) Execution-based evaluation with explicit interface specifications to enable unambiguous testing

(3) Empirical results show that more capable models achieve very low success rate, like Claude 4 Sonnet achieves only 7.5% success rate

### Strengths
**1. Problem Formulation**: This work proposes ACE-Bench that contains complex software engineering tasks. Its problem formulation explains its differences from existing benchmarks in task complexity and dataset construction.

**2. Rigorous Evaluation Protocol**: The execution-based evaluation with comprehensive anti-cheating mechanisms and two difficulty levels provides a reliable performance assessment.

**3. Significant Challenge**: The obvious performance drop between SWE-bench and ACE-Bench demonstrates the limitations of current agentic systems, providing meaningful direction for future research.

### Weaknesses
**1. Limitations In Scale and Diversity**:

- ACE-Bench uses 212 evaluation data from 16 repositories, which can be relatively small.
- Python-only instances limits generalizability to other programming languages and real-world scenarios.
- Repository selection criteria are not clearly justified.

**2. Limitations In Methodology**:

- Using LLMs to classify top-level objects introduces potential systematic biases and errors. However, no quantitative evaluation of classification accuracy is provided.
- The 100-line minimum and "10 F2P test points" filtering criteria lack justification.
- The paper sets m=5 P2P tests per instance, but doesn't justify why this number provides adequate coverage.

**3. Limitations In Evaluation**:

- The average of 1M+ input tokens per task raises concerns about practical applicability and cost.
- Near-zero success rates on L2 tasks suggest the difficulty may be unrealistically high, limiting the benchmark's ability to differentiate between models and provide meaningful insights for future research.
- Figure 5 shows minimal correlation between task creation time and performance, but deeper analysis of how feature complexity evolves over time can be valuable.

**4. Limitations In Interface Specification Dependency**:

- Table 7 shows remarkable performance drops without interfaces, suggesting that the benchmark may be testing interface-matching more than general coding ability.
- Real-world development often involves ambiguous or evolving requirements, while the explicit interface specification may not reflect realistic scenarios.

### Questions
My questions are following several aspects mentioned in weakness:

- How can you ensure the generalizability of ACE-Bench, given various programming languages and real-world scenarios?
- What are your repository selection criteria, and why do you believe 16 repositories are sufficient?
- What are the justifications and verifications for LLM classification, filtering criteria, and the number setting to ensure adequate coverage?
- How would you address the limitations in evaluation, such as the high input tokens and cost, undistinguishable performance, and inadequate analysis?
- How are you going to address these interface specification dependency problems in ACE-Bench?

### Soundness
3

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
3

### Summary
This paper proposes ACE Bench, a novel benchmark for evaluating coding agents. While existing benchmarks focus on evaluating PRs or bug fixs, ACE-Bench emphasizes assessing feature-level implementation capabilities of coding agents.

### Strengths
- Evaluating feature-level implementation is both novel and important. As evidenced by recent SWE-bench leaderboard results, modern coding agents can perform bug-fixing tasks with high accuracy. However, their capability to handle feature-level implementations remains largely unexplored. This paper addresses this gap by providing a benchmark specifically designed to evaluate this capability.

- The benchmark is designed with usability in mind. Given that evaluating the full set requires approximately one million tokens (with associated computational costs), the authors provide a "lite set" that reduces evaluation costs. Additionally, the "Passed Rate" metric (the average fraction of fail-to-pass tests passed per task) enables partial assessment of feature-level implementation capability.

### Weaknesses
- Allowing unrestricted library usage may enable agents to complete tasks by simply calling existing library functions, essentially testing library knowledge rather than implementation capability (The benchmark allows agents to use pip install to add arbitrary libraries (Figure 13)). While the authors prevent accessing ground-truth implementations via anti-cheating mechanisms,the policy on legitimate library usage remains unclear. The authors should clarify whether the evaluation assesses (a) the ability to select and leverage appropriate libraries as part of software development skills, or (b) pure implementation capability with a fixed set of libraries. This distinction is critical for interpreting what capabilities are actually being measured.

- The benchmark tasks are derived from commits created between May 2022 and September 2025, which overlaps substantially with the training periods of models (i.e. knowledge cutoff). Fig. 5 shows that task performance currently exhibits minimal dependence on commit time. However, as the authors acknowledge, the risk of data leakage may become more pronounced in the future. Therefore, continuous updates to the benchmark will be critical to maintain its validity as an evaluation tool.

- The benchmark only supports Python, limiting its generalizability to other programming languages.

### Questions
- I found it somewhat confusing that there exists another agent evaluation benchmark with the same name ACE Bench [Chen+ 2025]

[Chen+ 2025] "ACEBench: Who Wins the Match Point in Tool Usage?"

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
5

### Summary
The authors propose ACE-Bench a python-only, execution-based coding benchmark to evaluate coding agents' performance on feature development. Similar to related benchmarks, they (manually) create execution environments and extract fail2pass and pass2pass tests to evaluate whether a solution would solve the problem at hand without breaking other functionality. Problem statements are synthesized using a LLM and include invocation path, function signature (including input and output variables), as well as annotations. The authors developed an algorithm to extract functions that are relevant to a test patch from an object dependency graph. In the evaluation of the OpenHands agent with four LLMs shows that compared with SWE-Bench the resolve rates are much lower.

### Strengths
* The authors don't base their dataset on already existing ones but scrape their own data which lessens the risk of data leakage
* The paper is well written and easy to follow. Visualization illustrate the core aspects of the work well. 
* Assessing feature development capabilities is an important area which is under-explored
* The dataset is seems to be significantly more complex in terms of gold solution lines, files, functions and number of tests.
* The graph-based function extraction is novel and seems sound

### Weaknesses
* The authors do not provide a lot of analysis to show that their tasks are truly solvable. Given that the problem statements are LLM generated, this needs to be shown. The authors propose that AssertionErrors indicate problem statements contain sufficient information. However, runnable code does not correlate with solvability of the tasks.
* The data set is Python only which severely limits to which degree one can measure coding agent performance.
* Only a single agent (OpenHands) is evaluated hence the claim of "laziness" is specific to their scaffold.
* All "feature development" tasks are more feature extensions rather than new features. While this is due to the fact that truly new features are extremely hard to test, this is a major limitation for a benchmark that focuses on feature development assessment.
* The quality of the generated prompts is hard to quantify, yet the authors claim to have developed a "high-quality" data set. 
* Providing invocation path, function signature (including input and output variables), as well as annotations in the prompt seems unreasonable and not typical of a feature development task. More typical are natural language description that are rather vague. 
* You consider SWE-Bench a benchmark that doesn't contain any feature requests. This is not true. As quantified by Rashid et al.[1] (containing 22% feature requests), it actually contains 18% feature requests. 

1. Rashid, M. S., Bock, C., Zhuang, Y., Buchholz, A., Esler, T., Valentin, S., ... & Callot, L. (2025). SWE-PolyBench: A multi-language benchmark for repository level evaluation of coding agents. arXiv preprint arXiv:2504.08703.

### Questions
* You report 889 executable environments but only 212 tasks, can you explain the descrepancy?
* You configure the nnumber P2P tests. Doesn't this mean a there may be tests that would fail even though it would pass both P2P and F2P?
* Is the subset of 30 instances purely random or stratified in some way?
* In section 4.2.1 you say you conducted a "professional-level algorithm engineer" who revised prompts. Can you detail what this title means and how they were revised?
* I don't quite understand how you arrived at the $L_1$ vs. $L_2$ datasets. Can you elaborate?

### Soundness
2

### Presentation
4

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
This paper introduces ACE-Bench, an execution-based and continually updatable benchmark for feature-oriented agentic coding, built via a test-driven, dependency-trace pipeline that yields 212 tasks across 16 repositories and shows frontier agents solve only ~7.5%.

### Strengths
- The benchmark targets feature-level development (not just bug fixes) and pairs each task in two modes—L1 (extend an existing repo) and L2 (from scratch)—a clean formulation that isolates the role of context and raises the ceiling on task complexity.

- The evaluation is execution-based, with explicit interfaces and anti-cheating controls; the pipeline includes post-verification and ablations (e.g., hiding interfaces, step budgets, visible tests), plus clear metrics (Resolved/Passed/Token I/O). These choices make the results reproducible and the failure analysis informative. 

- The paper is well structured (pipeline figures, instance layout, and evaluation workflow are easy to follow), and it surfaces useful empirical trends (e.g., performance drops with longer required code; L2 is markedly harder than L1).

### Weaknesses
- Positioning vs. closely related work needs to be sharper. The paper should more directly compare and differentiate from SWE-Dev (feature-driven development on large existing codebases with runnable environments; 14k train / 500 test and developer-authored unit tests) and commit0 (from-scratch library generation with API spec + interactive tests).

- Dataset composition skew. Although spanning 16 repos, the task mass is concentrated (e.g., Transformers dominates), which risks domain bias and may understate generalization across diverse stacks (services, infra, build systems).

- Baseline coverage & fairness details. All agents are run inside OpenHands, which is reasonable, but diversity in agent frameworks (gemini-cli, kimi-cli) would triangulate where the difficulty lies.

```
SWE-Dev: https://arxiv.org/abs/2505.16975
commit0: https://arxiv.org/abs/2412.01769
```

### Questions
Precise distinction from SWE-Dev and commit0.

### Soundness
3

### Presentation
3

### Contribution
2
