# Ensuring Functional Correctness of Large Code Models with Selective Generation

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
The hallucination of code generation models hinders their applicability to systems requiring higher safety standards. One critical bottleneck in addressing code hallucination is the difficulty of identifying the functional correctness of generated code, due to its unnatural form. We address this core bottleneck by automatically generating unit tests using dynamic code analysis tools, leveraging the \emph{executable nature} of code. Accordingly, we propose \emph{selective code generator} that abstains from uncertain generations -- based on the functional correctness evaluated by generated unit tests -- to {\color{red}theoretically control the correctness among non-abstained answers, \ie the false discovery rate}. Finally, we propose to use generated unit tests in evaluation as well as in learning for precise code evaluation, calling this paradigm \emph{FuzzEval}. We demonstrate the efficacy of our method along with the controllability of code hallucination and reasonable selection efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Tackling hallucinations of code generation models with automatically generated unit tests using dynamic code analysis tools. They propose SCG to abstain from uncertain generations. Defines a probabilistic notion of correctness called α-code entailment, leveraging dynamic analysis tools to approximate functional equivalence. Uses fuzzing to generate large sets of unit tests.

### Strengths
Reduces functional hallucinations and increases confidence in correctness among non-abstained outputs.
Automated test generation with FuzzEval using a model-agnostic approach, making evaluation robust.

### Weaknesses
Assumes i.i.d. conditions, which may not hold under distribution shifts. In a real-time scenario, such a uniform distribution would not be found.

Efficiency tradeoff: abstention reduces hallucination but lowers coverage with less output generated now.

Relies heavily on underlying model quality.

Fuzzing limitation may not cover all edge cases or complex logic paths. And increases computational overhead for evaluation.

### Questions
- How does the proposed fuzzing approach ensure coverage of complex logic paths and rare edge cases? Can the idea of integrating symbolic execution or static analysis improve completeness here?
- Have you considered adaptive thresholds or confidence calibration techniques to optimize efficiency?
- How does the method handle distribution shifts (for unseen libraries and new coding styles)?
- Could the approach be extended to multi-turn code generation or interactive coding scenarios?
- Beyond FDR, other metrics (coverage, complexity, runtime performance) could provide a holistic view of correctness
- How can the computation overhead of fuzzing at scale be optimized?

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
This paper proposes a selective code generator (SCG) and the FuzzEval paradigm to automatically generate unit tests for controlling code hallucination rates. This work effectively addresses the bottleneck of insufficient manual test cases in traditional code correctness assessment. The authors validate the SCG method's effectiveness across multiple datasets, models, and baseline methods, presenting a valuable contribution to evaluating code generation.

### Strengths
1. The paper is well-written and analyzes the interesting and important problem of code hallucination in LLMs.
2. It proposes a novel approach (SCG and FuzzEval) for automated unit test generation to evaluate code correctness.
3. The empirical validation is thorough, covering multiple datasets, models, and baselines.

### Weaknesses
**Major:**

1. The framework's core reliance on a ground-truth "canonical solution" for $\alpha$-equivalence is a significant limitation, as such reference solutions are unavailable in most real-world code generation scenarios.
2. The accuracy of the proposed FDR-CE metric is highly dependent on the quality of the generated unit tests, but the impact of this quality (e.g., test coverage) is not quantified.
3. The evaluation benchmarks consist mainly of simple, stateless algorithmic problems, making it unclear how the approach scales to more complex, stateful, or repository-level programs.
4. The admission in Section 4.1 that solutions were "manually post-processed" for the fuzzing tool raises significant concerns about the framework's practical scalability and the required manual effort.

**Minor:**

1. The $\alpha$-equivalence check focuses only on functional correctness and fails to capture significant non-functional differences, such as algorithmic performance (e.g., $O(n)$ vs. $O(n^2)$).
2. The FDR-CE metric aggregates all functional incorrectness into a single number, which limits its diagnostic power in understanding the specific types or patterns of hallucinations.

### Questions
1. The computational overhead of the calibration process and the FuzzEval dynamic analysis seems potentially high. Is there an analysis of this time cost?
2. Why is the performance data for the SCG-SMALL variant missing on the MBPP-F, HUMANEVAL-F, and MERCURY-F datasets in Table 1?
3. What was the final value of $k$ chosen for SCG-manual in Table 1, and what is the explanation for the $0.000 \pm 0.000$ result?

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
# Summary

This paper addresses code hallucination in large language models by proposing a selective generation framework that provides theoretical guarantees on functional correctness. The core innovation is the introduction of **α-code entailment**, which defines that code `y` α-entails `ŷ` when $P_y\{\hat{y}(u) = v\} \geq 1 - \alpha$, where input-output pairs `(u,v)` are automatically generated using dynamic code analysis tools (fuzzing) rather than manual verification. Building on this definition, the authors develop a selective code generator $\hat{S}(x)$ that either returns generated code `G(x)` or abstains (returns "I don't know") to control the False Discovery Rate with Code Entailment (FDR-CE): $R_\alpha(\hat{S}) := P\{\hat{S}(x) \notin E_\alpha(y) \mid \hat{S}(x) \neq \text{IDK}\}$ with probability guarantee $P\{R_\alpha(\hat{S}) \leq \hat{U}\} \geq 1 - \delta_S$. The learning algorithm (ASCG) uses binomial tail bounds to estimate code entailment and learns a threshold to maximize selection efficiency while maintaining FDR-CE control. The paper also proposes the FuzzEval paradigm, using automatically generated unit tests for both learning and evaluation. Experiments across multiple datasets, models (GPT-4o, Gemini-1.5 Pro, DeepSeek-R1, CodeLlama-13B), and programming languages demonstrate successful FDR-CE control at desired levels (e.g., εₛ = 0.3) with reasonable selection efficiency (30-50%), outperforming baseline methods including exact match and heuristic approaches, though performance depends on base model quality and assumes i.i.d. data.

### Strengths
This paper makes theoretical and methodological contributions to addressing code hallucination in large language models. The introduction of α-code entailment represents a novel and necessary formalization for measuring functional correctness between code snippets, addressing the challenge that code's unnatural structure makes it difficult for humans to verify functional equivalence at scale. Additionally, fuzzing tools, traditionally used for bug detection, are repurposed to automatically generate unit tests for learning and evaluation. The FuzzEval paradigm addresses a critical bottleneck in existing benchmarks (such as HumanEval) that rely on limited manually created test cases.

The experiments demonstrate broad applicability across multiple dimensions. The evaluation covers 4 datasets, 5 models, and 4 programming languages, with ablation studies examining various parameters (α, εₛ, εₑ, δₛ) and scoring functions. The empirical results validate that the theoretical bounds hold in practice, with box plot whiskers consistently staying below the desired FDR-CE threshold.

### Weaknesses
1. The computational cost of fuzzing remains unanalyzed. While nₘₐₓ = 150 is specified, the paper provides no wall-clock time measurements, overhead comparisons relative to code generation time, or scalability analysis for larger codebases. The low selection efficiency for weaker models (CodeLlama achieves only ~7% efficiency in Figure 4a) suggests the method may be impractical below certain model quality thresholds.

2. The evaluation scope is limited to simple algorithmic problems in standalone functions. All datasets (APPS, HumanEval, MBPP, Mercury) avoid the complexities of real-world software including multi-file projects, API calls, I/O operations, and stateful systems. Whether the approach extends to such scenarios remains unclear.

3. The method critically relies on fuzzing adequately exploring execution paths, but complex code with exponentially many paths may have corner cases that fuzzing misses due to time limits or hard-to-reach branches. The paper provides no theoretical or empirical analysis of how incomplete fuzzing coverage affects the quality of FDR-CE guarantees.

4. The strong i.i.d. assumption is acknowledged but never tested empirically. Code generation often encounters distribution shift (new APIs, evolving languages), yet no out-of-distribution robustness evaluation is provided. The α parameter fundamentally determines what "correct" means, but lacks principled selection guidance beyond vague suggestions to "choose some small value." Despite α dramatically affecting efficiency (Figure 3b), practitioners receive no data-driven guidelines.

5. The evaluation misses important baselines (other uncertainty quantification methods, self-consistency approaches) and lacks human evaluation to validate whether abstentions occur on genuinely harder problems or whether accepted codes are perceived as more correct by developers.

6. The claimed asymptotic equivalence between PASS@1 and FDR-CE (Appendix K) requires α→0 and nᵧ→∞, but experiments use finite values without providing sufficient empirical support for this relationship.

### Questions
See the section above.

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
2

### Summary
This paper addresses code hallucination in large code models (LCMs)—a barrier to their use in safety-critical systems—by tackling the core challenge of verifying code functional correctness. It proposes three key elements: α-code entailment (a statistical definition of code correctness using fuzzing-generated unit tests to measure if a canonical solution y statistically validates a generated code y, a Selective Code Generator (SCG) that abstains from uncertain outputs to control the False Discovery Rate with Code Entailment (FDR-CE) while maximizing selection efficiency, and FuzzEval (a scalable evaluation paradigm using fuzz-generated unit tests instead of manual ones). Validated across 4 datasets, 5 LCMs, and multiple programming languages, SCG achieves controllable FDR-CE (e.g., ≤0.3 for S=0.3), with strong selection efficiency (e.g., 0.995 for DeepSeek-R1), outperforming baselines, while the authors release code and datasets for reproducibility.

### Strengths
1. Its originality lies in defining α-code entailment (statistical code functional correctness via fuzzing) and extending selective prediction to code with SCG, bridging gaps between natural language entailment and code’s structural complexity.
2. It demonstrates high quality through rigorous theoretical proofs (e.g., FDR-CE controllability guarantees) and well-controlled experiments (5 LCMs, 4 datasets, 50 random splits) that report statistical significance.
3. It maintains clarity by explaining complex concepts (e.g., binomial tail bounds) with intuitive examples (α-entailment code samples) and visualizations (SCG workflow), while appendices add depth without cluttering the main text.
4. Its significance stems from FuzzEval solving code evaluation scalability (replacing manual unit tests) and SCG’s FDR-CE guarantees making LCMs trustworthy for safety-critical applications.

### Weaknesses
Admittedly, I'm not an expert in this field, but I figure out these weaknesses with low confidence.

1. It relies on an i.i.d. assumption for FDR-CE guarantees but does not explore mitigations for distribution shift (e.g., unseen code), limiting real-world applicability.
2. Low-quality models (e.g., CodeLlama 13B) fail to meet desired FDR-CE due to uncalibrated scoring functions, yet the paper does not propose strategies to improve SCG for such models.
3. It only briefly compares FuzzEval with LLM-augmented datasets (e.g., MBPP+) and lacks direct comparison with LLM-based unit test generators like EvalPlus or SemCoder.
4. It does not explain why the verbalized scoring function ((f_{verb}) fails to bound FDR-CE (e.g., LCM confidence verbalization issues or poor prompt design), leaving a gap in analysis.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
