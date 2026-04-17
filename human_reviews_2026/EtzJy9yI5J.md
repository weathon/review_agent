# DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle

- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
Real-world enterprise data intelligence workflows encompass data engineering that turns raw sources into analytical-ready tables and data analysis that convert those tables into decision-oriented insights. 
We introduce DAComp, a benchmark of 210 tasks that mirrors these complex workflows.
Data engineering (DE) tasks require repository-level engineering on industrial schemas, including designing and building multi-stage SQL pipelines from scratch and evolving existing systems under evolving requirements. 
Data analysis (DA) tasks pose open-ended business problems that demand strategic planning, exploratory analysis through iterative coding, interpretation of intermediate results, and the synthesis of actionable recommendations.
Engineering tasks are scored through execution-based, multi-metric evaluation.
Open-ended tasks are assessed by a reliable, experimentally validated LLM-judge, which is guided by hierarchical, meticulously crafted rubrics.
Our experiments reveal that even state-of-the-art agents falter on DAComp. Performance on DE tasks is particularly low, with success rates under 20\%, exposing a critical bottleneck in holistic pipeline orchestration, not merely code generation. Scores on DA tasks also average below 40\%, highlighting profound deficiencies in open-ended reasoning and demonstrating that engineering and analysis are distinct capabilities.
By clearly diagnosing these limitations, DAComp provides a rigorous and realistic testbed to drive the development of truly capable autonomous data agents for enterprise settings.
Our data and code are available at \url{da-comp.github.io}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces DAComp, a comprehensive benchmark designed to evaluate LLM-based data agents across the entire enterprise data intelligence lifecycle, encompassing both data engineering (DE) and data analysis (DA) tasks.

DAComp includes 236 tasks divided into repository-level engineering (architecture, implementation, and evolution) and open-ended analysis tasks that require reasoning, planning, and synthesis of insights. The DE tasks are evaluated through execution-based metrics (Component Score, Cascading Failure Score, Success Rate), while DA tasks are judged using a validated LLM-as-judge framework with hierarchical rubrics and GSB scoring.

The paper presents extensive experiments across leading models (GPT-5, Claude-4, Gemini-2.5, DeepSeek-V3.1, Qwen3, etc.), showing that even state-of-the-art agents perform poorly (<50 % overall, <20 % DE success rate). The authors conclude that current LLM agents lack holistic orchestration and open-ended analytical reasoning.

Contributions:

First benchmark to jointly evaluate repository-level engineering and open-ended analytical reasoning.

Novel hierarchical rubric framework for open-ended evaluation with verified human–LLM agreement.

Large-scale empirical study exposing the limits of current LLM data agents.

### Strengths
Originality: Defines a new, holistic problem space combining engineering and analysis in realistic data workflows.

Methodological rigor: Combines automatic and rubric-based evaluations, validated with human judgments.

Scale and realism: Repository-level tasks (>4k LOC) and open-ended business questions reflect real enterprise workloads.

Comprehensive evaluation: Benchmarks diverse models, offering insights into distinct skill domains (engineering vs reasoning).

Actionable findings: Identifies orchestration and multi-step reasoning as key bottlenecks for next-generation data agents.

Excellent presentation: Clear structure, detailed analysis, strong alignment with related work.

### Weaknesses
Limited exploration of agent learning improvements: The paper benchmarks performance but does not propose or test training strategies to overcome observed limitations.

Restricted open-source availability: While the paper claims data/code release, the double-blind setup prevents verification; explicit examples of released task formats would strengthen reproducibility.

Evaluation cost: The LLM-as-judge pipeline, though validated, may pose high computational costs for community replication; approximate or lightweight scoring alternatives would be valuable.

Slight imbalance between DE and DA sections: DE is analyzed more deeply than DA in terms of ablations and case studies.

### Questions
1- How does the cost of LLM-judging scale with benchmark size, and can smaller-scale variants be used for quick evaluations?
2- Could the authors clarify whether the hierarchical rubrics generalize to unseen tasks, or are they manually constructed per task?
3- How is task realism maintained when synthetic data are used—were human engineers asked to validate their authenticity?

### Soundness
4

### Presentation
4

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
The paper introduces DAComp, a benchmark suite intended to evaluate autonomous data‑agent systems across the full data‑intelligence lifecycle. DAComp comprises 236 tasks split into two families: (i) DE (repository‑level data‑engineering) tasks that require agents to design, implement, or evolve multi‑stage SQL pipelines on large, enterprise‑style schemas, and (ii) DA (open‑ended data‑analysis) tasks that ask agents to produce reports, insights, and recommendations. The authors propose a mixed evaluation protocol: deterministic DE tasks are scored with three execution‑based metrics (Component Score, Cascading‑Failure Score, Success Rate), while DA tasks are judged by an LLM‑based evaluator using hierarchical rubrics and a Good‑Same‑Bad (GSB) component. A large experimental study compares a range of proprietary and open‑source LLM agents (GPT‑5, Claude‑4‑Sonnet, Gemini, DeepSeek, Qwen3, etc.) under two agent frameworks (a ReAct‑style baseline and OpenHands). The authors also validate the LLM‑judge against human annotators and report benchmark stability over multiple runs.

### Strengths
1. **Holistic scope** – By covering both repository‑level engineering and open‑ended analysis, DAComp fills a clear gap in existing agent benchmarks that usually focus on isolated code generation or single‑turn QA.  
2. **Realistic task design** – Tasks are built from 73 permissively‑licensed enterprise SaaS schemas, with synthetic data that respects realistic column distributions, referential integrity, and edge‑case noise.  The DE‑Impl/Evol tasks involve multi‑file, multi‑layer pipelines (≈ 4600 LOC), which is substantially larger than prior data‑science benchmarks.  
3. **Rigorous evaluation methodology** – The three‑tiered execution metrics for DE provide fine‑grained diagnostics (component vs. pipeline level).  The hierarchical rubrics for DA are carefully constructed, with multiple valid solution paths and a validation study showing high LLM‑judge/human agreement (Pearson ≈ 0.80, κ ≈ 0.65).  
4. **Comprehensive analysis** – The authors present detailed error‑type breakdowns (planning, execution, interpretation) and relate performance to concrete properties such as node count, line count, and turn‑count distributions.  
5. **Potential impact** – By exposing the engineering‑orchestration bottleneck, the benchmark can steer future research toward better planning, tool‑use, and pipeline consistency, which are crucial for socially relevant domains (healthcare analytics, climate data pipelines, sustainability reporting).

### Weaknesses
- While the authors report strong correlations with human ratings, the statistical choices and reporting could be more rigorous. Using Pearson r on rubric sums (often ordinal/heterogeneous across tasks) is suboptimal; intraclass correlation (ICC) or Kendall’s tau for ranking, plus confidence intervals, would be preferable.
- Weighted κ=65 for GSB indicates only moderate agreement; the paper calls overall alignment “high” but should temper claims or provide confidence intervals and calibration plots.
- The judge model (Gemini-2.5-Flash) belongs to the same family as one of the evaluated models (Gemini-2.5-Pro). Although Flash≠Pro, the potential for family-specific biases remains. The paper presents alternative judges, but a full ranking-stability analysis across judges is not reported.
- GSB depends on five baseline reports created from multiple LLM outputs; selection of baselines may anchor judgments. The process to ensure these baselines neither overfit to a model family nor penalize stylistic variance is under-specified.
- For principle-based (Tier-3) cases, safeguards to prevent over-crediting fluent but methodologically weak responses rely on judge prompts and principles, but inter-rater checks on these specific cases are not separated and reported.
- Token budgets, tool-call limits (200), timeouts (120s per action), prompts, and system settings are critical for agent performance but are only briefly summarized. A stricter ablation on these constraints (and their fairness across models) would clarify whether some models are disadvantaged.
- SQL dialect and tolerance: The evaluation requires exact schema+data equality, but it is unclear if numeric tolerances, null-ordering, and type coercions are handled consistently; small floating-point or timestamp discrepancies can introduce false negatives.
- Contamination risk: Some DE tasks originate from open-source dbt projects; without a contamination analysis, models might have seen parts of the ground truth or closely related code. The synthetic data helps, but existence of public repo code raises a nontrivial risk.
- For DA tasks, stability is reported on a subset (DE-Arch and DA) across 8 runs for a few models; confidence intervals and significance tests for between-model differences are not provided.
- For DE deterministic tasks, results appear single-run; while determinism reduces variance, reporting robustness across multiple seeds/decoding settings would be informative for agent frameworks that involve LLM sampling.
- Some interpretations could be more cautious: “Strong evidence that engineering and analysis are distinct capabilities” is supported by rank inversions, but causality is not established. Alternative explanations (prompt specializations, token policies, execution tool robustness) could contribute; a cross-task, cross-prompt ablation would strengthen this claim. “Even the top agents barely reach 50%” is accurate for aggregate scores, but the consequences for practical utility would benefit from error cost analysis (e.g., what classes of failures materially change business decisions vs. minor mismatches).
- While the paper includes a brief LLM‑usage disclosure, it does not discuss potential risks of releasing synthetic enterprise data (e.g., inadvertent leakage of real company patterns) or the impact of using proprietary LLM judges on reproducibility for the broader community.

### Questions
1. How do the DA scores change if a different LLM (e.g., GPT‑4.1 or Claude‑3) is used as the judge? Have you measured inter‑judge agreement?  
2. Can you provide an ablation study varying α (e.g., 0.5, 0.7, 0.9) and report its impact on model rankings?  
3. What is the distribution of difficulty (e.g., number of nodes, LOC) across DE tasks? Are the results driven by a small subset of very hard tasks?  
4. Have you measured human performance on a sample of DE and DA tasks to contextualize the reported model scores?  
5. Which random seeds, library versions, and hardware configurations were used? Are the execution scripts deterministic?  
6. How were the synthetic data distributions validated against real enterprise datasets? Could you provide quantitative similarity metrics (e.g., column‑wise KL divergence) to demonstrate realism?

### Soundness
2

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
This paper introduces DAComp, a comprehensive benchmark designed to evaluate data agents across the entire data intelligence lifecycle, spanning both repository-level data engineering and open-ended data analysis. The benchmark defines 236 tasks reflecting realistic enterprise workflows, combining deterministic engineering evaluations (execution-based) with open-ended analytical assessments (LLM-judge guided by hierarchical rubrics).

### Strengths
- DAComp is the first benchmark to unify data engineering and analysis within a single evaluation framework. 

- The benchmark construction pipeline is rigorous.

- The paper is dense but logically structured.

### Weaknesses
- While the LLM-judge method is well-validated, DAComp’s rubric framework is static. Whether adaptive rubric refinement is required?

- The experiments show low success rates but do not deeply isolate why orchestration fails? 

- The benchmark assumes one-shot or fixed-turn interactions. Yet many enterprise agents operate iteratively. DAComp currently lacks tasks or metrics reflecting closed-loop self-correction, which might undervalue agents with strong iterative reasoning skills.

- CS/CFS/SR performs strict schema and data equivalence checks. It is not tolerant of irrelevant equivalences (such as equivalent rewrites or minor floating-point differences), and may incorrectly flag semantically equivalent results as errors.

### Questions
see Weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
