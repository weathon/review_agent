# TestExplora: Can LLMs Write Tests to Find Potential Problems Existing in Repository?

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
As Large Language Models (LLMs) are increasingly applied to automate software development, their use for automatic test case generation has become a key area of research. However, existing benchmarks for evaluating LLMs fundamentally simplify the real-world testing challenge. They typically constrain the problem to either (1) reproducing known bugs at the repository level, or (2) generating tests for isolated code units, such as individual functions, detached from their broader project context. Both approaches fail to assess the crucial capability of LLMs for proactive, exploratory testing in projects defined by complex, cross-file dependencies.
To address this critical gap, we introduce TestExplora, the first systematic benchmark designed to evaluate the proactive defect discovery capabilities of LLMs at the repository level. Constructed from real-world pull requests, TestExplora challenges models to find bugs without any prior knowledge of bug manifestations. Our comprehensive evaluation, conducted in both black-box and white-box settings, reveals a stark capability gap. Even state-of-the-art models exhibit critically low success rates (e.g., GPT-5-mini: 17.56%, o3-min: 5.23%), and access to the full source code (white-box) yields only marginal improvement. Further Analysis reveals that existing models struggle mainly with assertion mismatches and misconfigured mocks. TestExplora thus establishes a principled foundation for advancing research towards the grand challenge of autonomous, repository-level defect discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces TestExplora, a new test case generation benchmark for proactive, exploratory testing at repository level. The authors construct the benchmark by taking a valid PR (after careful filtering and selection process) and extract 3 pieces of information: documentation about the intended functionality, entry point, and their dependencies. The overall task is to let the LLMs generate test cases for a piece of function given larger (repo-level) context without access to information pointing to bugs (e.g., issues, fixing commits).

The authors evaluate the LLMs with several metrics capturing different scopes, under two scenarios: white-box and black-box, being different in terms of whether the actual code is visible to LLMs or not. The paper's key finding is that even state-of-the-art models perform very poorly on this task, with a low success rate (e.g., 17.56% for GPT-5-mini). The analysis suggests models primarily fail due to assertion mismatches and misconfigured mocks, showing that there still exist a significant gap toward finding defects proactively.

### Strengths
* A solid contribution of large amount of highly curated data instances & PRs
* Good findings about the SOTA LLMs not being robust wrt finding the potential bugs without access to the actual errors. Proactively assessing the bug attracts interest in real-life cases where running the test suites are expensive. So I think it is a nice direction to pursue.

### Weaknesses
* The novelty is limited. SWT-bench is a repository-level test generation benchmark, but this is given the bug report. It sounds to me that the proposed benchmark only differs conceptually from SWT-bench in that the input is intended behavior instead of the bug report. Though still nice to see the limitations of SOTA models, the claimed novelty is somewhat incremental.
* Somewhat shallow analysis. Experimental settings are great, but the analysis doesn't fully dissect the results. For example, whitebox and blackbox testing show similar F2P scores. I understand both settings have similar error patterns, but what parts are different? If not so different, does that mean the models are simply not utilizing the provided context?
* DocAgent provides better documentation – is it longer, more structured, comprehensive? It might be expected if Human-written versions tend to have only brief documentations. If we want to simulate a real-world setting, should we modulate the DocAgent output to mimic human-written documentations?

—

Other notes:

* Sec 4.2 starts with a sentence repetition.
* TestExplora-Lite is only briefly mentioned in Sec 4.3. It'd be nice to define it if it also appears in the main results table.

### Questions
Please see above.

### Soundness
3

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
TestExplora is a new benchmark set to assess LLM’s abilities to proactively detect bugs through tests generated. The benchmark is composed of 2389 test generation tasks from 482 repositories. LLMs are provided test entry points along with their human and LLM generated documentation and prompted to generate tests. The performance of the LLMs are assessed using various metrics, notably with fail-to-pass rates which checks if generated tests exhibit a Fail-to-Pass transition with selected PRs. The best LLM in this task displays 17% success rate, demonstrating challenging aspects of this benchmark.

### Strengths
- The paper is well written and organized.
- The work is well motivated. Proactive detection of bugs through LLM-based test generation would be of interest to the field.
- The benchmark is a novel contribution motivating further advancements for coding performance of LLMs.

### Weaknesses
- My understanding is that the benchmark is only composed of python examples, which might be limiting.
- The evaluations are lacking error bars. As LLMs are not deterministic, I’d love to understand the variability in the results. Also lacking one of the strongest coding LLM from Claude line of models.

### Questions
- Curious to hear if the authors considered repeating the experiments where they provide a hint to the LLM that there is a bug and would love to see if it would help LLMs to be more critical and observe how the results change.
- In Table 3:
   - What is the “Num” column?
   - There are several cases where the results are lower in White Box cases as compared to Black Box cases (e.g. GPT4-o for EC and CFG). It felt counterintuitive to me, can you elaborate on this behavior?
- In the main discussion as well as Figure 2, it is indicated that Black box case does not include code implementation. However looking at the prompts in the appendix, I only see dependencies as the delta since test entry functions are included for both. Could you confirm what the delta is between White and Black Box?
- Typo in line 372: “TThese”

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces **TestExplora**, a benchmark targeting *proactive* test generation for real-world repositories without giving models explicit bug cues. Tasks are derived from GitHub PRs (2,389 tasks, 482 repos), and the benchmark enforces fail-to-pass behavior: the generated test must fail on buggy code and pass after patching. The benchmark evaluates both black-box and white-box modes and reports that state-of-the-art LLMs perform poorly (typically <13% F2P), suggesting current models struggle at autonomous exploratory testing. The authors also introduce DocAgent to synthesize/clean documentation for models.

### Strengths
The benchmark pipeline and fail-to-pass validation methodology are generally reasonable, with realistic environment setup. However:

- No statistical significance or variance reporting
- Potential leakage: PRs might be in model training data
- DocAgent-generated docs may introduce *task supervision*, contaminating “exploratory” nature
- Some pipeline heuristics lack ablation (entry-point selection, doc filtering)
- Limited robustness checks (e.g., determinism, flaky tests, environment noise)

### Weaknesses
#### 1. Benchmark purity & leakage concerns
- PRs and project history may be in model training sets
- No explicit contamination filtering (hashing, repo disambiguation)
- DocAgent can implicitly encode bug semantics → *post-hoc supervision*

#### 2. “Exploratory testing” assumptions questionable
- Providing synthetic docstrings is not realistic exploratory QA
- Entry-point constraints shape model behavior artificially
- Real exploratory testing = multi-turn reasoning + interaction + search
  → model evaluated as *static predictor*, not explorer

#### 3. Limited baselines
- No comparison to property-based testing or symbolic tools (Hypothesis, Pynguin)
- No LLM agent systems with tool use
- No RL / planning / iterative search baselines

Benchmark may favor static prompting over realistic agent loops.

#### 4. Metrics interpretation unclear
- F2P alone doesn't guarantee good test quality
- Coverage ≠ fault detection power
- No mutation-testing analysis
- No difficulty stratification or variance reporting

#### 5. Reproducibility risks
- Python dependency resolution is brittle
- Flakiness/oracle reliability unclear
- Many heuristics for filtering and environment setup insufficiently justified

### Questions
1. How do you ensure PRs were not in model training data? Any contamination checks?
2. How do you guarantee DocAgent never leaks patch semantics?
3. Why constrain to function-level entry points? Why not broader exploration?
4. Any evaluation with autonomous tool-using agents?
5. Can you include mutation-testing metrics?
6. How robust is pipeline to flaky tests / dependency conflicts?

### Soundness
2

### Presentation
2

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
The paper introduces TestExplora, the first systematic benchmark designed to evaluate the proactive defect discovery capabilities of LLMs at the repository level. The evaluation shows that state-of-the-art models exhibit critically low success rates, and access to source code (white-box) yields only marginal improvement.

### Strengths
**Originality**
The paper introduced the first large-scale benchmark from real-world GitHub PRs to evaluate the proactive defect discovery capabilities of LLMs at the repository level.

**Quality**
The paper made extensive experiments on various LLMs and white-box setting with entry function and dependency implementation, showing quantitative results of with multiple metrics like fail-to-pass rate and change-focused coverage.

**Clarity**
The paper is well-written with description of benchmark construction methods and evaluation metrics. The appendix provides detailed prompt.

**Significance**
The paper proposed a large-scale benchmark which can help researchers in AI and SE fields to measure LLM's capability of bug-finding in real-world repositories.

### Weaknesses
**Missing literature from SE field**

Even though the paper defines the problem as repo-level proactive exploratory testing, the benchmark construction from GitHub PRs is essentially similar to the regression testing problem, which has been explored by following recent work from SE field. The key difference is that TestExplora shows LLM with the entry code but not the diff/patch so it's harder for LLM to infer the bug.

These two papers focus on bug reproduction:
- Automated Generation of Issue-Reproducing Tests by Combining LLMs and Search-Based Testing (ASE '25)
- Issue2Test: Generating Reproducing Test Cases from Issue Reports (ICSE '26)

This paper finds unintended bugs introduced by PR:
- Testora: Using Natural Language Intent to Detect Behavioral Regressions (ICSE '26)

This paper feeds bug-introducing and bug-fixing commit for LLM to generate bug-triggering and bug-reproducing test input.
- Can LLM Generate Regression Tests for Software Commits? (arXiv:2501.11086)

**Lacking qualitative example about LLM inferring intended behavior**

Figure 2 shows the task for LLM includes "infer the intended behavior of the Test entry points' API from the documentation", which is crucial for LLM to generate the test. I don't see results about how successful LLM is on this task. Figure 6 shows some failure analysis result about assertion mismatch, but the classification criteria is unclear. It would be better to have an example showing the full output of LLM, especially on how it infers the intended behavior in the first place.

### Questions
1. How do you assess if LLM successfully infer the intended behavior of the Test entry points' API so it tries to generate test that exercises the intended functionality?
2. The proactive exploratory setting is naturally difficult for LLM and human developers as the entry function itself may contain other distracting lines that may even contain other unintended bugs. Would it be better if you identify the problematic lines or the bug-introducing diff so LLM has a clear goal and context to test? In real-world CI/CD scenario it's also natural to test code change rather than the entry function.

### Soundness
2

### Presentation
3

### Contribution
3
