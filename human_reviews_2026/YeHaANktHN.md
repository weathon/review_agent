# FeatBench: Evaluating Coding Agents on Feature Implementation for Vibe Coding

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
The rapid advancement of Large Language Models (LLMs) has given rise to a novel software development paradigm known as “vibe coding,” where users interact with coding agents through high-level natural language. However, existing evaluation benchmarks for code generation inadequately assess an agent’s vibe coding capabilities. Existing benchmarks are misaligned, as they either require code-level specifications or focus narrowly on issue-solving, neglecting the critical scenario of feature implementation within the vibe coding paradiam. To address this gap, we propose FeatBench, a novel benchmark for vibe coding that focuses on feature implementation. Our benchmark is distinguished by several key features: **❶ Pure Natural Language Prompts.** Task inputs consist solely of abstract natural language descriptions, devoid of any code or structural hints. **❷ A Rigorous & Evolving Data Collection Process.** FeatBench is built on a multi-level filtering pipeline to ensure quality and a fully automated pipeline to evolve the benchmark, mitigating data contamination. **❸ Comprehensive Test Cases.** Each task includes Fail-to-Pass (F2P) and Pass-to-Pass (P2P) tests to verify correctness and prevent regressions. **❹ Diverse Application Domains.** The benchmark includes repositories from diverse domains to ensure it reflects real-world scenarios. We evaluate two state-of-the-art agent frameworks with four leading LLMs on FeatBench. Our evaluation reveals that feature implementation within the vibe coding paradigm is a significant challenge, with the highest success rate of only 29.94%. Our analysis also reveals a tendency for “aggressive implementation,” a strategy that paradoxically leads to both critical failures and superior software design. We release FeatBench, our automated collection pipeline, and all experimental results to facilitate further community research. Our code is available at https://anonymous.4open.science/r/FeatBench-D3C5.
“The hottest new programming language is English.”
—Andrej Karpathy (Karpathy, 2025b)

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper curates a new benchmark, FeatBench, designed to measure the ability of LLMs to "vibe code": produce repository patches that satisfy a natural language request and its paired unit tests. FeatBench includes 157 tasks from dozens of high-quality open-source repositories covering several different topics. The authors then benchmark existing competitive agentic frameworks + LLMs on FeatBench, concluding with analysis on common failure modes of existing coding agents on their benchmark.

### Strengths
FeatBench is a topical release, given the current interest in developing powerful coding agents and the increasing prevalence of "vibe coding". Given the reported low performance of SOTA coding agents on FeatBench, it also seems useful for researchers and developers interested in testing the quality of their coding agent.

## Interesting preliminary analysis 
I appreciate the clean and simple analysis conducted in section 3, such as the details on how simple measures of both repository complexity and patch complexity affect feature implementation resolution rate. The analysis sheds light on difficult problems in using LLMs in the software development workflow such as how much design control to give the LLM, and how to develop code that is maintainable to future LLMs and human developers.

## FeatBench engineering should make it easy to use
The authors of FeatBench include Docker containers and thorough testing for LLM agents to propose and iterate on code modifications to complete the task. This engineering support is a welcome strength point for FeatBench and the released repository appears thorough + well-documented (though I cannot speak to the robustness or ease of use because I did not clone and interact with it myself). The repository also includes code to analyze repositories, which other researchers can use if they want to grow FeatBench for their own use cases.

### Weaknesses
## "Vibe coding" novelty is unclear

Several coding benchmarks have been released, all with various focus points, as the authors of this paper point out. It seems that the primary contribution of FeatBench over other similar coding benchmarks is that the input is standalone natural language following a "I want..." format. Other repository coding benchmarks detail the specification with natural language for refactoring (RefactorBench), in a github issues format (SWEbench) or docstrings (Commit0). This difference seems minimal, especially since the agents using FeatBench seem to still get the use of given unit test cases.

### Questions
* Are the unit tests visible to the agent before final evaluation? If so, wouldn't that make the unit tests part of the input as well and thus not necessarily "pure natural language prompt"s?
* The authors point out that the "Trae-agent framework paired with a top model like Claude 3.7 Sonnet(Anthropic, 2025) reached a 65.67% resolution rate on SWE-bench." The included twitter quote also references Claude + Cursor. Given these, I would have liked to see a Claude model + Trae-agent on FeatBench, to properly compare the two benchmarks.

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
3

### Summary
This paper introduces a benchmark for code models. The particularity of this dataset is that:

- it is agentic, in the sense that more than one LLM call can be issued before submitting the final patch. Two scaffold are benchmarks

- the prompts are requests of feature implementations, not bug fixes which are covered by other benchmarks


Various frontier models are benchmarked, there is some analysis and a few examples

### Strengths
More code benchmarks are always welcomed, as this is a fast-evolving field where existing frontier benchmarks are often overfit. This one has a decent size (157, from 27 repositories). The datapoints seem to be dockerized, which is always a cumbersome step: if this is indeed already done it would help replicability

### Weaknesses
The focus on "vibe-coding" seems to be done to ride an existing hype. In particular, this field is never really defined (beyond "this approach allows users to program by interacting with an LLM-powered coding agent through high-level, abstract requests in natural language"). This benchmarks is valuable in itself, but not necessarily because it measures progress for some ill-defined concept

I could not inspect the datapoints itself, as the anonymized repository only allowed access to the README.md file (for all others I just saw a `The requested file is not found.`)

Without reading the prompts, it is not clear if they are unambiguous: ambiguous descriptions are a key ingredient of "vibe coding". In particular, the error analysis seem to indicate that part of the problem could be that the preamble/prompt is underspecified (see below)


nitpick: authors should uniformize how they refer to figures (both "Fig." and "Figure" are used)

### Questions
The error analysis reveals that ~75% of the failure cases are due to "scope creep", where the provided patch breaks a regression test. Did you try with solving this through prompt-engineering? It seems an easier problem to solve than the other categories.

### Soundness
3

### Presentation
2

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
The authors introduce FeatBench, a novel benchmark designed to evaluate coding agents on feature implementation tasks under the vibe coding paradigm. Unlike previous benchmarks such as SWE-bench, FeatBench focuses on assessing agents’ ability to implement new software features based solely on pure natural language requirements, without code-level hints. The benchmark is built through a rigorous, automated data pipeline and provides comprehensive test cases to validate both new functionality and regression avoidance. Experimental results reveal that feature implementation from abstract natural language remains highly challenging for state-of-the-art agents, with top success rates significantly lower than on traditional issue-solving tasks.

### Strengths
1. **Practical Relevance:** 

The benchmark targets a highly practical and emerging domain. By constructing the dataset from a non-developer perspective, the authors make a valuable step toward more realistic and accessible evaluation scenarios for LLM-based coding agents.

### Weaknesses
1. **Limited Technical Contribution:** 

The paper mainly focuses on dataset construction and benchmark design. There are no novel technical methods or algorithmic contributions beyond the creation of the benchmark.

2. **Lack of Quantitative Data Quality Analysis:** 

The paper does not provide concrete quantitative indicators regarding the quality of the dataset (e.g., annotation accuracy, consistency checks, or human evaluation of correctness).

3. **Insufficient Quantitative Comparison with SWE-bench:** 

While the authors motivate FeatBench as being more challenging and realistic than SWE-bench, there is no detailed, quantitative analysis of the differences. In particular, the paper lacks experiments or error analyses that concretely illustrate why existing methods perform much better on SWE-bench but struggle on FeatBench, and what aspects of the tasks or data distributions lead to this performance gap.

### Questions
Same to Weaknesses.

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
5

### Summary
The authors propose FeatBench, a benchmark for coding agents with a focus on feature development rather than bug fixing. Feature PRs are mined from reference in release notes. An LLM is used to synthesize the prompt from the release note's feature description, PR title and descriptions, as well as file changes. The authors use an LLM to set up execution environments and derive F2P & P2P tests as done in comparable benchmarks. The authors then evaluate Agentless and Trae agent on the data set using several SOTA LLMs on pass rates, file retrieval metrics, and patch apply rate. Overall, the authors observe a good file retrieval accuracy of the Trae-agent but low pass and regression rates. Lastly, the authors perform case studies showing strengths and weaknesses of the evaluated coding agents and close with a statement about the duality (pros and cons) of aggressive implementation.

### Strengths
* Having a benchmark that focuses on feature development is important and the approach of using release notes is a good starting point.
* Paper is easy to follow and visualizations are insightful

### Weaknesses
* You consider SWE-Bench a benchmark that doesn't contain any feature requests. This is not true. As quantified by Rashid et al.[1] (a work you may want to include in Table 1 as it contains 22% feature requests), it actually contains 18% feature requests.
* I haven't seen Trae-Agent being an established agent scaffold. It would be great to have at least one other coding agent in the comparison (e.g., SWE-Agent, OpenHands or Aider)
* Features that are testable with existing unit tests are typically only small feature improvements but not the addition of large new features. For FeatBench, I'd expect a benchmark that can evaluate such big features. 
* Solvability: LLM-generated prompts pose the risk of making part of the dataset not solvable. How do you make sure that each instance can be solved with the provided information?
* The dataset only contains Python repositories which limits the way that coding capabilities as a whole can be evaluated

1. Rashid, M. S., Bock, C., Zhuang, Y., Buchholz, A., Esler, T., Valentin, S., ... & Callot, L. (2025). SWE-PolyBench: A multi-language benchmark for repository level evaluation of coding agents. arXiv preprint arXiv:2504.08703.

### Questions
* I assume you only include a PR if it is directly linked to a feature in the release notes, correct? How do you make sure of that?
* In line 186, you say you're curating high-quality repositories. How do you determine the quality? The same question applies to line 194.

### Soundness
3

### Presentation
3

### Contribution
2
