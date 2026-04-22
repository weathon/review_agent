# HARDTESTGEN: A High-Quality RL Verifier Generation Pipeline for LLM Algorithimic Coding

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Verifiers provide important reward signals for reinforcement learning of large language models (LLMs).
However, it is challenging to develop or create reliable verifiers, especially for code generation tasks.
A well-disguised wrong solution program may only be detected by carefully human-written edge cases that are difficult to synthesize automatically. To address this issue, we propose HARDTESTGEN, an approach to synthesize high-quality test cases for algorithmic coding problems. We curate a comprehensive algorithmic programming dataset HARDTESTS with 26.6k problems and high-quality synthetic tests.
Compared with existing tests, \method tests demonstrate significantly higher accuracy in verifying LLM-generated code (+11.22 percentage points in precision, the percentage of actually correct code within the predicted correct ones). We also show that downstream post-training --- including rejection sampling and reinforcement learning (RL) --- using HARDTESTS verifier results in improved performance of LLM code generation.  We open-source our dataset and synthesis pipeline at https://leililab.github.io/HardTests/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents HardTestGen, a pipeline that synthesizes “hard” test cases for algorithmic problems via 4 different prompts, an LLM-prompted validator, and multi-oracle consensus for outputs. Using this LLM powered pipeline, the authors create 26.2k problems and show comparable (at easy level) or higher (at higher difficulties) precision and recall than the existing methods of TACO and CodeContents. They further demonstrate that post-training with RLVR on their tests leads to higher scores than post training with tests generated from an existing method, TACO.

### Strengths
* Coding LLMs are a hot and popular topic. The authors do well to point out the motivation, survey related works (especially with the extra offered in the appendix), and show practical impact/results
* Their method is automated, and scalable, resulting in over 26,000 problems. 
* They provide good evaluations and ablations into their different prompting techniques.
* Concurrent work is acknowledged, well presented, and well positioned.
* Overall the method seems reproducible as the authors offered all of their prompts in the appendix and oracle program sources.

### Weaknesses
* I think the motivation could be rephrased. The authors seem to teeter between problems that have incorrect solutions (the solution is genuinely wrong and fails to pass all test cases), and inefficient solutions (solutions that are correct but can take too long to execute). In the end they group both of these distinct cases and “incorrect solutions” but it’s clear that correctness != inefficient. 
* The second paragraph of the introduction goes over a tree example where a naive algorithm works for randomly generated trees but gives the worst time complexity for a tree who is a chain. This isn’t well motivated. For example, quicksort has a worst case time complexity of n^2 yet is or has been the default sorting algorithm used in Java, C++, C#, and Julia. Despite the worst case time complexity it is favored over merge sort which has a better worst case complexity. 
* 25% of the problems require checking equality in a method other than string comparison and in these cases an LLM is prompted for a judging function. Then on line 256, the authors offer a set in python as an example. Sets in python have a built in equality operator and presumably an LLM would propose using that. It would have been nice to see in these cases how often do LLMs use simple built in solutions, and how often do they have to write something intricate from scratch.
* The authors say that two of their prompting techniques for test generation, RPGen and SPGen are mutually exclusive yet Table 1 does not include HackGen and does in fact combine these two mutually exclusive settings. Are the results in the table still valid?
* Given that this benchmark cost over $6,000 ($0.23/problem * 26,000 problems) I question whether the method is truly scalable. Arguably, maybe a program but certainly properties/contracts of a program could be expressed as a set of rules and an SMT solver could be used to find edge cases. One could imagine replacing the prompts for RPGen+SPGen+HackGen with a single prompt to generate such rules and then running the SMT solver. 
* Qualitative evaluations could use a slight boost. For example, Table 2 shows a perplexing result in which training on good examples leads to worse performance. A deeper dive as to why would have made for interesting auxiliary results.

### Questions
* The second bullet in weaknesses leads to an interesting gap in the analysis of their methodology. Of the test cases generated, how often do we end up with tests that are forcing the worst case time complexity? One can argue that generating the fastest code leads to better LLMs but without a deeper understanding, at what point do we cross into hyper optimized code that is no longer easily human readable? 
* Why was 90% chosen as the consensus threshold? 
* Section 5.1, line 426, was the resulting random sampling of trajectories near 50/50 in terms of correctness?
* Why did you not use KL divergence in your RL loss? Could that be why 
* Could you provide a taxonomy of programs that still slip through to guide future research?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the low-quality test case problem in algorithmic coding benchmarks used for LLM training and evaluation.  The authors propose HARDTESTGEN, a multi-stage LLM-based test synthesis pipeline that produces diverse and valid test cases through four modules: LLMGen, RPGen, SPGen, and HackGen.  These modules generate inputs covering functionality, scalability, categorical balance, and adversarial edge cases. Each generated input is validated by an LLM-written input checker and cross-verified with multiple oracle programs for ground-truth outputs, with consensus filtering to ensure reliability. Using HARDTESTGEN, the authors build HARDTESTS, a dataset of 26.6k problems from 13 competitive programming platforms.  Experiments on AtCoder and Codeforces (1,253 problems) and 800 human submissions show substantial improvements in precision and recall over prior datasets (TACO, CodeContests).  Additional studies demonstrate that test quality positively affects downstream post-training tasks such as rejection sampling and reinforcement learning, with notable performance gains on LiveCodeBench-105.

### Strengths
- The modular design of HARDTESTGEN (LLMGen, RPGen, SPGen, HackGen) captures diverse failure modes, a broad coverage rarely seen in prior test-synthesis systems. The inclusion of an input validator and consensus output filtering via multiple oracle programs substantially improves validity and robustness.

- HARDTESTS fills an important gap for competitive-programming-level test data. The evaluation on 26.6k problems and the analysis of precision/recall against established datasets is rigorous. The 11-point improvement in both precision and recall and detailed ablations support the paper’s claims.

- The authors go beyond static metrics and show that better verifiers translate into improved LLM training outcomes, especially in reinforcement-learning settings—an important and underexplored direction.

- Method descriptions, example prompts, ablations, and cost statistics (≈ $0.23 per problem) are detailed.

### Weaknesses
- HARDTESTGEN assumes access to multiple human-written correct programs to compute reference outputs and validate tests. This assumption limits scalability to unseen or synthetic problems without oracles. While the authors briefly mention an “oracle-free” variant in Appendix A.8, it is minimally evaluated (165 problems) and shows time-out issues. This weakens claims of general applicability.

- The “DOWNSTREAM EFFECTS” section evaluates only Qwen3-4B, and LiveCodeBench-105 contains just 105 questions. This small testbed cannot robustly demonstrate generalization across architectures or domains. Additional experiments on open-weight models (e.g., DeepSeek-Coder, StarCoder2) or larger LiveCodeBench subsets would strengthen the results.

- Although decontamination with LiveCodeBench is mentioned, it is unclear whether other datasets (e.g., CodeContests+, TACO) share problems or reasoning traces with HARDTESTS, which could inflate evaluation gains.

- While quantitative metrics are thorough, the paper could benefit from qualitative examples showing how HARDTESTGEN discovers subtle inefficiencies or corner-case failures that baselines miss.

### Questions
1. Can HARDTESTGEN operate when no oracle exists (e.g., synthetic benchmarks or new competition tasks)? What proportion of problems in the dataset had multiple oracles versus single or none?

2. How do the results change if GPT-4o is replaced with a smaller or open-source model (e.g., DeepSeek-Coder-33B)?  Appendix A.9 is mentioned but not summarized.

3. What criteria were used to select the 105-problem subset in LiveCodeBench-105, and how does it differ from the full benchmark?

4. Given the cost of $0.23 per problem, is HARDTESTGEN cost-effective compared to simpler mutation-based generation (e.g., CodeContests+) when scaled to millions of tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a pipeline for generating synthetic programming questions based on seed questions collected from programming contests. It provides a set of 26,000 generated questions for use in LLM training and shows that training on these questions outperforms training on two other synthetic code datasets in fine-tuning and RL contexts.

### Strengths
- The paper compares post-training performance to other synthetic code datasets
- The pipeline seems sensible for using consensus, judging, and automated methods to improve question quality

### Weaknesses
- a comparison against a comparable real data-set is not performed
- discussion of how many generated instances were rejected at various stages was not discussed, which would be interesting to know for judging the effectiveness of each component. In addition, without this it is entirely unclear what sort of cost and computational resources would be needed to replicate this study.
- Reviewing a sample to estimate quality does not appear to have been done.

### Questions
- I'm curious what impact this has on less in-distribution coding, for example SWE-bench

- Agreeing on 90 of inputs doesn't seem like a lot, I'm somewhat surprised this doesn't lead to quite a few errors in tests.
"If two oracle programs generate outputs that are equivalent on more than 90% of the inputs (i.e., semantically the same rather than strictly identical), we regard this agreement as valid. "

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
Hardtestgen is proposed as an LLM pipeline to generate verifiers for coding problems, with the key goal to achieve higher quality (precision and recall) verifiers than datasets from prior work. The core improvement is brought by augmenting the test generation with diverse test inputs and edge cases. This is accomplished by using LLM generated programs to generate test cases instead of generating test cases directly with an LLM. The resulting dataset is shown to be higher in quality than existing tests besides being larger in scale. When used to post-train code LLMs these proposed verifiers show higher effectiveness due to their quality and lead to stronger model performance.

### Strengths
- The space of datasets that provide execution-based feedback for RL algorithms is limited; this work is well-motivated and can advance research on training of stronger code LLMs.
- The analysis of current datasets’ limitations is fresh and under-addressed in prior work. The critique of prior test-suite generation methods is valid; the proposed improvement - explicitly generating diverse inputs and edge cases - is solid and novel.
- Even though only briefly discussed, the investigation into how verifier/test quality affects LLM post-training offers valuable insights.

### Weaknesses
- Precision estimates are based on a small subset. The paper tackles dataset quality but reports precision using <5% of the full set (~1.2k of 26k), which - while better than prior work - does not convincingly characterize the overall dataset quality.
- Verifier quality beyond correctness is not analyzed. Apart from precision/recall, the cost profile of the verifier is missing. In particular, the execution time of HARDTESTGEN’s test suites relative to prior work is not discussed; a costlier suite can be problematic. Ideally, we would prefer a minimal test count that still accurately verifies correctness, but this dimension of test-suite quality is not addressed.
- Limitations are under-discussed. The gap in precision raises questions: when a test suite is found to be incorrect (FP or FN), is that suite still retained in the final 26.6k-problem dataset? Even if improvements over prior work hold, precision in the 80% or 50–60% ranges (Figure 1) could inject noisy signals into post-training.
- Cascaded LLM components lack evaluation. The overall pipeline is a cascade of LLM calls (Sec 3.2/3.3), each prone to error, yet individual components are not sufficiently analyzed. For instance, the LLM-generated output-equivalence function (discussed around L259) needs a reported correctness rate; at minimum, a manually inspected subset could establish its reliability.

### Questions
- How does the quality of the generated datasets vary with stronger frontier models (which have moved beyond GPT-4o in coding/reasoning)?
- Filtering in 3.4. What percentage/number of test cases are filtered out? Have you tried automatic re-attempts (regeneration) to raise precision/recall when a generated suite is flagged as incorrect?
- Decontamination in 3.5. How exactly is decontamination performed (e.g., exact-match criteria, near-duplicate thresholds)?
- In Table 1: Why are precision/recall gains on human submissions smaller than on model-generated programs?
- Can you explain the rejection-sampling study design? A more informative experiment would be similar to Figure 3 with an A/B setup for rejection sampling: (A) use HARDTESTS verification feedback vs. (B) use TACO verification feedback, to isolate the impact of verifier quality in that setting.

### Soundness
2

### Presentation
2

### Contribution
3
