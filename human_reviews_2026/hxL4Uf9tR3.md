# Pushing Test-Time Scaling Limits of Deep Search with Asymmetric Verification

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Test-time compute can be scaled both sequentially and in parallel. Sequential scaling involves lengthening the generation process, while parallel scaling involves verifying and selecting among multiple candidate outputs. Combining these two strategies has led to the most powerful AI systems, such as Grok 4 Heavy, GPT-5 Pro, and Gemini-2.5 Pro Deep Think. A key observation is that, in certain contexts (e.g., solving Sudoku puzzles), verifying responses can be substantially easier than generating them. This property, referred to as \emph{asymmetric verification}, highlights the strong potential of test-time scaling. In this work, we study both sequential and parallel test-time scaling of deep search agents, motivated by the intuition that verification in this setting is often much easier than generation. In experiments, we first show that sequential scaling methods, such as budget forcing, can be effective initially but eventually degrade performance when over-applied in agentic search. Due to asymmetric verification, however, we are able to achieve substantial improvements by allocating only a modest amount of compute to the verifier. We conduct experiments with flagship open-source models, including GLM-4.5, K2, Qwen3-2507 and Tongyi-DeepResearch, and extend them to their ``Heavy'' variants through test-time scaling. These deep research agents achieve improvements of up to 20 absolute points on benchmarks such as BrowseComp. Remarkably, as an open-source alternative, GLM-4.5 Heavy reaches accuracy of {\bf 54.0\%} on BrowseComp, {\bf 66.0\%} on GAIA, and {\bf 68.0\%} on xbench-DeepSearch, placing it on par with the best proprietary choices such as OpenAI Deep Research and o3. Tongyi-DeepResearch Heavy pushes performance even further, attaining {\bf 69.0\%} accuracy on BrowseComp.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates test-time compute scaling for deep search agents through the lens of "asymmetric verification" - the observation that verifying answers is often cheaper than finding them. The authors experiment with sequential scaling (budget forcing, max tool calls) and parallel scaling strategies across search and verification agents. Testing on BrowseComp, GAIA, and other benchmarks with models like GLM-4.5, K2, and Qwen3-2507, they show that allocating compute to verification yields better cost-efficiency than pure search expansion. For instance, GLM-4.5 improves from 19% to 54% on BrowseComp when extended to a "Heavy" variant through their scaling approach.

### Strengths
**Empirical rigor**: The experimental setup is comprehensive, testing 4 models across 4 benchmarks with systematic ablations. The scaling curves and trade-off analyses are well-executed.

**Clear presentation**: The paper tells a coherent story. Figure 1 effectively summarizes the key findings, and the concept of asymmetric verification is intuitive and well-motivated through concrete examples.

### Weaknesses
**Limited technical novelty**: The core approach is standard Best-of-K selection with a reward model (here called "verifier"). This is well-established in RLHF and LLM literature. The authors essentially apply existing techniques to a new domain without methodological innovation.

**Lack of theoretical framework**: The paper doesn't provide any optimal resource allocation strategy between search and verification. What's the right balance given a compute budget? The paper only offers empirical observations from grid search, not principled solutions. No complexity analysis, convergence guarantees, or formal optimization framework.

**Missing predictive framework**: A critical limitation is the absence of any method to predict whether a task will benefit from verification optimization. The paper only provides post-hoc measurements (Table 1) without actionable criteria. Users can't determine a priori whether to apply this approach to new tasks.

**Case study rather than research contribution**: The authors acknowledge this is studying deep search "as a representative case" but don't provide generalizable insights beyond this specific application. The findings largely confirm expected behavior (verification is cheaper than search) without revealing new phenomena.

**Incomplete analysis**: Why does xbench-DeepSearch show minimal improvement? The paper mentions verification and search have similar difficulty but doesn't deeply analyze when and why asymmetric verification breaks down.

### Questions
1. **Can you provide a decision framework for task suitability?** 

    What specific features of a task predict whether verification optimization will help? Could you test a simple classifier or heuristic rules on held-out tasks?

2. **What's the optimal allocation strategy?** 

    Given compute budget B and measured costs for search (75 calls) and verification (18 calls), what's the theoretical optimal split? Even a simple analytical model would strengthen the contribution.

3. **How does this differ from standard reward modeling?** 

    The paper should explicitly acknowledge that the "verifier" is functionally equivalent to a reward model. What, if anything, makes this application special beyond the domain?

4. **Why not adaptive strategies?** 

    Rather than fixed configurations, why not dynamically adjust the search/verification balance based on observed success rates? This could lead to more efficient resource use.

5. **Generalization beyond deep search?** 
    
    The asymmetric verification property likely applies to many tasks (math problems, code generation, fact-checking). Can you demonstrate broader applicability or is this fundamentally limited to web search?

6. **Statistical significance?** 

    The experiments use only 100-sample subsets. Do you have confidence intervals or significance tests for the reported improvements? How stable are these results across different samples?

7. **Failure mode analysis?** 

    When verification disagrees across multiple runs for the same answer, what's happening? Understanding these cases could provide insights into the limits of the approach.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies the problem of test time scaling from a perspective of assymetric verification. The overall intuition is straightforward: verification is easier then generation. e.g. verifying if an agent or llm solved a task e..g sudoku is much easier then actually solving the puzzle. For test time scaling the paper considers two approaches: sequential scaling which is simply budget forcing (simple scaling paper). For parallel scaling, the authors consider parallel sampling with a verifier. The agent generates multiple parallel responses which are aggreagated through majority voting or external verifier. Overall, the paper shows that each of these approaches have limitations (fig. 2, 3), which can be solved to some extent with test time scaling with assymetric verification.

### Strengths
* The paper studies an important problem of test time scaling whe verification is easier then generation.

* Fig. 2 and Fig. 3 provide a nice visualization on limitation of budget forcing and parallel scaling, applied alone.

* The authors  show efficiacy of their method with performance gains on BrowseComp and GAIA.

### Weaknesses
* Some of the statements in the paper are unsubstantiated. For instance, the paper says that "in forward search must navigate an enormous, sparsely
informative space, while backward verification ... drastically shrinking the search space and making far more efficient use of compute." While it intuitively makes sense it needs supporting evidence or references showing that backward verification or per-step verification is shrinking the search space for test-time scaling or forward search.

* The introduction provides some overview of the approach but its not very clear until later sections on how exactly assymetric verification is used for test-time scaling.

* The idea that "verification is easier then generation" is not very novel and already used in prior works [1,2]. Thus, I feel its important to provide detailed discussion on the how the proposed approach differs from prior works using verification for better tts (test time scaling)

* For fig. 2, the authors use number of tool calls for specifying the budget. Have the authors considered standard number of tokens for measuring scaling of performance with the used budget?

* Minor question, Maj@K in Line 196 is same as standard Best@K right with majority verifier?

* Some of the details in Sec. 3 are not clear. Sec. 3 does a good job of providing high level details and intuition but it should have concrete details for how exactly assymetric verification is used for test-time scaling.

* Have the authors tried comparing their approach to recent test-time scaling approaches like hybrid scaling[1] which also studies the verification for test-time scaling?

* Also I would be interested in how this approach differs and compares to "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling"[2] which also combines use of verifiers with parallel sampling for test-time scaling?
    - the main difference seems to be that here the verifier is based on majority voting while in the large language monkeys paper the verifier could be both based on majority or as an exteernal verifier?
    - Thus, it will be important to compare the proposed approach with the "large language monkeys" paper in terms of both conceptual difference as well as empirical performance under different verifiers.

* Some of the future work mentioned in the Conclusion section "Looking ahead, we aim to adopt more flexible strategies. For example, verifiers could be applied at each step along the search trajectory, ... " has already been explored in context of coding agents[3]. Thus, I feel the paper will benefit from thorgh discussion with prior works in this area.


References:
[1] R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents, COLM 2025

[2] Large Language Monkeys: Scaling Inference Compute with Repeated Sampling, 2024

[3] SOTA on SWE-Bench Verified with Inference-Time Scaling and Critic Model, 2025

### Questions
* The agentic variant of budget forcing from simple scaling paper is nice as showin in fig. 14. Have the authors tried other budget forcing prompts for the same?

Please also see the weaknesses section for some additional questions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies how previous test-time compute scaling, including sequential scaling and parallel scaling, affects deep search agents. It identifies a critical property of deep search agents, which is referred to as asymmetric verification, that verifying an answer is easier than generating it. Based on that, this paper proposes a new test-time scaling (TTS) method by constructing a verification agent and allocating a modest amount of compute to the verifier to verify answers retrieved from the search agent. Experimental results demonstrate that the proposed TTS method yields better accuracy–cost trade-offs than scaling computation on the search agent alone. Then, this paper extends flagship open-source models (GLM-4.5, K2, Qwen3-2507, Tongyi-DeepResearch) into “Heavy” variants using the proposed TTS method, and these variants achieve or surpass the performance of proprietary systems like OpenAI o3 and OpenAI DeepResearch.

### Strengths
1. The paper is well written and structured, with clear motivation and logical flow. Figures effectively summarize trends and highlight how the proposed TTS method boosts the performance of deep search agents using the property of asymmetric verification.
2. Discovering the asymmetric verification property of deep search agents and designing a verifier-based TTS method based on this discovery is highly insightful.
3. Experimental results on multiple benchmarks show that the proposed TTS method achieves a better accuracy–cost trade-off than previous TTS methods (e.g., sequential scaling, parallel scaling). More importantly, the proposed verifier-based TTS method is orthogonal to previous TTS methods, so they can be combined to form a more powerful TTS method.
4. The “Heavy” variants of flagship open-source models, by using the proposed TTS method, achieve on par or better performance than commercial systems, further demonstrating the effectiveness of the proposed TTS method.

### Weaknesses
1. The paper does not explicitly discuss related works, such as previous verifier-based test-time scaling (TTS) methods, making it difficult to assess the contributions and innovations of this work.
2. In the experimental section, the authors randomly selected 100 tasks from the BrowseComp and BrowseComp-zh benchmarks for comparative experiments. They state that “This sampling balances cost efficiency with representativeness”. However, the paper does not analyze the representativeness of the selected subsets. Consequently, it is difficult to determine whether the improvements achieved by the proposed TTS framework on the BrowseComp and BrowseComp-zh benchmarks are representative.
3. The authors propose that the deep search scenario exhibits asymmetric verification, meaning verification is easier than generation. However, in the xbench-DeepSearch benchmark, verification and generation present similar levels of difficulty. The TTS method proposed in this paper does not demonstrate significant improvement over previous TTS methods. This demonstrates that the strength of asymmetric verification in specific datasets or scenarios impacts the applicability of the proposed TTS method. But the paper does not provide a detailed analysis in this aspect.
4. The TTS framework proposed in this paper employs different parameter settings when applied to various benchmarks, but it does not provide an in-depth discussion on parameter selection.
5. The experimental section of the paper claims that the proposed TTS framework yields a superior accuracy-cost trade-off. The figure captions in the experimental section also demonstrate this. However, the comparative results in Table 2 do not reflect this claim.

### Questions
1. Please analyze the differences between the TTS method proposed in the paper and the previous verifier-based TTS methods.
2. Please present the experimental results of the proposed TTS framework on the full dataset of the BrowseComp and BrowseComp-zh benchmarks using flagship open-source models, or provide an analysis demonstrating that the randomly sampled sub-sets used in the paper can represent the full datasets.
3. Please analyze and categorize the strength of asymmetric verification across different datasets, and analyze its impact on parameter selection for the proposed TTS framework (e.g., how to allocate computation when dealing with datasets with different strengths of asymmetric verification). This would provide explicit guidelines when using the TTS framework proposed in the paper.
4. For Table 2, please provide the performance of flagship open-source models using either the previous TTS methods or the baseline TTS methods in the paper when adding the same or similar computational resources as the “heavy version”.
5. Please review the experimental results. Mismatch in experimental results: the Maj@8 accuracy for GLM-4.5 reported in line 312 is 35.7%, whereas the Maj@8 accuracy for GLM-4.5 displayed in the upper right corner of Figure 5 is 30.6%. This discrepancy impacts certain experimental conclusions.
6. Please check typos. For example, "heavey" in line 430.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores test-time compute scaling for deep search agents through both sequential and parallel strategies. Sequential scaling extends generation, while parallel scaling verifies and selects among multiple outputs. Building on the idea of asymmetric verification, where verification is substantially easier than generation. The authors show that allocating limited compute to verification significantly boosts performance. Experiments with several open-source models demonstrate consistent improvements across multiple benchmarks.

### Strengths
The paper is well-motivated, clearly structured, and supported by comprehensive experiments that effectively validate its key claims:

1. This paper demonstrates the inherent asymmetry of deep search tasks, where generation is substantially harder than verification, and shows that allocating compute to the verifier during test-time scaling (TTS) yields a superior accuracy–cost trade-off.

2. It systematically decomposes TTS into a three-dimensional design space of scaling target × scaling strategy × aggregation metric, providing a principled framework for compute-optimal allocation.

3. It achieves "heavy" variants of open-source models through TTS, significantly narrowing the performance gap between open and top proprietary systems—and even surpassing closed-source models in certain benchmarks.

### Weaknesses
1. External Validity Risk from Benchmark Sampling: On BrowseComp and BrowseComp-zh, the authors evaluate only a random subset of 100 samples instead of the full benchmark of approximately 1,200 questions. This limited sample size increases sampling variance and may reduce the robustness of observed performance differences across methods. Although the paper mentions that this decision was made "to reduce computational cost," it does not report confidence intervals or conduct resampling-based robustness analyses to support the reliability of the results.

2. The paper's insight that verification is inherently easier than search is novel; however, its implementation exhibits notable simplifications and potential biases. The verification process still relies on prompt-based judgment by the large language model agent, meaning its reliability ultimately depends on the model's underlying capabilities and biases.

3. In addition, the verifier shares the same base architecture as the search agent (except for Tongyi Deep Research), differing only in prompts and objectives. This design introduces a risk of self-verification bias, as the verifier is not fully independent. Moreover, using GLM-4.5 as the verifier for Tongyi-DeepResearch raises fairness concerns, as this hybrid setup makes direct comparison difficult. A more objective evaluation would benefit from incorporating an external or model-agnostic verifier applicable across different search agents.

### Questions
1. Could the authors clarify how the random subset of 100 samples from BrowseComp and BrowseComp-zh was selected? Was any stratified or difficulty-based sampling strategy used to ensure representativeness?

2. When the same base model architecture is used for both search and verification, how do the authors ensure that the verifier's decision is independent and not overly correlated with the search agent's outputs?

3. Since Tongyi Deep Research uses GLM-4.5 as its verifier, could the authors discuss potential fairness concerns and whether a unified verification framework could address this issue?

### Soundness
3

### Presentation
3

### Contribution
3
