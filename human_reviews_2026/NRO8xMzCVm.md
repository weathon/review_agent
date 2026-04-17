# EAGER: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
With the rise of reasoning language models and test-time scaling methods as a paradigm for improving model performance, substantial computation is often required to generate multiple candidate sequences from the same prompt. This enables exploration of different reasoning paths toward the correct solution, however, allocates the same compute budget for each prompt. Grounded on the assumption that different prompts carry different degrees of complexity, and thus different computation needs, we propose EAGer, a training-free generation method that leverages model uncertainty through token-wise entropy distribution to reduce redundant computation and concurrently improve overall performance.
EAGer allows branching to multiple reasoning paths only in the presence of high-entropy tokens, and then reallocates the saved compute budget to the instances where exploration of alternative paths is most needed.
We find that across multiple open-source models on complex reasoning benchmarks such as AIME 2025, EAGer can reallocate the budget without accessing target labels, achieving the best efficiency-performance trade-off in terms of reasoning length and Pass@k. When target labels are accessible, EAGer generates up to 65% fewer tokens (hence saving compute) and achieves up to 37% improvement in Pass@k compared to the Full Parallel Sampling.
Our results show that EAGer consistently maximizes the efficiency-performance trade-off by enabling dynamic control over computation expenditure.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
An entropy-based branching technique in generation is proposed and evaluated over various datasets. It can achieve better trade-offs by improving the efficacy of test-time scaling. The sampling is also adaptive, adding extra effort to more challenging questions.

### Strengths
The proposed entropy-aware generation, where branches depend on entropy, is intuitive and simple. 

Experiments cover different datasets in different areas and show the improvement for better trade-offs for computation-accuracy. Also, experiments show that budgets are used adaptively, which is good in practice.

### Weaknesses
Figure 1 is not very intuitive and not very helpful to understand the method, despite the method itself is not complicated.

EAGER results need ground truth, so it is like a theoretical upper bound. So, for clarity, the tables should make this clearer, like adding footnotes to EAGER. EAGER-init and +budget should be the practical ones, useful for practice instead.

Focusing on two mathematics datasets in Table 2 rather than the same datasets as in Table 1 is not clearly motivated. Why are these two separate experiments instead of a comprehensive experiment comparing baseline, EAGER-init, +budget, and EAGER? Also, why is Figure 3 the union of the datasets of Tables 1 and 2 plus AIME 2024? It would be clearer if different tables/figures were on consistent datasets and represented different perspectives for holistic comparison; in such a case, the conveyed information would overlap.

Minor: More precision in Table 2, especially for the number of tokens, is recommended.

### Questions
Is it possible that the branching points concentrate on the early part of the generation in the case of saturation, so that the branches happen too early? In this case, the latter places where branches are also useful may be ignored. In the case where the challenging question has multiple parts and your method saturates, will diversity/correctness be biased to the earlier parts?

Do people need to tune M for each dataset / how generalizable is it? Is it just a trade-off, or are there still optimal ranges?

Can you explain the meaning of the line in Figure 2? It does not look like the trend line.

Is it possible to further improve branching by distinguishing useful entropy (where randomness is more essential/relevant to the problem) vs empty entropy?

Is it possible to make the entropy threshold soft? Is there an optimal strategy?

How is “saves up to 80%” estimated?

### Soundness
2

### Presentation
1

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
The paper presents EAGER, a training-free decoding method that adapts compute usage to prompt difficulty using token-wise entropy. It branches reasoning only when uncertainty is high, cutting redundant computation. On reasoning benchmarks like AIME 2025, EAGER reduces token generation by up to 65% while improving accuracy by 27%, achieving a better efficiency–performance trade-off.

### Strengths
1. The paper is well-written and easy to follow.
2. The proposed approach is simple and effective. Especially, it always outperforms the full parallel baseline, showcasing that it improves the overall reasoning performance as well as improving the efficiency.
3. The authors conducted extensive experiments using models from diverse families, across diverse scales.

### Weaknesses
1. Limited baselines. While comparison against full parallel baseline is valuable, there is a large body of existing works on training-free approaches for budget control [1]. Comparing against at least some of such existing work would be helpful to understand the relative position of the proposed approach in the big picture.

2. Limited efficiency analysis. While the main efficiency metric is the total number of decoded tokens, this may not directly translate to peak memory usage or inference latency, especially in batch generation scenarios where the sequences are generated in parallel. Further analysis regarding these two aspects would strengthen the claims regarding efficiency.

3. Minor formatting issues. Bold is missing for some entries in Table 1. Also, the method name is written as EAGER in the title, but EAGer in the rest of the text.

[1] Sui et. al., Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models, TMLR 2025

### Questions
Could you provide some additional analysis regarding additional baselines and efficiency measurements?

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
EAGER monitors token-level entropy during decoding and branches only at high-entropy steps (EAGER-init), reusing the prefix so low-entropy spans are not re-generated. The saved budget is then reallocated to saturating/hard prompts up to a global cap M. The paper reports a negative correlation between peak entropy and pass rate and shows up to 65% fewer tokens with up to +27% Pass@1 vs FULL-PARALLEL across math/science/code tasks and 3B-20B models. Key knobs include the entropy threshold, top-2 temporal branching, a no-branch window, and simple reallocation rules.

### Strengths
- Simple, general heuristic: Thresholding token-level entropy to trigger branching is conceptually clean and model-agnostic.
- Empirical signal: The correlation between peak entropy and failure provides a concrete control signal.
- Practicality / deployability: Prefix reuse (KV-cache effectiveness) makes the method easy to integrate into existing parallel sampling paths and likely efficient in real serving stacks.
- Quality–efficiency gains: Reports simultaneous token savings and accuracy lifts across models and budgets.

### Weaknesses
- While superiority over FULL-PARALLEL is clear, a head-to-head comparison under equal budget with other adaptive policies—especially DeepConf—would be highly informative. Both methods aim to allocate compute dynamically, yet take opposite strategies: DeepConf terminates low-confidence continuations, whereas EAGER branches on high-entropy points. It would be interesting to see which yields better token reduction, accuracy, and wall-clock efficiency under the same budget. I expect EAGER to have an advantage in runtime due to KV-cache reuse, but empirical evidence would clarify whether “cut-and-restart” or “branch-and-reuse” is ultimately more effective.

- Sensitivity/assumptions: Outcomes may hinge on θ, temperature M, and the top-2 branching choice; difficulty-binned analyses (e.g., overthinking/U-shape on easy items) are missing.

### Questions
- Can you report same-budget curves vs DeepConf under (a) equal tokens, (b) equal wall-clock (incl. TTFT), (c) FLOP proxies?
- What is the per-token overhead of entropy checks and branch management in a standard serving stack, and how does KV-cache sharing affect memory/throughput?
- How sensitive are results to θ, M, temperature/top-p? Any auto-tuning or per-prompt adaptation?
- Does top-2 branching trade off diversity vs sampling two tokens independently? Any study with top-k or stochastic branching?
- Can entropy be approximated (e.g., with a tiny probe or confidence surrogate) when logits are unavailable, and how does that impact gains?

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
4

### Summary
The paper introduces a training-free decoding scheme for test-time scaling called EAGER. So instead of generating M parallel CoT traces per prompt no matter what, it watches token-level entropy and branches only at high-entropy steps, reusing the shared prefix; easy prompts consume 1–2 traces, hard ones branch more. The unused “sequence budget” from easy prompts is then reallocated to prompts that failed Pass@1, so total compute stays within the original M×|D| but gets spent where it matters. Across multiple benchmarks the method showed to be using up to 65% fewer tokens and still beat or match full-parallel sampling, sometimes by as much as +27% Pass@1 on small models.

### Strengths
- Clear motivation and definition of the problem
- Training free method which makes it scalable and works across a good range of model sizes.
- Good performance across benchmarks

### Weaknesses
- Needs per-token entropy + tuned threshold θ; θ is model/task-dependent, so not totally plug-and-play. 
- Reallocation assumes you can tell which prompts failed ( access to answers/verifier); without that, they fall back to a heuristic. 
- Entropy is shown to correlate with difficulty on these reasoning tasks, but generality beyond that is argued more than proved in the paper.

### Questions
Please refer to weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
