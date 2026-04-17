# Scalable Chain of Thoughts via Elastic Reasoning

- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Large reasoning models (LRMs) have achieved remarkable progress on complex tasks by generating extended chains of thought (CoT). However, their uncontrolled output lengths pose significant challenges for real-world deployment, where inference-time budgets on tokens, latency, or compute are strictly constrained. We propose Elastic Reasoning, a novel framework for scalable chain of thoughts that explicitly separates reasoning into two phases—thinking and solution—with independently allocated budgets. At test time, Elastic Reasoning prioritizes the completeness of solution segments, significantly improving reliability under tight resource constraints. To train models that are robust to truncated thinking, we introduce a lightweight budget-constrained rollout strategy, integrated into GRPO, which teaches the model to reason adaptively when the thinking process is cut short and generalizes effectively to unseen budget constraints without additional training. Empirical results on mathematical (AIME, MATH500) and programming (LiveCodeBench, Codeforces) benchmarks demonstrate that Elastic Reasoning performs robustly under strict budget constraints, while incurring significantly lower training cost than baseline methods. Remarkably, our approach also produces more concise and efficient reasoning even in unconstrained settings. Elastic Reasoning offers a principled and practical solution to the pressing challenge of controllable reasoning at scale. Code is available in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Elastic Reasoning (ER), a framework that makes large reasoning models more efficient and controllable under limited inference budgets. By separating reasoning into thinking and solution phases with independent token budgets and training via budget-constrained reinforcement learning, ER achieves concise yet reliable reasoning—reducing token use by over 30% while maintaining or improving accuracy on math and coding benchmarks.

### Strengths
- Novel framework for budget-aware reasoning – The proposed Elastic Reasoning introduces a clear separation between thinking and solution phases, enabling fine-grained control over inference cost without sacrificing performance.
- Strong empirical efficiency and robustness – The method achieves reduction in token usage while maintaining or even improving accuracy on diverse math and coding benchmarks.
- Excellent generalization under unseen budgets – Models trained with a single budget configuration generalize effectively to new inference constraints, demonstrating strong adaptability and practical scalability.

### Weaknesses
- The method is only tested on strong reasoning models (DeepScaleR, DeepCoder); it’s unclear whether it generalizes to weaker models like Qwen2.5-Math, which lack explicit CoT structure or strong reasoning priors.
- The paper shows that most improvement comes from the solution phase, while increasing the thinking budget (e.g., 2K–3K tokens) brings little additional gain. This suggests that the model may not truly improve its reasoning efficiency—instead, it might rely on memorized solutions rather than performing deeper thinking.
- The evaluation lacks out-of-domain (OOD) reasoning benchmarks such as MMLU or GPQA. As all experiments focus on math and code, it remains unclear whether Elastic Reasoning generalizes to broader reasoning domains or tasks requiring factual and conceptual knowledge.

### Questions
- Can the authors evaluate Elastic Reasoning on out-of-domain reasoning benchmarks (e.g., MMLU, GPQA) to verify whether the method generalizes beyond math and code tasks?
- Can the authors test this by removing the thinking phase entirely (i.e., prompting the model to output only the solution) and reporting the resulting accuracy?
- Could the authors explore asymmetric budgets (e.g., 1.75K for thinking and 0.25K for solution) to test whether the model still maintains performance with a shorter solution phase? Most solutions require far fewer tokens than reasoning.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Elastic Reasoning, a method that enables large reasoning models to achieve scalable and adaptive length control. To further improve solution quality under incomplete reasoning, we introduce a novel training strategy called budget-constrained rollout, which teaches the model to generate high quality answers even with partial CoT trajectories. This method is integrated into GRPO training. This method produces E1-Math-1.5B and E1-Code-14B.

### Strengths
- Very simple method that solves the problem of truncated solution in long reasoning.
- The thinking-solution ablation (4.4.1) is interesting and is good evidence to understand what the proposed training method improves (i.e., generating a solution under an incomplete thinking process).

### Weaknesses
## Major

- Figure 1, 4, 5, 6 are a bit unclear (This may be a minor weakness, but I assigned this as a major weakness for now because it is a crucial experimental setup):
    - What are the points? Do they correspond to the whole AIME questions across different budgets?
    - What is the x-axis? Is it the average tokens used?
    - Could you include the error bars (x- and y-axis)? This is particularly important as there are cases where the Pass@1 and tokens used are not significantly different
- Lack of model variations
    - The authors only experimented with one model variant per setup (one for math, and one for coding).
    - I understand that such experiments can be costly. However, I cannot confidently argue for the generalizability of this finding.
- Lack of further analyses
    - I am curious about the qualitative difference between the thinking process before and after the training (e.g., do the models after fine-tuning commit less backtracking? less circular reasoning?)

## Minor

- Missing details of the GRPO training
    - What is the reward design? It is unclear whether when the training “converges” around 0.5 reward score is a good thing or not.
    - What about the other hyperparameters?
    - Please generally be thorough in describing the experiment

## Additional Suggestions

- Typo Section 4.2 title

### Questions
- Just to confirm that I am understanding the novelty correctly: Two phases of thinking and solution seems to be exactly what reasoning models like DeepSeek is doing, right? Am I missing a certain novelty claim by the authors here? Is it simply that prior works were enforcing the overall tokens count, but now budget-constrained rollout limits the thinking tokens count separately from the solution token?
- Are there qualitative difference between the reasoning patterns of naive truncation vs Elastic Reasoning?
- Appendix B mentioned that the training is conducted for only 30 steps, but that does not seem to match Figure 3.
- Any intuition why the performances of the trained models are still lower than the original model’s?
- Have you tried varying the solution budget instead? or perhaps a control experiment with 0 thinking token and 1K solution tokens may be interesting.

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
This paper propose a method to control the reasoning sequence length of large language models. The key idea is to have a separate "token budget" for the c-o-t and for the answer, so that when the budget is exhausted for the c-o-t, the model can still produce an answer. Training with GRPO uses rollouts produced with that process, which leads the model to learn to deal with limited budget. The method outperforms alternatives that either limit the complete sequence or train the model to generate tokens that "terminate" its reasoning.

### Strengths
Token budget is a key issue for reasoning, the method is very simple and sensical, performance are great.

### Weaknesses
Part 3.2.3 could probably be clarified, in particular the authors should provide a clearer description of the quantities involved and in particular the meaning of the conditioning in the policy, and the differences with a vanilla GRPO procedure.

### Questions
- How is the model informed of the budget during inference? simply because </think> is generated? Hence the model has no information about the budget before it actually exhausts it? Fig 2 gives the impression that additional tokens specify it (red squares)?

- This is not my direct domain of expertise, so are the baselines considered in the experimental part the best available?

### Soundness
3

### Presentation
3

### Contribution
3
