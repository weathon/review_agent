# Efficient LLM Collaboration via Planning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 2, 2

## Abstract
Recently, large language models (LLMs) have demonstrated strong performance, ranging from simple to complex tasks. However, while large proprietary models (e.g., models with over 100B parameters) achieve remarkable results across diverse tasks, they are often accessible through costly APIs, making frequent use too costly for many applications. In contrast, small open-source models (e.g., models with fewer than 3B parameters) are freely available and easy to deploy locally, but their performance on complex tasks remains limited. This trade-off raises a natural question: how can small and large models efficiently collaborate to combine their complementary strengths? To bridge this trade-off, we propose COPE, a test-time collaboration framework. A planner model first generates a plan, a high-level abstraction of the task, and this plan serves as a lightweight intermediate that guides a downstream executor model. Small and large models take turns acting as planner and executor, exchanging plans in a multi-stage cascade to collaboratively solve tasks. Through comprehensive experiments on benchmarks spanning mathematical reasoning, code generation, open-ended tasks, and agent tasks, we demonstrate that COPE achieves performance comparable to large proprietary models, while drastically reducing the inference API cost. These results highlight planning as an effective prior for cost-efficient inference.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes COPE, a test-time collaboration scheme between a small model and a large model. A planner writes a short plan, an executor answers conditioned on that plan, and the system defers to the larger model when a confidence rule is not met. The claim is comparable quality to a strong single model with lower API cost across math, code, open-ended QA, and an agent benchmark.

### Strengths
- paper is well written and structured.
- clear problem framing: save large-model calls by escalating only when needed.
- simple interface: plan as a short guidance that conditions the executor.
- broad evaluation across several task families with cost reporting.
- sensible ablations on sampling, thresholds, and truncating later stages.
- paper is easy to reproduce as most resources are available in appendix.

### Weaknesses
- Novelty is arguably modest. COPE looks like a heuristic cascade or router with task-specific confidence checks -- one question from me: "how does COPE handle those killer questions that would even need multiple rounds of large model involvement?". Mixture-of-Experts or learned routing can address the same goal in a more principled way. The paper does not compare to MoE or to strong learned routers, so the contribution reads (arguably) incremental.
- Missing core metric. There is no clear “defer rate” analysis. I need to see how many items are solved by small-only, how many are escalated, and how accuracy and cost move conditional on the stage. Without this, the cost-savings story is incomplete.
- Confidence rules are heterogeneous across tasks (vote for math, pass rate for code, perplexity or LLM-as-judge for open-ended). This makes cross-task comparisons hard and risks bias in open-ended settings.
- Planner overhead is under-analyzed. Plans add tokens and extra calls. The paper should separate plan-token cost from answer-token cost and report the win only at matched accuracy.

Minor comments:
- Robustness is unclear. Results may depend on current pricing, specific model pairs, and judge choice. There is little stress-testing for model updates or alternative judges.
- Insights on “who should plan for whom” are unsurprising. The page-3 observations mirror common intuition and existing engineering writeups (for example, multi-agent system notes by anthropic blogs). Presenting them as headline findings feels overstated.

### Questions
1. For each benchmark, what percentage of items are answered at Stage 1, Stage 2, and Stage 3? What is the accuracy conditional on each stage? What is the average cost per item at each stage, split into plan tokens and answer tokens?
2. What exactly is new relative to prior cascades and learned routing? How does COPE compare to a learned router that chooses small versus large using validation signals under the same budgets?
3. For open-ended tasks, do multiple independent judges or a small human study replicate the reported wins?
4. Relative to a no-plan executor at the same stage, how much does adding a plan improve accuracy?
5. Will you frame COPE as a practical cascade router rather than a new reasoning framework?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposed a test-time framework to enable collaborations between small and large LMS to solve various tasks. The collaboration is done through a two-stage planning and execution process that is first attempted by a small model and then can be escalated to a larger model if the small model is not confident about the final answer. Authors have tested their approach over on mathematical, coding, open-ended and agentic tasks and showed better or on par performance to large closed source models while reducing the API costs of only relying on large models.

### Strengths
- the paper tested their method on a wide range of tasks and benchmarks including mathematical, coding, open ended and agentic tasks.
- the paper is generally well written

### Weaknesses
- Some of the papers claims are overstated and rebranding of existing all-known approaches. For example, the use of "planning" in the paper is just as similar to the known "CoT (chain-of-thoughts)" and cannot be considered as the contribution of the paper. The authors need to provide a clear explanation on the novelty of their work compared to the prior works. 
- Relatedly, a more fair baseline to compare their method with is the vanilla Large/Small including CoT. That is prompting the model to first do some brief reasoning before attempting the problem
- Overall, the contribution of the paper seems more incremental

### Questions
1. How is the plan different from the commonly known CoT? are the authors trying to rebrand that as plan and planner instead of CoT and reasoner? While I agree that the current landscape of reasoning is more on long chain of thoughts but the earlier studies introducing this approach focused on short CoTs which to me is similar in nature to what you call plan=brief guideline. (as shown in Figure 1)
2. Some qualitative analysis/comparison between your plan and existing CoTs would be helpful to understand the difference (if there's any). 
3. Suggestion: Table can benefits from more self-explanatory captions to guide the reader. For example, better baseline naming, explaining how cost is computed etc.
4. for a fair comparison between your method and baselines you should consider including CoTs (or your so called Plans) as inputs to the model but generate those using the answer generation model itself as opposed to having a collaboration between small/large models. The plan/Cot generation is not a contribution of this paper and thus be leveraged by all other baselines to better show the impact of collaboration between small and large lm.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The main contribution of this paper is a multi-stage LLM inference pipeline that relies on plans: i) in the first stage, a small model generates a plan and tries to solve the problem; ii) if not satisfied, in the second stage, a large model generates a plan and then passes the plan to the small model to solve; iii) finally, if not satisfied, in the third stage, the large model executes the plan it generated in stage 2. Experiments on a couple of small models and a couple of large models and in comparison with several baselines (Cascade, ABC, ...) show improvements in accuracy with a decrease in cost. The cost savings are attributed to "early exits" by a model solving a problem after stage 1 or after stage 2 (avoiding the expense of inference on a large model as in stage 3).

### Strengths
+ I like the conceptual simplicity of the paper: it is presented for a small/large setup, but could easily be extended to n models of increasing strength.

+ The idea of planning at LLM inference is a subtle distinction from traditional C-o-T (and related approaches) wherein the model explicitly generates a plan before attacking the problem. 

+ The experimental study shows consistent improvements in accuracy while reducing costs. I also appreciate the two results (Table 6 and 7) that show off the method on open-ended and agent tasks.

### Weaknesses
- The approach is "system-ized" in that the core idea of using plans to yield better results is hidden through the three stages. I would be interested to see how the results play out for each of the settings: small planner, small executor; large planner, small executor; large planner, large executor. 

- The observations (Section 3) make good sense, and I appreciate that these are framed as just "observations", but it would be good to see more empirical support, e.g., considering other planners, executors, and ablations on the "plans aligned with its capacity". 

- To this last point, a simpler plan is used for smaller models vs a more systematic plan for larger models: I wonder what the impact of plan prompt is. These are hand-crafted as best I can tell (from the Appendix). I could imagine many levels of detail for these planner prompts that could impact the quality of the results.

### Questions
Can you show the results for the three settings in isolation? (without passing from one stage to the next?).

Did you observe any impact of varying the planning prompt design?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a three-stage collaboration framework for efficient large language model (LLM) usage, where smaller models handle simpler tasks and larger models are invoked based on confidence thresholds. The approach aims to balance accuracy and cost efficiency, and experiments on reasoning tasks demonstrate comparable performance and lower cost compared to existing methods.

### Strengths
- The paper is clearly written and easy to follow.
- The proposed approach is intuitive and straightforward.
- Code is provided for review.

### Weaknesses
- The proposed method is largely engineering-oriented, with the three-stage design appearing intuitive and lacking algorithmic or learning-based innovation.

- The proposed plan–then–execute paradigm raises concerns regarding its effectiveness. In our experience, prompting a model to first produce a plan and then execute it often yields worse results than directly applying CoT prompting. This limitation stems from the fact that the model must generate a plan based solely on the query, without access to any intermediate reasoning results, which may hinder its ability to form accurate or adaptive strategies.

- While using confidence as a trigger is empirically effective, it does not fully address the issue of overconfidence.

- Table 3, 4: The performance gains in both accuracy and cost over existing baselines (such as Cascade and ABC) are relatively limited.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes COPE, a test-time collaboration framework that enables small and large language models (LLMs) to work together more efficiently through planning. Instead of immediately generating an answer, a model first produces a high-level plan (e.g., goal or guideline), which then guides another model in solving the task. COPE proceeds in three escalating stages: 1) small model plans + executes; 2) large model plans + small model executes; 3) large model plans + executes. Besides, a confidence measure (e.g., majority vote) determines whether to accept the output or escalate to the next stage. This minimizes expensive large-model usage while preserving performance. Experiments across math reasoning, code generation, open-ended tasks, and agent tasks show the advantages of COPE.

### Strengths
1. This paper investigates the collaboration of small and large models to reduce the inference costs of LLM, which is a timely and important topic. The proposed method, COPE, reduces reliance on expensive LLM calls by: Letting small models solve easy problems Escalating to large models only when needed Example: On MATH-500, cost drops ~45–50% versus GPT-4o.
 
2. COPE introduces an explicit planning stage where models write high-level goals/guidelines. This plan provides structured task abstraction and helps small executors perform better, stabilizing multi-stage reasoning. This also enables cross-model synergy not captured by prior cascades.
 
3. Unlike baselines that require few-shot data and training of a scoring function, COPE needs no extra training, no supervised planning data and only needs inference-time prompting.

### Weaknesses
1. The major concern of this work is its innovation. There are many previous papers which decomposed the task or generated a high-level plan by a large model and then executed sub-tasks by a small model, e.g. [1-2]. It seems that this multi-level planning framework is only an extension of previous works.
 
[1] Kwon, M., Kim, Y. and Kim, Y.J., 2025, May. Fast and accurate task planning using neuro-symbolic language models and multi-level goal decomposition. In 2025 IEEE International Conference on Robotics and Automation (ICRA) (pp. 16195-16201). IEEE.
 
[2] Li, Z., Chang, Y., Yu, G. and Le, X., 2025. Hiplan: Hierarchical planning for llm-based agents with adaptive global-local guidance. arXiv preprint arXiv:2508.19076.
 
2. Authors claim that small language models are free to access. However, many of them are not. Because of their size, e.g., llama-3B needs more than 8GB VRAM, these models cannot work on small devices and people still need to pay if these models are installed and maintained in a data center.
 
3. Writing needs to be improved. There are many confusing statements in this paper. For example, in the introduction section, the third paragraph and last second paragraph are contradicting each other. The working mechanism of the previous method stated in the third paragraph is presented again in the last second paragraph as the proposed method, which is very confusing.
 
4. Authors evaluate the proposed framework in tasks across many areas. However, its capacity in each specific area is not evaluated deeply and comprehensively enough. For example, in agent tasks, only AFLWorld is used and other popular benchmarks, such as webshop, are not included. We suggest authors focus on one area, such as math reasoning and conduct more comprehensive evaluations in a single area.

### Questions
1. Authors propose to use majority voting over self-generated answers to calculate the confidence score of LLM. However, it is widely known that LLM could be overconfident on wrong answers. How to address the situation that LLM is confident but wrong?
 
2. Authors propose to use a plan generated by a large model to help the task execution. However, it is widely known that the planning capability of modern LLM is still limited [1]. Specifically, the plan, even generated by GPT 4o, may violate some constraints in the domain. If the generated plan has mistakes, it could mislead the downstream task execution. How to resolve this issue in the proposed framework?
 
[1] Cao, P., Men, T., Liu, W., Zhang, J., Li, X., Lin, X., Sui, D., Cao, Y., Liu, K. and Zhao, J., 2025. Large language models for planning: A comprehensive and systematic survey. arXiv preprint arXiv:2505.19683.
 
3. Authors report that the success rate of the proposed method in ALFWorld is only around 40%. However, only vanilla ReAct can achieve nearly 65% with GPT-4o. If the proposed executor is augmented by a generated plan, how is its performance worse than a vanilla ReAct?

### Soundness
2

### Presentation
2

### Contribution
2
