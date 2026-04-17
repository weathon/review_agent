# TRIM: Hybrid Inference via Targeted Stepwise Routing in Multi-Step Reasoning Tasks

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Multi-step reasoning tasks like mathematical problem solving are vulnerable to cascading failures, where a single incorrect step leads to complete solution breakdown. Current LLM routing methods assign entire queries to one model, treating all reasoning steps as equal. We propose TRIM (Targeted routing in multi-step reasoning tasks), which routes only critical steps$\unicode{x2013}$those likely to derail the solution$\unicode{x2013}$to larger models while letting smaller models handle routine continuations. Our key insight is that targeted step-level interventions can fundamentally transform inference efficiency by confining expensive calls to precisely those steps where stronger models prevent cascading errors. TRIM operates at the step-level: it uses process reward models to identify erroneous steps and makes routing decisions based on step-level uncertainty and budget constraints. We develop several routing strategies within TRIM, ranging from a simple threshold-based policy to more expressive policies that reason about long-horizon accuracy-cost trade-offs and uncertainty in step-level correctness estimates. On MATH-500, even the simplest thresholding strategy surpasses prior routing methods with 5x higher cost efficiency, while more advanced policies match the strong, expensive model's performance using 80% fewer expensive model tokens. On harder benchmarks such as AIME, TRIM achieves up to 6x higher cost efficiency. All methods generalize effectively across math reasoning tasks, demonstrating that step-level difficulty represents fundamental characteristics of reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes TRIM, a hybrid inference method that routes reasoning steps between a small and a large LLM. The key idea is to escalate only critical reasoning steps to the stronger model, improving efficiency while maintaining accuracy. The paper includes several routing strategies, from a simple threshold-based approach to more advanced RL and POMDP-based ones and demonstrates improvement over query level baseline results on multiple math reasoning benchmarks.

### Strengths
The overall presentation is good, and the paper is mostly easy to read and understand. The concept of step-level routing appears practical, bridging the gap between coarse query-level routing and fine-grained token-level control. The experimental setup seems reasonable, and the reward modeling and reinforcement learning training are well-motivated.

### Weaknesses
Some parts of the paper are confusing or underspecified, especially the step definition and regeneration process. The cost metric ignores potential latency from frequent model switching, which could matter in deployment. A few experimental details (like baseline performance numbers) seem inconsistent or not clearly justified. Clarifying these points would make the technical contribution much easier to trust and replicate. See questions section for more details. I am open to increase my rating if the issues raised are adequately addressed.

### Questions
1- Section 4 feels ambiguous. While Figure 3 suggests that routing and decisions happen step-by-step, the notation y_s = Ms(y1:t−1) makes it look like the strong model regenerates the whole prefix. Which interpretation is correct? The notation and explanation should be clarified.

2- What exactly is a “step”? Is it defined as a sentence, a reasoning line, or a fixed number of tokens? How do you prompt or constrain the model to generate only one step at a time? Is this step size consistent across different datasets or tasks?

3- In TRIM-Seq, how is the step-level correctness score computed? Is it purely from the PRM output or based on an external reference?

4- Since cost is measured only by the number of tokens from the expensive model, does this ignore switching overhead between models? Routing back and forth should add latency and communication cost, which could be nontrivial in real systems.

5- The reported baseline performance for Claude 3.7 seems off, while this model reportedly has ~96.2 % accuracy on MATH-500,  Figure 6 shows it closer to 80%. Can you clarify this discrepancy or explain what experimental setup causes this gap?

### Soundness
2

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
4

### Summary
The paper presents TRIM, a hybrid inference framework that uses a process reward model (PRM) to decide when to route reasoning steps from a smaller model to a larger one. The approach aims to balance inference cost and reasoning accuracy. It is primarily evaluated on mathematical reasoning tasks such as AIME using Qwen2.5-Math-PRM-7B and Claude 3.7 Sonnet. The authors introduce a Cost–Performance Threshold (CPT) metric and report improved cost–performance trade-offs compared with several baseline methods.

### Strengths
* **Clear motivation and relevance:** The paper tackles a practical problem of reducing inference cost while maintaining reasoning accuracy.
* **Reasonably structured framework**: The framework is generally well organized, though its design largely follows existing PRM-based routing paradigms.
* **Clarity:** The paper is well organized and easy to follow, with clear figures and explanations.

### Weaknesses
* **Limited novelty over existing PRM-based routing methods:** The paper does not clearly articulate how TRIM differs conceptually from prior stepwise routing approaches using process reward models (PRMs).
* **Unclear reliability of PRM supervision for AIME problems:** The use of Qwen2.5-Math-PRM-7B raises concerns about label reliability on AIME, where reasoning traces are substantially longer and harder to evaluate.
* **Model family mismatch and API issues:** Mixing models from different families (e.g., Qwen and Claude) may introduce inconsistencies in formatting and reasoning styles. In addition, Claude 3.7 Sonnet’s chat-completion API makes partial-trace continuation non-trivial.
* **Insufficient baseline coverage:** Several key comparisons are missing, such as non-routing anchors (Always-Small, Always-Large), query-level routing, and adaptive compute methods.
* **Code and data unavailable for review:** The absence of released code or data prevents verification and reproducibility of the reported results.

### Questions
* There is already a considerable body of work on stepwise routing with PRMs [1–3]. What is the fundamental difference between TRIM and these existing approaches?
* How effective is Qwen2.5-Math-PRM-7B as a PRM for AIME problems? Compared with the datasets reported in the official documentation [4], AIME problems are substantially more difficult, and the reasoning traces generated by LLMs are usually much longer. It is therefore more challenging to evaluate the correctness of intermediate steps.
* The Cost–Performance Threshold (CPT) metric is not commonly used. In contrast, budgeted accuracy, which compares methods under the same compute or token budget, is a standard and decision-useful metric for evaluating efficient reasoning or routing. Could the authors also report results under this metric for better comparability?
* The baseline coverage should be expanded. In addition to stepwise routing methods, please include non-routing anchors such as Always-Small-Model and Always-Large-Model, as well as query-level routing [5,6] and step-/token-level adaptive compute approaches [1–3].
* Line 420: The paper uses models from different families as the base (e.g., Qwen vs. Claude), which may have distinct formatting styles and output distributions. How do the authors handle these inconsistencies? Moreover, since Claude 3.7 Sonnet relies on a chat-completion API schema, continuing reasoning from a partial trace can differ from a true completion mode. Could the authors explain how this issue is addressed or justify this model choice?
* Figures 6 and 7: Should the x-axis represent the number of tokens from the expensive model?
* How generalizable is the proposed approach? Can the authors test TRIM with different combinations of cheap and expensive models, and on additional domains such as scientific reasoning or code to demonstrate broader applicability?
* Can the authors provide the code and data for review?

## Reference:
[1] Liao, Baohao, et al. "Reward-Guided Speculative Decoding for Efficient LLM Reasoning." Forty-second International Conference on Machine Learning.

[2] Pan, Rui, et al. "Specreason: Fast and accurate inference-time compute via speculative reasoning." arXiv preprint arXiv:2504.07891 (2025).

[3] Liu, Yuliang, et al. "AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence." Forty-second International Conference on Machine Learning.

[4] Zhang, Zhenru, et al. "The lessons of developing process reward models in mathematical reasoning." arXiv preprint arXiv:2501.07301 (2025).

[5] Ding, Dujian, et al. "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing." The Twelfth International Conference on Learning Representations.

[6] Guha, Neel, et al. "Smoothie: Label free language model routing." Advances in Neural Information Processing Systems 37 (2024): 127645-127672.

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
3

### Summary
This paper proposes a novel method that improves the inference efficiency of large language models by performing routing decisions at the step level. The authors formalize this process as a sequential decision problem and design several routing strategies, including a simple threshold-based policy (TRIM-Thr), two reinforcement learning–based policies (TRIM-Seq and TRIM-Agg) that balance accuracy and cost, and a POMDP-based policy (TRIM-POMDP) to handle uncertainty in step correctness estimation. Experiments on multiple mathematical reasoning benchmarks demonstrate that this approach can achieve performance comparable to strong models while significantly reducing computational cost.

### Strengths
1. The paper is clearly structured and well written, with intuitive figures and tables that effectively illustrate the key ideas and experimental results.
2. The proposed method demonstrates strong innovation by moving beyond traditional query-level routing, offering a substantial conceptual advance in improving multi-step reasoning efficiency, and showing significant empirical effectiveness.
3. The authors formalize the routing process as a sequential decision problem and propose multiple complementary strategies that balance methodological simplicity with theoretical depth.

### Weaknesses
1. The approach heavily relies on the Process Reward Model; inaccuracies or poor calibration in PRM estimation may negatively affect routing stability and overall performance.
2. Experiments focus primarily on mathematical reasoning tasks, and the generalization of TRIM to other multi-step reasoning domains (e.g., code generation or scientific reasoning) remains untested.
3. Stepwise generation introduces additional inference latency; although token cost is reduced, the paper does not provide a detailed analysis of runtime efficiency or latency trade-offs.

### Questions
See weakness.

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
This paper proposes TRIM (Targeted Routing in Multi-step reasoning tasks), a novel hybrid inference strategy designed to improve the cost-efficiency of multi-step reasoning. The central problem it addresses is that conventional "query-level" routing (assigning an entire task to either a cheap or an expensive LLM) is inefficient. Relying only on cheap models leads to cascading errors, while relying only on expensive models is cost-prohibitive.

TRIM's core idea is to perform **step-wise routing**. A cheap model generates each reasoning step by default. After each step, a router, guided by a Process Reward Model (PRM), decides whether to accept the cheap step or to regenerate *only that specific step* with an expensive model before handing control back to the cheap model. This "targeted intervention" aims to prevent cascading failures at critical steps while saving costs on routine ones.

The authors propose and evaluate four distinct routing strategies:
1.  **TRIM-Thr:** A simple threshold policy (regenerate if PRM score < *k*).
2.  **TRIM-Seq:** An RL-trained policy (using full step history).
3.  **TRIM-Agg:** A simpler RL-trained policy (using aggregated step statistics).
4.  **TRIM-POMDP:** A sophisticated policy that models the true step correctness as a hidden state and the PRM score as a noisy observation.

On mathematical reasoning benchmarks, the simple TRIM-Thr policy is shown to be 6.51x more cost-efficient than contemporary query-level routing. The more advanced RL and POMDP policies can match the performance of the expensive-only model while reducing the number of expensive tokens generated by 80%.

### Strengths
* **Novelty:** The core concept of step-wise routing, as opposed to query-level routing, is a significant and novel contribution. It directly targets the "cascading failure" weakness of multi-step reasoning.
* **Impressive Results:** The efficiency gains are substantial. Achieving an 80% reduction in expensive token usage while matching the strong model's accuracy is a highly compelling result. The 6.51x cost-efficiency improvement from the *simplest* policy (TRIM-Thr) over SOTA query-level routing is also very strong.
* **Methodological Thoroughness:** The paper explores four different router implementations, from a simple heuristic to a complex POMDP. This shows a deep understanding of the problem, from a practical, easy-to-implement solution to a more theoretically-grounded one that handles observational uncertainty (PRM noise).
* **Generalizability:** The cross-dataset generalization experiments are a key strength, validating that the model is learning fundamental properties of reasoning difficulty rather than just overfitting to a specific dataset's quirks.

### Weaknesses
* **Incomplete Cost Model (Latency):** The paper's cost model is defined *only* as "the number of tokens generated by the expensive model." This model completely ignores the very significant latency and compute costs incurred at *every single step* of the generation. The loop is: (1) $M_w$ generates a step, (2) generation pauses, (3) the 7B-parameter PRM runs, (4) the router policy runs (for TRIM-POMDP, this involves *re-solving* the policy), (5) $M_s$ *maybe* runs. This sequential, stop-and-go process introduces a massive amount of latency per step, which is not accounted for. A simple, streamed generation from the expensive model $M_s$ might be much faster in wall-clock time, even if it uses more "expensive tokens."
* **The "Regenerate" Action Definition:** The "regenerate" action is defined as $M_s$ regenerating the current step, after which control is *immediately* returned to the cheap $M_w$ for the *next* step. This seems suboptimal. If a step was so critical that $M_w$ failed and $M_s$ had to intervene, why would you trust $M_w$ with the very next step? This could lead to an immediate re-failure. A more intuitive action would be to let $M_s$ take over for the remainder of the generation. This design choice feels overly optimized for the paper's cost metric, not for maximum accuracy.
* **Complexity vs. Performance:** The TRIM-POMDP approach is by far the most complex, requiring ground-truth step-level annotations (from ProcessBench) to learn the observation function and a POMDP solver. However, in the results (e.g., Figure 6), its performance-cost curve is almost identical to that of the much simpler `TRIM-Agg` policy (a simple MLP trained with PPO on aggregated stats). The paper does not adequately justify why this massive increase in complexity is worthwhile for such a marginal gain.
* **Dependency on the PRM:** The entire system is critically dependent on the quality of the Qwen2.5-Math-PRM-7B model. While the POMDP policy *models* this uncertainty, the simpler and highly effective TRIM-Thr and TRIM-Agg policies *trust* the PRM scores. The paper doesn't analyze how the performance of these simpler policies degrades if a less-accurate, noisier, or miscalibrated PRM is used.

### Questions
1.  **Latency Cost:** Your cost model ignores the significant per-step latency of running the PRM and the router policy. How does the total wall-clock time of a TRIM-generated solution (with its N sequential stop-and-think steps) compare to a single, uninterrupted generation from the expensive model $M_s$?
2.  **"Regenerate" Action:** Why did you choose to return control to the weak model $M_w$ *immediately* after an expensive $M_s$ intervention? Have you experimented with an alternative "regenerate" action where $M_s$ takes over for all *remaining* steps, and how does that compare on the performance-cost curve?
3.  **PRM Sensitivity:** How sensitive are the simpler (and more practical) TRIM-Thr and TRIM-Agg policies to the quality of the Process Reward Model? How much does their performance drop if a smaller, faster, but less accurate PRM is used?
4.  **Justifying POMDP Complexity:** The results for TRIM-Agg and TRIM-POMDP in Figure 6 appear nearly identical. Given the extreme complexity of the POMDP (requiring annotated data to train the observation function, plus an online solver), what is the practical justification for using it over the much simpler RL-based TRIM-Agg policy?

### Soundness
3

### Presentation
3

### Contribution
3
