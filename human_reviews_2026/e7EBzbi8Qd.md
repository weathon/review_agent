# SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 6

## Abstract
While reasoning models demonstrate exceptional performance on complex tasks, they often exhibit tendencies of overthinking on simple problems. This phenomenon not only leads to excessive computational resource consumption but also significantly degrades user experience. To address this challenge, we propose SelfBudgeter - a novel user-friendly adaptive controllable reasoning framework that incorporates a budget estimation mechanism prior to reasoning. The framework adopts a dual-phase training paradigm: during the cold-start phase, the model learns to predict token budgets before executing reasoning in a standardized format; in the reinforcement learning phase, the model is trained to autonomously plan budgets based on problem difficulty and strictly adhere to them when generating responses. Since the model outputs budget estimates at the initial stage, users can immediately anticipate waiting duration, enabling flexible decisions on whether to interrupt or continue the generation process. Notably, our method supports manual control of reasoning length through pre-filled budget fields. Experimental results demonstrate that SelfBudgeter can dynamically allocate budgets according to problem complexity, yielding an average response length compression of 61% for the 1.5B model on GSM8K, MATH500, and AIME2025, and 48% for the 7B model, while maintaining nearly undiminished accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Self-Budgeter, a training-free, adaptive test-time framework that enhances reasoning efficiency in large language models. It enables models to self-regulate token usage by dynamically deciding when to stop generating reasoning steps based on internal uncertainty indicators such as entropy or logit variance. This design reduces redundant computation for simple queries while allowing more reasoning for complex ones, effectively lowering inference costs without compromising accuracy. Experiments across multiple benchmarks demonstrate consistent efficiency gains with comparable or improved performance.

### Strengths
- The paper tackles an important and practical problem in LLM reasoning: how to balance accuracy and efficiency by adapting token usage during inference, a challenge that is highly relevant for real-world deployment.

- The evaluation covers several reasoning benchmarks and includes detailed discussions and explanations of the results that help interpret the observed behavior.

- The writing is clear and well-organized, making the paper easy to follow and its contributions well-articulated.

### Weaknesses
- The novelty is limited. The idea of adaptively controlling reasoning length or token budget has been explored in prior works (e.g., early termination, entropy-based stopping). This paper mainly repackages existing ideas under a new name without offering substantial conceptual advancement. What is the key new insight beyond previous adaptive reasoning methods?

- The method lacks clear theoretical grounding. While it claims to use uncertainty for token budgeting, the approach remains heuristic, with no principled analysis showing why or when the chosen uncertainty metrics (entropy, logit variance) reliably correlate with reasoning sufficiency.

- The evaluation is narrow, focusing mainly on a single-domain, math-style reasoning dataset. This limits evidence of generalization to other reasoning types. Also, the approach is not compared to prior works that explicitly optimize reasoning token budgets, such as early termination or adaptive reasoning frameworks. Would the same approach hold for non-mathematical reasoning tasks, and how does it compare quantitatively to prior budget-optimization methods?


- The paper does not account for the computational overhead of monitoring uncertainty and making token-level stopping decisions. What is the actual end-to-end wall-clock speedup once overhead is included?



- The evaluation uses only one model family (DeepSeek-R1-Distill-Qwen) and relatively small models. This restricts the strength of the claims. How would the approach scale to larger LLMs or different architectures?



- The method’s robustness to incorrect token budget predictions is not analyzed. If the model mispredicts the required reasoning length, accuracy may degrade. How sensitive is performance to such mispredictions, and is there a fallback mechanism?



- The rationale behind the chosen reward function is unclear. The paper should explain how it aligns with the overall optimization goal and whether other formulations were tested.

### Questions
See the weaknesses section.

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
3

### Summary
This paper introduces SelfBudgeter, a method to mitigate the "overthinking" problem in large reasoning models. Overthinking refers to models generating excessively long reasoning chains for simple problems, wasting computational resources and increasing latency. SelfBudgeter trains models to first predict a "token budget" within < budget > tags and then generate a solution that adheres to this budget. The training involves a two-stage process: a cold-start supervised fine-tuning phase to learn the output format, followed by a reinforcement learning phase (using GRPO) with a custom reward function. This function balances answer correctness, budget prediction, and precise adherence of the output length to the predicted or user-provided budget. The authors demonstrate results on mathematical reasoning benchmarks (GSM8K, MATH500, AIME2025), showing significant reductions in output length while maintaining or even slightly improving accuracy compared to baselines.

### Strengths
* It tackles a well-known and practical issue ("overthinking") in modern reasoning models, which directly impacts inference cost and user experience.

* It demonstrates compelling results on compression, achieving significant length reduction (61%, 48%) while generally preserving accuracy. The ability to sometimes improve accuracy while shortening output is notable.

* The dual-mode operation (autonomous budget and user-specified budget) is a useful feature for real-world applications, allowing users to trade off between quality and speed/latency.

### Weaknesses
* The core components are not novel. The paper's contribution is an engineering-focused synthesis of existing ideas rather than a conceptual breakthrough. It feels more like an effective engineering solution than a foundational research contribution.

* The paper shows that it works but provides limited analysis into why the model is able to maintain accuracy with shorter reasoning chains. Is it simply removing redundant verbiage, or is it learning a more efficient reasoning strategy? A qualitative analysis of shortened CoTs would be highly valuable.

### Questions
* The cold-start and RL training phases incur significant computational costs. What is the total cost of training SelfBudgeter compared to the baseline model? Is the inference-time latency introduced by the initial budget prediction step negligible compared to the savings from shorter generation?

* Can you provide examples of problems where SelfBudgeter's shortened Chain of Thought is not just concise but more efficient or less prone to error than the baseline's longer CoT? Conversely, are there cases where the budget constraint leads to catastrophic failure by cutting off crucial reasoning steps?

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
2

### Summary
SelfBudgeter trains reasoning LLMs to autonomously predict token budgets before generating solutions, addressing overthinking via dual-phase training: (1) Cold-start SFT teaches output format <budget>X</budget><solution>Y</solution> where X is predicted token count; (2) RL training with GRPO using reward R = accuracy + budget_penalty + PreB_reward. PreB reward (Eq 4) is cosine-based, penalizes if |length-budget|/budget > α, encourages minimal length for correct answers (b_best=(1-α)b) and longer reasoning for wrong ones (b_best=(1+α)b). Sets b_max = response_length if base model correct, else infinity. Dynamic α schedule linearly decays from 6.0 to 0.1 to prevent reward hacking.

### Strengths
I like this real user benefit - predicting generation time before inference is genuinely useful UX feature. Clever reward design with asymmetric b_best for correct/incorrect answers makes intuitive sense. Dynamic α schedule addresses reward hacking observed in Figure 3. Strong empirical results - achieves both accuracy gains AND compression on GSM8K/MATH500 unlike baselines. Linear relationship between predicted budget and actual length (slope 1.025 on MATH500) shows precise control. Works across model scales (1.5B to 7B). Simple method - just format change plus standard GRPO, no architectural modifications. Good ablations on cold-start data (GSM vs s1k) showing initialization matters. Outperforms L1-Max which requires explicit prompting while SelfBudgeter is "autonomous" (although see note in next section about this).

### Weaknesses
Cold-start phase seems essential but undermines "autonomous" claim? - model needs 3630 curated samples where it answered correctly, requires knowing b_max beforehand. Circular dependency: computing b_max (Eq 5) requires running base model to get correct responses, so not truly autonomous at deployment? Is the comparison unfair? - L1-Max uses fixed 3600 token limit while SelfBudgeter autonomously allocates, but Table 2 doesn't clarify this makes them incomparable. L1 with 3600 tokens gets 79.56% GSM8K, SelfBudgeter gets 84.10% but uses 1231 tokens - what if L1 used 1200 tokens? E1 comparison is also a bit unclear - what do (0.5K,1K) and (4K,1K) mean? Not explained in text. PreB reward tolerance window +-α·b is arbitrary - why not +-α·b_max or absolute tolerance? Figure 3 shows reward hacking but solution (dynamic α) is heuristic fix not principled. Why linear decay specifically? b_max=infinity when wrong seems problematic - no penalty for predicting huge budgets on hard problems. Figure 4 shows budget increases with difficulty but doesn't validate budgets are actually minimal - maybe model just learned "hard problem = big number"? AIME2025 results are a bit poor - 21.11% vs 22.22% baseline, does this suggest the method fails on hard problems? It would be stronger with a comparison to simpler baselines like "train model to output length first then generate" without complex reward.

### Questions
How do you deploy this without knowing b_max? At inference user asks new question - you need base model response to compute b_max for training, but that defeats the purpose. Is cold-start strictly necessary? What if you just do GRPO from scratch with format constraint in reward? Table 4 shows cold-start models already achieve 71.95% GSM8K with good matching rates - is RL adding much? Why does s1k initialization hurt length control (matching rate 21.76% vs 85.82% for GSM) even though s1k has better accuracy? Suggests quality-control tradeoff not resolved. Figure 5 shows many outliers with 2-3x deviation from fit line - what causes these failures? How often does model violate budget at test time? Claim "96% within 50% deviation" means half the samples could be 1.5x over budget, is this acceptable? Why compare to L1-Max specifically when L1 paper has other variants? L1 without max limit might be fairer comparison. What happens if user specifies budget smaller than model's prediction? Does it fail or compress? Can you show examples where predicting budget helps users decide to terminate early? Otherwise it's just a progress bar.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tackles the problem of overthinking in reasoning-focused LLMs, where models use unnecessarily long chains of thought for simple problems. The proposed SelfBudgeter framework adds a “token budgeting” mechanism that predicts an appropriate reasoning length before generating an answer. The system is trained in two phases: (1) a cold-start stage where the model learns to output both a token budget and a solution in a new structured format, and (2) a reinforcement learning (GRPO) stage using a reward that balances correctness, adherence to budget, and efficiency. Experiments on GSM8K, MATH500, and AIME2025 show that SelfBudgeter reduces response length by up to 61% (1.5B) and 48% (7B) while maintaining near-original accuracy.

### Strengths
1. Well-motivated: Addresses a clear inefficiency in current reasoning models (overlong CoTs) and ties the issue to both compute and user experience.
2. Simple, clear, and novel methodology: The dual-phase training and structured output (<budget>, <solution>) are intuitively designed and implemented. The PreB (Precise Budget Control) reward combines cosine shaping with tolerance margins to enforce tight adherence to predicted token budgets, which is conceptually sound and well-explained.
3. Quantitative impact: Strong empirical results demonstrating large reductions in token usage without accuracy loss; comparisons across multiple datasets and model sizes strengthen claims.

### Weaknesses
1. Limited baselines: Evaluation compares mostly against L1 and E1 methods; omits other adaptive compute or early-exit frameworks (e.g., speculative reasoning, dynamic stopping).
2. Ablation coverage: Lacks systematic analysis of individual reward components (budget penalty vs. PreB vs. correctness term) and their effect on efficiency and accuracy.
3. Limited experiments: Non-math benchmarks are lacking, albeit the baselines are trained specifically for math. Also while the paper highlights user-specified budgets, empirical evaluation of human-in-the-loop or interactive scenarios is limited.

### Questions
1. Does SelfBudgeter generalize to non-math reasoning tasks (e.g., code or commonsense)?
2. Could the authors clarify whether the model ever violates user-provided budgets during inference, and if so, how often?
3. How does the approach interact with speculative decoding or routing frameworks? Could SelfBudgeter be combined with those for further efficiency?
4. How sensitive are the results to the α parameter schedule beyond the linear decay? Would nonlinear or task-specific schedules yield further gains?

### Soundness
3

### Presentation
3

### Contribution
3
