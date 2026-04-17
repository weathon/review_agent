# Adaptive Inference‑Time Scaling for LRMs using Uncertainty‑Aware RL

- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
The widespread adoption of Large Reasoning Models (LRMs), such as Gemini 2.5 Pro Deep Think, OpenAI GPT-5 Pro, and SuperGrok 4 Heavy, is bottlenecked by their computational inefficiency, primarily stemming from the “overthinking phenomenon”—the propensity to generate unnecessarily long Chain-of-Thought (CoT) sequences even for simple queries. This verbose output, while enhancing accuracy, substantially increases inference costs and latency. Current efforts to mitigate this rely on L1 methods like explicit token budget instructions or post-hoc truncation, which either lack precise control or struggle to generalize across varying task complexities. 

We propose Uncertainty-Guided Self-Braking Tuning (USBT), an L2 adaptive inference framework that addresses the overthinking issue by enabling LRMs to autonomously regulate their reasoning depth based on real-time internal uncertainty. We frame adaptive inference as a sequential decision-making process optimized via Reinforcement Learning (RL), building on core algorithms like Group Relative Policy Optimization (GRPO). Our novel contribution is integrating a confidence metric, such as certainindex based on semantic entropy, into the RL reward function alongside explicit length penalties. 

This reward function incentivizes the model to produce concise, correct reasoning paths and facilitates an early exit strategy. Techniques like Serial-Group Decaying-Reward Policy Optimization (S-GRPO), which serialize early-exit interventions and decay rewards for later completions, demonstrate that this paradigm achieves substantial token reduction (35.4%–61.1%) while boosting accuracy. Our USBT framework generalizes this approach by actively coupling the decay/penalty coefficients with the measured uncertainty, allowing the model to recognize and inhibit excessive reasoning, cultivating an intrinsic ability to self-regulate without relying on external control. Furthermore, integrating this uncertainty-based self-regulation with inference acceleration strategies, such as branch-parallel decoding, significantly reduces end-to-end latency. Experiments incorporating our self-braking mechanism consistently show dramatic reductions in token consumption (up to 60%) across complex benchmarks while maintaining high performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles overthinking in LRMs and frames when to stop reasoning as a sequential decision problem. It proposes USBT, an uncertainty-guided halting policy trained with RL that observes a confidence signal and decides CONTINUE vs HALT at each reasoning step. Technically, it adapts GRPO and introduces S-GRPO: a serial, step-indexed grouping with decaying final reward and per-step cost, explicitly favoring earlier correct halts. On GSM8K, MATH, MMLU, and HumanEval, USBT reportedly cuts tokens by 35-60% while maintaining or slightly improving accuracy, and yields better accuracy-vs-tokens Pareto points.

### Strengths
- The paper formulates the stopping criterion as a learnable policy rather than a fixed rule. The state representation incorporates sequence length and uncertainty, and the reward explicitly balances accuracy against computation cost, aligning objectives with observed behavior.

- The work extends the GRPO framework with stepwise grouping and geometric decay, effectively avoiding the instability of critic training and addressing key challenges in long-sequence reinforcement learning.

- The proposed method is evaluated on multiple benchmarks, demonstrating reduced token usage.

### Weaknesses
* The certainIndex primarily relies on answer distribution entropy or top probability. However, in multi-step numerical reasoning or verification tasks, low entropy does not necessarily indicate sufficient reasoning, and early stopping may increase under-thinking errors. It is unclear whether using low entropy as the dominant stopping signal is justified, or whether continuing after the predicted stop point necessarily leads to redundancy or degradation.

* Recent works have explored overthinking (e.g., Do Not Think That Much for 2+3=? On the Overthinking of o1-like LLMs; Learning to Think: Information-Theoretic Reinforcement Fine-Tuning for LLMs; T-Reg: Preference Optimization with Token-Level Reward Regularization), focusing on efficiency beyond length penalties. Although the paper claims better performance over length-penalty methods, it should further clarify its advantages relative to these efficiency-oriented but non-length-penalty baselines to better demonstrate its advantages in mitigating overthinking.

* The geometric decay factor $γ$ and per-step cost $β$ jointly bias the policy toward shorter trajectories, but shorter does not necessarily imply correctness. This raises concerns about potential conservative early stopping and insufficient reasoning for harder samples. Furthermore, since the state features include final-layer embeddings, the policy may access partial semantic information, potentially leading to data leakage or learning heuristics unrelated to uncertainty.

* (Minor) Details of the baseline settings should be clarified, including unified step limits, temperature, group size G, sampling strategy, and GPU-days. Most reported results are single-point means; confidence intervals would better support the claimed improvements. The statement that the method "reduces overthinking errors while introducing some underthinking errors, with a net benefit ratio of 3:1" would benefit from explicit classification criteria and qualitative examples.

### Questions
The questions are also mentioned in the weaknesses.

### Soundness
3

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
The paper introduces a method for training a policy that decides whether to continue or stop generation at each step. This decision is guided by a reward that encourages high answer accuracy per token, and it integrates a confidence measure—semantic entropy—as a criterion for stopping.

### Strengths
The proposed method is both novel and interesting, and the authors present supporting experiments to demonstrate its effectiveness.

### Weaknesses
1. The experimental results suggest that the performance is relatively weak, particularly in comparison to other baseline methods. For instance, it only surpasses the Fixed Budget approach on the GSM8K dataset in the context of answer length compression. Additionally, evaluating the method on more complex datasets such as AMC or AIME would provide a more comprehensive assessment.

2. The main results are based on a single 34B model, but the paper does not provide any specific name or further details. It would be better to include evaluations on a wider range of open-source models.

3. The writing and presentation of the paper need improvement. The figures are not well-polished, and the excessive use of \vspace throughout the paper affects the overall readability and formatting.

### Questions
NA

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an adaptive inference-time framework called USBT that improves reasoning efficiency in large reasoning models. Instead of using fixed reasoning lengths, USBT trains a halting policy that dynamically decides whether to continue or stop generating reasoning steps based on the model’s internal uncertainty. The policy is trained through a new reinforcement learning algorithm, S-GRPO, which encourages shorter yet correct reasoning trajectories by applying decaying rewards to later steps. The paper shows that USBT achieves efficiency gains across multiple reasoning benchmarks (GSM8K, MATH, MMLU, HumanEval).

### Strengths
- The paper addresses a timely and practically important problem of reducing unnecessary reasoning length in large reasoning models by learning when to stop reasoning instead of relying on fixed or heuristic limits.

- The proposed uncertainty-guided halting mechanism is intuitive and well-motivated.

- The paper is clearly written and well-structured, making the methodology, intuition, and results easy to follow.

- The evaluation includes multiple benchmarks from different domains (GSM8K, MATH, MMLU, HumanEval).

### Weaknesses
- The novelty is limited. The use of uncertainty signals such as certainIndex has appeared in prior work without reinforcement learning, and RL-based adaptive control is already a mainstream approach. This paper primarily combines the two, raising the question of whether the contribution lies in integration rather than new conceptual insight.

- It is unclear how the method generalizes to domains where all reasoning paths converge to similar lengths. If there is little variation in reasoning depth, can the model still learn a meaningful halting policy?

- Section 3 reads more like a technical report than a research narrative. The intuition behind key design choices is missing. For example, the claim that “the policy learns to balance the terms well, preferring correct answers with minimal steps” should be supported by quantitative evidence. How well does it balance, and by how much are the reasoning steps reduced?

- The overhead of using an on-policy RL approach is not analyzed. Training on-policy can be costly; an off-policy or hybrid alternative might achieve similar results with less computation. What are the trade-offs between accuracy, stability, and efficiency across these settings?

- The evaluation is limited, focusing only on a single model scale (a 34B-parameter reasoning model). Would the same trends hold for smaller or larger models, or across different architectures?



Minor comments:

- The related work section is overly long and could be condensed to emphasize the most directly relevant prior studies.


- The fonts in Figures 2, 3, and 4 are too small, making them difficult to read.

- There is unnecessary \vspace usage around line 92, which should be removed for cleaner formatting.

### Questions
See the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work reports a ICL based method, named USBT) to train a on-policy classifier to continue or halt rollouts in reasoning.  The work builds on GRPO to add a decay to the reward to incentivise earlier solutions, and a entropy based measure, called _certainIndex_ to measure prospective answer distributions.

### Strengths
* This work intuitively brings a decay reward factor into GPRO, adding the capability of judging the necessity of additional lengthened rollouts in reasoning.
* The token savings are significant in the analysed cases on GSM8K and MATH against a standard ACT baseline, and define a Pareto optimal frontier (at least against the experimental configurations tested).
* The authors also bin trajections by difficulty according to _certainIndex_ to hedge against sparsity, a useful step to better ensure convergence in training.
* The analyses over the different difficulty bins show intuitive patterns, which reinforce the method's soundness.

### Weaknesses
* The presentation of the work is not very well done.  While formatting problems with (LaTeX) quotation marks, charts, tables, and citations are not necessarily problematic, they give a strong impression that work was rushed to submission.  
* The experimental validation is somewhat narrow.  Experiments are restricted to a base 34B anonymous LRM.  This is problematic for two reasons: 
  * The model is somewhat large (34B), but smaller and less costly (7-8B and sub-1B models are not tested at all)
  * The model is anonymous, so replication and reproducibility are problematic.
* The reasoning datasets are somewhat limited to three, well-tested datasets, which are essential but should be supplemented with additional datasets that require less math/logic reasoning.  Testing on such datasets would help generalise the claims better
* There is insufficient analysis of the faults and micro-analysis.  The submission relies solely on the macroscopic evaluation results to claim
* There are other recent, published, RL-based methods that should be tested as baselines against this work.    
* The reward design is straightforward, and difficult to claim novelty.  Geometric rewards are standard practice.

### Questions
* How often does your method hit the ceiling rolllout length $T_{max}$?  How does that affect failure cases? (This should be reflected in Algorithm 1 as well)
* (See weaknesses) Why do you use an anonymous LRM and use REINFORCE and PPO as baselines?  There are related works in these areas, that would be more fitting to compare against.

### Soundness
2

### Presentation
2

### Contribution
2
