# RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Despite rapid advancements in large language models (LLMs), the token-level autoregressive nature constrains their complex reasoning capabilities.
To enhance LLM reasoning, inference-time techniques, including Chain/Tree/Graph-of-Thought(s), successfully improve the performance, as they are fairly cost-effective by guiding reasoning through external logical structures without modifying LLMs' parameters.
However, these manually predefined, task-agnostic frameworks are applied uniformly across diverse tasks, lacking adaptability.
To improve this, we propose **RL-of-Thoughts (RLoT)**, where we train a lightweight navigator model with reinforcement learning (RL) to generate task-adaptive logical structures at inference time, enhancing LLM reasoning.
Specifically, we design five basic logic blocks from the perspective of human cognition.
During the reasoning process, the trained RL navigator dynamically selects the suitable logic blocks and combines them into task-specific logical structures according to problem characteristics.
Experiments across multiple reasoning benchmarks (AIME, MATH, GPQA, etc.) with multiple LLMs (GPT, Llama, Qwen, and DeepSeek) illustrate that RLoT outperforms established inference-time techniques in most cases and improves up to 13.4% in challenging situations.
Remarkably, with less than 3K parameters, our RL navigator is able to make sub-10B LLMs comparable to 100B-scale counterparts.
Moreover, the RL navigator demonstrates strong transferability: a model trained on one specific LLM-task pair can effectively generalize to unseen LLMs and tasks.
Our code is open-source at https://github.com/tsinghua-fib-lab/RL-LLM-Reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes the technique RL-of-Thoughts (RLoT), an inference-time method that trains a tiny “navigator” with reinforcement learning to compose task-adaptive reasoning structures for LLMs. Long-sequence reasoning is cast as an MDP, where the state is a self-evaluation of the current reasoning, and 5 possible actions: Reason one step, Decompose, Debate, Refine, and Terminate. A process-reward model scores intermediate steps to provide the reward signal. Experiments show consistent accuracy gains and the authors also claim transferability of a single navigator across LLMs and tasks.

### Strengths
1. It presents a method that reframes reasoning as an MDP over a small set of actions, avoiding deep tree searches.
2. Results show performance gains across different tasks and models. The authors have analyzed a wide range of models.
3. According to the authors, the method is transferable and generalizable to other models and datasets.

### Weaknesses
1. The method repeats inferences with self-consistency filtering. Including the results of token budgets per method and normalize across baselines can help understanding the budget costs in a fair way. 
2.PRM calibration. How well do PRM scores (Math-Shepherd) correlate with eventual correctness on non-math tasks?
3. The paper fails to include confidence intervals of the results.

### Questions
1. How does the number of tokens generated to achieve the solution compare between RLoT and the baselines?
2. How does the number of actions affect the results? Why are these actions selected?

### Soundness
3

### Presentation
1

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
The paper proposes RLoT, a new inference-time technique to enhance the complex reasoning capabilities of LLMs. The core problem addressed is that existing inference-time methods use predefined, task-agnostic logical structures that lack adaptability. RLoT's key idea is to train a lightweight "navigator model" using Reinforcement Learning to dynamically construct task-adaptive logical structures during inference.

### Strengths
1. The primary strength is the RLoT framework itself. It reframes the problem from "how to build a better reasoning LLM" or "what is the best fixed reasoning structure" to "how to learn a policy that navigates a frozen LLM's reasoning space." This is a clever and effective conceptual leap.

2. The results show that a sub-10B model with a 3K-parameter navigator can close most of the performance gap to a 70B model on tasks like GPQA and StrategyQA is a massive win.

3. The method is tested on multiple SOTA LLMs and a wide array of difficult reasoning benchmarks.

### Weaknesses
1. The state space, which is the sole input to the RL agent, is based on the LLM's own self-evaluation. The authors provide a validation (82% accuracy) in Appendix F and show it works better than alternatives. However, this remains a potential point of noise or failure. One might question if an LLM that is failing at a reasoning task (the "hard questions" used for training ) can simultaneously be a reliable evaluator of its own failing state. This could be particularly problematic for smaller, less-capable models.

2. The five logic blocks are "inspired from human cognition"  and the quality of the entire system is upper-bounded by the quality and expressiveness of this predefined action set. However, they are hand-engineered component.

### Questions
1. The quality of the 5 logic blocks seems highly dependent on the quality of their prompts (listed in Appendix K.2 ). How much prompt-engineering was required to make these actions (especially Debate and Refine ) work reliably? Is the framework sensitive to the phrasing of these action-prompts?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes RL-of-Thoughts (RLoT): treat multi-step reasoning as an MDP where a tiny navigator (Double-Dueling DQN, <3k params) learns to select among five logic blocks (Reason one step, Decompose, Debate, Refine, Terminate) while the base LLM stays frozen. States are 7-aspect self-evaluations the LLM generates via a fixed prompt; rewards come from a Process Reward Model (PRM), e.g. math shepherd. The navigator is trained on "hard" questions only; at test time they use self-consistency to pick the final answer.

### Strengths
- Reasoning as an MDP with a standardized action set and state interface; prompts for states/actions are fully specified and the MDP is clearly defined. The navigator adds negligible runtime overhead; training cost is amortized and reported with token accounting.
- Good comparison with baselines (fixed workflows like DeAR, search methods like GoT, r\*, LiteSearch, DSPy, ...) with an "extended table" across GSM8K/GPQA/StrategyQA; The authors also report token usage vs. accuracy.
- Proper ablation:  Remove-one-block studies (for Decompose, Debate, and Refine),

### Weaknesses
Major problem is the insignificance of many results: For instance in Table 9, RLoT = 92.87 while several baselines cluster ≥92% (e.g. Buffer-of-Thoughts 92.35,  ...). The paper text claims overall wins, but GSM8K specifically looks like parity rather than a clear lead; without error bars the <1% differences are hard to interpret and they are rather frequent in the paper.

### Questions
1. Could you report mean ± 95% CI over ≥3 seeds for the results (especially Table 9) (GSM8K/GPQA/StrategyQA) and key math sets? If not, please justify why single-seed is sufficient given self-consistency randomness.
2. Is there a reason behind fix training at 3,000 episodes? Do curves show continued improvement (or overfitting) at 1k/5k?
3. Table 10 reports tokens/question, but wall-clock or FLOPs can be very informative as different methods assume different number of model. Can you add runtime and PRM calls to enable compute-fair comparisons vs. search-heavy methods (e.g., r\*, LiteSearch)?
4. Did you think of including SelfCheck[1] and Least-to-Most prompting [2] to experiments as baselines? Even a lightweight SelfCheck (PRM as a one-shot checker without RL) would clarify how much the RL policy contributes beyond access to a PRM.
5. You ablate −Decompose/−Debate/−Refine; can you also show fixed schedules (e.g., Decompose→Reason→Refine cycles à la DeAR) under the same token budget to separate "RL policy" gains from "having blocks at all"?

[1] Miao et. al.. Selfcheck: Using llms to zero-shot check their own step-by-step reasoning. ICLR, 2023.

[2] Zhou,et al. Least-to-most prompting enables complex reasoning in large language models. ICLR, 2022.

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
3

### Summary
The authors hypothesise that inference-time reasoning frameworks such as Chain of Thought (CoT), Tree of Thoughts (ToT), and Graph of Thoughts (GoT) are overly rigid and fail to adapt dynamically to different reasoning problems. To address this, they propose a reinforcement learning–based controller (“navigator”) that, given the current reasoning state representation, selects among five “logic blocks”: Reason, Decompose, Debate, Refine, or Terminate.

Reasoning is formalised as a Markov Decision Process (MDP) where states are defined by the LLM’s self-evaluation, actions correspond to logic blocks, and rewards are provided via a process reward model (PRM). The navigator is trained using a Double-Dueling DQN and subsequently used at inference time to guide the LLM’s reasoning adaptively.

### Strengths
- The idea of viewing reasoning as an MDP with composable cognitive actions is original and conceptually appealing. The method is explained clearly.

- The proposed navigator has fewer than 3K parameters, making it lightweight, computationally inexpensive, and compatible with frozen LLMs.

- The authors experimented across reasoning, math, STEM, and commonsense domains, and also covered both small- and medium-sized models.

- The authors show that their method can perform cross-model and cross-task generalisation.

### Weaknesses
- There’s no ablation on reward signals or PRM accuracy, so it’s unclear whether improvements come from RL navigation or simply from PRM-guided prompting.
- Some parts of the paper are not detailed enough, for example, section 4.5. How many instances were considered? The authors do not justify why Double-Dueling DQN is used. There is no exploration of policy stability, convergence, or sample efficiency. 
- The paper majorly lacks qualitative analysis. The paper presents only anecdotal examples of reasoning patterns (“Reason–Refine–Debate”) and no failure-case analysis, error typology, or robustness study. We don’t know when or why RLoT fails. 
- Minor point: The paper over-relies on the appendix for critical methodological and experimental details. The main text should ideally be more self-contained.

### Questions
- What happens if the self-evaluation state is noisy or replaced with random values — does the navigator still help?

### Soundness
2

### Presentation
3

### Contribution
3
