# ToTRL: Unlock LLM Tree-of-Thoughts Reasoning Potential through Puzzles Solving

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 8

## Abstract
Large language models (LLMs) demonstrate significant reasoning capabilities, particularly through long chain-of-thought (CoT) processes,
which can be elicited by reinforcement learning (RL). 
However, prolonged CoT reasoning presents limitations, primarily verbose outputs due to excessive introspection. 
The reasoning process in these LLMs often appears to follow a trial-and-error methodology rather than a systematic, logical deduction.
In contrast, tree-of-thoughts (ToT) offers a conceptually more advanced approach by modeling reasoning as an exploration within a tree structure. 
This reasoning structure facilitates the parallel generation and evaluation of multiple reasoning branches, allowing for the active identification,
assessment, and pruning of unproductive paths. 
This process can potentially lead to improved performance and reduced token costs.
Building upon the long CoT capability of LLMs, we introduce tree-of-thoughts RL (ToTRL), a novel on-policy RL framework with a rule-based reward.
ToTRL is designed to guide LLMs in developing the parallel ToT strategy based on the sequential CoT strategy. 
Furthermore, we employ LLMs as players in a puzzle game during the ToTRL training process. 
Solving puzzle games inherently necessitates exploring interdependent choices and managing multiple constraints,
which requires the construction and exploration of a thought tree, providing challenging tasks for cultivating the ToT reasoning capability. 
Our empirical evaluations demonstrate that our ToTQwen3-8B model, trained with our ToTRL,
achieves significant improvement in performance and reasoning efficiency on complex reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Tree-of-Thoughts Reinforcement Learning (ToTRL), an on-policy RL framework designed to move an LLM from linear chain-of-thought (CoT) reasoning to tree-of-thoughts (ToT) reasoning. The method comprises three main components: 

1. Policy optimization: The policy is trained using a clipped ratio objective, optionally augmented with a KL divergence term to a reference policy.

2. Two-stage training: Stage 1 (“no-thinking mode”) uses a special prompt template to suppress the model’s usual CoT trace and induce explicit ToT steps within markup tags. Stage 2 (“thinking mode”) trains the model in standard generation mode to internalize the learned ToT behaviors for inference.

3. Task-driven learning: The policy is trained via puzzle games that benefit from branching search, including 6×6 Sudoku and alphametic puzzles.

The resulting ToTQwen3‑8B model is evaluated on both in-distribution puzzles and several out-of-distribution logic tasks, using accuracy as the primary metric. Results show consistent improvements over several ~7–9B baselines.

### Strengths
S1: The paper addresses the well-known inefficiency and verbosity of long CoT reasoning by enabling branching exploration with a global perspective, aligning with prior work in ToT and graph-based reasoning.

S2: The use of a rule-based validator combined with an exact-match reward is straightforward to reproduce for puzzle tasks and eliminates dependence on human-labeled rationales, reflecting trends in O1/R1-style RL frameworks.

S3: Empirical results show that ToTQwen3‑8B achieves higher accuracy with fewer thinking tokens than the Qwen3‑8B baseline across multiple tasks, offering a notable practical advantage in efficiency and computational cost.

### Weaknesses
W1: Equation (1) resembles a PPO-style clipped objective with an optional KL term to a reference model. Calling it REINFORCE may obscure the actual optimization method used.

W2: The exact set-equality reward (Eq. 5) is brittle; success may hinge on precise formatting or extraction of answers, which could inflate performance or reduce reproducibility.

W3: Comparisons omit relevant search-based alternatives, including: (i) self-consistency over CoT traces, (ii) explicit ToT BFS/MCTS as in the original ToT paper, (iii) RAP (planning with MCTS), and (iv) TS-LLM (AlphaZero-style value-guided search).

W4: It is unclear whether all baselines were evaluated with identical token budgets, early-stop rules, and “thinking mode” support. Baselines lacking special thinking channels may be disadvantaged, making cross-model comparisons potentially unfair.

W5: The training tasks are restricted to puzzles with exact-rule validators. Real-world reasoning tasks (coding, planning, open-ended writing, tool use) often require partial credit, multi-step execution, debugging, environment interaction, or human judgment. Training only on puzzles may not stress the full range of reasoning, error exploration, and backtracking required in practical scenarios.

### Questions
Q1: Is Eq. (1) implemented as PPO with ε‑clipping? If so, why call it REINFORCE? 

Q2: What is the rollout size n per prompt, sampling temperature/top‑p, and maximum tokens per “thinking” segment? How is the stop instruction inserted when the thinking budget is reached, and how is a partial tree summarized before answering? 

Q3 Did you try partial‑credit rewards (e.g., counting correct solutions, Sudoku constraint satisfaction), or step‑level validators? 

Q4: Did every baseline receive the same token budget and stop rule? Several baselines lack special “thinking modes”, how did you ensure fairness?  

Q5: How does the method perform on open-ended reasoning tasks as described in W5, beyond puzzles?

### Soundness
2

### Presentation
3

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
The article introduce tree-of-thoughts RL (ToTRL) framework to guide LLMs to develop parallel ToT capabilities beyond sequential CoT. After ToTRL training process, the LLMs can solve puzzle games better, including in-domain and out-of-domain ones.

### Strengths
Introducing parallel thinking patterns into reasoning LLMs sounds a reasonable effort. 

The ToTQwen3-8B model shows significant performance gains on a variety of logic puzzles.

### Weaknesses
Since the authors are still leveraging the CoT prompt for mathematical problems, it is unclear to me why it improves OOD mathematical tasks. Can you provide analysis as to why it also helps mathematical tasks?

I am particularly curious why ToTQwen3-8B can “explore the solution space more effectively and efficiently” as the authors mentioned, given that ToT is often very costly. Some experiment setting details of Figure 3 in section 3.5 are unclear. How do you set the budgets as exactly (2^c) k tokens? Do you set a budget for each method and truncate the thinking length?

### Questions
“Initially, as illustrated in Figure 1, the LLM undergoes training to perform ToT reasoning in a non-thinking mode. The non-reasoning mode is achieved by introducing blanks between reasoning tags, which compels the model to suspend its conventional reasoning processes.” This is not explained clearly, even after referring to Figure 1. 

What does the separation line in Table 3 mean? Are there fundamental differences between the above 2 models and the middle 3 models?

The authors mention “Collectively, these efforts demonstrate the significant potential of internalizing ToT capabilities within the LLM itself, moving towards more autonomous reasoning.” I find the paper *Autonomous Tree-search Ability of Large Language Models* proposed the notion of autonomous ToT reasoning ability two years ago, and I believe there were relevant efforts in the literature. The authors can also consider including a discussion of the literature in this direction.

### Soundness
2

### Presentation
2

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
This paper proposed to apply RL to Tree-of-thoughts (ToT) with two-stage training, named ToTRL. This approach aims to narrow the gap between linear COT and parallel TOT generations, hence at the first stage the reasoning trace from the original model is turned off, then at the second stage both COT and TOT thoughts are turned on with RL for training. The paper tunes the model on two puzzle tasks for adaptation to the improved TOT reasoning patterns.

### Strengths
The paper tackles an important problem of how to design an effective training procedure of improving parallel thinking techniques like TOT.

### Weaknesses
- The motivation to adapt CoT to ToT reasoning is not well justified. It remains unclear in what sense is the linear COT unsuitable under the TOT setting, and whether the gain from 2-stage training is just due to extended-training.
- It doesn't seem to be convincing that by applying RL on only two puzzle tasks, the model performance can be improved over a wide range of reasoning tasks. The claim of the title that training on puzzle tasks can unlock the potential of ToT is very broad and needs deeper justification.
- The experimental result analysis did not reveal whether ToTRL truly improved tree search quality (diversity and depth).

### Questions
1. In the 2nd stage of ToTRL ("thinking mode"):
- How do thoughts between <think> and </think> differ from <tot> and </tot>? It seems what <tot> captures is just a summary of <think>, rather than novel thoughts that could further improve tree search quality.
- If thoughts between <think> and </think> come from the 1st stage ("no-thinking mode"), then since the base model did not go through ToT training yet, the thought quality is expected to be bad? This is exactly the problem the paper wants to address, not sure how effective it is to use these thoughts directly for 2nd stage training.

2. For fair comparisoin, it seems the baseline models in experiments should also adopt ToT style reasoning, rather than using on their native reasoning paths.

3. Was the improvement from ToTRL just an artifact of prolonged training, or it truly improved the quality of search trees? There isn't analysis on the difference between tree quality before and after ToTRL.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper adds reinforcement learning to the tree-of-thoughts (which is a generalization of CoT) to support better LLM reasoning for solving games/puzzles/reasoning tasks. CoT is linear (ie sequential) and ToT can support branching exploration of multiple pathways. The authors' contribution is to add an on-policy RL on top of a rule-based reward system to help the LLM transition from sequential CoT to parallel, tree-structured reasoning.

### Strengths
- The paper is extremely well written and details are fleshed out to support reproducibility
- The experimental results show significant improvements based on a Qwen model that the authors have trained/fine-tuned.
- The authors also demonstrate their approach in a test-time-scaling experiment and show that the learned policy is good to explore the search space better.

### Weaknesses
- In the beginning of the paper, the authors mention that "Initially, the LLM is trained to perform ToT reasoning in a non-thinking mode, leveraging more moldable thinking patterns to activate ToT reasoning. Once the LLM has developed a degree of ToT reasoning ability in the non-reasoning mode, it undergoes
further training in the reasoning mode." This wasn't re-referred back later in the paper. Can you show/demonstrate examples of these patterns that activate ToT? Can you show ablation results showing the necessity of this initial reasoning in non-thinking mode? Is it because the CoTs are not "faithful"?

### Questions
- Please think of an additional experiment to address my question above.

### Soundness
3

### Presentation
4

### Contribution
4
