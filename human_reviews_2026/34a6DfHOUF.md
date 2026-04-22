# Benefits and Pitfalls of Reinforcement Learning for Language Model Planning: A Theoretical Perspective

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 6, 2

## Abstract
Recent reinforcement learning (RL) methods have substantially enhanced the planning capabilities of Large Language Models (LLMs), yet the theoretical basis for their effectiveness remains elusive. In this work, we investigate RL's benefits and limitations through a tractable graph-based abstraction, focusing on policy gradient (PG) and Q-learning methods. Our theoretical analyses reveal that supervised fine-tuning (SFT) may introduce co-occurrence-based spurious solutions, whereas RL achieves correct planning primarily through exploration, underscoring exploration’s role in enabling better generalization. However, we also show that PG suffers from diversity collapse, where output diversity decreases during training and persists even after perfect accuracy is attained. By contrast, Q-learning provides two key advantages: off-policy learning and diversity preservation at convergence. We further demonstrate that careful reward design is necessary to prevent Q-value bias in Q-learning. Finally, applying our framework to the real-world planning benchmark Blocksworld, we confirm that these behaviors manifest in practice.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides a theoretical analysis of reinforcement learning (RL) methods, particularly policy gradient (PG) and Q-learning, for planning in large language models (LLMs). It builds on recent empirical works showing that RL enhances reasoning and planning beyond supervised fine-tuning (SFT), and offers a rigorous mathematical explanation of why and how that happens.

The authors adopt a graph-based abstraction of planning, where path-finding over a directed graph represents multi-step reasoning or decision-making (e.g., tool use, theorem proving, or task sequencing). They theoretically and empirically analyze three regimes 1) SFT: Learns co-occurrence-based correlations without transitive reasoning, leading to memorization rather than true planning; 2) Policy Gradient (PG): Outperforms SFT via exploration-based data augmentation, but suffers from diversity collapse (reduced output variety even after convergence). KL regularization mitigates this but can hurt accuracy; 3) Q-Learning: Addresses PG’s pitfalls by enabling off-policy learning and preserving output diversity at convergence—but risks reward hacking unless process rewards (stepwise feedback) are used instead of binary outcome rewards. The authors validate these behaviors using both synthetic Erdos–Rényi graphs and the Blocksworld planning benchmark

### Strengths
- This paper has a strong theoretical foundation: The paper provides a clean, formal framework for analyzing planning in LLMs, translating abstract reasoning tasks into tractable graph-based path-planning problems. 

- Novel Theoretical Insights into RL for LLMs: Identifies diversity collapse in PG as a fundamental phenomenon, theoretically proving that entropy monotonically decreases after convergence—a widely observed empirical issue in RLHF and GRPO models. It reveals that Q-learning avoids this trap by updating state-action values rather than direct log-probabilities, preserving diversity across valid next actions. As well, it formalizes the reward hacking issue and demonstrates how process rewards eliminate it.

- Empirical Validation with Clear Visual Evidence: Figure 1 visualizes adjacency reconstruction in Blocksworld—showing how SFT fails to recover the graph structure while PG and Q-learning succeed. Then, the paper connects theory to practice via Figures 2–4. Figure 2 shows how KL regularization balances accuracy and diversity in PG. Figure 3 demonstrates Q-learning’s superior test accuracy and stability. Figure 4 draws heatmaps of logits reveal that feasible transitions converge to uniform values (confirming theoretical diversity preservation).

- Clear Conceptual Takeaways: The paper summarizes each section with concise takeaway statements, synthesizing intuition for both theoretical and empirical results. This makes it approachable despite heavy mathematical content.

### Weaknesses
- Limited Practical Scope: The paper primarily focuses on simplified models (one-layer, single-head transformers and synthetic graphs). While conceptually illuminating, the connection to real-world LLM training (e.g., PPO/GRPO pipelines) remains somewhat indirect. It has no experiments on large-scale reasoning benchmarks (e.g., MATH, GSM8K, or ALPINE) are included to show external validity.

- Simplistic Reward Modeling: The outcome vs. process reward distinction is theoretically elegant but lacks discussion on how these correspond to real LLM reward signals (e.g., chain-of-thought correctness, factual consistency). Furthermore, reward hacking is discussed in the graph setting but not demonstrated empirically in language tasks.

- Empirical Validation is Narrow: All empirical results use synthetic environments. If authors can have results on real text-based reasoning or tool-use tasks, then the statements would be much more strengthen that the same dynamics hold for modern RL-tuned LLMs.

- Absence of Theoretical Analysis for PPO/GRPO: Although Section F shows PPO is mathematically equivalent to unclipped PG under stop-gradient treatment, this misses important effects like clipping or value normalization used in real systems.

### Questions
- How do the diversity collapse dynamics extend to multi-step, natural language reasoning, where outputs are discrete tokens instead of graph edges?

- Can Q-learning be feasibly integrated with large transformer-based LLMs at scale? Are there existing implementations (e.g., in VeRL or R1) that align with the off-policy formulation analyzed here?

- Would adding entropy regularization or temperature annealing alter Theorem 4.3’s diversity-collapse dynamics?

- How sensitive are the conclusions to the assumption of persistent exploration? Could adaptive exploration decay or real-world sampling noise violate this condition?

- Could you relate process rewards to verifiable subgoal rewards used in reasoning tasks (e.g., partial correctness of intermediate steps)?

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
3

### Summary
This paper provides a theoretical comparison of different training paradigms for enabling language models to learn to plan—namely Supervised Fine-Tuning (SFT), Policy Gradient (PG), and Q-learning—within a controlled graph-based path planning setting.

The authors first show that SFT tends to memorize observed co-occurrences limiting generalization. They then demonstrate that PG improves over SFT despite having a similar objective, because its on-policy data generation introduces exploration, effectively augmenting the training set. However, they identify a diversity collapse in PG, where the model’s output variety diminishes even after achieving perfect training accuracy—a problem that KL regularization can mitigate at the cost of lower accuracy.

Finally, the authors analyze Q-learning, showing that it faces convergence issues and detail a form of “reward hacking” when relying solely on final outcome rewards, but that these problems can be addressed by introducing a process rewards to supervise intermediate steps. Under this setup, Q-learning naturally preserves output diversity and supports off-policy learning—unlike PG.

All theoretical predictions are validated empirically on a small Blocksworld planning environment using a one-layer Transformer.

### Strengths
- The paper is well written and clearly presents the theoretical analyses, making it easy to follow the derivations and main arguments.

- The controlled graph-based path planning framework provides a clean setting to extensively study the differences between SFT, PG, and Q-learning for planning with language models.

- The theoretical claims are relevant to the real world challenges of training LLM to plan.

### Weaknesses
**Lack of empirical validation at scale**  
While the theoretical analyses are thorough and insightful, the empirical validation is limited to a small Blocksworld environment with a one-layer Transformer. It remains unclear how well the theoretical findings translate to larger, more complex models and real-world planning tasks. Additional experiments on larger datasets and models would strengthen the practical relevance of the results. Given the clearness of the theoretical contributions, it seems that it would be straightforward to validate them on larger models/environments.

**Figure 1 description**
Could you add more explanation to Figure 1? You only mention that SFT performs worse, but it’s not very clear how this is shown in the figure. For example, it seems that PG and Q-learning tend to hallucinate more than SFT — is that correct? I’m referring to the strong diagonal from bottom-left to top-right, which appears high for PG and Q-learning but low for SFT (as in the dataset). Also, Q-learning seems to produce a large bright block on the right — what does that correspond to?

**Lack of empirical validation for important assumption**  
Theoretical analyses throughout the paper hinge on Assumption 3.1 — motivated by Wang et al. (2024b) — that the model’s logits depend only on the current and target nodes. However, this assumption is adopted without further empirical verification. The paper does not test whether real Transformer models indeed satisfy this reduced dependency, leaving open whether the derived results and takeaways accurately reflect practical LLM behavior.

### Questions
- Your theoretical results rely on **Assumption 3.1**, which posits that the model's logits for the next token depend only on the current node $u_m$ and the target node $u_2$, effectively neglecting the influence of the preceding path and initial state.  
Could you discuss how your analysis would change if this assumption were relaxed—for instance, if the model's predictions also depended on earlier nodes in the trajectory (the full prefix $u_1, \ldots, u_m$)?
In particular, would the main results (e.g., Theorem 3.1’s co-occurrence characterization or the PG/Q-learning convergence properties) still hold qualitatively, or would new interaction terms or dependencies appear in the gradient dynamics?

- In your formulation, RL exploration appears to operate only over paths between fixed source–target pairs sampled from $D_{\text{Train}}$. Have you considered or analyzed the possibility of *exploring over source–target pairs themselves* (e.g., dynamically selecting or expanding the set of training pairs)? How might such goal-level exploration affect the generalization and theoretical results you present?

- In Figure 2, PG ($\lambda = 0$) achieves its highest test accuracy early in training before diversity collapse occurs, while KL regularization later limits accuracy gains. Wouldn't *early stopping* at this peak be a simpler and potentially more effective strategy than adding KL regularization, since it avoids both collapse and the regularization-induced accuracy loss? Did the authors consider this comparison?

- **How specific are your findings to transformer architectures?**
Your theoretical analysis and experiments use a one-layer Transformer as the backbone. How would the conclusions change for other model classes such as RNNs (e.g., LSTMs), state-space models (SSMs), or diffusion-based architectures? Do your results rely on properties unique to transformers (e.g., attention structure), or do they extend to other sequential models?

-**In real-world applications, process-level reward information is often unavailable.**
Given that your analysis shows Q-learning may fail to converge under outcome-only rewards, how could this limitation be addressed when such intermediate reward signals are not accessible?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper theoretically analyzes why LLMs work well for planning problems. They formulate the problem as  a graph where the prompt defines a start state and a target, and then the agent gets rewarded for generating a valid trajectory through the graph from start state to target.

They provide theoretical explanations for...
- why Policy gradient (PG) > SFT (exploration)
- why PG results in entropy collapse
- why KL regularization mitigates this
- why Q-learning on outcome rewards experiences "reward hacking" (their word for inaccurately optimistic Q-value estimates)
- How Q-learning with process rewards mitigates this

### Strengths
- Paper carefully cites where it extends prior work
- Claims are supported by experiments in Blocksworld, where they observe that KL regularization maintains output diversity during PG training and poor accuracy during Q-learning training with outcome rewards but good accuracy with process rewards.
- Graphs are clear and have error bars.

### Weaknesses
See the "Questions" section for more details, but at a high level, I think there are 3 areas for improvement:

- Make it clearer exactly how your theoretical analysis ties into real-world LLM setups. I think this could be made very explicit with a running example that you reference back to in each section (e.g. you could say something like "suppose you have a dataset consisting of math proofs. For each proof in the dataset, the starting state is [....]. The target is [.....]. Each state is [.....]. There is a valid transition between states if [....]..." Then when you introduce Q-learning you could say something like "The Q-function takes inputs in this form [...question, math proof up until step X...]. But when we optimize it, we see reward hacking [...predicting that invalid proofs are correct?...] Also discuss whether the assumptions you make when constructing graphs for your experiments are realistic assumptions that match data distributions in real natural language. (you have a line "a tool-use scenario can be modeled as identifying a valid call sequence within an API call graph" -- this is a good candidate to be fleshed out into a full example).

- The trajectory notation with u1 = start state, u2=target u3+ = actual trajectory was somewhat confusing to me. I'd suggest giving the start state and target different notation (e.g. s0, s_target) and then indexing the actual trajectory from 1.

- See my questions in the "Questions" section about the mismatch between the non-markovian setup in real LLMs and the markovian Q-learning setup.

- I think the phrase "reward hacking" is a bit misleading in this case. Typically this is used to refer to exploiting a misspecified external reward rather than having biased upward estimates of Q-value estimates within the policy.

### Questions
Disclaimer: it is quite possible that I have misunderstood some of the key arguments in this paper. If these could be clarified (both for myself and for future readers) my score could increase significantly.

Question 1 - could you clarify exactly what you mean by a "graph" in this setup? I think it was often ambiguous whether the paper is referring to a semantic graph (e.g. for a math proof, nodes are proof prefixes and an edge from A to B exists is B contains A, but with one more logically correct step added (or is it one valid next token added?)) or to a literal graph (vocabulary are nodes, graph is fully connected b/c any token can come after any other one). 

I'm also very confused about whether we are assuming the graph is markovian or not. In section 2.1, it says that each vocabulary item is a word. However, whether a transition from one word to the next is valid depends not just on the current word, but on all prior words in the sequence. Furthermore autoregressive transformers look at sequence prefixes not just words in isolation. So is a node a single word, or a sentence prefix? If it's a single word, I don't think this matches how LLM RL is done in practice.

This confusion further makes me question the Theorem 5.1 results. It makes sense that you would converge to equal Q-values for all next states if it is possible to select k' across all possible next nodes. However if nodes are prefixes it's not clear to me if this still holds. My mental model is a problem where the start token is "Hello" and the model gets reward 1 if its next token is "World", but reward 0 for anything else ever. If a transformer is trained on a batch of trajectories with "Hello World"-> 1 and [anything else]-> 0 it is not clear to me the theorem still holds. If a sequence is [random, sequence, of, words....] then there is no possible transition from that to a sequence [Hello, World], so I don't think Q-learning would ever give this sequence > 0 reward.

Question 2 - I think some of the discussion about SFT vs PG was a bit unclear:
- The paper states that "SFT memorizes" whereas "PG generalizes". The claim that SFT memorizes makes sense and was well supported. However, the claim that "PG generalizes" seems at odds with the analysis showing a loss of diversity. It seems like PG is not truly generalizing, merely memorizing a small set of paths from start state to the solution. The Fig 2a results do show stronger test performance of RL methods, but the "Continual SFT" setup was not described, so I'm not sure what it means.
- There are 2 differences between SFT and PG: (a) PG has a notion of reward which SFT lacks (b) SFT is static whereas PG collects new trajectories online. The theoretical analysis focuses on the benefit of (b) but I think does not address (a).

### Soundness
1

### Presentation
1

### Contribution
2
