# When Evolution Meets Momentum: Orchestrating Goal-oriented and Process-oriented reasoning for LLM Inference Scaling

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large language models (LLMs) have demonstrated strong reasoning ability when given additional compute at inference time. However, existing inference-time scaling methods are fundamentally limited by their design. On the one hand, gaol-oriented approaches, such as Line or Tree Search, refine candidate solutions using feedback but are vulnerable to sequential dependence, often collapsing into suboptimal reasoning trajectories. On the other hand, process-driven approaches such as Best-of-N sampling encourage diversity through random exploration but lack feedback mechanisms, leading to inefficient computation allocation and unguided search. In this work, we propose EvoMo, a novel inference-time scaling approach that unifies both paradigms by embedding a globally evolving strategy pool into MCTS, where each node expansion selects reasoning strategies under an $\varepsilon$-soft policy.
To further avoid stagnation in familiar strategies, we introduce a \textit{momentum-based optimization} mechanism that monitors similarity among generated solutions and encourages the exploration of underutilized strategies. Across benchmarks, EvoMo reveals significant performance gains over SOTA inference scaling methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The work proposes an inference time scaling technique that induces diversity into MCTS-like techniques to improve Best of N performance. The method observes the similarity of different approaches and through momentum based strategies, it encourages more utilization on under represented approaches.

### Strengths
- Increasing solution diversity is a promising avenue for this research that deserves more attention.

- The method is simple, well motivated and clearly explained. 

- Initial results look promising especially for the larger models

### Weaknesses
- I found that the improvements gained through the method are either modest or otherwise not very well highlighted in the paper. For example the larger models seem to have a larger gain but this comes very late in the paper. Also, the paper does not shown results for all datasets on all models to get a good picture of reality. I would suggest the authors to add a full page set of figures with number of tries in the x axis and accuracy +bon in Y for all datasets and all models they work with. Otherwise, the story is too distributed.

- In the diversity for inference time scaling field, there are some traditional methods that basically try to sample at different temperatures (the Self Consistency method) or via prompting by asking the model to follow a particular approach or style or persona or in-context examples to enforce diversity (see Diversity of thought improves reasoning abilities of large language models). Would be useful to compare with some of these more straightforward approaches to judge the benefits. The appendix talks somewhat about this but 1:1 comparisons are not available.

- More generally, it would be good to see benefits in other domains, beyond code.

### Questions
- What model is used in Figure 1?

- Why do the authors only track BoN? It would be useful to also see trends in majority vote and average accuracy.

- Improvements in Table 1 even for BoN seem very modest. What am I missing?

- Why do the authors only consider coding benchmarks?

- How does token cost usage compare for your method and the baselines?

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
3

### Summary
The paper proposes EvoMo, a hybrid inference-time scaling framework for large language models that unifies goal-oriented search (like MCTS) with process-oriented diversification (like sampling). It maintains a global pool of reasoning strategies and uses a momentum-based mechanism to trigger exploration when solution similarity indicates stagnation, evolving new strategies via crossover and mutation. Experiments on multiple code-generation benchmarks show consistent Pass@K gains (≈+1–4pp) and higher diversity compared with baselines such as SFS and BoN.

### Strengths
1. The paper presents an interesting combination of goal-oriented (MCTS) and process-oriented (diversity-based) methods, addressing fundamental limitations of each paradigm when used in isolation.
2. The evaluation spans multiple challenging benchmarks (APPS, HumanEval, MBPP+, LeetCode, CodeContests) with consistent improvements demonstrated across different search methods.
3. The method's ability to integrate with existing search frameworks (BoN, Tree Search, GA, SFS) without structural modifications is practically valuable.

### Weaknesses
1. The strategy pool appears domain-specific and manually designed. The generalizability to other domains beyond code generation is questionable.
2. The paper lacks detailed analysis of the computational overhead introduced by the momentum mechanism, particularly the repeated similarity computations.

### Questions
1. How would the strategy pool be adapted to non-coding domains? Is manual strategy design always necessary?
2. Integrating EvoMo presumably requires modifying the search loop, node data structure, and prompt construction logic. How large is this engineering overhead in practice? Did the authors quantify the actual gain after integration relative to the added complexity or runtime cost for each baseline (BoN, SFS, MCTS)?
3. Since EvoMo sometimes uses longer prompts and extra similarity computation, do gains still hold under strictly token- or time-normalized budgets?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes EvoMo, a test-time (inference-time) scaling method that aims to unify goal-oriented search (e.g., MCTS/ToT) with process-oriented diversification (e.g., Best-of-N). Concretely, the authors (i) embed a global strategy pool of “reasoning strategies” into MCTS via an (\epsilon)-soft selection policy, and (ii) introduce a momentum-inspired controller that monitors inter-solution similarity and, when a threshold is exceeded, forces exploration of under-used strategies and triggers lightweight “evolution” (crossover/mutation) of strategies. Experiments on coding and reasoning benchmarks (e.g., APPS, CodeContests, HumanEval, LeetCode, MBPP+) report modest to moderate gains in Pass@K over baselines (BoN, MCTS, SFS), with larger improvements at higher iteration budgets.

### Strengths
- Empirical gains: The method reports consistent improvements on several benchmarks under fixed search budgets, sometimes reaching the strongest Pass@K among compared methods; e.g., on CodeContests and APPS under 40 iterations, EvoMo surpasses BoN, MCTS, and SFS variants. These results suggest the approach can be a practical plug-and-play enhancer for existing test-time pipelines. 
- General framing: Positioning a strategy pool inside a search controller is a clean way to expose “reasoning modes” as first-class actions. The idea could generalize beyond coding to other inference-time search settings. 
- Scalability intuition: The claim that diversity helps escape sequential traps in tree-like search is plausible, and the paper provides curves suggesting that benefits grow with more iterations. 
- Attempted analysis: The paper sketches a theoretical perspective arguing that similarity-triggered “momentum” can help the strategy action space approach a near-optimal pool over time.

### Weaknesses
1. Originality / motivation not crisp:

The core components—MCTS with policy over “reasoning strategies,” an evolving pool, and a momentum-style trigger based on solution similarity—feel like a direct stitching of existing ideas (tree/beam/MCTS search, evolutionary pools, and optimization-inspired momentum/diversity controllers). The paper does not clearly isolate a singular conceptual contribution beyond combining these ingredients, nor does it convincingly argue why this particular combination is necessary or superior to simpler diversity controllers applied to SFS or BoN. 

2. Clarity / specification gaps: Several important mechanisms are underspecified, inhibiting reproducibility and interpretability:
- Similarity metric: The momentum trigger hinges on measuring the semantic similarity (\Phi(\cdot\Vert\cdot)) between generated solutions, yet the paper does not concretely define which representation(s) and distance(s) are ultimately used in the controller (beyond listing options later for analysis); thresholds, normalization, and how multi-metric signals are combined for the actual trigger are not made precise in the main text. 
- Momentum details: The “momentum” mechanism is described at a high level (inject under-used strategies; crossover/mutation of strategies; prompt augmentation), but the exact update rules, scheduling, parameter settings ((\epsilon), (\theta_{\text{sim}}), window sizes (k), tie-handling), and ablation isolating momentum vs. basic diversity are not clearly presented in the main paper. 
- What exactly is a “strategy”? The taxonomy, initialization, and evolution operators for strategies (and how they map to concrete prompts/tool uses) are not fully specified; it is unclear which parts are hand-engineered vs. learned, and how much human tuning is required for each domain. 

3. Experimental coverage appears insufficient:

- Backbone choices and sizes: The main coding experiments rely heavily on a single or very limited set of LLM backbones (e.g., GPT-4o-mini is mentioned repeatedly). It’s unclear what base models are used across all tables, how large they are, and whether results hold on 30B+ class open-weight models (or stronger closed ones) to support claims about inference-time scaling at realistic capability tiers. The paper also suggests some multi-LLM runs but does not systematically explore multiple families and sizes under the same protocol. 
- Baselines vs. SFS: From Table 1, the strategy pool alone sometimes yields limited gains over SFS, suggesting the improvements may be sensitive to settings and not always substantial (e.g., marginal Pass@K increases on certain datasets). A more thorough analysis of when/why EvoMo helps (or fails) is missing. 
- Compute / cost accounting: Since the method’s value proposition is test-time scaling, the paper should provide detailed token/latency budgets, variance across runs, and cost-normalized comparisons (e.g., improvements per token or per second) to tease apart algorithmic gains from simply “more tries.” 

4. Positioning relative to prior momentum/diversity controllers:

The paper acknowledges related work that adds diversity or multi-agent search at inference time, but it does not clearly differentiate its “momentum” trigger and evolution step from prior diversity-on-plateau heuristics. Without sharper contrasts or controlled ablations against simpler triggers, the incremental contribution of “momentum” remains unclear.

### Questions
1.	Similarity & Triggering

- What exact similarity function(s) (\Phi) feed the trigger in the core algorithm (not only diagnostic plots)? If multiple metrics are used, how are they aggregated (weighted average? learned combiner? max?) and what are the precise thresholds and window (k)? Please provide pseudocode for the trigger. 

2.	Momentum Mechanics

- How is “momentum” formally implemented? What are the update rules, scheduling, and hyperparameters? How do you pick which under-used strategy to inject, and how do you combine it with the best strategy (the “evolve” operator) deterministically and reproducibly? A step-by-step algorithm box would help. 

3.	Strategy Pool Definition

- Please enumerate the initial strategy set, give one-line operational definitions (e.g., prompt templates/tooling), and describe crossover/mutation operators with examples. How much manual engineering or task-specific tailoring is required? 

4.	Backbones & Scaling

- Report results across multiple model families and sizes, including ≥30B open-weight models where feasible, under identical budgets. Do gains persist or grow with stronger backbones? Provide variance across random seeds. 

5.	Cost/Benefit Analysis

- Provide token-normalized and latency-normalized comparisons against SFS/BoN/MCTS. Where is EvoMo most cost-effective? Include Ablations that isolate: (i) strategy pool only, (ii) momentum only, (iii) simpler “diversity-on-plateau” heuristics, (iv) your full method. 

6.	When does it help?

- Table-by-table analysis where the strategy pool delivers limited gains over SFS: what characteristics of tasks correlate with small vs. large improvements (e.g., solution length, unit-test density, reward sparsity)? Could task-aware trigger thresholds help? 

7.	Theoretical claims

- The analysis sketches terminality/near-optimality of the evolving pool. What assumptions on reward smoothness, trigger firing frequency, and pool capacity are needed? Can you include a finite-budget bound or a practical stopping criterion consistent with the empirical budgets?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors propose an inference-time scaling framework (they call it EvoMo) that tries to combine the benefits of (i) goal-oriented search methods (e.g., tree or line search and MCTS) and (ii) process-oriented diversification (e.g., Best-of-N and evolutionary search). Their key idea is to embed a global, evolving strategy pool (a set of distinct thinking modes) into the expansion step of MCTS. And also to further mitigate entrapment in local optima, the authors incorporate a momentum-inspired mechanism that monitors similarity among recently generated solutions. When a similarity threshold is exceeded, the system forces exploration by combining an under-used strategy with a strong one via simple evolutionary operators and injects the resulting evolved strategy for the next expansion. The intent is to re-diversify the search when it starts collapsing to near-duplicates. Detailed theoretical analysis is also provided by the authors to support their claims. The authors validate the effectiveness of EvoMo on APPS, CodeContests, LeetCode, HumanEval, MBPP+ and show certain effectiveness.

I'm in fact impressed by the abundant content provided in the appendix, which shows the authors' efforts to make it more understandable for the readers.

### Strengths
``S1``:  The global strategy pool and similarity-triggered momentum is an intuitive method that can potentially be incorporated into different search pipelines.

``S2``: The theoretical guarantees provided in Appendix A (When momentum meets search) support the authors’ claims. 

``S3``: The appendix is well organised and contains abundant details and information.

### Weaknesses
``W1``: Following ``S3``, in fact, I don’t think the paper is very well written. On the one hand, the motivation described in the introduction is not very clear and sharp. The authors claim their contributions as simply combining the strengths of goal-oriented and process-oriented methods. Personally, I may consider this contribution a bit incremental. Is there any other alternative method to achieve this target? Why the proposed one is better? Clarifying these points would greatly strengthen the clarity and contributions of this paper. On the other hand, abundant details are given in the appendix. If possible, I would suggest the authors move some of the content in the appendix to the main paper.

``W2``: For the experiments, most gains are on code datasets. The framework should also generalise to other broader non-code evaluations. This would strengthen the “general test-time scaling” claim.

``W3``: It would be interesting to discuss some extensions of EvoMo such as incorporating PRMs.

### Questions
``Q1``: Is it possible for EvoMo to incorporate process-level reward models (that is, PRMs) to bias strategy selection earlier in the tree?

``Q2``: Are there alternative potential approaches to combine the strengths of goal-oriented and process-oriented methods? How EvoMo outperforms these potential solutions?

### Soundness
3

### Presentation
2

### Contribution
3
