Now I have enough calibration data. Let me synthesize my review.

**Calibration summary:**
- "Transformers Provably Solve Parity Efficiently with CoT" (oral, scores 8,8,10): Strong theoretical contribution with clean proofs, limited empirical validation, but novel and significant for understanding CoT.
- "Expressive Power of Transformers with CoT" (poster, scores 6,8,8,8): Strong theoretical paper on CoT expressiveness.
- "Positional Description Matters for Transformers Arithmetic" (reject, scores 5,3,3,5): Similar domain (arithmetic + ordering), weak experiments, limited novelty, poor writing.
- "Arithmetic with Language Models" (reject, scores 5,6,3,3): Simple synthetic tasks, limited generalizability claims, weak contribution.
- "Arithmetic Transformers Can Length-Generalize" (poster, scores 5,8,5,8): Novel scratchpad approach, clear improvement, task-specific but well-scoped.

The current paper sits somewhere between "Positional Description Matters" (rejected) and "Arithmetic Transformers Can Length-Generalize" (accept poster). It has a novel problem formulation and methodology, but the tasks are contrived and the claims are overstated.</think>Now I have sufficient calibration data. Let me compose the final review.

## Summary

This paper introduces the task of "unraveling the chain of thought"—automatically discovering learning-friendly orderings of decoder output tokens for Transformers learning arithmetic tasks. The method trains a small Transformer on a mixture of target sequences in different orders, identifies those with faster early-stage loss drops ("loss profiling"), and uses a two-stage hierarchical search (global block-level then local token-level) to navigate the factorial permutation space. Experiments on three synthetic order-sensitive arithmetic tasks (RELU, SQUARE-19, INDEX) and a multiplication task (PROD) show that the method recovers known favorable orderings, improving success rates from ~10% to near 100%.

## Strengths

1. **Novel and well-motivated problem formulation.** The idea of systematically optimizing the output token order for autoregressive Transformers is genuinely new. While Shen et al. (2023) heuristically identified that reverse-digit order helps multiplication, no automated procedure existed. The formalization as optimization over permutations in $S_L$ (Eq. 3.2) is clean and precise.

2. **Principled method grounded in known learning dynamics.** The loss profiling approach leverages the well-established "easy-to-hard" learning phenomenon (Arpit et al., 2017). The key empirical observation—that learning-friendly orders produce faster loss drops early in training (Figure 3)—is clearly demonstrated and intuitive.

3. **Successful recovery of known results.** The method rediscovers the reverse-digit order for multiplication from Shen et al. (2023) (Table 2, PROD rows), which serves as a convincing sanity check. The dramatic success rate improvement from ~10% to near 100% on RELU and SQUARE-19 is substantive.

4. **Reasonable scalability strategy.** The hierarchical global-local approach is a pragmatic solution to the factorial search space. The method handles up to L=13 (13! > 6×10⁹) with random initialization and L=40 with structured initialization, with manageable computational cost (1–7 hours on a single GPU).

## Weaknesses

### Major:

1. **Narrow and partially circular task design limits generality claims.** The three custom tasks (RELU, SQUARE-19, INDEX) are explicitly constructed so that the forward order is trivially learnable (due to causal recurrence with non-injective maps) while almost any other order destroys identifiability (Section 5.1). This makes the evaluation somewhat self-fulfilling: the tasks are built to be maximally order-sensitive, and the method finds the expected order. As stated in the paper: "Any disruption of the natural left-to-right order—such as reversing or randomly permuting the targets—breaks the causal chain and substantially increases the learning difficulty." This means the evaluation primarily shows recovery of a known privileged causal order in specially-constructed recurrences, not general discovery of non-obvious reasoning orderings. The paper's framing as "unraveling the chain of thought" and "enhancing a Transformer's reasoning ability" (Abstract, Section 6) substantially overreaches what the evidence supports for general reasoning tasks, real chain-of-thought data, or settings with multiple valid reasoning pathways.

2. **Gap between stated optimization objective and the implemented algorithm.** The formal objective in Eq. 3.2 seeks the permutation minimizing the risk of a model trained to convergence on that permutation alone. The actual algorithm (P1–P2) selects the permutation with lowest early validation loss after training on a *mixture* of permutations with a shared model. The paper provides only Figure 5(b)—a correlation between early mixed-order loss ranking and final success rate on a small subset of permutations—as evidence that these two converge. This gap is never explicitly acknowledged. Mixed-order training may alter learning dynamics in ways that do not generalize to single-order training, and the paper lacks systematic evidence that the proxy objective aligns with the true objective beyond anecdotal examples.

3. **Missing baselines for the search procedure.** For the central algorithmic claim—loss profiling plus hierarchical search—there are no baseline search strategies (random search with full training, evolutionary search, greedy local swaps, etc.). Given that Table 1 shows success rates of 99–100% for forward order vs. 0–10% for reverse, the tasks have such extreme order disparities that even naive search could identify good orders. Section 5.4 shows that simply including the forward order among 127 random permutations and ranking by 1-epoch loss suffices to select it—which suggests the main challenge is *not* combinatorial complexity in these tasks, but rather detecting a dramatically easier order among poor ones. Without baselines, it is impossible to assess whether the elaborate hierarchical pipeline adds value over simpler alternatives.

### Minor:

4. **INDEX task results show partial failures without adequate analysis.** For INDEX with d=4 and d=8, the method does not recover the forward order (Table 2: final orders still deviate substantially). The paper mentions a "flattened loss landscape" but provides no quantitative analysis (e.g., loss differences between candidates, correlation between loss rank and true optimality). This leaves the reader unable to assess whether the method is finding good local optima or simply failing.

5. **Strong dependence on structured initialization for scalability.** With random initialization $P_r$, the method scales only to L=13. Scaling to L=40 requires structured initialization $P_b$, which encodes domain knowledge (block-level permutations). For genuinely novel tasks where no structure is known, the method's scalability is limited, and this is acknowledged only briefly in the conclusion as "extension to longer sequences... will be future work."

6. **No validation that small-model orders transfer to large models.** The method discovers orders using a 1-layer, 1-head model but deploys them on a 6-layer model. The assertion that "learning-friendly orders must be universal" (Section 4, computational overheads) is stated without empirical validation. If orders found by small models are suboptimal for large models, the entire pipeline is undermined.

### Trivial:

7. The formal notation for the hierarchical method (Eqs. 4.2–4.4) is somewhat dense and could benefit from algorithm pseudocode for full reproducibility, though the code is promised.

## Nice-to-Haves

- Testing on at least one task where the optimal ordering is genuinely unknown (not constructed to have a known forward order) would substantially strengthen the paper's claim to "discovery."
- Comparison with alternative search baselines (random search, Gumbel-Sinkhorn with stronger regularization, evolutionary search) to isolate the contribution of loss profiling.
- Validation that discovered orders transfer across model sizes and architectures.
- Out-of-distribution generalization experiments (longer sequences, different input ranges) beyond the fixed-length in-distribution evaluations currently presented.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Overstated search-space claims vs. actual explored set"** (Harsh Critic Point 3): The paper correctly states it searches through candidate sets of size up to T=7! per run with hierarchical pruning. The claim "identifies a learning-friendly order out of a few billion candidates" (Abstract) refers to the total size of the permutation space being searched, not that all 13! permutations are evaluated. This is standard in combinatorial optimization. However, the related concern about whether the method scales when many near-equivalent orders exist (rather than one clearly dominant one) is valid and retained as part of Point 1.

- **"No comparison to state-of-the-art techniques for improving reasoning or length generalization"** (Harsh Critic Point 4 sub-point): This is scope creep. The paper is about reordering output sequences, not about positional encodings, abacus embeddings, or curriculum learning for training data. These are different approaches to different (though related) problems.

- **"No confidence intervals or error bars"** (Spark Point, implicit): Single-run evaluation is the norm for large-scale Transformer benchmark experiments. The 1,000-sample evaluation set provides sufficient statistical power for the binary success metric being used.

- **"Out-of-distribution generalization claim is unsupported"** (Spark Point 2): The paper's abstract claims "generalizable to out-of-distribution samples," but this is directly supported by the PROD (multiplication) experiment which recovers the reverse-digit order known to improve length generalization (Shen et al., 2023; Figure 1). This is not a fabricated claim. However, the lack of explicit OOD experiments on the three custom tasks is a fair concern and is noted in Nice-to-Haves.

- **"Formatting/style nitpicks"**: Removed as per rules.

## Novel Insights

The key insight of this paper—that early training dynamics on a mixture of orderings can serve as a proxy for identifying learning-friendly token orderings—is genuinely useful and could extend beyond the narrow arithmetic tasks tested here. The hierarchical approach to searching factorial spaces mirrors multi-resolution optimization ideas in other domains (e.g., coarse-to-fine grid search). However, the current evaluation's reliance on tasks with a single privileged causal direction leaves open the more interesting question: in tasks where *multiple* reasonable orderings exist, can the method identify *which* orders are better and by how much? That would be the true test of "unraveling chain of thought."

## Suggestions

1. **Add at least one baseline search method** (e.g., random search with full training, or greedy pairwise swap) to isolate the contribution of loss profiling vs. simply having any search strategy in an easy task landscape.

2. **Include a task where no single "correct" order is known a priori**, such as a multi-step symbolic computation with multiple legitimate orderings, to demonstrate genuine discovery capability rather than recovery of built-in structure.

3. **Explicitly acknowledge the gap** between the formal objective (Eq. 3.2) and the practical algorithm (mixed-order early loss proxy), and provide ablation studies on the number of profiling epochs and mixed-permutation set size.

4. **Report success rates for non-standard discovered orders** (e.g., INDEX d=4, d=8 in Table 2) rather than only showing the permutation indices, so readers can assess whether partially correct orderings yield meaningful improvements.

## Score and Decision

**Calibration comparison:**

- **"Positional Description Matters for Transformers Arithmetic"** (reject, avg ~4.0): Similar domain (arithmetic + ordering), but weaker novelty (heuristic modifications to positional encoding), poorer writing, and less principled methodology. The current paper has a clearer contribution.

- **"Arithmetic with Language Models"** (reject, avg ~4.25): Simple synthetic tasks, limited generalizability claims, weak contribution. The current paper has more algorithmic novelty but similarly limited task scope.

- **"Arithmetic Transformers Can Length-Generalize"** (accept poster, avg ~6.5): Novel scratchpad approach, clear improvement, task-specific but well-scoped. The current paper's contribution is narrower and less thoroughly validated.

- **"Transformers Provably Solve Parity Efficiently with CoT"** (accept oral, avg ~8.7): Strong theoretical contribution with clean proofs. The current paper is purely empirical on very narrow tasks.

- **"Understanding Addition in Transformers"** (accept poster, avg ~5.5): Mechanistic analysis on one-layer models, similar to current paper's synthetic focus. The current paper has comparable limitations but more methodological novelty.

The paper poses an interesting question and proposes a reasonable first approach, but the evaluation is limited to contrived tasks designed to have a single privileged ordering, lacks baselines for the search procedure, and overclaims generality. The contribution is best viewed as a promising proof-of-concept rather than a general solution. This places it below the threshold for acceptance.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>