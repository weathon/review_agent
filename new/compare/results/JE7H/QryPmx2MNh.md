---
job_id: f40ae886-ad90-40d2-a99f-70ea054958b4
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: QryPmx2MNh.pdf
paper: Chain of Thought in Order: Discovering Learning-Friendly Orders for Arithmetic
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work studies how decoder token order affects learning in Transformer models on arithmetic tasks, proposes an algorithmic method to search permutations, and analyzes training dynamics and attention patterns. This sits squarely within representation learning, optimization, and reasoning with Transformers, which are core ICLR topics.

## Minimum Quality
Pass ✅.  
The paper is in English and has all major sections: Abstract, Introduction, Related Work, Method (Sections 3–4), Experiments (Section 5), Results (Tables/Figures), and Conclusion. The method is reasonably well specified, the experiments are nontrivial, and there are no obvious fatal methodological errors or test-set leakage. While there are weaknesses in scope, theoretical depth, and some clarity, they do not warrant desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any attempts to manipulate automated reviewing (no hidden prompts, no instructions addressed to reviewers, no suspicious formatting).

---

# Expected Review Outcome:

## Summary

The paper studies how the *order* of decoder target tokens affects the learnability of arithmetic tasks by Transformers. It proposes a *loss profiling* approach that trains a small Transformer for a few epochs on a mixture of permutations of the targets, ranks permutations by early validation loss, and then uses a hierarchical global–local search over block-level and intra-block permutations to find “learning-friendly” orders. On three synthetic order-sensitive arithmetic tasks (ReLU, Square-19, Index) and a multiplication task, the method reportedly recovers or approximates the forward / reverse-digit orders that make the task easy to learn and increases end-to-end success rates from around 10% to near 100% for sequence lengths up to 13, and further with structured initialization.

## Strengths

1. **Clear and interesting problem formulation.**  
   The paper isolates an under-explored but important axis: not just *what* intermediate steps are generated (chain-of-thought content), but *in what order* they are produced. The formalization via permutations on the target sequence in Section 3, with $\theta_{\mathrm{ERM}}^\pi$ in Equation (3.1) and the minimization over $\pi \in S_L$ in Equation (3.2), usefully frames the problem as searching over output-sequence permutations.

2. **Simple, task-agnostic search heuristic grounded in learning dynamics.**  
   The core idea of “loss profiling” (Section 4) leverages the empirically well-known easy-to-hard training dynamics: permutations that yield faster early loss drops are treated as learning-friendly. The procedure in Step 1–2 of Section 4 is simple to implement and does not require gradients through permutation parameters. Figure 3 nicely illustrates that even for two different orders, validation loss curves diverge early, which directly motivates using early loss as a search signal.

3. **Hierarchical global–local search for permutations.**  
   The global/local strategy (Figure 4, Equations (4.2)–(4.4)) is a sensible engineering choice to manage the factorial search space. The global stage permutes blocks, then the local stage refines token order within and between blocks. Table 2 shows that the global stage tends to recover coarse structure (neighbors in the original forward order often remain adjacent), and the local stage can refine this into near-forward orders in many cases. This decomposition is plausible and practically motivated.

4. **Well-designed synthetic tasks that make order effects very explicit.**  
   The ReLU, Square-19, and Index tasks (Section 5.1 and Equations (5.2)–(5.4)) are carefully constructed so that the forward order realizes a simple causal recurrence, while reverse/random orders destroy invertibility due to non-injective $f$. Example 5.1 concretely demonstrates the ambiguity in Square-19 when reversing the sequence. These tasks isolate the specific challenge of order without confounding factors and are conceptually clean.

5. **Empirical evidence that early loss is informative for order quality.**  
   Section 5.4 provides a nice sanity check: with one forward order and 127 random permutations in $\mathcal{P}_g$, Figure 5(a) shows that the forward order (ID = 0) consistently achieves the lowest validation loss across tasks. Figure 5(b) then correlates the rank (by early loss) with final success rates on ReLU and Square-19: higher-ranked orders tend to yield higher success, supporting the central “loss profiling” assumption.

6. **Rediscovery of previously known good order in multiplication.**  
   On the PROD task, the method finds the least-significant-digit-first order previously identified by Shen et al. (2023). This is an important sanity check that the search procedure is not overfitting weird synthetic tasks but can also align with prior heuristic insights on multiplication.

7. **Thoughtful negative result and analysis of soft-permutation optimization.**  
   The paper does not just claim that continuous relaxation fails; it actually runs a soft-permutation + attention sparsity regularization approach (Appendix B). Figure 2 and Figure 9 show that such soft permutation training causes information leakage (dramatic early loss drops) and does not converge to a hard permutation, even with an entropy-style regularizer. This strengthens the motivation for the discrete, loss-profiling approach.

8. **Use of figures and tables to support claims.**  
   - **Figure 1** provides a clear visual of the huge difference in success rates for multiplication in forward vs reverse output digit orders, motivating the whole work.  
   - **Figure 6(a,b)** shows that the discovered orders (yellow) achieve success rates comparable to forward (blue) and far better than reverse (red) across lengths and initializations.  
   - **Table 1** quantifies how drastically performance drops in reverse order for all tasks, evidencing that the tasks are indeed order-sensitive.  
   - **Table 4** comparing the proposed method with an evolutionary strategy baseline is particularly useful, showing that ES can sometimes find good permutations but degrades rapidly with length, which helps to position the proposed search method.

## Weaknesses

I will be quite explicit here, since the core idea is appealing but the paper has several limitations that affect how impactful and conclusive it is.

1. **Scope restricted almost entirely to toy arithmetic tasks; external validity unclear.**  
   All main experiments (Sections 5.3–5.5) are on the three synthetic recurrences and multiplication. These tasks are carefully designed but extremely structured, with short fixed-length outputs and simple deterministic mappings. There is no evidence that the approach would meaningfully help on more realistic sequence modeling: e.g., more complex algorithmic tasks, symbolic math beyond scalar recurrences, or natural-language reasoning with chain-of-thought. For instance, the method relies heavily on there being a *single global permutation that is beneficial for all samples*; in real-world reasoning, the “best” intermediate order may depend on the input or the specific derivation. The paper does not discuss this limitation or attempt any task where the target order might need to be input-dependent. This significantly limits the significance for the broader ICLR audience.

2. **Search objective tied only to early loss on IID validation, with no direct optimization for generalization or learning stability.**  
   The core assumption is that permutations with faster early loss drops lead to better final performance. While Figures 3 and 5 partially support this on the synthetic tasks, the paper does not probe when this assumption fails or how stable it is to hyperparameters (e.g., training duration $E$, batch size, learning rate). There is no ablation on $E$ in step 1 of Section 4: how short can training be while still reliably ranking permutations, and how sensitive are rankings to $E$? Additionally, the objective in Equation (4.1) is just empirical loss on in-distribution $D'$, but generalization to longer sequences or out-of-distribution settings is central in prior arithmetic work. The paper does not examine whether orders that minimize early *training/validation* loss also yield better length generalization, which is arguably the most interesting aspect of choosing reasoning orders.

3. **Hierarchical algorithmic description has ambiguities and some mathematical sloppiness.**  
   Several parts of Section 4 are imprecise or possibly inconsistent:
   - In Equation (4.2), $Q_i \in [0,1]^{L\times L}$ are said to be “block-level permutations”. For an actual permutation of blocks, $Q_i$ should be permutation matrices (entries in $\{0,1\}$, rows/columns summing to 1). Using $[0,1]$ here is confusing, especially given the earlier discussion of soft permutations. The same issue appears for $R_j^i$ in Equation (4.3).  
   - The description of the local stage says “Let $P_1 \in P_\mathrm{g}$ be the initial permutation.” This is a bit sloppy: earlier $P_\mathrm{g}$ is a single permutation, not a set. Presumably they mean “let $P_1 := P_\mathrm{g}$”, but this needs to be made precise for readers trying to re-implement the method.  
   - The range for block length $l$ is written as $l = \{2,3,\ldots,\lfloor L/2 \rfloor\}^{2}$, followed by a footnote “When $k=1$, the sequence is not split into blocks.” This notation is unclear: is $l$ squared, or is that a typo? What exactly is the schedule over $l$?  
   - In Appendix B, sparsity $S$ is defined via entropy in Equation (B.2), $S = -\frac{1}{L'}\sum_{i,j} a_{ij}\log a_{ij}$, but the optimization objective in (B.4) is $\min_{\tilde P} \frac{1}{L'} \sum_{i=1}^{L'}\sum_{j=1}^{L'} a_{ij}$, i.e., the *sum* of attention weights, which is constant at $L'$ because each row of $A$ is a probability vector. This is almost certainly a mistake: the objective as written is degenerate and provides no gradient signal about sparsity. If they intended to minimize entropy (as claimed in the surrounding text), Equation (B.4) should mirror (B.2) and involve $a_{ij}\log a_{ij}$. This suggests the math has not been carefully vetted, and it undermines the conclusions about “soft-permutation optimization via attention sparsity”.

4. **Theoretical story is shallow relative to the ambition of “unraveling chain-of-thought”.**  
   The paper mostly rests on empirical observations of early loss dynamics and attention sparsity, but does not provide any formal analysis or even simple toy theoretical results about when a permutation is “learnable” for a given recurrence. For example, for the recurrence in Equation (5.1) with a non-injective $f$, there is a strong structural reason why forward order is preferable: causal invertibility. This could potentially be formalized (e.g., characterizing when the reverse order is information-theoretically ambiguous or proving impossibility under limited context), but the paper stops at intuition and examples (Example 5.1). Similarly, there is no attempt to analyze why early loss curves differ for different permutations in terms of gradient flow or inductive bias, despite this being the linchpin of the method.

5. **Interpretation of attention sparsity and its relation to learnability is tentative and partly contradictory.**  
   Appendix A and B claim that learning-friendly orders lead to sparser attention (lower entropy $S$ in Equation (B.2)). Table 3 reports $S$ for forward vs reverse on multiple tasks and lengths and indeed forward orders often have lower $S$, but the differences are inconsistent (for Square-19 at $L=50$, forward actually has *higher* sparsity measure than reverse). **Figure 7** and **Figure 8** qualitatively show more localized attention for forward orders, but the paper then states in Appendix B that “attention sparsity…decreases even for static orders” and that entropy-based optimization did not help. The narrative ends up being: sparsity correlates somewhat with a good order but cannot be meaningfully optimized. This section is lengthy and somewhat confusing; it might be better to cleanly separate the descriptive observation (forward often yields more structured attention) from the failed attempt to use it as a differentiable objective.

6. **Limited and somewhat confusing story around the PROD (multiplication) task.**  
   Section 5.1 defines the PROD forward order as least-significant to most-significant digits (denoted $Y$), and the reverse as $Y^{\mathrm{r}}$. However, in Section 5.5 they say “it succeeds in rediscovering the least-significant-digit first order reported by Shen et al. (2023), and it finds the optimal order for target lengths up to 13.” This implies that the method finds what they have *defined* as the forward order, not some nontrivial new permutation, yet in Table 2 the POSD entry for $L=10$ just lists the identity order. The narrative around rediscovering Shen et al.’s finding could be much clearer: currently it sounds like a stronger result than what is actually demonstrated (which is: selecting between forward and reverse and some block permutations recovers the known best).

7. **Heavy dependence on initialization for longer sequences and missing quantitative analysis of failure modes.**  
   Section 5.5 distinguishes random initialization $\mathcal{P}_r$ vs structured block initialization $\mathcal{P}_b$. **Figure 6(a)** shows that, with $\mathcal{P}_r$, the method can find good orders up to $L=13$. **Figure 6(b)** shows that with $\mathcal{P}_b$ it scales to $L=30$ or $40$ on some tasks. However, there is no quantitative reporting of how often the method fails to recover the true forward order, or what the variance is across random seeds, especially in the harder Index task. The text notes that Index with large $d$ has a “flatter loss landscape” and that learning becomes difficult, but there is no systematic analysis of how robust the search is under different $T$, $K$, or block sizes. For a search method whose main claim is to navigate a factorial space, more rigorous reporting on success vs failure rates is important.

8. **Experimental design and reporting gaps.**  
   Several issues make it hard to fully trust the strength of the empirical evidence:
   - For **Figure 5(b)**, only the top 32 permutations are retrained, and Index results are omitted “because success rates are all close to zero.” This omission hides the fact that, for the hardest task, the early loss ranking did not translate into any meaningful performance differences at full training, which weakens the generality of the early-loss heuristic.  
   - There is no comparison to easier baselines like random search with more iterations using the *same* computational budget as the hierarchical method. Table 4 compares to an evolutionary strategy, but the hyperparameters and total compute for ES vs the proposed method are not carefully controlled or discussed.  
   - All main experiments use GPT-2–style architectures with specific dimensions (Section 5.2), but there is no investigation of whether the discovered “best order” is robust across architecture sizes, depth, number of heads, or positional encodings. The authors assert that “learning-friendly orders must be universal” (end of Section 4) but do not provide evidence beyond using a small model for search and a larger one for final training.

9. **Some confusion in notation and minor clarity issues.**  
   - In Section 3, the notation $f(x,y)$ is introduced and then used as $y_{i+1} = f(x_i + y_i)$, which suggests $f$ is actually a scalar function rather than a function of $(x,y)$. This is minor but contributes to a feeling of imprecise math.  
   - In multiple places permutations are denoted simultaneously as vectors “relative to forward” (e.g., Table 2) and as matrices $P$, without a bridging explanation. For a paper that is essentially about permutation combinatorics, some more careful exposition on how to interpret these indices would be helpful.  
   - A few typographical issues (e.g., “SQUARE-M19” vs “SQUARE-19” in Table 4) and occasional run-on sentences make reading slightly more difficult, though overall the writing is serviceable.

Overall, the core idea is interesting and empirically supported on carefully chosen synthetic tasks, but the work feels more like a solid workshop paper than a fully mature ICLR main-track contribution at this stage, mainly due to limited scope and some mathematical and methodological rough edges.

## Potentially Missing Related Work

1. **Wang, X., Wei, J., Schuurmans, D. (2022). “Self-Consistency Improves Chain of Thought Reasoning in Language Models.”**  
   This paper proposes self-consistency decoding over multiple chain-of-thought samples to improve reasoning, and directly speaks to *searching over reasoning trajectories* and their ordering. While the current paper cites Wei et al. (2022) on chain-of-thought, it does not discuss self-consistency or any methods that sample and rank multiple reasoning paths. This work should be cited in Section 2 (Related Work) as another example of optimizing over reasoning sequences, and discussed in Section 6 as a conceptual analog where one also searches over multiple candidate “orders” or trajectories and selects the best according to a score.

2. **Magister, L. C., Mallinson, J., Adamek, J. (2022). “Teaching Small Language Models to Reason.”**  
   This work transfers reasoning capabilities to small models, often involving structured intermediate steps and curriculum-like considerations. It is relevant because this paper also uses a small Transformer for exploration and then trains a larger model on the discovered order. It would be natural to reference it in Section 2 and compare in Section 5.2 when motivating why a small model suffices for permutation search and what assumptions are being made about transfer of “reasoning structure” across scales.

## Questions

These are points where a detailed rebuttal or additional experiments could significantly update my assessment.

1. **Clarify and correct the attention sparsity objective (Appendix B).**  
   - Is Equation (B.4) indeed a typo? Should it be minimizing entropy $-\sum a_{ij}\log a_{ij}$ rather than $\sum a_{ij}$, which is constant?  
   - If so, please correct the equation and clarify what was actually implemented. If not, please explain how $\sum_{ij} a_{ij}$ is non-constant given row-wise softmax normalization.

2. **Sensitivity of loss profiling to training duration and hyperparameters.**  
   - How sensitive are permutation rankings to the choice of $E$ (number of epochs/steps) in step 1 of Section 4? Can you provide a plot analogous to Figure 5(a) where you vary $E$ and show Kendall-$\tau$ correlation between rankings obtained with different $E$?  
   - Relatedly, does using a different optimizer or learning rate schedule materially change which permutations are ranked as best?

3. **Robustness across architectures and positional encodings.**  
   - You argue that orders are “universal” and can be found with a small model then used for a larger one. Can you add experiments where the search is done with a different architecture (e.g., different depth or fewer attention heads) and test whether the discovered order is still optimal for the 6-layer model?  
   - Have you tried alternative positional encodings (e.g., learned vs sinusoidal) in either the search or final training stage, and do they change which order appears learning-friendly?

4. **Generalization to longer lengths and OOD regimes.**  
   - For ReLU and Square-19, can you train on a given $L$ but evaluate on longer sequences in the same order (as Charton, Jelassi et al., and Shen et al. do) to see if the discovered permutation improves *length generalization* relative to forward or reverse? This would connect more directly to the literature on arithmetic transformers.  
   - If not feasible due to data generation limits, please clarify why and discuss how you expect your method to behave in such settings.

5. **Clarify the search space and number of training runs for PROD.**  
   - In Section 5.5 you state that the method finds the optimal order for $L \le 13$, “identifying a single solution among roughly $13!$ possibilities.” Given the hierarchical pruning, what is the actually explored subset of permutations for PROD?  
   - Could there exist alternative permutations (e.g., that interleave digits in some structured pattern) that are as good as or better than least-significant-digit-first but are effectively pruned out by your initialization? Some discussion or an empirical check would help.

6. **Provide more systematic statistics on success/failure rates.**  
   - For each task/length combination in Figure 6(a,b), how many random seeds (of the search procedure) were run, and what fraction recovered the true forward order or achieved success rate within, say, 1% of forward?  
   - A table summarizing these statistics would make the “6 billion candidates” claim more credible and give a clearer picture of robustness.

Addressing these questions would both clarify the correctness of the current claims and strengthen the case that the method is more than a neat heuristic on carefully constructed toy problems.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating
2: fair.  
The empirical methodology on synthetic tasks is mostly sound and carefully executed, but there are notable issues: the attention sparsity objective appears mathematically incorrect as written, the global–local algorithm is described somewhat ambiguously, and the validation of the early-loss heuristic is limited to a narrow domain without probing its stability or generalization properties.

## Presentation Rating
3: good.  
The paper is generally clear, well structured, and easy to follow, with helpful figures like Figures 1, 4, 5, and 6 and tables such as Tables 1, 2, and 4. However, several notational inconsistencies and ambiguous descriptions (especially in Section 4 and Appendix B) detract from clarity and should be cleaned up.

## Contribution Rating
2: fair.  
The conceptual contribution, “searching over target permutations using early loss profiling,” is interesting and somewhat original within the arithmetic Transformer niche, and the empirical results on synthetic tasks are convincing within that scope. However, the lack of experiments beyond toy recurrences, absence of theoretical development, and unclear applicability to real-world reasoning tasks substantially limit the broader impact.

## Overall Rating
4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The work contains a neat and well-argued idea, with nontrivial experiments and clear demonstrations of order effects on arithmetic recurrences and multiplication. At the same time, the scope is narrow, the theoretical story is shallow, and there are some concrete mathematical and methodological issues that need fixing. For ICLR main track, I would lean to a reject at this stage, while acknowledging that with stronger empirical breadth (especially on more realistic tasks) and clearer algorithmic/theoretical treatment, this line of work could become more impactful.

## Reviewer Confidence
4: confident.  
I am familiar with Transformers for arithmetic and chain-of-thought reasoning, carefully read the math and experimental sections, and feel confident in the identified strengths and weaknesses, though some empirical questions (e.g., sensitivity analyses) would benefit from additional author clarification.