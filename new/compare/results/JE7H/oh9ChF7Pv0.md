---
job_id: cc191b39-3530-4c9f-ba10-e5c7de29f131
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: oh9ChF7Pv0.pdf
paper: Egg-SR: Embedding Symbolic Equivalence into Symbolic Regression via Equality Graph
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.  

## Topic Compatibility
Pass ✅.  
The paper proposes equality-graph based methods to accelerate symbolic regression via MCTS, DRL, and LLMs, with theory and experiments. This fits squarely within neurosymbolic & hybrid AI, reinforcement learning, optimization, and scientific ML, all core ICLR topics.

## Minimum Quality
Pass ✅.  
All required sections are present: Abstract, Introduction, Methodology, Experiments, Results (Tables 1–2, Figures 3–5), Related Work, and Conclusion. The work is technically nontrivial, theoretical claims are supported by detailed proofs in the appendix, and experiments are reasonably substantial. No obvious fatal methodological or statistical flaws or misuse of test data are apparent from the text.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no attempts to manipulate automated reviewing systems, no hidden prompts, and no suspicious formatting or instructions targeting AI reviewers.

---

# Expected Review Outcome:

## Summary

The paper introduces Egg-SR, a framework that embeds symbolic equivalence into symbolic regression via equality graphs (e-graphs). Egg-SR provides a grammar-aware e-graph module (Egg) and integrates it into three families of SR methods: MCTS (Egg-MCTS), deep RL (Egg-DRL), and LLM-based SR (Egg-LLM). Theoretically, the authors derive a tighter regret bound for Egg-MCTS and a variance-reduced policy gradient estimator for Egg-DRL; empirically, they show consistent NMSE improvements and modest overhead across several benchmarks, particularly on trigonometric and scientific-law datasets.

## Strengths

1. **Clear and well-motivated problem formulation (symbolic equivalence in SR).**  
   The paper articulates the redundancy induced by symbolic equivalence very clearly, with concrete motivating examples already in the abstract and Introduction (e.g., $\log(x_1^2 x_2^3)$ variants). Section 2’s definition of equivalence via rewrite systems and Equation (1) is precise and sits well within existing rewriting / formal methods literature.

2. **Technically careful adaptation of e-graphs to grammar-based SR, with solid exposition.**  
   Section 3.1 provides a concrete, implementable design of Egg for grammar-based expressions, including how e-nodes and e-classes map to grammar rules, and how equality saturation and extraction work.  
   - **Figure 1** effectively illustrates construction, matching, substitution, and merging on the toy example $\log(x_1^3 x_2^2)$, making the e-graph operations much easier to follow than the text alone. The visual distinction between e-classes (dashed) and e-nodes (solid) clarifies the structure.  
   - Appendix B.2 and B.3 give explicit code-level skeletons (e.g., `Rule`, `ENode`, `EGraph`) and rule tables (Table 3), which strongly supports reproducibility.

3. **Reasonable theoretical grounding for claimed benefits in MCTS and DRL.**  
   The regret analysis in Theorem 3.1 plugs Egg-MCTS cleanly into the “near-optimal branching factor” framework of Munos / Leurent & Maillard. The definitions around $\kappa$ and $\kappa_\infty$ (Definitions 1 and 3, Appendix A.2) are coherent, and the stated bound  
   \[
   \texttt{regret}_{\texttt{egg}}(n)=\widetilde{\mathcal{O}}\left(n^{-\frac{\log(1/\gamma)}{\log \kappa_\infty}}\right),\quad \kappa_\infty \le \kappa  
   \]  
   matches the intuition that merging equivalent nodes reduces the effective branching factor.  
   For DRL, Theorem 3.2 formally identifies the Egg-based estimator as a Rao–Blackwellization of the standard gradient (via Lemma 1 and the use of $q_\theta(\phi)=\sum_{\tau\in\mathcal{S}_\phi}p_\theta(\tau)$ in Equation (10)), which is a clean and recognizable statistical argument.

4. **Consistent empirical improvements across three quite different SR paradigms, with concrete figures and tables.**  
   - **Table 1** (trigonometric datasets) shows that Egg-MCTS improves median NMSE substantially over MCTS, especially on more complex settings such as (4,4,6) and (5,5,5) under noiseless data (e.g., 0.006 vs 0.144, 0.009 vs 0.147). Egg-DRL yields smaller but still systematic improvements over DRL in most settings.  
   - **Table 2** shows Egg-LLM improving IID/OOD NMSE on real scientific tasks for both GPT-3.5 and Mistral. The improvements are moderate but consistent, which is interesting given that the only change is richer equivalence-based feedback prompts.  
   - **Figure 3 (left)** indicates that Egg-MCTS explores substantially more nodes than vanilla MCTS over iterations for the “sincos(3,2,2)” dataset, supporting the claim that pruning equivalent subtrees frees budget to explore more distinct regions of the tree.  
   - **Figure 3 (right)** visualizes reduced variance in the DRL objective proxy $R(\tau_i)\log p_\theta(\tau_i)$ for Egg-DRL versus DRL; the shaded region for Egg-DRL is visibly narrower, consistent with the variance reduction story.

5. **Convincing case for space and time efficiency of the e-graph module itself.**  
   - **Figure 4** provides a very clear comparison of memory usage between array-based storage and Egg on families of expressions with $2^{n-1}$ equivalent variants. The log-scale plots show array-based memory exploding while Egg’s memory grows slowly with $n$, making the asymptotic sharing argument tangible.  
   - **Figure 5** decomposes runtime in Egg-DRL into four components and shows that EGG’s contribution is small compared to coefficient fitting and neural updates, for both LSTM and Transformer decoders. This alleviates concerns that e-graph saturation might dominate runtime.

6. **Good breadth of integration and discussion of scope.**  
   The paper does not just bolt Egg onto one baseline. It systematically integrates with:  
   - MCTS via equivalence-aware backpropagation (Section 3.2, “Egg-based Backpropagation” and **Figure 2** pipeline),  
   - DRL via a modified gradient estimator (Equations (3)–(4), **Figure 8**), and  
   - LLM-based SR via equivalence-augmented feedback (**Figure 9**).  
   Section 3.3 candidly discusses other architectures (SymNet, Transformer-based E2E SR) and sketches how Egg might be used there, highlighting open questions rather than over-claiming.

7. **Implementation transparency and reproducibility.**  
   The authors provide a Python implementation of Egg, detail how rewrite rules are encoded at the grammar level, and explain their integration with baselines (MCTS, DRL, LLM-SR) in Appendix B and C. The explicit pseudocode for `apply_rewrite_rules` / `equality_saturation` and the discussion of using BFGS, fixed seeds, and dataOracles are all good practice.

## Weaknesses

1. **The DRL variance-reduction theory assumes full equivalence classes, but the algorithm actually samples only a subset, so the estimator is generally biased.**  
   - Theorem 3.2 and Appendix A.3 derive unbiasedness and variance reduction under the assumption that $g_{\texttt{egg}}(\theta)$ uses  
     \[
     q_\theta(\phi) = \sum_{\tau\in\mathcal{S}_\phi} p_\theta(\tau)
     \tag{10}
     \]  
     i.e., sums over *all* sequences in an equivalence class $\mathcal{S}_\phi$.  
   - However, the practical algorithm in Section 3.2 uses only the original sampled sequence $\tau_i$ plus $K-1$ sampled equivalents $\{\tau_i^{(2)},\dots,\tau_i^{(K)}\}$ from the e-graph and then computes  
     \[
     g_{\texttt{egg}}(\theta) \approx \tfrac{1}{N} \sum_i (\texttt{reward}(\tau_i)-b') \nabla_\theta\log\Big[\sum_{k=1}^K p_\theta(\tau_i^{(k)})\Big],
     \tag{4}
     \]  
     which is *not* equal to $\nabla_\theta \log q_\theta(\phi)$ unless $K$ covers the full $\mathcal{S}_\phi$ or they reweight by conditional probabilities.  
   - There is no discussion of this discrepancy between the theoretical estimator (over complete equivalence classes) and the approximate estimator (over a sampled subset), nor any bias analysis in terms of $K$ and the sampling scheme. As stated, the unbiasedness claim does not hold for the implemented algorithm. This is a central theoretical inconsistency that should be explicitly acknowledged and either corrected (e.g., with an importance-weighted estimator) or reframed as approximate variance reduction.

2. **MCTS analysis relies on strong assumptions and idealized settings that are not well matched to the actual implementation.**  
   - The regret bound in Theorem 3.1 is derived in Appendix A.2 using the OPD algorithm analysis and a deterministic finite-horizon MDP with discount $\gamma<1$. But Section B.4.1 states that the discount factor is set to $1$ in practice (“we set the discount factor to 1”), which violates the $\gamma\in(0,1)$ assumption used in Theorem A.1 and Definition 1.  
   - Moreover, the analysis assumes perfect detection and merging of *all* equivalent nodes, so that the effective branching factor is truly $\kappa_\infty \le \kappa$. In the real system, EGG samples only a finite number of equivalent sequences from the saturated e-graph and checks whether the tree contains these paths (Section 3.2, “Egg-based Backpropagation”). There is no guarantee that all equivalent paths are found and merged, or that cycles / depth truncation do not break the MCGS equivalence.  
   - As a result, the regret “tightening” is more of a qualitative transfer of existing MCGS theory than a rigorous guarantee about this concrete implementation. The paper would benefit from clearly delimiting the gap between theory and practice (e.g., making explicit which parts are asymptotic / ideal and which are approximations).

3. **Limited experimental scope and missing key SR baselines.**  
   - The main quantitative results focus on: (i) trigonometric synthetic datasets from Jiang & Xue (2023) for MCTS / DRL (Table 1 and **Figure 3**), and (ii) four scientific benchmarks from LLM-SR (Table 2). These are relevant, but the comparison set is narrow.  
   - There is no comparison against widely recognized SR baselines that are not trivially sequence-based, such as AI-Feynman / AI-Feynman 2.0, SRBench’s best-performing methods, or recent differentiable / neural-guided approaches like SINDy-type or neural equations beyond Petersen et al. (2021).  
   - Since Egg-SR is sold as a “unified framework that enhances a class of modern SR algorithms,” it would be much more convincing to show that an Egg-augmented MCTS/DRL is competitive with the broader state of the art, not only relative to its own non-Egg counterpart.

4. **Evaluation metrics and analysis of success are somewhat shallow.**  
   - The experiments report median NMSE of the best expression per method, but they do not systematically report *exact-recovery* rates or structural similarity measures, which are standard in symbolic regression benchmarks. It is quite possible for two methods to have similar NMSE while one discovers the true functional form much more often.  
   - On the LLM side (Table 2), results are given only as NMSE with no breakdown of success rates per seed, nor any qualitative examples of expressions found by EGG-LLM vs. LLM-SR. Without an inspection of typical discovered formulas, it is hard to know whether Egg-LLM helps the LLM escape particular local patterns or simply slightly tweaks coefficients.  
   - **Figure 3 (right)** does not plot the actual gradient variance, but instead the variance of the quantity $R(\tau_i)\log p_\theta(\tau_i)$, which is only loosely related to the variance of the gradient estimator in Equation (3). This is an odd metric choice; a more direct empirical check of Var$(g(\theta))$ vs Var$(g_{\texttt{egg}}(\theta))$ would make the theoretical claims more concrete.

5. **Limited discussion and quantification of potential numerical/semantic issues from rewrite rules.**  
   - Section B.2 acknowledges that many rewrite rules (e.g., $\log(ab)=\log a+\log b$) are only valid on restricted domains and that applying them blindly can yield $-\infty$ / NaN values when data contain negative inputs. However, the main text of the paper does not quantify how often such domain violations occur, nor does it evaluate the performance impact.  
   - In symbolic regression contexts with noisy or real-world data, domain violations and numerical instabilities are common; an aggressive rewrite system can significantly distort search if many candidates become invalid. Robust handling of these cases (e.g., domain-aware rules, masking, or algebraically safe rewrites) is left entirely to future work.

6. **Rewrite rule set is hand-crafted, domain-biased, and its breadth is not analyzed.**  
   - The strongest gains in Table 1 happen on trigonometric datasets, where Table 3’s rich trig identities are very helpful. However, it is less clear how Egg behaves on more general SR benchmarks with heterogeneous operators and no strong trig structure.  
   - The paper does not provide ablations on the rule set itself: for example, what happens if only basic algebraic rules vs a full trig set are available? How does performance scale as more rules are added, given that saturation can blow up graph size? The only hint is the qualitative remark under **Figure 5** that more rules increase EGG runtime.  
   - Without such ablations, it is hard to disentangle the generic benefit of equivalence-aware learning from the very specific choice of identities tailored to the datasets.

7. **Clarity issues around some equations and notational consistency.**  
   - Equation (2) defines UCT, but the theoretical analysis in Section 3.4 explicitly abandons UCT in favor of OPD. This switch of planning algorithm is only mentioned later (Appendix A.2 notes UCT’s issues), which can mislead readers into thinking Theorem 3.1 applies to the UCT-style Egg-MCTS implementation described in Section 3.2 and **Figure 2**.  
   - In the policy gradient section, Equation (3) uses $b$ while Equation (4) uses $b'$, but there is no clear explanation of how $b'$ is chosen and how its variance effect compares to $b$. Since baseline choice is nontrivial in variance reduction, this omission weakens the practical guidance.  
   - The definition of the objective in Figure 3 (right) as “$R(\tau_i)\log p_\theta(\tau_i)$” is somewhat arbitrary and is not explicitly tied back to Equation (3) or (4). A clearer derivation would help.

Overall, these weaknesses do not undermine the qualitative conclusion that e-graphs are helpful in SR, but they do limit the theoretical rigor and breadth of the empirical support.

## Potentially Missing Related Work

1. **M. Cranmer, A. Sanchez-Gonzalez, P. Battaglia, “Discovering symbolic models from deep learning with inductive biases”, 2020.**  
   - This work learns neural models with inductive biases and extracts symbolic equations, directly relevant to the theme of combining learning with symbolic structure.  
   - It should be discussed in Section 4 (“Knowledge-Guided Scientific Discovery” or “Equivalence-aware Learning”), emphasizing similarities and differences in how inductive biases / symmetries are exploited versus Egg-SR’s equality-graph approach.

2. **L. Biggio, M. Lippi, M. Maggini, “Neural symbolic regression using mixed-integer programming”, 2021.**  
   - Proposes a hybrid neural + MIP approach for SR, focusing on efficient search in a combinatorial space. This is another way to control search complexity, analogous in intent to Egg’s reduction via equivalence.  
   - It should be cited in the Related Work section when discussing modern SR methods that reduce search complexity (around the discussion of MCTS / DRL baselines), and briefly compared experimentally or conceptually.

3. **G. Martius, C. H. Lampert, “Extrapolation and learning equations”, 2017.**  
   - Focuses on learning equations that extrapolate beyond training data, central to evaluating SR methods on physical systems.  
   - This is relevant to the OOD aspects in Table 2 (IID vs OOD NMSE); it should be cited in the section describing scientific benchmarks and potentially in Section 4 as prior work on extrapolative SR.

4. **M. Cranmer, S. Greydanus, S. Hoyer et al., “Lagrangian neural networks”, 2020.**  
   - Incorporates Lagrangian mechanics into neural architectures to learn physical laws, another instance of injecting structured prior knowledge.  
   - It should be referenced in Section 4’s “Knowledge-Guided Scientific Discovery” subsection, contrasting physics-informed architectures with Egg-SR’s algebraic equivalence prior.

## Questions

1. **On DRL unbiasedness vs practical approximation:**  
   In Equation (4) and Theorem 3.2, the unbiasedness proof assumes sums over all trajectories in an equivalence class (via $q_\theta(\phi)$). In practice, you only sample $K-1$ additional equivalent sequences from the e-graph per original sample. Can you clarify whether the implemented estimator is unbiased, and if not, provide a bias analysis or an importance-weighted variant that would preserve unbiasedness for finite $K$?

2. **Discount factor inconsistency in MCTS analysis:**  
   The regret analysis assumes $\gamma\in(0,1)$, but B.4.1 sets the discount factor to 1 for the SR MDP. How do you reconcile this discrepancy? Is there a version of the regret bound (Theorem 3.1) that applies with $\gamma=1$, or is the theorem intended more as a qualitative justification than a formal guarantee for your implementation?

3. **Domain constraints and numerical robustness of rewrite rules:**  
   Do you have any statistics on how often rewrite rules lead to invalid expressions (e.g., NaNs due to $\log$ of negative numbers) in your experiments? Did you observe cases where aggressive rewriting degraded performance, and if so, how did you mitigate that? Some quantitative analysis here would be helpful.

4. **Ablation on rewrite-rule sets and generality beyond trigonometric identities:**  
   Could you add or at least describe ablations where you vary the richness of $\mathcal{R}$ (only algebraic vs algebraic+trig, etc.), and report how that affects the gains in Table 1 and Table 2? This would clarify to what extent Egg’s benefits are generic vs heavily dependent on domain-specific identities.

5. **More direct evidence of gradient variance reduction:**  
   Figure 3 (right) plots variance of $R(\tau_i)\log p_\theta(\tau_i)$. Can you provide results on the actual empirical variance of the gradient estimator $\hat g(\theta)$ vs $\hat g_{\texttt{egg}}(\theta)$ (e.g., squared norm variance across minibatches), to better connect with Theorem 3.2?

6. **Comparisons to broader SR baselines:**  
   Is there a reason AI-Feynman / modern SRBench leaders or hybrid methods (e.g., Cranmer et al. 2020) were not included in the evaluation? Even a small-scale comparison on one or two datasets would help position Egg-SR’s practical effectiveness against the wider SR ecosystem.

7. **LLM-LLM qualitative analysis:**  
   For Table 2, could you provide a few representative expressions discovered by LLM-SR vs Egg-LLM, highlighting cases where equivalence-augmented feedback changed the qualitative form of the discovered law? This would make the LLM-side story more convincing.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A. The work uses public scientific datasets and standard ML methods; no apparent concerns around privacy, safety, or human subjects.

## Soundness Rating

3: good.  
The core ideas are technically sound; the e-graph construction is well implemented, and the qualitative empirical findings are consistent. However, the DRL variance-reduction proof does not fully match the implemented estimator, and the MCTS regret bound relies on assumptions (full equivalence merging, $\gamma<1$) that are not fully met in practice.

## Presentation Rating

3: good.  
The paper is generally clear, with helpful figures (especially Figures 1–5, 8–9) and detailed appendices. Some theoretical assumptions and algorithmic choices (OPD vs UCT, exact vs approximate $q_\theta(\phi)$) are not clearly aligned, and the description of the DRL variance plots could be sharper.

## Contribution Rating

3: good.  
The paper makes a meaningful contribution by unifying equality-graph based symbolic equivalence with three major SR paradigms and backing it up with both theory and experiments. While not completely exhaustive in baselines or ablations, it advances the field in a direction likely to be useful for future SR work.

## Overall Rating

8: Accept, good paper (poster).  
Despite some theoretical and experimental limitations, the work offers a coherent and practically useful framework for embedding symbolic equivalence into SR, demonstrates consistent empirical gains, and provides nontrivial theoretical insight. The idea of using e-graphs as a unifying abstraction across MCTS, DRL, and LLM-based SR is sufficiently impactful and well developed to merit acceptance.

## Reviewer Confidence

4: confident.  
I am familiar with symbolic regression, RL-based search, and equality-graph methods, and I have read the math and implementation details carefully. Some empirical nuances (e.g., exact rule coverage on all datasets) are naturally hard to fully verify from the paper alone.