# ICLR Benchmark Results

Date: 2026-04-12 20:30
Critic/Merger: gpt-5.4 (OpenRouter)
Neutral: ollama:glm-5.1:cloud, Related Work: qwen/qwen3.5-flash-02-23:online (OpenRouter)

## 1c73HCZpbo

- GT: Reject (avg 4.0)
- Predicted: N/A (4.1/10)
- Match: N/A

### Final Review

## Summary
This paper proposes REVEAL-IT, a framework that visualizes policy weight updates during RL training, trains a GNN predictor to estimate learning progress from those policy-update graphs, and uses a GNN explainer to highlight the most important updated regions. The intended payoff is twofold: interpret the agent’s learning process and use that information to optimize subtask sequencing during training. The idea of explaining *learning dynamics* rather than isolated actions is interesting, but the paper’s empirical validation much more convincingly supports a curriculum-style training mechanism than an interpretability contribution.

## Strengths
- **The paper targets an underexplored interpretability angle: explaining the *learning process* rather than only single decisions.** The framing in the introduction and method is specific: REVEAL-IT tries to connect policy evolution, subtask structure, and final performance, which is a more longitudinal perspective than standard action-level explanation methods.
- **The coupling between policy-update graphs and subtask sequencing is a concrete, nontrivial idea.** Algorithm 1 and Sec. 4.2 operationalize a pipeline where policy changes are turned into graphs, a GNN predictor estimates learning progress, and task sequences are adapted accordingly. This is more specific than a generic “visualize the network” paper.
- **The qualitative analysis is at least suggestive rather than purely decorative.** In Fig. 2 and Sec. 5.3, the paper attempts to identify overlaps in highlighted policy regions across related subtasks (e.g., shared regions for tasks involving the microwave or the apple), which gives some face-validity to the claim that common skills may correspond to shared updated substructures.
- **The OpenAI Gym results do suggest some sample-efficiency benefit across multiple RL backbones.** Table 2 reports several cases where PPO/A2C/PG with REVEAL-IT attains better return with reduced environment interaction budgets (e.g., 0.9x or 0.8x of the baseline budget), which supports the narrower claim that the framework can act as a useful training signal for task selection.

## Weaknesses

### Major:
- **The paper conflates two distinct contributions—interpretability and curriculum optimization—and the experiments mainly validate the latter.**  
  The strongest evidence in the paper is improved task performance (Tables 1–3), but those results primarily support that a learned signal based on policy-update graphs can help choose training tasks. They do **not** directly establish that the produced explanations are faithful, meaningful, or causally informative. This mismatch is visible in the paper’s own structure: in Sec. 4.2, “The learning objective of GNN predictor” explicitly drives task optimization via predicted learning progress, while the explainer is presented as highlighting important updates. The paper then uses downstream RL success as evidence for the explanation mechanism. By ICLR standards, a paper centered on interpretability needs evaluation aligned with interpretability claims.

- **The ALFWorld headline comparison does not isolate the proposed contribution.**  
  Table 1 compares REVEAL-IT against VLM/LLM agents and vanilla PPO, and the text claims that “REVEAL-IT demonstrates significantly better performance than other VLM agents.” But REVEAL-IT’s training setup includes explicit subtask sequencing and curriculum-style optimization, while the listed baselines are heterogeneous systems with different training pipelines and inductive biases. This means Table 1 does not tell the reader how much of the gain comes from the proposed explanation/predictor machinery versus the much stronger training decomposition itself. A comparison against RL agents with alternative curriculum/task-ordering strategies, or at least stronger same-backbone curriculum baselines, is needed to support the paper’s central empirical narrative.

- **The interpretability claims are not directly validated.**  
  The paper’s evidence for explanation quality is mainly Fig. 2 plus narrative interpretation in Sec. 5.3. The closest thing to supervision is in Sec. 4.2, Step 1: “the activated nodes in the policy will be tagged and utilized as the ground truth for the GNN explanation.” But the paper does not justify why evaluation-time activated nodes should be considered valid ground truth for which **training-time weight updates** explain eventual performance. Nor does it provide standard explanation-oriented tests such as sufficiency/comprehensiveness-style perturbation checks, stability analyses, faithfulness tests, or human-centered evaluation of whether the visualizations are actually understandable. Table 3 only shows downstream success after swapping explainers, which is useful as a pipeline ablation but still not a direct explanation-quality evaluation.

- **The method is underspecified at several points that matter for assessing the claims.**  
  The high-level pipeline is understandable, but important operational details are missing or too vague: how exactly \(G_O\) is built from policy updates; what node/edge features are used; how “activated nodes during evaluation” are thresholded/defined; the exact GNN predictor/explainer architectures; how the partition in Eq. (2) is trained in practice; and, especially, how task sequences/subtasks are defined in the non-ALFWorld settings of Table 2. This is not just a minor reproducibility complaint: because the empirical gains are large, these missing details make it difficult to judge what mechanism is actually responsible.

- **The paper’s “complex environments” claim is broader than what the experiments substantiate.**  
  The paper repeatedly argues that prior methods are limited in simple settings and presents REVEAL-IT as addressing explanation in complex environments. But the evidence consists mainly of ALFWorld plus standard control benchmarks in Table 2. The latter are useful sanity checks for algorithm-agnosticity, but they are not especially compelling evidence for interpretability in high-dimensional complex environments. As written, the breadth of the claim exceeds the empirical support.

### Minor
- **The roles of predictor and explainer remain conceptually muddled.**  
  Sec. 4.2 distinguishes them, but in the overall narrative the predictor appears to do the heavy lifting for training improvement, while the explainer is treated as central to the paper’s identity. The paper would be stronger if it more clearly separated “what improves learning” from “what provides human-understandable explanation.”

- **Table 2 contains meaningful information, but its presentation is too opaque.**  
  The parenthetical values are explained as training environment steps, and the table suggests that REVEAL-IT often obtains better returns with fewer steps. That is an interesting result, yet the presentation is not clear enough to let the reader quickly interpret whether the gain is from better sample efficiency, a different curriculum protocol, or some other budget mismatch.

- **The visualization approach may have limited scalability, and the paper does not discuss that limitation enough.**  
  The demonstrations focus on a relatively small actor MLP (4 layers, 64 nodes each). For much larger policies, raw node-link diagrams are likely to become hard to read, and the paper does not really address how the visualization component would scale.

### Trivial
- None.

## Nice-to-Haves
- Add a clean ablation that uses the **GNN predictor without the explainer**, and ideally the visualization/explainer without curriculum optimization, to disentangle which component contributes what.
- Provide side-by-side qualitative outputs for REVEAL-IT vs. the swapped explainers in Table 3, not just task success numbers.
- Include failure-case analysis for settings where REVEAL-IT does not help or slightly hurts performance in Table 2.
- Clarify the exact meaning of the training budgets in Table 2 and consider reporting learning curves or area-under-curve metrics.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper should prove generality over many more domains / only tested on 4 domains.”**  
  Removed/softened because the paper does test both ALFWorld and multiple standard control environments; the real issue is not simply number of domains, but that the experimental evidence does not fully support the broad interpretability claims.

- **Claims about nonexistent or questionable cited systems / release status / independent verification.**  
  Removed by policy. If the paper cites a benchmark/model/tool, it is treated as existing.

- **Purely generic reproducibility complaints about omitted hyperparameters.**  
  Removed in generic form. However, I retained the more substantive version where missing method details block interpretation of the core mechanism.

- **Requests for unrelated external baselines by name.**  
  I do not rely on naming specific missing literature. The valid underlying criticism is simply that the paper needs stronger same-problem curriculum/task-ordering baselines.

- **Reviewer import about dependence on \(\pi_{ref}\) / BC-derived policies.**  
  Removed as factually inapplicable to this paper; the submission does not use the referenced setup.

- **“Any online RL algorithm can be accepted” as inherently invalid.**  
  Softened. The issue is not that the claim is false, but that the empirical evidence for broad generality is limited.

## Novel Insights
The most interesting synthesis across the paper and reviews is that REVEAL-IT appears to contain the seed of **two different papers**: one on curriculum optimization from policy-evolution signals, and one on interpretability of policy learning dynamics. Right now, the first is much better supported than the second. The qualitative policy-overlap visualizations hint at a potentially valuable notion of “shared skill substructures” across subtasks, but the current experiments only show that this signal is *useful* inside training, not that it is a *faithful explanation* for humans. Reframing the contribution around policy-dynamics-informed curriculum design, with interpretability as a secondary qualitative lens, would make the paper read as more technically coherent.

## Suggestions
- **Reframe the contribution more honestly and sharply.** If the main validated result is improved subtask sequencing from policy-update graphs, say so directly, and present interpretability as a secondary qualitative benefit unless stronger explanation evaluation is added.
- **Add direct explanation-quality evaluation.** For example: test whether removing/highly perturbing highlighted subgraphs changes the predictor’s output more than removing non-highlighted regions; measure sparsity/stability; or run a small human study on whether the visualizations help diagnose learned skills.
- **Strengthen the ALFWorld baselines around the actual mechanism being proposed.** Compare against same-backbone RL systems with random curriculum, fixed handcrafted curriculum, or other task-ordering heuristics.
- **Disentangle predictor vs. explainer.** A predictor-only variant is particularly important, because the current results do not show whether the explainer contributes beyond the task-progress prediction signal.
- **Specify the graph construction and training protocol in full.** In particular: node/edge features, update frequency, architecture details, and how subtasks/task sequences are defined in each environment family.
- **Tone down or qualify the strongest claims about “complex environments” and “explanation.”** The current evidence does not fully support the breadth of those statements.

---

## 3lXZjsir0e

- GT: Reject (avg 5.6)
- Predicted: N/A (2.9/10)
- Match: N/A

### Final Review

## Summary
This paper studies offline robust learning in finite-horizon tabular two-player zero-sum Markov games under transition uncertainty. It proposes a model-based algorithm, RTZ-VI-LCB, built from empirical robust value iteration plus a pessimistic Bernstein-style penalty, introduces a clipped concentrability notion tailored to unilateral robust best responses, and proves uncertainty-dependent upper and lower sample-complexity bounds; it also sketches an extension to multi-player general-sum games.

## Strengths
- **The paper identifies and formalizes a nontrivial setting that combines three hard ingredients at once: offline learning, robustness to transition uncertainty, and self-play in Markov games.** The contribution is not just “robust RL” or “offline MARL” separately; the analysis is specifically for robust offline two-player zero-sum Markov games with partial coverage.
- **The sample-complexity dependence on state and action spaces is materially improved over the comparison highlighted in the paper.** Table 1 claims a shift from \(S^2AB\) to \(S(A+B)\), and the main theorem/lower bound indeed both scale in \(S(A+B)\), which is a meaningful structural improvement if correct.
- **The robust unilateral clipped concentrability coefficient is a specific conceptual contribution.** Assumption 1 is designed to avoid requiring full coverage of all state-action pairs by clipping occupancy ratios at \(1/(S(A+B))\), which is more nuanced than a raw worst-case density-ratio assumption and is central to the partial-coverage message of the paper.
- **The paper makes uncertainty-level dependence explicit rather than hiding it in constants.** The appearance of \(f(\sigma^+,\sigma^-,H)\) in the upper bound and the two uncertainty regimes in the lower bound are useful theoretical insights about when robust RTZMG learning is no harder than standard TZMG learning and when robustness introduces additional cost.
- **The lower-bound component is substantive and helps frame the contribution.** Theorem 2 is not decorative: it supports the claim that the \(S\) and \(A+B\) dependence is essentially unavoidable and clarifies the role of the uncertainty radius.

## Weaknesses

### Fatal
- **The paper’s core solution concept is not convincingly established under the asymmetric robustification it defines, and this undermines the meaning of the main guarantee.**  
  In Section 2, the max-player and min-player are evaluated in two different robustified problems: the max-player uses \(\inf_{P\in \mathcal U^{+\sigma^+}(P^0)}\) (Eq. 3, Eq. 9), while the min-player uses \(\sup_{P\in \mathcal U^{-\sigma^-}(P^0)}\) (Eq. 4). Eq. (10) then defines a single “robust NE” by requiring both unilateral optimality conditions simultaneously. That is not the standard saddle-point value of one robust zero-sum Bellman operator; it is an equilibrium notion assembled from two player-specific robust objectives, potentially with different uncertainty sets/radii.  
  The paper states after Eq. (9) that “there is at least one policy referred to as \(\mu^*\)... and \(\nu^*\)... [that] simultaneously achieve” the corresponding best-response values, and later says existence “has been proved for general divergence functions ... by Blanchet et al. (2024).” But in the main paper, there is no explanation of why this asymmetric formulation yields a coherent single game object, nor why the gap in Eq. (11) is the right exploitability notion for the returned pair. Since the main theorem is exactly about learning an \(\varepsilon\)-robust NE, this is not a peripheral omission.

- **Algorithm 2’s returned policy pair is not adequately justified as solving the stated equilibrium problem.**  
  The algorithm computes one Nash pair from \(\hat Q_h^+\) and another from \(\hat Q_h^-\), then outputs the cross-paired combination \((\hat\mu,\hat\nu) = (\{\mu_h^-\}, \{\nu_h^+\})\). This is explicitly in Algorithm 2. The main text does not explain why a max-player policy extracted from the “minus” recursion and a min-player policy extracted from the “plus” recursion should jointly form an approximate equilibrium of any single stage game or dynamic game.  
  This is the most serious algorithmic gap in the submission: even granting the definitions, the central construction of the returned policy is ad hoc from the main-paper presentation. Without a clear argument connecting the two separate recursions to the mixed final output, the main performance guarantee is hard to trust.

### Major:
- **The paper overstates optimality/tightness relative to its own theorems.**  
  The abstract and introduction repeatedly claim the method is “optimal” and “tight,” and Table 1 presents this very prominently. But the paper itself also acknowledges in Section 1.1 that optimality holds “except for the finite-horizon \(H\).” This qualification is important, not minor.  
  More concretely, Theorem 1 gives
  \[
  \tilde O\!\left(\frac{C_r^* H^4 S(A+B)}{\varepsilon^2} f(\sigma^+,\sigma^-,H)\right),
  \]
  while Theorem 2 gives lower bounds of order
  \[
  \Omega\!\left(\frac{C_r^* H^3 S(A+B)}{\varepsilon^2}\min\left\{\frac{1}{\min\{\sigma^+,\sigma^-\}},H\right\}\right).
  \]
  So the result supports strong claims about \(S\) and \(A+B\), but not full tightness. The headline language should be narrower and more careful.

- **There are substantial presentation/notation inconsistencies exactly in the quantities the paper claims as main innovations.**  
  Examples visible in the main text include:
  - The uncertainty factor \(f(\sigma^+,\sigma^-,H)\) is written inconsistently across Table 1, Section 1.1, and Theorem 1.
  - Assumption 1 refers in the text to \(C_r^*\) but the displayed equation uses \(C_\epsilon^*\).
  - Eq. (20) and Algorithm 2 appear inconsistent in the Bernstein penalty form.
  - Eqs. (12) and (14) switch to \(N\) after the dataset was defined with \(K\) episodes.  
  Some notation issues are survivable, but here they affect the uncertainty dependence, the data-quality coefficient, and the penalty term—the core technical content. This substantially hurts technical clarity and confidence.

- **The “partial coverage” framing is somewhat stronger than what the theorem actually guarantees.**  
  The paper is right that it does not assume full uniform coverage of all state-action tuples, and the clipped concentrability condition is a meaningful relaxation. However, Theorem 1 still requires a burn-in/sample lower bound depending on
  \[
  d_m^n = \min_{h,s,a,b}\{d_h^n(s,a,b): d_h^n(s,a,b)>0\},
  \]
  through Eq. (24), i.e., inverse dependence on the smallest positive occupancy mass. If \(d_m^n\) is tiny, the required sample size can be very large. So the result is a partial-coverage guarantee under a nontrivial occupancy lower-bound condition, not a broad statement that sparse offline data is generally enough.

- **Computational tractability is not adequately discussed, despite the paper making strong algorithmic claims.**  
  The method requires solving `ComputNash` at every state and time step, and also solving the robust Bellman backup via the TV dual in Eq. (18). Yet the paper provides no overall computational complexity analysis. Moreover, the “Policy estimation” paragraph says “Solving these robust matrix games is generally PPAD-hard,” which is confusing in context because Algorithm 2 applies `ComputNash` to ordinary zero-sum payoff matrices \(\hat Q_h^\pm(s,\cdot,\cdot)\), for which equilibrium computation is standard. The presentation leaves unclear what is actually hard and what is efficiently solved in the proposed method.

- **The multi-player general-sum extension is too underdeveloped in the main paper for the strength of its claims.**  
  Theorem 3 is only stated, with essentially no substantive derivation or discussion in the main text. Yet the paper describes this as “a breakthrough in breaking the curse of multiagency.” For a general-sum robust Markov game extension, equilibrium definition, computation, and selection are delicate; a theorem statement plus slogan is not enough support for such a strong claim.

### Minor
- **The paper is restricted to TV-distance uncertainty in the actual results, despite the problem formulation listing many divergences.**  
  Section 2 discusses KL, Wasserstein, \(\ell_q\), etc., but the algorithmic tractability and analysis in the main results rely on the TV dual form (Eq. 18). This is acceptable as scope, but the paper should more clearly state that the contribution is TV-specific rather than implying broader divergence generality.
- **Assumption 1 is difficult to parse and its interpretation is under-explained.**  
  It quantifies over policies/kernels tied to robust best responses and is analysis-centric rather than observable from data. That is not inherently invalid, but it needs much better intuition in the main text.
- **The paper lacks empirical validation entirely.**  
  For a theory-focused submission this is not fatal, but even a minimal tabular experiment would have helped validate the interaction between the subsampling step, pessimistic penalty, and uncertainty-level dependence.

### Trivial
- None.

## Nice-to-Haves
- Add a short tabular experiment showing NE-gap versus sample size and versus uncertainty radius \(\sigma\), ideally including comparison to the cited prior baseline.
- State the scope of optimality precisely in the abstract/introduction: optimal in \(S\) and action dependence, but not in \(H\).
- Add a concise computational complexity discussion for the TV dual backup and the per-state matrix-game solves.
- Include a short limitations paragraph on rectangularity, TV-only analysis, tabular scope, and the role of \(d_m^n\).
- Give an intuition paragraph before Assumption 1 explaining which occupancy measures are compared and why clipping is the right device.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Eq. (8) has malformed expectations / \(\nu_h(a)\)”** — removed as likely parser/extraction noise rather than a reliable paper issue.
- **“The primal robust backup is exponentially hard in \(S\)”** — the paper indeed says this, but the exact phrasing is more of an overstatement than a substantive flaw in the contribution.
- **Generic complaint that more ablations are needed for the two-stage subsampling and penalty** — removed as too generic for a theory paper, especially without experiments being standardly required for the core claim.
- **Criticism that the multi-player extension is merely ‘expected’ in model-based tabular settings** — weakened/removed in that form. The valid issue is not that the extension is unsurprising, but that it is under-justified relative to the strength of the claim.
- **Broad scope-creep criticisms asking for robustness against unrelated attack models or broader non-tabular evaluations** — removed as outside the paper’s stated scope.

## Novel Insights
The strongest synthesis across the reviews is that the paper’s most interesting idea is not just the \(S(A+B)\) sample-complexity improvement, but the attempt to define a *unilateral* clipped concentrability notion tailored to robust best-response occupancies under partial coverage. That is a potentially valuable lens for offline robust game learning. However, the paper pairs that promising analytical idea with a much shakier equilibrium formulation: the submission effectively learns from two separately robustified objectives and then cross-combines policies from two distinct recursions. This makes the paper feel split between a genuinely interesting sample-complexity analysis direction and an insufficiently grounded game-theoretic target/algorithm interface.

## Suggestions
- **Clarify the solution concept first.** Explicitly define whether the paper studies one robust zero-sum game with a single saddle-point operator, or a pair of player-specific robust best-response problems. If the latter, prove existence and explain why Eq. (11) is the correct gap notion.
- **Justify the cross-pair output in Algorithm 2.** The main paper needs a direct lemma/theorem showing why outputting \((\mu^-, \nu^+)\) yields an approximate equilibrium under the stated definitions.
- **Tone down and localize the optimality claims.** Replace broad “optimal/tight” language with precise claims: optimal in \(S\) and \(A+B\), with an open gap in \(H\).
- **Clean up the technical presentation.** Ensure consistency of \(f(\sigma)\), \(C_r^*\), the penalty formula, and sample-count notation. Right now, these inconsistencies significantly reduce confidence.
- **Add a computational discussion.** Give the complexity of each robust backup and each Nash solve, and clarify the relation between the claimed PPAD-hardness and what Algorithm 2 actually computes.
- **If space allows, add a minimal synthetic experiment.** Even one small tabular RTZMG would help demonstrate that the method behaves as the theory suggests.

---

## 33P4evE2ej

- GT: Reject (avg 4.8)
- Predicted: N/A (5.1/10)
- Match: N/A

### Final Review

## Summary
This paper proposes **DynaMer Adapter**, a parameter-efficient adaptation method that jointly uses a general-domain ViT and a medical-domain ViT by dynamically merging their token representations through a **shared gated MoE adapter** and a **layer-wise skipping router**. Empirically, the method delivers consistently best average results across Med-VTAB, patient-split evaluations, and some general-domain transfer benchmarks, while using fewer tunable parameters than prior dual-expert adapter variants.

## Strengths
- **The central design is more specific than a routine adapter tweak:** rather than adapting one backbone, the paper explicitly targets the practically relevant case where **general and medical pretraining provide complementary strengths**, and operationalizes this via **token-level cross-model fusion** with a shared MoE adapter plus per-domain gating. This is a concrete architectural idea, not just “add another adapter.”
- **The paper shows unusually broad benchmark coverage within its target setting.** Results are reported across **23 medical datasets** spanning color, X-ray, and OCT/CT/MRI tasks (Tables 1–3), plus **patient split evaluations** (Tables 8–9) and **general-domain transfer** on FGVC/VTAB-1K (Table 10). The wins are not isolated to one modality.
- **DynaMer is consistently stronger than the closest adapter baselines while using fewer added parameters than prior dual-model MoE variants.** For example, compared to GMoE-Adapter, DynaMer improves nearly every reported dataset while using **1.21X total params vs. 1.35X** in Tables 1–3 and 10. That supports the claim that the shared adapter design is parameter-efficient.
- **The ablations do verify that some proposed components matter.** Tables 4–6 show gains from adding gates on both streams and applying them across more layers; Table 7 shows a meaningful internal speed/accuracy tradeoff, with **50% token processing reducing batch inference time from 0.165s to 0.086s** while slightly improving accuracy on the color-image suite.
- **The generalization beyond purely medical evaluation is at least partially supported.** Table 10 shows that the same machinery remains competitive and slightly best on FGVC and VTAB-1K, which strengthens the paper’s claim that the token-merging principle may extend beyond the immediate medical setting.

## Weaknesses
### Fatal
- None.

### Major:
- **The paper’s efficiency claims are only partially substantiated, because the reported timing evidence is not comparative and does not account for the cost of using two backbones.**  
  The paper repeatedly makes strong efficiency claims: e.g., the abstract says the skipping router “optimiz[es] inference time,” and the introduction claims DynaMer “achieves few costs in both training and inference time.” However, the only explicit wall-clock evidence is **Table 7**, which reports inference time **only for DynaMer at different token-retention ratios**. There is **no direct runtime/FLOPs/memory comparison against baselines** such as Adapter, MoE-Adapter, or GMoE-Adapter, despite Figure 1 visually positioning the method as comparatively efficient. This matters especially because the system runs **both a general ViT and a medical ViT**, which is a real architectural cost not quantified in the paper. Parameter-count savings alone are not sufficient to support the broader efficiency narrative.
- **Several headline claims are stronger than the empirical evidence warrants, especially for patient-OOD and few-sample performance.**  
  The abstract and introduction claim DynaMer “particularly excel[s] in patient out-of-distribution settings and tasks with only few samples.” The paper does show improvements in Tables 8–9 and Figure 1(c), but the margins over the strongest baselines are generally **small** (often only a few tenths of a point), and there is **no uncertainty reporting** anywhere. Likewise, the few-shot/data-efficiency evidence is concentrated in **Figure 1(c)** without a corresponding detailed table or protocol description. Given the strength of the wording (“particularly excelling”), the empirical support is not fully commensurate with the claim.
- **The method description is underspecified in a few places that matter for interpretation of the architecture.**  
  This is not a minor implementation-detail complaint; several core aspects are genuinely unclear from the current text:
  - Sec. 3.2 introduces top-\(k\) expert routing but does not clearly specify the actual values of **number of experts \(n\)** and **selected experts \(k\)** in the paper body.
  - The interaction between the two streams is not fully transparent: the method assumes token-wise pairing \((\mathbf{x}_{\text{gen},i}, \mathbf{x}_{\text{med},i})\), but the exact alignment assumptions across the two frozen ViTs are not discussed.
  - Sec. 3.3 claims the skipping router can reduce the number of tokens processed in deeper layers, but **Eq. (4)** only explicitly partitions/reorders tokens into selected and skipped subsets. The text says skipped tokens “will skip the adapter” and that processed and skipped tokens are concatenated and sent to the next layer, which makes the exact computational savings mechanism somewhat ambiguous unless one infers that skipping applies only to the adapter path, not the full transformer.
  - There is also a genuine inconsistency between Sec. 3.3 (“They are optimized end-to-end”) and Sec. 4.1 (“Each expert within the MoE architecture was optimized individually before the gating mechanism was trained”). Those two descriptions suggest different training procedures and should be reconciled.
- **The paper does not test whether the benefit really comes from the proposed dynamic token fusion, versus simpler ways of combining two frozen experts.**  
  The comparisons are mostly against prior prompt/adapter methods, including prior MoE-style adapters, but the paper omits some highly relevant simple baselines for its central claim: e.g., **feature averaging, concatenation plus linear fusion, or logit ensembling** of the general and medical models. Since the conceptual contribution is specifically “dynamic merging” of two experts, it is important to show that the elaborate routing/gating machinery outperforms straightforward two-expert fusion strategies, not only prior adapter families.

### Minor
- **The absolute gains over the strongest baseline are often modest.**  
  Tables 1–3 and 10 show consistent improvements, which is positive, but many are incremental relative to GMoE-Adapter. This does not negate the contribution, but it does reduce the significance of claims framed as a large advance.
- **The choice of medical expert is narrow relative to the breadth of evaluated modalities.**  
  Sec. 4.1 states that the medical expert is a ViT-B/16 pre-trained on **1.6M cell images**, yet the downstream evaluation spans retinal, skin, X-ray, CT, and MRI tasks. The paper’s thesis is that combining general and medical priors is broadly useful, but it does not analyze whether this result depends on this specific medical checkpoint or whether other medical pretraining choices would materially change the conclusions.
- **The analysis of the skipping router is incomplete, particularly because 50% token retention outperforms 100%.**  
  Table 7 is interesting, but also raises a question the paper does not answer: why does discarding half the tokens slightly improve every reported metric on that benchmark slice? This could reflect useful regularization or denoising, but without analysis it is hard to know whether the router is learning meaningful saliency or whether the full-token configuration is simply less well tuned.
- **The qualitative evidence is weaker than the text suggests.**  
  Figure 3 only visualizes DynaMer attention maps, yet the surrounding discussion makes comparative claims about “previous methods” and about mitigating “spatial and prompt forgetting.” Without side-by-side baseline visualizations under identical conditions, those interpretability claims are suggestive rather than demonstrated.
- **There is a naming inconsistency that hurts clarity.**  
  Table 9 refers to “GL-MoF Adapter (ours)” rather than DynaMer Adapter, which looks like a leftover name/version mismatch and should be corrected.

### Trivial
- None.

## Nice-to-Haves
- Report **mean ± std over multiple seeds** for the main benchmark tables, especially because many gains over the strongest baseline are small.
- Add **comparative compute reporting**: total FLOPs, wall-clock inference/training time, and peak memory for DynaMer versus Adapter/GMoE-Adapter, including the cost of running two frozen ViTs.
- Include **simple two-expert fusion baselines** such as feature averaging, concatenation with a learned linear head, and logit ensembling.
- Analyze the learned **gating weights** and **expert-routing patterns** across modalities to verify that the method is actually learning meaningful domain specialization.
- Test at least one additional **medical pre-trained checkpoint** from a different modality family to assess how much the result depends on the current cell-image expert.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper relies too heavily on limited medical benchmarks / lacks external datasets.”**  
  Removed because it is not supported by the paper text. The paper evaluates on a broad Med-VTAB suite spanning many datasets and modalities, plus patient-split and general-domain benchmarks. One can still question specific claims, but not dismiss the evaluation as narrowly confined.
- **Pure writing/style issues** such as grammar problems (“four four folds”) or generic presentation complaints.  
  Removed per instruction as formatting/style nitpicks, except where a wording problem reflects overclaiming or genuine ambiguity.
- **Broad criticism that ablations are absent or wholly inadequate.**  
  Removed in that absolute form because the paper does include several real ablations (Tables 4–7). The fairer retained criticism is narrower: the ablations do not test some of the most relevant simple fusion alternatives or fully explain the skipping behavior.
- **Any concern doubting the existence/availability of cited models or benchmarks.**  
  Removed by rule.

## Novel Insights
The most interesting aspect of the paper is not just that two experts are combined, but that the **adapter itself is shared across the two backbones while the gates/routers remain stream-specific**. That is a potentially elegant compromise: it encourages a common fusion mechanism while still allowing domain-specific control over how much adaptation to apply. The empirical pattern in Table 7 is also more revealing than the paper makes it seem: the fact that moderate token skipping can slightly improve accuracy suggests that the model may benefit from **selective suppression of low-value cross-domain interactions**, not merely from saving compute. If validated with routing analysis, that could become a stronger conceptual contribution than the current presentation emphasizes.

## Suggestions
- Reframe the abstract/introduction claims to better match the current evidence, especially around **efficiency**, **patient-OOD**, and **few-sample** advantages.
- Add a table reporting **full compute costs** versus baselines: runtime, FLOPs, and memory, including the overhead of the second frozen backbone.
- Clarify the training procedure so Sec. 3.3 and Sec. 4.1 are consistent: is the method trained **end-to-end**, or are experts pretrained separately before gate training?
- Explicitly specify the key architectural hyperparameters in the main paper: **number of experts, top-k routing choice, adapter placement, and whether the skipping router is used during training, inference, or both**.
- Add **simple dual-expert fusion baselines** to isolate the value of the proposed dynamic token-merging mechanism.
- Provide a short analysis of **why 50% token retention beats 100%**, ideally with token-selection visualizations or per-layer selection statistics.
- If space permits, test one more **medical expert checkpoint** from a non-cell modality to strengthen the claim of broad medical applicability.

---

## EUAxxrxOM8

- GT: Reject (avg 5.0)
- Predicted: N/A (7.1/10)
- Match: N/A

### Final Review

## Summary
This paper studies infinite-horizon average-reward restless Markovian bandits and analyzes a rolling-horizon LP / model predictive control policy (“LP-update”) that replans from the current empirical state at every step, then applies randomized rounding to satisfy the hard per-step budget constraint. The main contribution is a new analysis based on dissipativity that yields asymptotic optimality with an \(O(1/\sqrt N)\) gap under a broad mixing assumption, and an exponentially small gap under additional local stability / non-degeneracy assumptions.

## Strengths
- **The core theoretical contribution is real and technically interesting:** the paper gives an infinite-horizon average-reward analysis of a finite-horizon MPC-style policy for RMABs, which the paper explicitly distinguishes from prior uses of finite-horizon LPs mainly in finite-horizon settings or under stronger assumptions. The main theorem (Theorem 4.1) establishes an \(O(1/\sqrt N)\) gap under Assumption 1, and Theorem 4.2 recovers exponential convergence under stronger stability assumptions.
- **The dissipativity viewpoint is a genuine conceptual contribution in this context:** Section 5 does more than repackage existing arguments. The rotated-cost construction, the storage function from the LP dual variable, and the monotonicity of \(L_\tau\) provide a coherent bridge from finite-horizon planning to infinite-horizon average reward. That is the most novel part of the paper.
- **The algorithmic object being analyzed is simple and practically plausible:** at each time step, solve a linear program on the empirical state and apply randomized rounding. This is substantially cleaner than many RMAB policies that require more specialized structural assumptions.
- **The assumptions for the \(O(1/\sqrt N)\) result are meaningfully broader than classical indexability/UGAP-style assumptions:** the paper does support that Theorem 4.1 does not rely on indexability or UGAP, and Assumption 1 is at least a credible broad mixing-type condition. This strengthens the significance of the general theorem.
- **The empirical section contains some genuinely insightful diagnostics, not just reward curves:** Figures 2 and 3 attempt to explain behavior via rotated cost and state-space trajectories, which is well aligned with the paper’s theoretical perspective and is more informative than a pure aggregate-performance table.

## Weaknesses
###: Fatal
None.

### Major:
- **The empirical evidence does not fully support the paper’s broad practical-positioning claims.**  
  The abstract says the method “performs very well in practice when compared to the state of the art,” and the introduction says it “beat[s] state of the art algorithms in our benchmarks.” But Section 6 compares only against LP-priority and FTVA, and explicitly justifies these baselines because they are “natural and simple to implement,” not because they represent the strongest practical baselines for the regimes considered. This leaves the empirical claim overstated relative to the evidence actually shown. The theory remains valuable, but the practical superiority claim should be toned down or supported more convincingly.

- **The paper does not analyze computational cost enough to justify its practical-efficiency messaging.**  
  The method solves a \(\tau\)-horizon LP at every time step. Yet Section 6 presents no timing or scaling study, despite claims in the abstract/introduction/main contributions that the method is easy to implement and performs well in practice, including “in terms of computational time horizon \(T\).” This omission matters because rolling re-optimization is the main practical tradeoff of MPC relative to simpler heuristics. Reward-only plots are insufficient to substantiate practical competitiveness.

- **There is a nontrivial clarity gap between the analyzed algorithm and the algorithm reportedly used in practice due to the treatment of the terminal term \(\lambda \cdot x(\tau)\).**  
  Section 3.1 defines the finite-horizon problem (8) using the terminal term with \(\lambda\) equal to the dual multiplier of the LP, and Section 5’s dissipativity proof relies centrally on this construction. But the paper then states: “our proofs will hold with minor modification by replacing \(\lambda\) by 0 and in practice we do not use this multiplier for our algorithm.” As written, it is unclear whether the experiments solve (8) with \(\lambda\), solve a different LP with \(\lambda=0\), or use some other variant. Since the proof sketch leans heavily on the storage/terminal term, this mismatch should be explained much more explicitly.

- **Some central assumptions / constants are not operationalized well enough for the paper’s “easy to verify / quantified loss” narrative.**  
  Assumption 1 is presented as an easily verifiable mixing assumption, but in the main text it is defined through the worst-case quantity \(\rho_k\) over all initial states and action sequences (Eq. 10). The paper gives a sufficient condition via ergodicity of \(P^0\), which helps, but does not really explain how one would verify or estimate the assumption in realistic models. Likewise, the bound in Eq. (11) depends on \(k,\rho_k,C_\lambda,C_\Phi\) and \(\tau(\epsilon)\), but the main text gives only limited guidance beyond \(\tau(\epsilon)=O(1/\epsilon)\). This does not invalidate the theorem, but it weakens the practical interpretability of the guarantee.

- **Assumption 4 for the exponential result is under-motivated and insufficiently discussed.**  
  Theorem 4.2 additionally assumes uniqueness of the LP solution of (8) for all \(\mathbf{x}\), described only as “a technical assumption that simplifies the proofs.” That is a strong parametric uniqueness requirement, and the paper gives little guidance on when it holds, how restrictive it is, or whether the result can be obtained with a tie-breaking rule or local uniqueness instead. Since Theorem 4.2 is one of the headline results, this deserves better justification.

### Minor
- **The empirical evaluation is somewhat narrow for a paper making broad practical claims.**  
  The experiments cover a few representative examples and useful parameter sweeps, but they do not test failure modes, difficult regimes, or instances specifically designed to showcase the claimed advantage of avoiding stronger structural assumptions. For example, the paper argues its generality relative to prior structurally dependent methods, but the experiments do not clearly demonstrate a regime where that generality is decisive.

- **The main text compresses some important proof steps too aggressively.**  
  In the sketch around Eqs. (16)–(19), the argument switches between stochastic \(U(t)\) and deterministic/mean-field \(u(t)\), and several key bounds are pushed into imported lemmas. The overall line is plausible, but one more level of explicitness in the main text would improve auditability for such a theory-heavy submission.

- **The paper should be more precise about what is novel: the main novelty is the proof framework and infinite-horizon analysis, not the asymptotic rates themselves.**  
  The paper mostly acknowledges this, but some wording could still better distinguish “new algorithmic analysis and viewpoint” from “new convergence rates.” The strongest contribution is the dissipativity-based analysis of MPC for this setting.

### Trivial
- **One small empirical presentation issue is that some figure/caption references are not fully contextualized in the main text** (e.g., Figure 3’s example naming), which slightly hurts readability.

## Nice-to-Haves
- Add a runtime/scaling study as a function of \(|\mathcal S|\), \(\tau\), and perhaps \(N\), alongside reward.
- Clarify whether experiments use the \(\lambda\)-terminal term or \(\lambda=0\), and if the latter, state the theorem variant explicitly in the main text.
- Give more practical guidance on choosing \(\tau\), especially how the truncation error and finite-\(N\) error should be balanced.
- Include at least one example where stronger structural assumptions behind classical approaches are violated, to better illustrate the motivation for this MPC approach.
- If feasible, add exact-optimum comparisons on more small instances, since the LP upper bound can hide absolute finite-\(N\) gaps.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **“Missing Whittle Index comparison” as a core weakness.**  
  It is reasonable as a suggestion, but it should not be elevated to a main criticism here. The paper does not claim to benchmark every classical RMAB policy, and I cannot verify from the paper alone whether the chosen examples are indexable or whether a WI implementation is straightforward/fair in those cases. So this is better treated as a nice-to-have rather than a substantive flaw.

- **Any criticism doubting release status / existence / verifiability of cited methods or references.**  
  Not applicable and should be ignored.

- **Generic complaints about omitted implementation minutiae or appendix placement alone.**  
  For example, the rounding procedure being in the appendix is not itself a meaningful weakness; the real issue is only whether the main text explains the theorem-relevant role of rounding sufficiently.

- **Overstated novelty attack claiming the method is “not new” because finite-horizon LP ideas already existed.**  
  The paper itself explicitly acknowledges that the LP-update idea existed for finite-horizon RMABs and positions its contribution as the infinite-horizon average-reward analysis via dissipativity. So “algorithm not new” is not a fair standalone criticism.

## Novel Insights
The most compelling synthesis across the reviews is that this paper’s real contribution is not a new asymptotic rate, but a new *control-theoretic explanation* for why a rolling-horizon LP policy works in average-reward RMABs. The rotated-cost/dissipativity lens gives a unifying interpretation of “operate optimally at the fixed point” that helps connect mean-field control, dual LP structure, and MPC truncation error in one framework. That perspective may outlast this specific theorem, especially if the authors can make good on the claim that the proof methodology extends to broader constrained MDP settings.

## Suggestions
- **Sharpen the claims.** Rephrase broad statements like “compared to the state of the art” and “beats state of the art algorithms” unless the empirical section is expanded accordingly.
- **Clarify the exact algorithm used in experiments.** State plainly whether Algorithm 1 solves (8) with the terminal term \(\lambda \cdot x(\tau)\) or the \(\lambda=0\) variant; if the latter, include the corresponding theorem statement or proposition in the main text.
- **Strengthen the discussion of Assumption 1 and Assumption 4.** For Assumption 1, provide a clearer checklist or computable sufficient conditions beyond the ergodic \(P^0\) example; for Assumption 4, discuss typicality/restrictiveness or how it might be relaxed.
- **Add computational evidence.** Report wall-clock times or solver scaling with \(\tau\) and \(|\mathcal S|\), and compare against the baselines used in Section 6.
- **Improve the proof sketch around Eqs. (16)–(19).** Make the role of \(u(t)\) vs. \(U(t)\), the rounding discrepancy, and the conditional-expectation step more explicit.
- **If space permits, add one regime specifically illustrating the advantage of avoiding stronger structural assumptions.** That would connect the theory more directly to the motivating narrative.

Overall, this is a strong theory paper with a meaningful conceptual contribution and credible core theorems. Its main limitations are over-ambitious practical framing and underdeveloped discussion of computational cost and some theorem-facing assumptions, rather than a failure of the central theoretical idea.

---

## 3b9SKkRAKw

- GT: Accept (Spotlight) (avg 8.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper proposes **LeFusion**, a lesion-focused diffusion framework for synthesizing pathology on top of real normal anatomy rather than regenerating full medical volumes. The key design combines inpainting-style background preservation with a lesion-only training objective, and extends this with histogram-based control for multi-peak lesion textures, multi-channel decomposition for multi-class lesions, and a diffusion model for lesion masks (DiffMask). On two 3D settings—lung nodule CT and cardiac lesion MRI—the method yields strong downstream segmentation gains, especially when the full system is used.

## Strengths
- **The paper targets an important but unusually well-scoped synthesis problem: generate lesions while preserving real anatomy exactly.** Eq. (3) explicitly composes generated lesion foregrounds with forward-diffused real backgrounds, so the background outside the lesion mask is preserved by construction during inference. This is a meaningful modeling choice for medical imaging, where anatomical realism outside the lesion often matters more than fully generative flexibility.

- **The lesion-focused objective is conceptually clean and well matched to the task.** Instead of spending model capacity on reconstructing both anatomy and pathology, Eq. (4) restricts the diffusion loss to the lesion region. That is a simple modification, but it is a task-specific inductive bias that is much more targeted than generic conditional generation.

- **The paper identifies and addresses two practically relevant synthesis challenges that are often glossed over: multi-peak texture distributions and multi-class correlated lesions.** The histogram-conditioning mechanism for lung nodules is a specific, annotation-light control signal derived directly from image-mask pairs, and the multi-channel decomposition for MI/PMO is a concrete way to jointly model correlated lesion types.

- **The empirical gains of the full system are substantial, not just marginal.** On LIDC, the best configuration improves nnU-Net Dice from 78.26 to 83.44 and SwinUNETR from 78.38 to 83.13. On Emidec, the strongest gains are especially notable for the difficult PMO class, e.g. nnU-Net PMO Dice rises from 36.32 to 43.54. These are large enough to be practically interesting even if statistical uncertainty is not fully characterized.

- **The evaluation is downstream-task-centric rather than relying only on image realism metrics.** Given the paper’s stated goal—synthetic data that improves training data for segmentation—evaluating impact on nnUNet and SwinUNETR is more meaningful than optimizing purely perceptual metrics.

- **The paper demonstrates useful control knobs rather than only unconditional synthesis.** Histogram control changes lesion attenuation patterns in lung nodules, and DiffMask gives control over size and location through the control sphere and boundary mask design. This controllability is a meaningful asset for data augmentation.

## Weaknesses

### Major:
- **The paper does not cleanly isolate the effect of its core methodological claim—the lesion-focused diffusion objective—from the other additions.**  
  The strongest results come from augmented variants such as **LeFusion-H + DiffMask** and **LeFusion-J + DiffMask**, not from the core lesion-focused mechanism alone. While the tables do compare LeFusion variants, they do **not** include the most direct internal ablation: same architecture, same inpainting-style inference, same training setup, but **global diffusion loss vs. lesion-masked loss**. RePaint is not an adequate substitute for this, because it is a different training/inference setup and the paper itself frames it as a standard inpainting baseline. As a result, the paper supports the usefulness of the **overall recipe**, but does not decisively prove that the lesion-focused objective itself is the main driver.

- **The evidence for “significant” downstream improvement is weaker than the wording suggests, especially on Emidec.**  
  The abstract says the generated data “significantly improves” segmentation performance, but the reported results are single numbers with no confidence intervals, no repeated runs, and no statistical tests. This matters because the cardiac evaluation uses only **10 pathological test cases**, and some intermediate comparisons in Table 2 are mixed or small. The largest gains are convincing as point estimates, but the paper does not establish robustness to training randomness or small-sample evaluation noise strongly enough to justify strong significance language.

- **DiffMask is under-specified relative to its importance in the final results.**  
  Section 3.3 describes the “boundary mask” and “control sphere” at a high level, but gives no mathematical formulation, no explicit diffusion objective, and little detail on constraints for anatomically plausible placement. Since DiffMask materially boosts the best reported numbers in both datasets, it should be described much more rigorously. At present, a reader can understand the idea, but not fully assess or reproduce why it works.

- **Several causal claims in the experimental analysis are plausible but not directly validated.**  
  In Section 4.2/4.3, the paper attributes baseline failures to “background disruption,” “bias toward healthy appearances,” or “ignored correlation between lesions,” but these are mostly post hoc explanations from visual examples and end-task outcomes. There is little direct diagnostic analysis of boundary continuity, category confusion, or lesion-property distribution matching to substantiate these causal interpretations.

### Minor
- **The histogram-conditioning component would benefit from more quantitative validation and clearer specification.**  
  The paper explains the high-level idea and shows a useful qualitative result in Fig. 6, but important details are missing or relegated: how the histogram is represented (e.g., bins, normalization), how sensitive performance is to that representation, and whether generated histograms truly align with the real lesion distribution beyond pairwise PSNR/SSIM diversity proxies.

- **The claim that lesion-focused training “simplifies the learning process” is intuitive but not directly demonstrated.**  
  There is no convergence-speed, sample-efficiency, or optimization-stability comparison supporting this statement. The claim may be true, but the paper currently treats it as motivation rather than evidence-backed conclusion.

- **The multi-channel decomposition is only demonstrated for two lesion classes.**  
  The method is reasonable for MI/PMO, but the paper does not discuss how memory or training scales when the number of lesion classes increases. This does not invalidate the cardiac experiments, but it limits how broadly the approach can be interpreted.

- **The practical claim is narrower than the broad framing.**  
  For LIDC, experiments are conducted on ROI crops rather than full scans. That is a valid setup for lesion synthesis, but the downstream conclusion should be read as improving ROI-level augmentation for segmentation, not as demonstrating full clinical pipeline impact. Similarly, the introduction and conclusion mention fairness/privacy and broader anomaly domains, but those are motivations and possible extensions, not evaluated outcomes here.

### Trivial
- **A more explicit failure-case analysis would improve trustworthiness.**  
  The paper mainly shows successful examples. A small gallery of implausible masks, texture failures, or difficult boundary cases would make the empirical story more complete.

## Nice-to-Haves
- Add a direct ablation: identical architecture and inpainting inference, comparing **global diffusion loss vs. lesion-focused masked loss**.
- Report multi-seed downstream results or confidence intervals, especially on Emidec.
- Formalize DiffMask with equations and isolate the contributions of boundary mask vs. control sphere.
- Add boundary-interface analysis showing lesion/background continuity near the mask edge.
- Quantify histogram-control fidelity beyond diversity proxies, e.g. alignment between generated and real histogram clusters.
- Include a simple baseline such as copy-paste + DiffMask to separate the value of better mask diversity from better texture synthesis.
- Discuss computational cost and memory scaling, particularly for multi-channel modeling.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Complaint about missing comparisons to concurrent work because code/models were unavailable.**  
  The paper explicitly states: “Due to differences in research focus or/and the unavailability of their code/models, a comprehensive comparison could not be conducted.” Under the review policy here, criticism rooted in questioning availability/release status should be removed.

- **General requests for broader clinical endpoints, radiologist studies, or cross-site robustness as core weaknesses.**  
  These would strengthen the paper, but they are outside the paper’s stated scope, which is algorithmic lesion synthesis evaluated through downstream segmentation. They are better treated as future extensions rather than central flaws.

- **Reproducibility complaints about implementation minutiae such as patch size/crop overlap/hyperparameter granularity.**  
  The paper already gives the main dataset splits and experimental setup, provides code/models, and these requests are too implementation-specific to be central weaknesses for this review.

- **Criticism that the comparisons are unfair because the baselines are less controllable than LeFusion.**  
  This is only partly reasonable. The paper’s contribution explicitly includes controllability (histogram control, multi-channel decomposition, DiffMask), so outperforming less controllable baselines is not inherently unfair. The valid core concern is narrower: the experiments do not isolate which added control mechanism drives the gains.

- **Strengths such as “the paper is well written” or “the topic is important.”**  
  These are too generic and were omitted.

## Novel Insights
The paper’s real contribution is less “a new diffusion model” in the abstract and more a **reframing of lesion synthesis as selective generation under exact anatomical preservation**. That framing is stronger than many generic medical image generators because it matches the asymmetry of the problem: pathology is scarce and variable, while anatomy is abundant and should often be copied rather than reimagined. The empirical results suggest that this selective-generation viewpoint is especially effective when paired with explicit control over lesion statistics (histograms) and lesion support (DiffMask). At the same time, the paper currently proves the strength of the **integrated system** more convincingly than the specific necessity of its core lesion-masked objective.

## Suggestions
- Add the decisive ablation: same model and inference, with and without lesion-focused loss.
- Replace “significantly improves” with more measured wording unless statistical support is added.
- Formalize DiffMask with equations, training target, conditioning parameterization, and placement constraints.
- Add multi-seed downstream experiments or uncertainty estimates, prioritizing Emidec.
- Provide direct diagnostics for the claimed mechanisms: boundary smoothness, lesion-type distribution matching, and category correlation modeling.
- Include a brief limitations section discussing ROI-level evaluation, two-class multi-channel scope, and known failure modes.

---

