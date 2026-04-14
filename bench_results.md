# ICLR Benchmark Results

Date: 2026-04-14 01:36
Critic/Merger: openai/gpt-5.4 (OpenRouter)
Neutral: qwen/qwen3.5-plus-02-15, Related Work: qwen/qwen3.5-flash-02-23:online (OpenRouter)

## SI6zocV2SS

- GT: Reject (avg 1.5)
- Predicted: Reject (2.0/10)
- Match: YES

### Final Review

## Summary

CAN (Continuously Adapting Networks) proposes to mitigate catastrophic forgetting by combining selective neuron freezing with Hebbian learning. The method maintains a separate Hebbian weight matrix updated by local co-activation signals, averages those weights per neuron to produce importance scores, and uses those scores to scale SGD gradients via PyTorch hooks — effectively constructing binary masks that prevent previously-assigned neurons from being modified when learning subsequent tasks. Experiments are conducted on 2-task splits of MNIST and CIFAR-10 under task-incremental (TIL) and domain-incremental (DIL) settings, compared only against a vanilla ANN baseline.

---

## Strengths

- **Using unsupervised, local Hebbian co-activation as the importance signal is a genuinely distinct mechanism for continual learning.** Most parameter-isolation methods compute importance from global, supervised signals — Fisher information (EWC), path integrals of gradient norms (SI), or loss-based saliency. Anchoring importance scoring entirely in unsupervised local activations, requiring no global error signal, is a different design point that is task-agnostic by construction and avoids the computational overhead of second-order gradient methods. This is a specific choice, not a generic observation.
- **Evaluating both TIL and DIL with different masking protocols for each is more complete than single-setting evaluation.** The paper correctly identifies that in the DIL setup the output heads overlap and masking at inference is inapplicable, and adjusts the protocol accordingly (gradient scaling only, no forward masks). This distinction is handled consistently.

---

## Weaknesses

### Fatal

- **Near-random performance makes the paper's core claim unverifiable.** Table 1 reports CAN at 27.1% average accuracy on CIFAR-10 TIL, on a 5-class problem where random chance is 20%. The vanilla baseline is 22.5% — also near-random. An improvement of 4.6 percentage points over a nearly non-functional baseline does not demonstrate that catastrophic forgetting has been meaningfully reduced; it demonstrates that neither model is learning the tasks. A system cannot be said to "remember" tasks it never successfully learned. The paper provides no single-task accuracy upper bound (i.e., what accuracy is achieved if the model is trained only on one task with no CL constraint), making it impossible to determine whether the bottleneck is the continual learning mechanism, the architecture, or a fundamental implementation bug. Until single-task accuracy is established as reasonable, no claim about forgetting can be evaluated.

- **Figure 3's y-axis is labeled "Acc" but displays values of 2.0–2.45, which are physically impossible for accuracy on any standard scale (0–1 or 0–100%).** These values fall in exactly the same numerical range as the loss curves in Figures 4–6, which are correctly labeled as loss. The accompanying table inside Figure 3 also lists "Task 1 (Acc)" and "Task 2 (Acc)" values such as 2.05 and 2.45. This is almost certainly a loss curve mislabeled as an accuracy curve. The consequence is that all visual evidence in the results section is either mislabeled or uninterpretable, and the only usable quantitative evidence is Table 1 — whose values are themselves near-random.

### Major

- **No comparison against any established continual learning method.** The sole comparison in the paper is against a vanilla ANN trained with standard SGD. For a continual learning submission, comparison against EWC, SI, Experience Replay, or parameter-isolation methods such as PackNet is a minimum expectation. Without such baselines, there is no way to know whether the modest improvement over vanilla SGD is competitive, redundant, or worse than trivial alternatives. The absence of these baselines, combined with near-random absolute performance, makes the empirical section essentially uninterpretable.

- **Oracle task identity at inference — the system is incomplete as presented.** Section 4.3 explicitly states: *"we are manually selecting the mask and measuring the metrics but it can be done using a gating system that automatically selects the relevant gate according to the given task during inference"*, and identifies the gating mechanism as future work. Requiring a human to select the task mask at test time means the reported TIL results are not reproducible in any realistic deployment scenario. The paper's broader language about "allowing the network to learn continually" overstates the current system. This should be clearly scoped to task-ID-known TIL evaluation.

- **Core methodology is underspecified and not reproducible.** After reading the full paper, the following questions remain unanswered:
  - Are the Hebbian weights a separate parameter matrix maintained in parallel to the ANN weights, or are they the ANN weights themselves? (Section 3.1.1 refers to "Hebbian parameter" *w* but this is never clarified structurally.)
  - Are Hebbian updates accumulated per sample, per batch, or per epoch, and over what window?
  - How exactly are Hebbian weights converted into gradient scaling values (normalization procedure, layerwise vs. global)?
  - Is Oja's rule (Section 3.1.2) actually used in the experiments? The paper says it "can be used in practice" but never states which rule is used experimentally.
  - Is lateral inhibition (Section 3.1.3) implemented in the reported experiments? No equation, no hyperparameter, and no mention in the experiment section is provided. Written in present tense as a method contribution ("we introduce competition"), yet absent from any implementation detail.
  - What threshold defines "selected" neurons (Section 3.3 references "a pre-defined threshold" but no value is given)?
  - What loss function is used? The paper specifies "Sigmoid activation at the end" for multi-class classification, which is non-standard (softmax + cross-entropy is the norm), and the loss is never named. This likely contributes directly to the near-random performance.
  - What optimizer configuration, batch size, weight decay, and learning rate schedule are used?
  - No pseudocode or formal algorithm is provided.

- **MNIST results are missing.** The paper states "All our experiments were conducted on the MNIST and the CIFAR-10 Dataset" but Table 1 — the only accuracy table — explicitly says "The above values are the results by using the CIFAR-10 dataset." No MNIST accuracy results are reported anywhere.

- **Single seed; no variance estimates.** All results come from one fixed seed (720). For a small MLP on CIFAR-10, running 3–5 seeds is inexpensive and expected. A single run cannot distinguish true performance from noise, especially given the small margins reported.

### Minor

- **Two-task evaluation only.** Both CIFAR-10 and MNIST are split into exactly two 5-class subsets. With only two tasks, long-term forgetting dynamics, capacity exhaustion, and mask accumulation behavior over many tasks cannot be assessed. Standard CL benchmarks (Split CIFAR-100, Permuted MNIST with 5–20 tasks) exist precisely for this reason.

- **The stated constraint "we can't use a continuous stream of data belonging to a variable number of classes" (Section 4.1.1) is a significant architectural limitation** mentioned only in passing. This restricts the method to fixed-output-head, fixed-task-count settings — a narrower scope than the introduction implies. It should be stated prominently as a limitation.

- **Abstract claim "new tasks can be trained without changing parameter weights" is inaccurate.** The method does change parameter weights — it trains the selected subset. The correct characterization is that *previously-assigned* parameters are not changed. This should be corrected.

- **"Time to Stability" (Section 5.3)** provides no quantitative table, no operational definition, no baseline comparison, and no analysis. The observation that Task 2 requires 20 epochs vs. 10 for Task 1 is presented without interpretation.

### Tiny

- Citing Zenke et al. (2017) (the Synaptic Intelligence paper) under meta-learning approaches in Section 2.5 is a citation misplacement; it belongs under regularization, where it is correctly cited in Section 2.1.
- The loss notation `L(θ) = L_n(θ_n) ∈ T_n` is ill-formed (a loss value is not an element of a task set); the meaning is recoverable from context but should be stated precisely.

---

## Nice-to-Haves

- **Ablation on the Hebbian importance mechanism.** Without comparing Hebbian-based neuron selection against random selection, magnitude-based selection, or gradient/Fisher-based selection, there is no evidence that the Hebbian component specifically — rather than the freezing strategy alone — contributes to any gains.
- **Capacity utilization analysis.** Report the fraction of neurons consumed per task, the overlap (if any) between masks, and a projection of how many tasks the 256-128-64 network supports before exhaustion.
- **Visualization of learned masks.** Heatmaps showing which neurons are selected per task per layer would reveal whether the method is finding meaningful task-specific subnetworks or partitioning neurons near-randomly.
- **Per-task single-task accuracy as upper bound.** Reporting the accuracy of a model trained solely on one task establishes a ceiling against which CL performance can be measured.
- Implement and evaluate the autoencoder-based gating mechanism to remove the oracle task identity requirement.

---

## Removed Points

*These points were raised in sub-reviews but are flagged for removal. They are preserved here for transparency.*

- **"Hou et al. (2025) is a future-dated citation"** (Review 2): The reference appears in the bibliography with a valid DOI (Information Sciences, 687:121368, July 2025). Per review policy, if the paper cites a reference, it is assumed to exist. Removed.
- **"The paper should use CNNs for CIFAR-10"**: The paper explicitly uses and scopes itself to MLPs. Demanding CNNs imposes architectural requirements outside the stated contribution. The low accuracy is a genuine concern but should be attributed to the near-random performance issue, not to the architecture choice. Removed as a standalone criticism.
- **Demand for theoretical convergence proofs**: This is an empirical systems paper. Theoretical guarantees on Hebbian convergence are not a standard expectation for this type of contribution. Removed.
- **Generic strengths** ("the paper is well-written," "the topic of continual learning is important"): These apply to any paper in the field and are not evidence of specific quality. Removed.

---

## Novel Insights

The three reviews collectively point to a deeper diagnostic concern worth highlighting directly for the authors: the paper's experimental results likely reflect an implementation-level problem rather than a merely weak method. The combination of (1) a non-standard sigmoid output layer for multi-class classification with an unnamed loss function, (2) loss curves for all conditions hovering near or above the entropy of a random 5-class predictor, (3) accuracy figures that are visually identical to loss figures yet labeled differently, and (4) final accuracy only marginally above random chance collectively suggest the network may not be converging to a working classifier at all — independent of the continual learning mechanism. This means the paper is not yet in a position to evaluate whether Hebbian importance scoring is a good strategy for subnetwork selection: the experiment cannot distinguish "Hebbian scoring is unhelpful" from "the network isn't trained correctly." Establishing a working single-task baseline with proper loss function and evaluation protocol is the essential first step before any CL claim can be made.

---

## Suggestions

1. **Establish single-task accuracy first.** Train the model on each 5-class split in isolation and confirm it achieves reasonable accuracy (>85% on MNIST, >55–60% on CIFAR-10 with an MLP). If it does not, fix the architecture and loss function before making any continual learning claims.
2. **Use softmax + cross-entropy** as the standard for multi-class classification. The sigmoid output is almost certainly causing the near-random performance. Explicitly state the loss function in the experimental setup.
3. **Correct Figure 3.** Replace mislabeled "Acc" curves with actual accuracy curves (0–100% scale) for both the vanilla and CAN models, measured on held-out test data after each epoch.
4. **Add at minimum EWC and a simple replay baseline** on the same 2-task setup. This is the single most important addition for the empirical section.
5. **Provide a complete algorithm description** specifying: (a) whether Hebbian weights are maintained as a separate matrix, (b) which Hebbian rule (basic or Oja's) is used in experiments and why, (c) whether lateral inhibition is implemented, (d) the exact normalization procedure for converting Hebbian weights to gradient scaling factors, and (e) the numerical threshold used for neuron selection.
6. **State the evaluation protocol explicitly** as task-ID-known TIL and remove language suggesting a more general continual learning solution until a gating mechanism is implemented and evaluated.
7. **Report MNIST results** alongside CIFAR-10 results, or remove the claim that experiments were conducted on MNIST.
8. **Run at minimum 3 seeds** and report mean ± standard deviation for all accuracy numbers.

---

## E6kQ51yfAj

- GT: Reject (avg 4.5)
- Predicted: Accept (5.0/10)
- Match: NO

### Final Review

## Summary
This paper proposes Game-Theoretical Preference Optimization (GPO), which conceptualizes LLM alignment as a two-player zero-sum game between an adversarial agent (prompt generator) and a defensive agent (response generator). Unlike prior self-play RLHF work that fixes a prompt set and optimizes only responses, GPO jointly trains both agents: the adversarial agent generates increasingly challenging and diverse prompts targeting current model weaknesses, while the defensive agent optimizes responses against those prompts. The paper provides a convergence guarantee to an approximate Nash Equilibrium for an idealized version of the algorithm, incorporates lexical and semantic diversity rewards to prevent adversarial collapse, and demonstrates consistent safety improvements over RLHF baselines across multiple in-distribution and out-of-distribution datasets.

---

## Strengths

- **Genuinely novel framing at the intersection of prompt generation and response alignment.** Prior self-play LLM alignment work (e.g., SPPO, self-play preference optimization) fixes the prompt set and has both players compete on response quality. Prior automated red-teaming trains attackers against static defenders. GPO's joint optimization of adversarial prompt generation and defensive response generation within a single co-adaptive game is a conceptually distinct contribution, and Section 5 articulates this distinction clearly with respect to both families of prior work.

- **Diversity mechanism is well-motivated, theoretically grounded, and empirically supported.** Section 3.3 provides a formal justification: without diversity regularization, the adversary's equilibrium degenerates to a point distribution (the single worst-case prompt), while diversity constraints force a richer distribution. This is not just asserted — the paper demonstrates it via the contrast between Algorithms 1 and 3. Table 2 confirms the practical value: GPO+Div achieves both higher ASR (48.57%) and higher diversity (0.70) than GPO (45.06%, 0.52), a combination that plain RLHF with diversity rewards (RLHF+Div) fails to achieve — RLHF+Div actually reduces ASR (33.60%) relative to RLHF (37.72%) while gaining diversity. This comparison in Table 2 provides strong evidence that it is the game-theoretic coupling, not diversity rewards alone, that allows attack strength and diversity to coexist.

- **Both sides of the game are evaluated systematically.** Table 1 evaluates the defender on three datasets (one in-distribution, two OOD); Table 2 evaluates the attacker's transferability to three held-out third-party models (Llama-2-7b-chat, Vicuna-7b, RLHF-trained model); Table 3 evaluates generalization to unseen jailbreak formats. Testing attack transferability to third-party models that were not part of training reduces evaluation circularity on the attacker side.

- **OOD evaluation shows non-trivial safety gains.** GPO+Div reduces ASR on ToxicChat from 24.06% (RLHF) to 14.37% and on PKU-BeaverTails from 8.28% to 3.44%, despite the model being trained exclusively on Anthropic red-teaming prompts. These are meaningful generalization improvements, not in-distribution artifacts.

---

## Weaknesses

### Fatal
None identified.

### Major

- **No comparison to MART or any iterative red-teaming baseline — the most critical experimental gap.** The paper explicitly discusses MART (Ge et al., 2023) in Section 5, distinguishing GPO by arguing that MART's SFT-based approach "makes it difficult to balance the capabilities of attackers and defenders." Despite this, MART is entirely absent from Tables 1–3. The paper's central claim — that game-theoretic coupling of RL-trained attacker and defender is valuable — cannot be substantiated without comparing to an iterative alternating baseline. An "iterate RLHF red-team + RLHF defender" loop without game-theoretic co-adaptation would serve as the minimum control, and MART as the most directly relevant prior work. Its omission leaves the incremental contribution unestablished.

- **No ablation on number of game iterations T, despite convergence being the central theoretical claim.** Theorem 3.2 provides an $\mathcal{O}(T^{-1/2})$ Nash-gap bound, yet no experiment plots performance as a function of the number of outer-loop iterations $T$. Figure 2 shows reward over gradient steps within training but does not identify how performance changes across game rounds. Without this, it is impossible to verify that convergence is occurring in practice, to determine how many iterations are practically sufficient, or to assess cost-benefit. This gap is especially salient because the theorem is for an idealized algorithm (see below), making empirical convergence evidence all the more necessary.

- **The theoretical guarantee applies to an idealized algorithm, but the prose incorrectly attributes it to Algorithm 1.** Section 3.3 explicitly changes the practical algorithm by returning averaged rather than last-iterate policies, assuming uniform initialization (whereas practice starts from SFT), and ignoring optimization error. These changes are reasonable for theory, but the sentence immediately following Theorem 3.2 states: "Theorem 3.2 demonstrates that **Algorithm 1** can find an $\mathcal{O}(T^{-1/2})$-approximate Nash equilibrium in $T$ iterations" — which is incorrect, as the theorem is for the "theoretical version of Algorithm 1." This prose-level misattribution compounds the theory-practice gap. The paper also provides no empirical proxy (e.g., tracking the Nash gap defined in Eq. 3.6 during training) to check whether the practical PPO-based algorithm exhibits the theoretical behavior.

### Minor

- **Equation (3.5) uses squared norms rather than norms in the denominator, inconsistent with the stated cosine similarity definition.** The text says: "The cosine similarity between two embeddings corresponds to the semantic similarity between the sentences," but Eq. (3.5) reads $\phi(x)\cdot\phi(x') / (\|\phi(x)\|^2 \|\phi(x')\|^2)$, whereas standard cosine similarity has $\|\phi(x)\|\cdot\|\phi(x')\|$ in the denominator. If the sentence embedding model normalizes outputs to unit length (common in SBERT-style models), the formula is functionally equivalent to cosine similarity, but this assumption must be stated explicitly. If embeddings are not unit-normalized, the formula is incorrect. The paper should clarify or correct this, since the diversity reward is a core mechanism.

- **No statistical uncertainty reported across any table.** Tables 1–4 present point estimates with no standard deviations, confidence intervals, or multi-seed results. GPO involves stochastic PPO for two LLMs across multiple outer iterations; small differences such as the MT-Bench scores (6.22 vs. 6.11) cannot be reliably interpreted without uncertainty estimates, and marginal improvements on safety metrics deserve statistical support.

- **Circular evaluation of attack quality in Table 2.** The "unsafe reward" column ($r_{\text{unsafe}}$) in Table 2 is computed by the same toxicity classifier family used to compute the training reward. This means the metric partly measures how well the attacker has optimized the training signal rather than how genuinely dangerous the generated prompts are. The ASR against third-party models is more trustworthy, but the $r_{\text{unsafe}}$ column should be flagged as potentially circular.

- **Training schedule asymmetry is unexplained.** The defender trains for 200 steps and the adversary for 400 steps per game round, starting with the defender. This asymmetry is stated but not justified. In two-player iterative training, schedule balance can substantially affect which agent dominates, making this an important reproducibility and fairness concern.

- **Over-refusal is not tested.** MT-Bench measures general instruction-following quality but does not capture the safety-helpfulness tradeoff directly. A model that over-refuses borderline benign prompts would still score comparably on MT-Bench while being less useful. A refusal rate on benign prompts or a safety-helpfulness benchmark would be more informative for the quality-preservation claim.

### Tiny

- The abstract states "this iterative reinforcement learning optimization converges to a Nash Equilibrium" without qualification. The theorem proves $\epsilon$-NE convergence for an idealized averaged-policy algorithm, not the practical PPO-based implementation. A one-clause qualification would calibrate the claim appropriately.
- Section 3.3 contains confusing wording: "the absence of the entropy regularizer in equation 3.7 causes the adversarial agent to converge to a one-point distribution." Equation 3.7 *includes* the entropy term $-\eta \mathcal{H}(\mu)$. The intended meaning — that a diversity-free variant of the game (Algorithm 1 without $R_{\text{div}}$) leads to point-distribution collapse, whereas Algorithm 3's entropy regularization prevents this — is logically correct but the sentence as written is self-contradictory. This should be reworded.
- The definition and computation of ASR (threshold, span-level vs. response-level, classifier used) are deferred to the appendix. A one-sentence operationalization in the main text would aid reproducibility.

---

## Nice-to-Haves

- **Human evaluation of safety.** Since training and primary evaluation share the same classifier family, even a small human annotation study (100–200 sampled outputs) would validate that safety gains reflect actual behavior rather than classifier gaming.
- **Compute comparison.** A table of approximate GPU-hours for GPO vs. RLHF would help practitioners evaluate the efficiency-performance tradeoff, given that GPO trains two LLMs iteratively.
- **Qualitative examples of adversarial prompts across game iterations.** Showing what the adversary generates at round 1 vs. round 5 vs. round 10 would reveal whether it discovers structurally novel attack strategies or merely amplifies a narrow toxicity template — directly corroborating or challenging the "diverse and challenging" claim.
- **Sensitivity analysis on the asymmetric training schedule.** Ablating step counts (e.g., equal allocation, reversed ratio) would support the claim that the reported schedule is not a confound.

---

## Removed Points
*These points were evaluated and removed; treat them with caution.*

- **"Fully trains agents" is unsupported** (Harsh Critic): This phrase in the abstract is informal language about the training process rather than a technical claim requiring formal definition. The experiments do demonstrate that both agents improve. Removed.
- **"Stale opponent" narrative is too strong** (Harsh Critic): The harsh critic argued that the prose "the defensive agent achieves the highest reward under the prompt distribution given by the adversarial agent" overstates the implemented algorithm. On close reading, this sentence (Section 3.1, final paragraph) is describing what it means to reach Nash Equilibrium — a property of the *equilibrium state*, not of each individual training step. The claim is not that each update is a best response; it is that the converged policy pair satisfies mutual optimality. This is correct. Removed.
- **"Adversarial tutor" framing overstates curriculum** (Harsh Critic): The paper uses "tutor-student" as an intuitive analogy for the introduction, not as a claim that the adversarial training constitutes formal curriculum learning. This is a rhetorical choice, not a technical overclaim. Removed.
- **Entropy discussion in Section 3.3 is "conceptually muddled"** (Harsh Critic): The harsh critic believed the paper contradicts itself about the entropy term. The underlying logic is correct: Algorithm 3 optimizes Eq. 3.7 which includes entropy, preventing adversarial collapse; the diversity-free version of Algorithm 1 would collapse. The wording is imprecise (flagged as a Tiny writing issue above) but the conceptual direction is right. Removed as a substantive weakness.
- **$\beta, \eta = \mathcal{O}(\sqrt{T})$ is "practically awkward"** (Harsh Critic): This is the standard parameter schedule in FTRL and no-regret theory; it is not unusual or problematic in the theoretical context. The paper does not claim this schedule is applied in practice. Removed.
- **Safety reward "too narrow to support broad alignment claims"** (Harsh Critic): The paper explicitly scopes its experiments to safety alignment in Section 3.2, and the conclusion acknowledges this as a limitation. Demanding that a safety-focused paper simultaneously address helpfulness and mathematical reasoning is scope creep. Removed.
- **MT-Bench quality gains are "statistically meaningless"** (Harsh Critic, strong form): The legitimate concern about statistical uncertainty is retained as a Minor weakness above. The extreme form — that the results are worthless — is rejected. MT-Bench provides a useful sanity check that safety training does not catastrophically hurt generation quality. Removed.
- **Novelty relative to iterative red-teaming is insufficiently established at the conceptual level** (Harsh Critic): The paper's conceptual distinction from MART and other automated red-teaming methods is articulated in Section 5. The legitimate concern — that MART is not tested experimentally — is retained as a Major weakness. The purely conceptual form of this criticism is removed.
- **Missing related works** (multiple reviewers): Per review policy, missing related work criticisms are excluded as external sources cannot be independently verified.

---

## Novel Insights

The most structurally revealing signal in the paper, not fully foregrounded by the authors, is the contrast between RLHF+Div and GPO+Div in Table 2. Adding diversity rewards to standard RLHF (RLHF+Div) actually *reduces* attack ASR relative to plain RLHF (33.60% vs. 37.72% on Anthropic) while improving diversity. This indicates that diversity and attack effectiveness are genuinely in tension when the training opponent is weak: diversifying against an easily-satisfied classifier causes the attacker to explore prompts that are varied but not dangerous. GPO+Div breaks this tension — it achieves *both* higher ASR (48.57%) and higher diversity (0.70) than any other method. The mechanism is that the game-theoretic coupling forces the defender to become strong, which in turn creates selection pressure for the adversary to produce prompts that are both diverse *and* effective. This attacker-diversity / attacker-strength tension, and the precise role the game coupling plays in resolving it, is the paper's most compelling empirical finding and deserves to be stated as a central result rather than emerging indirectly from the tables.

---

## Suggestions

1. **Add MART (or an equivalent iterative RLHF attacker + RLHF defender loop) as a baseline in Tables 1–3.** This is the single highest-priority revision. The paper explicitly acknowledges MART in the related work; its absence from experiments is the main barrier to establishing the marginal contribution of game-theoretic co-adaptation over simple iterative red-teaming.

2. **Clarify or correct Equation (3.5).** Verify whether the sentence embedding model used normalizes outputs to unit length. If yes, state this assumption explicitly and note the formula reduces to cosine similarity for unit vectors. If no, correct the denominator to $\|\phi(x)\|\cdot\|\phi(x')\|$.

3. **Add a performance-vs-outer-iteration plot.** Report ASR (Table 1 metric) and attack diversity (Table 2 metric) as a function of the number of GPO rounds $T$ to empirically validate the convergence claim and provide practical guidance on how many rounds are needed.

4. **Fix the prose following Theorem 3.2.** Change "Theorem 3.2 demonstrates that Algorithm 1 can find an $\mathcal{O}(T^{-1/2})$-approximate Nash equilibrium" to accurately state that the theorem applies to the theoretical version of Algorithm 1, and discuss what the gap implies for the practical method.

5. **Report variance across at least two training runs** for all tables. Even a single replication would substantially strengthen confidence in marginal results (especially Table 4).

6. **Rewrite the entropy discussion in Section 3.3.** Clarify that the comparison is between Algorithm 1 *without* diversity reward ($\beta_{\text{div}}=0$) and Algorithm 3 *with* entropy regularization; the current phrasing "absence of the entropy regularizer in equation 3.7" is self-contradictory since Eq. 3.7 contains the entropy term.

7. **Elevate the attacker-diversity/attack-strength trade-off finding.** The contrast between RLHF+Div and GPO+Div in Table 2 is the paper's clearest evidence for the game-theoretic mechanism. Discussing it explicitly as a key result — not leaving it implicit in table numbers — would substantially strengthen the paper's argument.

---

## EW6bNEqalF

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (6.3/10)
- Match: YES

### Final Review

## Summary

This paper studies offline Reinforcement Learning in episodic Regular Decision Processes (RDPs) — non-Markovian environments whose hidden dynamics are captured by a finite-state automaton. The core contribution is a novel **language metric** $L_X$ grounded in formal language theory that replaces the $L_\infty^p$-based statistical test in the prior ADACT-H/RegORL framework, yielding PAC sample complexity bounds that scale with $\log|\mathcal{X}|$ rather than exponentially with the episode length $H$ for structured instances. A secondary contribution applies Count-Min-Sketch (CMS) to reduce the memory footprint of the original suffix-counting approach. Experiments across five domains demonstrate that the language-metric variant achieves superior reward and automaton compactness compared to FlexFringe and the CMS variant.

---

## Strengths

- **Formally proven exponential distinguishability gap (Theorem 1 + Example 4).** The paper rigorously constructs a family of RDPs ($\mathbf{R}_N$) and shows that the $L_\infty^\ell$-distinguishability decays as $\mathcal{O}(2^{-N})$ while the $L_{\mathcal{X}_{2,1}}$-distinguishability is $\Omega(1)$ — an exponential gap. This is a concrete, non-trivial result that directly justifies the language metric rather than merely asserting superiority, and targets a specific structural pathology (distinguishing signal carried by event co-occurrence patterns rather than any individual suffix string).

- **Unifying language metric formalism.** Definition 2 cleanly unifies $L_\infty$, $L_1$, total variation, and both prefix distances as special cases of $L_X$ under different choices of $\mathcal{X}$. This consolidation is genuinely novel and provides a principled analytical lens on the $L_\infty$-vs-$L_1$ tension that motivates the paper.

- **Two-dimensional language hierarchy with formal language roots.** The $\mathcal{X}_{i,j}$ hierarchy — constructed via the $C_k^\ell$ operator inspired by the dot-depth hierarchy of star-free regular languages — is a creative and principled way to interpolate between cheap local tests and richer long-range pattern matching. The connection to formal language theory gives this hierarchy structural justification beyond ad hoc design.

- **Identification and correction of a mistake in Cipollone et al. (2023).** The analysis uncovers a missing $\sqrt{H}/\mu_0$ factor in the original RegORL sample complexity proof. This correction applies both to the prior work and to the new bounds in Theorems 2–3, adding scientific credibility.

- **Clear empirical validation of the scaling claim (Figure 2).** The T-maze corridor-length scaling experiment directly demonstrates linear vs. exponential growth in both runtime and automaton size, with the language-based approach reaching $N=100$ while the CMS approach exceeds 1800 seconds at $H=15$. This is clean, focused experimental evidence directly connected to Theorem 1.

---

## Weaknesses

### Fatal
None identified.

### Major

- **No sample complexity experiments — the headline claim is unvalidated empirically.** The paper's central stated contribution is improved *sample efficiency*, yet every experiment in Table 1 uses a fixed dataset of $K=100$ episodes with no variation. There is no learning curve showing policy reward as a function of $|\mathcal{D}|$, no recovery accuracy vs. dataset size, and no empirical estimate of sample complexity improvement. For a paper whose primary theoretical contribution is Theorem 3 (a PAC sample complexity bound), this is the single most consequential gap between the stated contribution and the empirical evidence.

- **No comparison to RegORL / original ADACT-H.** The paper explicitly frames itself as improving upon RegORL (Cipollone et al., 2023), and the pseudocode of the original ADACT-H (the direct baseline) is included in Appendix A. Yet Table 1 compares only against FlexFringe — a general automata learner with no RL guarantees and different optimization objectives — and against the CMS internal variant. The original ADACT-H with the $L_\infty^p$ test, which is the method the paper claims to surpass, is absent from all empirical comparisons. Without this comparison, the claim to practical improvement over the direct prior method is unsubstantiated.

- **The $1/d_m^*$ term can dominate and may be exponential in $H$, undermining the overall bound.** Both Theorems 2 and 3 scale as $1/d_m^*$, where $d_m^* = \min_{u,a,o} d_t^*(u,a,o)$ is the minimum occupancy of the optimal policy. This can be exponentially small in $H$ if any RDP state is reachable only via a specific chain of transitions — even in fairly structured settings. The paper acknowledges this in one sentence ("The constant $1/d_m^*$ depends exponentially on $H$ if there exists an RDP state that is very hard to reach") but offers no further analysis. If $1/d_m^*$ is the dominant factor, the claimed removal of exponential $H$-dependence via $\log|\mathcal{X}|$ may be illusory end-to-end. At minimum, the authors should characterize the $d_m^*$ regime for the T-maze family, where the benefits are most prominently claimed.

- **Notation error in the estimator definition.** Section 4.1 writes $\hat{p}_1 := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in \mathcal{X}_{i,j})/|\mathcal{Z}_1|$. This is type-inconsistent: $\mathcal{X}_{i,j}$ is a *set of languages*, not a language, so "$e \in \mathcal{X}_{i,j}$" would require the trace $e$ to be a language. The intended definition is clearly: for each fixed $X \in \mathcal{X}_{i,j}$, $\hat{p}_1(X) := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in X)/|\mathcal{Z}_1|$, with the test maximizing over $X$. Since this estimator is the statistical object underpinning Theorem 3, the precise definition must appear correctly in the main text.

### Minor

- **CMS width formula appears inverted relative to standard parameterization.** The paper defines the column width as $w = \lceil \varepsilon/\delta_c \rceil$ (Section 2). The canonical Cormode & Muthukrishnan (2005) parameterization sets width $w = \lceil e/\varepsilon \rceil$, making the sketch *wider* for *smaller* error tolerance. The paper's formula has $w$ growing with $\varepsilon$, which has the opposite sense. If this is a deliberate reparameterization suited to the specific application, it requires explicit justification, since the CMS guarantees are invoked in the proof of Theorem 2.

- **No memory measurements for the CMS variant.** Theorem 2's claimed practical advantage over vanilla ADACT-H is *memory reduction*. Yet Table 1 reports only runtime and automaton size — not peak memory usage. The primary claimed benefit of the CMS contribution is empirically unquantified.

- **No ablation over language hierarchy parameters $(i,j)$.** All experiments exclusively use $\mathcal{X}_{3,1}$ without justification. The $\mathcal{X}_{i,j}$ hierarchy is presented as a key structural contribution, but there is no empirical demonstration of when $\mathcal{X}_{1,1}$ suffices or when larger $j$ is needed. The hierarchy remains a conceptual device rather than an empirically validated design tool.

- **No characterization of when Assumption 1 holds in practice.** Assumption 1 requires $\mu_0 > 0$ for the chosen $\mathcal{X}_{i,j}$. The paper demonstrates this for T-maze with $\mathcal{X}_{2,1}$ (Theorem 1), but provides no general characterization of which RDP classes admit positive $L_X$-distinguishability for small $j$ while having exponentially small $L_\infty^p$-distinguishability. Practitioners cannot determine when to apply the method without this guidance.

- **Chain from RDP recovery to policy optimality is implicit in the main text.** Theorems 2–3 establish recovery of the minimal RDP with high probability; the reduction to an $\varepsilon$-optimal policy guarantee is deferred entirely to the prior RegORL construction in Appendix A. The main text should state explicitly how RDP recovery translates into policy suboptimality bounds, including the contribution of the failure-probability event.

### Tiny

- The objective in Section 2.3 writes $V_\circ^*(h) - V_{\hat{\pi}}^*(h) \leq \varepsilon$ — the subscript appears to be a formatting artifact for $V_0^*(h) - V_0^{\hat{\pi}}(h)$ and should be corrected.
- The conclusion says the language approach "remov[es] the dependency on $L_\infty^p$-distinguishability parameters." More precisely, it *replaces* that dependency with $L_X$-distinguishability under the chosen $\mathcal{X}$, which must still be assumed positive.

---

## Nice-to-Haves

- **Guidance or heuristic for selecting $(i,j)$ at deployment time.** A model-selection criterion using held-out log-likelihood on a validation split of $\mathcal{D}$ could potentially identify the sufficient complexity level without prior knowledge of the RDP structure.

- **Visualization of learned automata on T-maze.** Showing the automaton recovered by $L_X$ vs. the CMS/$L_\infty^p$ approach on T-maze would make tangible why the language metric recovers correct structure while the baseline fails — directly connecting the motivating example to the learned artifact.

- **Results at larger $H$.** The paper claims exponential gains in $H$; all Table 1 domains have $H \leq 15$. Even two results at $H = 30$–$50$ would strengthen the practical relevance claim and demonstrate the approach beyond toy scales.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"Comparison to sequence-model baselines (RNNs, transformers) is missing"** (Harsh Critic). This is scope creep. The paper explicitly scopes itself to methods with formal PAC guarantees and notes that sequence-model approaches "lack correctness guarantees." Evaluating against them is orthogonal to the stated contribution.

- **"PSR connection is underleveraged"** (Harsh Critic). The paper appropriately cites the PSR connection and notes that existing PSR bounds involve different parameters. Demanding deeper structural comparison to PSRs is outside the paper's scope.

- **"T-maze state definition is malformed / difficult to parse"** (Harsh Critic). The apparent structural ambiguity in the state set definition in Example 3 is almost certainly a PDF text-extraction artifact; the example is coherent in substance and the construction is consistent with the corridor dynamics described.

- **"Notation is inconsistent among $L_\infty^\circ$, $L_\infty^p$, $L_\infty^\ell$"** (Harsh Critic). These are genuinely distinct metrics in the paper: $L_\infty^\circ$ uses prefix matching (any trailing suffix, defined in Section 2.2), $L_\infty^p$ is the prefix distance used in ADACT-H's test, and $L_\infty^\ell$ is the $L_\infty$ over strings of exact length $\ell$. The notation is deliberate and internally consistent, not erroneous.

- **"FlexFringe comparison is unfair due to heuristics"** (implicit, Harsh Critic). The paper explicitly acknowledges that FlexFringe uses heuristics that do not preserve sample complexity guarantees, and notes "The RDPs output by FlexFringe are not always directly comparable." This comparison is intentionally asymmetric in FlexFringe's favor — FlexFringe being computationally less constrained makes it a strong practical baseline, not a weak one. Beating it strengthens the paper's claims.

- **"No statistical significance for entries reporting 1.0 and 4.0"** (Harsh Critic). In episodic environments such as Corridor ($H=5$) and T-maze(c) with a fixed goal structure, an optimal policy can achieve deterministic reward. Reporting a point value without variance in these cases is appropriate.

- **"Contribution statement mixes theory and implementation without clearly isolating novelty"** (Harsh Critic). The paper separates the language metric and CMS contributions clearly in both the abstract and the contributions paragraph. This is a stylistic critique without technical substance.

---

## Novel Insights

The most genuinely novel observation synthesized across the reviews is that the standard $L_\infty^p$-based statistical test for state merging in automaton learning is not merely "practically inconvenient" but is *provably catastrophic* for a natural and important class of RDPs: those where the distinguishing signal is carried by event co-occurrence patterns (e.g., observing a specific reward upon a specific action *anywhere* in a suffix) rather than by any single suffix string. The language metric $L_X$ — by measuring probability mass over *sets* of strings defined by pattern-matching rather than individual strings — captures exactly the structural feature that makes T-maze-like domains tractable. The connection to the dot-depth hierarchy of star-free regular languages provides a theoretically grounded organizing principle, and the $\mathcal{X}_{i,j}$ hierarchy is a concrete implementable instantiation. A key open question raised by this synthesis is whether $1/d_m^*$ can be tamed for the same structured RDPs where $1/\mu_0$ improves via the language metric — if yes, the end-to-end sample complexity improvement would be genuinely polynomial in $H$.

---

## Suggestions

1. **Add learning curves.** Run each domain with dataset sizes $K \in \{10, 25, 50, 100, 250, 500\}$ and plot average reward vs. $K$ for Language metric, CMS, and original ADACT-H. This is the most important missing experiment and directly validates the sample efficiency claim.

2. **Include original ADACT-H ($L_\infty^p$ test) as a baseline in Table 1.** Since the pseudocode is already in Appendix A and all three variants share the same codebase, this comparison is straightforward and is necessary to substantiate the claim of improvement over RegORL.

3. **Fix the estimator definition.** Replace $\hat{p}_1 := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in \mathcal{X}_{i,j})/|\mathcal{Z}_1|$ with the correct per-language formulation: for each $X \in \mathcal{X}_{i,j}$, define $\hat{p}_1(X) := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in X)/|\mathcal{Z}_1|$, and state the test explicitly as the maximum discrepancy over $X \in \mathcal{X}_{i,j}$.

4. **Justify or correct the CMS width formula.** Clarify the parameterization $w = \lceil \varepsilon/\delta_c \rceil$ relative to the standard Cormode & Muthukrishnan (2005) formulation, and verify that downstream proofs use the correct guarantee.

5. **Add memory measurements for CMS.** Include peak memory usage alongside runtime in Table 1, since reduced memory is the primary theoretical advantage of Theorem 2.

6. **Ablation over $(i,j)$.** Add a table or figure showing how reward and automaton size vary as $(i,j)$ is changed on one or two domains, to ground the hierarchy empirically and guide practitioners.

7. **Characterize the $d_m^*$ regime for T-maze.** Add a short analysis or corollary showing what $d_m^*$ equals for the T-maze family as a function of $N$, to determine whether the end-to-end sample complexity bound is indeed polynomial in $H$ for this motivating example.

---

## slZZnzlITo

- GT: Reject (avg 4.8)
- Predicted: Reject (4.8/10)
- Match: YES

### Final Review

## Summary

This paper proposes Multimodal Open Set Recognition (MMOSR), extending OSR to multimodal input settings (image-text, audio-visual, RGB-depth). The authors empirically demonstrate that naively combining multimodal fusion with standard OSR regularization causes "fusion degradation," where OSR's compaction pressure suppresses modality-specific representations and reduces unknown detection ability. To address this, they introduce the Multimodal Representation Reactivation Network (MRN), combining a cross-attention-based mutually enhanced fusion module with an MoE-based adaptive fusion module, and benchmark it across four datasets against a broad set of baselines.

---

## Strengths

- **Concrete evidence of a previously unreported failure mode.** Table 1 presents a specific, reproducible result: applying OpenAUC on top of addition-fused multimodal features (Fusion-OSR) degrades AUROC by up to 5.60 points versus naive Fusion alone, and underperforms the single text modality (Text-OSR achieves 91.57 vs. Fusion-OSR's 86.43 at the 20-class split). This is a genuine empirical finding that motivates the paper's direction regardless of the theoretical explanation.

- **Broad multi-dataset, multi-modality benchmark construction.** The paper evaluates on image-text (Food-101, Flower-102), audio-visual (CREMA-D), and RGB-depth (SUN RGB-D) data, covering three fundamentally different modality types. The baseline comparison spans single-modal OSR, multimodal fusion, fusion+OSR combinations, and pretrained vision-language models (CLIP, CoOp, MaPLe). This breadth is a practical contribution to the community beyond what most OSR papers offer.

- **MRN as a drop-in backbone consistently improves paired OSR methods.** When MRN replaces other fusion strategies underneath ARPL and CSRR (Table 2, "Multimodal fusion with OSR methods" rows), it improves both AUROC and OSCR consistently across all four datasets and both OSR methods — including CREMA-D (ARPL-MRN: 64.37/56.79; CSRR-MRN: 66.86/56.90, both best in their group). This is a more robust demonstration of MRN's value than the standalone comparison.

- **MRN, trained from scratch, outperforms large pretrained vision-language models on fine-grained OSR.** Table 3 shows MRN consistently exceeds CLIP (zero-shot), CoOp, and MaPLe (16-shot fine-tuned) across all known/unknown splits on Food-101, despite having substantially fewer pretraining resources. This is a non-obvious and practically meaningful finding about the limits of foundational model transfer to OSR-specific tasks.

---

## Weaknesses

### Fatal
None.

### Major

- **CREMA-D failure is unreported and actively misrepresented.** In Table 2, standalone MRN scores 66.78 AUROC and 57.32 OSCR on CREMA-D, while MLA scores 67.83 AUROC and 57.50 OSCR — MRN is worse on both metrics. The Gain row explicitly confirms this with (1.05↓) and (0.18↓). Yet the paper bolds the MRN row for CREMA-D and in the text claims MRN "consistently demonstrates exceptional MMOSR performance across various datatypes." This is a factual misrepresentation: the caption says "best results are marked in bold," but MRN's CREMA-D values are bolded despite not being best. The paper provides no discussion of this failure or why the reactivation mechanism does not help for audio-visual data.

- **No statistical significance reporting for marginal gains.** Gains on SUN RGB-D (+0.37 AUROC, +0.01 OSCR over MLA) and Food-101 (+0.72 AUROC, +1.38 OSCR) are very small; no standard deviations, confidence intervals, or multiple-seed results are reported anywhere. Without variance information, it is impossible to determine whether these differences are meaningful or within random variation. This is particularly damaging alongside the CREMA-D failure.

- **Ablation study does not cover the adaptive fusion (MoE) component.** Table 4 ablates only cross-attention branches C1 and C2; the very first row (neither C1 nor C2) already includes the adaptive fusion module. There is no experiment removing MoE entirely or substituting it with an equivalent-capacity single MLP. The MoE adaptive fusion is half the proposed method, and its independent contribution to MMOSR performance is entirely unverified.

- **The core motivating experiment (Section 3.2) is too narrow.** The "fusion degradation" analysis uses only one dataset (Food-101) and one OSR method (OpenAUC). The paper then generalizes this to a universal phenomenon and uses it to justify a new task. CREMA-D in Table 2 does not exhibit the same pattern (MLA, a pure fusion method, outperforms MRN there). A robust motivating claim requires evidence across multiple modality pairs and multiple OSR methods in the motivating section.

- **The most natural alternative baseline is absent: score-level fusion of per-modality OSR.** Running OSR independently on each modality and combining rejection scores (e.g., taking the max/product/mean of per-modality MSP) is the most obvious alternative to early-fusion MRN. Without this baseline, there is no evidence that the proposed architectural complexity is necessary, rather than simply ensembling single-modal decisions.

### Minor

- **Unknown rejection relies entirely on standard MSP thresholding, with no OSR-specific mechanism.** Section 4.3 uses maximum softmax probability with a percentile threshold. The training loss (Eq. 3) is classification plus load balancing, with no open-set objective, margin shaping, or uncertainty term. The paper frames MRN as an MMOSR method, but its open-set detection is post-hoc and no different from applying MSP to any classifier. The gains may stem purely from better representations; this should be acknowledged and the scoring function examined.

- **Threshold calibration is underspecified.** Section 4.3 states τ is "set to ensure 95% of the known samples are correctly classified," but does not specify which split (training, validation, or held-out known set). If the training set is used, behavior at test time may differ; if a validation set is used, the protocol must be stated for reproducibility. This also raises the question of sensitivity: how much do AUROC/OSCR change under different percentile thresholds?

- **"Fusion degradation" is not quantitatively operationalized.** The concept is central to the paper's motivation but defined only via t-SNE plots and one performance table on one dataset. No measurable quantity — such as effective feature rank, inter-class margin, modality contribution entropy, or feature norm statistics — is computed to verify the phenomenon or confirm it is resolved by MRN.

- **Grad-CAM comparison (Figure 7) is against a single-modal baseline (ARPL), not a multimodal one.** Comparing MRN's visual attention maps to ARPL — which operates on one modality — cannot demonstrate that MRN's cross-modal reactivation mechanism is responsible for better attention. A comparison against MLA or GQA would be far more informative.

- **Ablation study reports ACC but not OSCR.** Table 4 evaluates the fusion modules using AUROC and ACC. For a paper framed around MMOSR, ablations should report OSCR, which jointly evaluates open-set detection and closed-set classification, rather than ACC alone.

### Tiny

- Sensitivity analysis figures (Figures 4 and 5) show metric curves without error bars, making stability claims qualitative.
- No computational overhead analysis (parameter count, FLOPs, inference time) is provided for the MoE component, relevant given the motivating deployment scenario of robotic systems.
- The text in Section 3.2 refers to both Figure 2c (Fusion) and 2d (Fusion-OSR) while writing "Fusion-OSR methods over-compress," slightly obscuring which phenomenon applies to which model.

---

## Nice-to-Haves

- **Quantitative measurement of fusion degradation**: Compute feature rank, intra-class compactness, or modality contribution entropy before/after OSR regularization and after MRN reactivation across multiple datasets. This would transform a qualitative narrative into a verifiable mechanistic claim.
- **Alternative OSR scoring functions on top of MRN representations**: Compare energy-based scores, logit margins, or prototype distances on MRN features to determine whether MSP is a bottleneck and whether a dedicated scoring function could recover CREMA-D performance.
- **Deeper characterization of CREMA-D failure**: Analyze whether audio-visual modality structure differs in ways that cause the cross-attention "class-relevant = cross-modally correlated" assumption to break down — e.g., modality-private prosodic cues that are discriminative but not aligned with visual frames.
- **Extension of Section 3.2 to at least two modality pairs**: Include CREMA-D or SUN RGB-D in the motivating fusion degradation analysis, even if the result there is weaker, to establish the generality and boundary conditions of the phenomenon.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **Training from scratch disadvantages pretrained baselines** (Critic): The paper explicitly states this choice is "to avoid introducing unknown information," and the design is applied uniformly to all methods in Table 2. Pretrained models appear only in Table 3 as a separate comparison group, where the asymmetry favors those baselines (they get pretraining; MRN does not). This is an intentionally conservative comparison that strengthens, not weakens, the paper's claims. Removed per the rule on unfair comparisons that favor baselines.
- **Missing related works on multimodal OOD detection / multi-view uncertainty** (Critic, Spark): Removed per instructions — cannot confirm existence of specific references without external sources.
- **"MLLMs era" discussion is aspirational** (Critic): The forward-looking paragraph in Section 5.3.3 is appropriately framed as future motivation and does not make empirical claims. Not a methodological flaw.
- **GQA not being an appropriate fusion baseline** (Critic): GQA is a published method used here as a multimodal fusion architecture with grouped-query attention; its use as a fusion baseline is reasonable and its existence is confirmed by the citation.
- **Equation 1 tensor shape ambiguity** (Critic): The cross-attention formulation follows standard conventions; the intent is unambiguous and this is a minor notation concern, not a reproducibility failure at the level warranting inclusion.
- **Broader impact section missing**: Not relevant to technical merit evaluation at this venue.
- **"The new task is just a reframing"** (Critic): The benchmark construction across four heterogeneous datasets, with reproducible experimental protocols and the new MMOSR framing, constitutes a genuine contribution independent of how novel the formal definition is.

---

## Novel Insights

The paper's most transferable insight is a *negative* finding: OSR regularization, whose purpose is to compact known-class representations to leave decision space for unknowns, is structurally at odds with what multimodal fusion needs — diverse, modality-specific representations. The concrete evidence in Table 1 (Fusion-OSR underperforming both Fusion and the single best modality at AUROC) is a novel empirical finding with architectural implications beyond this paper. A second observation from Table 3 is equally noteworthy: well-trained task-specific multimodal fusion models, even those trained entirely from scratch, substantially outperform large pretrained vision-language models (CLIP, CoOp, MaPLe) across all tested known/unknown ratios on fine-grained OSR tasks, challenging the assumption that foundation models transfer effectively to rejection-aware recognition. Together, these observations — that OSR and early fusion interact adversarially, and that task-specific training dominates large-scale pretraining in this setting — provide a useful framing for follow-on work.

---

## Suggestions

1. **Correct and discuss the CREMA-D reporting.** Remove erroneous bolding of MRN in Table 2 for CREMA-D, acknowledge the underperformance explicitly in the text, and analyze why audio-visual data may not benefit from cross-attention reactivation (e.g., modality-private discriminative cues).

2. **Add a score-level fusion baseline.** Run OSR independently per modality and combine rejection scores via max/product/average pooling. Report this on all four datasets. This is the first experiment a reviewer will ask for and its absence is the paper's most conspicuous gap.

3. **Report mean ± std over multiple random known/unknown splits.** Even three to five seeds would allow readers to assess whether marginal gains (e.g., +0.37 AUROC on SUN RGB-D) are reliable. This is especially necessary given the CREMA-D failure.

4. **Add a proper MoE ablation row to Table 4.** Replace the MoE module with a single MLP of matched parameter count, and report AUROC and OSCR (not just ACC). Without this, the contribution of the adaptive fusion component cannot be assessed.

5. **Expand Section 3.2 to include at least one additional modality pair.** Rerun the Fusion vs. Fusion-OSR motivating experiment on CREMA-D or SUN RGB-D to establish whether fusion degradation is a general phenomenon or specific to image-text data with a dominant modality.

6. **Quantify fusion degradation with a concrete metric.** Compute effective feature rank or inter-class margin statistics for Fusion, Fusion-OSR, and MRN across datasets. This would convert the central motivating claim from a qualitative observation into a verifiable and replicable finding.

---

