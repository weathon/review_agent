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

## HozsY9Gdcl

- GT: Reject (avg 5.0)
- Predicted: Accept (5.8/10)
- Match: NO

### Final Review

## Summary
This paper introduces **Set-MI**, a wrapper method that improves membership inference (MI) for language models by aggregating per-document MI scores over metadata-defined groups ("sets") under the assumption that all documents in a set are either entirely in or entirely out of the training corpus. The authors construct five new MI benchmarks spanning Wikipedia, Arxiv, language identity, data license, and instruction-tuning datasets, and demonstrate an average AUROC gain of 0.14 over four Individual-MI baselines. Additional ablations analyze the effects of model size, training-data deduplication, document length, and set size, plus a controlled robustness analysis under simulated membership noise.

---

## Strengths

- **Broadly applicable wrapper design.** Set-MI is formulated as a zero-overhead wrapper over any existing Individual-MI scoring function, verified empirically across four qualitatively different base methods (Loss Attack, LiRA, Min-K%, zlib entropy). The positive correlation between Individual-MI and Set-MI performance (r = 0.824, p = 0.0002) also provides an actionable prescription: improving base MI methods will compound into larger Set-MI gains.

- **Novel benchmark suite.** To the authors' knowledge, these five benchmarks are the first set-structured MI benchmarks for LMs, and collectively the most domain-diverse. Covering temporal cutoffs (Wikipedia/Arxiv), language identity, license category, and fine-tuning datasets addresses a real gap since prior work typically evaluates on a single domain.

- **Concrete motivating examples grounded in real data-pipeline practice.** The paper provides specific, verifiable examples (DOLMA's March 2023 Reddit cutoff, SILO license categories, Tulu instruction-dataset composition) rather than toy constructions, making the set assumption credible for practitioners.

- **Insightful deduplication finding.** The result that deduplication widens the gap between Duplicated and Deduped models *more* for Set-MI than for Individual-MI is a genuinely novel empirical observation about memorization dynamics that is not obvious a priori and has implications for data-curation research.

- **Scaling analysis.** The finding that Set-MI benefits disproportionately more from larger model sizes, while Individual-MI improvement is modest, adds concrete empirical content to the general understanding that memorization scales with model capacity.

---

## Weaknesses

### Fatal
None. The core contribution is sound and the empirical gains are real.

### Major

- **Benchmark statistics are internally inconsistent, harming reproducibility.** Table 1 states Wikipedia has 1,000 sets / 100,000 documents and Arxiv has 1,000 sets / 100,000 documents, but the construction text says "We subsample 100 sets with 100 documents per set" for each — giving 10,000 documents, not 100,000, and 100 sets, not 1,000. For the Language benchmark, the text says "resulting in 130 sets" but 20 languages × 10 subsets = 200 sets, matching Table 1 (200 sets / 20,000 docs). For License, the text again says "resulting in 130 sets" but 19 source datasets × 10 subsets = 190 sets, matching Table 1 (190 / 19,000). These appear to be systematic copy-paste errors in the text, but until resolved a reader cannot reproduce the benchmarks or interpret the scale of the evaluation. This must be corrected for the paper to be replicable.

- **No uncertainty quantification.** The entire empirical contribution rests on AUROC point estimates, but no confidence intervals, bootstrap intervals, standard deviations, or significance tests are reported anywhere. This matters because many improvements are modest in absolute terms (e.g., Loss Attack on Wikipedia: 0.524 → 0.575; zlib on License: 0.647 → 0.674), and each estimate is computed from a single random 1,024-token span per document. Without error bars it is impossible to assess whether these gains represent reliable signal or sampling noise. This is a core requirement for an empirical claim paper, not a methodological nicety.

- **Several benchmarks plausibly measure domain/distribution shift rather than true membership.** (a) *Language*: Bloom's per-language loss differences could reflect tokenizer coverage and overall language competence rather than document-level membership, because the model was trained with language selection as a first-order design choice. (b) *License*: different license categories often correspond to qualitatively different dataset topics and writing styles, so a model may separate them without memorizing any specific documents. (c) *Instructions*: the target model (Tulu-v1) is fine-tuned rather than pretrained on the instruction datasets, and the set label is the dataset identity itself — the model may recognize the format of an unseen ShareGPT conversation without that specific conversation being in training. These confounds do not invalidate the benchmarks outright, but the paper provides no control (e.g., removing the distributional cue while keeping the membership cue) to disentangle the two effects. The scientific claim — "Set-MI leverages membership signals" — requires at least acknowledging and, ideally, partially ruling out this alternative explanation.

- **Robustness analysis (Section 6) is too narrow to support general claims.** The robustness experiments use a single base method (Loss Attack), a single domain (Wikipedia), and a single model (Pythia 2.8B-dedup) under synthetic noise generated by random replacement of members/non-members. This does not substantiate the claim that Set-MI is robust "under practical settings" across the paper's five benchmarks and four base methods. Different base methods have different tail shapes and calibration, and real-world violations of the set assumption (version updates, partial crawls, deduplication artifacts) do not follow a uniform random replacement model.

### Minor

- **The zlib + Set-MI failure on Instructions (0.458 → 0.429, below random) is not adequately analyzed.** The paper lists this number in Table 2 and notes the general caveat that poor Individual-MI can hurt Set-MI, but provides no domain-specific explanation for why zlib specifically fails here while the other three methods improve. This is a direct counterexample to the claim that "Set-MI significantly improves Individual-MI on most settings," and understanding it would strengthen the paper.

- **Date-based sets create near-perfect correlation between the grouping variable and the membership label.** For Wikipedia and Arxiv, documents are labeled as members iff their creation date precedes the collection cutoff, and sets are defined by creation date. Set-MI on these benchmarks is therefore largely testing whether the model encodes the temporal training boundary, not individual document memorization. The 13-gram overlap validation in Section 6 partially addresses this for the robustness experiment, but not for the main results in Table 2. The paper should discuss whether temporal-cutoff monotonicity is the dominant driver of gains on these two benchmarks.

- **The 30% threshold for MAX/MIN aggregation in Section 6 is unjustified.** No sensitivity analysis is provided and it is not clear whether this value was selected on the same data used for evaluation. A brief justification or sensitivity curve is needed.

- **No random-set control.** Aggregating over any batch of documents reduces estimator variance. The paper does not show that aggregating over randomly assembled (non-membership-correlated) groups fails to achieve similar gains. Such a control would confirm that the set assumption — and not mere variance reduction — is what drives performance.

### Tiny

- A single random 1,024-token span is drawn per document for all experiments. The sensitivity of AUROC estimates to this random draw is not reported. Even a brief note on the variance across multiple draws (or evidence that results are stable) would strengthen confidence in the reported numbers.

- The set-size ablation (Figure 4 right) keeps set count fixed while varying set size, but does not control for total tokens observed. A comparison at equal total token budget (e.g., 1 doc × 1,024 tokens vs. 4 docs × 256 tokens) would better isolate the benefit of the set assumption from the benefit of observing more tokens.

---

## Nice-to-Haves

- **Random-set baseline.** Run Set-MI by grouping documents into sets with no membership correlation (e.g., random date assignments) and compare AUROC. This would directly quantify how much of the gain is due to meaningful shared membership versus pure averaging noise reduction.

- **Evaluation on a model with genuinely unknown training data.** All target LMs have published training-corpus details used to define ground truth. Applying Set-MI to a semi-unknown model (e.g., using later-revealed information about GPT-2 or an early LLaMA checkpoint) would validate the practical use case more convincingly.

- **Score distribution visualization.** Plotting per-document score distributions for member vs. non-member sets before and after aggregation would make it visually clear whether aggregation genuinely separates the classes or uniformly shifts all scores.

- **More sophisticated aggregation alternatives.** Median, trimmed mean, or a simple confidence-weighted average are natural competitors to the mean that might be more robust to outliers. The paper explores MAX/MIN only in the noise robustness section; briefly comparing these alternatives in the main experiment would strengthen the design choice.

- **Automatic set discovery.** The current method requires a practitioner to know which metadata attribute defines membership-correlated sets. A brief discussion or preliminary experiment on clustering-based set discovery would broaden applicability.

- **LiRA reference-model sensitivity.** The paper notes that finding a good reference model is difficult in practice. A brief ablation varying the reference model quality would clarify whether LiRA-based Set-MI is stable or highly sensitive to this choice.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **"The method is too simple for ICLR"** (Harsh Critic): Simplicity is not a disqualifying weakness when the empirical contribution and benchmark construction are substantive. Many impactful systems papers succeed through principled evaluation of simple methods. Removed.

- **"Missing stronger or more recent LM-specific baselines"** (Harsh Critic): No specific missing method is named; this is a generic criticism that applies to any paper. The four methods chosen span the main families of black-box MI scoring (loss, LiRA, n-gram, compression). Removed.

- **"Loss Attack notation uses probability rather than log-probability"** (Harsh Critic): The paper consistently uses probability notation throughout Section 2.2 for all methods and Figure 2 illustrates the same. While log-probability is more common in practice, this is a presentation choice, not a technical error. Removed.

- **"Demanding theoretical proofs for when averaging improves AUROC"** (Harsh Critic): This paper is an empirical systems contribution. Requesting formal proofs of AUROC improvement under averaging is not a standard expectation for this type of work. Removed.

- **"Calibration/threshold discussion required"** (Harsh Critic): The paper evaluates with AUROC throughout, which is threshold-free. Requesting threshold calibration analysis is a nice-to-have at best and not a substantive weakness. Removed.

- **Strength: "the paper is well-written / the topic is important"** (Generic): These are not retained as named strengths as they apply to virtually any paper in the area.

---

## Novel Insights

The most insightful observation that emerges from synthesizing all three reviews is that Set-MI's gains on date-based benchmarks (Wikipedia, Arxiv) may reflect the model having internalized the **temporal training boundary** — a smooth monotonic signal — rather than per-document memorization. This interpretation, if correct, changes the scientific claim: the paper would partly be showing that LMs encode coarse temporal data-selection policies, which can be recovered by aggregating loss signals over time-cohort groups. This is itself a meaningful finding, but it is distinct from (and arguably stronger and more tractable than) the document-level membership inference framing. A clean disentanglement — for example, using 13-gram overlap labels for Wikipedia/Arxiv in the main experiments rather than only in Section 6 — would clarify whether gains come from temporal policy recovery, genuine per-document memorization, or both, and would substantially sharpen the paper's narrative.

---

## Suggestions

1. **Fix benchmark statistics:** Reconcile Table 1 with the construction text. For each domain, state clearly whether the reported numbers are the full collected benchmark or the experimental subsample, and ensure these are consistent throughout.

2. **Add uncertainty quantification:** Report AUROC estimates with 95% bootstrap confidence intervals (or at minimum standard deviations across random token draws). For large-scale benchmarks where this is expensive, report at least a small-scale variance study.

3. **Add a random-set control:** Include one experiment where Set-MI is applied over randomly composed groups (size-matched to real sets). This is a cheap but decisive experiment that verifies the mechanism.

4. **Use 13-gram overlap labels in the main Wikipedia/Arxiv experiments:** Since this cleaner labeling is already computed for Section 6, applying it to the main results table would partially address concerns about proxy ground truth and temporal confounding.

5. **Analyze the zlib + Instructions failure case explicitly:** Identify whether this failure is due to zlib Individual-MI being below 0.5 (and thus aggregating in the wrong direction), or whether something about the Instructions domain specifically causes Set-MI to degrade. A sentence or two in the main paper would suffice.

6. **Widen the robustness analysis to at least two base methods and two domains:** Even adding Min-K% on Arxiv alongside the existing Loss Attack on Wikipedia setup would substantially strengthen the generalizability of Section 6's claims.

7. **Explicitly separate the temporal-boundary effect from per-document memorization:** Use 13-gram overlap labels in the main Wikipedia/Arxiv experiments, and add a discussion of whether removing the exact temporal ordering of sets changes the AUROC, to verify whether the set assumption is doing the work rather than temporal monotonicity.

---

## cojJ2s1e35

- GT: Reject (avg 5.5)
- Predicted: Accept (5.2/10)
- Match: NO

### Final Review

## Summary
WLA (World modeling through Lie Action) introduces an unsupervised framework that models video transitions as linear Lie group actions in an object-centric, slot-partitioned latent space, enabling continuous and compositional dynamics. A single shared model is trained across multiple environments simultaneously, and a lightweight supervised adapter (`Ctrl_adapt`) maps user-specified action labels to the learned Lie algebra parameters. The method is evaluated on Phyre (qualitatively), ProcGen (quantitatively against Genie on 8 games), and the 1X Android robotic dataset.

---

## Strengths

- **Single multi-environment model with strong quantitative gains:** The paper trains one model across all 8 ProcGen environments and outperforms Genie substantially on both temporal metrics (Δ_t PSNR: e.g., 9.03 vs. 0.48 in coinrun; 4.06 vs. 0.05 in ninja) and LPIPS in 7 of 8 environments (Table 2). This is a concrete demonstration of cross-environment representation sharing, not merely an architectural story.

- **Novel synthesis of Lie group structure with object-centric modeling:** The combination of slot attention with per-slot Lie algebra dynamics—where each slot evolves under structured rotation+scaling operators—is architecturally distinct from both pure slot models and generic Koopman/state-space approaches. The explicit connection to equivariant autoencoders (Eq. 2) provides a principled theoretical grounding.

- **Least-action slot alignment principle:** The proposal to resolve temporal slot permutations by solving a linear assignment problem that minimizes the Lie-algebra operator norm is novel and motivated. The ablation (Table 1) confirms it reduces MSE meaningfully on both seen and unseen environments.

- **Compelling Android robotics results:** The dramatically better FVD (131.02 vs. 393.85 for Genie, Table 3) and better Δ_t PSNR on real-world robot video indicate that the temporal coherence advantages generalize beyond synthetic game environments. The tradeoff (worse per-frame PSNR but much better FVD) is consistent with the hypothesis that WLA better captures action-conditional dynamics rather than static frame quality.

- **Unsupervised pretraining + modular adapter design:** The decoupling of structure learning (unsupervised, label-free) from action mapping (`Ctrl_adapt`, small and supervised) is a clean and reusable design. The ablated version without rotation is explicitly noted to resemble diagonal-SSM models (Mamba), situating the contribution clearly.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Unseen-environment evaluation is critically thin.** The headline claim of cross-environment generalization is only partially supported. Table 1 (right) reports ActionACC for unseen ProcGen environments, but full metrics (PSNR, Δ_t PSNR, LPIPS) for unseen settings are absent from the main paper. This is a direct evidentiary gap: the core generalization claim rests almost entirely on the seen-environment results in Table 2.

- **Single baseline throughout.** The paper compares exclusively against Genie in all experiments. For an ICLR submission claiming a new structured-dynamics framework, this is insufficient. Critically absent: (a) an ablation with general linear (non-Lie) transition operators to isolate the rotation+scaling bias contribution; (b) an object-centric video predictor with MLP transitions to separate the object-centric contribution from the Lie structure contribution; and (c) a continuous latent-action model (e.g., LAPO, which directly addresses unsupervised action discovery). Without these, the source of WLA's improvements—whether structure, object-centricity, or something else—cannot be attributed.

- **No few-shot label scaling experiment despite "minimal labels" claim.** The abstract and introduction prominently claim "minimal or no action labels" for adaptation, but Section 4.3 and the experiments never vary the number of labeled sequences used to train `Ctrl_adapt`, report the actual label budget used, or show a performance curve as a function of label count. This claim cannot be evaluated, making it the most poorly substantiated assertion in the paper.

- **Long-horizon rollout degradation is unanalyzed.** All quantitative results appear to be over fixed 16-step rollouts. The exponential latent dynamics (Eq. 9) could compound errors at longer horizons, yet no rollout-length vs. error analysis is presented. For a paper positioned as a "world model," this analysis is essential to understand practical utility.

- **Training details for per-sample (λ, θ) parameters are insufficient for reproducibility.** Footnote 3 notes that these trajectory-specific Lie algebra parameters "are not to be stored as parts of the model," implying per-trajectory optimization during training. However, the paper never explains how these are initialized, whether they are optimized jointly with network weights at every gradient step, what the computational overhead is, or whether there is any test-time inference procedure. These are essential for reproducibility and for understanding the effective inductive bias.

- **Ablation study is incomplete.** Only two ablations are tested (rotation and least-action). Missing ablations that are necessary to understand the contribution: (a) no object slots (fully dense latent vs. slotted), (b) no shared cross-environment training (per-environment model vs. joint), and (c) shared vs. per-environment `Ctrl_adapt`. These are needed to verify whether the object-centric decomposition and the cross-environment sharing are each adding value.

### Minor

- **ActionACC values are low in absolute terms and the scale is under-explained.** The paper reports 21.07 (Ours) vs. 10.25 (Genie) for seen-environment ActionACC (Table 1 right). While WLA doubles Genie's score, 21% absolute accuracy on a classification task is low, and the number of action classes is not stated in the main text. The paper should clarify the chance-level performance for context. If there are 5 classes, 21% is near chance; if there are 15 classes, it is better. This is important for interpreting whether `Ctrl_adapt` is actually learning useful action correspondence.

- **Phyre evidence is entirely qualitative.** Phyre is presented as validation of the core continuity and compositionality claims, yet only cherry-picked frames are shown (Figures 3 and 4). There is no numerical interpolation error, no composition error metric, and no baseline comparison. This weakens the foundational empirical support for the paper's inductive bias.

- **Android experimental protocol is underdescribed.** The paper says the architecture was "slightly adapted" for the Android dataset without specifying what was changed, what action space is used, or how sequences are split. Without this, the Android results cannot be reproduced.

- **Identifiability of (λ, θ) not discussed.** Nothing in the training objective (Eq. 7–9) prevents the encoder from absorbing dynamics while the Lie algebra parameters become weakly meaningful, or produces unique/stable representations across runs. The paper should discuss whether the learned parameters are stable or if degenerate solutions are observed.

- **Eq. (3) ordering notation.** The Fact states F(h·g) = F(g)·F(h), reversing the standard group homomorphism order M(hg) = M(h)M(g). Whether this is intentional (e.g., right action convention) or an error should be clarified explicitly, as it affects how compositionality is interpreted.

- **Key hyperparameters N and J absent from main text.** The number of slots and rotation components used in each experiment are relegated to the appendix but are central to understanding model capacity and reproducibility.

### Tiny

- The claim "the first of its kind as a generative interactive framework that is based on a state-space model" (Section 7) is overreaching and should be softened or precisely scoped given the substantial related SSM/Koopman literature.
- The commutativity assumption in Eq. 9 (∑A[ℓ] inside the matrix exponential) should be foregrounded in Section 4 alongside the formal equations where it is used, rather than deferred to the limitations in Section 7.

---

## Nice-to-Haves

- **Stochastic extension.** Extending the Lie algebra parameters (λ, θ) to distributions (e.g., Gaussian) would address environmental stochasticity—a stated limitation—and broaden applicability to RL settings. This is noted as future work and would be a natural extension.
- **Visualization of (λ, θ) trajectories alongside ground-truth actions.** Showing whether learned Lie algebra parameters cluster by action type or are interpretably disentangled would strengthen the "compositional and continuous action representation" claim and provide important mechanistic insight.
- **Commutativity violation analysis.** A controlled experiment measuring composition error when ground-truth action sequences are explicitly non-commutative (e.g., "up then right" vs. "right then up") would quantify when the core assumption holds and when the model is expected to fail.
- **Sensitivity analysis for N and J.** A brief sweep over the number of slots and rotation components would allow future practitioners to set these hyperparameters for new domains.

---

## Removed Points
*These points are flagged as removed; treat them with caution.*

- **[REMOVED] Genie comparison unfairness (doubled training iterations).** The harsh critic raises the concern that Genie was given 0.4M training iterations instead of the original 0.2M. Per review policy, comparisons that are asymmetric in favor of the baseline (Genie receives more compute) are intentionally stronger baselines and do not constitute a weakness of the paper. The authors explicitly state this was done to accommodate multi-environment training.

- **[REMOVED] No reconstruction loss per frame.** The harsh critic claims there is no reconstruction loss on x[t] independently of the prediction loss. This is incorrect: the forward and backward prediction losses (Eq. 8) include reconstruction of all frames x[t] via rolled-out latent dynamics, which amounts to frame-level reconstruction supervision.

- **[REMOVED] Statistical rigor / confidence intervals.** Requesting confidence intervals or multiple-seed results for ProcGen evaluations is not standard practice in the video generation and world modeling community, where single-run evaluation on fixed benchmarks is the norm.

- **[REMOVED] Missing related works.** Specific related works were requested by reviewers; per policy, we do not evaluate claims about missing citations without access to external literature.

- **[REMOVED — formatting/scope] Formalism mismatch in CIP (Eq. 1 type signature vs. history input).** The paper explicitly acknowledges the abuse of notation in the text following Eq. (1) and Section A provides a formal definition. While the mismatch is slightly confusing, it is an acknowledged notation convenience, not a substantive error.

- **[REMOVED — acknowledged] Deterministic environment assumption.** Fully acknowledged as a limitation in Section 7 with a proposed future direction (stochastic process modeling). It is a real constraint but not a hidden flaw.

- **[REMOVED] Human analogy in introduction as scientific evidence.** The harsh critic flags the human analogy ("after mastering basic movements in a few 2D action-adventure games…") as unscientific. This is standard motivational framing, not a methodological claim, and is appropriately cited with cognitive science references.

---

## Novel Insights

The most genuinely novel structural insight in this paper—underemphasized even by the authors—is the connection between the ablated "w/o rotation" WLA variant and diagonal-state-space models like Mamba. By explicitly identifying that restricting to scaling-only Lie group actions collapses the framework to a diagonal SSM, the paper provides a principled generalization of the SSM family toward richer, non-diagonal structured dynamics. This framing suggests that the rotation+scaling Lie group structure is not just an arbitrary inductive bias but a specific augmentation of the SSM with rotational degrees of freedom in the latent space, which could motivate a broader class of structured world models. The spark finder's observation that the (λ, θ) parameters could be interpreted and visualized to verify whether the Lie algebra dimensions correspond to semantically meaningful action axes (e.g., θ for orientation, λ for speed/magnitude) is a valuable diagnostic not pursued in the paper—its absence is a missed opportunity to substantiate the "compositional representation" claim mechanistically.

---

## Suggestions

1. **Report full unseen-environment metrics.** Add PSNR, Δ_t PSNR, and LPIPS for out-of-domain ProcGen environments to Table 2 or a companion table. The data likely already exist given the setup.
2. **Add a label-efficiency experiment.** Plot `Ctrl_adapt` performance (Δ_t PSNR or ActionACC) as a function of the number of labeled trajectories (e.g., 1, 5, 10, 50, 100). This single experiment would directly validate the "minimal labels" claim.
3. **Add at least one structural ablation baseline.** Either a general linear (non-rotation-constrained) transition operator, or an object-centric MLP dynamics model, is needed to isolate the Lie group contribution from the object-centric contribution.
4. **Clarify and quantify the per-sample (λ, θ) training procedure.** Explain whether these are optimized per-batch via gradient steps, amortized via a recognition network, or otherwise. Include training time and memory requirements relative to Genie.
5. **Clarify ActionACC scale.** State the number of action classes, chance-level accuracy, and compute a normalized metric (e.g., accuracy above chance) so readers can interpret the absolute numbers in Table 1.
6. **Include long-horizon rollout analysis.** Report a metric (e.g., MSE or PSNR) as a function of rollout length (e.g., 4, 8, 16, 32 steps) for at least one ProcGen environment to characterize error accumulation.
7. **Visualize learned (λ, θ) parameters.** Show scatter plots or trajectory plots of the inferred Lie algebra parameters colored by ground-truth action class for a ProcGen environment to test whether the latent action space is disentangled and semantically interpretable.

---

## lNuGCXxvkn

- GT: Reject (avg 5.2)
- Predicted: Accept (5.5/10)
- Match: NO

### Final Review

## Summary

This paper develops an asymptotic Sobolev-norm learning curve for kernel ridge and ridgeless regression applied to elliptic linear inverse problems governed by PDEs. The central theoretical finding is that the PDE forward operator — by amplifying high-frequency components — effectively stabilizes the variance of min-norm interpolators, enabling benign overfitting in fixed spatial dimensions where standard regression would produce tempered or catastrophic overfitting. A secondary contribution is characterizing how the choice of Sobolev-norm inductive bias (parameter β) affects convergence, establishing a smoothness threshold above which the rate becomes independent of the specific inductive bias, and showing this threshold matches one previously identified in the Bayesian inverse-problem literature.

---

## Strengths

- **Fixed-dimensional benign overfitting via PDE structure.** Theorem 4.2 and Remark 7 constitute a genuine and specific finding: the negative exponent p of the differential operator shifts the variance bound exponent from `max{λβ', −1}` (pure regression) to `max{2p + λβ', −1}` (inverse problem), and since p < 0, this can push variance below the regression baseline even without dimensional growth or kernel engineering. This mechanism — the inverse problem operator acting as a spectral smoother — is clearly articulated through the spectral transformation Σ̃ = A²Σ^β and gives a principled reason for variance stabilization that is new relative to prior kernel-interpolation analyses.

- **Unified regularized + ridgeless framework recovering known rates.** The paper analyzes both ridge-regularized (Theorem 4.1) and min-norm interpolating (Theorem 4.2) estimators in a single spectral framework. Critically, Remark 5 shows the regularized bound reproduces the minimax-optimal rate from Lu et al. (2022) at the optimal γ, providing a meaningful sanity check that the framework is correctly calibrated. Simultaneously extending to interpolators in the same setting, and establishing where the dominant terms depend vs. do not depend on β, is a non-trivial analytical step.

- **Smoothness threshold matching the Bayesian literature.** The finding that the threshold λβ ≥ λr/2 − p — above which the convergence rate becomes independent of the inductive bias — coincides with the analogous condition identified in Bayesian inverse problems (Knapik et al., 2011; Szabó et al., 2013) and with empirical understanding in semi-supervised learning is a surprising and useful connection. It elevates the result from an isolated technical bound to a structurally motivated condition. Extending this threshold to the ridgeless/interpolating regime is new.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Bounded-observation assumption inconsistent with Gaussian noise model.** Assumption 2.2(a) stipulates that observations y are almost surely bounded by M, yet Section 3 explicitly sets ε ∼ N(0, σ²I_{n×n}), making y unbounded almost surely. If proofs use boundedness to invoke standard concentration inequalities, the Gaussian model is not formally covered by the stated assumptions. The paper needs to either relax Assumption 2.2(a) to sub-Gaussian or finite-variance noise, or clarify that the Gaussian model satisfies the technical conditions actually used in the proofs (rather than the stated ones).

- **Critical dependence of benign-overfitting claims on ρ_{k,n} is underemphasized.** The headline claim — benign overfitting in fixed dimension — is established under Theorem 4.2, but both the variance bound (scaled by ρ²_{k,n}) and bias bound (scaled by ρ³_{k,n}) critically depend on the concentration coefficient. Remark 6 acknowledges that ρ_{k,n} = Θ(1) requires sub-Gaussian features, and in the worst case can grow as Õ(n^{2p+βλ−1}), which can substantially weaken or even eliminate the benign-overfitting conclusion. The abstract and main body consistently present benign overfitting as a consequence of the PDE structure, without foregrounding that this requires a separate, non-trivial assumption on the feature behavior. The paper should state clearly, in the main theorems, under exactly which feature conditions the benign-overfitting exponents hold, and what happens in the worst case for ρ_{k,n}.

- **Experiments are too limited and indirect to substantiate the theoretical claims.** All experiments are conducted on a single 2D Poisson equation with one ground-truth function, using finite-width neural networks — not the kernel estimators the theory covers. There are no kernel experiments, no systematic variation of PDE order p (which is the central determinant of variance stabilization), no variation of the inductive bias parameter β directly (activation smoothness is an indirect proxy), and no comparison of regularized vs. interpolating estimators under controlled conditions. For a theory paper at ICLR, this leaves the theory entirely unvalidated in its own setting and makes Figure 1(Left)/(Middle) illustrative rather than confirmatory. At minimum, a controlled synthetic kernel experiment — e.g., Matérn kernel + Laplacian operator with known spectral decay — varying n at different values of p and β to check the predicted exponents, would substantially strengthen the paper.

- **No lower bounds; benign vs. tempered regime is one-sided.** All results are upper bounds. "Benign overfitting" in the strict sense requires that risk vanishes, but without matching lower bounds the upper bounds may be loose. It is not possible from the present results to determine whether the benign/tempered/catastrophic trichotomy is tight or an artifact of proof looseness, especially given the max{·,−1} and max{·,−2p+λ(β'−2β)} exponents in Theorem 4.2. The paper should acknowledge this limitation, or present even partial lower bounds for the variance in the inverse-problem interpolation setting.

### Minor

- **Theory-to-experiment gap is not adequately bridged.** The experiments use overparameterized finite-width neural networks, and the connection to kernel regression is justified only informally via NTK heuristics. The paper frames Figure 1 as validating theory "beyond kernel methods" but this is an overstatement; the NTK approximation is not verified to hold in this inverse-problem setting, and the activation function smoothness is at best a proxy for the spectral decay parameter β. Section 5 should explicitly label these results as heuristic evidence rather than validation of the stated theorems.

- **Diagonalizability assumption (Assumption 2.2(d)) limits scope more than framing suggests.** The requirement that A and Σ be simultaneously diagonalizable is strong and excludes most practical geometries beyond the torus. While this is standard in theoretical kernel-based inverse-problem analysis (acknowledged in Remark 2), the paper's broad framing around "physics-informed machine learning" and PINNs is at odds with this restriction. A short discussion of what happens qualitatively when this assumption fails, or under which conditions it approximately holds, would calibrate the reader's expectations.

- **Practitioner takeaway about activation smoothness is heuristic, not theorem-derived.** Section 4.3 states that higher-order PDEs "require smoother activation functions," presenting this as a consequence of the theory. However, the formal results are for kernel estimators with prescribed spectral decay; the link between activation smoothness and the β parameter for finite neural networks is not established analytically. This guidance should be explicitly labeled as conjectural.

### Tiny

- The abstract says "the convergence rate is actually independent to the choice of (smooth enough) inductive bias" without the qualifier that this independence holds for β above a threshold and is subject to the ρ_{k,n} caveat. The abstract should reflect these conditions.
- No dedicated limitations section; the key caveats (diagonalizability, kernel-only theory, ρ_{k,n} dependence) are scattered across remarks rather than consolidated.

---

## Nice-to-Haves

- **Spectral visualization of the transformed kernel eigenspectrum.** A plot of Ã² Σ^β eigenvalues vs. standard kernel Σ eigenvalues for a concrete example (e.g., Matérn + Laplacian) would make the variance-stabilization mechanism intuitive and help readers assess whether the concentration assumptions hold in practice.
- **Bias-variance decomposition plots.** Plotting bias and variance separately vs. n for both PINN and standard NN interpolators would directly validate the claim that the PDE operator specifically suppresses variance, rather than improving the combined risk through other mechanisms.
- **Explicit benign vs. tempered parameter-regime table.** A table stating: for these ranges of (p, λ, r, β, β'), the variance exponent is negative (benign); for these ranges it is only bounded (tempered); for these it diverges (catastrophic) — would make Theorem 4.2 much more accessible and clarify the practical scope of the main result.
- **Characterization of ρ_{k,n} for physics-informed kernels.** Showing that for, e.g., Matérn kernel + Laplacian on the torus the sub-Gaussian feature condition holds, and thus ρ_{k,n} = Θ(1), would close the gap between the theoretical claim and the setting for which it is actually established.

---

## Removed Points

*These points are flagged to be removed; treat them with caution as they either misread the paper or are overly pedantic.*

- **Threshold inconsistency between Section 1.1 and Remark 5.** The harsh critic flags that Section 1.1 writes `λβ ≥ λ^r/λ^p − p` while Remark 5 writes `λβ ≥ λr/2 − p`. The former is almost certainly a PDF-to-text parsing artifact of `\frac{\lambda r}{2}` rendering as `λ^r/λ^p` (which would equal λ^{r−p}, a nonsensical expression in context). The Remark 5 expression is the coherent one and appears consistently in the applications. This is not a scientific inconsistency.

- **Novelty is "just a reparameterization."** The critic suggests the results might follow by a black-box application of Barzilai & Shamir (2023) to the transformed kernel K̃. However, the extension to the inverse-problem operator setting, the derivation of the closed-form representer theorem under the Sobolev norm with operator A, the introduction of the concentration coefficient for the transformed kernel Σ̃ = A²Σ^β, and the derivation of threshold conditions involving p are genuine technical contributions, not trivial substitutions.

- **Concern about φ and ψ basis confusion.** The critic raises notation inconsistency between L₂ and RKHS inner products. The paper defines φ_i = √λ_i ψ_i as the RKHS basis and ψ_i as the L₂ eigenbasis, and introduces ψ-maps and φ-maps explicitly with their relationships stated. The notation is dense but internally consistent given careful reading; this is not a mathematical error.

- **Claim that experiments should include confidence intervals/error bars.** For neural-network experiments demonstrating qualitative phenomena (noise profiles, convergence trend with activation smoothness), single-run plots are acceptable; demanding multiple-run statistics here would be an atypical rigor requirement for this type of demonstration figure.

- **Criticism of the "first rigorous upper bound" claim.** The paper scopes this claim specifically to min-norm kernel interpolators for fixed-dimensional physics-informed settings. Within that narrow scope, the claim is plausible; the critic's objection is largely about lack of a comprehensive literature survey, which does not mean the claim is false.

---

## Novel Insights

The most significant insight synthesized from the reviews goes beyond the paper's own framing: the paper implicitly establishes a *spectral duality* between the forward PDE operator and the inductive bias in the interpolation regime. The forward operator A with p < 0 amplifies high-frequency components in the forward direction, but this amplification means the inverse-problem objective *penalizes* high-frequency errors more heavily, effectively acting as spectral regularization without any explicit regularizer. This is structurally dual to adding Sobolev norm regularization: increasing |p| has the same qualitative effect on variance as increasing β. The smoothness threshold condition λβ ≥ λr/2 − p makes this duality precise — p and β enter symmetrically in the admissibility condition. This perspective suggests that for practitioners, the choice of activation smoothness (proxy for β) and the PDE order (p) should be co-designed, and that for sufficiently high-order PDEs, relatively weak inductive bias may suffice for benign behavior. The connection to the Bayesian threshold suggests this duality may be fundamental rather than an artifact of the proof technique.

---

## Suggestions

1. **Add at least one synthetic kernel experiment** on a 1D or 2D Poisson / Schrödinger problem using the actual kernel estimator (Matérn or RBF kernel + discretized Laplacian), varying n at multiple values of p and β, and plotting empirical excess risk against the theoretically predicted exponents. This is the single highest-value addition to strengthen the paper.

2. **Resolve the bounded-y / Gaussian-noise inconsistency** in Assumption 2.2(a). Either extend the assumption to sub-Gaussian/finite-variance noise and verify the downstream inequalities hold, or add a footnote in Section 3 explaining that the proof uses only finite-variance properties and Gaussianity is for concreteness.

3. **Make Theorem 4.2 self-contained with a regime table.** Add after Theorem 4.2 a corollary or remark that explicitly lists the parameter conditions under which (i) V → 0 and B → 0 (benign), (ii) V bounded but nonzero (tempered), and (iii) V → ∞ (catastrophic), so readers can immediately identify which operating regime applies to their problem.

4. **Characterize ρ_{k,n} for at least one concrete kernel-operator pair.** Show (in the appendix) that the shift-invariant Matérn kernel on the torus with the Laplacian satisfies the sub-Gaussian feature condition, so ρ_{k,n} = Θ(1) in that case, providing a complete end-to-end statement of benign overfitting.

5. **Separate heuristic from rigorous in Section 4.3.** Label the practitioner guidance on activation smoothness and higher-order PDEs explicitly as "conjectural extension" and reference the NTK connection, to avoid overstating what is formally established.

6. **Add a limitations paragraph** consolidating: (i) joint diagonalizability of A and Σ; (ii) kernel theory vs. NN experiments gap; (iii) ρ_{k,n} assumption; (iv) linear self-adjoint elliptic operators only.

---

## F6s7OApF0n

- GT: Reject (avg 4.8)
- Predicted: Accept (5.8/10)
- Match: NO

### Final Review

## Summary

Cost-sensitive Multi-fidelity BO (CMBO) proposes a framework that reframes hyperparameter optimization from maximizing asymptotic validation performance under a fixed budget to maximizing a user-defined utility function that trades off BO performance against computational cost. The method introduces a utility-aware EI acquisition function with a dynamically chosen optimization horizon, a probabilistic stopping criterion interpolating between regret-based and probability-of-improvement signals, and a transfer learning scheme for Prior-Fitted Networks (PFNs) based on learning curve (LC) mixup across datasets and configurations. Extensive evaluation across LCBench, TaskSet, PD1, and a collected real-world object-detection dataset shows consistent and substantial improvements over multi-fidelity BO and transfer-BO baselines.

---

## Strengths

- **Genuinely novel problem framing for multi-fidelity BO.** Existing freeze-thaw methods (DyHPO, iFBO, DPL) optimize either greedy one-step EI or asymptotic performance at a fixed horizon. Reformulating the objective as maximizing a user utility U(b, ỹ_b) over the BO trajectory—and deriving both acquisition and stopping from this objective—is a conceptually clean and practically relevant departure. The dynamic horizon selection (max over Δt in Eq. 2) is a direct consequence of this framing and is not present in prior work.

- **LC mixup preserving inter-configuration correlations.** The two-stage mixup—across datasets first with a shared λ₁ applied to all configurations, then across configurations—is a simple yet thoughtful technique. Using a single shared λ₁ in the first stage explicitly preserves the correlation structure encoded in each dataset's LC matrix L_m. Fig. 6 directly demonstrates that the mixup reduces overfitting of the PFN surrogate and translates to improved BO regret, providing concrete evidence for the mechanism rather than just end-to-end performance.

- **Coherent interpolation between two extreme stopping rules.** The BetaCDF(p_b; β, β)^γ formulation in Eq. (4) provides an interpretable one-parameter family: β→0 recovers the regret-only threshold used by baselines (δ_b = 0.2 when γ = log₂5), while β→∞ recovers a hard PI-based threshold. The smooth interpolation at β = e⁻¹ is well-motivated and tested across all three benchmarks in Fig. 7d.

- **Empirical breadth and real-world validation.** The method is tested on four distinct LC benchmarks spanning tabular classification (LCBench), diverse NLP tasks (TaskSet), large-scale vision and biology tasks (PD1), and a self-collected object detection dataset with three heterogeneous architectures. Multiple utility function shapes (linear, quadratic, square root, staircase, estimated) are evaluated, and Table 2 shows robustness across all of them with CMBO achieving rank 1.0 in every setting.

---

## Weaknesses

### Fatal
None.

### Major

- **Table 3 ablation inconsistency undermines the incremental gain narrative.** Rows 3 and 4 in Table 3 are both labeled (p_b ✓, Acq. ✓, T. ✓) yet produce very different results (e.g., 4.4 vs. 0.9 for α=2e-04). The text claims "performance improves sequentially as each component is added," implying four distinct configurations, but only three distinct checkmark patterns appear (missing one intermediate ablation row). This makes it impossible to cleanly attribute gains to the stopping criterion vs. acquisition vs. transfer learning, which is a core claim of the paper. At minimum, one row is mislabeled, and the correct labels must be provided for the ablation to be interpretable.

- **Stopping-criterion confound in baseline comparisons.** As the paper notes in footnote 2, the PI-based term in Eq. (5) depends on the utility-aware acquisition, so baselines cannot directly use the same stopping rule. This is a principled justification, but it means the cost-sensitive results in Table 1 and Fig. 5 conflate: (i) better configuration selection from the utility-aware acquisition, and (ii) better stopping from the mixed PI+regret criterion. The ablation in Table 3 shows the stopping criterion alone contributes substantially (4.4→0.9 at α=2e-04), yet the baseline comparison gives baselines only the inferior regret-only stopping. A cleaner decomposition—e.g., showing CMBO's acquisition with regret-only stopping vs. full CMBO—would make the contribution boundaries clearer. The dotted "achievable regrets without stopping" lines in Fig. 5 are helpful but insufficient to resolve this.

- **Uniform per-step cost assumption undermines the motivating scenario.** The utility U(b, ỹ_b) is defined over BO steps b and evaluated over "total epochs spent." This implicitly assumes each BO step (each epoch evaluation) has identical cost. The paper's motivating examples invoke cloud credits and Slurm allocations, where wall-clock cost is the relevant resource. The real-world object detection experiment includes ResNet-50, HRNet, and MobileNetV2 evaluated jointly, which almost certainly have different per-epoch wall-clock costs. The mismatch between the motivation (heterogeneous wall-clock costs) and the formulation (step counting) is a genuine gap that goes unacknowledged in the main text.

### Minor

- **Utility elicitation is empirically under-validated.** The Bradley-Terry preference model is demonstrated only via synthetic recovery in Fig. 2 (1,000 pairwise labels, no sensitivity to fewer/noisier labels). In the main experiments, all reported results use predefined utility functions (linear, quadratic, etc.). The single "Estimated" row in Table 2 constructs preferences from iFBO's trajectory assuming "the user wants a better tradeoff than iFBO"—this is an artificial proxy for user preference, not evidence that the end-to-end pipeline (elicitation → BO → stopping) works in practice.

- **Algorithm 1 notation inconsistency.** Line 4 reads: n* ← argmax_{n∈C} A(n), where C = {(x, t, y)} is the history of partial LC observations. At initialization C = ∅ (line 2), making this argmax undefined. Furthermore, the text in §3.1 says "we predict for all x∈X the remaining part of the LCs," implying the argmax should range over the full configuration pool X, not C. This discrepancy should be corrected for reproducibility.

- **Notation inconsistency between Eq. (2) and Eq. (5).** Eq. (2) uses ỹ_{b+Δt} (tilde-y, the best-so-far BO performance), while Eq. (5) uses ȳ_{b+Δt} (bar-y). Whether these are identical quantities should be clarified explicitly, as the distinction between the running best performance (line 10: ȳ_b = max(ȳ_{b-1}, y_{n*,t_{n*}})) and the extrapolated BO performance matters for the stopping criterion computation.

- **ESBO baseline is undefined.** ESBO appears in Tables 2 and 4 but is not described anywhere in the baselines section (§4). Its definition, source, and relationship to CMBO (it appears to be a strong baseline in Table 4) must be provided; its absence from Table 1 also suggests it is not applicable in all settings, which should be explained.

- **γ parameter is fixed without sensitivity analysis.** β is ablated in Fig. 7d across all three benchmarks, but γ is fixed at log₂5 (corresponding to δ_b = 0.2) without any analysis. As γ and β jointly determine the stopping behavior, a sensitivity test on γ is warranted.

### Tiny

- **Mixup validity for discrete/categorical hyperparameters is not discussed.** Convex combinations of configuration vectors (step 2 of the mixup) may produce invalid hyperparameter settings when some hyperparameters are categorical or integer-valued. The paper should at minimum state that this is applied only to continuous hyperparameters or discuss how categorical cases are handled.

- **Key PFN architecture details are deferred to appendices (§E, §G) that are not available in the main text.** Architecture size, tokenization of partial LCs, number of meta-training examples, and inference procedure are relevant for assessing the method's practical overhead and reproducibility.

---

## Nice-to-Haves

- **EI/cost as a baseline.** The standard cost-aware acquisition divides EI by expected evaluation cost; including it (even as a black-box surrogate variant) would clarify whether the utility formulation offers advantages beyond this simpler cost-weighting approach.

- **Utility trajectory visualization with oracle stopping point.** Showing U(b, ỹ_b) over BO steps for CMBO and baselines, with the actual stopping point marked and the oracle b* indicated, would directly demonstrate whether the stopping criterion is well-calibrated.

- **Sensitivity to utility misspecification.** A brief analysis of how CMBO degrades when the estimated utility deviates from the true utility (e.g., wrong penalty weight α) would quantify the practical risk of the elicitation approach.

- **Wall-clock time experiment.** One experiment with actual per-configuration compute time (rather than epoch counting) would validate the cost-sensitivity claim in a realistic heterogeneous-cost setting.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **Zero variance (±0.0) as a credibility concern.** Quick-Tune† and FSBO are deterministic methods; ±0.0 is expected and not suspicious. The paper explicitly uses 30 runs only for methods with large variance.

- **β ablation limited to PD1 only.** This misreads Fig. 7d, which plots normalized regret vs. β for LCBench, TaskSet, PD1, and Average simultaneously, with asterisks marking optima for each. The ablation covers all three benchmarks.

- **Demand for theoretical guarantees.** This is an empirical systems paper in the freeze-thaw BO tradition; no prior competing method provides theoretical stopping or regret guarantees, and demanding them would impose a non-standard bar.

- **Finite configuration pool being too restrictive for "general BO."** All freeze-thaw BO methods operate in this setting (DyHPO, iFBO, DPL). The paper targets tabular HPO benchmarks where this is the standard setup; criticism for not covering continuous-space BO is scope creep.

- **Criticism about Quick-Tune† modification being unfair.** The modification removes the model-selection component to isolate the transfer learning mechanism, which makes the comparison fairer for the hyperparameter selection task—the baseline is weakened in a way that benefits it (more compute per HP eval), not the proposed method.

- **Concern about comparison to methods not yet released / not existing.** The paper cites iFBO, Quick-Tune, DPL, etc.—these are assumed to exist.

---

## Novel Insights

The acquisition function's dynamic horizon selection (Eq. 2: max over Δt) induces a principled behavioral transition from non-greedy to greedy over the course of BO—not by scheduling but as a direct consequence of cost-dominated utility. Fig. 7b visualizes this concretely: early BO steps select large Δt (look-ahead), while late steps collapse to Δt≈0 (myopic exploitation) as the cost term dominates. This is a cleaner explanation of the exploration-exploitation transition in cost-sensitive BO than ad hoc schedule designs, and the analysis in Fig. 7c shows the resulting configuration-selection concentration matches intuition. The LC mixup's cross-dataset shared λ₁ to preserve inter-configuration correlation is a subtle but nontrivial design choice that distinguishes it from naive per-curve augmentation and deserves attention from the broader PFN training community.

---

## Suggestions

1. **Fix Table 3**: Identify and correct the mislabeled row so the four ablation rows correspond to four distinct component combinations, enabling clean attribution of gains to stopping criterion, acquisition function, and transfer learning separately.

2. **Clarify Algorithm 1 initialization**: Specify how the first configuration is selected when C = ∅, and correct the domain of the argmax in line 4 from n∈C to n∈[N] (or n∈X) to match the described procedure.

3. **Address uniform-cost assumption explicitly**: Add a paragraph acknowledging that U(b, ỹ_b) treats per-step cost as homogeneous, and either justify this for the benchmarked settings or describe a simple extension to variable per-step costs (e.g., replacing step count b with cumulative wall-clock time).

4. **Define ESBO**: Add a description of the ESBO baseline in §4, clarify its relationship to CMBO, and explain why it appears only in Tables 2 and 4.

5. **Add stopping criterion decomposition**: Report CMBO with regret-only stopping (β→0) alongside baselines with the same stopping rule, to isolate acquisition quality improvements from stopping policy improvements in the cost-sensitive setting.

---

## AnPEfzBstD

- GT: Reject (avg 3.5)
- Predicted: Reject (3.8/10)
- Match: YES

### Final Review

## Summary
This paper presents a large-scale empirical benchmark comparing 1D (MolFormer/LLM), 2D (MPNN), and 3D (Equiformer v2) molecular representations for Bayesian Optimization in materials discovery. Across four datasets (QM7, QM9, GEOM MoleculeNet, GEOM DRUGS) with multiple surrogates (GP, LLA), the authors find that simpler 1D and 2D representations generally match or outperform 3D, and that 3D representations require considerably more training data to close the gap. The benchmark spans over 2100 runs and also investigates transfer learning and sample-complexity regimes.

---

## Strengths
- **Fills a genuine, documented gap in BO benchmarking**: Prior BO benchmarks for molecular discovery (Olympus, Summit, Griffiths et al. 2024) explicitly omit 3D representations. This work is the first systematic study that adds equivariant GNNs (Equiformer v2) to the comparison, across both GP and LLA surrogates—a non-trivial engineering effort not attempted elsewhere.
- **Sample complexity analysis has a concrete mechanistic anchor**: The finding in Section 5.2 that 3D (equivariant) models are substantially less data-efficient than 2D models is grounded in existing theory (Elesedy & Zaidi, 2021) and is demonstrated across all four datasets with four training-set sizes. This is the most original and actionable finding in the paper.
- **Multi-axis evaluation**: The study simultaneously varies representation dimensionality, surrogate type (GP vs. LLA), data regime (four sizes), and task type (single-property vs. transfer learning), producing structured evidence rather than a single-condition comparison.
- **Reproducibility practice**: 15 seeds per condition with reported standard errors, and an anonymous code repository, are above the norm for this class of benchmarking paper.

---

## Weaknesses

### Fatal
None. The core empirical findings are real, but a central methodological confound severely limits the scope of the strongest headline claims (see Major #1).

### Major

- **The 1D vs. 3D comparison conflates representation dimensionality with model scale and pretraining**. MolFormer is a masked language model pretrained on 1.1 billion SMILES strings (Ross et al. 2022), while the 2D/3D GNNs are constrained to ~1.5 million parameters trained from scratch on the benchmark tasks. The paper never accounts for this asymmetry. The dominant finding—"LLM/1D outperforms 2D and 3D"—is therefore at least as likely to reflect large-scale pretraining as it is to reflect anything about 1D representation dimensionality. This conflation is not minor: it makes the paper's central framing ("is 3D a step too far?") largely unanswerable from the presented experiments as the comparison is not isolating the dimensionality axis. The 2D vs. 3D comparison (both at ~1.5M parameters) is the paper's most internally fair comparison, and its conclusions should be foregrounded accordingly.

- **No computational cost measurements, despite cost being a core claim**. The paper repeatedly argues that 3D's "computational overhead" outweighs its gains—this framing appears in the abstract, introduction, results, and conclusion. Yet no wall-clock times, GPU hours, or FLOPs are reported anywhere. The cost claim is entirely qualitative. For a paper whose thesis is explicitly about cost–accuracy trade-offs, this is not a stylistic gap: without cost numbers, the trade-off cannot be evaluated.

- **Conformer handling for 3D models is never described**. GEOM datasets provide multiple conformers per molecule. The paper does not state which conformer is used for Equiformer v2 inputs—lowest-energy, random, or some other selection. This decision materially affects 3D model performance; if ground-truth minimum-energy conformers are used, 3D has an oracle advantage unavailable in real BO settings. If poor conformers are used, 3D's underperformance may reflect data quality rather than dimensionality. The confound directly undermines the interpretation of 3D vs. 2D results.

- **Task selection bias undermines the generality of the main conclusion**. All four benchmark targets—atomization energy (QM7), HOMO-LUMO gap (QM9), absolute energy (MoleculeNet/DRUGS)—are quantum mechanical scalar properties that are primarily determined by molecular topology and composition, not by specific 3D conformation. The paper itself acknowledges in the conclusion that "future research should focus on tasks where 3D information might be more important, e.g. protein docking." This acknowledgment, however, is not sufficient: it means the paper's headline "3D is a step too far" is tested only on tasks where 3D is not theoretically expected to win. The finding is valid for these tasks, but should not be presented as a general verdict on 3D representations in BO.

- **The acquisition function is never specified**. Section 4 describes datasets, feature extractors, and surrogates in detail, but never names the acquisition function (EI, UCB, Thompson sampling, etc.) or its hyperparameters. For a BO benchmark, this is a reproducibility-critical omission that prevents independent replication of any individual run.

### Minor

- **Only one 3D architecture tested**. The paper draws conclusions about "3D representations" using only Equiformer v2. Poor results could reflect architecture-specific failure modes (e.g., insufficient expressive power for the GP/LLA interface, initialization sensitivity) rather than a dimensionality-level verdict. Including even one additional 3D model (SchNet, DimeNet) would substantially strengthen the claim.

- **Factual inconsistency between abstract/conclusion and body**: The abstract and conclusion state "LLMs consistently outperformed" all methods, but Section 5.1 explicitly says "LLMs performed worse than 2D and 3D models" on QM9. This is not a minor phrasing issue—it is a factual contradiction that misleads readers who read only the abstract or conclusion.

- **Transfer learning analysis is incomplete and overclaims**: Section 5.3 and Fig. 5 show transfer learning results only for QM7 and QM9. Yet the text draws general conclusions about transfer learning and invokes "foundation model" potential. The claim "Foundation models prove a good tool" is overstated from two datasets with a limited fine-tuning protocol (only the final layer).

- **Potential MolFormer data leakage not investigated**: MolFormer was pretrained on 1.1 billion SMILES from ZINC and PubChem. QM9 and MoleculeNet molecules are small, well-known, and could plausibly appear in those corpora. If so, MolFormer's strong performance could partly reflect memorization rather than generalization. A membership overlap check is warranted.

- **Sample complexity analysis (Section 5.2) excludes the 1D/LLM comparison**: Despite MolFormer being a key performer, the sample complexity plots compare only 2D vs. 3D. If the paper's goal is a comprehensive 1D/2D/3D benchmark, omitting LLM from this axis is inconsistent.

- **GP kernel on learned embeddings is not specified**. For the GP surrogate using pretrained/trained feature embeddings, the kernel (RBF, Matérn, ARD, etc.) is never stated. This matters because kernel choice interacts with embedding geometry and directly affects uncertainty calibration.

### Tiny

- **Laplace approximation notation is inconsistent**: The paper writes $p(\theta|\Omega_t) \approx \mathcal{N}(\theta_*, \Sigma_*^{-1})$ and then defines $\Sigma_*^{-1} = -\nabla_\theta^2 \log p(\theta|\Omega_t)$, treating $\Sigma_*^{-1}$ simultaneously as a covariance parameter and as the Hessian. Standard convention is to write $\Sigma_* = H^{-1}$ where $H$ is the (positive-definite) negative log-posterior Hessian. The current notation will confuse readers.
- **"35 setups per dataset" is never broken down**. The abstract and introduction cite this number prominently, but the main text never enumerates the exact combination of representation × surrogate × regime × seed that generates the count. An explicit table in the appendix would make the benchmark auditable.
- **GAP metric notation inconsistency**: The definition uses $y_i$, $y_0$, $y_*$ but the body text refers to $y^*$. Minor but worth fixing.

---

## Nice-to-Haves

- **Include at least one genuinely conformation-dependent task** (e.g., docking score, stereoselective reaction yield, conformer-dependent binding affinity). Even a single dataset where 3D is theoretically expected to win would transform the negative results into a more principled and bounded statement rather than a potentially task-specific finding.
- **Uncertainty calibration evaluation**: BO performance depends on calibrated posteriors, not just point prediction accuracy. Reliability diagrams or ECE plots per model type would help distinguish whether 3D underperforms because its features are uninformative vs. because its uncertainty estimates are poorly calibrated—a distinction with distinct implications for practitioners.
- **Matched-pretraining controls**: A pretrained 2D GNN foundation model (e.g., from graph self-supervised pretraining) alongside a pretrained 3D model compared to MolFormer would allow cleaner disentanglement of the pretraining vs. dimensionality effect.
- **Additional 3D architectures (SchNet, DimeNet, or SphereNet)** to ensure the 3D conclusions generalize beyond Equiformer v2.
- **Per-subset breakdowns**: Reporting results for subsets of molecules by size, flexibility, or chirality would help practitioners understand when (not just whether) 3D helps.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"Why do nobody use them" is too informal** (Harsh Critic, Introduction): This is a pure style/tone complaint with no scientific content. Removed.
- **Fingerprints are described as "unique identifiers"** (Harsh Critic, Section 2.3): While technically ECFP fingerprints can have collisions, the characterization is conventional shorthand used throughout cheminformatics and does not affect any experimental result. Removed as inconsequential.
- **Comparison with Tanimoto GP is "unfair"**: The Tanimoto GP uses simple fingerprints and is explicitly used as a baseline *to prove a stronger point* for alternative methods. Any asymmetry favors the baseline, which strengthens rather than undermines the authors' claims. Removed per editorial rules.
- **Writing quality nitpicks** (subject-verb agreement, mid-sentence cuts in parsed PDF): These are artifacts of PDF text extraction or minor grammatical issues that do not affect scientific content. Removed as formatting/style nitpicks.
- **Requests for theoretical proofs of sample complexity bounds** (Spark Finder): This is an empirical benchmarking paper; demanding sample complexity theorems is not standard for this paper's scope or community setting. Moved to Nice-to-Have at most.
- **Criticism that "MoleculeNet is not a standard headline target"**: The paper uses absolute energy from GEOM's augmented version of MoleculeNet, and GEOM is a published dataset. This is a legitimate benchmark target. Removed.

---

## Novel Insights

The most genuinely novel finding beyond the benchmark results themselves is the *sample-complexity asymmetry* between 2D and 3D representations in BO: the paper provides converging evidence, consistent with equivariant model theory (Elesedy & Zaidi 2021), that equivariant 3D GNNs require substantially more training data before matching 2D performance, and that this crossover threshold (~10,000 examples in the cleaner comparisons) is well above what is typically available in realistic molecular BO campaigns with expensive oracle evaluations. This finding is actionable and not obvious: 3D models might be expected to learn more efficiently due to stronger inductive biases, but the data requirements of equivariance appear to dominate in practice. However, a clean mechanistic explanation—separating equivariance overhead from conformer noise and architecture-specific factors—remains an open question the paper does not fully answer.

---

## Suggestions

1. **Reframe the headline finding**: Present the key contribution as "2D outperforms 3D under realistic BO data budgets" (the most defensible and interesting result), rather than "1D/LLM outperforms everything," since the latter conflates pretraining scale with representation dimensionality.
2. **Report wall-clock time or GPU hours for each model class**—even a single table in the appendix comparing training and inference time per step would make the cost–accuracy trade-off argument empirically grounded.
3. **Explicitly state the acquisition function, its hyperparameters, and the GP kernel** in the main experimental setup section.
4. **Describe conformer selection policy for 3D models** (e.g., "we use the lowest-energy conformer from GEOM's precomputed ensemble") and, ideally, include a sensitivity analysis with random conformer selection.
5. **Fix the abstract/conclusion claim about "consistent LLM outperformance"** to acknowledge the QM9 exception, and qualify all major claims to specify that results hold for topology-dominated quantum chemistry properties under the data budgets studied.
6. **Add a breakdown table of the 35 setups** (representation × surrogate × task type × data regime) so the benchmark scope is auditable.

---

