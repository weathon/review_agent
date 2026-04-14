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