---
job_id: 4cf3ee64-7892-4bc5-b52c-b3f7bcb283de
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: CNrU5kGJYG.pdf
paper: TrojanTO: Action-Level Backdoor Attacks Against Trajectory Optimization Models
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies backdoor attacks against trajectory-optimization (transformer-style) offline RL models, clearly within ICLR’s scope of reinforcement learning, representation learning, and safety/security.

## Minimum Quality
Pass ✅.  
Abstract, Introduction, Related Work, Methodology (Sec. 5), Experiments (Sec. 6), Results (Tables/Figs), Discussion/Conclusion (Sec. 7 & Appendix B.2/B.3) are all present. The work is technically non-trivial, experiments are extensive on standard D4RL benchmarks, and exposition is adequate to understand and reproduce most of the methodology. I do not see fundamental methodological errors that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden prompts, AI-targeted instructions, or unusual formatting intended to manipulate automated reviewers.

---

# Expected Review Outcome:

## Summary
The paper studies action-level backdoor attacks against trajectory optimization (TO) models in offline RL, such as Decision Transformer (DT), Graph Decision Transformer (GDT), and Decision ConvFormer (DC). It first analyzes how target actions, trigger dimensions/values, and reward manipulation affect attack efficacy, showing that action and state design matter whereas reward hacking is largely ineffective for TO models. Building on these findings, the authors propose TrojanTO, a post-training backdoor attack that leverages trajectory filtering, batch poisoning, and alternating trigger/model optimization to implant backdoors with very low poisoning budgets; experiments on six D4RL tasks and three TO architectures report high attack success rates with minimal benign performance degradation.

## Strengths
1. **Clear problem motivation and framing of a relevant threat model.**  
   The paper makes a compelling case that TO-style offline RL models are becoming widely used, that most RL backdoor work has focused on Bellman-based training-time attacks, and that post-training attacks against large TO models are both realistic and underexplored (Sec. 1, 3.3, Appendix A). The supply-chain attack scenario (pretrained model tampering without access to original training data) is well articulated and practically relevant.

2. **Systematic empirical analysis of key factors in TO backdoors.**  
   Section 4 revisits three important factors: target action, trigger design, and reward manipulation.  
   - **Table 1** demonstrates that ASR varies dramatically with target action type (e.g., near 1.0 for boundary actions “1”/“-1” and as low as 0.11 for an interior “0” target on Walk), which is an important and often ignored subtlety for continuous action spaces.  
   - **Table 2** and **Table 3** show strong dependence on trigger dimensions and values, with some dimension choices giving ASR≈0 and others >0.9, and with learned triggers outperforming handcrafted or Baffle-style triggers.  
   - **Figure 1** (and **Figure 7** in Appendix K.1) visually support the claim that reward manipulation has negligible influence on ASR and BTP during backdoor training for TO models: the ASR/BTP trajectories over epochs are nearly identical across reward hacks. This is a useful empirical clarification that reward hacking, central in many RL backdoors, is not the right handle on TO models.

3. **Methodological contribution: post-training TrojanTO framework.**  
   TrojanTO is a fairly concrete algorithmic framework combining:
   - **Trajectory filtering (Sec. 5.1)**: selecting longer trajectories to better match high-return behavior, reducing distribution shift and preserving benign task performance. Table 20 corroborates that filtering improves CP compared to using short “bad” or random trajectories.  
   - **Batch poisoning (Sec. 5.2, Eq. (5)–(6))**: duplicating each batch and poisoning only one random transition in the poisoned copy, while keeping an unaltered copy, to ensure consistent trigger context and stabilize training. **Table 18** indicates that the proposed joint optimization over separate poisoned and clean losses yields substantially higher ASR and CP than a naive single-objective formulation.  
   - **Alternating trigger/model optimization (Sec. 5.3; Fig. 2)** with MI-FGSM-based trigger updates (Eq. (8)) and multi-step alternation. **Figure 2** is a helpful schematic that clarifies how trajectory filtering, batch poisoning, and alternating updates interlock.

   While none of these ingredients is conceptually exotic in isolation, combining them into a coherent, post-training TO backdoor pipeline is a meaningful methodological step relative to existing RL backdoor work.

4. **Strong empirical results across models, tasks, and target actions.**  
   - **Table 4** is the central result, comparing TrojanTO with Baffle and IMC on six D4RL tasks and three TO models. TrojanTO yields markedly higher average CP (0.701 vs 0.342 for Baffle and 0.551 for IMC) at only ~0.3% poisoning rate, versus 10% for Baffle. Average ASR improves from 0.369/0.575 to 0.719 while BTP is maintained or improved (0.914 vs 0.792/0.853).  
   - The breakdown in **Table 24** across three different target action types (“1”, “arithmetic”, “fixed random”) confirms that TrojanTO maintains a CP advantage for more challenging, non-boundary targets where ASR is harder to achieve for all methods.  
   Overall, the experimental suite is quite comprehensive: multiple TO architectures (DT, GDT, DC), multiple control domains (locomotion, navigation, manipulation), different target actions, persistent backdoors (Table 6), noise robustness (Table 7), and extensive ablations.

5. **Insightful observations about defenses and detection difficulty.**  
   Appendix B.1 evaluates several backdoor defenses (weight pruning, provable projection, spectral analysis, activation clustering, fine-tuning). The t‑SNE visualization in **Figure 3** nicely illustrates that activations of benign and backdoored models on clean data are almost indistinguishable, explaining why activation clustering fails here. Tables 8–11 clearly show that commonly discussed defenses either barely reduce ASR while heavily damaging BTP, or are ineffective; only simple fine-tuning with a small clean dataset is reasonably successful. This is a useful “negative result” for practitioners interested in defending TO models.

6. **Good reproducibility and experimental transparency.**  
   The paper provides non-trivial details: formal ASR/BTP/CP metrics (Eqs. (2)-(4)), clear poisoning budgets (Table 12, Appendix C.1 & J.5), training hyperparameters (Table 13), raw clean performance (Table 14), and algorithmic pseudocode (Algorithm 1). The public code link and explicit note that CP is computed per-run rather than from averaged ASR/BTP are welcome.

## Weaknesses
1. **Conceptual and novelty positioning could be sharper relative to existing RL backdoor work, especially bi‑level or post-training attacks.**  
   The method combines known pieces: bi-level-like alternating optimization between trigger and model parameters, MI‑FGSM-style trigger updates, and batch-wise poisoning. These ideas are reminiscent of IMC (Pang et al., 2020) and other optimization-based backdoors in supervised learning. Section 2 acknowledges IMC and Baffle, but the actual novelty over IMC is not dissected in depth. For instance, in **Table 4**, TrojanTO’s average CP improvement over IMC (0.701 vs 0.551) is significant but not transformative, and in some individual entries IMC performs comparably or better (e.g., DC‑Ant: IMC CP 0.752 vs TrojanTO 0.559; DC‑Pen: 0.655 vs 0.477). The paper would benefit from a crisper conceptual discussion of *what exactly* makes TrojanTO more suitable for TO models than generic IMC or similar bi-level optimization frameworks, beyond empirical tuning.

2. **Some mathematical formulations are inconsistent or under-specified.**  
   - In Section 3.3, the adversary’s objective (Eq. (1)) is given as  
     \[
     \min_{\tilde\pi} \sum_s \|\tilde\pi([a],[s]+\delta,[\hat R])_t - a^\dagger\| + \lambda \|\tilde\pi([a],[s],[\hat R])_t - \pi([a],[s],[\hat R])_t\|.
     \]  
     Later, in Section 5.2, the training objective is  
     \(\mathcal L = \mathcal L_p + \lambda \mathcal L_c\), with \(\mathcal L_p\) and \(\mathcal L_c\) defined by Eq. (5)–(6). Then in Eq. (7) (bi-level formulation) the inner objective for \(\tilde \pi_\star\) is  
     \[
     \arg\min_{\tilde\pi}\mathbb E_{\tau\in F_\tau}\big[\lambda \mathcal L_p(\tau,\delta_\star;\tilde\pi) + (1-\lambda)\mathcal L_c(\tau;\tilde\pi)\big],
     \]  
     which swaps the weighting pattern used in Sec. 5.2. This inconsistency in how \(\lambda\) is used is not discussed, and it is unclear what exact loss was implemented. Clarifying whether Eq. (7) is a reparameterization (with a different \(\lambda\)) or a typo matters because the tradeoff between ASR and BTP is governed precisely by this coefficient.  
   - Eq. (5) and Eq. (6) implicitly use MSE over continuous actions, but the normalization differs: Eq. (5) uses a squared norm for one step, Eq. (6) averages over timesteps via a factor \(1/T\). This mismatch biases optimization toward benign reconstruction when \(\lambda\) is not extremely small, yet the paper does not discuss this design choice or how \(\lambda\) was selected per environment. It would be good to see sensitivity plots for \(\lambda\) or justification that the values yield the desired ASR/BTP balance.

3. **Threat model realism and attacker capabilities need clearer justification.**  
   The threat model (Sec. 3.3) emphasizes that the attacker lacks access to the original training dataset but can modify a pretrained TO model and “uses a minimal set of poisoned trajectories (e.g., 0.3%).” However, Appendix C.1 clarifies that these trajectories are obtained via environment interaction with the target model or another agent. This is a somewhat strong assumption in many supply-chain scenarios: an attacker who can run 10 trajectories in the true environment (e.g., a real robot or simulator behind a service boundary) may not be as unconstrained as assumed. The paper should discuss:  
   - Whether the same method could be instantiated purely as *data poisoning* on a public fine-tuning dataset (no environment access), and if not, why interaction is essential.  
   - What happens if the attacker can only query the policy on logged states but cannot alter environment dynamics or collect trajectories that follow the poisoned behavior distribution.  
   Currently, the “no access to original data” phrase oversells the realism: the attacker does need access to the environment or a behaviorally equivalent simulator.

4. **Interpretation of some empirical results is somewhat optimistic, underplaying failure cases.**  
   While **Table 4** shows strong average gains, per‑cell performance is more nuanced:  
   - For *DC‑Pen*, TrojanTO’s CP (0.477) is substantially worse than IMC’s 0.655, despite similar BTP (~0.98–0.98) and significantly lower ASR for TrojanTO (0.428 vs 0.657).  
   - For *DC‑Ant*, TrojanTO’s CP (0.559) trails IMC (0.752), mainly due to lower ASR.  
   - For DT‑Hopp, TrojanTO CP (0.365) is only slightly above Baffle (0.313) and below IMC’s 0.013 is indeed degenerate, but here the framing in Sec. 6.1 emphasizes average improvements and “complete performance collapse” of baselines in some settings, with less attention to the cases where TrojanTO underperforms.  
   A more balanced discussion of when and why TrojanTO struggles (especially on certain DC tasks) would improve the scientific value.

5. **Limited exploration of automated trigger-dimension/value selection and their generality.**  
   Section 4.2 and Appendix F show that trigger dimensions significantly affect ASR (Table 2): some triples achieve ASR >0.9, while others yield near-zero ASR. The authors report that Grad‑CAM-based dimension choice did not help, and NTK spectrum analysis showed no correlation. However, they ultimately *fix* dimensions to (1,2,3) for subsequent experiments and rely on MI‑FGSM to tune values, effectively assuming the attacker can search for a “good” dimension set offline. This makes the attack less plug‑and‑play than advertised; in realistic settings an attacker may not have the budget to brute-force many dimension combinations, especially in high-dimensional states like Kitchen. The paper briefly acknowledges this as future work (Appendix B.2), but the main text could be more transparent about the search cost and robustness of TrojanTO to suboptimal dimension choices.

6. **Reward-manipulation study, while interesting, is somewhat narrow.**  
   The conclusion of Section 4.3, that reward manipulation is “ineffective” for TO backdoors, is supported by **Figure 1** and **Figure 7**, which show very similar ASR/BTP curves for a few reward hacking strategies. However, these experiments:  
   - Use a single trigger dimension triple ((8,9,10)) and a specific target type;  
   - Are conducted in a subset of tasks (Walk and Hopper);  
   - Consider “reward hacking” mainly by overwriting rewards to a few constant values.  
   This is an informative initial study, but the general statement that reward manipulation is negligible for TO models is stronger than what the evidence strictly supports. It would be more accurate to state that *the particular class of reward perturbations tested* had little effect, and that more sophisticated reward-shaping or RTG-target manipulations might still be viable.

7. **Defense discussion is incomplete regarding adaptive attackers and fine-tuning.**  
   Appendix B.1 finds that most standard defenses fail, while fine-tuning on 10 clean trajectories over 10k steps reduces ASR to near zero with good BTP (Table 11). However:  
   - The analysis does not consider an *adaptive attacker* anticipating fine-tuning (e.g., designing the backdoor to survive modest fine-tuning, or to be reactivated by a second post-training step).  
   - The residual effect mentioned (actions remaining unusually close to the target even when ASR is low) is not quantified.  
   - From a defender’s perspective, this suggests a simple and practical mitigation, which slightly undercuts the severity of the threat. The paper would be stronger if it evaluated whether TrojanTO can be re-instantiated after fine-tuning with even less data, or whether certain fine-tuning regimes leave hidden vulnerabilities.

8. **Clarity and notation issues.**  
   There are several minor but cumulative clarity problems:  
   - In the experimental description of ASR (Eq. (2)), the notation \([a]_{i_k}\), \([s]_{i_k}\), \([\hat R]_{i_k}\) is somewhat overloaded relative to Sec. 3.1, where \([a]\) denotes sequences of length \(K\). It would help to spell out sequence lengths, indices, and how context truncation is handled.  
   - In Algorithm 1, the notation \(\delta_i^k, \tilde\pi_j^k\) is not clearly tied back to the definitions in Sec. 5.3, and the relationship between inner iteration counts \(N_1, N_2\) and the “alternation frequency” used in Table 19 is not fully explicit.  
   These do not invalidate the method but make precise reimplementation more difficult.

Overall, while the paper is solid and provides useful insights and a reasonably effective attack, it falls short of being exceptional due to limited conceptual novelty and some analytical gaps.

## Potentially Missing Related Work
1. **B. Zhang, J. Li, L. Zheng, “Stealthy Backdoor Attack in Reinforcement Learning via Bi-level Optimization”, 2025.**  
   This work also employs bi-level optimization for RL backdoors, focusing on stealthiness. It is directly related to the TrojanTO alternating optimization in Eq. (7). It should be discussed in the Related Work section (Sec. 2) alongside IMC and Baffle, with a clear comparison: how does TrojanTO’s bi-level scheme differ from Zhang et al.’s in terms of objectives (policy-level vs action-level), optimization strategies, and data requirements?

2. **E. Rathbun, A. Oprea, C. Amato, “Adversarial Inception Backdoor Attacks against Reinforcement Learning”, 2025.**  
   This paper introduces a strong RL backdoor method under strict reward constraints. Even though it may target online RL and Bellman-based agents, its methodology and threat model are close enough that it should be cited and contrasted in Sec. 2 and possibly Sec. 3.3. The authors could explain why those techniques are not directly applicable to TO models, and how TrojanTO overcomes those limitations (e.g., operating post-training, acting on sequences instead of reward shaping).

If these papers differ substantively in problem setting (e.g., no TO models), that distinction should be explicitly drawn.

## Questions
1. **Clarification on the loss weighting \(\lambda\).**  
   - Did you actually implement \(\mathcal L = \mathcal L_p + \lambda \mathcal L_c\) as stated in Sec. 5.2, or the Eq. (7) formulation with \(\lambda \mathcal L_p + (1-\lambda)\mathcal L_c\)?  
   - What specific values of \(\lambda\) were used per environment/model, and how sensitive are ASR/BTP/CP to \(\lambda\)? A plot similar to Table 19 for \(\lambda\) would significantly increase confidence in the robustness of your design.

2. **Trigger-dimension selection overhead.**  
   Beyond the small-scale experiments in Table 2 and Appendix F, how many different dimension triples did you try in practice before settling on (1,2,3) for the main experiments? Approximate this search cost in terms of extra training/MI‑FGSM runs. Could TrojanTO still be effective if the attacker is restricted to a single pre-selected dimension set (e.g., based on a generic heuristic) without any empirical search?

3. **Dependence on environment interaction.**  
   Your main attack assumes access to a small set of trajectories collected by interacting with the environment using the victim model (Appendix C.1). If instead the attacker only obtains a small number of *logged* trajectories (e.g., from the original dataset or a public benchmark) but cannot interact with the environment, does TrojanTO still work as-is? If not, what part breaks, and could the batch poisoning plus alternating optimization be adapted to this stricter setting?

4. **Fine-tuning as a defense and adaptive attacks.**  
   Table 11 shows that fine-tuning on 10 clean trajectories for 10k steps effectively eliminates the backdoor. Can you comment on whether you attempted to design TrojanTO to *survive* such fine-tuning, for instance by injecting the backdoor into weights that are less likely to move under standard supervised fine-tuning? Do you expect a straightforward extension of your alternating optimization (e.g., operating after the defender’s fine-tuning) to re-insert a backdoor using even fewer trajectories?

5. **Reward manipulation generality.**  
   In Fig. 1 and Fig. 7, you experiment with a few constant reward-replacement schemes. Did you also try manipulating the initial RTG \(\hat R_0\) at evaluation, or shaping rewards only around the poisoned time steps while leaving others untouched? If you have negative results for such more targeted manipulations, including them would strengthen your claim that reward hacking is indeed a weak lever for TO backdoors.

## Flag For Ethics Review
- No ethics review needed.

## Details Of Ethics Concerns
N/A. The work studies attacks on RL agents but is framed in a security context with an explicit ethics statement (Appendix B.3) and discusses defenses. No specific sensitive data, human subjects, or deployment are involved.

## Soundness Rating
3: good.  
The methods are technically plausible and empirically well supported across many settings, though there are some inconsistencies in the loss formulation and limited analysis of certain assumptions (e.g., dimension selection, threat model).

## Presentation Rating
3: good.  
The paper is generally well structured and readable; figures like **Figure 2** and tables like **Table 4** are informative. Some notational inconsistencies and over-strong statements (e.g., about reward manipulation) slightly reduce clarity.

## Contribution Rating
3: good.  
The contribution is a solid, empirical and methodological study of post-training action-level backdoors for TO models, with useful insights and a practically effective attack. Conceptual novelty is moderate rather than high, but the work is well executed and relevant for ICLR.

## Overall Rating
6: marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper offers a timely and well-executed study of backdoor attacks on trajectory-optimization offline RL models, with a practical post-training method that significantly outperforms prior baselines on average and a careful empirical exploration of trigger/target design and defenses. The main limitations are modest conceptual novelty over existing optimization-based backdoors, some unclarities in the mathematical formulation and threat model, and under-discussed failure cases. On balance, the strengths outweigh the weaknesses, and I lean slightly toward acceptance.

## Reviewer Confidence
4: confident.  
I am familiar with RL and backdoor attack literature, carefully checked the core equations and tables, and feel reasonably confident in my assessment, though I did not independently reproduce experiments.