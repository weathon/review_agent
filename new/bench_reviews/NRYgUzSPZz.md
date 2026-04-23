Now let me run calibration searches in parallel.Now I have enough information to write the final review. Let me synthesize everything.---

## Summary

This paper argues that discrete diffusion models overcome "subgoal imbalance" in complex reasoning and planning tasks, where autoregressive (AR) models struggle because predicting certain tokens requires long-range context unavailable under causal conditioning. The authors propose Multi-Granularity Diffusion Modeling (MGDM), which adds token-level reweighting to focus training on harder subgoals. Evaluations on Countdown, Sudoku, and Boolean SAT show dramatic improvements over comparably-sized AR models (e.g., 91.5% vs. 45.8% on Countdown-4, 100% vs. 20.7% on Sudoku), and a synthetic planning task is used to motivate the theoretical framing.

---

## Strengths

1. **Dramatic and consistent empirical improvements across multiple tasks (Table 1, Figure 4)**: A 6M MGDM achieves 100% on Sudoku where LLaMA-13B gets only 32.9%, and 91.5% on Countdown-4 vs. 45.8% for the same-size AR model. These gaps are large enough that they are not attributable to minor tuning differences. Multiple diffusion baselines (VDM, D3PM, RDM) are included, allowing readers to assess how much MGDM adds within the diffusion family.

2. **Clean synthetic task motivating "subgoal imbalance" (Figure 2, §3.1)**: The planning graph task with controllable planning distance is well-designed. The key empirical finding—AR models barely outperform random chance for PD ≥ 2 with 50k training instances, while diffusion achieves near-perfect accuracy—is clearly presented, and the data-scaling curves provide additional quantitative evidence that AR requires exponentially more data for harder subgoals.

3. **Insightful "Regretful Compromise" error analysis (Figure 6b, §4.4)**: The finding that AR models concentrate 48.9% of their calculation errors in Equation 3 (the last equation), vs. 0.2% in Equation 1, provides a concrete mechanistic picture of how left-to-right decoding cascades early planning errors into late calculation mistakes. This characterization is both novel and practically informative.

4. **Speed-accuracy trade-off characterization (Figure 6a)**: MGDM with a single diffusion step achieves ~75% on Countdown-4 (vs. 45.8% for AR) at 10× the throughput, demonstrating practical utility across the decoding budget spectrum.

---

## Weaknesses

### Fatal
None.

### Major

- **Teacherless baseline absent from all main-task experiments.** This is the paper's most significant structural gap. In §3.1 the paper explicitly states: *"Both teacherless training and diffusion models exhibit a similar U-shaped curve in their performance. This similarity can be attributed to the fact that teacherless training can be conceptualized as a special case of diffusion without an iterative denoising process."* Teacherless training is a bidirectional model that predicts all output tokens jointly from the input—no iterative denoising, but also no causal constraint. If teacherless training matches diffusion's data efficiency in the synthetic task (both outperform AR dramatically), the decisive question for the paper's core claim is whether the same holds on Countdown, Sudoku, and SAT. The paper never tests this. Note: the fixed-50k synthetic experiment does show teacherless produces illegal paths, suggesting a difference from diffusion in that specific regime. But this limited observation is insufficient to justify omitting teacherless from all three main-task evaluations. Without it, the paper cannot cleanly attribute the large AR-vs-diffusion gaps to the diffusion mechanism as opposed to simple bidirectional conditioning.

- **Theoretical framing does not disentangle bidirectionality from iterative denoising.** The multi-view learning interpretation in §3.2 applies equally well to any bidirectional prediction model, including masked language models or teacherless training. Equation (6) reformulates the diffusion ELBO as $-\log p_\text{DM}(\mathbf{x}_n | \mathbf{x}_{\neq n})$, but this reflects an expectation over noisy contexts $\mathbf{x}_t$—not a single bidirectional conditioning step. The theoretical argument for why multi-step denoising (as opposed to one-shot bidirectional prediction) is necessary for the gains observed is not developed. "Multi-view learning" as described (Xu et al., 2013) does not require iterative denoising, so the framework does not uniquely characterize diffusion models. The paper would be substantially strengthened by either (a) a formal argument showing the iterative denoising contributes beyond bidirectionality, or (b) an empirical disentanglement via a one-step bidirectional baseline.

### Minor

- **MGDM provides modest gains over RDM baseline, with anomalous scaling behavior.** Table 1 shows MGDM (85M) improves Countdown-4 from 87.0% to 91.5% (+4.5pp) but barely moves Countdown-5 (45.8% → 46.6%). More problematically, the 303M MGDM underperforms the 85M MGDM on CD4 (88.3% vs. 91.5%). Well-behaved methods should improve with scale. Table 3 selects the best hyperparameter combination (α=0.25, β=2, linear reweighting, TopK) on the evaluation set itself without a held-out protocol or multiple seeds, raising the possibility of post-hoc tuning. The contribution of MGDM beyond a well-tuned RDM is therefore uncertain.

- **Game of 24 comparison overclaims.** §4.1 concludes "it is challenging for model scaling and decoding strategies to substitute the advantages of modeling paradigm," comparing MGDM (85M) trained on 500k task-specific instances against GPT-4 prompted with 5 examples. The more precise conclusion is that a task-specific small model trained on abundant labeled data outperforms a general-purpose model given minimal supervision. This is not evidence about modeling paradigm per se.

- **SAT experiments use only n = 5, 7, 9 variables**, which are trivially small instances. The paper frames SAT as a benchmark for "a wide range of constraint satisfaction problems," but near-threshold random 3-SAT with 9 variables is not representative of practically hard SAT. The advantage of the diffusion approach at meaningful scales (n ≥ 20–50) is untested.

### Trivial

- Figure 4 (Sudoku results) is a figure with no associated table, making it harder to precisely report the "100% accuracy" headline result. Adding a tabular comparison would improve clarity.

---

## Nice-to-Haves

- Adding a BERT-style masked LM fine-tuned on Countdown/Sudoku would cleanly test whether bidirectionality alone (without the diffusion training objective) accounts for the performance gap.
- Sudoku constraint-satisfaction analysis: verifying that generated outputs satisfy all row/column/box constraints explicitly would rule out near-valid but non-compliant solutions.
- SAT instances with n ≥ 20 variables would meaningfully extend the benchmark to practically relevant scale.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"The Sudoku 100% claim contradicts Figure 4 showing ~40% accuracy."** The PDF parser extracted approximate values from a figure image. The paper text at §4.2 clearly states "our model, which has only 6M parameters, is able to perfectly solve all the problems," and the abstract states "100% accuracy on Sudoku." Figure descriptions produced by image-to-text parsing are unreliable; there is no real contradiction. **Removed as parser artifact.**

- **"Figure 3 conflates training loss with an oracle lower bound."** The harsh critic claims that $-\log p_\text{DM}(\mathbf{x}_n|\mathbf{x}_{\neq n})$ in Equation (6) conditions on clean surrounding tokens. But reading Equation (6) directly, it is defined as $\sum_t w(t)\mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)} u(\mathbf{x}_0,\mathbf{x}_t,n;\theta)$—an expectation over *noisy* $\mathbf{x}_t$, not clean context. The notation is shorthand for the diffusion ELBO contribution for token $n$. The criticism misreads the formulation. **Removed as factually incorrect.**

- **"The synthetic task is an adversarially constructed condition not representative of real tasks."** The task is explicitly symmetric by design to rule out reverse-AR solutions—this is methodologically deliberate, not a confound. Its purpose is to isolate the subgoal imbalance phenomenon. **Removed as misunderstanding of experimental design.**

- **"LLaMA 7B resolving all PDs contradicts the 'fundamental limitation' framing."** The paper itself reports this (§3.1): scaling to 7B with fine-tuning does work. The paper's argument is about data efficiency and small/medium-scale model performance, not strict impossibility. The framing is a bit strong, but this is not a real contradiction—it is a scope clarification. **Removed as strawman.**

- **"MGDM with 1 diffusion step is essentially teacherless, undermining the speed claim."** A single-step MGDM achieves 75% on Countdown-4 vs. 45.8% for AR. Even if this approximates teacherless training, the practical speed advantage is real and the accuracy advantage is demonstrated. **Removed as not substantive.**

---

## Novel Insights

The paper's most genuinely novel observation is the "Regretful Compromise" phenomenon: AR models commit to incorrect planning choices early and then amplify calculation errors at the last step to compensate, producing a characteristic error distribution where ~49% of arithmetic mistakes concentrate in the final equation. This is not just a description of AR failure—it is a mechanistically illuminating account of *how* left-to-right commitment undermines structured generation. The complementary finding that diffusion achieves near-zero planning errors at the first equation step provides strong evidence that global coherence during generation prevents this cascade. This insight has implications beyond the specific tasks studied.

---

## Suggestions

1. **Add teacherless training to Countdown, Sudoku, and SAT experiments.** Even a single result (e.g., teacherless vs. MGDM on Countdown-4) would substantially clarify whether iterative denoising or bidirectionality is responsible for the AR gap.
2. **Report MGDM results with multiple random seeds and error bars** (especially for Table 1 and Table 3), and use a held-out validation set to select the best hyperparameter combination in Table 3.
3. **Tone down the "fundamental limitation of AR" framing** in the introduction to match the evidence: the limitation is a data-efficiency and small/medium-scale phenomenon, not an intrinsic impossibility (LLaMA-7B fine-tuned does resolve all planning distances).
4. **Expand SAT evaluation** to n ≥ 20 variables to test claims about hard combinatorial reasoning.

---

## Score and Decision

**Calibration anchors:**

| Paper | Avg Human Score | Decision | Comparison |
|---|---|---|---|
| `/human_reviews/Xe6UmKMInx.md` (Latent Diffusion for Reasoning) | 3.0 | Reject | Most similar claim (diffusion > AR for reasoning); weaker execution — only 2 synthetic tasks, no real-task evaluation, vague architecture. This paper is clearly above this anchor. |
| `/human_reviews/1pTlvxIfuV.md` (Reparameterized Discrete Diffusion) | 5.5 | Reject | Discrete diffusion technical improvement; mixed reviews. This paper has more dramatic empirical results but a larger theoretical gap. Roughly comparable anchor. |
| `/human_reviews/CfdPELywGN.md` (LLM path planning extrapolation) | 5.2 | Borderline | Empirical reasoning study with moderate contributions, similar score band. This paper has stronger results but similar completeness issues. |
| `/human_reviews/XsgHl54yO7.md` (Discrete Guidance) | 6.5 | Accept Poster | Better theoretical grounding; principled contribution with formal derivation. This paper falls below this anchor due to the missing teacherless baseline and incomplete theoretical disentanglement. |
| `/human_reviews/tyEyYT267x.md` (Block Diffusion) | 8.0 | Accept Oral | Strong SOTA results on language modeling, comprehensive ablation, novel architecture. This paper's empirics are impressive but narrower in scope and weaker in theory. Clearly below this anchor. |
| `/human_reviews/wM2sfVgMDH.md` (Diffusion-Based Planning) | 7.5 | Accept Oral | Diffusion for autonomous driving planning with flexible guidance. Strong technical contribution with formal analysis. This paper is below this anchor. |

**Positioning:** The paper sits between Xe6UmKMInx (3.0, weak) and XsgHl54yO7 (6.5, poster accept). The empirical results are more impressive than the 5.5 anchor (1pTlvxIfuV), but the structural gap of the missing teacherless baseline—which the paper explicitly acknowledges as a conceptual close relative of diffusion yet never evaluates on real tasks—mirrors the kind of incomplete disentanglement that causes borderline papers to be rejected. The MGDM contribution is real but modest. Overall, this paper is a borderline case centered around the 5–5.5 range.

**Originality:** Moderate. The idea of framing diffusion's advantage through "subgoal imbalance" is novel, but the connection to multi-view learning is qualitative and the key competing hypothesis (bidirectionality alone) is not ruled out.

**Importance of research question:** High. Whether discrete diffusion can meaningfully outperform AR on structured reasoning is an important open question for the field.

**Claim support:** Partial. The empirical superiority of diffusion over AR is well-supported. The attribution of this advantage to the *diffusion mechanism specifically* (vs. bidirectionality) is not.

**Soundness of experiments:** Mixed. The main task experiments are solid and multi-baseline. The synthetic task analysis is clean. The MGDM ablation lacks cross-validation and seed variance.

**Clarity of writing:** Good. The paper is well-organized and the key results are accessible.

**Value to the research community:** Meaningful. The empirical demonstration is compelling and the error analysis is insightful, even if the theory is incomplete.

**Final score: 5.5** — Above the weak anchor (3.0) and the rejected medium paper (5.5 with weaker results), but below the poster-level accepted papers (6.5+) due to the missing teacherless baseline and the modest incremental contribution of MGDM over RDM.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>