Now I have all the information needed. Let me write the final review.

## Summary

This paper identifies the "priming vulnerability" in Masked Diffusion Language Models (MDLMs), where affirmative tokens appearing at intermediate denoising steps can steer subsequent generation toward harmful responses even in safety-aligned models. The authors analyze this vulnerability through an anchoring attack (hypothetical intervention in the denoising process) and derive First-Step GCG (a tractable surrogate objective for optimization-based attacks that exploits the priming effect). They then propose Recovery Alignment (RA), which trains models to generate safe responses from contaminated intermediate states, and demonstrate that RA effectively mitigates the vulnerability while preserving general capability and improving robustness against conventional jailbreaks.

## Strengths

- **Clear identification of a novel, MDLM-specific vulnerability**: The priming vulnerability is precisely defined (Section 4) and its connection to the MDLM's parallel, iterative denoising mechanism is well-articulated. Figure 2 cleanly demonstrates that even a single-token intervention at step 1 raises ASR from 2% to 21% on LLaDA Instruct, and ASR exceeds 80% by step 16 across all models.

- **First-Step GCG is a genuine technical contribution**: Theorem 4.1 provides a tractable surrogate objective that converts an intractable stochastic optimization into a directly differentiable one. Table 1 confirms ~20× speedup and up to 4× ASR improvement over Monte Carlo GCG, making it practically useful.

- **RA has strong empirical support with crucial ablation**: Table 2 shows RA reduces ASR from 68.7%→3.0% at t_inter=8 on LLaDA Instruct, far outperforming baselines. The "RA w/o inter" ablation is critical: without contaminated-state training, ASR remains at 49.0% vs. 3.0% at t_inter=8, directly confirming that conventional alignment is insufficient and conditioning on contaminated states is essential.

- **RA generalizes beyond the priming vulnerability**: Table 3 shows RA also reduces ASR against conventional conversational attacks (e.g., PAIR on LLaDA: 44.3%→10.0%), suggesting it learns a general recovery capability rather than narrowly patching one attack vector.

- **General capability is preserved**: Table 4 shows average benchmark scores remain effectively unchanged (LLaDA: 52.2→52.6; LLaDA 1.5: 52.7→52.8), with TruthfulQA improving (47.6→53.4 on LLaDA).

- **Comprehensive evaluation scope**: 3 MDLMs, 7 attack methods (intervention-based and black-box), 3 automatic evaluators, and 11 capability benchmarks make the findings robust.

## Weaknesses

### Fatal
None.

### Major

- **Missing over-refusal evaluation on benign queries**: The paper claims "minimal impact on task performance" (Abstract), but the general capability evaluation (Table 4) consists entirely of multiple-choice and code-generation benchmarks that do not measure whether the model over-refuses benign requests. RA trains the model to refuse from contaminated intermediate states, and the No Attack ASR drops to 0.0% on LLaDA (Table 2). Without measuring false refusal rates on benign or borderline queries (e.g., using XSTest), the utility-preservation claim is incomplete for the dimension most relevant to safety alignment. This is standard practice in the safety alignment literature, and the closely related backtracking paper (Bo62NeU6VF) received the same criticism from its reviewers.

- **Theoretical contribution limited by the monotonicity assumption**: Theorem 4.1 requires log π_θ(r̃_{t+1}=r|q,r_t) ≥ log π_θ(r̃_1=r|q,r_0) for the harmful response r, which essentially assumes that revealing more of the harmful response makes the model more confident about it. While the paper empirically validates this assumption (Appendix C.2), the theoretical contribution is weaker than presented: the assumption encodes the vulnerability itself, and the theorem shows that if the vulnerability holds, a simpler objective suffices. The theorem does not explain *why* the vulnerability arises or provide non-trivial predictive power beyond the empirical observation.

### Minor

- **Anchoring attack results foregrounded despite unrealistic threat model**: The most dramatic results come from the anchoring attack (Table 2), which requires directly modifying the model's internal denoising state—an extremely strong assumption. The paper acknowledges this is "hypothetical" (Section 4.1), but the abstract and introduction lead with these results. Results under the more realistic First-Step GCG and conversational attacks (Tables 1, 3) show more modest but still meaningful improvements. The paper would benefit from clearer contextualization of results by threat model realism.

- **HumanEval drops meaningfully on LLaDA**: HumanEval drops from 22.0 to 17.1 with RA (a ~22% relative decrease), the largest relative change in Table 4. The paper attributes PIQA decreases to "potential forgetting effects" but does not discuss this HumanEval drop, which is more substantial and directly relevant to code-generation capability.

- **Abstract overclaims about "existing" attacks**: The abstract states "the vulnerability allows existing optimization-based jailbreak attacks to succeed on DLMs." While MC GCG does benefit somewhat from the vulnerability (20% ASR on LLaDA), the most dramatic improvements come from First-Step GCG, which is a *new* attack derived in this paper. The phrasing could be more precise.

### Trivial
None.

## Nice-to-Haves

- Over-refusal evaluation on a benchmark like XSTest to complete the utility-preservation claim.
- A compute-matched comparison for MC GCG vs. First-Step GCG to disentangle optimization efficiency from priming amplification.
- Qualitative examples of recovery trajectories showing token-level denoising before and after RA, to make the recovery mechanism concrete.
- Analysis of which harmful queries still succeed against RA at large t_inter, to characterize the fundamental limits.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Circularity of Theorem 4.1 (from Harsh Critic, elevated to "partially circular")**: The harsh critic claims the theorem is "partially circular" because it "assumes the vulnerability exists in order to prove a result that motivates an attack exploiting it." While the monotonicity assumption does encode a property closely related to the priming effect, this is not truly circular—many theorems make assumptions that align with observed phenomena. The assumption is empirically validated (Appendix C.2) and the theorem's value is in providing a tractable optimization objective, not in "proving" the vulnerability. Downgraded from the critic's structural-level concern to a minor theoretical limitation.

- **Unrealistic threat model dominates evaluation (from Harsh Critic, elevated to "methodological gap")**: The critic overstates this. The paper explicitly frames the anchoring attack as a "hypothetical" diagnostic tool and devotes separate sections and tables to the realistic threat model (Section 4.2, Tables 1 and 3). The anchoring attack is a well-designed *diagnostic*—varying t_inter provides a precise characterization of vulnerability scaling (Figure 2), which directly motivates the curriculum in RA. This is appropriate methodology. Retained as a minor point about presentation/contextualization only.

- **MMaDA baseline essentially unaligned (from Harsh Critic)**: The critic notes MMaDA's 79.7% No Attack ASR indicates the model is poorly aligned, making RA's effect less informative. This is true but the paper's main claims are supported by LLaDA and LLaDA 1.5 results. MMaDA results are supplementary. Removed as a standalone weakness.

- **Compute-matched MC GCG comparison (from Harsh Critic)**: Interesting but not critical—the paper's contribution is the First-Step GCG method and its empirical effectiveness. The theoretical justification provides the connection to priming. Moved to Nice-to-Have.

- **DPO-style RA (from Harsh Critic)**: Already acknowledged as a limitation in Section 7. Moved to Nice-to-Have.

- **Confidence intervals on small samples (from Harsh Critic)**: The 21% ASR at t_inter=1 on 100 examples. The paper reports standard deviations (±4-5% across experiments), which appropriately conveys uncertainty. This is standard practice and not a meaningful concern.

- **Generic strengths without specific citations (from Strength Finder)**: Dropped strengths that were too generic ("well-written," "timely topic") or conflicted with verified weaknesses.

## Novel Insights

The paper reveals a fundamental structural asymmetry in how safety alignment works for MDLMs versus ARMs: in ARMs, prefilling attacks exploit the causal left-to-right constraint, but the defense (refusal at any position) is conceptually straightforward; in MDLMs, the parallel denoising structure means that any affirmative token at any position can serve as an anchor, and the model must learn to recover from arbitrarily contaminated states rather than simply override a prefix. This distinction explains why RA—training on contaminated intermediates rather than just improving initial-step safety—is necessary and why the "RA w/o inter" ablation shows such a large gap. The generalization of RA to conventional jailbreaks further suggests that the recovery capability learned from contaminated states is not attack-specific but reflects a deeper property of how safety manifests in iterative denoising.

## Suggestions

- Add evaluation on XSTest or a similar benign-query benchmark to verify that RA does not increase over-refusal, completing the utility-preservation claim.
- In the abstract, replace "existing optimization-based jailbreak attacks" with more precise language distinguishing MC GCG's moderate improvement from First-Step GCG's stronger results.
- Acknowledge the HumanEval drop on LLaDA directly in Section 6.3 rather than only discussing the PIQA decrease.
- When presenting anchoring attack results, briefly reiterate that this is a diagnostic tool for understanding vulnerability scaling, not a realistic attack scenario, to help readers contextualize the headline numbers.

## Calibration

**Anchors used:**

- **6Mxhg9PtDE.md** (avg 9.5, Oral): "Shallow safety alignment" paper—discovers that ARM safety alignment is shallow (first few tokens), proposes deep realignment. Closest analog conceptually. This paper is broader and has cleaner theory, earning a higher score. The current paper is more specific to MDLMs and has a limited theoretical contribution.

- **Bo62NeU6VF.md** (avg 8.0, Oral): Backtracking paper—proposes [RESET] token for recovery from unsafe partial generation. Very similar "recovery" concept. Also lacked over-refusal evaluation (noted by its reviewers) but scored 8.0. The current paper has broader evaluation but a weaker theoretical contribution and the MDLM-specific scope is narrower.

- **r42tSSCHPh.md** (avg 7.0, Spotlight): Generation exploitation attack—discovers decoding-strategy vulnerability, proposes alignment defense. Similar structure (vulnerability + defense). The current paper has stronger defense results and better ablation.

- **kUH1yPMAn7.md** (avg 6.0, Poster): Safety layers vulnerability—discovers vulnerability, proposes defense. Weaker empirical support than the current paper.

- **41uZB8bDFh.md** (avg 6.0, Poster): Q-Misalign attack—novel vulnerability with overclaiming flagged. The current paper has stronger ablation and broader evaluation but similarly limited theory.

- **v6tPaf8V09.md** (avg 2.0, Reject): Arms race paper—unrealistic threat model, no systematic evaluation. Far below the current paper.

- **5kMwiMnUip.md** (avg 1.4, Reject): NEMESIS—no results section, superficial analysis. Far below the current paper.

The current paper sits above the 5-6 range papers (which had overclaimed or weaker results) but below the 8+ papers (which had broader implications and cleaner theoretical frameworks). The over-refusal gap is a real but addressable concern, and the theoretical contribution is limited. The empirical work is strong with comprehensive evaluation across 3 models, 7 attacks, and 11 capability benchmarks.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>