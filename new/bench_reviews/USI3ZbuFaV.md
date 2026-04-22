Now I have a thorough understanding of the paper and the calibration anchors. Let me write the final consolidated review.

Let me organize my analysis carefully:

**Harsh Critic's Claim 1: Input-dependent noise distribution invalidates the certification framework**

The paper's fuzzed text randomization (FTR) uses MCTS to identify vulnerable areas for each input x', then applies differential randomization probabilities ω_H (higher) to identified areas and ω_L (lower) to other areas. This means the noise distribution depends on x', which varies across inputs. Standard randomized smoothing requires a fixed noise distribution. Let me think about this more carefully.

Looking at Corollary 1 (Eq. 18), it compares R_r^new vs R_r^old. The derivation seems to assume that ω at the trigger location is ω_H (i.e., the trigger falls in the MCTS-identified vulnerable area). But this is only true if MCTS correctly identifies the trigger. If MCTS misses, then the trigger gets ω_L < ω_M, which yields a *worse* certified radius than uniform. The paper essentially conditions on the trigger being correctly identified, which is a circular assumption for a certified guarantee.

However, I need to be careful about what the theory actually claims. The certification in Theorem 1 conditions on a fixed randomization probability ω. Corollary 1 then shows that if the trigger happens to be in the vulnerable area (ω_H region), the radius improves. The issue is that for certification to be meaningful, this should hold for *all* possible adversarial inputs, not just those where MCTS happens to identify the trigger correctly. If the adversary can craft inputs that evade MCTS identification, the certified radius doesn't hold.

But wait - in randomized smoothing for backdoor defense, the noise is applied during inference (randomizing the input text). The MCTS runs on the perturbed input x'. The adversary doesn't know which segments will be identified as vulnerable (unless they are a strong adaptive adversary that can manipulate the MCTS output). The adversarial trigger is embedded in the text before the defense processes it. So the question is whether the MCTS will reliably identify segments containing backdoor triggers.

This is still a legitimate concern because (a) the certification should hold unconditionally, and (b) the paper's Corollary 1 implicitly conditions on the trigger being in the vulnerable area. But it's not exactly the same as the bDooTVT4t2 case where input-dependent sigma directly invalidates the certification theorem. Here, the issue is more subtle - the certification in Theorem 1 is about the probability of randomizing the trigger, and Corollary 1 assumes the trigger is randomizable with probability ω_H. If MCTS correctly identifies triggers most of the time, this is a reasonable assumption, but it's not a certified guarantee in the formal sense.

**Harsh Critic's Claim 2: Assumption 1 is asserted without justification and is likely false**

Assumption 1 states that f̃(x, θ̃_F) = f(x, θ_F) on benign inputs. The paper claims this "can be approximately guaranteed" by setting η small. This is a strong assumption - it says that after adding noise to model parameters during fine-tuning and inference, the model still makes the same predictions on clean data as a model fine-tuned without any noise. The fact that CA drops in some ablations (Table 3) when BMPS is removed (not when BMPS is added) is actually consistent - removing BMPS means no noise, so CA could go up (which it slightly does in some cells). But adding parameter noise could still change clean predictions. This assumption is standard in randomized smoothing (the base model should predict correctly on uncorrupted inputs), but its justification here is thin. The standard RS approach uses an uncorrupted base classifier; here, the biphased smoothing modifies the model parameters, so Assumption 1 is not guaranteed.

Actually, looking more carefully at Table 3: the "-BMPS" row shows CA of 82.48%, 85.82%, 91.60% on SST-2, while the full FRS shows 82.36%, 85.67%, 91.64%. These are very similar. The CA doesn't drop much with BMPS. But this comparison is between "no BMPS" vs "BMPS" - it doesn't compare to the clean model without any defense. Looking at Table 1, the best CA numbers for undefended models are around 81-91%, so the CA seems preserved.

However, the core theoretical issue remains: the paper doesn't provide a rigorous argument for why adding noise during fine-tuning and inference preserves clean accuracy. The claim that "setting η small enough" achieves this is informal. In practice, the results show CA is maintained, but the theoretical certification depends on this assumption holding exactly.

**Harsh Critic's Claim 3: No baseline for biphased parameter smoothing vs naive ensemble**

This is a valid point. The paper motivates BMPS as an efficient alternative to training K models on K noisy datasets (Eq. 3), but never compares against this baseline. The ablation only removes components from FRS. Without the full ensemble baseline, we can't assess whether BMPS sacrifices robustness for efficiency.

**Harsh Critic's Claim 4: Trigger-neutralization model is binary/unrealistic**

The Δ = 1 - ω^{R_rL} model assumes the trigger is either completely randomized or not. This is a simplification but is standard in randomized smoothing analysis (all-or-nothing reasoning). This is a minor concern, not major.

Now let me assess severity:

1. **Input-dependent noise / certification validity** - This is the most serious issue. The certification in Corollary 1 implicitly assumes MCTS correctly identifies trigger locations. If MCTS fails, the trigger could be in a low-randomization area, and the certified radius would not hold. However, this is not identical to the bDooTVT4t2 case where Theorem 3.2 explicitly requires fixed noise but Section 4.3 uses adaptive noise. In this paper, Theorem 1 doesn't technically require fixed noise — it conditions on a given randomization probability ω. The issue is with Corollary 1, which claims a broader radius by substituting ω_H, but this only holds if the trigger is in the vulnerable area. This is a significant gap in the certification claim but not necessarily fatal to the empirical method.

2. **Assumption 1** - This is a standard simplification in randomized smoothing. The empirical results suggest it approximately holds (CA is maintained). However, the theoretical certification depends on it holding exactly, and the paper provides no formal bound on the violation. This weakens the certification but doesn't completely invalidate the method.

3. **Missing baseline** - This is a valid and significant experimental gap but not a fatal flaw.

4. **Binary trigger model** - Minor simplification, standard in the field.

Let me calibrate against anchors:

- bDooTVT4t2 (avg 3.0, Reject): Input-dependent noise directly violates the assumption of Theorem 3.2, received soundness 1/5. The FRS paper has a similar but not identical issue — its Corollary 1 assumes triggers are correctly identified, which is not a formal certification guarantee. This is less severe than bDooTVT4t2 where the theorem itself is violated.

- 85Eej2kUHQ (avg 2.33, Withdrawn): Main theorem has a concrete counterexample. FRS doesn't have a provably wrong theorem, but its certification relies on an unverified assumption (trigger identification).

- cd79pbXi4N (avg 7.0, Accept Poster): Clean certified robustness paper with proper theoretical guarantees, limited to small models/edits. FRS has more serious theoretical gaps.

- ExUC9dQJhQ (avg 6.0, Reject): Certified robustness against poisoning with loose bounds, rejected primarily for loose and impractical bounds. FRS has a different issue (assumption validity rather than looseness).

- xq7h9nfdY2 (avg 4.5, Reject): SmoothLLM, provable guarantees that aren't really formal guarantees. Similar positioning issue.

- vZ6r9GMT1n (avg 5.75, Accept Poster): Defense with theorem assumptions questioned but still accepted. FRS is somewhat similar.

Overall assessment: FRS has meaningful empirical results and a creative approach, but its core theoretical claim (certified broader robustness radius) has foundational gaps. The certification relies on Assumption 1 (unverified) and on MCTS correctly identifying triggers (not certified). The paper frames FRS as providing "certified defense" but the certification is conditional on assumptions that are not verified/boundable. This is similar to papers that received scores in the 4-6 range.

I'll place this around 4.5-5.0 based on the calibration anchors. The empirical results are solid and the MCTS+fuzzing approach is creative, but the theoretical claims overreach given the conditional nature of the certification.</think>Now I have enough information to write the consolidated review. Let me carefully weigh all the points.

## Summary

FRS (Fuzzed Randomized Smoothing) proposes a certified defense against pre-training-phase backdoor attacks on pre-trained language models. It combines two ideas: (1) biphased model parameter smoothing (BMPS), which adds noise to model parameters during both fine-tuning and inference to avoid the O(K) training cost of ensemble-based randomized smoothing, and (2) fuzzed text randomization (FTR), which uses MCTS to identify vulnerable text segments and concentrate randomization probability on those segments. The paper claims FRS achieves a provably broader certified robustness radius than uniform randomized smoothing, supported by experiments across multiple attack types, datasets, and model sizes.

## Strengths

- **Novel and well-motivated problem formulation.** The paper correctly identifies that existing certified defenses mostly target evasion attacks or assume access to poisoned training data (in-attack), while the pre-training-phase backdoor setting requires post-attack defense without access to the original poisoned data. This is a practically important and underexplored threat model (Sections 1–2).

- **Creative integration of fuzzing with randomized smoothing.** The MCTS-based FTR (Section 4.3) proactively identifies segments likely containing triggers and concentrates randomization probability on those segments (Eq. 11), departing from the standard uniform randomization. The Damerau-Levenshtein space formulation (Section 3) naturally accommodates character-level, word-level, and sentence-level triggers, providing a unified framework.

- **Empirical effectiveness.** FRS outperforms TextGuard (the only other certified defense) across all 9 attack-dataset configurations (Table 1), with ASR reductions of 7–15 percentage points and consistent PA improvements. Table 2 shows 26–35% larger average certified radii. Table 4 demonstrates generalization to LLaMA3-8B, though with diminishing margins.

- **Biphased parameter smoothing as a practical efficiency mechanism.** BMPS (Section 4.2, Eqs. 4–5) avoids training K models on K noisy datasets by adding noise to parameters during fine-tuning and inference. This is a practical contribution for scaling to larger models, and the ablation (Table 3) confirms both BMPS and FTR contribute positively.

## Weaknesses

### Fatal
None.

### Major

- **The certified robustness radius in Corollary 1 depends on the trigger being correctly identified by MCTS, which is not guaranteed.** Corollary 1 (Eq. 18) shows the radius improves by replacing the uniform probability ω_M with ω_H (the higher randomization probability for MCTS-identified segments). This substitution is valid *only* if the trigger falls within the MCTS-identified vulnerable area T(x'). If MCTS fails to locate the trigger, the trigger lies in a ω_L zone (where ω_L < ω_M), yielding a *worse* certified radius than uniform randomization. The paper states "with more MCTS iteration budget, the confidence that the trigger is successfully captured can be higher" (after Eq. 18), but this is an empirical claim, not a formal guarantee. A certified defense requires the robustness guarantee to hold unconditionally or under clearly quantified assumptions. Here, the guarantee holds only when MCTS correctly identifies the trigger—a condition that itself depends on the very backdoor the defense is trying to neutralize (if the backdoor is effectively erased by parameter smoothing, the model won't exhibit trigger-sensitive behavior for MCTS to detect; if the backdoor persists, then the certification over ω_H may not apply). This circularity undermines the paper's central theoretical claim.

- **Assumption 1 is asserted without formal justification.** The certification in Theorem 1 requires that the smoothed model matches the clean model on benign inputs: f̃(x, θ̃_F) = f(x, θ_F). The paper claims this "can be approximately guaranteed" by setting η small (Section 4.4), but offers no formal bound on the approximation error. In standard randomized smoothing (Cohen et al., 2019), the base classifier is unmodified and this holds trivially. Here, BMPS modifies the model parameters with noise and clipping during both fine-tuning and inference, so Assumption 1 is non-trivially violated. Empirically, the CA metric measures accuracy on benign inputs after defense, not agreement with a hypothetical clean model (which is unavailable in the threat model). Without quantifying the gap f̃(x, θ̃_F) ≠ f(x, θ_F), the certification in Theorem 1 and Corollary 1 is conditional on an assumption whose violation probability is unknown. This doesn't invalidate the empirical method but significantly weakens the formal certification claims.

- **No comparison against the naive ensemble baseline that BMPS is designed to replace.** Section 4.1 explicitly formulates the standard approach as training K models on K noisy datasets (Eq. 3). BMPS is motivated as an efficient alternative, but no experiment compares FRS (with BMPS) against this naive ensemble baseline. The ablation in Table 3 only removes BMPS from FRS, which tests whether BMPS helps versus no parameter smoothing—not whether BMPS achieves comparable robustness to the full ensemble it replaces. Without this comparison, the claim that BMPS effectively substitutes for data-level randomization is unsupported, and we cannot assess the accuracy–efficiency tradeoff that motivates the entire approach.

### Minor

- **The trigger neutralization model (Eq. 17) assumes an all-or-nothing binary outcome** — the trigger is either completely randomized or not. Partial disruption of triggers could also neutralize backdoors, or partial trigger presence could still activate them. This binary model makes the certified radius a loose approximation. However, this type of simplification is standard in randomized smoothing analysis and is not uniquely problematic here.

- **Diminishing improvements for larger models.** Table 4 shows that the ASR improvement over TextGuard shrinks from 7.50 pp (BERT-base) to 3.13 pp (LLaMA3-8B) on SST-2. While the paper honestly acknowledges this trend, it raises questions about scalability to even larger models—the very scenario BMPS is designed to address.

### Trivial
None.

## Nice-to-Haves

- Quantitative inference-time cost comparison between FRS (including MCTS overhead) and TextGuard, to substantiate the "efficient" claim in the title.

- Sensitivity analysis of MCTS trigger identification: measure the overlap between MCTS-identified segments and actual trigger locations, and report the rate at which the trigger falls inside the identified vulnerable area. This would empirically ground the key assumption behind Corollary 1.

- Evaluation against adaptive adversaries aware that randomization is MCTS-guided, to assess whether the input-dependent noise distribution creates exploitable weaknesses.

## Removed Points

- **"The output of the smoothed model doesn't match the clean model" (harsh critic, citing Table 3 CA drops):** Table 3 actually shows CA is *preserved* or slightly improved by FRS vs. removing BMPS. The CA drops the harsh critic references (e.g., the "-BMPS" row for OffensEval BadPre) appear to be a formatting artifact in the parsed table. The empirical results do not strongly contradict Assumption 1 in practice; the real issue is the *formal* lack of justification, not a demonstrated empirical violation. Kept as a theoretical concern (see Major weakness above) but removed the empirical CA claim.

- **"MCTS circularity: if backdoor effectively neutralized, model won't exhibit trigger-sensitive behavior" (harsh critic, Section 4.3 note):** While interesting, this is speculative. The MCTS step occurs on the *poisoned* (unsmoothed) model's outputs to detect vulnerable areas, and the smoothing is applied separately. The two mechanisms operate at different stages. Nonetheless, the related concern about the certification depending on MCTS correctness is kept in Major weaknesses.

- **"β=1 condition (full training convergence) is strong and unverified" (harsh critic, Section 4.4):** This is a standard convergence assumption in randomized smoothing analysis and not uniquely problematic for this paper. Minor enough to not warrant inclusion.

- **"Comparison with empirical defenses is unfair" (implied by harsh critic):** The paper compares against both empirical and certified defenses. Comparisons with empirical defenses establish practical value and are not claimed as certified defense comparisons. Not an unfair comparison.

- **"Formatting/style issues and typos" (harsh critic section-by-section notes):** Removed per instructions—these are parser artifacts.

- **Strength Finder's claim about "Clean ablation validating both components":** Partially kept, but weakened—the ablation shows removing components hurts, but doesn't compare against the naive ensemble, so "validates" is too strong.

- **Strength Finder's claim about "Comprehensive experimental evaluation":** Kept but the missing ensemble baseline comparison limits comprehensiveness.

## Novel Insights

The paper's most interesting insight is the fuzzing-inspired idea of concentrating randomization on MCTS-identified vulnerable text segments, which is genuinely novel for certified defenses. However, the tension between this adaptive randomization and the uniformity requirement of traditional certification frameworks reveals a fundamental design challenge: making certification *input-adaptive* (as fuzzing does) and *input-agnostic* (as standard certification requires) are at odds. Resolving this tension—for example, by providing a formal bound on the probability that MCTS correctly identifies the trigger, or by making the randomization schedule a deterministic function that satisfies the certification conditions—would be a significant theoretical contribution. The current paper leaves this unresolved.

## Suggestions

- Add a comparison against the naive K-ensemble baseline (K models on K noisy datasets) to validate that BMPS does not sacrifice certified robustness for efficiency. This is the most critical missing experiment.

- Provide an empirical analysis of MCTS trigger identification accuracy: for each attack type, report the Jaccard overlap between MCTS-identified segments and ground-truth trigger locations. This would directly assess the key assumption behind Corollary 1.

- Either (a) formally bound the violation probability or magnitude of Assumption 1 under the BMPS noise schedule, or (b) explicitly reframe the certification as conditional on Assumption 1 and discuss its practical implications.

- Reframe the theoretical claims to acknowledge that Corollary 1's certification holds conditional on MCTS correctly identifying the trigger, and report the empirical frequency of correct identification.

## Score and Decision

Calibration anchors used:

1. **bDooTVT4t2** (avg 3.0, Reject): Input-dependent noise violates Theorem 3.2's assumptions, soundness rated 1/5. FRS has a similar but less severe issue—its Corollary 1 conditions on MCTS correctness rather than directly violating its own theorem.

2. **85Eej2kUHQ** (avg 2.33, Withdrawn): Main theorem has a concrete counterexample, a fundamentally incorrect result. FRS's issues are about unverified/conditional assumptions rather than provably wrong theorems.

3. **xq7h9nfdY2** (avg 4.5, Reject): Certifies defense with "provable guarantees" that aren't truly formal, relying on restrictive assumptions. Most similar to FRS in positioning.

4. **vZ6r9GMT1n** (avg 5.75, Accept Poster): Defense paper where theorem assumptions were questioned by a reviewer but still accepted. FRS has somewhat stronger theoretical gaps.

5. **cd79pbXi4N** (avg 7.0, Accept Poster): Clean certified robustness paper with proper formal guarantees for text classifiers. FRS's certification is weaker due to its conditional assumptions.

6. **ExUC9dQJhQ** (avg 6.0, Reject): Certified robustness with loose bounds. FRS's bounds are arguably less loose but rest on shakier assumptions.

FRS is empirically strong with creative ideas, but its core theoretical contribution—certified broader robustness radius—is conditional on assumptions (Assumption 1, correct MCTS identification) that are neither formally justified nor empirically validated in the necessary way. This places it below papers like cd79pbXi4N (proper certification, score 7) but above papers like 85Eej2kUHQ (provably wrong theorem, score 2.3). The most comparable anchors are xq7h9nfdY2 (4.5) and vZ6r9GMT1n (5.75). FRS has substantial empirical contributions and a well-motivated approach, which elevates it above xq7h9nfdY2, but the theoretical gaps are more significant than vZ6r9GMT1n. I place FRS at 4.5.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>