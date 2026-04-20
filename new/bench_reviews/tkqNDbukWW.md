## Summary

The paper proposes DeCoRe, a training-free decoding strategy that mitigates LLM hallucinations by contrasting the outputs of a base model against a variant with masked "retrieval heads," with contrastive strength dynamically controlled by the conditional entropy of the next-token distribution. The method demonstrates substantial improvements on contextual faithfulness tasks (e.g., +18.49 factKB on XSum for LLaMA3-8B) and achieves the highest aggregated scores across faithfulness, factuality, and reasoning benchmarks. However, the paper contains factual overstatements about task-level improvements, insufficient justification for key design choices, and a disconnect between how retrieval heads are identified (context-dependent NiTH protocol) and how they are evaluated on closed-book factual recall.

## Strengths

- **Novel integration of mechanistic interpretability with contrastive decoding.** DeCoRe operationalizes the retrieval-head discovery of Wu et al. (2024) into a practical, inference-time intervention. Unlike DoLa (contrasts layers), CAD (contrasts with/without context), or standard CD (contrasts model sizes), it contrasts the base model against a structurally modified version of itself with specific attention heads zeroed out. This mechanistic grounding—validated by the controlled ablation in Figure 3 showing positive correlation between masked heads and faithfulness task performance (XSum r=0.92, MemoTrap r=0.98)—is a principled and creative contribution.

- **Comprehensive and competitive empirical results on faithfulness tasks.** Table 1 shows DeCoRe<sub>entropy</sub> achieves the best overall faithfulness scores for both model sizes (LLaMA3-8B: 64.86, LLaMA3-70B: 68.52), with especially large gains on XSum factKB (47.61→66.10) and MemoTrap Micro Accuracy (64.40→74.87) for the 8B model. Table 2 further demonstrates robustness under the challenging Lost-in-the-Middle setup with 9 distractors, where DeCoRe<sub>entropy</sub> achieves avg EM of 49.10 (8B) vs 47.74 baseline, while ITI and CAD severely degrade.

- **Best aggregated performance across task categories.** Table 5 shows DeCoRe<sub>entropy</sub> achieves the highest overall score for both LLaMA3-8B (48.72 vs 46.29) and LLaMA3-70B (54.98 vs 54.80), demonstrating broad effectiveness rather than gains limited to a single task type.

- **Effective with CoT prompting and low computational overhead.** Table 4 shows improvements on MuSiQue open-book CoT (69.84→74.47 for 8B). The training-free nature avoids the compute costs of baselines like ITI, which requires labeled TruthfulQA data and linear classifier training.

## Weaknesses

### Fatal

None identified. The paper's core claims on faithfulness tasks are empirically supported.

### Major

- **Factual misrepresentation of task-level results in the text.** The paper states on line 169: "DeCoRe<sub>static</sub> and DeCoRe<sub>entropy</sub> both improve the base models in all tasks." This is incorrect. Table 1 shows DeCoRe<sub>entropy</sub> *decreases* IFEval Prompt Acc (70.34→68.39) and Instruct Acc (78.30→76.38) for LLaMA3-8B. Table 3 shows DeCoRe<sub>entropy</sub> *decreases* TruthfulQA-MC1 for LLaMA3-8B (39.41→38.43) and does not improve several other metrics (e.g., PopQA EM: 80.82→80.82, TruthfulQA Gen %Reject: 29.04→28.96). The paper selectively highlights gains while ignoring regresses, which undermines the credibility of the claims about broad effectiveness.

- **The DeCoRe<sub>entropy-lite</sub> variant is functionally indistinguishable from standard Contrastive Decoding.** Section 3.3 (line 133) defines entropy-lite as "employing a smaller LLM with the same vocabulary space as the masked LLM," using "LLaMA3-70B-Instruct and LLaMA3-8B-Instruct as the base and masked LLMs." This means the "masked" model is actually an entirely different, natively smaller model—not the 70B model with retrieval heads masked. The formulation in Equation 7 is mathematically identical to CD (Li et al., 2023) when the amateur model is a smaller model rather than a masked version. The paper presents this variant alongside the main method without clearly acknowledging that it strips away the claimed mechanistic novelty (retrieval head masking) entirely. This blurs the line between what is genuinely novel about DeCoRe and what is simply applying an existing method.

- **Disconnect between retrieval-head identification scope and closed-book evaluations.** Retrieval heads are identified via the Needle-in-a-Haystack protocol, which evaluates *context-dependent* information extraction. Yet the paper heavily promotes and evaluates on closed-book factual recall (TriviaQA, PopQA, closed-book NQ-Open), where no external context exists. The authors acknowledge the negative correlations on these tasks (Figure 3: TriviaQA r=−0.98, PopQA r=−0.94) but hand-wave them by stating "truthfulness... may require different retrieval mechanisms." This is an admission that the core mechanistic hypothesis—that retrieval heads are the locus of hallucination—does not generalize beyond context-dependent tasks. The modest gains on closed-book factuality cannot be confidently attributed to the proposed retrieval-head mechanism and may be artifacts of generic logit scaling. The paper's claims about factuality improvement are therefore overreaching.

### Minor

- **No theoretical or empirical justification for α = H(x<sub>t</sub>).** Section 2.3 proposes setting the contrastive scaling factor to the conditional entropy of the next-token distribution, but provides no rationale for this specific linear relationship. The claim that "higher next-token distribution entropy increases α, reducing the likelihood of selecting potentially hallucinated generations" is not analyzed: entropy can exceed log|V| nats, meaning (1+α) can scale base logits by more than 11×, aggressively amplifying the base model's priors under high uncertainty. The paper provides no ablation comparing this choice to constant, logarithmic, or other scaling functions beyond the static variant.

- **Masking heads zero-dimension residual stream contributions without renormalization.** Section 2.1 sets masked head outputs to zero (m<sub>h</sub><sup>(l)</sup> = 0) but does not discuss whether this introduces distributional shifts in the residual stream. If masked heads contribute significant magnitude to the concatenation, zeroing them without compensatory scaling or bias correction could introduce numerical instabilities unrelated to faithfully removing retrieval capability.

## Trivial

*None.* (The table header mismatch "DeCoRe<sub>entropy-siu</sub>" vs defined "DeCoRe<sub>entropy-lite</sub>" in Table 5 is a typo; removed per rules.)

## Nice-to-Haves

- Providing concrete examples where p<sub>base</sub> confidently selects a wrong token, p<sub>masked</sub> selects a different wrong token, and DeCoRe correctly recovers the ground truth would help validate the contrastive hypothesis visually.

- Reporting and diagnosing the tasks where DeCoRe underperforms the baseline (IFEval, TruthfulQA-MC1 for some models) rather than burying them would strengthen the paper's credibility.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

1. **Harsh critic's claim about structural flaw in entropy scaling:** The critic argues α = H(x<sub>t</sub>) gives minimal contrastive weight when the model is "confidently wrong" (low entropy). While this is a valid theoretical concern in the abstract, the paper's empirical results show DeCoRe<sub>entropy</sub> consistently outperforms the baseline on faithfulness tasks, so the mechanism appears to work in practice despite the concern. The empirical evidence outweighs the theoretical worry.

2. **Harsh critic's claim about missing renormalization/bias correction:** Standard head-masking papers (e.g., Wu et al., 2024; Elhage et al., 2021) zero out head outputs without renormalization. This is standard practice in mechanistic interpretability work, not an omission.

3. **Harsh critic's claim about ROUGE-L/BERTScore not measuring faithfulness:** The paper does include factKB alongside these metrics for the XSum task, and faithfulness evaluation is standardly multi-metric. This is a fair point but not a weakness of substance.

4. **Harsh critic's claim about masking head identification vs factuality recall being contradictory:** The paper explicitly scopes its analysis to note that retrieval heads may not play a primary role in factual recall (line 331: "we observed that DeCoRe offers only marginal enhancements in factual recall tasks, suggesting that retrieval heads may not play a primary role in factual recall"). The critique is partially addressed by the paper itself.

5. **Harsh critic's request for token overlap/KL-divergence analysis of p<sub>masked</sub> distributions:** Moved to nice-to-have. This would strengthen the analysis but is not required to validate the core contribution. It is non-standard for this area of decoding-intervention work.

6. **Harsh critic's claim about "active degradation of factual pathways":** Figure 3 shows a correlation analysis about how many heads to mask, not evidence that masking destroys factual pathways. The baseline in Figure 3 is the unmodified model; the declining performance with more masked heads simply shows that masking too many heads harms overall model capability. This is expected and does not invalidate the method.

7. **Harsh critic's claim about missing ablation on masked vs natively smaller model:** This is partially already present in the paper—DeCoRe<sub>entropy</sub> (masked) and DeCoRe<sub>entropy-lite</sub> (smaller model) are both evaluated. However, no direct head-to-head comparison isolates whether the masking adds value over simply contrasting model scales. Addressable in a rebuttal.

## Novel Insights

The paper makes an interesting observation that the effect of retrieval head masking varies dramatically by task type: positive correlation for faithfulness tasks (XSum, MemoTrap), moderate for TruthfulQA, and strongly negative for closed-book factuality (TriviaQA, PopQA). This suggests there may be fundamentally different mechanisms underlying contextual faithfulness versus parametric factual recall, and a single "hallucination mitigation" method cannot address both equally. However, this insight is underdeveloped—the paper acknowledges the pattern but does not explore why retrieval heads, by definition context-dependent, would have any role in closed-book settings at all, beyond vague appeals to "information movement" in the residual stream.

## Suggestions

1. **Correct the false claim on line 169.** Replace "both improve the base models in all tasks" with a balanced statement acknowledging the IFEval and TruthfulQA regressions, and discuss possible reasons (e.g., that retrieval head masking may interfere with instruction-following capabilities that rely on different attention mechanisms).

2. **Clarify the relationship between DeCoRe<sub>entropy-lite</sub> and standard CD.** Either add a direct ablation showing whether retrieval-head masking provides additional benefit over simply using a smaller model (which would strengthen the claim), or acknowledge that the lite variant is a standard CD instantiation and position it as a computationally cheaper variant rather than a novel contribution of the method.

3. **Add theoretical intuition or ablation for the α = H(x<sub>t</sub>) choice.** A simple comparison with α = constant, α = log H(x<sub>t</sub>), or α = 1/H(x<sub>t</sub>) would help justify why linear entropy scaling was selected and rule out that the choice was arbitrary.

4. **Tighten the scope of claims.** The paper's strongest results are on contextual faithfulness tasks. Consider reframing closed-book factuality improvements as secondary findings that may arise from secondary effects (logit sharpening, residual stream perturbation) rather than the stated retrieval-head mechanism.

---

**Calibration:** I compared against several anchor papers:
- **DoLa** (Th6NyL07na.md, scores 8,8,5,8, Accept): Similar training-free decoding with mechanistic grounding. One reviewer gave 5, questioning the hypothesis and noting anecdotal evidence. DoLa had clearer novelty and stronger factuality results. This paper has comparable faithfulness gains but weaker justification and misleading claims.
- **488A64eOf6.md** (scores 8,5,6,6, Accept): Decoding method with weak mathematical justification but strong experiments. This paper is similar—good results but unclear mechanism for key design choices.
- **rYyu3jpk8z.md** (scores 5,5,6,5,3, Reject): Paper with marginal improvements and overclaiming. This paper has stronger empirical results than this anchor.
- **n9xeGcI4Yg.md** (scores 6,8,6,10, Accept Spotlight): Training-free method with clear mechanistic motivation and strong results across many tasks. This paper is weaker in mechanism clarity.

This paper sits between the accepted posters (6-8 range) and the rejected borderline ones (3-5 range). The empirical results on faithfulness are genuinely strong and comparable to DoLa. However, the factual misstatements, the entropy-lite/CD equivalence, and the scope overreach on closed-book factuality pull it down. It is not strong enough for high acceptance (7+) but not weak enough for rejection (4-). The pattern most closely matches 488A64eOf6 (avg 6.25) and DoLa (avg 7.25, but one reviewer gave 5 for similar concerns). I place it at 6.0—marginally above threshold, with clear empirical value but substantively flawed claims that need correction.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>