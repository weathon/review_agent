=== CALIBRATION EXAMPLE 16 ===

# Final Consolidated Review
## Summary
This paper studies a real tension in offsite tuning: protecting both private data and the model owner’s intellectual property. It proposes LLEOT, which builds an emulator by first aligning it to the original model for soft-prompt transfer (CPKD) and then intentionally degrading its inference capability via a loss-margin objective (LLE), with the goal of retaining transferability while reducing emulator misuse.

The paper’s central idea is interesting and the empirical results are promising on several QA benchmarks and three model families. However, the strongest claims—especially the theorem-level guarantees about exact gradient preservation and perplexity amplification—are not justified by the actual training procedure used in the paper, and the privacy evaluation remains too narrow to fully substantiate the claimed protection against emulator misuse.

## Strengths
- **The paper identifies a concrete and underexplored failure mode of offsite tuning: emulator capability leakage.** This is not a generic privacy claim; the paper specifically argues that existing OT-style emulators can remain useful enough for unauthorized inference or repackaging, and it backs this concern with direct comparisons in Figure 1 and Table 1.
- **The proposed CPKD component is a specific and useful contribution for soft-prompt transfer.** The paper correctly notes that standard distillation on discrete tokens is mismatched to continuous soft-prompt optimization, and introduces proxy-prompt distillation to align hidden representations under random prompt perturbations. This is empirically supported: in Table 3, removing \(L_{PPD}\) causes the largest degradation among the CPKD loss components.
- **The experiments show a nontrivial utility/privacy tradeoff improvement over OT and CRaSh under the paper’s chosen metric.** In Table 1, LLEOT usually attains lower CPL than OT/CRaSh while remaining competitive or better in transferred downstream accuracy across Qwen2-1.5B, Gemma-2-2B, and Llama-3.2-3B.
- **The ablations are reasonably informative about the method’s internal mechanics.** Table 2 shows that removing CPKD mainly hurts transfer, while removing LLE mainly hurts capability privacy; Table 6 also gives some evidence that the specific elevation strategy matters relative to a naive “negative LM loss” alternative.
- **The method is efficient in the specific soft-prompt setting it targets.** Table 7 highlights that the proposed implementation uses tiny prompt parameters compared with the baseline adapters reported there. Even though this complicates fairness claims (see weaknesses), it is still a practically relevant aspect of the proposed setup.

## Weaknesses

###: Fatal
- **The main theoretical guarantee is not established for the actual algorithm.**  
  Theorem 1 claims exact statements such as
  \[
  PPL_E = e^H \cdot PPL_M,\quad \nabla_P L_E(P;x)=\nabla_P L_M(P;x).
  \]
  But the method does not enforce the identity \(L_E(P;x)=L_M(P;x)+H\) for all prompts \(P\) and inputs \(x\). Instead, Algorithm 1 / Eq. (7) only *optimizes an empirical objective*
  \[
  \min_E \mathbb{E}_{x\sim X_e,\;P'\sim \mathcal N(\mu,\sigma^2)} |L_E(P';x)-L_M(P';x)-H|,
  \]
  over sampled proxy prompts and an elevation dataset. This can at best encourage approximate margin matching on sampled regions; it does not justify exact equalities over the continuous prompt space encountered during downstream tuning. The proof in Appendix D proceeds as if Eq. (6) holds exactly and globally, but that is an idealized assumption, not what the algorithm guarantees. Since exact gradient preservation is central to the paper’s core claim, this mismatch is a serious issue.

### Major:
- **The perplexity/privacy theorem overstates what follows from matching scalar cross-entropy differences.**  
  Appendix D derives \( \hat p_E(x|P) = e^{-nH} \hat p_M(x|P)\) and therefore \(PPL_E=e^H PPL_M\), but this derivation again assumes exact equality of sequence-level losses for every \((P,x)\). In the implemented method, only an empirical residual is minimized. Thus the theorem should be presented as an idealized property of a perfectly offset loss function, not as a guarantee for the trained emulator. As written, the theory is disconnected from the optimization actually performed.
- **The privacy evaluation is too narrow to support the broader claim of “mitigating emulator misuse.”**  
  Capability Privacy Leakage (CPL) is defined only as the ratio of zero-shot task scores. This is a useful descriptive signal, but it does not test whether the emulator can be repurposed or recovered by an adversary through additional fine-tuning, prompting, distillation, or extraction-style attacks. Since the paper motivates the method by concerns like knowledge extraction and repackaging, the evaluation should include at least one concrete attack-oriented protocol; otherwise the claimed privacy protection is only partially demonstrated.
- **Claims of broad adapter generality are not supported by the experiments.**  
  The paper states, “In theory, our method is applicable to various types of adapters. In this paper, we focus on soft prompts...” This scoping is fine, but the broader generality claim should be weakened. All experiments are with soft prompts of length 5, and CPKD is specifically engineered around proxy soft prompts. For common PEFT choices such as LoRA, no evidence is provided.
- **The empirical support for “preserving gradient alignment” is indirect.**  
  The paper relies on Theorem 1 for exact gradient equality, but given the theory/algorithm gap, the empirical section should have directly measured prompt-gradient alignment or margin residuals on held-out prompts. Instead, the evidence is inferred from downstream accuracy and CPL. That is suggestive but not sufficient for such a strong mechanistic claim.

### Minor
- **The comparison setup mixes methodological gains with adapter-choice differences, which muddies attribution.**  
  Table 7 reports dramatically smaller adapter size for the proposed method than OT/CRaSh, and the paper states it focuses on soft prompts. It is not fully clear from the paper whether the OT/CRaSh baselines in Table 1 are reconfigured to use the same adapter family or remain closer to their original larger-adapter forms. Because of this, some of the observed differences may reflect adapter design choices rather than LLE itself. At minimum, the paper should clarify this more precisely.
- **The scope of empirical validation is somewhat limited for the breadth of the claims.**  
  The experiments cover three models, which is good, but only 1.5B–3B scale and only four English QA benchmarks. This is enough for a proof-of-concept, but not enough to convincingly support broad claims about modern LLM deployment settings or capability privacy more generally.
- **The paper acknowledges but does not really characterize the privacy-utility tradeoff induced by \(H\).**  
  The text says downstream accuracy is “remarkably robust” to increasing \(H\), while Figure 4 is described as showing CPL decreases then plateaus. Without the actual figure values in the extracted text, I would not overstate this either way, but the paper should provide a more quantitative discussion of the transfer/privacy tradeoff and recommend practical \(H\) selection.

### Trivial
- **CPL would benefit from better interpretation guidance.**  
  The metric is easy to compute, but the paper does not define operational thresholds for what constitutes acceptable leakage, and ratios can be unstable when the original model’s zero-shot score is low. This does not invalidate the metric, but it limits interpretability.

## Nice-to-Haves
- Add **attack-based privacy evaluations**, e.g., whether a third party can recover capability by further tuning or distilling the emulator.
- Measure **empirical gradient alignment** directly (e.g., cosine similarity between \(\nabla_P L_E\) and \(\nabla_P L_M\)) and report **margin residuals** \(L_E-L_M-H\) on held-out prompts and data.
- Test at least one **additional adapter family** such as LoRA to support the paper’s broader applicability claims.
- Expand to at least one **generation-style or non-QA benchmark**, since “capability privacy” is broader than multiple-choice QA.
- Clarify more explicitly how **OT and CRaSh are instantiated in the soft-prompt setting**, so the reader can separate gains from emulator construction versus gains from adapter choice.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper is unfair because the asymmetry favors the proposed method in adapter size / parameter budget.”**  
  Kept only in weakened form as a clarification issue. It is valid to ask for clearer attribution, but under the review policy this should not be framed as an unfair-comparison flaw when the asymmetry does not advantage the baseline.
- **Generic complaint that larger models (7B/13B/70B) are required for validity.**  
  Removed as a core weakness. The current evaluation on three models is a reasonable proof-of-concept. Scaling to larger models would strengthen the paper, but absence of 7B+ experiments is not by itself a decisive flaw.
- **Formatting/readability complaints about equation rendering.**  
  Removed because the user explicitly noted parser artifacts and such points are not substantive.
- **Reproducibility complaints about missing release status, missing artifacts, or whether cited models/datasets/tools exist.**  
  Removed per instruction.
- **Claim that the paper should compare to unrelated missing prior work.**  
  Removed because external related-work completeness cannot be verified here.

## Novel Insights
The most important synthesis is that the paper has two qualitatively different contributions, and they should be evaluated separately. The **problem framing and the CPKD mechanism** are materially useful and empirically supported. In contrast, the paper’s **headline theory for LLE is stronger than what the implemented procedure can justify**: the proof assumes a globally exact loss offset, while the algorithm only fits that condition approximately on sampled prompts and data. This means the paper is strongest when read as an empirical method for reducing emulator utility while retaining transfer in a soft-prompt setting, and substantially weaker when read as providing a theorem-backed guarantee of exact gradient preservation and privacy. Reframing the contribution in those more modest but still interesting terms would make the work more credible.

## Suggestions
- **Reframe Theorem 1** as an idealized proposition and explicitly distinguish it from what Eq. (7) can guarantee in practice.
- **Add empirical verification** of the theorem’s intended mechanism: report held-out \(L_E-L_M-H\) residuals and prompt-gradient cosine similarity during downstream tuning.
- **Strengthen the privacy section with adversarial evaluations**, not just CPL. Even one concrete recovery/extraction attack would materially improve the paper.
- **Clarify the baseline instantiation**: specify whether OT/CRaSh are adapted to the same soft-prompt regime in Table 1 and what exactly differs besides emulator construction.
- **Tone down the generality claims** beyond soft prompts unless supported by new experiments.
- **Improve the discussion of CPL** by explaining when it is informative, when it can be unstable, and how model owners should interpret specific values in practice.



# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0]
Average score: 3.3
Binary outcome: Reject
