Now let me do calibration searches to anchor my score.Now I have enough to write the final review. Let me synthesize everything.---

## Summary
SEPIT (Structure-Enhanced Protein Instruction Tuning) proposes a framework for general-purpose protein understanding by: (1) designing a structure-aware module that injects 3D structural information into protein language models (pLMs) via Gaussian-kernel attention biases and structural positional encoding, (2) connecting these enhanced pLMs to LLMs through a linear projector, and (3) employing a two-stage instruction tuning pipeline where MoE upcycling is applied in Stage 2. The authors also construct a 10M+ instruction dataset from Swiss-Prot and RCSB PDB, which they use for both training and evaluation.

---

## Strengths

- **Independent encoder benchmark (Table 3):** The SEPIT encoder achieves F_max of 0.893 on EC prediction and 0.674/0.497 on GO-MF/CC, outperforming ESM-2 (0.874 EC), ProST-ESM2 (0.878 EC), and all other baselines on these standard held-out tasks. This is the most reliable evidence in the paper that Stage 0 structure-aware pre-training genuinely improves protein representations.

- **Structural cross-modal transfer via training (Table 4):** SEPIT-TinyLlama trained with structure but evaluated with sequence-only input achieves METEOR 67.91, compared to PIT-TinyLlama's 66.19 using the same sequence-only inference — a clear demonstration that structure training transfers knowledge to sequence-only representations, beyond what a purely sequence-trained model learns.

- **MoE expert routing analysis (Figure 3):** Protein and text tokens follow systematically distinct expert pathways across layers, unlike in vision-language MoE models. This non-trivial mechanistic observation validates the design choice of preserving all protein token representations rather than compressing them as in vision-language work.

- **Honest negative TrEMBL ablation (Table 2, Section 5.3):** Doubling training data with lower-quality TrEMBL annotations (annotation score ≥ 4) actually reduces performance (↓2.69% BLEU-2, ↓2.13% ROUGE-L), even at double the GPU cost. This honest reporting of a negative result provides genuinely practical guidance for protein instruction dataset construction.

---

## Weaknesses

### Fatal
None.

### Major

- **The headline architectural claim — that structural information meaningfully enhances protein understanding — is insufficiently supported at inference time.** Table 4 directly confirms this: SEPIT-Llama with structure at inference achieves BLEU-2 of 60.81 vs. 60.64 without structure (0.17 absolute; 0.28%), and SEPIT-TinyLlama-MoEs similarly scores 60.28 with vs. 59.98 without (0.30 absolute). These differences are negligibly small. The structure module's contribution is real during training (the "w/o Structure" ablation in Table 2 shows a 4.08% relative BLEU-2 drop), but this effect is attributable to encoder pre-training rather than to using structural signals at inference. The paper's headline framing — "structure-enhanced" — implies runtime structural reasoning that the numbers do not support.

- **Stage 0 ablation is unavailable, leaving the contribution of structural pre-training to the full downstream pipeline unquantified.** The paper honestly acknowledges this: under FP16 AMP, the randomly initialized structure-aware module causes gradient overflow, so the "w/o Stage 0" row in Table 2 is empty. The authors substitute the independent encoder evaluation in Table 3 to partially fill this gap, which does confirm the encoder's quality. However, this does not establish whether Stage 0's benefit flows through structural information or simply through additional pre-training of the pLM. The central Stage 0 contribution to downstream generation and closed-set accuracy remains unverified.

- **Primary evaluation is on an author-constructed benchmark using n-gram metrics that heavily reward format-matching.** The test set is drawn from the same Swiss-Prot annotation distribution as training. The enormous BLEU gap over zero-shot GPT-4-turbo (60.81 vs. 4.21) primarily reflects that SEPIT reproduces Swiss-Prot annotation phrasing, not superior protein biology knowledge. The BERTScore gap is far smaller (95.76 vs. 85.71), and even the plain TinyLlama-Chat fine-tuned on the same data achieves BLEU-2 of 51.16. The closed-set accuracy metric is more informative and harder to dismiss (79.97% vs. 58.58%), but no protein-disjoint (e.g., <30% sequence identity) verification between train and test is reported.

### Minor

- **The 15%/85% sequence vs. structure probability in Stage 1 is puzzling given the paper's stated scarcity of structural data.** Section 4.2 states "we randomly input protein data, both those with only sequences and those paired with structures, into our protein encoder at probabilities of 15% and 85%, respectively." If 85% of Stage 1 data is structure-paired, this seems inconsistent with the repeated claim that structural data is "rare." The paper does not clarify whether AlphaFold-predicted structures are used for Swiss-Prot proteins in Stage 1; without this clarification, a key training detail is ambiguous.

- **The ablation study reports only relative percentage changes, not absolute scores with variance.** Table 2 reports values like "↓ 4.08%" without stating the absolute metric values or any uncertainty estimates. This makes it impossible to judge whether small differences (e.g., "↓ 0.52% METEOR" for w/o MoEs) are meaningful or within noise.

- **Case studies are cherry-picked and no failure analysis is presented.** Table 5 selects two examples where SEPIT answers correctly. Case 2 (GO term: "fatty acid beta-oxidation") is trivial — even the PIT model answers correctly. A systematic failure analysis across underrepresented protein families would better characterize generalization.

### Trivial

- The coefficient ω in Equation 6 that regulates the magnitude of structural positional encoding is introduced without any discussion of how it was set or its sensitivity.

---

## Nice-to-Haves

- Evaluate the **full model** (not just the encoder) on the same EC and GO benchmarks used in Table 3, using LLM-generated text outputs parsed for function labels. This would provide independent validation of the generation pipeline beyond the self-constructed benchmark.
- Include a **human or LLM-judge evaluation** of open-ended responses for factual biological correctness. BLEU/ROUGE cannot measure whether answers are biologically accurate.
- Report explicit **protein-level train/test disjointness** statistics (e.g., percentage of test proteins with < 30% sequence identity to any training protein) to rule out near-duplicate leakage.
- Analyze model performance on **protein families with very few training examples** to test generalization vs. memorization.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"SEPIT achieves superior performance over GPT-4" as a core strength (Strength Finder):** This is partially a formatting/distribution artifact as discussed above. The strength in closed-set accuracy (79.97% vs. 58.58%) is real and retained, but the BLEU-2 gap (60.81 vs. 4.21) framed as demonstrating "superior protein understanding" is misleading. Removed from strengths.

- **"Well-designed ablation study isolating each contribution" (Strength Finder):** The Stage 0 ablation is explicitly unavailable (Table 2 row is empty), so the claim of systematic isolation is false. The partial substitution via Table 3 is acknowledged in the paper. Removed from strengths.

- **"Two-stage pipeline avoids training instability" as a strength (Strength Finder):** The gradient overflow problem is an engineering limitation caused by random initialization, not a positive contribution. Removed.

- **Harsh critic's claim that the paper's central contribution is entirely invalidated by evaluation design:** Overclaimed. The encoder evaluation (Table 3) on independent EC/GO benchmarks provides genuine independent validation, and the closed-set accuracy metric is more robust. The benchmark concern is a major weakness, not a fatal one.

- **Harsh critic's claim that "zero-shot models cannot accomplish protein understanding tasks well" is unfairly stated:** Partially valid observation, removed as a separate weakness since it is subsumed under the BLEU/ROUGE metric discussion.

---

## Novel Insights

The most genuinely novel finding in SEPIT is the cross-modal transfer demonstrated in Table 4: a model trained with structural information consistently outperforms a purely sequence-trained model even when structure is withheld at inference time (e.g., METEOR 67.91 vs. 66.19 for TinyLlama variants). This suggests that structural pre-training instills geometric regularities into sequence representations that persist without runtime structure input — a mechanistically interesting result, though one that undercuts the case for deploying structural information at inference. The MoE routing visualization (Figure 3) further reveals that protein and text tokens self-organize into distinct expert pathways across layers, a qualitatively different behavior from vision-language MoEs and offering modest mechanistic insight into how heterogeneous molecular and linguistic tokens interact in shared transformer decoders.

---

## Suggestions

1. **Re-evaluate the headline claim:** Frame the structural contribution honestly — it primarily benefits encoder pre-training, not inference. The paper's title and abstract should reflect this distinction.
2. **Add full-model evaluation on EC/GO benchmarks** (Table 3-style but for LLM output) to provide at least one independent generative benchmark beyond the self-constructed test set.
3. **Clarify structural data proportion in Stage 1** and specify whether AlphaFold predicted structures are used; this affects reproducibility of the entire training setup.
4. **Address Stage 0 ablation** by rerunning with BF16 or mixed-precision safeguards, or quantify its contribution via a proxy experiment that does not cause overflow.
5. **Report absolute values + standard deviations** in Table 2 ablations rather than only relative changes.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Human Score | Comparison to SEPIT |
|---|---|---|
| `Tlsdsb6l9n.md` (Mol-Instructions) | 7.0 | Most comparable topically — also a protein instruction dataset + LLM paper; accepted. SEPIT adds structure and MoE but has weaker evidence for its key structural claim and a more suspect evaluation setup. |
| `BksqWM8737.md` (ProteinBench) | 6.5 | Evaluation framework for protein foundation models; accepted. SEPIT is narrower in scope but provides more new methodology. |
| `DkhYlWZq84.md` (Protein Captioning) | 4.5 | Narrower protein captioning paper with similar dataset/metric issues; rejected. SEPIT is broader and has more thorough analysis, including independent encoder validation. |
| `AK9uRqzLjt.md` (LLaPA) | 4.75 | Protein enzyme function prediction with LLM, narrower scope; rejected. SEPIT has better experimental coverage. |
| `nbia2X0urs.md` (Multimodal Protein Function) | 4.75 | Multimodal protein function, narrower; rejected. SEPIT is more comprehensive. |

SEPIT falls clearly above the rejected protein papers (4.5–4.75) due to its broader scope, the independent encoder benchmark (Table 3), the honest negative TrEMBL result, and the MoE analysis. However, it falls meaningfully below Mol-Instructions (7.0) because (a) the headline structural claim is only weakly supported at inference, (b) Stage 0 ablation is unavailable, and (c) the benchmark design inflates the appearance of improvement. The paper sits between these clusters, closer to a borderline accept — enough genuine contribution and engineering effort, but the core claim is overstated relative to what the experiments support.

**Originality:** Moderate — integrates existing ideas (structure denoising, CLIP, MoE upcycling) into a protein-specific pipeline.  
**Importance:** Relevant research direction.  
**Claim support:** Partially supported — encoder improvements are real and independently validated; structure-at-inference claim is not.  
**Soundness:** The experiments are competently executed but the Stage 0 ablation gap is a real methodological hole.  
**Clarity:** Adequate; the training proportion ambiguity (15%/85%) is a notable clarity issue.  
**Value to community:** The dataset and encoder could be genuinely useful resources.

**Final score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>