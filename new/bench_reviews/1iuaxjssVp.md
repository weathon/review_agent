Now let me run calibration searches in parallel to anchor the score.Now I have sufficient calibration data to write the final review. Let me synthesize everything.Now I have sufficient information to write the final review.

---

## Summary

InvMSAFold proposes a two-stage inverse folding approach: a neural network (built on the ESM-IF1 encoder + 6-layer Transformer decoder) generates the parameters of a low-rank Potts pairwise model in a single forward pass, which is then sampled efficiently on CPUs. The model is trained on multiple sequence alignments (MSAs) rather than single native sequences, directly targeting the "one-to-many" nature of inverse folding. The paper demonstrates dramatically better coverage of natural sequence space (KL divergence 0.49 vs 15.8 for ESM-IF1) and covariance reconstruction (median Pearson 0.53 vs 0.31), with maintained structural fidelity at large sequence distances from the native.

---

## Strengths

- **Linear-time pseudo-likelihood computation via low-rank parameterization** (Eqs. 4–7): The derivation reduces both memory and computational cost from O(L²) to O(L), making training and inference tractable without materializing the full coupling tensor. This is technically non-trivial and clearly presented.

- **Dramatic sequence space coverage improvement**: Table 1 reports average KL divergences of 0.49 (inter-cluster) and 0.67 (intra-cluster) for InvMSAFold-AR vs. 15.8 and 11.9 for ESM-IF1—roughly 25–30× improvement. Figure 6 provides a vivid visual confirmation for domain 1xqiA00 (KL=0.27 vs. 18.28). Results hold on a hard inter-cluster test set with novel folds not seen during training.

- **Superior covariance reconstruction**: Figure 5 and its accompanying table show median Pearson correlations of 0.53 for InvMSAFold-AR vs. 0.31 for ESM-IF1 on both test splits—directly supporting the paper's core claim of capturing evolutionary landscape statistics.

- **Structural fidelity maintained at high sequence distances**: Figure 8 shows that while ESM-IF1's AlphaFold-predicted RMSD degrades sharply at sequence distances >0.75, InvMSAFold-AR and InvMSAFold-PW maintain substantially lower RMSD, demonstrating that diversity does not come at the cost of fold preservation.

- **MSA training supervision as principled design choice**: Section 2.4 introduces a training approach using subsampled MSAs (64 sequences per step), directly addressing the known one-to-many limitation of single-sequence training objectives. This is a clear and well-motivated departure from ESM-IF1/ProteinMPNN.

- **Well-designed hierarchical test splits** (Section 3): Inter-cluster, intra-cluster, and MSA test sets provide progressively harder generalization tests with controlled homology levels, giving an unusually careful experimental framework for this subfield.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing DCA/plmDCA upper-bound baseline**: The paper's central evaluation measures how well structure-conditioned Potts parameters reproduce MSA covariance statistics. However, the most informative reference point—a Potts model fitted *directly* to each test MSA via plmDCA or ArDCA (the exact methods cited in the paper, e.g., Trinquier et al., 2021; Cocco et al., 2018)—is never included. While DCA requires the MSA at inference time and is therefore not a direct competitor (InvMSAFold operates from structure alone), it provides the natural performance ceiling: how well can a pairwise model capture MSA statistics when it has direct access to the data? Without this reference, the absolute metric values (Pearson 0.53, KL 0.49) cannot be contextualized. Concretely, if DCA achieves Pearson 0.95 from the MSA, the 0.53 score signals there is significant room for improvement; if DCA achieves 0.60, InvMSAFold is near-optimal. This comparison is standard in the DCA literature and its absence means the performance quantification is incomplete.

- **Speed comparison is hardware-asymmetric and excludes one-time NN overhead**: Figure 4 compares InvMSAFold-AR on a single CPU core (i9-13905H) against ESM-IF1 on a GPU (RTX 4060), which is not a like-for-like hardware comparison. More importantly, Figure 4 shows sampling time *after* the Potts parameters have been generated—it excludes the one-time forward pass through the ESM-IF1 encoder and 6-layer Transformer decoder that InvMSAFold requires per structure. For settings where few sequences are needed per structure, this amortized upfront cost could dominate total wall time. The paper presents "orders of magnitude faster" as a headline result without showing the crossover point at which InvMSAFold's amortized cost actually wins overall. The practical deployment story is compelling, but the headline claim is not fully established in its current form.

### Minor

- **Missing ablation isolating MSA supervision vs. Potts architecture**: The paper compares InvMSAFold (trained on MSAs with Potts output) against ESM-IF1 (trained on single native sequences with autoregressive output), mixing two design decisions: (a) MSA-based training supervision, and (b) the Potts pairwise architecture. The conclusion in Section 5 itself acknowledges "the idea is not specific to our model formulation and could also be applied to other architectures, such as ESM-IF1." This implies that MSA training supervision might alone drive the diversity improvements, independent of the Potts parameterization. A fine-tuned ESM-IF1 (or ProteinMPNN) trained with MSA supervision—while not the paper's goal—would clarify what component of InvMSAFold's advantage is attributable to architectural choices versus supervision.

- **Small sample size for structural fidelity analysis**: Figure 8 averages over 14 structures from the intra-cluster test set. For a metric as variable as AlphaFold-predicted RMSD, 14 structures is insufficient to establish reliable conclusions, and no variance estimates are reported. The inter-cluster results (where InvMSAFold's performance degrades) are relegated to the appendix.

- **"Functional integrity" overclaimed in the abstract**: The abstract states that InvMSAFold generates sequences "while preserving structural and functional integrity." Functional integrity is evaluated solely via computational proxies (Thermoprot thermostability score and Protein-Sol solubility prediction), and the property sampling analysis is not conditioned on sequences that pass the structural fidelity check from Section 4.5. The claim exceeds what the evidence supports; "preserving structural integrity (as predicted computationally)" would be accurate.

### Trivial

- Section 2.1 does not explicitly state whether the ESM-IF1 encoder is frozen or fine-tuned during InvMSAFold training. The text says embeddings are "created" using the encoder and Gaussian noise is added, suggesting it is frozen, but this should be stated explicitly for reproducibility.

---

## Nice-to-Haves

- Including ProteinMPNN in the speed comparison (Figure 4) would be informative, since it is more commonly used than ESM-IF1 and computationally lighter.
- A structure-ablation control (using random/shuffled structure encodings) would demonstrate whether the structure signal actually guides the Potts parameter generation, as opposed to the decoder learning per-superfamily statistics from MSA training alone.
- Conditioning the property sampling analysis (Figure 9) on sequences that pass a structural fidelity filter (e.g., AF2 RMSD < threshold) would strengthen the virtual screening application claim.
- More examples in the main text (beyond the single 1ny1A00 domain) for the property sampling analysis would make Section 4.6 more convincing.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Circular evaluation design" (Harsh Critic, Major)**: The criticism is that evaluating on MSA statistics is circular because InvMSAFold is trained on MSAs. This misunderstands the contribution: the paper explicitly trains on MSAs to capture diversity and demonstrates this works better than a model not designed for this task. The paper itself acknowledges ESM-IF1 wasn't trained for MSA diversity. This is not circular — it IS the claimed contribution. The legitimate version of this concern (the missing ablation isolating supervision vs. architecture) is retained as a Minor weakness above.

- **"Thermoprot binary classifier used as continuous distribution" (Harsh Critic)**: The paper's Figure 9 caption explicitly says "the y-axis [shows] the output of Thermoprot, a binary classifier (0 is mesophilic and 1 thermophilic)." The y-axis range 0–1 and the smooth contour plots in Figure 9 indicate that Thermoprot outputs a probability score, not a hard binary label. Using the classifier's probability output as a continuous feature for visualization is standard practice, not misleading. This criticism reflects a misreading.

- **"Coupling rescaling by 1/max(i,j) introduces systematic bias" (Harsh Critic)**: The paper justifies this rescaling from Ciarella et al. (2023), and the explanation in Section 2.3 is reasonable: the rescaling compensates for the position-dependent number of coupling terms in the autoregressive sum. The "systematic bias" claim is speculative and unsupported by evidence.

- **"90/10 MSA split inflates results" (Harsh Critic)**: The paper provides three test sets at different levels of generalization. The MSA test set is the least stringent, which the paper explicitly states. The inter-cluster test set (10% of clusters held out during training) is the clean evaluation, and results there are the primary evidence. The paper already addresses this concern.

- **Speed comparison strength (Strength Finder)**: Retained as a Major weakness due to the hardware asymmetry and missing NN overhead — moved out of the Strengths section accordingly. The practical CPU-deployability claim is kept as context.

- **"Broader range of predicted properties" strength**: Figure 9 shows only one domain (1ny1A00) in the main text. This is too limited for a standalone strength; more evidence is in the appendix (which the parser strips). Retained as a Nice-to-Have suggestion.

---

## Novel Insights

The two-stage amortization strategy — a large neural network generates the complete parameterization of a cheap, analytically tractable distribution, which is then sampled via fast direct methods — is a clean architectural pattern with broader applicability beyond protein design. The observation that training a pairwise model generator on MSA supervision yields substantially better recovery of evolutionary covariances than single-sequence supervised models suggests that the training objective (diversity matching) is at least as important as the choice of generative architecture. The explicit low-rank constraint (rank K ≪ L) enabling O(L) pseudo-likelihood and sampling is a practically useful contribution that may generalize to other settings where Potts-like models are needed for long sequences.

---

## Suggestions

1. **Add a DCA upper-bound comparison**: Fit plmDCA or ArDCA directly to each test-set MSA and report Pearson correlation and KL divergence. This single experiment would resolve the most significant interpretability gap and either greatly strengthen the paper (if InvMSAFold is near-ceiling) or identify a clear direction for improvement.

2. **Report complete pipeline timing**: Add to Figure 4 (or a companion figure) the one-time cost of the encoder + decoder forward pass, showing total time-to-first-sample and time-per-additional-sample for both methods. Report ESM-IF1 on CPU as well.

3. **Clarify encoder frozen/fine-tuned status** in the Methods section.

4. **Expand Figure 8 to more structures** with variance estimates; include inter-cluster results in the main text.

---

## Score and Decision

**Calibration anchors:**

- *ProfileBFN* (PSiijdQjNU, 8/8/8/6, Oral): A comparable paper on MSA-based protein family generation with extensive biological validation and multiple baselines. InvMSAFold is similar in scope but lacks a ceiling comparison and ablations, placing it below this level.
- *KW-Design* (mpqMVWgqjn, 6/6/6, Poster): Accepted protein inverse folding paper with strong empirical results on standard benchmarks. InvMSAFold has a more novel problem formulation but has comparably important evaluation gaps.
- *MSA Generation* (bM6LUC2lec, 6/6/5, Reject): Borderline paper with MSA-related focus and comparable evaluation completeness; was ultimately rejected.
- *AlphaFold Distillation* (3pgJNIx3gc, 3/5/3, Reject): Severely flawed methodology; much weaker than InvMSAFold.

InvMSAFold clearly exceeds the AlphaFold Distillation anchor (3) and the non-comprehensive latent diffusion papers (3–4). It is comparable to the KW-Design/MSA Generation anchor cluster (5–6), with a more novel problem framing and stronger empirical results, but with the important gap of no ceiling comparison. The missing DCA baseline and the incomplete speed analysis prevent the paper from reaching the ProfileBFN level (8). The center of the comparable anchor cluster is ~5.5–6, and InvMSAFold sits at the lower end due to the evaluation gaps on its central claims.

**Originality:** High — the two-stage structure→Potts approach for diversity-focused inverse folding is genuinely novel.  
**Importance of research question:** High — diverse sequence generation from structure is practically important and underexplored.  
**Claim support:** Moderate — the improvement over ESM-IF1 is convincing, but without a DCA ceiling the absolute performance level is uninterpretable.  
**Experimental soundness:** Moderate — well-designed test splits but key baselines missing; small sample sizes in Figure 8.  
**Clarity:** Good — architecture and derivations are clearly explained.  
**Value to community:** Moderate-high — fast CPU-based sampling of diverse sequences is practically useful.

**Final Score: 5.5**  
**Decision: Reject**

The paper presents a genuine contribution — a principled and technically sound approach to diversity-focused inverse folding with practical deployment advantages — but does not yet provide sufficient experimental evidence to fully support its headline claims. The missing DCA upper-bound comparison leaves the primary metrics uninterpretable in absolute terms; the incomplete speed analysis weakens the "orders of magnitude" headline. These are not unfixable issues: adding the DCA comparison and a complete timing analysis would likely bring this work up to an acceptable level.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>