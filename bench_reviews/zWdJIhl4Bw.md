## Summary
This paper studies how to adapt a perspective-pretrained 3D Transformer (VGGT) to equirectangular panoramic inputs without retraining the backbone. The proposed “projection-domain adaptation” combines ERP-consistent ray lifting, ray-field token augmentation, head-only dual-branch LoRA, and latitude-aware depth uncertainty weighting, and shows strong gains over naïve ERP fine-tuning on a curated Matrix-3D subset as well as indoor 360° datasets, while using far fewer trainable parameters and training compute.

## Strengths
- **The paper isolates two concrete projection-domain failure modes and ties them to specific design choices.** Section 3.2 is more than generic motivation: it distinguishes **measure mismatch** (planar loss on spherical pixels) from **proxy-focal entanglement** (using fictitious pinhole intrinsics for ERP), and the proposed remedies in Sections 3.3–3.5 map cleanly onto these failures.
- **The parameter-efficiency result is genuinely strong and specific.** Across the main tables, the head-only LoRA variant is consistently close to the authors’ own full-FT variant while updating only ~0.6M parameters versus ~35M, and with substantially lower training time (e.g., Appendix A.10: 28h on 1 A100 for LoRA vs. ~185h on 4 A100 for the authors’ full-FT variant).
- **The paper shows that naïve full fine-tuning can be worse than minimal geometric interface correction.** This is an interesting and nontrivial empirical outcome: plain VGGT full fine-tuning under ERP is much worse than the proposed interface-aware adaptation across depth, pose, and 3D point quality.
- **The work goes beyond a single synthetic benchmark.** It includes indoor real 360° evaluation on Stanford2D3D and Matterport3D, plus OOD transfer from Matrix-3D to indoor data, which helps support the claim that the method is not narrowly overfit to one training setup.
- **Some diagnostic analysis is unusually useful.** Table 7 (“Where to adapt?”) directly addresses adaptation locus, and Table 9 provides evidence that the predicted uncertainty is not arbitrary: high-σ pixels indeed concentrate a disproportionate share of squared depth error.

## Weaknesses

###: Fatal
None.

### Major:
- **The central “interface, not backbone” claim is only partially supported because the strongest counterfactual is missing.**  
  The paper argues that correcting the projection interface is the right locus of adaptation and that head-only LoRA suffices, but the full comparison to a *successfully trained* backbone+head fine-tuning model under the same corrected interface is absent. In Section 4.6 / Appendix A.6, the authors state that “**Full backbone+head finetuning under the ERP interface was unstable in preliminary runs**” and Table 7 only compares head-only LoRA against backbone LoRA variants, not against a converged full-FT-with-interface model. This does **not** invalidate the empirical value of the proposed method, but it does weaken the stronger architectural claim that backbone adaptation is unnecessary in principle rather than merely harder to optimize in this setting.
- **The Matrix-3D evaluation relies on very aggressive curation, which narrows the scope of the claims.**  
  Section 4.1 and Appendix A.2 show that the dataset was reduced from **116,759** sequences to **2,196**, explicitly emphasizing “mid- and near-range geometry” and filtering out many extreme long-shot, sky/grass-dominated cases. This is a reasonable benchmark-design choice for studying geometric adaptation, but it also means the evidence is strongest for scenes with substantial visible structure and weaker for the most challenging panoramic regimes. Since the paper’s framing is broad (“projection-domain adaptation” for panoramic scene reconstruction), this curation should be treated as a substantial scope limitation rather than a minor implementation detail.
- **Key implementation details of how ray-augmented tokens enter the frozen backbone are underspecified.**  
  Equation (6) defines token augmentation as \(t_i^{(0)}(u,v)=t_i^{RGB}(u,v)\oplus \Phi(r(u,v))\), i.e., concatenation of ray embeddings to image tokens before the frozen backbone. But the manuscript does not clearly explain how this increased dimensionality is reconciled with VGGT’s fixed input feature size, nor whether an additional projection layer is used and, if so, where it sits and whether it is trainable. Since the method’s core mechanism is token-level ray augmentation into a frozen pretrained model, this is an important technical omission affecting clarity and reproducibility.

### Minor
- **Some wording overstates the geometric guarantees.**  
  The paper sometimes uses language like “restores the geometric invariances broken by ERP” and says the tokenization “enables directional equivariance” or “SO(3) directional consistency.” What the method clearly provides is an explicit directional coordinate prior and better projection-consistent supervision; it does not establish formal SO(3)-equivariance of the frozen transformer. This is mainly a claim-calibration issue, but the current phrasing is stronger than what is demonstrated.
- **The ablation coverage is good but still incomplete on the most mechanism-specific design choices.**  
  Table 4 covers geometric interface, loss design, and LoRA rank, and Table 7 covers placement. However, the paper does not isolate sensitivity to the **ray embedding design itself** (e.g., embedding dimension or encoding choice), even though this is central to the proposed interface.
- **The efficiency headline could be contextualized more carefully.**  
  The repeated “~25× lower cost” comparison appears to be against the naïve full fine-tuning baseline in the main tables, while Appendix A.10 indicates that compared to the authors’ own interface-corrected full-FT variant the wall-clock reduction is smaller on a per-training-run basis (28h on 1 A100 vs. ~185h on 4 A100). The main claim is directionally correct—LoRA is much cheaper—but the paper should be more explicit about which full-FT comparator each efficiency number refers to.
- **3D evaluation is narrower than the paper’s world-model framing suggests.**  
  The 3D point quality metric in Table 3 is useful, but it is derived from reconstructed points via predicted depth+camera and evaluated after Umeyama alignment. This supports reconstruction quality, but it does not fully establish scale-faithful, long-horizon, temporally consistent 3D world modeling. The paper’s claims are strongest for multi-view depth/pose transfer, and somewhat broader than the current evaluation directly proves.

### Trivial
- **Latitude-specific evidence for the claimed correction would make the story sharper.**  
  Since the paper’s theoretical motivation emphasizes ERP distortion varying with latitude, showing error as a function of latitude or pole-vs-equator breakdowns would directly validate that the proposed weighting and ray interface are fixing the intended problem rather than just improving average performance.

## Nice-to-Haves
- Add a stronger optimization study for full fine-tuning under the corrected ERP interface, since this is the most important missing comparator for the paper’s main conceptual claim.
- Provide an explicit architectural diagram or formula for how concatenated ray features are projected into the frozen VGGT token dimension.
- Report latitude-resolved depth/pose errors or targeted pole/equator analyses to validate the geometric story more directly.
- Include uncertainty calibration analysis beyond correlation (e.g., calibration plots), since the current evidence mainly shows usefulness for weighting rather than calibrated confidence.
- Evaluate longer sequences than the fixed \(K=10\) frame protocol to better support the “world model” framing with respect to drift and temporal consistency.
- Clarify more prominently in the main text that many depth metrics are median-aligned per sequence, and distinguish claims about relative depth quality from claims about metric-scale 3D reconstruction.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Concerns about code/weights release status or institutional approval.**  
  Removed per instruction. The paper explicitly discusses intended release in Appendix A.9/A.12, and release-status concerns should not be treated as scientific weaknesses.
- **Complaints about missing comparisons to uncited external panoramic or PEFT methods.**  
  Removed because I cannot verify omitted related work beyond the paper, and the review should not speculate about missing baselines from outside the submission.
- **Claims that comparisons are unfair because baselines are weaker under ERP than the proposed method.**  
  Removed. The paper also includes cubemap variants for several baselines, and asymmetries that favor the baseline are not a valid weakness under the stated rules.
- **Generic reproducibility nitpicks about hyperparameters.**  
  Removed. Appendix A.10 already provides optimizer/schedule/augmentation/LoRA details at a level that is broadly standard for this kind of work.
- **“Not even a paper” style objections about the existence or release of referenced tools/datasets/models.**  
  Removed per instruction and because they are not grounded in the manuscript.
- **A pure demand for more geometry metrics such as Chamfer/normal accuracy as a core flaw.**  
  Weakened rather than kept as a major issue: Table 3 already reports 3D point quality (Acc/Comp/Overall after alignment), so it is inaccurate to say quantitative 3D evaluation is absent. The fair criticism is that the 3D evaluation is somewhat limited relative to the broad framing, not that it is missing.

## Novel Insights
The most interesting synthesis across the paper and reviews is that the work is strongest not as a universal statement that “backbones should never be adapted,” but as evidence for a more specific principle: **for projection shift, the highest-leverage intervention may be to repair the sensor interface before spending capacity on the model interior**. The results suggest that much of VGGT’s pretrained geometric prior survives panoramic transfer once two projection-specific mismatches are corrected—ray geometry and spherical sampling measure. At the same time, the missing successful full-FT-with-interface comparison means the paper currently supports this as a strong empirical design heuristic, not yet as a definitive architectural law.

## Suggestions
- **Clarify token integration rigorously.** Add one explicit equation or diagram showing the dimensional path from \(t^{RGB}\oplus \Phi(r)\) to the frozen VGGT input size, and state whether any projection layer is frozen/trainable and included in the parameter count.
- **Temper the strongest claims.** Replace wording implying formal equivariance/invariance restoration with more precise language about projection-consistent geometric conditioning and supervision.
- **Strengthen the central comparison.** If possible, add a more thoroughly tuned full fine-tuning experiment under the corrected interface; if not, explicitly narrow the claim from “head-only is sufficient/optimal” to “head-only is the most reliable and efficient strategy we found.”
- **Elevate the dataset-curation limitation into the main text.** State clearly that the curated Matrix-3D subset emphasizes scenes with stronger mid-range structure and that the conclusions are therefore best supported in that regime.
- **Add latitude-resolved diagnostics.** A plot of error versus latitude for naïve ERP, plain full FT, and the proposed method would directly test the paper’s core geometric hypothesis.
- **Be explicit about metric scale.** Since depth is median-aligned when scale is ambiguous and 3D points are Umeyama-aligned, the paper should clearly separate claims about geometric consistency from claims about absolute metric reconstruction.