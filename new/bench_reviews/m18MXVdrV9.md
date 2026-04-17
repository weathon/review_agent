## Summary

INFO-SEDD introduces a method for estimating mutual information (MI) and entropy on high-dimensional discrete data by leveraging Continuous Time Markov Chains (CTMCs) and discrete diffusion score functions. The core idea is to express KL divergences as time integrals over CTMC transition rates using Dynkin's lemma, and to exploit an absorbing-state construction that enables computing both joint and marginal scores from a single trained model. The method is evaluated on synthetic benchmarks, text summarization model selection, and genomics motif discovery.

## Strengths

- **Novel and important problem framing**: The paper directly targets MI estimation for high-dimensional discrete data, a genuinely underserved setting where existing neural estimators (designed for continuous data) struggle via the "embedding trick." This is a meaningful contribution.

- **Elegant architectural insight**: The absorbing-state approach (Eq. 6) that allows a single score model trained on the joint distribution to recover marginal scores is a clever and practical design that reduces training overhead. The sparse rate matrix construction using one-token-at-a-time transitions also directly addresses the combinatorial explosion inherent in discrete state spaces.

- **Strong synthetic benchmark results**: Table 1 shows INFO-SEDD consistently tracking ground-truth MI across challenging regimes (MI=10–50, D=10–50), where all baselines either saturate, underestimate, or diverge badly. The gap widens substantially at MI≥20, which is compelling.

- **Principled theoretical positioning**: The error bound decomposition (Eq. 7) into estimation error and exponentially-decaying truncation bias provides a clear conceptual framework for understanding when the estimator works, even if the bound's practical tightness is not empirically validated.

- **Well-chosen applications**: Text summarization (model selection aligned with human evaluation metrics) and genomics (motif discovery via sliding-window MI) are natural domains for discrete MI and showcase the practical value of the method.

## Weaknesses

### Major:

- **Opaque core derivation (Eq. 4→5)**: The transition from Dynkin's formula (Eq. 3) to the explicit KL integrand (Eq. 4) is sketched without specifying the function f used in Dynkin's lemma, and the notation in Eq. 4 is garbled (e.g., $\overrightarrow{x}_t(x)$, $\overrightarrow{p}_i$, $\overrightarrow{q}_i$). The subsequent substitution of parametric scores for exact ratios (Eq. 5) also lacks an explicit justification that DWDSE-trained scores converge to the *ratios* needed. While the paper references Appendix E for proofs, the main text presentation is insufficient for a reader to verify that the estimator targets the correct quantity. This is the method's central contribution, and its inaccessibility undermines confidence. (Note: this is a presentation/clarity issue, not a claim that the derivation is wrong—the appendix may be sufficient, but key steps should be more transparent in the main text.)

- **Overclaiming on "arbitrary subsets"**: The introduction states the method enables "MI across arbitrary subsets of variables," but Eq. 6 specifically proves the marginal score property for full absorption ($Y_t = \emptyset$). The genomics application uses masking with sliding windows, which works through a different mechanism (conditioning on partially-masked data via the joint model), not through the subset absorption property claimed. The paper extrapolates from the full-absorption result to arbitrary subsets without demonstrating or proving that analogous identities hold for partial absorption.

- **Baseline fairness confound on real-world experiments**: In Sections 4.2 and 4.3, INFO-SEDD uses pretrained backbone models (MDLM-SMALL, CADUCEUS) in their native discrete token space, while competitors project tokens into learned embedding spaces and operate continuously. This architectural asymmetry confounds the comparison: the observed performance gap could stem from the backbone incompatibility rather than the estimation methodology itself. The paper's stated contribution is the estimation method, not the backbone choice.

- **Missing baselines on real-world tasks**: GAN-DIME is the most competitive baseline on synthetic data (Table 1) but is absent from the text summarization and genomics experiments without justification. Lee & Rhee (2024), which the introduction specifically critiques for relying on the "embedding trick," is not evaluated against. No classical discrete MI estimators (plug-in, Miller-Madow, NSB) are included as baselines despite being the most natural comparison class for discrete data.

### Minor:

- **No computational cost analysis**: The paper claims the method is "lightweight and scalable," but provides no wall-clock time, FLOPs, or memory comparisons against any baseline. Training a discrete diffusion model with DWDSE objectives is non-trivial computationally.

- **Error bound (Eq. 7) lacks practical grounding**: The constants $C_1, C_2$ and approximation errors $\epsilon_p, \epsilon_q$ are never measured or bounded empirically. It is unclear whether the bound is tight or vacuous in realistic regimes. No ablation on the time horizon $T$ and its effect on truncation bias is provided.

- **Qualitative-only motif discovery evaluation**: Figure 5 shows a visually compelling TATA-box detection via MI profiles, but the result is purely qualitative. No precision/recall or localization error metrics are reported, and no comparison to existing motif-finding tools (e.g., MEME) is provided.

- **Real-world "ground truth" references are heuristic**: The text summarization consistency test assumes MI grows linearly with $\rho$ based on entropy-rate estimates, and the genomics reference curve uses a classifier-based proxy $I(X,Y) = H(Y) - H_b(\text{Acc})$. These are reasonable but rough approximations; the paper should be more cautious in treating them as ground truth.

### Trivial:

- The notation is dense throughout, with overlines and arrows on symbols that compound readability issues (though some of this is attributable to PDF parsing artifacts).

## Nice-to-Haves

- An ablation study on the time horizon $T$ to empirically validate the truncation bias term behavior.
- A principled comparison where at least one competitor benefits from pretrained embeddings (e.g., using frozen LM embeddings rather than learned-from-scratch embedding tables).
- Guidance on when INFO-SEDD-J vs INFO-SEDD-C is preferred, beyond the empirical observation in genomics that the conditional variant is easier to optimize when Y is low-dimensional.

## Removed Points

- **Claim that Eq. 4-5 derivation is "not correct" or "targets the wrong quantity"**: The paper references a full derivation in Appendix E. While the main text is insufficiently transparent, the existence of the appendix derivation means we cannot definitively claim the estimator is wrong—only that it is hard to verify from the main text. Downgraded from fatal to major (clarity concern).

- **Claim that Eq. 6 "may not be exactly correct" for the described CTMC**: Without evidence that the property fails, and given the proof in Appendix A.3, this is not a justified fatal objection. The valid concern is the overgeneralization to "arbitrary subsets," which is retained above.

- **Demand for comparison with models "not yet released" (e.g., Lee & Rhee, specific baseline versions)**: Per the hard rule, all cited models are assumed available. The concern about missing Lee & Rhee is retained as a missing baseline, not as a question of availability.

- **Reproducibility concerns about undisclosed hyperparameters or model sizes**: The paper provides a code repository and detailed appendix descriptions. Minor implementation details are not a core weakness.

- **Formatting nitpicks**: Removed per hard rules.

- **Demand for confidence intervals / multiple seeds**: Single-run evaluation is standard in this community for large-scale benchmarks. Downgraded to nice-to-have.

## Novel Insights

The absorbing-state construction that enables joint-to-marginal score sharing (Eq. 6) is a genuinely novel architectural contribution specific to the CTMC framework. It exploits a structural property of the absorbing transition matrix—once a component is fully absorbed, its conditional distribution factors out of the joint dynamics, leaving the marginal intact. This is not a generic diffusion trick but a discrete-diffusion-specific insight, and it is what makes single-model MI across marginal/joint distributions feasible. The distinction between INFO-SEDD-J and INFO-SEDD-C also reveals an interesting practical trade-off: the conditional method is substantially easier to optimize when one variable is low-dimensional (as in classification), while the joint method is needed when masking subsets of a high-dimensional variable (as in motif discovery).

## Suggestions

- Add a step-by-step derivation walkthrough in the main text (or at minimum a clearly referenced proof sketch) for Eq. 4→5, specifying the choice of function f in Dynkin's formula and justifying the change of measure from reverse to forward process.
- Include GAN-DIME and at least one classical discrete estimator (e.g., plug-in with Miller-Madow correction for moderate sizes) in real-world experiments, or provide explicit justification for their exclusion.
- Report wall-clock training and inference times alongside the synthetic results.
- Conduct an ablation on T and empirically measure how the truncation bias manifests in practice.
- Add quantitative metrics for the motif discovery experiment (localization accuracy relative to known TATA-box positions).

## Score and Decision

**Calibration comparison:**

- **MINDE (Accept poster, scores 8/6/6/6, avg 6.5)**: The direct predecessor for continuous diffusion-based MI estimation. INFO-SEDD extends this to discrete data with novel architectural contributions (absorbing-state trick). MINDE had similar concerns (computational cost, missing comparisons, dense notation) but was accepted. INFO-SEDD shares these weaknesses but adds baseline fairness concerns on real data and an overclaim about arbitrary subsets. Roughly comparable, slightly weaker due to baseline concerns.

- **SEDD (Reject, scores 8/6/8/5/6, avg 6.6 but rejected)**: A foundational discrete diffusion paper with initially incomplete theory and preliminary experiments. INFO-SEDD's theory is similarly compressed but with an appendix, and experiments are more comprehensive. Slightly stronger empirically.

- **Discrete Diffusion Convergence (Accept poster, scores 8/6/8/6, avg 7.0)**: Pure theory paper with no practical experiments. INFO-SEDD has weaker theory (opaque main-text derivation, undefined constants) but much stronger empirical validation.

- **Reference Distributions MI (Reject, scores 1/5/5/5/5, avg 4.2)**: Trivial theoretical contribution and weak experiments. INFO-SEDD is clearly far above this.

- **F-DIME (Reject, scores 5/5/6/6/6, avg 5.6)**: These are the competitors! INFO-SEDD clearly outperforms them experimentally but has its own methodological concerns.

Given that INFO-SEDD has genuine novelty, strong synthetic results, and meaningful applications—but also has significant presentation issues in the core derivation, overclaims about subset support, and real concerns about baseline fairness on empirical tasks—I place it slightly below MINDE (6.5) and slightly above the rejected F-DIME-level papers. The theoretical presentation is the main issue holding it back; if the derivation were clearer and baselines more complete, this could be a 7.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>