Now I have sufficient information to write the final consolidated review. Let me compile everything.

## Summary

This paper studies whether data contamination necessarily invalidates LLM benchmark evaluations. Through controlled experiments training models from scratch with explicitly inserted contaminated examples, the authors quantify how contamination effects scale with model size, token count, and repetition frequency. They show that contamination effects *vanish* when training data is scaled beyond Chinchilla-optimal levels (demonstrated at 124M parameters), identify novel-data exposure as the key mechanism for forgetting (contrasting with prior work), and derive a theoretical framework based on cumulative weight decay in AdamW that bounds forgetting rates and is applied to OLMo-7B and Llama 3 405B.

## Strengths

- **Controlled experimental paradigm with explicit contamination insertion**: Unlike prior contamination work that must infer effects without training data access, this paper trains models from scratch with inserted contaminated examples at known positions and frequencies (Section 3.2), enabling direct causal measurement of contamination impact and forgetting. Near-duplicate filtering (Section 3.3, Figure 1) ensures contamination effects are correctly attributed.

- **Key finding that novel-data exposure drives complete forgetting**: Figure 3c shows that after 5 Chinchilla tokens of continued clean training, accuracy gaps from all contamination levels (4×–144×) completely disappear. The contrast with Figure 3d—where repeating the same 100M tokens causes forgetting to stabilize above zero—identifies novel data exposure as the critical mechanism, resolving the discrepancy with prior work (Tirumala et al., 2022) that found forgetting approaches a stable baseline.

- **Rapid forgetting demonstrated at 1B scale in OLMo-1B**: Figure 4 shows that contamination inserted at step 369,000 of OLMo-1B causes a 15pp accuracy increase, but 96% of this increase is reduced within just 1% of the remaining training time (~2000 gradient steps out of ~370,000 remaining). This provides strong evidence that forgetting operates on much faster timescales than full pre-training at a meaningful model scale.

- **Novel finding that uniformly distributed repetition causes strongest overfitting**: Figure 3e/f shows that questions seen uniformly throughout training exhibit stronger overfitting than those seen only early or late, suggesting spaced repetition aids memorization. This has practical implications since real contamination is often uniformly distributed across training, making it harder to forget than window-based contamination.

- **Weight decay framework provides useful mechanistic insight**: Equation (3) decomposes final model parameters as a weighted sum of all past gradient updates with exponentially decaying weights. Figure 5's visualization of how different training deciles contribute to the final model is an original way to reason about training dynamics, and the framework correctly predicts that empirical forgetting occurs faster than the theoretical bound (Figure 6).

- **Useful "N-times Chinchilla" framework**: Section 3.1 provides a clean vocabulary for distinguishing training regimes where contamination matters (1× Chinchilla) from those where it likely does not (5×+), aligning with how modern LLMs are actually trained.

## Weaknesses

### Fatal
None.

### Major

- **The central vanishing-contamination claim is demonstrated only at 124M parameters, creating a significant scale gap**: The paper's headline finding that "even 144 times of contamination can be forgotten if the training data is scaled beyond five times Chinchilla" (abstract) is supported exclusively by 124M-parameter experiments (Figure 2b, 3a-c). Meanwhile, Figure 2a shows contamination effects *increase substantially* with model size (5pp at 124M vs. 20pp at 1.6B for 4× contamination on 7B tokens), creating a direct tension: if larger models memorize more effectively, the Chinchilla multiple required to forget contamination may scale with model size in an unknown way. The OLMo-1B experiment (Section 4.3) provides partial evidence at 1B/17.5× Chinchilla, but only for mid-training window contamination with 4× repetition—not the 144× uniform contamination that is the paper's strongest claim. Without establishing how the forgetting threshold scales with model size, the paper cannot confidently project its 124M results to the 7B–405B regime where the practical question actually lives.

- **The theoretical framework establishes gradient contribution decay, not information forgetting—the paper's framing overclaims what it proves**: The derivation in Section 5.1 correctly shows that individual gradient contributions decay exponentially via cumulative weight decay in AdamW (Proposition 1, Equation 3). However, the paper frames this as demonstrating *forgetting of information*—e.g., "many LLMs, including Llama 3, have forgotten the data seen at the beginning of training" (abstract). A gradient term decaying to zero does not entail that the *information* it encoded is absent: later gradient updates from clean data can reinforce representations initially shaped by contaminated examples, creating indirect pathways through which contamination persists. The paper acknowledges this in one paragraph (around line 261: "if the weight updates at a later time step t₂ were aligned with past updates at t₁…"), but then applies the framework confidently to Llama 3 405B (Figure 5c) and states in the abstract that Llama 3 has "forgotten" early training data. The empirical results in Section 4 *do* show genuine informational forgetting (accuracy gaps vanishing), so the overall thesis has empirical support—but the Llama 3 405B conclusion rests entirely on this incomplete theory, not on experiments at that scale.

### Minor

- **The "3pp from single contamination under Chinchilla" claim is derived via linear extrapolation from 4× data**: The paper states (Section 4.1) that "a single time of contamination can lead to overfitting of as much as 3 percentage points" by dividing the 774M model's 4×-contamination gap (~15.6pp) by 4. However, Table 1 shows the repetition–overfitting relationship is strongly sub-linear (4×→15.6pp, 12×→32.1pp, 32×→43.8pp, 144×→46.9pp for 774M), meaning the first repetitions have larger marginal effects than later ones. This sub-linearity actually suggests dividing by 4 may *underestimate* the 1× effect, so the paper's "as much as 3pp" framing may be conservative rather than exaggerated. The paper's own footnote 1 reports 1× experiments at 124M produced "one or two percentage points," consistent with the linear approximation at that scale. However, the 774M claim lacks direct 1× verification, and the direction of the error is worth clarifying.

- **Uniformly spaced contamination (the hardest to forget) is the least-tested pattern in the forgetting experiments**: The forgetting experiments in Section 4.2 use window-based contamination (between Chinchilla 1× and 2×), while Figure 3e/f shows uniformly distributed contamination produces the strongest overfitting. This means the experimental setup that demonstrates complete forgetting actually uses the *easiest-to-forget* contamination pattern, while the practically most relevant pattern (uniform contamination throughout training) is not directly tested in the forgetting paradigm.

### Trivial
None.

## Nice-to-Haves

- A 2D sweep of model size × Chinchilla multiple at even 2–3 points would directly address the joint interaction between model size and token count, which is the crucial practical question.
- Per-benchmark breakdown of forgetting dynamics would reveal whether the vanishing effect is uniform across different types of tasks (commonsense reasoning vs. factual knowledge).
- A controlled experiment where clean training data is explicitly filtered to exclude knowledge relevant to contaminated benchmarks would disentangle "direct gradient decay" from "information re-encoding via clean data."

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh critic: "Abstract does not disclose the 124M scale limitation"** — The abstract says "even 144 times of contamination can be forgotten if the training data is scaled beyond five times Chinchilla, a regime characteristic of many modern LLMs." The "characteristic of many modern LLMs" phrase does implicitly connect the result to real systems, but this is a framing concern rather than a factual error. The actual scale limitation is the real issue (addressed as a Major weakness above), not the abstract's wording.

- **Harsh critic: "Figure 2c shows contamination effects that are not clearly decreasing with scale"** — Re-reading the paper, the Chinchilla scaling panel (Figure 2c) is explicitly about models trained at *1× Chinchilla*, which is the regime where the paper *expects* contamination to matter. The paper's argument is precisely that going *beyond* Chinchilla (Figure 2b) makes contamination vanish. The harsh critic conflates two different claims.

- **Harsh critic: "OLMo-1B contamination is mid-training, not early"** — The paper explicitly discusses this in footnote 3, noting that the model is "already fairly trained after the first Chinchilla" and that this is intentional because "there is evidence that observations are more quickly forgotten if the model has not yet learned representations." The mid-training placement is a deliberate design choice, not a gap.

- **Harsh critic: "The proposition is elementary and straightforward"** — This is a value judgment on novelty of the mathematical derivation, not a substantive weakness. The value of Proposition 1 lies in its application to real training runs, not in the difficulty of the proof.

- **Harsh critic: "The paper should discuss why uniformly spaced contamination produces strongest overfitting"** — The paper does discuss this: "suggesting that this spaced form of repetition helps the model remember" (Section 4.2). A deeper cognitive-science explanation would be nice but is beyond the paper's scope.

- **Strength Finder: "Full reproducibility" cited as a core strength** — While the paper provides code and data references, reproducibility is a baseline expectation rather than a distinguishing strength. Moved to removed.

- **Strength Finder: "Theory serves as valid upper bound with empirical forgetting consistently faster"** — While technically accurate, this is better framed as a property of the theory rather than an independent strength. The theory being conservative is expected for an upper bound.

- **Harsh critic: "Faster-than-predicted forgetting could mean information is being re-encoded through later clean data"** — This is speculative and conflates two different phenomena. Faster forgetting means the contamination effect decays faster, which is the opposite of information persistence. The re-encoding concern is valid for the theory's *lower bound* on forgetting (i.e., the theory might say forgetting is slow when it's actually fast), not for the empirical observation that forgetting is fast.

## Novel Insights

The paper's most novel insight is the identification of *novel data exposure* as the critical mechanism distinguishing complete forgetting (Figure 3a/c) from stable partial memorization (Figure 3d). This resolves a genuine discrepancy in the literature: prior work (Tirumala et al., 2022) found forgetting approaches a stable baseline, while this paper finds complete forgetting—the difference is entirely explained by whether the model continues to see the same data (multi-epoch) or encounters new data (streaming). This has a direct practical implication: the multi-epoch training regime that is common in small-scale academic settings is actually the *worst case* for contamination persistence, while the data-intensive streaming regime of modern LLMs is inherently more forgiving.

## Suggestions

- Narrow the framing of the Llama 3 405B analysis: explicitly state that the cumulative weight decay analysis shows the *direct gradient contribution* of early training data has decayed, and that indirect information persistence through aligned clean-data updates cannot be ruled out by the theory alone.
- Add a discussion of how the forgetting threshold might scale with model size, even as a theoretical argument (e.g., if contamination effects scale roughly linearly with model size at fixed token count, and the token-count forgetting rate is independent of model size as suggested by Figure 6d, then the Chinchilla multiple needed for forgetting might scale slowly or not at all with model size—this would strengthen the extrapolation argument substantially).
- Test 1× contamination at the 774M Chinchilla scale to directly validate the "3pp" claim, or reframe it as "at least 3pp" given the sub-linear relationship suggests the true value may be higher.

## Evaluation

**Originality**: The experimental paradigm of controlled contamination insertion with systematic scaling is novel. The cumulative weight decay framework, while mathematically simple, provides a new perspective on temporal contributions to model weights. The novel-data-vs-repeated-data distinction is a genuine insight. 

**Importance of research question**: High. Data contamination is a central concern in LLM evaluation, and understanding when it matters vs. when it doesn't is practically important and under-studied.

**Claims support**: The core empirical claims at 124M and OLMo-1B scales are well-supported. The extrapolation to 7B–405B models and the Llama 3 405B "forgotten" claim are not well-supported by the evidence presented.

**Soundness of experiments**: Strong at the scales tested—controlled, well-designed, with appropriate confidence intervals and near-duplicate filtering. The gap is in scale, not in methodology.

**Clarity**: Well-written with clear structure. The "N-times Chinchilla" framework is a useful conceptual contribution that aids clarity.

**Value to community**: Significant—provides the first controlled experimental evidence that contamination effects can vanish under realistic training conditions, which directly challenges the common assumption that any contamination invalidates evaluations.

## Score and Decision

**Calibration anchors**:
- **High (7.0–7.75)**: KS8mIvetg2 (7.5, oral, contamination detection with exact guarantees), 2Rwq6c3tvr (7.0, spotlight, contamination detection), syThiTmWWm (7.75, oral, benchmark gaming). These papers have cleaner theoretical guarantees or more direct empirical evidence. The paper under review has weaker theoretical grounding and a larger scale gap.
- **Medium (5.25–6.75)**: m2NVG4Htxs (6.75, poster, longitudinal contamination analysis using natural experiments), iZeQBqJamf (6.5, poster, scaling laws with controlled 104-model testbed), 79ZkWgY2FI (5.25, poster, small-to-large extrapolation with limited evidence), MF7ljU8xcf (6.0, poster, theoretical generalization bounds with limited empirical evaluation). The paper under review is stronger than m2NVG4Htxs (more controlled experiments), comparable to iZeQBqJamf (similar controlled experimentation ethos), and notably stronger than 79ZkWgY2FI (which had similar extrapolation concerns but weaker experimental design and no larger-scale validation).
- **Low (4.0–4.75)**: lwtaEhDx9x (4.75, reject, contamination/memorization with weak methodology), GbEmJmnQCz (4.40, reject, challenges memorization-generalization with flawed methodology). The paper under review is clearly above these—its experimental methodology is rigorous and its claims at the tested scales are well-supported.

The paper sits above the medium anchors because its experimental methodology is more controlled than most, and it provides partial evidence at 1B scale (OLMo-1B) that goes beyond pure small-model results. It sits below the high anchors because the central claim about contamination vanishing relies on 124M-only evidence for the strongest version (144× forgotten), and the Llama 3 405B analysis overclaims based on an incomplete theory. A score of 6.5 reflects genuine contributions with significant but addressable limitations.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>