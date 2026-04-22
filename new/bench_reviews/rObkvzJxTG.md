Now I have a thorough understanding of the paper and calibration anchors. Let me synthesize the final review.

**Key verification of harsh critic claims:**

1. **Ablation Table internal inconsistency**: Table 5 (lines 330-331) does show two rows with ✓✓✓ but different numbers (54.3 vs 59.3). This is a real issue — confirmed.

2. **IQR Smoothing's core mechanism**: The critic claims there is "no theoretical reason why the IQR of raw feature values should correlate with OOD severity." Let me verify — Equation 3 applies IQR to the concatenated representation vector h. The paper says in line 131-143 that they use representations (not raw inputs) because they are "informative dense vectors that contain less noise." The t-SNE visualization (Figure 3b) is the only evidence for the ordering. This is a valid concern — the mechanism is not theoretically justified.

3. **IQR importance claim contradicted by data**: Section 4.3 (line 313) says "IQR smoothing brings the most improvements to the model." But Table 5 shows IQR alone gets 37.1 on Kinetics50-C severity 3, while UA alone gets 52.1 and MIS alone gets 49.4. And IQR+UA (47.4) is worse than UA alone (52.1). The text claim is indeed contradicted by their own numbers. However, the claim might be interpreted in context of the full system (IQR+MIS = 58.0 vs UA+MIS = 56.0), where IQR adds more when combined with MIS. But the paper doesn't make this nuanced argument. This is a valid major weakness.

4. **Missing conservative adaptation baseline**: The baselines compared are unimodal TTA methods. A simple conservative baseline (skip adaptation when entropy is high) would isolate whether SuMi's gains come from specific mechanisms. This is a reasonable concern.

5. **IQR+UA underperforms UA alone**: Confirmed from table (47.4 vs 52.1). The paper doesn't address this.

6. **f(t) = t/iter requires knowing total iterations**: Confirmed from line 131. Valid concern for online TTA.

7. **MIS is not mutual information in the info-theoretic sense**: The loss (Eq 6) is indeed KL divergence, not mutual information in the Shannon sense. The naming is misleading but this is a minor issue.

Now let me write the final review:

## Summary
The paper proposes "multimodal wild TTA," a setting where test data contains a mixture of weak OOD (single modality corrupted) and strong OOD (multiple modalities corrupted or missing) samples, and introduces SuMi, a method with three components: IQR smoothing for gradual sample selection during adaptation, unimodal assistance for selecting low-entropy samples with rich multimodal information, and mutual information sharing for cross-modal alignment. Experiments on Kinetics50-C and VGGSound-C show substantial improvements on strong OOD scenarios where existing methods collapse.

## Strengths
- **Novel and practical problem formulation**: The multimodal wild TTA setting — where target data contains mixed weak and strong OOD samples — is genuinely more challenging and realistic than prior multimodal TTA (e.g., READ), which only addresses weak OOD under a single shift type. Figure 1(c) effectively shows existing methods collapsing from ~65% on weak OOD to ~10% on strong OOD, providing strong motivation.

- **Substantial and consistent improvements on strong OOD**: On Kinetics50-C strong OOD (Table 2), SuMi achieves 33.4% avg vs. next-best READ at 29.1%, while all other methods fall below 15.6%. On the hardest "Mix" scenario, SuMi reaches 18.4% where most baselines are at 0–5%. On VGGSound-C strong OOD (Table 4), SuMi achieves 19.7% vs. READ at 14.5%. These are large, consistent gaps.

- **Unimodal assistance insight is well-motivated**: The counterintuitive finding that very low unimodal entropy indicates samples that do not rely on multimodal fusion (Figure 3c), motivating selection of samples in the [20,40] quantile rather than [0,20], is well-supported by empirical evidence and Table 6's area analysis (Area 1: 39.4% vs. Area 4: 24.3%).

- **Comprehensive evaluation scope**: Two datasets, 21 weak OOD corruption types, 4 strong OOD types, 5 severity levels, 10 mixing ratios (Figure 5), and mixed severity levels (Figure 6) provide thorough coverage of the proposed problem space.

- **Graceful degradation under increasing strong OOD ratios**: Figure 5 shows SuMi degrades gracefully as the proportion of strong OOD samples increases, while all other methods collapse below the source model baseline.

## Weaknesses

### Fatal
None.

### Major

- **Ablation table contains contradictory duplicate rows**: Table 5 contains two rows both marked IQR ✓, UA ✓, MIS ✓ (rows 7 and 8 on Kinetics50-C) yielding substantially different results (54.3 vs. 59.3 at severity 3; 33.4 vs. 38.4 on VGGSound-C severity 3). This makes the ablation unreliable — it is unclear which row represents the actual full method, and the discrepancy leaves component-level claims unverifiable. This must be corrected.

- **Textual claim about IQR's importance is contradicted by the paper's own ablation data**: Section 4.3 states "IQR smoothing brings the most improvements to the model." However, Table 5 shows IQR alone is the worst-performing standalone component (37.1 on Kinetics50-C severity 3) vs. UA (52.1) and MIS (49.4). Adding IQR to UA *decreases* performance (52.1→47.4). IQR+MIS (58.0) does exceed UA+MIS (56.0), suggesting IQR's value lies in combination with MIS rather than being the "most important" component. The paper's characterization is misleading and should be corrected to accurately reflect which components drive performance.

- **No conservative adaptation baseline isolates mechanism contribution from cautiousness**: All compared baselines (Tent, EATA, SAR, etc.) are unimodal TTA methods that catastrophically fail on strong multimodal OOD. A simple conservative baseline — e.g., skipping adaptation for high-entropy batches, or using aggressive entropy-based sample exclusion without IQR/UA/MIS — is needed to determine whether SuMi's gains come from its specific mechanisms or simply from being more conservative about when/how much to adapt. Without this, it is unclear whether the method works because the proposed components are correct or because it avoids destructive updates.

### Minor

- **IQR smoothing lacks theoretical justification for its core mechanism**: The paper claims IQR smoothing preferentially selects weak OOD samples early and strong OOD samples later, but provides only a t-SNE visualization (Figure 3b) as evidence. There is no theoretical reason why the IQR of concatenated feature vectors across a batch should correlate with OOD severity. The mechanism may not generalize across architectures, datasets, or corruption types. A quantitative validation of sample ordering across varying strong OOD ratios would strengthen this claim.

- **IQR combined with UA underperforms UA alone**: Table 5 shows IQR+UA (47.4) performs substantially worse than UA alone (52.1) on Kinetics50-C severity 3. If IQR truly selects higher-quality samples, combining it with UA should help, not hurt. This negative interaction is never discussed or explained, which raises questions about component compatibility.

- **The smoothing function f(t) = t/iter presumes knowledge of total iterations**: In online TTA, the length of the test stream is typically unknown. The paper does not analyze what happens when iter is misestimated, which limits the method's practical applicability.

- **"Mutual information sharing" is a misnomer**: The MIS loss (Equation 6) is a KL divergence between unimodal predictions and a mixture of complementary unimodal and multimodal predictions — i.e., a consistency/distillation objective — not mutual information in the information-theoretic sense I(X;Y) = KL(p(x,y) ∥ p(x)p(y)). The naming could mislead readers.

- **Majority of samples discarded from optimization**: Table 6 shows 78.5% of samples fall in Area 2 (high multimodal entropy, rich unimodal information), but these yield only 27.6% accuracy. The method's UA selects from Area 1 (13.1% of samples), meaning most data is effectively excluded from gradient updates. Whether this limits adaptation signal over many iterations is not discussed.

### Trivial
- None.

## Nice-to-Haves
- Test SuMi on at least one additional multimodal architecture (e.g., a CLIP-based fusion model) to demonstrate generalizability beyond CAV-MAE.
- Validate IQR's sample ordering quantitatively (fraction of weak vs. strong OOD samples at each iteration) rather than relying only on t-SNE.
- Analyze sensitivity to misestimation of iter.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **"Multimodal wild TTA" is overstated as a new task**: The harsh critic claims the framing is just a natural combination of wild TTA (Niu et al. 2023) and multimodal TTA (Yang et al. 2024). While derivative, the specific intersection — mixed weak/strong OOD in multimodal settings — does pose unique challenges (evidence: all existing methods collapse). The formulation adds practical value even if components of the setting exist separately. Downgraded to minor observation.

- **Figure 1 bar charts use approximate values**: The critic notes approximate values ("~45", "~30") in Figure 1. This is a presentation style choice for an overview figure; the exact numbers appear in subsequent tables. Removed as formatting nitpick.

- **Missing modality handling not specified**: The critic notes that how missing modalities are handled (zeroed input, learnable mask token) is not specified. Line 259 states the datasets are described in Appendix A, which was stripped. The original submission likely contains these details. Removed as missing appendix concern.

- **VGGSound-C baselines collapse to near-random**: The critic argues that comparing against collapsed baselines is uninformative. However, the collapse itself is part of the problem statement, and READ (14.5%) is a non-collapsed baseline showing meaningful competition. This is a feature of the evaluation, not a bug.

- **Weak OOD improvements are small**: The critic notes ~1% absolute improvements on VGGSound-C weak OOD. But the paper's focus is on strong OOD; weak OOD parity with existing methods is expected and acceptable.

## Novel Insights
The ablation data reveals an unexpected negative interaction between IQR smoothing and unimodal assistance (IQR+UA underperforms UA alone), suggesting these components target conflicting subsets of samples — IQR selects "safe" samples based on feature-space proximity while UA selects "informative" samples based on multimodal entropy. The fact that MIS rescues this interaction (IQR+UA+MIS > UA+MIS) hints that the mutual information sharing loss acts as a regularizer that coordinates the competing selection mechanisms, a dynamic the paper does not acknowledge.

## Suggestions
- Fix Table 5's duplicate ✓✓✓ rows by clarifying which configuration each represents (perhaps one is with MIS loss applied for all iterations vs. only the first t₀ iterations), and re-run if needed.
- Revise the claim "IQR smoothing brings the most improvements" to accurately reflect that MIS provides the largest marginal improvement when added to existing components, while IQR's value is primarily realized in combination with MIS.
- Add a conservative adaptation baseline (e.g., entropy-threshold-based adaptation skipping) to isolate mechanism contribution from cautiousness.

## Evaluation Axis Assessment
- **Originality**: Moderate. The problem formulation extends existing work (wild TTA + multimodal TTA), and the IQR smoothing mechanism is novel though heuristic. UA is a sensible but straightforward sample selection strategy. MIS is a standard consistency loss.
- **Importance of research question**: High. Multimodal TTA under mixed strong/weak OOD is a genuinely practical and underexplored setting.
- **Claims well supported**: Partially. Empirical gains are large and consistent, but the text mischaracterizes which components drive performance, and the ablation table contains inconsistencies.
- **Soundness of experiments**: Good scope but compromised by the Table 5 inconsistency and missing conservative baseline.
- **Clarity of writing**: Adequate; the core ideas are understandable but some claims are loosely stated (e.g., "IQR brings most improvements," "mutual information sharing").
- **Value to community**: The problem setting and benchmarks have lasting value; the method's value is somewhat clouded by the ablation issues.

## Calibration Comparison
- **High anchor**: TPZRq4FALB (READ, avg 8.0, Accept poster) — Same domain (multimodal TTA), strong baselines, clean ablations, new benchmarks. SuMi is less clean methodologically (conflicting ablation, overclaimed component importance) but targets a harder problem setting.
- **Medium anchors**: sEMJ1PLSZR (AEA, avg 6.25, Accept poster) — TTA with strong empirical results but mechanism clarity issues. UhKkWHkvfg (MDAA, avg 5.0, Reject) — Multimodal TTA with incomplete ablations and methodological inconsistencies. SuMi is stronger than MDAA in empirical gains but has similar ablation issues; somewhat weaker than AEA which had cleaner mechanism validation.
- **Low anchor**: GxmltrqVNn (GABins, avg 2.5, Reject) — Overclaimed contribution, no ablation. SuMi is far above this with real empirical gains and actual ablation data (despite inconsistencies).

SuMi sits between the medium anchors — it has stronger empirical results than the rejected multimodal TTA papers but its ablation table inconsistency and overclaimed component importance are substantial issues that a reviewer would weigh against acceptance. The paper's gains are real and substantial, the problem formulation is valuable, but the inconsistency in Table 5 and the characterization of component importance need correction.

## Score and Decision

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>