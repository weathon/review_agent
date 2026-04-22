Now I have sufficient calibration data. Let me write the final review.

## Summary

TrajGPT proposes a Selective Recurrent Attention (SRA) mechanism with data-dependent decay for irregular time-series representation learning on healthcare data. By interpreting SRA as a discretized ODE, TrajGPT enables time-specific inference that directly predicts at arbitrary target timesteps, and demonstrates strong empirical results across diagnosis forecasting, drug usage prediction, and phenotype classification on two clinical datasets (PopHR: 489K patients; eICU: 139K patients).

## Strengths

- **Principled time-specific inference mechanism**: The approach of evolving learned ODE dynamics to predict at arbitrary timesteps via S_{n'} = D_Δ S_n is a practical and well-motivated mechanism. The ablation (Table 3) consistently shows time-specific inference improving over auto-regressive inference by 4.6–6.2 percentage points in top-K recall, which is a meaningful and reliable gain. This is the paper's most substantiated contribution.

- **Comprehensive empirical evaluation**: The paper evaluates across two large-scale datasets, four task types (forecasting, drug prediction, phenotype classification, sepsis detection), and 17+ baselines including both regularly-sampled Transformers and irregular-time-series-specific models (Tables 1–2). The breadth is commendable and the experiments demonstrate TrajGPT's strong performance, particularly in forecasting and zero-shot classification.

- **Interpretable trajectory visualizations**: Figure 4 provides clinically meaningful disease risk trajectory analyses that demonstrate how TrajGPT interpolates and extrapolates risk, connecting predicted trajectories to real comorbidity patterns (e.g., chronic IHD, hypothyroidism, and obesity preceding diabetes onset).

- **Efficient dual-mode formulation**: The parallel (Eq. 3) and recurrent (Eq. 2) forms of SRA enable O(N) training and O(1) inference, a practical engineering advantage over standard Transformers and ODE-solver-based models like ContiFormer.

## Weaknesses

### Fatal
None.

### Major

- **The central architectural contribution—data-dependent decay—has marginal empirical impact**: The paper's first listed contribution is SRA with data-dependent decay, described as enabling "selective forgetting" analogous to clinical reasoning. However, the ablation in Table 3 shows that removing data-dependent decay (replacing γ_n with a fixed value) reduces top-10 recall from 71.7% to only 70.3%—a gain of 1.4 percentage points. Meanwhile, removing RoPE causes a larger drop (to 67.8%), and replacing linear attention with GPT-2 drops to 61.2%. The data-dependent decay, which is the paper's headline innovation, is the smallest contributor among all components. This significantly undermines the framing of SRA as a mechanism that "adaptively forgets irrelevant past information based on contexts" and weakens the novelty claim relative to the more impactful RoPE and linear attention components.

- **Time-specific inference formulation has a logical gap**: Section 3.2 states that forecasting a target point (x_{n'}, t_{n'}) uses S_{n'} = D_{Δ} S_n + K_{n'}^⊤ V_n. However, K_{n'} = X_{n'} W_K e^{-iθt_{n'}} (Eq. 1) requires X_{n'}, the very observation being predicted, creating a circular dependency. The paper never resolves this: either K_{n'} should be omitted (using only the evolved state D_{Δ} S_n), or K_{n'} and V_{n'} should be derived from the last observed input, but this must be explicitly stated. As written, the key mechanism for time-specific inference—the paper's second contribution—cannot operate as described, though we note the empirical results suggest the implementation likely handles this correctly.

- **Asymmetric comparisons in zero-shot and fine-tuning settings**: In Table 1, irregular-time-series baselines (mTAND, GRU-D, RAINDROP, SeFT, ODE-RNN, HeTVAE, MGP-TCN) have "—" entries for zero-shot and few-shot columns because they lack pre-training paradigms. This makes TrajGPT's zero-shot numbers appear as the best by default, but they reflect an inherent capability asymmetry rather than a contestable methodological comparison. Meanwhile, in the fine-tuning column where baselines are also fully trained, TrajGPT does not consistently win: mTAND achieves 85.4% on CHF (vs. TrajGPT's 83.9%), and BiTimelyGPT achieves 75.8% on insulin (vs. 75.5%). The paper's headline framing of "strong zero-shot performance" is valid but could be perceived as overstated given the asymmetric evaluation setup.

### Minor

- **Zero-shot classification procedure is underspecified**: The 67.2% and 72.8% AUPRC numbers for zero-shot classification on insulin and CHF respectively are never explicitly mapped to a classification procedure. Section 5.1 and Figure 3b suggest sequence representations naturally separate by class, but the paper does not specify whether a nearest-neighbor classifier, linear probe, or next-token probability threshold is used to produce these numbers. This is important for reproducibility.

- **The ODE connection is presented with excessive novelty claims**: The ZOH discretization in Eq. 5 connecting SRA to an ODE is standard SSM theory (as used in S4, Mamba). The paper acknowledges this derivation comes from Gu et al. (2022) but frames the connection as a novel contribution. The actual novelty is in applying this discretization to enable time-specific inference, not in establishing the ODE connection itself.

- **Trajectory analysis is anecdotal**: Section 5.3 provides case studies on two cherry-picked patients. While clinically meaningful, these lack quantitative validation (e.g., correlation between predicted and actual risk across a held-out cohort), making claims about "forecasting unseen diseases" somewhat overextended based on two examples.

- **Missing citation of RetNet**: The SRA formulation (Eq. 2–4) closely mirrors RetNet's retention mechanism with input-dependent decay substituted for learned decay. The paper cites linear attention (Katharopoulos et al., 2020) but not RetNet (Sun et al., 2023), which would more precisely position the contribution.

### Trivial
None.

## Nice-to-Haves

- An ablation on the classification tasks (not just forecasting) showing the effect of data-dependent decay and RoPE on AUPRC for insulin/CHF would strengthen claims about the mechanism's generality.
- Quantitative cluster evaluation (e.g., silhouette score) for the zero-shot classification representations in Figure 3b.
- Analysis of the learned γ_n distributions: Are they nearly uniform, or do they meaningfully vary by patient/condition? This would clarify whether data-dependent decay is learning meaningful specialization.
- Comparison with mTAND or GRU-D also pre-trained with next-token prediction on the same data to create a fairer zero-shot baseline.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Harsh critic Claims all baseline comparisons are unfair**: Including regular-time-series Transformers (Informer, etc.) is standard practice and provides useful reference points. Their poor performance is informative, not misleading. The paper includes them as common baselines across all time-series work.

- **Harsh critic Claims "zero-shot forecasting is a mischaracterization"**: The paper explicitly states "Since next-token prediction is inherently forecasting, TrajGPT enables zero-shot forecasting without requiring fine-tuning" (Section 4.2). This is transparent about the relationship. Zero-shot in the context of pre-trained models typically means "without task-specific fine-tuning," which is accurate here. For classification tasks, zero-shot IS genuine task transfer.

- **Harsh critic Claims SRA is essentially RetNet without citation**: While the similarity is notable and RetNet should be discussed, the data-dependent decay mechanism IS a meaningful modification (even if its empirical impact is small). Calling it "Replicating RetNet" overstates the overlap—SRA has different design motivations and context.

- **Harsh critic Claims ODE interpretation is "standard SSM theory repackaged"**: While the ZOH connection is standard, the application to time-specific inference for irregular time steps is novel and empirically validated. The contribution is in the application, not the theory itself.

- **Harsh critic's "BitMimicGPT" typo claim**: This is a parser/formatting artifact from the original paper, not a substantive issue.

- **Harsh critic's claim that trajectory analysis is "not validated against ground truth"**: This is partially valid but overstated—the trajectory analysis is explicitly described as qualitative/case-study analysis, and the forecasting results provide the quantitative validation.

- **Strength Finder's claim that "zero-shot transfer across multiple clinical tasks" is a core strength**: This conflates forecasting (which is same-task transfer) with classification. The zero-shot classification results are genuinely interesting, but the "strong zero-shot" framing includes the forecasting task which is same-task evaluation. Partially removed—classification zero-shot is kept as a strength, forecasting zero-shot is weakened.

## Novel Insights

The most insightful observation from reviewing the evidence is that the paper's true contribution appears to be the engineering combination of (1) data-independent mechanisms (RoPE, linear attention) that handle irregular time intervals, (2) a pre-training paradigm adapted to clinical discrete code sequences, and (3) the practical time-specific inference trick—rather than the individually claimed novel mechanisms (data-dependent decay, ODE interpretation). The 1.4pp ablation impact of decay gating, combined with the larger impact of RoPE (2.5pp) and replacing the entire architecture with GPT-2 (10.5pp), suggests the architecture's overall design matters more than any single component. Time-specific inference (4.6–6.2pp gain over auto-regressive) is the real workhorse of the empirical gains, and it relies on the ODE framework's ability to evolve hidden states through arbitrary time deltas—a practical insight that stands independent of the data-dependent decay claim.

## Suggestions

- Rename or reframe the "zero-shot forecasting" to "direct forecasting" or "task-free forecasting" to avoid the implication of cross-task transfer in the forecasting setting.
- Explicitly resolve the K_{n'} dependency in the time-specific inference formulation—either state that K_{n'} uses the prior token's embedding, that only D_{Δ} S_n is used, or clarify the implementation.
- Reposition data-dependent decay as a regularizer/fine-tuning contribution rather than the primary architectural contribution, given its small ablation impact. Emphasize time-specific inference and the pre-training + SRA combination as the main contributions.
- Add RetNet to the related work and explicitly discuss how SRA differs from retention mechanisms.

## Score and Decision

**Calibration anchors**:
- TimelyGPT (similar topic, avg 5.5, Reject): Similar architecture for irregular time series, also built on retention-like mechanisms. TrajGPT has more comprehensive experiments and the time-specific inference contribution, but shares the overclaimed novelty concern.
- XTSFormer (similar topic, avg 5.0, Reject): Irregular time-event prediction Transformer with modest novelty gains.
- TimeDiT (time-series foundation model, avg 5.25, Reject): General time-series foundation model with zero-shot claims and some overclaimed novelty.
- ACSSM (irregular time series, avg 8.0, Accept Oral): Rigorous theoretical grounding with strong empirical results and clear novelty in the SDE formulation.
- "Old dog" paper (overclaimed novelty, marginal gains, avg 5.0, Reject): Combining existing techniques with overclaimed novelty.

TrajGPT sits in a similar space to TimelyGPT/XTSFormer—solid empirical results on an important problem, with a few genuine contributions (time-specific inference, comprehensive evaluation, trajectory visualization) undermined by overclaiming of the data-dependent decay mechanism and a logical gap in the time-specific inference formulation. The paper is above the lowest-quality anchors (which had fundamental methodology issues) but clearly below the high-quality anchors (which had novel theoretical contributions). Its positioning is in the borderline zone alongside TimelyGPT (5.5) and TimeDiT (5.25), with slightly more empirical breadth but similar novelty concerns.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>