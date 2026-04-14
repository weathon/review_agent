## Summary
Self-Evolved Reward Learning (SER) proposes an iterative self-training framework for reward models (RMs) that reduces human annotation requirements by having the RM pseudo-label its own unlabeled data. The key mechanism is a curriculum-based two-status data filtering strategy: Status 1 selects pairs where the RM confidently distinguishes clearly good from clearly bad answers, and Status 2 (activated when Status 1 is exhausted) selects pairs where the RM can discern subtle quality differences. Experiments on HH-RLHF, UltraFeedback, StackOverflow, and Summarize across Llama 2/3 and Mistral models show that, starting from a seed RM trained on 15% of labeled data, SER achieves performance close to or exceeding full-dataset training, with an average 7.88% gain over the seed model.

---

## Strengths

- **Two-status curriculum filtering is a specific and well-motivated mechanism**: Rather than applying a single confidence threshold uniformly (vanilla pseudo-labeling), SER distinguishes between an "easy" phase (global discrimination of good vs. bad, Status 1) and a "hard" phase (amplifying differences between similar-quality answers, Status 2). Figure 2 and Figure 3 provide concrete evidence that this transition prevents performance plateauing or regression that occurs when only easy samples are used: Loop 2 (Status 1 with easy data) shows diminishing returns for some models, while Loop 3 (Status 2 with harder data) recovers performance gains. This curriculum structure is not the default in standard self-training and represents a genuine algorithmic contribution.

- **Consistent empirical results across a broad and diverse experimental grid**: The method is evaluated across 4 datasets spanning quite different domains (helpfulness/harmlessness, code Q&A, summarization, instruction-following), 4 model families and sizes (Mistral 7B, Llama 3 8B, Llama 2 13B, Llama 3 70B), yielding consistent and non-trivial improvements over the seed model in every configuration. Table 1 shows average gains of 7.88% over the seed and an average gap to the full-data baseline of only 0.3%, which is a meaningful empirical signal.

- **Downstream PPO validation**: Unlike most reward model papers that stop at RM accuracy, SER validates the evolved RM by using it to guide PPO training and evaluates the resulting LLMs via GPT-4 judgments. SER-trained models consistently outperform SFT baselines and are competitive with (or superior to) PPO trained with full-data RMs (Figure 4), closing the loop on practical utility.

---

## Weaknesses

- **Absence of semi-supervised / vanilla pseudo-labeling baselines — the most critical gap**: SER's core claim is that its *specific* status-based curriculum mechanism provides gains over simply training on 15% of labeled data. However, Table 1 contains no comparison against vanilla pseudo-labeling (e.g., confidence-thresholded self-training without curriculum), mean-teacher, or other standard semi-supervised approaches applied to RM training. Without such baselines, it is impossible to determine whether the observed gains arise from the curriculum structure or from the simpler effect of adding any self-labeled data. This fundamentally limits the attributable novelty of the contribution.

- **No statistical significance reported for key comparisons**: No standard deviations, confidence intervals, or significance tests appear anywhere in Table 1 or Figure 4. Several headline results rest on very small margins — e.g., Mistral 7B surpassing the Full Dataset by 0.13% on HH-RLHF, or SER vs. Full differences of roughly 1% in several cells. With single-run experiments, it is unknown whether these differences exceed noise, which is especially problematic for the claim that SER "can approach or even exceed" full human-labeled data performance.

- **Circular label assignment in Status 2**: In Status 2, pairs are selected where |RM(Q, A¹) - RM(Q, A²)| > δ, and the *sign* of the RM's score difference implicitly determines which answer is treated as preferred in the pairwise loss. The RM thus serves simultaneously as the filter and the label oracle. While pairwise loss provides some robustness and the confidence threshold filters for higher-certainty predictions, this circularity can amplify systematic errors in the RM: if the RM has a consistent bias (e.g., preferring verbosity), Status 2 will reinforce that bias rather than correct it. This is a well-known risk of self-training and is not discussed or experimentally probed in the paper.

- **Source of "unlabeled data" is never made explicit**: The paper never clearly states whether the unlabeled data is the withheld 85% of preference *pairs* (i.e., questions + answer pairs already exist, only labels are withheld) from the same dataset, or genuinely new Q/A pairs from an external source. This distinction is critical: if the unlabeled pool is simply the labeled dataset with labels removed, SER is within-distribution semi-supervised learning over a fixed corpus, and its annotation cost reduction refers only to *labeling* cost, not *data collection* cost. The abstract's claim that "only 15% of human-annotated seed data is required" should be qualified to reflect that all Q/A pairs are still required from the original dataset.

- **Threshold sensitivity is unanalyzed**: The method relies on three non-trivial thresholds (τ_high=0.55, τ_low=0.45, τ_Δ=0.3) and a "sufficient count" of 600, justified only by stating they "provided the most consistent improvements." No ablation over these values is presented. Given that these thresholds determine which data is selected, the proportion of Status 1 vs. Status 2 transitions, and ultimately when training stops, their robustness is central to the method's generalizability. Small perturbations could shift the data selection dramatically.

- **PPO evaluation is statistically fragile**: Figure 4 shows high tie rates (42–71%) and small win-rate differences (e.g., SER: 22% vs. Full: 24% wins for Llama 8B on HH-RLHF). No statistical testing is applied. The GPT-4-as-judge protocol, while standard, introduces position and length biases; no consistency or calibration checks are reported. Claims such as "SER models outperform the full models to a certain extent" are difficult to substantiate given this noise.

- **Theoretical section overpromises**: Section 3.2 describes theoretical results on RM convergence and PPO optimality, but all proofs are deferred to Appendix A (removed from the submitted text). The only substantive claim in the main body — "when initial accuracy exceeds 50%, iterative training with high-confidence samples can further improve performance" — is a mild result that follows naturally from basic self-training arguments. Critically, the Discussion section (Section 5) itself concedes "a rigorous theoretical analysis of its effectiveness is still needed," which undercuts Section 3.2's framing as a theoretical contribution.

- **Data accumulation across iterations may create labeling conflicts**: The paper defines D_filtered = D_filtered^n + D_filtered^{n-1} — data from previous loops is retained and mixed with new data. Because the RM's scores change between iterations, the same pair could receive different relative orderings in different loops. No reconciliation strategy is described or evaluated, and this could introduce noisy or contradictory training signal.

---

## Nice-to-Haves

- **Ablation on thresholds**: Vary τ_high, τ_low, τ_Δ, and the sufficient-count criterion to show the method degrades gracefully; this would significantly strengthen robustness claims.

- **Direct RLAIF comparison**: Including a baseline where a same-sized or slightly larger model annotates the unlabeled pairs (standard RLAIF) would sharpen the contribution, as the motivation explicitly positions SER against RLAIF.

- **Reward score histogram analysis per iteration**: Visualizing reward score distributions across loops would reveal whether the RM is maintaining discriminative power or shifting scores in degenerate ways (reward hacking), addressing the bias amplification concern directly.

- **Self-label noise quantification**: Measuring the error rate of RM-generated pseudo-labels against ground-truth labels per iteration (on a held-out calibration set) would concretely show how noise evolves and validate the theoretical claim that noise stays within tolerable bounds.

- **Status transition breakdown visualization**: Explicitly plotting the proportion of data selected under Status 1 vs. Status 2 across iterations per dataset/model would verify that the curriculum progresses as intended and is not trivially collapsing to one regime.

- **Compute/wall-clock comparison**: Reporting total training time or FLOPs relative to full supervised training would give a complete picture of the efficiency gains, since multiple iterative loops add overhead.

- **Out-of-distribution generalization test**: Evaluating the SER-trained RM on a held-out preference dataset not used in training would assess whether the evolved RM generalizes or overfits to its self-labeled distribution.

- **Discussion of minimum viable seed quality**: The Discussion briefly mentions future failure modes; explicitly characterizing how weak the seed RM can be (e.g., what happens if the initial accuracy is near 50%) before SER collapses would be practical guidance for users.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Critic's claim of "logically inverted" status ordering**: The critic argues that "If the RM does not meet the criteria for Status 1 (easier task), we then check for Status 2" is contradictory. However, the paper's logic is that Status 1 activation means *enough clearly easy pairs* remain in the unlabeled pool for productive training; when the RM has already absorbed those (fewer clear good/bad pairs survive the threshold), Status 1 is no longer satisfiable, and the harder Status 2 takes over. This is a coherent curriculum progression, not a logical inversion. The paper explicitly states "Status 1 is the easier task...Status 2 is the harder task...If the RM does not meet the criteria for Status 1 (i.e., few or no samples satisfy the thresholds), we then check for Status 2." This is internally consistent.

- **Critic's claim about "missing" Figure 6**: The critic notes Figure 6 (RM score distributions) appears to be referenced but missing. This is likely an artifact of the review copy with appendix removed; the main text describes these results and they do not appear to be fabricated.

- **Strength: "The paper is well-written / well-structured"** (Reviewer 2): Generic and applies to any adequately formatted paper; removed.

- **Criticism that SER requires "stronger LLMs" in the same way as RLAIF**: The critic argues the self-labeling requirement (seed RM must have >50% accuracy) is "parallel" to RLAIF's need for a strong AI annotator. Technically true, but the bar for SER (any RM trained on 15% data, no external model required) is materially lower than typical RLAIF setups requiring GPT-4-class models. The distinction is worth noting but does not undermine the motivation.

- **Criticism about the title being "slightly misleading"**: Stylistic preference; removed.

---

## Novel Insights

The most genuinely novel conceptual insight in the paper — and one worth emphasizing — is the *adaptive curriculum in data selection driven by the model's own evolving confidence profile*. Rather than applying a static filtering rule, SER monitors the RM's current probability outputs on the unlabeled pool to determine *which skill the model is ready to develop next*. The empirical evidence in Figure 2 and 3 substantiates that Status 2 data (subtle differences) is not useful early (when the model cannot reliably rank them) but becomes critical later to push past the performance ceiling reached with easy data. This is a meaningful operationalization of curriculum learning in the reward modeling context, and the observation that "similar data becomes marginally helpful or harmful after multiple iterations" (Loop 2 showing weakest gains) is an empirically grounded and underexplored phenomenon in RM training. If supported by ablations against vanilla pseudo-labeling, this mechanism would constitute a concrete design principle for data-efficient reward learning.

---

## Suggestions

1. **Add vanilla pseudo-labeling as a baseline**: Train an RM with 15% labeled data and apply simple confidence-thresholded pseudo-labeling (without status-based curriculum) for the same number of iterations. This is the most critical experiment needed to validate the specific contribution of the two-status mechanism.

2. **Explicitly define the unlabeled data pool in the main text**: State clearly whether unlabeled data consists of held-out preference *pairs* from the same dataset (labels removed) or truly new Q/A instances. Adjust the annotation-cost-reduction claims accordingly.

3. **Report variance or repeat key experiments with multiple seeds**: At minimum, for the headline comparisons in Table 1 (SER vs. Full Dataset), report results over 2–3 runs to establish whether differences are meaningful or within noise.

4. **Add an ablation table for thresholds**: Fix two thresholds and vary the third across a grid; show resulting RM accuracy to quantify sensitivity. Even a 3×3 grid in an appendix would substantially address this concern.

5. **Address bias amplification in Status 2 empirically**: Track the correlation between SER pseudo-labels and ground-truth labels across iterations. If the circular label assignment is not causing error amplification, show it; if it is, discuss mitigation strategies (e.g., label smoothing, ensemble filtering).

6. **Strengthen the theoretical section or reclassify it**: Either provide the key assumptions and sketch of proofs in the main body (moving material from Appendix A), or reframe Section 3.2 as "theoretical motivation" rather than "theoretical analysis," consistent with the Discussion's honest admission that rigorous theory is still needed.