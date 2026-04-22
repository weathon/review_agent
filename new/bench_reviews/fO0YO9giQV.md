Now I have all the information I need. Let me write the final review.

## Summary

AnyECG introduces a family of ECG foundation models (254M–1.7B parameters) that use a two-stage self-supervised pre-training pipeline: first, an ECG Tokenizer with a vector-quantized Rhythm Codebook and a Multi-View Synergistic Decoder (reconstructing morphology, frequency, and demography), then masked code-index prediction over the discretized tokens. A Cross-Mask Attention (CMA) mechanism restricts patch interactions to same-lead and same-position patches. The model is evaluated on anomaly detection, arrhythmia detection, corrupted lead generation, and ultra-long ECG recognition across seven datasets.

## Strengths

- **Principled domain-adapted architecture**: The Rhythm Codebook (Eq. 2) using cosine-similarity-based VQ to discretize ECG patches into noise-resilient tokens, combined with the Multi-View Synergistic Decoder (morphology/frequency/demography, Eqs. 3–6), is a well-motivated adaptation of the BEiT paradigm to ECG. The demography decoder directly addresses demographic shift (Challenge 3), which most prior ECG SSL methods ignore.

- **Cross-Mask Attention as a physiologically-informed inductive bias**: CMA (Eq. 1) restricts patch interactions to same-lead or same-position patches with positional tolerance for conduction delays — a domain-appropriate design that exploits known multi-lead ECG physiology rather than applying vanilla self-attention.

- **Consistent outperformance of ECG-FM**: AnyECG-XL outperforms the prior ECG foundation model ECG-FM across all shared tasks: anomaly detection (0.8255 vs 0.7788 accuracy, Table 2), arrhythmia detection (0.3449 vs 0.2212 accuracy, Table 3), and additionally handles two tasks (corrupted lead generation, ultra-long ECG recognition) that ECG-FM cannot perform due to architectural limitations (Tables 4–5), demonstrating genuine functional advantages from the tokenizer design.

- **Functional capabilities beyond classification**: The ability to perform corrupted lead generation (Table 4, Figure 2) and hierarchical ultra-long ECG recognition (Table 5) are genuine capabilities enabled by the tokenizer+masked modeling design that prior ECG SSL methods lack.

- **Scaling behavior and reporting rigor**: Results across three model sizes (Tables 2–5) generally show improvements with scale, and all results are reported with standard deviations across five random seeds.

## Weaknesses

### Fatal

None.

### Major

- **Potential data leakage between pre-training and evaluation**: Section 3.1 states "we utilized a comprehensive set of ECG datasets that include all available unlabeled data during pretraining," and Section 3.2 describes an 80/20 downstream split. The paper never explicitly states that the 20% test/validation recordings were excluded from the self-supervised pre-training corpus. If the same ECG recordings appear in both pre-training and test, the tokenizer and masked modeling objectives would have seen and encoded these exact signals, inflating all downstream metrics. The paper says "final evaluations were conducted on separate test sets" (Section 3.2) but the relationship between these test sets and the pre-training data is never clarified. This ambiguity threatens every generalization claim and must be addressed. If pre-training did cover test data, the results are not indicative of true generalization.

- **Systematically asymmetric baseline comparison**: All non-AnyECG baselines in Tables 2, 3, and 5 are trained from scratch (Pretrain = ×), while AnyECG benefits from large-scale self-supervised pre-training on all seven datasets. The single other pre-trained model, ECG-FM, was pre-trained on different data and performs poorly, but this could reflect data mismatch rather than architectural inferiority. The paper cannot attribute its improvements to AnyECG's architectural contributions (CMA, codebook, multi-view decoder) without controlling for the pre-training data advantage. Pre-training at least one baseline architecture (e.g., XResNet1D or Inception1D) on the same data with a standard SSL protocol, or evaluating AnyECG without pre-training, would disentangle the data advantage from the architecture advantage. Without this, the results are consistent with the well-known finding that pre-training on more data helps.

- **Arrhythmia detection accuracy is low and uncontextualized**: AnyECG-XL achieves only 34.5% accuracy on arrhythmia detection (Table 3). The paper frames this as "strong ability to handle arrhythmia detection effectively" but never reports the number of classes, class distribution, or a random-chance baseline. Given all models struggle here (ECG-FM: 22%), these low numbers suggest a very challenging multi-class problem, but the reader cannot interpret whether 34.5% is meaningful without this context. The claim of "strong" performance is unsupported as written.

### Minor

- **ECG-FM excluded from two of four tasks**: ECG-FM is omitted from corrupted lead generation (Table 4) and ultra-long ECG recognition (Table 5) due to architectural limitations. While this may be legitimate (the model physically cannot handle these tasks), it means the paper's most directly comparable baseline is missing from half the evaluation, and the narrative of "outperforming all SOTA" is inflated.

- **No ablation for CMA vs standard self-attention**: CMA is presented as a key differentiating contribution (Eq. 1), but the paper does not provide a comparison replacing CMA with standard self-attention. The ablation studies mentioned (Section 7.4/7.3) are in the appendix and appear to cover hyperparameters and two-stage necessity, not the attention mechanism itself. Without this, the contribution of CMA remains conjectural.

- **Undisclosed dataset**: Table 1 lists an "Undisclosed Dataset" of 10,000 recordings (roughly 20% of total pre-training data) with no information about provenance, collection protocol, or contents. The note says "geographically distinct test set" but this raises more questions than it answers — why is a test set included in pre-training? This undermines reproducibility and verifiability.

- **Positional tolerance not parameterized or studied**: The CMA mask includes a "positional tolerance (mask width)" to account for conduction delays (Section 2.1, Eq. 1), but this parameter is never specified, studied, or ablated.

### Trivial

- Table 5 row 238 has no method name — a formatting issue in the table.

## Nice-to-Haves

- Cross-dataset generalization test: evaluating AnyECG on a dataset entirely excluded from pre-training would be a much stronger test of generalization than the current mixed-dataset evaluation.
- Codebook visualization showing what learned rhythm codes correspond to qualitatively (nearest-neighbor ECG patches for selected codes) to substantiate the claim that codes are "clinically meaningful."
- Analysis of information loss through the codebook bottleneck, especially for subtle pathological waveforms.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Demographic bias risk from demography decoder**: The harsh critic raises that the demography decoder may encode demographic shortcuts into rhythm codes, propagating bias. While this is a valid fairness consideration, it is speculative — the paper does not show evidence of such bias, and the decoder is designed to make demographics explicit rather than implicit. This is a nice-to-have analysis, not a current flaw in the paper.

- **Claim that quantization "effectively mitigates signal noise" is unsupported**: The harsh critic argues this is asserted without evidence. However, the paper provides empirical evidence that AnyECG outperforms baselines that reconstruct raw signals. The claim is implicitly supported by the results, even if a direct quantization information-loss analysis would strengthen it. This is a minor point already captured above.

- **Missing related works**: Per the hard rules, I do not mention missing related works as I cannot independently verify their existence.

- **Model scaling details not provided**: The harsh critic notes no details on how architecture scales. The paper states "the increase in parameters is achieved by deepening the Transformer encoder and expanding the hidden layer sizes" (Section 3.2). While more detail would help, this is standard for conference submissions and not a substantive weakness.

- **Reproducibility concerns about undisclosed hyperparameters/large artifacts**: Removed per hard rules on reproducibility nitpicks.

- **Formatting/style nitpicks**: Removed per hard rules.

- **Typos and parser artifacts**: Removed per hard rules.

- **Missing appendix proofs/references**: Removed per hard rules (parser strips appendices).

- **Concern about ECG-FM's pre-training data being different**: While this is partially valid (captured in the major weakness about asymmetric comparison), the specific framing that ECG-FM should be retrained on the same data is unreasonable — ECG-FM is a published model with its own pre-training corpus, and asking authors to retrain someone else's model is not standard practice. The fairer request is to pre-train one of the from-scratch baselines on the same data.

## Novel Insights

The paper's Multi-View Synergistic Decoder that jointly reconstructs morphology, frequency, and demography is an underappreciated strength: by forcing the tokenizer to encode demographics explicitly, it makes the model's reliance on demographic features transparent and controllable, whereas most ECG SSL methods let the model learn implicit demographic shortcuts that are invisible and unaccountable. This design choice could set a precedent for fairness-aware pre-training in medical signal foundation models, though the paper itself does not frame it this way.

## Suggestions

- Explicitly document the data splitting protocol: whether the 20% held-out evaluation data was excluded from self-supervised pre-training. If it was not, re-run experiments with proper separation and report the results.
- Add at least one from-scratch baseline pre-trained on the same data using a standard SSL method (e.g., masked autoencoding of raw signals) to isolate the contribution of AnyECG's architecture from the pre-training data advantage.
- Report the number of classes and random-chance baseline for arrhythmia detection so readers can interpret the 34.5% accuracy.
- Add an ablation replacing CMA with standard self-attention to validate this claimed contribution.

## Score and Decision

### Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| LaBraM (EEG foundation model) | QzTpTRVtrP.md | 7.33 | Similar VQ tokenizer + masked modeling paradigm for bio-signals. LaBraM was accepted at spotlight with better-controlled evaluation and ablations but similar novelty level. AnyECG is weaker due to data leakage ambiguity and baseline unfairness. |
| NeuroLM (EEG multi-task FM) | Io9yFt7XH7.md | 6.25 | Also 1.7B parameter bio-signal foundation model with VQ tokenizer. Accepted poster despite concerns about alignment effectiveness and theoretical grounding. AnyECG has more severe evaluation issues (data leakage, baseline fairness). |
| PaPaGei (PPG foundation model) | kYwTmlq6Vn.md | 6.25 | Physiological signal foundation model with open code and strong reproducibility. Accepted poster. Stronger evaluation rigor than AnyECG but similar domain. |
| TA-PCLR (ECG FM) | 7zJDTnogdG.md | 3.33 | ECG foundation model rejected partly for unfair pre-training comparisons and limited baseline evaluation — the same issues AnyECG has. |
| CuPID (ECG SSL) | QjrC77Nyu6.md | 2.50 | ECG SSL method rejected for unfair baseline configuration and missing details. |
| Grammar induction unfair comparison | 63r6HyqyRm.md | 2.33 | Rejected for unfair pre-trained vs from-scratch comparison — the exact issue AnyECG has. |
| Data contamination in LLMs | lwtaEhDx9x.md | 4.75 | Rejected despite interesting findings, partly due to data contamination evaluation concerns. |

AnyECG has meaningful architectural contributions (VQ codebook, multi-view decoder, CMA) and real functional advantages over ECG-FM, but the evaluation is weakened by (1) unclear data separation between pre-training and test sets, and (2) no control for the pre-training data advantage in baseline comparisons. These are not fatal (the model does genuinely outperform ECG-FM, which is also pre-trained), but they significantly undermine the claims about architectural contributions specifically. The paper falls below LaBraM (7.33) and PaPaGei (6.25) due to these evaluation gaps, but above TA-PCLR (3.33) and CuPID (2.5) because it has a more complete system and genuine functional capabilities. It is closest to the borderline 4–5 range, but its architectural novelty and capability to handle tasks that prior models cannot push it slightly above that band.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>