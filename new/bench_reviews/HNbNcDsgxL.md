Now I have sufficient information to write the final review. Let me compose it.

---

## Summary

Delta is a training-free, inference-time method for mitigating hallucinations in LLMs. It randomly masks input tokens, runs a second forward pass through the LLM on the masked input (which amplifies prior-driven hallucinations), and subtracts the resulting logits from the original logits via a contrastive decoding formula. The idea adapts Visual Contrastive Decoding (VCD) from vision-language models to the text domain, replacing Gaussian noise with random token masking. Delta is evaluated on SQuAD v1.1/v2, TriviaQA, Natural Questions, CommonsenseQA, and MMLU using a single Llama 3.1 8B Instruct model.

---

## Strengths

- **Practical, deployment-friendly design**: Delta requires no retraining, no external models, and no additional data, making it immediately applicable to existing LLMs.
- **Strong SQuAD v2 no-answer results**: The 14.53 and 11.81 percentage-point improvements on NoAns_EM (with and without sampling, respectively) are substantial and directly relevant to the hallucination problem — specifically, the model better withholds answers when context is insufficient.
- **Honest limitation reporting**: The paper explicitly acknowledges near-zero gains on CommonsenseQA and MMLU and provides a coherent explanation: without grounding context to mask, there is no hallucination-amplifying signal to contrast against.
- **Ablation shows hyperparameter robustness**: All parameter configurations in the heatmap (SQuAD v1.1) exceed the baseline, with low standard deviations (0.66 EM / 0.21 F1), indicating the method is not fragile to exact hyperparameter tuning.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing comparison to Context-Aware Decoding (CAD)**: The paper explicitly acknowledges in Section 2 that CAD "demonstrated a similar outcome to our Delta method by adjusting the output probabilities of LMs, amplifying the differences between outputs generated with and without the given context." The paper's only claimed differentiator is generalizability, but by its own evidence Delta shows "marginal or no improvement" on context-free benchmarks (CommonsenseQA, MMLU), making it equally restricted to context-driven scenarios. CAD is never included as a baseline in any experiment. This is not a secondary ablation — it is the primary competing method. Without this comparison, the core novelty claim is unverifiable and the paper cannot establish that Delta contributes beyond an acknowledged prior equivalent.

- **Misleading abstract framing of TriviaQA/NQ results**: The abstract claims "gains of 7 and 2 percentage points on TriviaQA and Natural Questions under-sampling decoding." However, under standard greedy (no-sampling) decoding, Delta is *worse* than baseline: TriviaQA 48.27 → 48.13 (−0.14 pp), NQ 14.88 → 14.57 (−0.31 pp), as shown in Table 1. The gains exist only under temperature=1 sampling, which the paper frames as the informative condition in Section 5.2 but which represents non-default inference. The positive gains are measured against a degraded sampling baseline that already underperforms the greedy baseline (TriviaQA sampling baseline: 35.39 vs. greedy 48.27). The abstract does not acknowledge this, creating a misleading overall picture.

- **Single-model evaluation**: All experiments use one model — Llama 3.1 8B Instruct in 4-bit quantization. The paper claims Delta is a "scalable solution for real-world LLM applications," but this claim is unsupported without testing on other model families, sizes, or quantization settings. Whether the masking mechanism generalizes — or whether results are idiosyncratic to this model — is entirely unknown.

### Minor

- **EOS token used as MASK token — unjustified and unablated**: Section 4.2 states "All experiments utilize the end-of-sequence (eos) token as the MASK token," with no motivation, reference, or comparison. The EOS token carries specific generational semantics and may interact with the model's attention machinery differently from a true [MASK] or [UNK] token. Since the method's entire claim rests on the masked input "amplifying hallucinations," the nature of the masking token is not peripheral. The observed effects could partly be artifacts of EOS semantics rather than of contextual deprivation.

- **Computational cost not discussed**: Delta requires two full autoregressive forward passes per generation step, doubling inference latency. The paper describes the method as "computationally efficient" without ever quantifying or discussing this 2× overhead. This is a significant omission for a method positioned as practical for deployment.

- **HasAns_EM regression in SQuAD v2 not analyzed**: Under no-sampling decoding, HasAns_EM drops from 59.08 to 57.47 when Delta is applied (Table 1). This suggests Delta may increase abstention rates broadly rather than selectively suppressing hallucinated answers. The NoAns_EM gain and HasAns_EM loss are not discussed together, and no analysis is provided to determine whether Delta genuinely corrects wrong answers or merely increases the refusal rate.

- **No multi-run variance or statistical testing**: Delta applies random masking, so results vary across runs. The ablation reports SD 0.66 for EM on SQuAD v1.1, which is non-trivial relative to claimed gains of ~3 pp. Single-run results are reported throughout with no indication of stability.

- **Hyperparameter ablation limited to one dataset**: The ablation varies r_mask and α on SQuAD v1.1 only; β = 0.1 is never varied. Whether the fixed parameters (r_mask = 0.7, α = 0.3) are also near-optimal for TriviaQA or NQ is untested.

### Trivial

- **ICD citation error**: In Section 2, both VCD and ICD are attributed to "Leng et al. (2024)," but the reference list contains only the VCD paper (Leng et al., CVPR 2024). ICD is a distinct work and should be cited separately.

---

## Nice-to-Haves

- Evaluation on at least one additional model (e.g., Mistral 7B, Gemma 2) to test generalizability across architectures.
- Token-level analysis showing which logits are suppressed under Delta on real dataset examples — the "moldy banana" illustration is hand-constructed; empirical token-shift evidence from actual benchmark instances would substantially strengthen the mechanistic claim.
- Ablation over MASK token type (EOS vs. [UNK] vs. a dedicated MASK token vs. random token).
- Refine scope claims: "could apply to all textual inputs" should be qualified to context-grounded generation tasks, consistent with empirical evidence.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Demand for DoLa comparison**: DoLa (Chuang et al., 2024) targets factuality hallucinations by contrasting internal layer representations, not contextual grounding, making it a different mechanism and setting from Delta. Including DoLa as a baseline would be a nice-to-have, not a critical omission given the different problem framing.
- **Hyperparameter information leakage concern**: The harsh reviewer suggests SQuAD v1.1 was used to tune hyperparameters that were then applied to other datasets. However, the ablation is explicitly framed as a post-hoc robustness check rather than a search, and the paper shows all configurations beat the baseline — this concern, while valid in principle, is not concretely evidenced.

---

## Novel Insights

None beyond the paper's own contributions. The key idea — replacing VCD's Gaussian image noise with random token masking for text-domain contrastive decoding — is a straightforward domain adaptation. The observation that Delta is most beneficial under sampling (where suppressing hallucination-prone logits has more practical impact) is a coherent but modest mechanistic insight.

---

## Suggestions

1. Run CAD on all experimental datasets and include the comparison; this is the single most important revision to make the contribution legible.
2. Perform multi-run (≥3 seeds) evaluation and report mean ± SD across all benchmarks to establish that gains exceed the noise floor from random masking.
3. Add at least one ablation comparing EOS as MASK token against UNK or a dedicated MASK token.
4. Revise the abstract to accurately reflect that TriviaQA/NQ gains only appear under sampling decoding, not standard greedy decoding.
5. Quantify inference latency overhead (2× forward passes) and compare wall-clock time with baseline.

---

## Score and Decision

**Calibration:**

The closest analogous paper found in the human-reviewed pool is **RITUAL** (aNYabH9Th4): a training-free, contrastive-decoding approach to hallucination in LVLMs using random image transformations instead of VCD's Gaussian noise — a direct analogue of Delta's adaptation of VCD to text via token masking. RITUAL was rejected (scores: 5, 5, 5, 5). Critically, RITUAL is *stronger* than Delta on multiple dimensions:
- RITUAL evaluates on multiple dedicated hallucination benchmarks (POPE, CHAIR, MME) and compares against VCD and other contrastive-decoding baselines.
- Delta omits its most direct competitor (CAD), evaluates on one model, and has results that degrade under the default decoding mode (greedy) on half of its context-rich benchmarks.

The **SID** paper (rsZwwjYHuD), a more principled and theoretically motivated contrastive decoding paper for LVLMs, was accepted as a poster (scores 6, 6, 5, 8). Delta is well below SID in novelty, evaluation breadth, and methodological rigor.

**Assessment:** Delta is weaker than RITUAL, which itself scored 5s (rejected). The missing CAD baseline, performance regressions under greedy decoding, single-model evaluation, unjustified design choices, and very thin experimental section (8 pages including references) place this paper below the bar for acceptance. I calibrate at **3.0**.

**Originality**: Low — direct adaptation of VCD with minor modification; acknowledged equivalent (CAD) not distinguished.  
**Importance**: Moderate question (hallucination mitigation is important), but addressed only partially and narrowly.  
**Claim support**: Weak — headline gains are selectively reported; standard decoding regressions omitted from abstract.  
**Experimental soundness**: Weak — single model, no variance reporting, no comparison to most relevant baseline.  
**Writing clarity**: Adequate, though the abstract is misleading.  
**Community value**: Low in current form; the idea may have merit if rigorously evaluated.

MY FINAL SCORE: <pineapple>3</pineapple>
MY FINAL DECISION: <orange>Reject</orange>