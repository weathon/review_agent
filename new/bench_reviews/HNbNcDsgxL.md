Now let me look at a few of the human review files for calibration.Now I have enough information. Let me write the final consolidated review.

---

## Summary

The paper proposes Delta, an inference-time contrastive decoding method that mitigates hallucinations in LLMs by randomly masking a portion of the input prompt, running the LLM on both the original and masked inputs, and subtracting the masked logits from the original logits (scaled by α). The method is evaluated on SQuAD v1.1/v2, TriviaQA, Natural Questions, CommonsenseQA, and MMLU using a single model (Llama 3.1 8B Instruct, 4-bit quantized). Results show meaningful gains on SQuAD v2's no-answer category but inconsistent improvements elsewhere, with slight regressions on context-free benchmarks.

---

## Strengths

- **Clear, implementable algorithm**: The formulation in Eq. 3 — `P_delta(y_t|z) = softmax[(1+α)·logit_θ(y_t|z) − α·logit_θ(y_t|mask(z))]` — is straightforward, training-free, and easy to deploy on top of any LLM.
- **Compelling result on unanswerable questions**: The 14.53 percentage point improvement on SQuAD v2 NoAns_EM (23.63 → 38.17 without sampling) is the strongest and most directly hallucination-relevant result in the paper. Preventing the model from generating answers when none exist is a meaningful operationalization of hallucination reduction.
- **Honest reporting of negative results**: The paper transparently discloses marginal declines on CommonsenseQA (−0.25 pp) and MMLU (−0.29 pp) and discusses why the method is limited there, rather than omitting those results.
- **Robustness of hyperparameters**: The ablation heatmap (Figure 2) shows that all 15 (mask ratio × logit ratio) configurations exceed the baseline on SQuAD v1.1, with a standard deviation of only 0.66 EM. Delta is not brittle to tuning within the tested range.

---

## Weaknesses

### Fatal
*None that individually invalidate the entire paper.*

### Major

- **Negligible novelty over Context-Aware Decoding (CAD)**: The paper's own Section 2 correctly describes CAD (Shi et al., 2024) as "amplifying the differences between outputs generated with and without the given context," which is structurally `(1+α)·logit(y_t|context,x) − α·logit(y_t|x)`. Delta is `(1+α)·logit(y_t|z) − α·logit(y_t|mask(z))` — mechanically almost identical, with the sole difference being "remove entire context" vs. "randomly mask 70% of tokens." The paper claims Delta is "more generalizable than CAD" because "CAD is mainly based on context-driven datasets," but (a) CAD is an algorithm, not a dataset, and (b) Delta itself fails on context-free tasks. Without a head-to-head empirical comparison with CAD on the same benchmarks and model, there is no evidence that Delta provides any practical advantage over prior art. This is the central novelty gap.

- **No experimental comparison with directly related baselines**: The only baseline is vanilla Llama 3.1 8B Instruct. CAD (Shi et al., 2024) and DoLA (Chuang et al., 2024) are cited in the related work but never compared against experimentally. These are the most directly comparable inference-time decoding methods targeting the same problem. Without these comparisons, the reader cannot determine whether Delta is competitive, redundant, or inferior to existing approaches.

- **Inconsistent results raise robustness concerns**: On TriviaQA and NQ **without sampling**, Delta actually decreases performance (48.27→48.13 on TriviaQA; 14.88→14.57 on NQ). Gains only appear with sampling (temperature=1). The explanation that "sampling is more prone to generating hallucinations due to the higher likelihood of sampling lower logit tokens" (§5.2) is ad hoc and empirically unsupported. This inconsistency suggests the method's benefits may be an artifact of the interaction between high-temperature decoding and logit rescaling, rather than genuine hallucination reduction.

- **Single-model, single-precision evaluation**: All experiments use Llama 3.1 8B Instruct with 4-bit quantization. There is no evidence the method generalizes to other architectures, model sizes, or non-quantized models. Given that 4-bit quantization itself perturbs logit distributions, it is unclear whether the gains are robust or specific to this setup.

### Minor

- **QA accuracy as an indirect hallucination proxy**: Standard EM/F1 on SQuAD v1.1, TriviaQA, and NQ do not directly measure hallucination (spurious spans, fabricated statements). The SQuAD v2 NoAns_EM metric is the closest to a direct hallucination measure, and it does show a strong result. However, the broader claim that Delta "mitigates hallucinations" across all tested benchmarks is overstated given that most gains are on span-extraction accuracy, not hallucination-specific metrics.

- **EOS as MASK token is non-standard and unexplored**: §4.2 states "All experiments utilize the end-of-sequence (eos) token as the MASK token." For an autoregressive instruction-tuned model, EOS has specific semantic meaning (generation termination). Replacing 70% of input tokens with EOS mid-sequence may induce pathological behavior that is unrelated to the hypothesized "hallucination amplification" effect. No alternative masking strategies are compared, and no analysis is provided of whether EOS insertion causes early stopping or other artifacts.

- **Efficiency claims are unsubstantiated**: The abstract and §1 claim Delta is "computationally efficient and easily deployable in real-time systems," but the method requires two full forward passes per decoding step (~2× inference cost). No latency, throughput, or FLOP measurements are provided.

### Trivial

- The ablation study (§6) covers only SQuAD v1.1 with sampling. No ablations are shown for the other benchmarks, different MASK tokens, or the contribution of APC (with vs. without).

---

## Nice-to-Haves

- Run at least one experiment with a structured masking strategy (e.g., masking only named entities or verbs via POS tagging) to begin validating the "future work" direction the paper already proposes.
- Add a direct head-to-head comparison table with CAD and DoLA using the same model — this single addition would substantially clarify the contribution.
- Evaluate on a dedicated hallucination benchmark (e.g., TruthfulQA or HaluEval) to strengthen the hallucination narrative.
- Report qualitative examples: show a case where vanilla decoding hallucinated and Delta corrected it, and a case where Delta failed.

---

## Removed Points

*These points are flagged for removal; treat with caution.*

- **Harsh Critic point: "Use of EOS as MASK is entirely opaque and possibly artifact-driven (structural/fatal)"** — Downgraded to Minor. The concern is real and worth investigating, but there is no evidence of systematic pathological behavior (the method does consistently outperform baseline on SQuAD). This is a legitimate gap in ablation, not a fatal flaw.

- **Harsh Critic point: "No statistical significance / single run"** — Kept in principle but weakened. Single-run evaluation is the norm in this community for large-scale QA benchmarks. The standard deviation across hyperparameter settings (0.66 EM) provides a weak proxy for robustness. The concern about small gains is noted under Major (inconsistent results) where it is more relevant.

- **Harsh Critic point: "Computational cost claims are unsubstantiated" (labeled as a major structural issue)** — Downgraded to Minor. The 2× forward-pass cost is a real concern, but for a short ICLR paper on a training-free decoding method, this is a common omission. The claim of efficiency is overclaimed but not the core issue.

- **Harsh Critic point: "Delta does not improve context-free benchmarks it claims to help (fatal)"** — Removed as fatal; paper explicitly acknowledges this limitation in §1 ("a fundamental limitation"), §5.3, and §7. The authors do not hide this. The concern about overclaimed generalizability is preserved under Major.

---

## Novel Insights

The reviewers collectively surface one genuinely useful observation: the improvement on SQuAD v2 no-answer cases is disproportionately larger than all other gains (~14 pp vs. ~3 pp), suggesting that the mechanism by which Delta helps is primarily **abstention induction** — learning to withhold an answer — rather than improved answer extraction per se. This implies the method functions more as a calibration tool for unanswerable questions than a broad hallucination mitigator. This framing would actually better align the paper's claims with its evidence and could be a stronger, more honest contribution story if pursued.

---

## Suggestions

1. **Compare directly against CAD and DoLA on the same benchmarks and model.** This is the single most important revision. Even a negative result (Delta ≈ CAD) would be informative and honest.
2. **Reframe the contribution around abstention on unanswerable questions**, where the 14.53 pp gain is compelling, rather than broad "hallucination mitigation."
3. **Ablate the MASK token choice**: compare EOS, a neutral unused token, and deletion (i.e., simply removing masked tokens). This would address the most mechanistically opaque aspect of the method.
4. **Test at least one additional model** (e.g., a non-quantized model or Mistral-7B) to establish that gains are not specific to 4-bit Llama 3.1.
5. **Report inference latency** relative to the vanilla baseline — a single table row showing overhead would address the efficiency claim.

---

## Score and Decision

**Calibration:**

- *Instructive Decoding* (Accept Spotlight, ~7.5 avg): Tested across many instruction-tuned models and tasks, consistent gains, clear differentiation from prior work. Delta is much more limited in scope and novelty.
- *DeCo* (Accept Poster, 6.0 avg): Contrastive decoding for hallucination, evaluated on multiple MLLM benchmarks with a clear empirical motivation. Accepted at borderline.
- *Contrastive Decoding Improves Reasoning* (Reject, ~4.3 avg): Inconsistent results across tasks, single model family, no comparison with other decoding strategies. Delta shares these problems and is weaker in scope.
- *CID paper* (Withdrawn/Reject, ~5.3 avg): Criticized for near-identical formulation to CAD with insufficient differentiation. Delta has the same issue, arguably more severely since CID at least added a theoretical privacy framing.

**Assessment:** Delta is clearly below DeCo (6.0) and the CID paper (5.3 avg). It sits closer to the Contrastive Decoding Reasoning paper (4.3 avg) but somewhat above it due to the honest reporting and the one genuinely strong result (NoAns_EM on SQuAD v2). The paper is below the acceptance bar: it presents a minimally novel variant of CAD with a single model, no baseline comparisons against the most directly related methods, and inconsistent results across settings. The contribution as currently validated does not justify acceptance.

**Score: 3.5**

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>