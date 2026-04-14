## Summary

MERLIM is a multi-modal evaluation benchmark for Instruction-Tuning Large Vision-Language Models (IT-LVLMs) covering three fundamental computer vision tasks: object recognition, object counting, and inter-object relationship understanding. The benchmark's core methodological contribution is an inpainting-based procedure that detects "hidden hallucinations"—cases where a model produces an apparently correct prediction that lacks any actual visual grounding, revealed by comparing model responses on original versus object-removed edited images. Experiments across 11 IT-LVLMs reveal that the best-performing models are also the most susceptible to hidden hallucinations, suggesting that strong aggregate scores are partially driven by language priors and global visual context rather than fine-grained visual grounding.

---

## Strengths

- **Operationalized hidden hallucination detection**: The inpainting-based methodology provides a concrete, automated way to test whether a correct prediction is genuinely grounded. Unlike POPE (which simply probes for object presence), MERLIM directly manipulates ground truth and verifies the same query on the modified image, enabling a causal attribution of the prediction to visual evidence. This distinction is well-motivated and the benchmark fills a real gap.

- **Counterintuitive empirical finding**: The discovery that the best-performing IT-LVLMs exhibit the largest precision gaps between original and edited sets (averaging 8.05% drop in object recognition precision) is a non-obvious result that strengthens the benchmark's diagnostic value. This finding—that capability and hallucination susceptibility are positively correlated—has real implications for how progress in IT-LVLMs should be measured.

- **Instruction bias analysis with carefully varied prompts**: The five semantically equivalent but syntactically different prompts for object recognition surface a meaningful and underappreciated failure mode: most models vary their output quality substantially (e.g., xGen-MM shows 26.50% F1 variability between best and worst prompt), while only BLIP-2 and Kosmos-2 remain consistent. This targeted design reveals which architectural properties confer robustness to prompt phrasing.

- **Random vs. Curated relationship sets**: The distinction between randomly sampled and LLM-curated negative relationships in the relationship task is a thoughtful design choice. The curated set (plausible but false relationships) nearly doubles the ΔAcc gap relative to the random set (16.15% vs. 7.49%), quantitatively demonstrating that language-solvable relationships inflate apparent performance.

- **Broad model coverage including 2024-era models**: MERLIM evaluates 11 models including xGen-MM (Phi-3-Mini), InternLM-XComposer2-VL, and Qwen-VL-Chat alongside the 2023 generation, providing reasonable contemporaneous coverage.

---

## Weaknesses

- **Figure 4 model naming inconsistency**: The model list in Section 4 specifies MiniGPT-4 with Vicuna-7B v0 and Vicuna-13B v0, and InstructBLIP with Vicuna-7B/13B/FlanT5xl. However, Figure 4's extracted tables list "MiniGPT-Vicuna-33B," "MiniGPT-Vicuna-65B," and "InstructBLIP-Vicuna-33B/65B"—variants never introduced or described anywhere in the text. This is a direct inconsistency that undermines confidence in the experimental details and needs to be resolved explicitly.

- **Inpainting artifact validation relegated to supplementary**: The entire "hidden hallucination" methodology rests on the assumption that the inpainting edits are visually imperceptible to models. If the edited images contain detectable artifacts (blurring, texture discontinuities), a precision drop on the edited set could reflect model sensitivity to image quality degradation rather than hallucination. This is the foundational assumption of the paper and its quantitative validation belongs in the main text, not the supplementary.

- **Gradient attribution and LLM-only baseline in supplementary**: Section 4.1 cites Table 5 (Supplementary) as evidence that "language tokens dominate over image tokens" and Section 4.2 states the LLM-versus-IT-LVLM comparison is in supplementary. Both of these are central to the paper's core claim that performance is language-biased rather than visually grounded—they should appear in the main paper.

- **Parsing pipeline reliability unquantified in main text**: The evaluation chain (spaCy → WordNet → ChatGPT synonym matching) is complex and each step can fail silently. The paper acknowledges "unsuitable outputs" and defers their frequency to supplementary. This is a significant omission: if a non-trivial fraction of responses are discarded or misparsed, the reported precision/recall/F1 values are not interpretable without knowing the noise floor. A table reporting parser error rates belongs in the main paper.

- **MS-COCO training data contamination unaddressed**: The benchmark is built entirely on the MS-COCO validation set, and several evaluated models (InstructBLIP, LLaVA-1.5) are trained on MS-COCO-derived data. The paper never addresses whether models may have partially memorized MS-COCO annotations, which could inflate both recognition performance and the apparent stability of predictions across edited images. This is a genuine threat to validity that is not acknowledged even in the limitations section.

- **Kosmos-2 degenerate behavior not analyzed**: Table 2 shows Kosmos-2 achieves ~90% Acc_neg (near-universal "No" response) while its Acc_org on the Random Set is only 19.34% and on the Curated Set 4.42%—a clear "always No" strategy. The paper notes high negative accuracy but does not flag this as a degenerate failure mode. A model defaulting to "No" achieves low ΔAcc without performing any visual reasoning; this needs explicit discussion to prevent misinterpretation of results.

- **Hidden hallucination rate not formally quantified**: Contribution (ii) claims that "hidden hallucination errors unfairly increase the accuracy of IT-LVLMs," but the paper never computes by how much accuracy would change if hidden hallucinations were removed, nor does it re-rank models after correction. The precision gap on edited images is indirect evidence; a direct quantification of the inflated performance is absent.

- **Low recall not discussed**: Figures 2 and 3 together imply very low recall values (F1 of ~20–48% with precision of ~40–58% implies recall well below 50% for most models). The paper focuses almost exclusively on precision because it connects to hallucinations, but severe under-prediction is a distinct failure mode that deserves analysis alongside over-prediction.

- **No statistical significance testing for relationship task**: Table 2 presents ΔAcc values ranging from 0.54% (MiniGPT-4 Vicuna-13B, Random Set) to 37.19% (LLaVA-1.5, Curated Set) on equal footing. Without confidence intervals, it is unclear whether the 0.54% difference is meaningful or noise, and conflating it with large effects undermines the analysis.

---

## Nice-to-Haves

- **Hallucinated class distribution analysis**: A breakdown of which object categories are most frequently hallucinated (e.g., head vs. tail classes in LVIS) would reveal whether models are biased toward frequent dataset priors rather than visual evidence—a natural and informative extension.

- **Model scaling analysis**: An analysis of hidden hallucination rate versus model parameter count would address whether scale systematically worsens or improves grounding, a question of interest to the ICLR community.

- **Coverage of frontier proprietary models (e.g., GPT-4V)**: Including at least one frontier model would give the benchmark contemporary relevance beyond the open-source ecosystem. The current model set is reasonable but skewed toward 2023-generation open-source models.

- **Human-verified subset**: A small human-annotated validation of the parsing pipeline's output would provide an upper-bound error estimate for the automated metrics and increase trust in the evaluation chain.

- **Explicit dataset release commitment**: The paper states the code is available but does not explicitly commit to releasing the pre-computed synonym corpus, the ChatGPT-generated question sets, and the inpainted images as static artifacts. Reproducibility of the benchmark depends on this.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"Title is unnecessarily cryptic"** (Harsh Critic): Pure stylistic preference; not a substantive concern.

- **"300K pairs claim is misleading"** (Harsh Critic): The 300,664 count is the legitimate product of prompts × images across all tasks. The paper does define how the count arises (Table 1), and the recommended usable subset is disclosed in Section 5. The framing is somewhat optimistic but not dishonest.

- **"Counting uses a single prompt is unexplained asymmetry"** (Harsh Critic): Counting tasks require a numeric response by nature, limiting syntactic variation. The paper uses 2 prompts (main count + consistency check). The asymmetry relative to 5 recognition prompts is reasonable given task structure.

- **"Benchmark should address object segmentation, spatial reasoning, captioning"** (Harsh Critic): This is scope creep. The paper explicitly scopes to three tasks and evaluates them well. Demanding additional tasks is not a weakness of the paper as written.

- **"Broader societal impact statement missing"** (Harsh Critic): Not a standard requirement for benchmark papers at ICLR.

- **"ChatGPT-generated curated set is not reproducible"** (Harsh Critic/Reviewer 2): The paper explicitly states "We pre-compute a corpus of noun-to-noun relations totaling 334,608 tuples and stored them, therefore future changes in ChatGPT model/API will not affect our benchmark predictions." The concern is addressed; what remains is only whether the pre-computed artifacts will be released publicly.

- **"Limited Discussion on Computational Cost"** (Reviewer 2): While useful, this is a nice-to-have practical detail, not a scientific weakness.

---

## Novel Insights

The most genuinely novel observation synthesized across all three reviews is the **inverse grounding paradox**: the best-performing IT-LVLMs are simultaneously the most susceptible to hidden hallucinations. This suggests that instruction tuning, rather than improving visual grounding, may be primarily teaching models to generate more fluent and contextually plausible language responses—creating an illusion of visual competence that standard VQA-style benchmarks cannot detect. The inpainting-based controlled comparison is an elegant methodological tool for exposing this gap, and the parallel finding in the relationship task (where the curated set doubles the ΔAcc gap) reinforces that the phenomenon is systematic across task types. The gradient attribution evidence (in supplementary) further suggests a mechanistic explanation: top-performing models weight previously generated language tokens more heavily than image features when completing a response.

---

## Suggestions

1. **Move inpainting validation, gradient attribution analysis, and the LLM-only baseline into the main text**; these three pieces of evidence are load-bearing for the paper's central claims and do not belong in supplementary.

2. **Resolve the Figure 4 model naming inconsistency** (MiniGPT-Vicuna-33B/65B, InstructBLIP-Vicuna-33B/65B): either correct the figure labels or add a table entry explaining what these model variants are.

3. **Report parsing error rates in the main paper**: a single table showing the fraction of responses discarded as "unsuitable" per model, along with a precision/recall estimate on a human-verified subset, would substantially increase confidence in the reported metrics.

4. **Add a direct computation of the accuracy inflation due to hidden hallucinations**: rather than only reporting ΔPrecision on edited images, compute what the F1 score would be if all hidden hallucination events were corrected, and report whether model rankings change.

5. **Acknowledge MS-COCO contamination as a limitation** and discuss how it could be mitigated in future versions (e.g., evaluating on a disjoint held-out dataset or restricting analysis to classes not present in training splits).

6. **Explicitly discuss Kosmos-2's degenerate "always No" behavior** and consider excluding or flagging models with extreme response bias when reporting ΔAcc summaries.

---

**Novelty**: The "hidden hallucination" concept and inpainting-based detection methodology are genuinely novel and address a specific gap that POPE-style benchmarks cannot reach.

**Technical soundness**: Moderate. The methodology is creative but several critical validation steps are deferred to supplementary, the parsing pipeline's reliability is unquantified in the main paper, and the Figure 4 naming inconsistency introduces doubt about experimental details.

**Empirical support**: Adequate for the qualitative trends, but weakened by absent significance testing and the unresolved contamination question. The core directional findings (hidden hallucinations exist, best models are most susceptible) are plausible and consistent across multiple tasks.

**Significance**: Meaningful for the ICLR community as a diagnostic tool—the benchmark exposes a failure mode invisible to standard VQA evaluation. Practical impact is limited by the need to resolve reproducibility and parsing reliability concerns before widespread adoption.

**Clarity**: The high-level exposition is clear and Figure 1 effectively communicates the hidden hallucination concept, but several sections require key supporting evidence currently buried in supplementary material.

MY FINAL SCORE: <pineapple>5.3</pineapple>