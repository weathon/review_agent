Now I have enough calibration context. Let me assemble my final review.

**Calibration summary:**
- FusionBench (benchmark for model fusion): scores 3,6,5,6 → rejected
- UQ-Merge (MLLM merging method): scores 6,6,5,5 → withdrawn/rejected
- Realistic Evaluation of Model Merging: scores 5,6,5 → rejected
- KnOTS (SVD-based LoRA merging): scores 6,6,5,8 → accepted poster
- MMER (multimodal expansion via merging): scores 5,6,5,6 → rejected
- Multimodal Lego: scores 5,8,6,6 → accepted poster

This paper (OptMerge) is comparable to UQ-Merge and KnOTS in scope, with similar strengths (novel method + evaluation) and similar weaknesses (limited validation, some overclaiming). The key differentiator is the benchmark contribution. KnOTS was accepted with scores 6,6,5,8. UQ-Merge was rejected with 6,6,5,5.

This paper has:
- A genuine benchmark contribution (first MLLM merging benchmark with capability categories)
- A method that shows consistent but modest improvements over WUDI
- A significant overclaiming problem with "surpasses mixture training"
- Limited modality merging evidence
- Theoretical result disconnected from experiments

Given the calibration, I'd place this around 5.5 — it has real contributions but significant weaknesses in its central claim and limited experimental rigor.

## Summary

The paper introduces OptMerge, a model merging benchmark and method for Multimodal LLMs (MLLMs). The benchmark categorizes MLLM capabilities into five tasks (VQA, Geometry, Chart, OCR, Grounding) across two model families (InternVL2.5-1B with full fine-tuning and Qwen2-VL-7B with LoRA), evaluates 10 merging algorithms, and explores modality merging (vision-language, audio-language, video-language). The proposed OptMerge method improves upon WUDI Merging by using SVD-based denoising of task vectors and tailored optimization strategies (SGD + mean initialization for LoRA), achieving average improvements of ~2.48% over WUDI.

## Strengths

- **First MLLM merging benchmark with clear capability categorization**: The paper fills a genuine gap by providing a structured benchmark that divides MLLM tasks into five capability categories, trains expert models on ≥100K samples each, and releases checkpoints. This is more systematic than prior work (AdaMMS merges only 2 models; UQ-Merge lacks capability categorization). The inclusion of both full fine-tuning and LoRA settings and HuggingFace community checkpoints adds practical value.

- **Emergent multi-ability results (Table 10) are compelling**: On general multimodal benchmarks (MMMU, DocVQA, ScienceQA, AI2D, InfographicVQA), merging five specialized models yields 10.85% average improvement over the best individual expert. This is a strong empirical demonstration that merging produces genuinely multi-capable models, not just averaged ones.

- **Practical relevance demonstrated via HuggingFace checkpoints (Table 6)**: Merging four independently developed community models (math, Pokemon, OCR, Vietnamese) and surpassing any individual model (66.70 avg) shows real-world applicability beyond curated benchmarks.

- **Methodological contribution (OptMerge)**: The insight that LoRA and full fine-tuning models have fundamentally different task vector distributions (multi-modal vs. right-skewed) and require different merging strategies is motivated by the analysis in Section 3.2 and Figure 2. The ablation (Table 4) cleanly shows the contribution of each component (+4.65% total gain over WUDI).

## Weaknesses

### Major:

- **The central claim that "model merging potentially surpasses mixture training" is not properly supported**: This claim appears prominently in the abstract, introduction, and conclusion, yet the evidence is at best mixed:
  - **InternVL2.5 (Table 2)**: Mixture Training average = 57.66, OptMerge = 57.44. Mixture training actually wins by 0.22 points.
  - **Qwen2-VL (Table 3)**: The "mixture training" baseline is Qwen2-VL-Instruct, a model trained on different, broader data than the 5 benchmark-specific datasets. This is not a controlled comparison of merging vs. training on the same data mixture. The 1.1-point advantage of OptMerge (63.30 vs. 62.23) could easily be explained by data distribution differences.
  - **HuggingFace (Table 6)**: Again compared against Qwen2-VL-Instruct, not a model trained on the union of the four checkpoint domains.
  - There is no experiment where the same base model is jointly fine-tuned on the exact union of the expert datasets and compared against the merged model in a controlled fashion. The InternVL2.5 comparison (the closest to fair) shows mixture training winning, not losing. This overclaiming significantly undermines the paper's narrative.

- **Limited modality merging evidence**: The claim that "complementarity among multiple modalities outperforms individual modalities" (abstract) rests on only 2 evaluation benchmarks (MUSIC-AVQA and AVQA) using a single LLM backbone (Vicuna-7B-v1.5). There is no comparison against a jointly-trained tri-modal model, and no failure analysis or broader evaluation across more diverse cross-modal tasks. The conclusion that this "advances Omni-language models" substantially overstates what two benchmarks on one backbone can demonstrate.

- **No error bars or statistical significance**: All results appear to be single runs. Since the margins between OptMerge and the next-best method are often 0.2–1.5 points on noisy generative evaluation benchmarks, the reliability of these rankings is uncertain. The "2.48% average improvement" is measured only relative to WUDI, not the overall best baseline; actual margins over the strongest non-OptMerge competitor are typically much smaller (e.g., 57.44 vs. 57.00 on InternVL2.5, 63.30 vs. 61.88 on Qwen2-VL).

### Minor:

- **Theorem 3.1 lacks empirical validation**: The theoretical upper bound on merged model loss (depending on learning rate η, iterations T, PL factor γ, and cross-task interference δ) is presented as "the first theoretical explanation" but is never empirically validated. The reference to "experiments in App. B.1" showing rise-then-fall behavior is not shown in the main text, and no quantitative connection between the bound's terms and the actual fine-tuning hyperparameters used in the benchmark is established. The theory is conceptually interesting but currently decorative.

- **"Data-free" claim is partially misleading**: The paper claims OptMerge requires "no hyperparameter search" compared to AdaMMS, yet still searches λ over [0.1, 0.3, 0.5, 0.7, 1.0, 1.5] using evaluation data. While this is standard for static merging methods, the framing of "data-free" overstates the degree to which the method avoids data dependence. The claim should be qualified (e.g., "data-free during merging vector optimization" rather than broadly "no hyperparameter search").

- **Ad-hoc nature of LoRA-specific techniques**: The switch from Adam to SGD for LoRA models (Section 4.2) is justified empirically rather than theoretically. Table 4 shows that SGD alone *reduces* performance by 9.77% compared to WUDI, and recovers only with mean initialization. This suggests the method's success depends on a specific initialization strategy rather than a principled optimizer choice.

- **Iso-C failure on Qwen2-VL is not adequately analyzed**: Iso-C catastrophically fails (26.69% avg.) on the LoRA benchmark, which the paper briefly attributes to LoRA task vectors already being low-rank. This is an important finding for the merging community that deserves deeper analysis.

### Trivial:

- The Frobenius norm analysis (Figure 2) could be more quantitative — current presentation is qualitative for motivating the different merging strategies.

## Nice-to-Haves

- A proper controlled mixture-training baseline (same data, same base model) for at least one setting would substantially strengthen or qualify the main claim.
- Error bars across 2–3 seeds for the main comparison tables.
- Evaluation of the merged model on general LLM benchmarks (e.g., MMLU) to verify base capabilities are preserved after merging.
- Connecting Theorem 3.1 to the actual fine-tuning schedules used (measuring δ, verifying the bound's tightness).
- More diverse modality-merging evaluation benchmarks.

## Removed Points

- **"Benchmark novelty is incremental"** (Human Finder): While the benchmark curates existing datasets, this is standard in benchmark papers and addresses a real gap (no MLLM merging benchmark existed). The value is in the curation, checkpoint creation, and evaluation protocol, not in inventing new datasets. — *Kept as minor consideration but not a standalone weakness since benchmark creation inherently involves curation.*

- **"Limited architectural diversity"** (Human Finder): The paper uses 3 model families across different scales (1B, 7B, 32B) and fine-tuning paradigms (LoRA, full SFT). This is reasonable for an initial benchmark, though more architectures would strengthen generality.

- **"Missing evaluation of general LLM abilities"** (Spark): This is outside the paper's stated scope (MLLM capability merging) but would be a useful addition. Not a core flaw.

- **"Task scope is vision-centric"** (Neutral reviewer): The five tasks are deliberately vision-language tasks because those are the primary MLLM capabilities in current practice. The modality-merging section addresses non-vision modalities. This is a scope complaint.

## Novel Insights

The analysis showing that LoRA and full fine-tuning models have fundamentally different task vector distributions (multi-modal vs. right-skewed) and require different merging strategies is a genuine insight. The empirical finding that SGD with mean initialization significantly improves LoRA merging (though SGD alone is catastrophic) suggests that optimization pathologies in the low-rank null space are a key challenge. The Table 10 results showing emergent multi-ability from merging are perhaps the most compelling finding — that merging specialized MLLMs can produce capabilities on complex integrated benchmarks that no individual specialist possesses.

## Suggestions

1. **Add a controlled mixture-training baseline**: Fine-tune Qwen2-VL-Base on the union of all five task datasets and compare directly with the merged model. This is the single most important addition to properly evaluate the paper's central claim.
2. **Qualify the "surpasses mixture training" claim**: At minimum, note that InternVL2.5 shows mixture training matching or slightly exceeding merging, and that the Qwen2-VL comparison is against a differently-trained instruct model. Currently, the claim is overstated relative to the evidence.
3. **Report per-task retention ratios**: Show merged/expert performance ratios alongside averages to transparently display which tasks lose ground after merging.
4. **Expand modality merging evaluation**: At least 3–5 more cross-modal benchmarks would substantially strengthen the "Omni-language model" narrative.

## Evaluation

- **Originality**: Moderate. The benchmark is the first for MLLM merging with capability categorization, and the LoRA/full fine-tuning distinction is valuable. OptMerge is an incremental improvement over WUDI with technically motivated modifications. The theoretical contribution is conceptually interesting but disconnected from experiments.

- **Importance of research question**: High. MLLM merging is practically important for the open-source community, and a structured benchmark fills a real gap.

- **Claims support**: Mixed. The core claims about OptMerge's improvements over WUDI are well-supported. The headline claim about surpassing mixture training is not well-supported by controlled experiments. Modality merging claims are preliminary.

- **Soundness of experiments**: Moderate. The benchmark design is solid, but the lack of error bars, the uncontrolled mixture-training comparison, and limited modality evaluation weaken the experimental rigor.

- **Clarity**: Generally clear, though the "data-free" and "surpasses mixture training" framings oversell what is demonstrated.

- **Value to community**: High. The benchmark, checkpoints, and comprehensive comparison of 10 methods on MLLMs will be useful to the merging community regardless of the method's marginal improvements.

## Score and Decision

Calibration against similar papers:
- **KnOTS** (SVD-based LoRA merging, accepted poster, scores 6/6/5/8): Similar scope — novel merging method + benchmark contribution. OptMerge has a broader MLLM benchmark but a less novel core method and more overclaiming.
- **UQ-Merge** (MLLM merging, withdrawn/rejected, scores 6/6/5/5): Similar domain. OptMerge has a stronger benchmark contribution but similar issues with limited evaluation scope.
- **Realistic Evaluation of Model Merging** (benchmark paper, rejected, scores 5/6/5): Benchmark-focused evaluation, no novel method. OptMerge adds a method but with modest improvements over the prior art.
- **FusionBench** (benchmark only, rejected, scores 3/6/5/6): Pure infrastructure benchmark with no novelty. OptMerge clearly exceeds this.
- **MMER** (multimodal expansion via merging, rejected, scores 5/6/5/6): Similar domain, similar modality-merging scope issues. OptMerge has stronger empirical results on capability merging.

OptMerge is stronger than pure benchmark or narrowly-evaluated merging papers (FusionBench, MMER), competitive with UQ-Merge, but weaker than KnOTS (which had a cleaner contribution). The main issue dragging it down is the overclaiming around "surpasses mixture training" without a controlled baseline, which inflates the narrative beyond what the data supports. The benchmark contribution is real and valuable.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>