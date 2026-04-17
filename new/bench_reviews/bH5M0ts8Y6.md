## Summary

The paper proposes VINCIE, a method to learn in-context image editing models directly from video data, bypassing the need for task-specific paired image-editing datasets. The core idea is to transform videos into interleaved multimodal sequences (frames, textual transition annotations, segmentation masks) and train a Diffusion Transformer via three proxy tasks (next-image prediction, current/next segmentation prediction). The authors also introduce MSE-Bench, a 100-instance, 5-turn multi-turn image editing benchmark. Results on MagicBrush and MSE-Bench show competitive or strong performance, especially after supervised fine-tuning on pairwise editing data.

## Strengths

1. **Novel and well-motivated core idea**: Leveraging videos—which naturally encode sequential visual changes—as a scalable source for learning in-context editing is an elegant and important insight. The paper is the first to demonstrate the feasibility of this approach, and the data construction pipeline (VLM CoT annotation + GroundingDINO + SAM2) is practically viable and scalable.

2. **Strong scalability evidence**: Figure 5 shows nearly log-linear improvement in 5-turn success rates when scaling from 0.25M to 10M sessions (5% → 22% at Turn-5), clearly demonstrating the scalability advantage of the video-based approach. This is one of the most compelling empirical results in the paper.

3. **Effective proxy task design with ablation support**: The three proxy tasks (NIP, CSP, NSP) are well-justified, and Table 3 shows clear gains from adding segmentation tasks (CLIP-I: 0.784 → 0.823 on Turn-3; Turn-5 success rate: 0.113 → 0.173 on MSE-Bench). The context ablation in Table 4 also provides useful insight into multi-turn editing dynamics.

4. **Artifact accumulation mitigation**: Figure 6 provides a compelling demonstration that in-context editing naturally mitigates the artifact accumulation problem common in sequential single-turn editing—a practically important finding.

5. **Video sequence data effectiveness**: Table 5 shows a dramatic improvement from pairwise data to sequence data (Turn-5: 1.0% → 22.0%), and combining both yields 25.0%, suggesting genuine complementarity between video-derived and pairwise editing data.

## Weaknesses

### Major

1. **"Learned solely from videos" claim is overstated in key results**: The paper's central framing is that in-context editing can be "learned solely from videos" (Abstract, Sec. 1, Sec. 5). However, the strongest results in both Table 1 and Table 2 come from models fine-tuned on pairwise image-editing data ("+ SFT" rows). On MagicBrush, the paper claims state-of-the-art results, but this is achieved with SFT on editing data—not video-only training. The video-only models are described as "comparable to SOTA" on MagicBrush, which is a substantively weaker claim. On MSE-Bench, the 7B+SFT model (48.7% at Turn-5) still falls well short of proprietary models (62.7-64.3%). The abstract and conclusion blur this distinction, stating the model "achieves state-of-the-art results on two multi-turn image editing benchmarks" without clearly flagging that SOTA requires non-video SFT data.

2. **MSE-Bench evaluation relies solely on GPT-4o with no human validation**: The paper's new benchmark has only 100 test instances, provides no ground-truth images (by design), and relies exclusively on GPT-4o as judge. No human annotation, inter-annotator agreement, or calibration against human judgments is reported. On a 100-instance cascading evaluation where errors propagate across turns, this is insufficient for reliable quantitative claims. The success rates reported in Table 2 (e.g., 48.7% vs. 62.7% at Turn-5) are presented as definitive comparisons, but without variance estimates or human validation, the differences may not be statistically meaningful. Furthermore, since the video annotation pipeline also uses a VLM, there is a potential systematic bias in the data generation and evaluation pipeline.

3. **Insufficient isolation of video-sequence contribution from confounds**: Table 5 compares "pairwise" vs. "sequence" data, but the "pairwise" baseline (Wei et al., 2024) is a different dataset with different scale, domain coverage, and annotation quality. There is no size-matched or domain-controlled comparison. The MM-DiT initialization is pretrained on text-to-video tasks—a powerful prior that already encodes visual dynamics—but the paper does not include an ablation against a model initialized from a standard text-to-image backbone (e.g., an SDXL-scale DiT). Without these controls, it is unclear how much of the observed benefit comes from the proposed interleaved sequence training vs. the base model's video pretraining vs. simply having more diverse data.

### Minor

4. **No full-attention vs. block-wise causal attention comparison in main text**: The paper explicitly states (Sec. 3.2) that "both variants are compared to provide a direct assessment of their differences," but no comparison appears in any table or figure in the main paper. The reference to Appendix C.4 is insufficient given the explicit claim of comparison in the methodology section.

5. **No single-turn editing benchmark evaluation**: The model is evaluated only on multi-turn benchmarks. Showing performance on established single-turn editing benchmarks (e.g., EditVal, Emu Edit) would demonstrate whether video-derived training produces competent base editing capability or specializes narrowly to the multi-turn regime.

6. **Position-shift artifact from video training is acknowledged but not quantified**: Section 4.4 and Figure 7 note that native video training introduces subject position shifts, and that segmentation prediction mitigates it. However, no quantitative metric (e.g., object centroid displacement, unchanged-region IoU) is provided to assess the frequency or severity of this issue or the degree of mitigation achieved.

7. **Emergent capabilities lack systematic evaluation**: Multi-concept composition, story generation, and chain-of-editing are showcased in Figure 1 but supported only by cherry-picked qualitative examples with no task-specific evaluation or failure analysis. The paper's claim that these are "emergent" (Sec. 1, Sec. 4.5) is not substantiated beyond visual examples.

## Nice-to-Haves

- Human evaluation or human correlation study on a subset of MSE-Bench to validate GPT-4o judgments.
- Per-category breakdown on MSE-Bench (remove, replace, position, etc.) to reveal where video training excels vs. struggles.
- Analysis of VLM annotation quality/noise and its impact on downstream performance.
- Size-matched comparison between video-sequence data and pairwise data to isolate the effect of sequential structure vs. data volume/diversity.
- Inference cost analysis (latency/FLOPs per turn) as context grows.

## Removed Points

1. **"GPT-4o is being compared against itself"**: The harsh critic claimed GPT-4o evaluates itself as a competing model. In Table 2, GPT Image 1 is the image generation system being evaluated, while GPT-4o is used as the evaluation judge. These are different systems (text+vision model vs. image generation model), though both are from OpenAI. The concern about judge bias is valid, but the framing that GPT-4o directly evaluates "itself" is misleading and removed.

2. **"Annotation pipeline replaces one set of expert models with another"**: The neutral reviewer noted the pipeline depends on VLM, GroundingDINO, and SAM2, suggesting this undermines the "no expert models" claim. However, the paper never claims to eliminate all expert models—it claims to eliminate task-specific paired data pipelines. The annotation models (VLM, GroundingDINO, SAM2) are used for data construction, not as part of the editing model itself. This distinction is meaningful and the critique is a misrepresentation of the paper's actual claim.

3. **"3B model underperforms without SFT, contradicting the video-only claim"**: On MSE-Bench Turn-1, the 3B model without SFT (0.913) outperforms OmniGen (0.847/0.853) and matches many baselines. The underperformance is mostly at later turns, which is expected for multi-turn settings where context accumulation matters. This does not fundamentally contradict the video-only training claim.

4. **"Reproducibility concerns about in-house MM-DiT"**: The paper links to released code (https://vincie2025.github.io/) and provides detailed implementation descriptions. Per our rules, we do not flag reproducibility concerns about undisclosed hyperparameters or implementation details. The use of an in-house pretrained model is standard practice and the paper commits to code release.

5. **"Unfair comparison with proprietary models"**: The paper clearly marks proprietary models in gray in Table 1 and separates them from open-source methods. Comparing against proprietary systems is standard practice for context-setting and does not inherently constitute an unfair comparison, as long as the distinction is clear (which it is).

6. **Formatting/notation nitpicks**: Minor notation inconsistencies (RoE vs RoI, M00/M01 vs Mi/Mi+1) are trivial and do not affect understanding.

## Novel Insights

The most interesting finding is not just that video data can serve as a proxy for multi-turn editing data, but that the *structure* of the interleaved sequence matters substantially. Table 4 shows that adding even a dummy context (original image + "generate the same image") before Turn-1 nearly halves L1/L2 distances, and Table 5 shows that sequence data dramatically outperforms pairwise data (1.0% → 22.0% at Turn-5). This suggests that the primary bottleneck in multi-turn editing is not per-turn editing quality but rather the model's ability to maintain and leverage context—something video sequences naturally teach. However, the extent to which this benefit comes from the sequential training paradigm vs. simply from having more diverse data remains an open question that the current experiments do not fully disentangle.

## Suggestions

1. **Clearly separate video-only and video+SFT results in all claims**: In the abstract and conclusion, either report video-only results as the primary claim or explicitly frame the SFT-augmented results as a separate contribution. Currently, the abstract's "achieves state-of-the-art results" is misleading without acknowledging the SFT requirement.

2. **Add human evaluation on a subset of MSE-Bench**: Even 20-30 instances with 2-3 human annotators would provide calibration against GPT-4o and substantially strengthen the benchmark's credibility.

3. **Report variance/confidence intervals on MSE-Bench**: With only 100 instances, bootstrap confidence intervals or per-turn instance counts are essential for interpreting the reported success rates.

4. **Include size-matched ablation**: Train a model on pairwise data at the same scale (number of unique images/instructions) as the video sequence data to isolate the effect of sequential vs. individual training examples.

## Score and Decision

I calibrated against papers with similar patterns: GPT-based evaluation without human validation (EditVal: 5-6, DreamBench++: 6), interleaved multimodal generation with evaluation gaps (OpenLEAF: 3-5), and unsupervised/automated data construction for editing (UIP2P: 5-6, MGIE: 6-8).

VINCIE is stronger than OpenLEAF (which had more fundamental methodology issues and weaker empirical support) and comparable to MGIE in terms of novelty and engineering effort. However, it has more serious evaluation concerns than MGIE due to the small, unvalidated MSE-Bench. The overstated "video-only" framing in light of best results requiring SFT is a significant presentation issue. The core idea is genuinely novel and the scalability evidence is compelling, but the evaluation foundation is not yet rigorous enough to sustain the strongest claims.

**Calibration reasoning**: Papers with strong ideas but overclaimed results and weak evaluation methodology (e.g., GPT-only judges, no human validation, small benchmarks) typically land in the 5-6 range at top venues. VINCIE's novel idea and strong scalability evidence push it above the midpoint, but the evaluation gaps and overclaiming pull it back. It falls between EditVal/Multi-Reward (5-6) and MGIE (6-8).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>