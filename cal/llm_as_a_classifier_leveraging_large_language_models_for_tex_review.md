=== CALIBRATION EXAMPLE 15 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Does the title accurately reflect the contribution?** Yes. The title clearly indicates the focus (using LLMs for classification) and the modality scope (text and vision).
- **Does the abstract clearly state the problem, method, and key results?** The abstract succinctly frames the latency problem with standard LLM decoding, introduces the single-token constrained generation approach with PEFT, and reports accuracy/latency metrics. However, it references `GPT-5`, which is not a publicly verified or benchmarkable model in the current literature. This undermines the reproducibility and credibility of the reported `51.8%` baseline.
- **Are any claims in the abstract unsupported by the paper?** The claim of "zero-shot adaptation" (reassigning label tokens at inference time to adapt to new tasks without retraining) is heavily contradicted by empirical results in Appendix A.6, where accuracy drops from `62.7%` to `~44.3%` under unseen token-label permutations. This discrepancy must be reconciled before publication.

### Introduction & Motivation
- **Is the problem well-motivated? Is the gap in prior work clearly identified?** Yes. The tension between LLM flexibility/generality and encoder-based speed/efficiency is well-established, and the specific pain point of unpredictable, multi-token generation latency in real-time deployments is clearly articulated.
- **Are the contributions clearly stated and accurate?** The three pillars (accuracy, latency, generality) are clearly stated. However, the contribution of "generality/zero-shot adaptation" is overstated given the significant performance degradation when the token-label mapping changes (Sec A.6). The novelty of the method is also more of a practical engineering design (special tokens + LoRA) than a fundamental algorithmic advance, which should be calibrated in the introduction to meet ICLR's novelty expectations.
- **Does the introduction over-claim or under-sell?** It overclaims on the `O(1)` latency guarantee (discussed in Method) and the seamless nature of zero-shot task switching. It under-sells the importance of training data distribution; the gains heavily depend on constructing a diverse multi-domain corpus, which is a major engineering effort in itself.

### Method / Approach
- **Is the method clearly described and reproducible?** Largely yes. The use of LoRA, cross-entropy on a single output position, and tokenizer extension are standard but clearly integrated. Prompt templates are referenced in the appendix. The randomization of class-to-token mappings during training is well-justified to prevent token-ID memorization.
- **Are key assumptions stated and justified?** The method assumes that label descriptions in the prompt will sufficiently ground the model's single-token output. This is stated but not deeply analyzed. The assumption that multi-token verbalizers inherently scale poorly with vocabulary size is also slightly overstated; decoding length is typically bounded by 2-3 tokens, and the primary latency bottleneck is KV-cache management and autoregressive loop overhead, not linear scaling with label count.
- **Are there logical gaps in the derivation or reasoning?** The claim of `O(1)` latency (Fig 1, Intro) is mathematically imprecise. LLM inference complexity is `O(L_in + L_out)`. Since `L_out = 1`, complexity scales linearly with input context length `L_in`, which is often the dominant factor. The claim should be corrected to "single decoding step" or "constant output-step latency."
- **Are there edge cases or failure modes not discussed?** The method forces a deterministic choice. The model cannot naturally abstain, express uncertainty, or request clarification. In safety-critical or open-world classification, forcing a single token without an explicit "unknown/refuse" class or calibrated confidence threshold is a significant failure mode.

### Experiments & Results
- **Do the experiments actually test the paper's claims?** They test accuracy and latency well, but the zero-shot generalization claim is inadequately tested (addressed below).
- **Are baselines appropriate and fairly compared?** This is a major concern. Table 1 compares a *fine-tuned* 27B/24B model (trained on 28k examples, including 14k multimodal data from domains overlapping with MINTREC) against *zero-shot* GPT-4o. This is an inherently asymmetric comparison. ICLR standards typically require either fine-tuning the proprietary baseline (via API adaptation if possible) or constructing a rigorous few-shot prompt baseline with comparable context to ensure the accuracy gap is due to the LaaC framework rather than just the LoRA adaptation to in-domain data. The text classification evaluation (Table 2) mitigates this by testing zero-shot cross-domain generalization, but the multimodal results remain skewed.
- **Are there missing ablations that would materially change conclusions?** Yes. 
    1. *Training data ablation:* What is the exact contribution of the 14k text vs 14k multimodal examples? Section 4.8 touches on this but lacks a systematic study (e.g., removing specific sources like FollowBench or AgentGym).
    2. *LoRA vs Full Fine-Tuning:* Did LoRA saturate early, or would full fine-tuning yield significantly better accuracy/latency profiles?
    3. *Randomization Ablation:* How much does the randomized token mapping actually contribute compared to a fixed mapping? The paper claims it prevents memorization, but without a control group, this remains an assumption.
- **Are error bars / statistical significance reported?** No. All accuracy and latency metrics are point estimates. Given the stochastic nature of LLM inference and the sensitivity to seed/prompts, reporting standard deviations across multiple runs (especially for the 200 sample subsets in text benchmarks) is essential for ICLR.
- **Do the results support the claims made, or are they cherry-picked?** The latency results are robust and well-supported (Tables 1, 2, A.5, A.7). The accuracy claims are supported for in-domain fine-tuning but overstated for generalization. The Appendix A.6 results (accuracy dropping to `~44%` under permutation) directly undermine the "seamless zero-shot adaptation" narrative.
- **Are datasets and evaluation metrics appropriate?** Datasets are standard. Latency measurement via vLLM on a single A100 is appropriate for controlled comparison. However, batch-1 latency does not reflect production throughput; while Appendix A.7 addresses this, the main text's emphasis on P50/P95 single-batch metrics may mislead readers about real-world system-level latency.

### Writing & Clarity
- **Are there sections that are confusing or poorly explained?** Section 4.5.1 claims the fine-tuned Gemma-3-27B achieves `62.7%` on MIntRec 2.0, yet Table 1 appears truncated in the parsed text (cutting off after Mistral's results). If this is a parser artifact, it's fine; if the original submission omitted it, it's a major flaw. I assume artifact, but the discrepancy in the permutation results (Sec 4.5.2 vs A.6) creates conceptual confusion around what "zero-shot adaptation" means.
- **Are figures and tables clear and informative?** Table 2 and 3 are structurally clear. Figure 3 (scaling analysis) effectively communicates the accuracy/latency trade-offs. The Pareto figure (Fig 10) is useful but relies on the contested accuracy claims. The axis definitions for "generality" in Fig 10 are not precisely defined in the text (is it a composite score of cross-dataset average accuracy?).

### Limitations & Broader Impact
- **Do the authors acknowledge the key limitations?** They note extension to audio, need for calibration/robustness/multilingual analysis, and combining with reasoning.
- **Are there fundamental limitations they missed?** 
    1. *Prompt Sensitivity:* Collapsing classification to a single token places immense weight on the quality and phrasing of the label descriptions in the system prompt. Poor descriptions or long context windows that push label definitions out of the attention window will severely degrade the single-token prediction, yet the method lacks a fallback or verification step.
    2. *Calibration & Abstention:* As noted, forcing a single token eliminates the LLM's ability to naturally express low confidence. For high-stakes classification, the lack of a calibrated probability thresholding or an explicit abstention token is a critical deployment limitation.
    3. *Token Vocabulary Bound:* The method caps at 500 reserved tokens (Sec 3.2 & 3.3). Real-world industrial classification systems often require thousands of fine-grained intents (e.g., Banking77 has 77, but enterprise taxonomies can exceed 500). The paper mentions this indirectly for multimodal but does not rigorously test scalability beyond 77 classes (Table 7) or propose a hierarchical decoding strategy for `K > 500`.
- **Are there failure modes or negative societal impacts not discussed?** The latency gains could accelerate automated moderation or content classification, which raises standard dual-use concerns. More specifically, single-token black-box classifiers lack interpretability compared to verbose CoT or encoder attention visualizations, making debugging bias in automated decisions harder.

### Overall Assessment
The paper addresses a practical and well-motivated problem: optimizing LLM classification for low-latency, throughput-sensitive deployments. The core engineering design—extending the vocabulary with atomic class tokens, randomizing mappings to prevent memorization, and restricting generation to a single step via PEFT—is clean, effective, and yields empirically verified latency improvements. However, the paper falls short of ICLR's acceptance bar in its current form due to several critical issues. First, the experimental design compares a domain-fine-tuned model against zero-shot proprietary baselines, inflating the perceived accuracy gains of the framework. Second, the central claim of "zero-shot adaptation" via token reassignment is directly contradicted by the `~18%` accuracy drop reported in the appendices when mappings are unseen. Third, the absence of statistical variance (error bars) and the use of an unverifiable `GPT-5` baseline undermine reproducibility and rigor. Finally, the `O(1)` latency claim is mathematically inaccurate, and the method's inability to abstain or calibrate confidence is under-discussed. The contribution stands as a solid engineering system design with strong latency gains, but it requires stricter baseline symmetry, reconciliation of the zero-shot claims, proper statistical reporting, and a more nuanced discussion of failure modes before it can be considered complete.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes LaaC (LLM as a Classifier), a framework that reformulates text and multimodal classification as constrained, single-token generation using atomic special tokens and parameter-efficient fine-tuning (LoRA). By randomizing token-label mappings during training, the method prevents token memorization and enables cross-dataset generalization without task-specific retraining. Experiments across multiple benchmarks demonstrate that LaaC achieves competitive or superior accuracy to proprietary models and encoder-based baselines while delivering substantially lower and more predictable inference latency.

### Strengths
1. **Clear and practical methodological design:** The formulation of classification as a deterministic one-step decoding problem via atomic special tokens is well-motivated and elegantly executed. The use of randomized label assignments during training (Section 3.3) is a strong design choice that prevents the model from overfitting to static token-class associations and directly supports the claimed cross-dataset adaptation.
2. **Comprehensive and rigorous empirical evaluation:** The paper provides extensive latency measurements (P50/P95) alongside accuracy, using a consistent vLLM setup (Section 4.4). Tables 1 and 2, along with Appendix A.7, thoroughly quantify the latency advantages across single-example and batched inference, showing consistent speedups (e.g., sub-200ms P95 on text tasks vs. ~0.8–1.2s for GPT-4o).
3. **Strong emphasis on real-world deployment constraints:** Unlike many LLM papers that focus solely on accuracy, this work explicitly addresses latency-critical applications. The breakdown of encoder pipeline latency (Appendix A.5) and the analysis of constrained decoding overhead for GPT-4o (Appendix A.4) provide valuable context, demonstrating a nuanced understanding of production inference bottlenecks.

### Weaknesses
1. **Terminology mismatch regarding “zero-shot”:** The paper claims zero-shot capability, but the model is actually fine-tuned on a 28k mixed-domain corpus. In standard ML terminology, this represents cross-dataset transfer or training-free adaptation to unseen label spaces rather than true zero-shot learning. Evaluating on SST-2, AG News, etc., without fine-tuning on those specific datasets is cross-domain evaluation, which should be explicitly reframed to avoid confusion with established prompt/few-shot literature.
2. **Inconsistent accuracy trends on text benchmarks:** While LaaC matches or exceeds GPT-4o on most tasks, it slightly underperforms its own base model on AG News (81.5% vs 84.0% for Gemma-3-27B Base) and DBpedia (95.0% vs 97.0% Base) in Table 2. The paper does not deeply analyze this regression. Possible causes include dataset shift from the training corpus, label semantic mismatches, or the restrictive single-token loss masking limiting nuanced probability calibration.
3. **Proprietary model benchmarking reproducibility concerns:** The paper benchmarks against “GPT-5” and “GPT-5-NANO” (Table 1). As these models lack standardized, publicly verifiable API versions or snapshot dates in current academic literature, their inclusion may raise reproducibility and fairness concerns typical of ICLR reviewing. Additionally, comparing a mixed-domain fine-tuned LaaC model against highly prompt-optimized proprietary systems without standardized prompt tuning baselines slightly complicates the fairness of the accuracy comparison.

### Novelty & Significance
- **Novelty:** Moderate to High. Cloze-based classification and constrained decoding are established, but the systematic combination of atomic single-token outputs, randomized token-class mappings, and cross-modal PEFT is a well-engineered methodological advance. It represents a strong system/design contribution rather than a fundamental theoretical breakthrough.
- **Clarity:** High. The paper is exceptionally well-structured, with clear mathematical formalization (Section 3), intuitive diagrams (Figure 1), and thorough experimental reporting. The methodology, training objectives, and inference rules are precise and easy to follow.
- **Reproducibility:** High. The authors provide explicit hyperparameters (LoRA rank 8, α=16, dropout 0.05, LR 2e-5, 30 epochs), hardware/deployment details (8×A100, DeepSpeed ZeRO-3, vLLM configuration), a promised code repository, and detailed dataset formulations. The only minor gap is the exact class distribution/composition ratios of the 28k training corpus.
- **Significance:** High. The work directly addresses a critical industry and research pain point: deploying LLMs for low-latency, high-throughput classification without sacrificing generality or accuracy. The findings are highly relevant for real-time AI systems, effectively bridging the efficiency-generalization gap between specialized encoders and flexible decoders.

### Suggestions for Improvement
1. **Clarify evaluation terminology and baseline framing:** Reframe “zero-shot classification” as “cross-dataset generalization” or “training-free adaptation to unseen label spaces.” When comparing to BERT/RoBERTa/LM-BFF, explicitly note that those baselines use few-shot fine-tuning on the target domain, whereas LaaC relies on domain-shifted pre-training. Consider adding a minimal few-shot adaptation comparison to LaaC to demonstrate how quickly it can peak on a new task.
2. **Investigate accuracy regressions on AG News/DBpedia:** Provide a brief analysis or error characterization for why LaaC underperforms the base model on certain text benchmarks. Confusion matrices, label-level performance breakdowns, or logit analysis would clarify whether the bottleneck stems from label vocabulary mismatch, insufficient training coverage, or the single-token masking strategy affecting calibration.
3. **Expand deployment analysis beyond latency:** Since the latency advantage narrows at large batch sizes (Appendix A.7), include a throughput (samples/sec) vs. VRAM usage comparison. Serving engines optimize batched autoregressive decoding differently than single-token constrained outputs, and a cost-benefit analysis (e.g., memory overhead of tokenizer extensions/LoRA weights vs. latency savings) would greatly enhance the paper’s practical value for ICLR reviewers and practitioners.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Evaluate on full test splits for all text benchmarks; sampling only 200 examples yields high statistical variance that undermines accuracy claims and violates standard ICLR evaluation rigor.
2. Include a baseline fine-tuning the same base models with a standard linear classification head on the final hidden state; without this, accuracy gains cannot be isolated from generic LoRA benefits versus the atomic token mechanism.
3. Benchmark true throughput (requests/sec under concurrent mixed loads) rather than isolated latency sweeps, as single-example latency on one GPU does not reflect real-world serving efficiency or the actual compute cost of 27B models.
4. Evaluate out-of-domain (OOD) detection or abstention rates; real-world classification deployments require calibrated rejection of out-of-scope inputs, which the current deterministic argmax mechanism entirely ignores.

### Deeper Analysis Needed (top 3-5 only)
1. Resolve the critical contradiction between the 62.7% baseline accuracy and the 44.35% accuracy under control-token permutations (Table 4); this ~18% drop directly invalidates the claim that the model learns task-agnostic descriptions for zero-shot label reassignment.
2. Provide error analysis and confusion matrices for larger label spaces (AG News, DBpedia, Banking77) where LaaC underperforms GPT-4o to determine if arbitrary atomic tokens lose semantic grounding and cause systematic class confusions.
3. Quantify the impact of masking loss on all non-final tokens on the model's retained generative capabilities; the paper claims preserved flexibility but offers no instruction-following or text-generation benchmarks post-fine-tuning.
4. Apply bootstrapping or repeated random sampling to all accuracy metrics and report 95% confidence intervals, as point estimates on small or single runs are statistically insufficient to claim superiority over baselines.

### Visualizations & Case Studies
1. Visualize the embedding space of learned atomic tokens versus standard verbalizer words (e.g., via t-SNE/PCA) to verify whether the method captures meaningful semantic class boundaries or merely memorizes arbitrary mappings.
2. Present qualitative failure cases where the single-token constraint forces high-confidence incorrect labels, contrasting with unconstrained baselines that successfully generate multi-step reasoning or correct classifications.
3. Plot full latency distribution histograms rather than only P50/P95 summaries to expose whether the O(1) latency claim holds or if hardware jitter and serving overhead create unpredictable tail behavior.

### Obvious Next Steps
1. Replace the 200-example text evaluations with full-dataset runs immediately, as the current sampling strategy is the single largest threat to the paper's credibility regarding cross-domain generalization.
2. Conduct a strict ablation isolating the special token design by comparing LaaC directly to standard decoder fine-tuning with a linear classification head, proving the single-token constraint is necessary for the reported gains.
3. Update multimodal encoder baselines to modern, competitively fine-tuned architectures (e.g., modern DeBERTa-v3 variants or contemporary VLM encoders) rather than relying on dated pipelines like MAG-BERT/MulT.
4. Retract or substantially weaken the zero-shot adaptation claim until the token-parmutation accuracy drop is explained and resolved by either architectural changes or stricter fine-tuning regularization.

# Final Consolidated Review
## Summary
LaaC reframes text and multimodal classification as constrained single-token generation by extending tokenizer vocabularies with atomic control tokens and applying parameter-efficient fine-tuning (LoRA). The method replaces verbose, multi-token autoregressive decoding with a deterministic one-step argmax over a finite label set, delivering predictable sub-200ms latency while preserving the generative architecture of decoder-style LLMs.

## Strengths
- **Clean, deployment-aware engineering:** The formulation of classification as a restricted vocabulary mapping effectively eliminates multi-token generation overhead. The integration of LoRA with single-position loss is straightforward, reproducible, and directly addresses the latency unpredictability that hinders real-time LLM deployments.
- **Rigorous latency characterization:** Reporting P50/P95 across single-example and batched regimes, coupled with a transparent breakdown of encoder pipeline bottlenecks (e.g., vision feature extraction vs. encoder inference), demonstrates a mature understanding of production serving constraints beyond raw accuracy metrics.
- **Competitive zero-shot cross-domain transfer:** Without target-domain fine-tuning, LaaC matches GPT-4o accuracy on SST-2 and Amazon Reviews while cutting tail latency by nearly an order of magnitude, validating the framework’s practical value for latency-sensitive pipelines.

## Weaknesses
- **Statistically underpowered text evaluation undermines accuracy claims:** Evaluating standard NLP benchmarks on only 200 randomly sampled examples (Sec 4.1) is insufficient for ICLR. Small sample sizes introduce high variance and prevent robust statistical claims, especially when reporting point estimates without confidence intervals or error bars. This makes the reported cross-domain parity with proprietary APIs and encoder baselines scientifically fragile.
- **Core "zero-shot adaptation" claim is empirically contradicted:** The paper asserts that reassigning label tokens at inference time enables seamless task switching without retraining. However, Appendix A.6 reveals an ~18% absolute accuracy drop (62.7% → 44.35%) when control-token mappings are permuted. This indicates the model heavily relies on learned token embeddings and inductive biases rather than purely grounding predictions via prompt descriptions, directly invalidating the stated generality advantage.
- **Missing critical ablation isolates the true contribution of the design:** The paper lacks a baseline fine-tuning the same decoder architectures with a standard linear classification head (pooled or final-token hidden state). Without this, it is impossible to determine whether accuracy gains stem from the atomic token constraint or simply from standard supervised LoRA adaptation. Additionally, the method underperforms its own untuned base model on AG News (81.5% vs 84.0%) and DBpedia (95.0% vs 97.0%) with zero analysis provided.
- **Deterministic argmax eliminates calibration and abstention:** Forcing a single-token prediction removes the model’s natural ability to express low confidence or reject out-of-scope inputs. In safety-critical or open-world classification deployments, the absence of a calibrated thresholding mechanism or explicit "unknown" class is a substantial operational risk that the paper treats as an afterthought.

## Nice-to-Haves
- Benchmark end-to-end throughput (requests/sec under concurrent load) and VRAM memory overhead to contextualize latency gains in batched serving environments.
- Visualize learned atomic token embeddings relative to baseline verbalizers to assess whether control tokens capture meaningful semantic boundaries or act as arbitrary indexers.
- Provide a detailed error analysis/confusion matrices for AG News and DBpedia to diagnose the base model regression.

## Novel Insights
The work effectively highlights a pragmatic but underexplored design axis: collapsing classification into a constrained vocabulary selection problem rather than an open-ended generation or encoder-pooling task. The randomized token-label training strategy is conceptually sound, aiming to force contextual grounding over memorization. However, the sharp performance degradation under unseen token permutations reveals a subtle truth: even with explicit prompt descriptions and randomized mapping, decoder-based LLMs still form strong distributional priors over specific token embeddings during fine-tuning. This suggests that "zero-shot" task switching via token reassignment is not purely a semantic reasoning capability but remains partially tethered to the geometric structure of the refined embedding space, a valuable observation for future work in dynamic classifier adaptation.

## Potentially Missed Related Work
- **Classifier heads on decoder-only architectures:** Recent work on training explicit linear classification heads on decoder-only models (e.g., recent Olmo/Llama instruction-tuning pipelines) serves as a more direct efficiency/accuracy comparison than masked encoder models.
- **Speculative decoding for classification:** Methods that use draft models to accelerate short-sequence generation could alternatively address the multi-token latency problem without requiring vocabulary extension.

## Suggestions
1. **Expand text evaluations to full test splits** with bootstrapped 95% confidence intervals. The 200-sample protocol must be retired to meet standard statistical rigor for benchmarking claims.
2. **Add a standard LoRA + linear classifier head baseline** trained under identical data/epoch/hardware conditions. This will isolate the exact latency-accuracy trade-off introduced by the atomic token constraint.
3. **Investigate and resolve the permutation accuracy drop.** If the goal is true zero-shot remapping, introduce a lightweight calibration step (e.g., cosine-similarity scaling between token embeddings and prompt descriptions) or explicitly reframe this limitation in the main text.
4. **Address the accuracy regression on AG News/DBpedia.** Provide a brief analysis (e.g., class-wise F1 or logit spread) explaining why fine-tuning for single-token output degrades performance on these specific domains compared to the untuned base model.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
