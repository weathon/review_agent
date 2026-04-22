# Imitating the Truth: Attention-aware Truth-Guided Enhancement for Hallucination Mitigation in Large Vision-Language Models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 2, 8

## Abstract
Large Vision-Language Models (LVLMs) achieve impressive multimodal reasoning but remain prone to hallucinations, generating content inconsistent with visual evidence.
Existing mitigation methods often rely on auxiliary modules or coarse decoding-time adjustments, overlooking the fine-grained dynamics that distinguish truthful (real) tokens from hallucinatory ones.
In this paper, we introduce \textbf{AGE (Attention-aware Truth-Guided Enhancement)}, a training-free framework that performs fine-grained, layer-wise interventions guided by attention patterns of real tokens. 
Our analysis reveals that real and hallucinated tokens follow distinct stage-specific attention behaviors, and hallucinations emerge when models fail to reproduce these behaviors.
AGE addresses this by introducing two lightweight interventions: (i) Imitating the image attention, derived from discrepancies between real and hallucinated tokens, and (ii) Imitating the text attention when semantic grounding is required.
Extensive experiments on widely used benchmarks, including COCO Image Captioning, POPE, and MME, demonstrate that AGE consistently mitigates hallucinations across diverse LVLMs such as LLaVA, MiniGPT-4, and mPLUG-Owl2, without additional training or loss of fluency.
Our results highlight that imitating truth-grounded attention dynamics is a simple yet powerful principle to improve the reliability of LVLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper observes layer-wise attention disparities between real and hallucinated tokens across three LVLMs, which motivates a method for calibrating stage-specific attention behaviors using a pre-calculated directional vector on a labeled subset of data.

### Strengths
1. The method is clearly motivated by empirical observations, offering new insights into attention behavior in LVLMs. Comprehensive experiments demonstrate that it effectively mitigates object hallucinations and outperforms prior work.

2. The paper is well written and supported by clear figures.

### Weaknesses
1. My main concern is that this method relies on access to a labeled subset of data to calculate the directional vector that captures the attention correction tendency. This limits its applicability in real-world scenarios where labeled data from the same distribution are not available, and there is no validation showing whether the learned attention behavior remains valid.

2. Related to W1, it is also unclear how the stage-specific settings (line 300–302) are determined in practice, since they seem to be tuned like hyperparameters.

### Questions
1. In Section 3.3 and Figure 3, the statement that “the final reasoning stage (Layers 26–31) exhibits a stable and pronounced positive gap” (lines 204–205) is less convincing. The earlier layers (Layers 0–4) appear to show a more consistent and significant gap.

2. The related work section could be more up to date. most of the papers are from 2024.

Happy to raise score if my concern is addressed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents AGE (Attention-aware Truth-Guided Enhancement), a training-free, decoding-time framework aimed at reducing hallucinations in large vision-language models (LVLMs). It leverages an in-depth analysis of attention patterns distinguishing truthful from hallucinated tokens across layers and modalities, applying targeted, stage-specific interventions to realign attention toward grounded responses. The approach is evaluated on benchmarks including COCO (with CHAIR metric), POPE, and MME, across three LVLMs (LLaVA, MiniGPT-4, mPLUG-Owl2), showing reduced hallucination rates while preserving or enhancing overall performance metrics like BLEU.

### Strengths
1. AGE offers a lightweight, training-free, and model-agnostic solution that can be integrated seamlessly during inference, making it practical for deployment.

2. The experimental setup is comprehensive, covering three diverse LVLMs and multiple benchmarks (COCO/CHAIR, POPE, MME), with results indicating hallucination mitigation alongside maintained or improved generation quality.

### Weaknesses
1. The core idea of using attention calibration or intervention to mitigate hallucinations by aligning with truthful patterns has been explored in several prior works, such as AGLA, which assembles global and local attention to reduce object hallucinations, DeCo, which employs dynamic correction decoding, and DeGF, which uses generative feedback for self-correction. Token-level and layer-wise analyses of hallucinations are also featured in these and related studies. While AGE contributes through finer-grained, stage-specific interventions, the paper does not sufficiently delineate its novelty beyond these approaches and lacks citations or direct comparisons to them.

2. The intervention stages (e.g., layer 20 for text, 30–31 for vision) are determined empirically (Appendix A.5), but the paper provides limited guidance for adapting these to novel architectures or LVLM variants. It notes that early-layer interventions severely degrade BLEU scores yet fails to propose a systematic methodology for identifying optimal stages or assessing transferability to deeper or wider models.

3. The formulation of δ as a "universal" intervention vector is empirically motivated and tuned, but lacks theoretical grounding on its effectiveness, particularly under distribution shifts between calibration and test data.

4. Notation in equations for attention weights exhibits inconsistencies, reported baseline metrics appear inconsistent with those in the original papers, potentially leading to unfair comparisons.

5. The method relies on numerous empirically set hyperparameters, with insufficient analysis of their robustness or generalizability across different settings.

### Questions
1. Given the similarities to prior works like AGLA, DeCo, and DeGF, could you clarify the novel contributions of AGE and provide comparisons (both qualitative and quantitative) to these methods? Additionally, please address discrepancies in baseline metrics compared to their original reports to ensure fair benchmarking.

2. Could you provide examples of failure cases for AGE, including their common characteristics (e.g., specific image types, prompt complexities, or distributional shifts)? This would help assess the method's limitations more concretely.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The goal of the paper is to address the hallucination in LVLMs. The paper conducts the layer-wise analysis by introducing the concept of Average Attention Sum and computing the difference between non-hallucinated and hallucinated ones. The analysis shows the patterns of $Diff^{l}_{image}$, which is positive at the last stage. The paper proposes an attention direction to imitate the attention pattern, which is used during inference. The paper validates the proposed method on hallucination benchmarks and conducts an ablation study, including sample size and variants of AGE.

### Strengths
1. The inference cost is lower than that of other methods, such as VCD. Once the attention direction is computed to imitate attention, inference requires only a simple shift operation.
2. Latent steering is widely adopted to mitigate hallucinations. But attention steering is a novel approach.

### Weaknesses
1. There is a logical gap in the hypothesis of the layer-wise analysis.
- Both MiniGPT-4 and mPLUG-Owl2 exhibit a consistently positive difference across all layers. Given this observation, is it valid to imitate Early and Late attentions for MiniGPT-4 and mPLUG-Owl2?
- In contrast, the difference in text is negative in general. This suggests that the corresponding $\lambda_t$ should be negative to reflect this. However, the paper assigns the positive value to $\lambda_t$.
2. The way to select the final hyperparameters, including the choice of layers and the scaling factor, is not clear. Details on the hyperparameter tuning method would be helpful.
3. The proposed approach steers the attention after computing the attention direction, which is similar to the latent steering methods [VTI, Nullu]. A detailed discussion and comparison are needed to validate the advantages of the proposed method.

### Questions
1. In Sec. 3.2, do humans manually annotate tokens as 'real/hallucinated', or is it done automatically?
2. In Eq. (3), the left term is divided by $I^{(i)}_{hall}$. Is it correct?
3. What common characteristics between MiniGPT and mPLUG-Owl lead to the positive $Diff^{l}_{image}$ for all layers?
4. After applying the proposed method to LVLMs, are the general task capabilities preserved?
5. After steering the attention, does the total attention still sum to 1?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes AGE, a training-free, decoding-time framework that mitigates hallucinations in LVLMs by imitating the stage-specific attention patterns observed in truthful tokens. From a token- and layer-wise analysis, the authors find that real tokens systematically allocate more visual attention in late layers, while some models (e.g., LLaVA-1.5) also require stronger text attention mid-stage. AGE injects a directional visual correction vector δ (estimated from a small set of COCO samples) into late-stage layers and applies self-multiplicative text attention in the mid-stage when model-specific analysis indicates it helps. Across COCO/CHAIR, POPE, and MME, AGE outperforms recent SOTA methods while maintaining or slightly improving BLEU, and requires no additional training. Ablations show the vector-guided visual intervention (AGEI) is substantially more effective than coarse self-scaling, and that targeted, stage-aware interventions beat blanket modifications. The method is simple, interpretable at the attention level, and appears robust to the number of samples used to estimate δ.

### Strengths
- A precise, stage-aware recipe—directional visual correction in late layers + optional mid-stage text boost—plugs into existing LVLMs without retraining or extra modules.
- The token-/layer-wise diagnosis directly motivates the interventions; ablations (SMA vs. AGEI, AGEM/AGEL variants) convincingly show why direction + stage targeting matter.
- Strong gains on CHAIR/POPE/MME across LLaVA-1.5, MiniGPT-4, mPLUG-Owl2, with BLEU preserved; δ estimated from M=10 samples yet still competitive—good efficiency story.

### Weaknesses
- The approach assumes attention patterns of “real” tokens are the right supervisory signal; while effective, this remains heuristic and may not universally reflect grounding quality across tasks or architectures.
- Fixed layer choices (20, 30–31) and large λᵥ=100 / λₜ=3 suggest potential sensitivity; it’s unclear how easily settings transfer to unseen models, domains (e.g., medical, diagram, chart), or alternative decoders.
 -“Real vs. hallucinated” token labeling depends on object annotations (COCO); generalization to open-ended VQA without dense GT, long-context reasoning, or video remains untested, and the latency/throughput overhead of per-step intervention isn’t reported.

### Questions
- How sensitive are results to the layer indices and λᵥ/λₜ? Could you automatically pick stages (e.g., by monitoring live attention gaps) to avoid manual choices?
- δ is computed from COCO captions—how does AGE transfer to domain-shifted settings (e.g., medical images, charts) or text-heavy VQA where “real vs. hall” labels are less object-centric?
- Have you considered online estimation of δ during decoding (or per-image calibration) to reduce dependence on an offline sample pool and improve robustness to image/content diversity?

### Soundness
4

### Presentation
4

### Contribution
4
