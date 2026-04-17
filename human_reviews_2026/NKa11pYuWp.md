# Look&Learn: Where to Look? Bridging Perception and Grounding Gap in Vision-Language Models

- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Vision-Language Models (VLMs) excel at perception and reasoning, yet their ability to precisely ground visual concepts remains limited. 
A common remedy is to attach a segmentation decoder (e.g., SAM) to VLMs, which indeed equips them with segmentation capabilities but simultaneously erodes their understanding performance. 
We call this the understanding–grounding gap, where enhancing grounding comes at the expense of perception. In this work, we propose Look&Learn, a segmentation-free grounding framework that closes this gap by enabling the LLM itself to act as a segmentor. At the core is Where to Look?, a lightweight, plug-and-play loss applied directly to the attention heatmaps of arbitrary VLMs. Rather than relying on an additional decoder, our loss guides the model to focus on relevant image regions during text generation, effectively leveraging the LLM’s own attention as high-quality segmentation masks. Your native LLM is your segmentor. Look&Learn is efficient and versatile: it can be integrated seamlessly into any VLM architecture and applied during pre-training, fine-tuning, or even post-training. To enable joint training of understanding and grounding, we further construct a scalable pseudo-grounding dataset that aligns textual and visual entities using state-of-the-art grounding models combined with LLM-based adjudication. Extensive experiments demonstrate that Look&Learn consistently improves grounding performance without sacrificing perception. 
On LLaVA, it delivers +3% gains on grounding benchmarks while also boosting Understanding by 1–2%. On GLaMM, our loss further provides +1% improvements across three grounding datasets. These results confirm our central hypothesis: grounding and understanding are not competing objectives but can be unified within a single model, without explicit segmentation supervision. Our work paves the way for more interpretable, efficient, and multimodal-aware VLMs, and more broadly, opens the door to using LLMs themselves as segmentors.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Look&Learn, a novel segmentation-free grounding framework for VLMs that unifies understanding and grounding. The core idea is “LLM as Segmentor”, where the large language model itself generates usable attention-based segmentation masks, without relying on external segmentation decoders (e.g., SAM). 
The key component, Where-to-Look loss, aligns the model’s attention maps with pseudo-ground-truth masks during text generation. 
The authors also develop a scalable pseudo-grounding dataset built from multiple grounding models with LLM adjudication. 
The approach yields consistent improvements on grounding benchmarks while enhancing reasoning performance on VQA-style benchmarks.

### Strengths
1. The paper reframes visual grounding as an intrinsic property of VLM attention rather than an external task, which is conceptually appealing. 
2. Evaluations cover both understanding and grounding benchmarks. The results show consistent gains over LLaVA and GLaMM without adding decoders or segmentation heads.
3. The work enhances interpretability by making attention maps explicitly align with visual entities, which is a step toward explainable VLMs.

### Weaknesses
1. Although the pseudo-mask generation pipeline is robust, it still relies on existing grounding models and LLM adjudication. The quality of supervision may limit scalability or introduce biases from upstream models.
2. While qualitative results are reasonable, this paper heavily emphasizes empirical evidence but lacks a quantitative analysis of pseudo label quality.

### Questions
How consistent are attention-based segmentations across different prompts describing the same object? Have you measured stability or robustness to paraphrasing?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the "understanding-grounding gap" in Vision-Language Models (VLMs), where enhancing grounding capabilities, typically by adding external segmentation decoders, often leads to a significant degradation in the model's core perception and reasoning performance. To bridge this gap, the authors propose Look&Learn, a segmentation-free framework centered around a novel loss function called "Where-to-Look". This lightweight, plug-and-play loss directly supervises the VLM's internal attention maps, guiding the model to focus on relevant visual regions during text generation. The central idea is to enable the LLM to act as its own segmentor, using its refined attention maps as high-quality grounding masks, thus obviating the need for cumbersome external modules. To facilitate this, the paper also introduces a scalable pipeline for creating a large-scale pseudo-grounding dataset. Experiments show that this approach successfully improves both grounding and understanding metrics on top of strong baselines like LLaVA and GLaMM.

### Strengths
1.	The paper introduces a scalable pipeline for creating a large-scale pseudo-grounding dataset. 
2.	The evaluation is thorough, assessing performance on both understanding (VQA benchmarks) and grounding (referring segmentation) benchmarks. The inclusion of detailed ablation studies provides strong support for the proposed design choices and validates the effectiveness of the approach. The consistent gains across multiple models and tasks are convincing.

### Weaknesses
1.	Novelty and Positioning: The core idea of leveraging attention maps from a VLM to generate segmentation masks is not entirely new. Several prior works have explored similar directions. 
2.	Marginal Performance Gains and Inconsistent Narrative:
o	The performance improvements on core understanding benchmarks in Table 3, are modest when compared to the LLaVA baseline. This raises questions about the practical impact of the proposed method on the model's general reasoning capabilities.
o	There is a tension between the paper's central claim of a "native segmentation-free" approach and the reported results. In several experiments (e.g., Table 2c and Table 3), the variant that uses a special [SEG] token (Ours-Seg) achieves performance comparable or even superior to the purely entity-based approach (Ours-Ents). This undermines the argument that special tokens are unnecessary and complicates the paper's primary narrative.
3.	Clarity of Presentation:
o	The experimental section (Section 4) lacks a clear and logical flow. It is difficult for the reader to identify the main experimental setup, distinguish it from ablations, and synthesize the key takeaways. The structure makes it challenging to follow the authors' line of reasoning and verify their central claims.
o	The visual presentation quality is subpar. Figures and tables are often cluttered and poorly laid out (e.g., the proximity of Figure 2 and Table 2). The diagrams in Figures 3 and 4 are aesthetically unrefined and complex, hindering comprehension. Furthermore, crucial details are missing; for instance, the meaning of the "10:1" ratio in Table 2(c) is never explained.
o	There is a minor but notable inconsistency in terminology between the title ("perception and grounding") and the abstract ("understanding and grounding"), which could be easily harmonized.

### Questions
1.	Could you please elaborate on the role of the [SEG] token? Given that the Ours-Seg variant performs so well, does this suggest that a dedicated, explicit mechanism for grounding is still beneficial, challenging the core premise of a purely "native" attention-based solution? How do you reconcile these findings with your main claim?
2.	In Table 2(c), you compare loss weightings like "Ents (10:1)" and "SEG (10:1)". Could you please clarify what this ratio represents? Is it the weighting between the standard cross-entropy loss and your proposed "Where-to-Look" loss? This information is essential for interpreting the results.
3.	Considering the existing literature on using attention for grounding, could you more explicitly pinpoint what makes your loss formulation or overall framework novel?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Look&Learn, a framework that solves the "understanding-grounding gap" in Vision-Language Models (VLMs), where adding segmentation decoders to improve grounding typically harms perception. Its core is a lightweight *Where to Look?* loss function that guides the VLM's own attention to relevant image regions during text generation. This "LLM as Segmentor" approach, trained with a scalable pseudo-data pipeline, successfully improves both grounding performance (e.g., +3%) and reasoning (~1-2%) simultaneously, demonstrating that the two tasks can be unified.

### Strengths
1. The paper proposes an intuitive and effective method to enhance VLM perception and reasoning by guiding the model's attention maps to align with task-relevant visual entities. This approach of correcting the LLM's "focus" not only improves performance but also offers a promising path toward better model interpretability.

2. The generalizability of the proposed "Where to Look?" loss is well-supported by solid experiments. The method shows consistent benefits when applied to different base models (e.g., LLaVA and GLaMM) and in various training paradigms, including full fine-tuning and as a lightweight post-training (PT) step. This demonstrates its robustness and practicality.

3. The paper introduces a robust and scalable pseudo-mask generation pipeline. This pipeline, which leverages consensus from multiple grounding models (GLaMM, OWL-v2, etc.) and LLM-based adjudication (using GPT-4o), is a valuable contribution in itself and could be adapted to enhance other multimodal datasets.

### Weaknesses
1. Frozen Visual Encoder: A significant limitation is the decision to keep the visual encoder frozen during fine-tuning. As noted in prior work [ref1], ViT encoders can aggregate crucial visual information in "register" or background patches, not just within the primary object's region. By forcing the LLM's attention onto specific regions, without allowing the vision encoder to adapt, the model might be constrained in its ability to acquire all necessary visual information. This could be a contributing factor to the relatively modest performance gains (e.g., ~1-2% on VQA benchmarks ). It is encouraged to provide additional analysis on training LLaVA with visual encoders unfrozen.

2. Text-Rich Images: The pseudo-mask generation pipeline appears not suited for text-rich images. The grounding models employed (e.g., GLaMM, OWL-v2, Grounding-DINO) are not specialized for text segmentation. This limitation is critical as OCR is a key VLM capability. The observed performance drop on the TextQA benchmark (57.7 for Ours-Ents vs. 58.0 for LLaVA) may be direct evidence of this weakness.

3. A minor point on presentation is that the main paper is quite dense. The authors could improve readability by moving some of the more detailed experimental ablations (e.g., parts of Table 2) or pipeline details to the appendix, allowing for a clearer focus on the core concepts in the main body.

[ref1] Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski. Vision Transformers Need Registers. ICLR 2024.

### Questions
In addition to the Weaknesses mentioned above, there are some questions to be clarified:

1. OCR-VQA Statistics in Table 1: The statistics for the OCR-VQA dataset in Table 1 are not well-explained. The table reports 100% IoU Acc. and 0% Judge Acc. This 100% accuracy implies a ground-truth source was used. However, the caption states that for datasets with existing bounding boxes (highlighted in gray), SAM was used directly, and OCR-VQA is not highlighted. Given the pipeline's aforementioned weakness in text segmentation, how did it achieve 100% accuracy on this dataset? Please clarify how the masks for OCR-VQA were generated.

2. Contradiction in Table 2(b): There appears to be a direct contradiction between the text and Table 2(b) regarding visual-token aggregation across the K dimension. The text states that "supervising **all** visual tokens yields the strongest grounding signal". However, Table 2(b) shows that the "All" method (47.8 VQA-v2 score) performs worst, while the "Avg." method (50.0 VQA-v2 score) performs best. Please clarify this discrepancy.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Look&Learn, an attention-guided loss that supervises cross-attention maps in vision-language models (VLMs) to align with visual object regions, enabling the LLM to act as its own segmentor. The work aims to address the challenge that grounding and segmentation modules in current VLMs (e.g., SAM-attached architectures) often degrade perception and reasoning abilities, creating a trade-off between understanding and grounding. To mitigate this, the authors design a loss function that aligns cross-attention maps with pseudo-grounding masks without altering the model architecture or introducing decoder modules. They evaluate their approach by integrating it into LLaVA and report modest improvements on visual question answering and referring expression segmentation tasks such as RefCOCO and GQA. The method is conceptually simple and compatible with various VLM architectures, but the evaluation is limited to an outdated model (LLaVA) and relies on a complex pseudo-labeling pipeline involving multiple vision-language models and GPT-4.

### Strengths
### Conceptually simple approach

The method adds a straightforward auxiliary loss to encourage the model’s attention weights to match given segmentation masks or regions. This idea is easy to plug into an existing model (LLaVA in this case). The approach does not require architectural changes, only an extra training signal, and it reportedly yields slight improvements on LLaVA’s VQA and RefCOCO performance.

### Weaknesses
### Limited Novelty

The core idea – supervising or guiding attention maps to align with known relevant regions – is not novel. Several prior works have already explored aligning model attention with human or ground-truth signals. For example, HINT encouraged vision-language models to focus on the same image regions as humans by optimizing alignment between human attention maps and model importance weights. Likewise, earlier VQA and captioning research introduced explicit attention supervision (e.g. using human-annotated attention maps) and showed that guiding attention can improve performance. In open-vocabulary segmentation, methods like PnP-OVSS and MaskCLIP++ leverage the cross-attention maps of pretrained vision-language models directly to obtain segmentation masks without additional training. Furthermore, end-to-end grounding models (e.g. MDETR and the Referring Transformer) already handle detection/segmentation and grounding in a unified framework. MDETR, for instance, integrates text and image features in a transformer and achieved state-of-the-art results on phrase grounding and referring expression benchmarks. The Referring Transformer similarly uses a one-stage transformer to directly predict boxes and masks for referred expressions, outperforming previous two-stage methods by a large margin. Given this rich prior art, the submission does not clearly explain what is new beyond applying an attention alignment loss in the LLaVA training pipeline. The authors cite the goal of bridging perception and grounding, but the technique itself appears to be a straightforward amalgamation of known ideas (attention supervision using segmentation cues). The lack of a compelling novelty claim is a major weakness.

### Insufficient Experiments and Baselines

The evaluation is narrow and leaves questions about the method’s general usefulness. All experiments are conducted on LLaVA, which is an early vision-language model known to be weaker than more recent models (e.g. Qwen-VL, Emu2, Kosmos-2, or MM1). The reported gains on LLaVA’s VQA and RefCOCO tasks are relatively small. It’s uncertain whether the proposed attention-guidance would yield similar improvements on stronger baselines or modern multimodal models. The paper does not compare against any state-of-the-art models that are already strong at grounding or segmentation. For instance, SEEM/SEEM++ (specialized segmentation models) or a larger model like Qwen-VL-Max are not included in comparisons, even though those models likely excel at the tasks in question. Without testing on or against these, it’s hard to gauge the significance of the contribution. In essence, the results show a minor tweak improving a single (weaker) model, but do not demonstrate closing the gap with the best-in-class approaches. This limited scope weakens the paper’s empirical evidence.

### Heavy and Impractical Pipeline

The method’s reliance on a complex pseudo-label generation pipeline raises concerns. To generate training targets for the attention supervision, the authors use multiple large models and even GPT-4. This is a heavyweight and elaborate process, which might be impractical for others to reproduce or deploy. The paper doesn’t detail the cost or latency of this pipeline, but one can infer it involves significant computation and coordination (running advanced vision models and a large language model to annotate images with segmentation masks or region descriptions). Such a pipeline could hinder reproducibility and scalability – it may be difficult for researchers without extensive resources to recreate the exact training setup. The need for GPT-4 (which is a proprietary model) in the loop is especially problematic for open-source or academic replication. Overall, even if the idea of attention supervision is sound, the way it’s implemented here seems cumbersome and not easily generalizable to other settings.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
