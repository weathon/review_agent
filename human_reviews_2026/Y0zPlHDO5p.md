# AntigenLM: Structure-Aware DNA Language Modeling for Influenza

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 6

## Abstract
Language models have advanced sequence analysis, yet DNA foundation models often lag behind task-specific methods for unclear reasons. We present AntigenLM, a generative DNA language model pretrained on influenza genomes with intact, aligned functional units. This structure-aware pretraining enables AntigenLM to capture evolutionary constraints and generalize across tasks. Fine-tuned on time-series hemagglutinin (HA) and neuraminidase (NA) sequences, AntigenLM accurately forecasts future antigenic variants across regions and subtypes, including those unseen during training, outperforming phylogenetic and evolution-based models. It also achieves near-perfect subtype classification. Ablation studies show that disrupting genomic structure through fragmentation or shuffling severely degrades performance, revealing the importance of preserving functional-unit integrity in DNA language modeling. AntigenLM thus provides both a powerful framework for antigen evolution prediction and a general principle for building biologically grounded DNA foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces AntigenLM, a structure-aware DNA language model designed to improve influenza forecasting. It identifies a key limitation: task-specific evolutionary models (like beth-1) often treat mutations as independent events, while general-purpose DNA foundation models are often trained on fragmented, unaligned data, causing them to miss biological context.

To address this, AntigenLM is pretrained on over 54,000 complete influenza A genomes, with each training sample consisting of all eight gene segments concatenated in a fixed order. The hypothesis is that preserving these intact functional units allows the model to learn co-evolutionary dependencies across the entire genome. The model itself is a 6-layer GPT-2-style Transformer with a 13k-token context window to accommodate full genomes.

The evaluation shows that AntigenLM outperforms conventional baselines (beth-1, LBI) in next-season forecasting, reducing amino acid mismatches by over 50% relative to beth-1 on key tasks. Ablation studies show that this Full-genome model outperforms two variants: one trained on Incomplete-genome (randomly cropped) sequences and another on isolated Segment-wise data. The model also shows strong generalization to unseen subtypes (H7N9) and geographies.

### Strengths
**(S1)** A clear and important research problem. This work clearly demonstrates the limitations of both site-based evolutionary models and general-purpose genomic foundation models. The hypothesis that a DNA LM for viruses must be pretrained on aligned, intact functional units is intuitive. These arguments thus motivates the need for a new approach. 

**(S2)** The integration of biological structure in method. AntigenLM enforces gene- and segment-level boundaries during pre-training and fine-tuning. This offers a compelling biological motivation for functional-unit-aware modeling.

**(S3)** Complete experiment setup. The breadth of the experiments is excellent. It spans several axes such as short-term and long-term sequence forecasting, cross-geographical generalization, rare subtype transfer, and explicit subtype classification. This builds a robust case for the model's capabilities and shows practical utility on real-world scenarios. I particularly respect that some practical considerations like replicate vaccine decision timelines and public health applicability are thoughtfully integrated in experiment framing and data curation.

**(S4)** Interpretability and Visualization. The use of t-SNE visualizations in Fig. 5 and confusion matrices in Fig. 3 offers great interpretability to learned representations. It clearly shows that AntigenLM’s embeddings can yield more distinct and biologically meaningful subtype clusters compared to baselines.

**(S5)** Rigorous Ablation Studies. I respect the design of ablation studies. The comparison of the Full-genome model against its Incomplete-genome and Segment-wise variants counterparts provides a direct evidence of the value of structure preservation. Especially, the performance drop of the ablated models (e.g., perplexity jumping, and near-perfect classification vs subtype confusion) provides a strong support for this paper’s main claim.

### Weaknesses
**(W1)** Severe formatting errors (about title). The most immediate and severe issue in my view is that the manuscript is improperly compiled and appears to be in a draft state. The main title of the paper is a placeholder. The actual title, "AntigenLM: Structure-Aware DNA Language Modeling for Influenza," is incorrectly placed as a line of text above the Abstract section. This should be a major flaw for a submission to a top-tier conference. However, given that ICLR permits revisions and iterative author-reviewer discussions, I will reserve final judgment on this shortcoming. I strongly encourage the authors to provide a corrected, complete manuscript in the rebuttal phase.

**(W2)** Incomplete literature review especially for some highly relevant previous work. I encourage the authors to include related discussions in the revised manuscript.

- [1] Learning the Language of Viral Evolution and Escape, Science 2021. This work develops a viral sequence LM directly tied to immune escape and evolutionary prediction for influenza. It is almost exactly the stated target domain here. It helps show what is actually novel in AntigenLM relative to contemporaneous, domain-specialized LMs.
- [2] VQDNA: Unleashing the Power of Vector Quantization for Multi-Species Genomic Sequence Modeling, ICML 2024. This work studies learnable vocabularies which has shown a path beyond fixed tokenizers like BPE and k-mer. Its motivation is similar to AntigenLM which also critiques hand-crafted k-mer and fragmented approaches and offer a learned-representation instead, differing from AntigenLM's explicit functional-unit.
- [3] Dinucleotide Evolutionary Dynamics in Influenza A Virus, Virus Evolution 2019. This work examines sequence-level constraints and evolution, providing insights for the biological basis behind AntigenLM's structural modeling.

**(W3)** Comparisons to DNA foundation model. It is stated that general-purpose DNA foundation models underperform because they are trained on fragmented data. However, none of these models (e.g., ESM, DNABERT, HyenaDNA) are included as baselines for direct comparison. IMHO, the Incomplete-genome model may not be representative enough for these SOTA models, which use different tokenization and pre-training objectives. I encourage the authors to fine-tune at least one genomic LM like DNABERT on their exact downstream tasks and report its performance to support the claim.

**(W4)** In visualization, the interpretive narrative is a bit thin. For example, Fig. 2 showcases reduced mismatch for the AntigenLM variants. But neither the main text nor the caption interprets which features (e.g., epitope conservation vs. overall sequence) drive the model’s superior forecasting. Panel D (geographic generalization) would benefit from a breakdown by outlier scenarios (as mentioned, H3N2-NA had worse US transfer—why?). Similarly, t-SNE visualization in Fig. 5 suggests tighter clusters for the full-genome variant, but quantitative cluster metrics (e.g., inter/intra-class distances) could substantiate claims.

**(W5)** Some unclear method details. 

- In Sec. 3.2, while the core causal LM objective is standard, details are missing around the batch sampling strategy, positional encoding modifications for highly variable-length and segmented inputs, and the explicit formulation of the dual-task loss function (especially the weight or scheduling between the two heads).
- Equations in Sec. 4.2 does not clarify if nucleotide padding, masking, or non-coding regions are handled during autoregressive loss computation. These are real concerns in genome application. How are sentinels encoded and differentiated from standard tokens? Are transitions across gene boundaries soft or hard? The method should specify if and how the model penalizes boundary crossing in generation mode.
- The paper states that "All pre-training sequences were aligned by segment". In my view, this could mean (a) the 8 segments for a single virus were simply concatenated in a fixed order, or (b) a full Multiple Sequence Alignment (MSA) was performed across all 54,000 genomes for each segment before concatenation. If (b), this is a massive, non-trivial step, and the method for handling gap tokens is missing. I encourage the authors to clarify this in rebuttal. If it is simple concatenation, please replace the word "aligned" with "concatenated in a fixed order" for precision. If it was an MSA, the full method (tool, gap-token handling) is essential for reproducibility.

**(W6) Minor Points**:

- Model efficiency is described in texts in Sec. 3.3), but no specific metrics such as runtime, FLOPs, or GPU memory footprint.
- The distinction between segment-wise and incomplete-genome variants in Fig. 4 could be more intuitively clarified early on for readers less familiar with genomic ML.

### Questions
Most of my major concerns and related recommendations have been stated in the Weaknesses section. I recommend focusing the efforts on addressing those points, as they are critical for strengthening the manuscript in the rebuttal stage.

The following are more specific, minor questions to help the authors think more deeply about certain design choices and experiment setups, which might be helpful for this and future work:

**(Q1)** In Sec. 3.2, how does the model encode sentinel tokens, and how are transitions across gene boundaries handled during both pretraining and decoding? What happens if a generation crosses into a functionally distinct segment? Does this terminate generation, penalize the loss, or trigger any corrective mechanism?

**(Q2)** Beyond t-SNE visualization, are there any more quantitative cluster separation scores or biological validation of embedding space distinctions, especially for rare subtypes?

**(Q3)** Are improvements in Fig. 2 and related tables statistically significant? Is it possible to quantify confidence in reported gains? From example, utilizing p-values or bootstrap intervals.


---
## Justifications:

In summary, this paper presents a promising and intuitive pretraining method for viral genomics. The core idea is strong, and the ablation studies provide clear support for it.

However, the current manuscript seems to be not yet complete and with severe issues. The claims of superiority over other foundation models are not supported by direct comparisons, and the claims about learning co-evolution are unproven. IMHO, these are not minor issues but strike at the overall quality of the paper.

Therefore, I cannot recommend acceptance at this stage. I would be glad to raise my rating if thoughtful responses and improvements are provided in the rebuttal stage. I am also open to follow-up discussions with the authors to help further strengthen this work.

I hope these comments help my fellow reviewers and ACs understand the basis of my recommendation.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The influenza virus evolves continuously to evade human adaptive immunity, leading to seasonal epidemics. As a result, influenza vaccine strains must be updated each year to maintain their effectiveness for the upcoming flu season. The paper introduces AntigenLM, an autoregressive DNA language model pretrained on aligned influenza genomes. It robustly captures evolutionary constraints and transfers effectively to the generative tasks of strain forecasting for upcoming influenza seasons. Furthermore, to support the pretraining on aligned genome data, the paper showed that training on unaligned or fragmented gene sequences degrades performance. AntigenLM was compared to an evolutionary model, beth-1, and a tree-based predictor, LBI, on forecasting tasks, where it showed superior forecasting performance on most of the influenza genome segments.

### Strengths
- The paper presented a sound way of collecting and preprocessing the influenza genomes, taking care of gene segments alignment and functional unit preservation, which led to an efficient autoregressive model pretraining. Additionally, it has been shown that overlooking alignment and functional unit preservation can lead to performance degradation.
- The proposed model improved influenza forecasting over evolutionary models and tree-based predictors. The model achieved lower genetic mismatches between predicted and observed consensus amino-acid sequences.
- The paper showed geographic and cross-subtype generalization. Specifically, the model was tested on held-out U.S. sequences during finetuning. Additionally, unlike baseline models, the proposed model is able to forecast minor-subtype sequences. The minor subtype sequences were present in pretraining and finetuning, but in much lower amounts (<5% and <1% in pretraining and finetuning corpora, respectively).
- Even though the paper lacks ML novelty, I think the work was technically done well. The data preparation is sound, the model pretraining and finetuning were done effectively, resulting in outperforming the baseline models.
- The authors promised to release code, documentation and data retrieval guidance subject to acceptance.

### Weaknesses
- The presentation of the paper requires improvement. The overall writing style and, in particular, the clarity of explanations are of low quality. While the pretraining procedure of the model is well described, the fine-tuning section is poorly explained. The fine-tuning process for each specific task should be described in greater detail. I recommend illustrating the fine-tuning procedures with clear indications of the prompts and the regions being forecasted. The current Figure 1B does not contribute to clarity.
- Although segment-wise pretraining is a reasonable approach, the incomplete-genome pretraining with random cropping as a baseline model appears conceptually weak. It would be more meaningful to pretrain on incomplete genomes while excluding entire segments that are not used during fine-tuning and forecasting, retaining only the complete HA and NA segments in their correct order. This would demonstrate whether information from other segments contributes meaningfully, or if comparable performance can be achieved using only the properly aligned HA/NA segments.
- In the segment-wise pretraining, all segments—including those not used during fine-tuning and inference—are utilized. The non-HA/NA segments are likely less informative. The authors should elaborate on their choice of baseline models (segment-wise pretraining and incomplete-genome pretraining) and clarify why these represent sound alternative approaches. At present, the baseline models appear to have been selected post hoc, possibly after the development of AntigenLM, merely to provide comparison points.
- Epitopes are specific regions within influenza HA/NA segments that are targeted by the host immune system. While AntigenLM outperforms beth-1 and LBI on HA epitopes, the authors do not discuss the superior performance of beth-1 on NA epitopes. This omission should be addressed.

**Minor comments:**
- There are typos in Fig. 1B. "Finetuneing" should be written as "finetuning."
- In Fig. 1B, the heads (classification and LM) blocks and the captions do not align with the text lines 134 -- 138.

### Questions
1. Please comment and elaborate on the choice of the baseline pretraining methods. It's important to include this in the paper.
2. Please comment on the superior performance of beth-1 on NA epitopes. It's important to include this in the paper as well.
3. How exactly was fine-tuning for next month and next season done? During inference, what was a prompt? Could you elaborate on this in the paper and illustrate, if possible?

### Soundness
3

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
This paper presents AntigenLM, a transformer-based DNA language model designed to predict how influenza viruses evolve. Instead of treating genomes as random strings of nucleotides, the model keeps the natural structure of the virus—eight gene segments joined in the right order—so it can learn both local and global relationships. The authors pretrain AntigenLM on more than 50,000 influenza A genomes and then fine-tune it to forecast future HA and NA sequences, which are key for vaccine design. They compare it with existing evolutionary and sequence models like beth-1 and show that AntigenLM gives lower amino-acid mismatches and better subtype classification. Ablation studies also suggest that keeping the full genome intact during pretraining matters for accuracy. Overall, it’s a clear attempt to bring structure awareness into DNA foundation models for real biological forecasting.

### Strengths
The idea of preserving full functional units during pretraining feels biologically meaningful and easy to justify. It makes sense that genomes should be learned as whole systems, not chopped-up pieces. The model is compact and computationally reasonable, which makes the method more accessible to other labs. The overall contribution to treating DNA as structured language is simple but potentially important for future genomic foundation models.

### Weaknesses
1. The paper claims a big jump over existing models, but I’m not fully convinced it’s a fair comparison. The baselines feel a bit old — why not include newer genomic LMs like HyenaDNA or TITAN? That would make the improvement more believable.

2. Figure 1 is visually appealing and central to understanding the paper, but it could better communicate data imbalance, temporal coverage, and architectural specifics. A supplementary figure showing training sample counts per region/year and a more explicit schematic of sentinel-token flow would improve interpretability.

3. The functional-unit aware idea sounds interesting, but it feels mostly like a data-organization trick, not a new modeling method. Could similar results be achieved just by using longer context windows?

4. The results in Figure 2 look visually impressive, but some doubts arise. The reported 70% mismatch reduction seems unusually large, given that beth-1 already performs well on H1N1/H3N2; could these differences stem from dataset splits or evaluation metrics rather than genuine model ability? Error bars are wide and overlap in several cases, yet statistical significance isn’t shown. It’s also unclear whether test sets are truly unseen in time (to avoid leakage) or whether the consensus targets overly smooth the diversity of real viral populations.

5. H1N1 and H3N2 dominate the dataset, while minor subtypes like H7N9 have extremely small test sets. The paper interprets success on H7N9 as cross-subtype generalization, but with <50 examples, this could just reflect memorization or noise.

6. The paper doesn't seem to have been revised well. The title of the paper is not written in the correct place. Figure 2 caption includes the typographical symbol “¿70%” instead of a proper “~70%” or “≈70%.” Appears to be a stray inverted question mark (encoding artifact) at p.7, caption line. Add proper appendix separation and consistent figure numbering. 

7. The model architecture (6 layers, 384 dim) is very small compared to typical DNA LMs. Are we sure it isn’t underpowered — or that the gain isn’t just from better preprocessing?

8. The authors say the model “scales to population-level datasets,” but training was done on ~54 k genomes. That’s not really population scale; what happens at millions?

9. The “near-perfect” subtype classification (99.8 F1) seems almost too good to be true. Was the classifier evaluated on strictly unseen subtypes or just held-out samples of known ones?

### Questions
1. The t-SNE plots in the appendix look clean, but t-SNE always makes clusters look nice. Could we see a quantitative metric instead of a pretty visualization?

2. There’s no comparison to protein-level models (e.g., ESM-2). Would those perform equally well or even better for HA/NA sequence forecasting?

3. In the results, H3N2-NA seems slightly worse under U.S. testing. Any insight why that happens? Could it reveal overfitting to regional sequence biases?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors developed a new DNA language model, AntigenLM, for predicting the evolution of influenza genomes. This is the first work on applying DNA LM to this task to my knowledge. The training data appear carefully constructed, and the evaluation tasks were well designed. It showed notable performance gains over traditional methods. Although I have some concerns regarding excluding protein language models from the benchmark, I overall recommend a weak acceptance of this work.

### Strengths
1. It is a novel idea to use a DNA language model for predicting viral evolution, which is an important problem for public health.
2. There are very few DNA language models for eukaryotic viruses, and this work has filled this gap.
3. The training data is carefully curated and documented in detail. It could be a useful resource to the community.
4. The evaluation tasks are well-designed, and data splitting was done thoughtfully.

### Weaknesses
1. I am not sure if the authors covered all the appropriate baselines. I am not super familiar with the field, but there seem to be many works on using protein language models to predict viral evolution. For example, https://www.science.org/doi/10.1126/science.abd7331, https://www.nature.com/articles/s41392-024-02066-x, https://www.biorxiv.org/content/10.1101/2025.08.04.668423v1. Although this model is trained at the DNA level, the evaluations seem mostly at the amino acid level. The authors should justify excluding those models from the comparisons.
2. Following up on the previous point, the authors should also demonstrate the necessity or advantage of modeling at the DNA level rather than the protein level.
3. The description of the model architecture is obscure. For example, does it have 6 heads per layer or 1 head per layer? A more detailed diagram of the architecture might be helpful.

### Questions
1. Is the sentinel token not used during pretraining?
2. I am a bit confused about how the notion of time is built into the model. For example, in the next-month prediction, is the model just fine-tuned on sequences at month t, and then used to generate sequences supposedly for month t+1 without any constraint or guidance?

### Soundness
2

### Presentation
3

### Contribution
3
