# HAD: Hybrid Architecture Distillation for Bridging Large-Transformer Knowledge into Compact Genomic Models

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Inspired by the great success of Masked Language Modeling (MLM) in the natural language domain, the paradigm of self-supervised pre-training and downstream fine-tuning has also achieved remarkable progress in the field of genomic sequence modeling.
However, existing research often either relies on scaling up pre-training data and parameters, which brings a heavy computational burden, or lacks a systematic method to avoid the loss of prior information with compact architectures. 
In this work, we propose a **H**ybrid **A**rchitecture **D**istillation (**HAD**) approach, leveraging both distillation and reconstruction tasks for more efficient and effective pre-training.
Specifically, we employ the NTv2-500M as the teacher model and devise a grouped masking strategy to align the feature embeddings of visible tokens while concurrently reconstructing the invisible tokens during MLM pre-training.
To validate the effectiveness of our proposed method, we conducted comprehensive experiments on the Nucleotide Transformer Benchmark and Genomic Benchmark. Compared to models with similar parameters, our model achieved excellent performance. **More surprisingly**, it even surpassed the distillation ceiling-teacher model on some sub-tasks, which is more than **500×** larger. 
Lastly, we conducted a comprehensive analysis of the HAD architecture, including linear probing representation evaluation, which demonstrates both the strong representation capacity of HAD and the validity of our teacher model selection for distillation. t-SNE visualization further supports these findings, providing an intuitive view of the model's representation ability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes HAD (Hybrid Architecture Distillation), a compact genomic sequence model that learns from a large teacher via a dual‑branch pretraining objective: 

(i) visible‑token feature alignment to the teacher with an MSE loss;

and (ii) masked‑token reconstruction using a decoder that performs cross‑attention from masked queries to visible keys/values. 

A two‑stage masking scheme addresses the tokenizer mismatch between a 6‑mer teacher and a char‑level student by masking at the teacher’s k‑mer level, then mapping those masks to characters. The student backbone is a hybrid Bi‑Gated Delta Net (GDN) plus a single FlashAttention layer, with a chunkwise parallelization of the recurrence.

HAD is pretrained for 10k steps on GRCh38 with RC augmentation and then fine‑tuned on the Nucleotide Transformer Benchmark and Genomic Benchmarks. Ablations indicate the benefit of both the visible‑token distillation branch and the single attention layer.

### Strengths
(1) The teacher‑group masking + student mapping prevents easy leakage and aligns the visible‑token supervision with the student’s tokenizer is a novel architecture innovation.

(2) The model achieves good performance (even outperform the teacher model) using only 1/500 of the size by distillation on Genomics Benchmark and NT.

(3) The clear ablation study shows that removing visible‑distillation or attention hurts performance. Teacher size matters (50M/100M < 500M). t‑SNE and linear probes suggest cleaner structure and better few‑shot generalization.

### Weaknesses
(1) Several typos exist: at line 254, "Equation equation 1", at line 858, "Pra-training".

(2) The NT and genomics benchmark are known to be fluctuant and many tasks' results largely rely on the finetuning recipes. The teacher and student models could have different fine-tuning recipes for the optimal behavior. Consider using a more convincing benchmark like BEND.

### Questions
(1) Can you further demonstrate the performance gain on some other well known benchmark like BEND?

(2) You align the final‑layer teacher features. Did you try multi‑layer or early‑layer distillation (or relational/contrastive feature matching)? Any gains?

### Soundness
3

### Presentation
2

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
The authors propose a new architecture and distillation strategy for genomic language models. The architecture is a variant of Gated Deltanet with added softmax attention. The authors pretrain on the Human reference genome and then compare finetuning performance on the genomic benchmarks and the NT dataset.

### Strengths
0) The authors report the performance of many models alongside theirs, and use 2 datasets for downstream performance eval.

1) I find it interesting to use new architectures such as GDN for genomic foundation models. The authors propose a model which is actually new in this space.

2) The distillation strategy is smart and well documented -- can serve as a good guideline for future research!

3) Some ablations are presented

4) All results have error bars

### Weaknesses
I think the paper is a bit underdeveloped at this stage. Most importantly, there are some methodological flaws and overclaims.

0) The architecture design is a bit arbitrary -- looks like pure engineering. The motivation for architecture choice is vague, generic, and superficial, e.g., "This hybrid approach harnesses combined strengths, effectively integrating GDN’s proficiency in capturing local and long-range sequential patterns with attention’s capacity for unifying global context". 

1) I appreciate the author's engineering skills and their drive towards improving benchmark results, yet there is a huge overclaiming of the results. The $\Delta$ proposed by the authors to motivate their claim "it even surpassed the distillation ceiling-teacher model on some sub-tasks, which is more than 500 × larger", is misleading: NT is a model with a different architecture, trained on a different reconstruction loss compared to what you have. Downstream performances are not comparable. There is no reason to say that you get better with 500 times fewer parameters -- you changed the architecture completely!! It is known that you could use many fewer parameters (HyenaDNA and Caduceus papers). Your hybrid model and distillation strategy do not change the fact that your model has an architecture much similar to HyenaDNA and Caduceus compared to NT.

2) To compare your approach safely, I would train on exactly the same pipeline (e.g., same sequence length and same distillation loss) a Caduceus model of the same size.  Caduceus results (the ones reported) are also not comparable, indeed. If the pipeline changes, I expect that anything could happen (e.g., was Caduceus trained on the same exact number of tokens?)

3) In some way, the authors propose a complete package: have a teacher model and use GDN + attention. Results are good, but I am (a) not surprised that GDN works better than Mamba 1 (i.e. in caduceus) and (b) not surprised that distillation helps. It is indeed known that distillation losses can accelerate learning, even when using a weak model as a teacher, such as random teachers. (ref: https://arxiv.org/abs/2302.12091). A standard reference is also the classic DeiT paper (https://arxiv.org/abs/2012.12877).

4) The ablation point to the fact that distilling large NT variants is better. However, the gap is relatively small: 50M also works well. What if you dont distill at all? I think this ablation should be present (sorry if I missed it).

### Questions
See above.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents HAD (Hybrid Architecture Distillation): a dual-branch pretraining framework that distills high-level knowledge from NTv2 into a compact student based on a bidirectional GDN backbone and a single self-attention layer. On the Nucleotide Transformer and Genomic Benchmarks, the student outperforms similarly sized baselines and exceeds the teacher on several NT subtasks.

### Strengths
- Well-structured KD+MLM scheme: align only on visible tokens and reconstruct masked tokens conditioned on visible context; clean separation of objectives.
- Strong results for tiny models, with multiple wins over compact baselines and selected wins over the large teacher.

### Weaknesses
Major
- The conceptual novelty is limited, as the method combines several known components (feature-based KD, cross-attention MLM, and a hybrid GDN-Attention architecture). The paper demonstrates that this combination works, but lacks deeper insight into why. For instance, the reason visible-only alignment consistently surpasses masked-only alignment is not adequately explained.

Minor
- Line 88: aliment -> alignment.

### Questions
- The paper states the surprising "student surpassing teacher" result, but does not deeply analyze why it occurs. This is the most interesting finding of the work. Is the Hybrid-GDN architecture simply a better inductive bias for these genomic tasks than the Transformer? Or does the hybrid distillation+MLM task act as a powerful regularizer?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method for creating compact genomic sequence models through knowledge distillation. The approach combines a 1.1M parameter student model (bidirectional Gated Delta Net + self-attention) with a dual-branch pretraining framework: (1) feature alignment between visible nucleotides and a 500M parameter NTv2 teacher model, and (2) masked nucleotide reconstruction via cross-attention. One key technical component is a two-stage masking strategy designed to handle tokenizer mismatches between the “k-mer teacher” and “character-level” student. The authors evaluate on Nucleotide Transformer and Genomic Benchmarks, claiming their compact model matches or exceeds much larger models, including the teacher, on several tasks.

### Strengths
- Addresses practical need: Targets computational efficiency in genomic applications where resources are constrained
- Rigorous evaluation: Multiple benchmarks with proper statistical analysis, cross-validation, and representation quality assessment beyond task-specific metrics
- Strong empirical results: Achieves competitive or superior performance to NTv2, even with 500× parameter reduction
- Comprehensive validation: Includes linear probing and t-SNE visualization to validate learned representations

### Weaknesses
**Novelty**
- The background and references section omits relevant masked language modelling work such as MAE-LM (Meng et al, 2024), and a previous application of an MAE decoder to DNA models by BarcodeMAE (Safari et al, 2025). The cross-attention based decoder is similar to that used in CrossMAE. Together, these omitted citations lead to an overstatement in the architectural novelty of the proposed method.

**Soundness**
- Critical ablations are missing: the paper lacks distillation-only and reconstruction-only comparisons, preventing assessment of individual component contributions.
- It is not spelled out in the paper that pretraining data used for the distillation is **not** the same as the data used to train the teacher model, NTv2. L319 says the data is the same as used in two citations, but it is to transparent that this is the papers for training the HyenaDNA and Caduceus baselines, not the NTv2 baseline. The authors indicate surprise that the student was able to outperform the teacher (L027, L483), but do not make it clear that the student was better than the teacher at two types of tasks (Histone Markers, Enhancer Annotation) and worse at two other types (Promoter Annotation, Splice Site Annotation). I suspect that this difference in performance vs the teacher is likely due to the change in pretraining data - by changing the training distribution, the model will become better at the data seen during distillation than the teacher, and less good at data not shown during distillation. However, this may not be the case - the difference could be due to the shift in training task instead of the distributional shift of the data (Lowe et al, 2024; Marks et al, 2024). Hence experiments are needed to assess and evaluate the impact of these factors.
- The tokenizer mismatch necessitates two-stage masking; character-level teachers or a 6-mer student could eliminate this unnecessary engineering burden, but neither of these options were explored.
- It would be helpful to see the teacher's performance alongside the graphs in Fig 5. I am also confused as to this subset of the experiments is shown for the figure.

**Presentation**
- Tables 2 and 3 captions need to explicitly say that this is fine-tuning evaluation (instead of linear probe, kNN, etc.) so a reader can understand the table at a high-level without searching the main text for the section describing the methodology.
- Table 3 font size is very small. Usually we would want to have a row per comparator (model) and a column per comparison metric (evaluation dataset). If the table doesn't fit that way, it is okay for it to be transposed (as it is now), but the title cell widths need to be narrower so the column widths are narrower and font is larger. Probably this is solved simply by adding a line break in between the method names and \citep references.
- Fig 5 text is far too small. Text within figures should not be smaller than 70% of the size of the main body text, i.e. scriptsize.
- Fig 7: You should bold the best method for each dataset. This will make it clearer which datasets the teacher (NTv2) does best on and which the student (HAD) does best on.
- Fig 7 is not referred to in the text.
- L086 Need to explicitly introduce the NTv2 abbreviation here.

**References**
- MAE-LM: Meng et al (2024). "Representation Deficiency in Masked Language Modeling". ICLR 2024. https://openreview.net/forum?id=b3l0piOrGU
- BarcodeMAE: Safari et al (2025). "Enhancing DNA Foundation Models to Address Masking Inefficiencies". https://arxiv.org/abs/2502.18405
- CrossMAE: Fu et al (2025) "Rethinking Patch Dependence for Masked Autoencoders". TMLR. https://openreview.net/forum?id=JT2KMuo2BV
- Lowe et al (2024). "An Empirical Study into Clustering of Unseen Datasets with Self-Supervised Encoders". https://arxiv.org/abs/2406.02465
- Marks et al (2024). "A Closer Look at Benchmarking Self-Supervised Pre-training with Image Classification". https://arxiv.org/abs/2407.12210

### Questions
- Fig 5 shows the student performs better when distilling the 50M than the 100M NTv2 teacher. Can the authors comment on why this might be?

**Typos**
- L20 "we employ [the] NTv2-500M as"
- L21 "grouping masking strategy" -> "grouped masking strategy"
- L453 Truncated sentence

### Soundness
3

### Presentation
3

### Contribution
3
