# ECHO: Toward Contextual Seq2Seq Paradigms in Large EEG Models

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4, 8

## Abstract
Electroencephalography (EEG), with its broad range of applications, necessitates models that can generalize effectively across various tasks and datasets. Large EEG Models (LEMs) address this by pretraining encoder-centric architectures on large-scale unlabeled data to extract universal representations. While effective, these models lack decoders of comparable capacity, limiting the full utilization of the learned features.
To address this issue, we introduce ECHO, a novel decoder-centric LEM paradigm that reformulates EEG modeling as sequence-to-sequence learning. ECHO captures layered relationships among signals, labels, and tasks within sequence space, while incorporating discrete support samples to construct contextual cues. This design equips ECHO with in-context learning, enabling dynamic adaptation to heterogeneous tasks without parameter updates.
Extensive experiments across multiple datasets demonstrate that, even with basic model components, ECHO consistently outperforms state-of-the-art single-task LEMs in multi-task settings, showing superior generalization and adaptability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ECHO, a decoder centric framework for EEG modeling as a _seq2seq_ model, enabling in-context adaptations to heterogeneous tasks given supporting samples and their labels. This shift paradigm shows promising results across tasks in-distribution or out-of-distribution. However, there is selective reporting of the results in the main body, so I am leaning towards not acceptance.

### Strengths
- A large EEG model with ICL capability
- Pretraining across multiple EEG dataset with distinct channel configurations, made possible by a standardization procedure guided by prior knowledge
- Learned model transferable to unseen datasets

### Weaknesses
1. If I understand correctly, the reported values are not complete. It is mentioned in the appendix that there were 14 datasets involved in pretraining, but only 6 are reported in section 4.1 and 4 more are reported in appendix. Because the proposed model outperformed baselines in the 6 tasks reported in section 4.1 and showed comparable or worse in the 3 of 4 tasks reported in appendix (outperformed by the same baseline in 2 tasks), I believe it is worth mentioning that in the main body of the paper. 
2. Normalization process of diverse EEG channels is guided by prior knowledge, which could limit future scalability when dataset with unseen EEG channel configurations is put in the pretraining data.
3. Overall, discussion on limitations is lacking.

### Questions
1. Please see weakness number 1. Could the authors elaborate on why the 6 tasks in section 4.2 representative? In general, I believe the it is worth mentioning where the model is not working as good, given there is at least 3 datasets out of 14. Also, I count 14 datasets in total in the pretraining dataset, but it says 12 in the paper (line 329, page 7). Please comment on the difference.
2. In equation 11, it seems that next-token prediction is conducted for every textual token, given all prior tokens including the support samples ones. Are predicted support label being used in the subsequent generation, or are there any uses of teacher forcing?
3. Could authors provide more details on the standardized templated channel set used in the pretraining? In the paper, a 75-channel set is advocated, is there a theoretic reason, or it is chosen empirically?
4. Other than mapping everything to a standardized channel set, have the authors considered other ways to mitigate channel diversity, for example, channel embedding based on channel identity (like unit embedding proposed in POYO+, reference shown below) or coordinates?
5. Are the reported values in Table 2 using 8 or 12 supporting samples? Either way, it would be interesting to see how the number of supporting samples affect decoding performance.
6. One substantial difference between the proposed approach and existing LLM decoder is the use of support samples, ultimately transforming the paradigm to meta-learning. But LLM is known for enabling in-context learning, so have authors tried to equip the support samples to LLM decoder as well?
7. Did the authors use a pretrained model for the warm-up stage? Because in Figure 2 (c), encoder is marked as frozen during warm-up.

### Soundness
3

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
ECHO proposes a decoder-centric paradigm for Large EEG Models (LEMs) using sequence-to-sequence learning with in-context learning (ICL). Unlike existing encoder-centric or LLM-centric LEMs, ECHO structures EEG samples, tasks, and labels as sequences, allowing the model to adapt to multiple tasks without fine-tuning. Trained on 12 datasets with multi-task learning, ECHO outperforms single-task state-of-the-art baselines and demonstrates zero-shot generalization to unseen datasets.

### Strengths
First to show ICL works for continuous time series (not just discrete text).
Strong multi-task performance: Beats single-task baselines despite harder setting.
Code promised, all hyperparameters documented, 5 seeds reported.
Decoder-centric paradigm enables multi-task learning in single model.

### Weaknesses
Computational cost not fully discussed. Training time not reported for the warm-up 90 epochs + contextual 40 epochs. Need to process support samples every time.

For ablation studies: Only ablates positional encodings (Figure 3), missing: decoder size, support sample count, sequence length. Why using 8 support samples in Round 1 and 0-12 in Round 2? Lacking comparison with different decoder architectures.

Current baselines are mostly single-task models, what about multi-task learning baselines? (shared encoder + task-specific heads) For now it remains unknown if seq2seq is better than standard multi-task learning.

### Questions
Can you compare ECHO multi-task vs baselines trained in a multi-task manner? (fair comparison)

What's the training time and inference time cost?

How does performance scale with decoder size?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes **ECHO**, a novel **decoder-centric seq2seq paradigm** for Large EEG Models (LEMs), reframing EEG decoding as in-context sequence modeling. It addresses critical limitations in existing LEMs—decoder bottlenecks, poor cross-task generalization, and lack of in-context learning. The method is evaluated on a broad set of EEG datasets with extensive ablation and zero-shot experiments, demonstrating consistent gains over strong baselines.

### Strengths
1. **Novel paradigm shift**: Introduces the first decoder-centric, seq2seq formulation for EEG modeling, moving beyond encoder-finetune or LLM-as-decoder paradigms.

2. **Effective use of in-context learning**: Demonstrates that EEG models can benefit from discrete support samples for dynamic adaptation without parameter updates.

3. **Comprehensive evaluation**: Multi-dataset, multi-task, zero-shot, and ablation studies provide strong empirical support for generalization and robustness.

### Weaknesses
1. **Unverified channel averaging in Eq. (5)**: The paper assumes averaging over aliased channels (e.g., T9 and M1) is valid, but no validation is provided—phase misalignment may introduce unnatural signal artifacts.

2. **Missing signal normalization**: It is unclear whether the transformed EEG tokens undergo normalization (e.g., z-score), which is crucial for stable training across datasets.

3. **Fixed-length tokenization in Eq. (8)**: Are all EEG signals be split into equal-length K patches? How do you encode patches with different sampling points?

4. **Ambiguity in embedding nature**: It is not clarified whether the EEG token embeddings are continuous or quantized. If continuous, how is cross-entropy loss applied during next-token prediction?

5. **Lack of warm-up stage evaluation**: The warm-up phase resembles standard multi-task supervised learning. Reporting its direct performance (without contextual training) would clarify whether gains stem from multi-task pretraining or true in-context learning.

6. **Biased task selection**: Downstream tasks focus on emotion and motor imagery. Missing key EEG applications like epilepsy detection or sleep staging limits claims of universality.

7. **Incomplete ablation of positional encoding**: The ablation removes sample- and text-level encodings, but does not test a baseline using only a single, global absolute position encoding across the entire sequence—this would better isolate the necessity of the hybrid design.

8. **Limited generalization beyond seen tasks**: The model relies on labels and tasks seen during training. No evidence is shown for few-shot adaptation to entirely new tasks, limiting claims of true generalization.

9. **Unreported model capacity differences**: Baseline models may vary significantly in parameter count. Moreover, no attempt is made to train baselines in the same multi-task setting, risking unfair comparison.

10. **No scaling law analysis for support samples**: The impact of increasing support count (e.g., 0 → 8 → 12) shows diminishing returns, but no systematic study (e.g., performance vs. number of supports) is conducted.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
ECHO reframes Large EEG Models as a decoder-centric seq2seq problem that serializes support EEG samples, task/label tokens, and a target EEG, letting a Transformer decoder infer the target’s task and label via next-token prediction—thereby enabling in-context and multi-task EEG learning without parameter updates.  Technically, it adds (i) a simple channel-alignment preprocessor to standardize heterogeneous electrode layouts, (ii) a hybrid positional encoding spanning token-, sample-, and text-levels to fuse continuous EEG with discrete control tokens, and (iii) a staged training scheme that progressively increases support examples to instill contextual reasoning.

### Strengths
The strengths of this article are as follows:

1. It introduces a decoder-centric paradigm that mitigates the limitations of prior approaches, such as poor generalization and heavy reliance on inductive biases, while enabling in-context learning.
2. The argumentation is well structured and easy to follow.
3. The experimental evaluation is comprehensive: it benchmarks ECHO against various baselines and includes ablations on two forms of positional encoding, demonstrating that both are necessary.

### Weaknesses
This paper presents several weaknesses:

1. While the paper rightly highlights the generalization bottlenecks of encoder-centric methods, the proposed method, ECHO appears confined to a single task family—classification. By contrast, encoder-centric approaches can be lightweight and adaptable across prediction, classification, and captioning, which may weakens the paper's contribution.
2. The proposed approach depends on a pre-trained encoder, atop which a decoder supporting multiple classification tasks is trained. Clarifying how this design substantively departs from an encoder-centric adaptation tailored to general classification would help strengthen the methodological positioning.
3. The validation of ECHO’s zero-shot capability could be more persuasive. First, using SEED, data visible during training for “zero-shot” prediction may not constitute a strict zero-shot setting. Second, the evaluation is limited to the BCIC 2020-T1 for Motor Imagery task with only one baseline. A broader assessment across additional datasets and stronger, diverse baselines would make the zero-shot claims more compelling.
4. The paper’s central contribution seems to lie in its decoder-centric training paradigm. To further underscore its robustness, it may be helpful to evaluate additional combinations of encoder and decoder architectures.

### Questions
1. Different classification tasks involve varying numbers of classes. How does ECHO accommodate variable class cardinalities within a unified model architecture? I wasn’t able to find an explicit explanation; if it is already covered, could you point me to the relevant section?
2. Although the overall structure is clear, a few presentation details were confusing. For instance, the output implied by Eq. (11) appears inconsistent with Fig. 2: the figure does not show $S_{\text{out}}$. I initially interpreted $T_{2S+3}$​ as the classification output, which seems at odds with Eq. (11).
3. Could the authors clarify why removing textual positional encoding prevents the model from making meaningful predictions? What does a “no valid predictions” look like here? Intuitively, for classification, the worst case should approximate random guessing.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes ECHO, a decoder-centric sequence-to-sequence (seq2seq) paradigm for EEG modeling that concatenates support samples (with known task and label) and a target sample into one sequence, forcing the model to generate <task> then <y> (then <EOT>). The goal is to overcome the “decoder bottleneck” in Large EEG Models (LEMs) and enable in-context learning (ICL) across heterogeneous datasets. The system includes a channel-template alignment step (same/nearby electrodes averaged; missing padded with zero) and hybrid positional/role encodings. Training uses a curriculum (first fixed 8 support examples, then randomizing the count in [0,12]); testing uses k∈{0,8,12} with no task tokens fed.

### Strengths
1.Paradigm-level breakthrough: a decoder-centric seq2seq with a fixed solution order
ECHO hard-codes the reasoning path—support → target → generate <task> → generate <y> → <EOT>—and trains via next-token prediction. This lets the model learn within a single sequence space how to reason from examples to the target, instead of relying on a small classifier head or fragile text-bridging.


2.Hybrid positional/role encodings that fuse continuous EEG with discrete control tokens
The method explicitly distinguishes functional roles (e.g., support vs. target) and balances heterogeneous inputs by combining token-level, sample-level, and text-level encodings. As a result, the decoder preserves fine-grained temporal EEG structure and the logic of <task>/<y> prediction.


3.Strong evidence of generalization and “zero-context” robustness
Even without any support samples, the sequential formulation internalizes the mapping between signal, task, and label. On unseen datasets (e.g., BCIC 2020-T1) it maintains consistent generalization, and under a strict train-once, evaluate-everywhere protocol it can infer both the task and the label without being given <task> at test time.

### Weaknesses
ICL evaluation under-characterized
Training uses a two-stage curriculum (8 → random [0,12]), while testing uses k∈{0,8,12}. However, tables report only “No Support” vs an ambiguous “Support” (k=8? k=12? average/best/mix?). There is no k→performance curve (e.g., 0,2,4,8,12 with CIs) and the support sampling protocol (same/cross subject or device, adjacency, fixed indices) is not fully specified. This obscures how performance scales with context size and complicates attribution to the paradigm rather than context amount.


Ablations do not test core claims
Current ablations toggle only parts of the hybrid positional encodings.Missing:
Seq2seq necessity & order sensitivity: shuffle support order; swap output order (<y> before <task>); remove support-recap generation; compare with a non-sequential, information-equivalent baseline that ingests the same support/target encodings and predicts via a feed-forward head.
Decoder capacity: vary layers/hidden/heads/FFN and compare to an equal-capacity hierarchical discriminative baseline (task-head → label-head) to substantiate the “decoder bottleneck” thesis.
Channel alignment: isolate its contribution via ablation.


Fairness: multi-task ECHO vs single-task baselines
ECHO trains jointly across tasks and is evaluated everywhere; most baselines are trained/fine-tuned per task. A multi-task baseline (e.g., shared encoder with task-specific heads) or single-task ECHO is needed to separate seq2seq benefits from multi-task training effects.


Channel alignment methodology
Averaging mapped electrodes may smooth meaningful spatial nuances and inject template bias. There is no comparison to stronger alternatives (e.g., spherical/spline interpolation, learnable channel mappings, or permutation-robust architectures), nor pre/post alignment PSD/topography analyses.


 Important extensions buried in the appendix
The long-sequence variant (ECHOL) and the MI-specialized variant (ECHOMI) appear only in the appendix. They should be summarized in the main text (motivation, setup, key findings) to illustrate scope and limitations; ensure variant names match the paper (avoid superscripts like ECHO^L / ECHO^MI).

### Questions
1.Technical Writing and Presentation
Typos/formatting should be fixed:
(1)“proprocess” → preprocess (p.5)
(2)Missing ) (Page 5)
(3)Sentence-initial “if” → If (p.5, under Eq.(5))
(4)“Postional" → Positional (p.6)
(5)“encodding" → "encoding" (Page 6, twice)
(6)Extra ) near Table 12 (p.24)

### Soundness
3

### Presentation
3

### Contribution
3
