# Don’t Forget the Context: A Multitask Transformer for Intracortical Speech Decoding

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
We present a transformer-based sequence-to-sequence model for human speech decoding from intracortical neural recordings. Unlike prior framewise recurrent approaches trained with connectionist temporal classification, our approach jointly models neural and linguistic dynamics and generates open-vocabulary word sequences directly from the neural signal. To address the limited-data regime of human brain–computer interface datasets, we adopt a multitask framework that combines phoneme and word decoding with auxiliary supervision from Mel-frequency cepstral coefficients, and we introduce Neural Hammer \& Scalpel day-specific transformation to mitigate cross-day nonstationarity. The model establishes a new benchmark in phoneme decoding on the Willett et al. dataset and improves over previous end-to-end systems in word decoding. Attention visualizations reveal interpretable temporal chunking aligned with speech segments, shedding light on emergent neural dynamics. Finally, a scaling analysis shows favorable power-law trends, suggesting that continued data growth could yield substantial gains and positioning transformers as strong candidates for future brain-to-text

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present a sequence-to-sequence (seq2seq) Transformer model for decoding open-vocabulary speech from intracortical neural signals. To address data scarcity, the model is trained in a multitask framework that includes phoneme decoding, word decoding (using a pretrained BART head), and an auxiliary regression task on MFCCs. The paper also introduces a novel day-specific transform, the "Neural Hammer & Scalpel" (NHS), to mitigate cross-session non-stationarity. The authors claim their model sets a new state-of-the-art in phoneme decoding and improves upon a previous end-to-end baseline for word decoding.

### Strengths
- The paper provides a strong rationale for moving beyond RNN+CTC models, clearly outlining their limitations, such as the conditional-independence assumption and the separation of the neural encoder from the language model.
- The model claims to achieve a new state-of-the-art (SOTA) Phoneme Error Rate (PER) of 14.3%. This demonstrates that the seq2seq architecture and multitask setup are effective for learning robust phoneme-level representations.
- The scaling analysis (Fig. 2) provides good evidence that the model architecture can improve with more data. The attention visualizations (Fig. 3) offer valuable qualitative insights into the model's learned alignments.

### Weaknesses
- The primary weakness is the model's performance on the main task. The paper's best-reported WER is 25.6%. This is significantly worse than the established 17.8% WER from the hybrid RNN-CTC + LM baseline (Willett et al., 2023b) reported in the same table. An absolute performance gap of 7.8% on the primary metric is too large to overlook, especially for a clinically-motivated application.
- The paper's methodological contributions are largely combinations of existing techniques.
  - The use of seq2seq Transformers is a standard, established practice in automatic speech recognition (ASR). Applying it to neural signals is a logical, but incremental, step.
  - The "Neural Hammer & Scalpel" (NHS) transform is a novel combination of a per-day affine transform (which the authors note is similar to a prior baseline) and a FiLM-style modulation. This is a good engineering contribution but not a fundamental new method for adaptation.
  - Multitask learning is a common regularization technique.
- The paper claims to improve "over previous end-to-end systems in word decoding" by comparing its 25.6% WER to the 26.3% WER of Feng et al. (2024). While true, this comparison obscures the fact that both end-to-end systems perform substantially worse than the existing, simpler hybrid-model baseline (17.8% WER).

In my opinion, the paper's strongest result is its SOTA phoneme decoding. The work might be better received if it were reframed to focus on this achievement. By framing the paper as a word decoder (as in the title and abstract), it invites a direct comparison to the SOTA word decoder, a comparison it does not win. Focusing on the value of seq2seq and multitask learning for phoneme-level representation learning would be a more defensible claim.

### Questions
- The authors have shown the model is a superior phoneme decoder (14.3% PER vs. 17.4% for the RNN-CTC). What happens if the authors use their Transformer model as just a phoneme generator and feed its output into the same WFST + LM rescoring pipeline used by the Willett et al. (17.8% WER) baseline? This would provide a direct, apples-to-apples comparison of the neural encoder quality and test if the improved PER can actually lead to a better WER.
- NHS transform introduces 24 separate sets of "hammer" and "scalpel" parameters for the 24 days. How does this model generalize to a hypothetical 25th day not seen in the training set? Does this per-day parameterization risk overfitting and limit the model's ability to generalize to new, unseen sessions?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a multitask seq2seq Transformer for intracortical speech decoding, directly mapping neural activity to phoneme and word sequences. It introduces Neural Hammer & Scalpel (NHS), a day-specific calibration module that mitigates cross-day drift, and adopts multitask training with auxiliary MFCC prediction to improve data efficiency. Experiments on the Willett et al. dataset show state-of-the-art phoneme decoding and competitive word-level results. Analyses of attention patterns and scaling laws further reveal interpretable temporal structure and consistent data–performance trends. Overall, the paper demonstrates that Transformer-based decoding can effectively capture contextual and linguistic information from intracortical recordings.

### Strengths
(1) Originality
- The paper presents a meaningful step forward in intracortical speech decoding by introducing a unified seq2seq Transformer architecture that jointly models neural and linguistic dynamics. While Transformers themselves are not new, their application in this specific domain, particularly with multitask supervision and explicit day-wise calibration—is novel and well-motivated. The proposed Neural Hammer & Scalpel (NHS) module is an way to address day-to-day nonstationarity, a major challenge often glossed over in prior work.

(2) Quality
- The experiments are carefully executed, with fair baselines and ablations that isolate the impact of each design component (e.g., NHS, MFCC supervision, BART conditioning). The inclusion of scaling law analysis and attention visualization adds depth and scientific rigor beyond simple performance reporting. The model achieves state-of-the-art phoneme decoding accuracy and competitive word-level results despite limited data.

(3) Clarity:
- The paper is generally clear and well-structured. Each architectural choice is motivated by a concrete empirical or neuroscientific problem (e.g., temporal drift, weak audio supervision). Figures and tables effectively support the arguments, and the explanations of the day-adaptation mechanism are particularly easy to follow.

(4) Significance:
- This work bridges a methodological gap between modern sequence modeling and practical neural prosthetic applications. By showing that Transformer-based decoding can capture contextual and linguistic dependencies directly from intracortical activity, the paper paves the way for future foundation-style “brain-to-text” models. It provides both a technical contribution and a broader conceptual shift for the speech BCI field.

### Weaknesses
(1) Limited novelty in model design
- Most components like Transformer backbone, multitask setup, FiLM modulation are adaptations of existing techniques rather than fundamentally new inventions. The originality lies mainly in integration and application. The paper could be strengthened by articulating why this particular combination works better than other possible architectures (e.g., conformer-based, latent-alignment models).

(2) Insufficient evaluation diversity
- Experiments focus heavily on a single dataset (Willett et al.), with no cross-subject or cross-task validation. This limits claims of generalization and practical robustness. A small-scale transfer or held-out-day test would better demonstrate NHS’s effectiveness beyond memorizing per-day patterns.

(3) Ablation depth and analysis scope
- While ablations exist, some design choices (e.g., the gating function or MFCC weight) lack sensitivity analysis. It would help to quantify how each auxiliary loss contributes to performance and stability over time. Similarly, results are mostly quantitative; additional qualitative error analyses (e.g., semantic vs. phonetic errors) would provide richer insight.

### Questions
(1) The proposed NHS module effectively addresses day-to-day nonstationarity, but it appears to rely on per-day embeddings learned jointly with the training set.
- Have you tested the model’s ability to generalize to unseen days (e.g., a held-out-day split)?
- If not, could you comment on whether NHS can handle new sessions without retraining, or how a continuous-time version might perform?
  
(2) The paper shows that adding MFCC and BART supervision improves performance, but the mechanism remains somewhat unclear.
- Do you have any analysis (e.g., layer probing, representation similarity) indicating how MFCC or BART signals influence encoder representations?
- Would the same benefit persist if MFCCs were randomly shuffled or misaligned?
  
(3) The scaling law results are interesting but extrapolated from relatively small data fractions.
- How sensitive are these fits to the chosen data fractions (0.1–1.0)?
- Have you validated that the power-law trend holds when adding or removing entire recording days rather than random subsets?
  
(4) The model uses a BART decoder with partially frozen layers.
- Did you explore alternative strategies, such as training from scratch or using a smaller LM head?
- How crucial is the BART initialization compared to a randomly initialized decoder for achieving good WER?
  
Overall, the experimental setup is comprehensive, with ablations and analyses. However, the work feels more like a domain-focused integration study rather than a conceptual or algorithmic innovation typical of ICLR. Its strength lies in methodological rigor and neuroscientific relevance, which might make it a better fit for a specialized journal.

### Soundness
3

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
4

### Summary
This paper presents a transformer-based sequence-to-sequence model for decoding speech from intracortical neural recordings in a patient with ALS and anarthria. The authors propose a multitask framework that jointly learns phoneme and word decoding with auxiliary MFCC supervision, and introduce a "Neural Hammer & Scalpel" (NHS) day-specific transformation to handle cross-day nonstationarity. the model achieves 14.3% phoneme error rate (PER) and 25.6% word error rate (WER), improving over previous end-to-end approaches. The authors demonstrate favorable power-law scaling trends and provide attention visualizations showing interpretable temporal chunking aligned with speech structure.

### Strengths
1. Strong ablation studies, Table 1 systematically evaluates architectural choices (Transformer vs. RNN), auxiliary tasks (MFCC, BART), and day transformations (NHS vs. Linear vs. None)
2. Strong phoneme performance and clear improvements over framewise CTC approaches
3. Excellent presentation and contextualization
4. Good details for facilitating reproducibility

### Weaknesses
1. Scaling law extrapolations (Section 3.4) are based on single seeds; quantitative uncertainty is lacking.
2. The paper’s power-law scaling analysis (Section 3.4; Appendix B; Figure 2) extrapolates phoneme and word error rates from ∼10 k to ∼100 k training trials under the assumption that the neural data distribution remains stationary. This assumption is unlikely to hold for intracortical recordings. Therefore, the statement of the projected “low single-digit PER and WER” estimates (page 6, line 319) may be overly optimistic.
3. While Figure 3 presents intriguing qualitative patterns, the attention interpretation (Section 3.6) is based on single representative trial; needs statistical validation across dataset. The authors should quantify: (a) what fraction of trials exhibit clean "box" structure in Layer 6, (b) how cross-attention entropy varies across decoder types, (c) whether attention peaks align with envelope features statistically across the test set
4. The authors state "training stability and performance to be highly sensitive to hyperparameters" but provide no ablation over critical choices like model dimension, number of layers, dropout rate, or warmup schedule.

### Questions
1. Can you provide confidence intervals or multiple-seed averages for the power-law fits in Figure 2? How sensitive are the extrapolations to the chosen functional form?
2. Why freeze the first 3 BART decoder layers? What is the impact of different freezing strategies or fine-tuning more/fewer layers?
3. Your power-law fits project substantial gains at 100k trials. How do you reconcile this with known long-term nonstationarities (electrode drift, scarring, etc.) that would violate the stationary-distribution assumption?
4. In the NHS module, how sensitive are results to the FiLM modulation strength?
5. Could you quantify attention alignment, for example, by correlating peak cross-attention with speech envelope events?
6. Given that inference is much faster than two-stage systems, have you considered beam search or other techniques to generate multiple hypotheses for rescoring? This might close the WER gap in Table1.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a seq2seq Transformer that decodes intracortical signals directly into phonemes and words. It uses (i) a multitask setup with an auxiliary MFCC head, (ii) a frozen-part BART word decoder for language priors, and (iii) a day-specific “Neural Hammer & Scalpel (NHS)” transform (global affine + FiLM) to address across-day drift.

### Strengths
1) The paper tackles a relevant and challenging problem: the limited-data issue of human brain–computer interface datasets.
2) The paper includes clear, well-documented architecture and training details which helps reproducibility.

### Weaknesses
1) The novelty of the work is limited. The core contribution is a conventional seq2seq Transformer with (a) an auxiliary MFCC prediction head, (b) a partially frozen BART decoder, and (c) a day-wise affine+FiLM calibration (NHS). None of these ingredients are algorithmically new in ML; NHS is essentially per-day affine re-mix + FiLM gating. The work reads as careful engineering, not a conceptual advance.

2) All core results are reported on one intracortical participant from Willett et al. Moreover, the paper discards one of the implanted areas and uses only area 6v (128 channels), further narrowing scope. There is no cross-subject or cross-implant generalization. Therefore, conclusions about general utility are not supported.

3) The evaluation scope of the paper is narrow. It primarily contrasts variants of the authors’ own model and omits comparison with other baselines for intracortical or speech decoding  in Table 1 (e.g., transducer-based, CTC-Transformer, or hybrid pipelines). As a result, the contribution cannot be properly contextualized within the existing literature on brain signal-based speech decoding.

### Questions
Were the WFST + LM results recomputed under your preprocessing pipeline, or were they taken directly from Willett et al. (2023)?

### Soundness
2

### Presentation
2

### Contribution
1
