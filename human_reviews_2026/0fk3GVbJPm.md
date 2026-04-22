# Towards Unsupervised Speech Recognition at the Syllable-Level

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Training speech recognizers with unpaired speech and text -- known as unsupervised speech recognition (UASR) -- is a crucial step toward extending ASR to low-resource languages in the long-tail distribution and enabling multimodal learning from non-parallel data. However, existing approaches based on phones often rely on costly resources such as grapheme-to-phoneme converters (G2Ps) and struggle to generalize to languages with ambiguous phoneme boundaries due to training instability. In this paper, we address both challenges by introducing a syllable-level UASR framework based on masked language modeling, which avoids the need for G2P and the instability of GAN-based methods. Our approach achieves up to a 40\% relative reduction in character error rate (CER) on LibriSpeech and generalizes effectively to Mandarin, a language that has remained particularly difficult for prior methods. Code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents SylCipher, a syllable-based unsupervised ASR (UASR) system that jointly predicts syllable boundaries and embedding tokens from raw speech using a unified self-supervised objective. The authors conduct experiments across domains (LibriSpeech, SpokenCOCO) and languages (English, Mandarin). On Mandarin, SylCipher achieves a 12.2% phone error rate, outperforming previous GAN-based UASR methods.

Overall, this is a paper revisiting the UASR task with a new unit of representation (syllable). While the approach is interesting and the results are solid compared to prior UASR systems, the contribution is mainly the change of processing unit rather than a conceptual or methodological advance. The scope feels somewhat narrow for ICLR and would likely be a better fit for a speech-focused venue such as Interspeech.

Overall, this is a well-executed study with solid experiments, but the novelty and relevance are limited. The work would be more impactful if applied to truly under-resourced languages.

### Strengths
* Proposes a new speech unit (syllable) for the UASR problem.
* Achieves competitive results compared to previous UASR approaches.

### Weaknesses
* The motivation for UASR is strongest in the context of truly under-resourced languages, but the experiments are limited to English and Mandarin—two well-resourced languages. This makes the work feel a bit outdated, given that by 2025 we expect to see these methods applied to real low-resource settings where they are most needed.

* Limited discussion on whether syllables are suitable units across different language types.

### Questions
UASR is not the only approach for building ASR systems for low-resource languages. It would be useful to compare your results with semi-supervised methods, as starting from a small seed (e.g., around one hour of aligned speech–text data) is often realistic even in low-resource or language documentation scenarios.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tackles the task of unsupervised speech translation, where no paired speech and transcript are included in the training data. The vast majority of UASR systems operate at the phoneme level, which calls for a grapheme-to-phoneme converter. This is not always available for all languages. This paper instead proposes to operate on syllable level.

The overall training paradigm comes in three parts:

1. Use an unsupervised syllable detector to train with an initial boundary
2. Joint E2E training where the soft-polling mechanism is trained along side the rest of the model, hence improving the syllable boundaries
3. Positional unigram and skipgram matching (PUSM) training, which further improves the model with explicit distribution matching between the speech feature and text tokens.

Experiment result shows that the system is able to achieve better performance than all the existing G2P-free approaches, but was unable to push beyond G2P-based method. The method is also shown to be robust in terms of domains and also works for Mandarin.

### Strengths
1. Being able to get rid of pronunciation dictionary will significantly help with expanding UASR systems to more low-resource languages.
2. The performance improvement over existing G2P-free approaches is quite remarkable.
3. I liked the fact that the experiments covered both matched and unmatched setups, and examined robustness to domains, scripts, and syllable boundary detection model.

### Weaknesses
1. I understand that SylCipher is not entirely language-universal yet because different writing systems exist, but since one of the main benefit of getting rid of pronunciation dictionary was to save work when expanding to new languages, I think it is important to test this on more languages than just English and Chinese.
2. The presentation of the paper could be significantly improved. Here are my suggestions:

* 2a. Re-draw Figure 1 and properly combine the parts that are shared between speech and text inputs.

* 2b. Formula (1) doesn't quite help me understand how I can recover $y^\*$ given only unpaired $X$ and $Y$. You probably want to think about if you can re-write $y^\*$ as an argmax of an objective function that you are optimizing.

* 2c. There are quite few places where aggressive space hacks were used, which I believe you should remove. I would consider removing the theoretical guarantee on Page 4 to make some space for that.

* 2d. I could use a little more intuition apart from Formula 10 as to why PUSM provided more explicit inductive bias for distribution matching.

* 2e. L357: "..., ~~though~~ it's performance is limited by ..."

### Questions
1. Why is the evaluation constrained to English? In other words, what's the hurdle to evaluating this also on French? Spanish? Italian? Finnish?
2. L140: I don't follow the last sentence of the paragraph ("As shown in Wang et al. (2023b), ..."). Can you explain?
3. Table 2: I'm confused about why you switched from CER to PER for Mandarin?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SylCipher, which is a novel UASR system that operates at the syllable level rather than the phoneme level, eliminating the need for G2P convertors. The approach uses an iterative training approach and experiments across two English domains and Mandarin show substantial improvements over existing G2P-free approaches. While the approach still requires language-specific syllabification methods to obtain strongest performance and its theory relies on idealized assumptions, the work makes a clear contribution by establishing syllable-level UASR as a viable alternative to phoneme-based systems.

### Strengths
Motivation for the study is clearly stated and the manuscript aims to tackle the important problem of making ASR more inclusive of the world’s languages and enabling multimodal learning from non-parallel data. The paper makes a solid contribution by demonstrating that syllable-level modeling can be effective for UASR. The empirical results are strong and the work includes thoughtful ablations on syllabifier choice, pooling mechanisms, and vocabulary size, and evaluates cross-domain robustness. The iterative training procedure, while complex, shows clear improvements at each stage, and the boundary refinement mechanism successfully improves upon the teacher model’s boundaries.

### Weaknesses
The paper makes a clear case that UASR is a crucial step toward extending ASR to low-resource languages in the long-tail distribution. However, experiments involving such low-resource languages are missing from the paper, and instead the authors perform experiments on high-resource languages for which well-performing ASR systems exist. Empirical evidence on low-resource languages would substantially strengthen the contribution of this work.

The paper shows empirically that the system works with approximate syllable boundaries, but the theorem has much stronger assumptions and provides no formal analysis of why the theoretical guarantees should hold when its core assumptions are violated. The paper would benefit from such an analysis.

Citation style is inconsistent throughout the text. Narrative and parenthetical forms are sometimes applied in the wrong context. Please standardize. This is a minor issue.

### Questions
Could the authors provide results on at least one low-resource language to validate that the method works in the target scenario?

Relatedly, could the authors provide insight into how the amount of training data affects performance of their method?

The performance gains of SylCipher are clear, but the best-performing model requires a 3-stage training procedure. Could the authors provide an analysis of computational costs of training time compared to baselines? That would help assess the practicality of their method.

In Section 4.2, the authors mention using the first transformer layer instead of the last improves performance, citing "over-contextualization" as the cause. Could the authors provide the results for both layer 1 and layer 2 outputs for inference? Does this pattern hold across all three datasets?

### Soundness
3

### Presentation
2

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
The paper introduces SylCipher, a novel framework for unsupervised automatic speech recognition (UASR) at the syllable level. Unlike prior UASR systems that operate at the phoneme or word level (often requiring grapheme-to-phoneme converters or suffering from instability), SylCipher leverages masked language modeling and recent advances in unsupervised syllable boundary detection to align unpaired speech and text. The approach is resource-efficient (no G2P required), theoretically grounded, and empirically validated across English (LibriSpeech, SpokenCOCO) and Mandarin (AISHELL-3). SylCipher achieves up to a 40% relative reduction in character error rate (CER) over previous G2P-free UASR methods and demonstrates robustness to domain and language shifts.

### Strengths
The strengths of the paper are:
1. Originality: 
   - First to propose syllable-level UASR, bridging the gap between phoneme- and word-level approaches.
   - Avoids reliance on G2P or pronunciation dictionaries, making it applicable to truly low-resource languages.
   - Theoretical analysis provides guarantees for distribution matching and zero-error UASR under certain conditions.

2. Quality:
   - Extensive experiments on multiple datasets (LibriSpeech, SpokenCOCO, AISHELL-3) and languages (English, Mandarin).
   - Careful ablation studies on segmentation, vocabulary size, and pooling mechanisms.
   - Robustness to domain and language shifts is empirically demonstrated.

3. Clarity:
   - Clear motivation for syllable-level modeling and its advantages over phoneme/word-level approaches.
   - Well-organized presentation of architecture, training objectives, and results.
   - Visualizations and tables directly support the claims.

4. Significance:
   - Enables unsupervised ASR in languages lacking G2P resources.
   - Advances the state of the art in UASR, with potential impact on speech technology for low-resource and endangered languages.

### Weaknesses
The weaknesses of the paper 
1. The method is not yet language-universal; languages with non-syllabic scripts (e.g., Hebrew, Arabic) may pose challenges for syllabification.

2. While the method is robust to segmentation noise, performance still depends on the quality of unsupervised syllable boundary detection.
The approach to syllabification (e.g., Pyphen+, BPE+) may require further tuning for new languages.

3. The iterative training procedure (fixed boundary, JE2E, PUSM) could be further simplified into a fully end-to-end approach.

### Questions
The work is with good quality in general. I have several questions:

1. In natural language processing nowadays, languages are covered into one tokenizer. For speech, phone-level modeling can somehow cover most of the languages by sharing one pronunciation vocab, e.g.. How about the syllable-level? Is it possible to train a multi-lingual UASR system by leveraging a more universal/ language agnostic speech tokenizer? It is interesting to see the method effective on more than single language individually.

2. Based on the results in table 1, REBORN still holds the best performance in unmatched case. Does it mean phone-based method still has the advantage?

3. How to perform down-stream ASR finetuning given limited paired data? This would bring the system into a more practical usage. Any experiment to show the effectiveness of the ASR finetuning?

### Soundness
2

### Presentation
3

### Contribution
3
