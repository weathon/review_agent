# D&R: Recovery-based AI-Generated Text Detection via a Single Black-box LLM Call

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Large language models (LLMs) generate increasingly human-like text, raising concerns about misinformation and authenticity. Detecting AI-generated text remains challenging: existing methods often underperform, especially on short texts, require probability access unavailable in real-world black-box settings, incur high costs from multiple calls, or fail to generalize across models. 
We propose Disrupt-and-Recover (D\&R), a recovery-based detection framework grounded in posterior concentration. D\&R disrupts text via model-free Within-Chunk Shuffling, performs a single black-box LLM recovery, and measures semantic–structural recovery similarity as a proxy for concentration. This design ensures efficiency, black-box practicality, and is theoretically supported under the concentration assumption. Extensive experiments across four datasets and six source models show that D\&R achieves state-of-the-art performance, with AUROC 0.96 on long texts and 0.87 on short texts, surpassing the strongest baseline by +0.08 and +0.14. D\&R further remains robust under source–recovery mismatch and model variation. Our code and data is available at https://github.com/Yuxia-Sun/D-R.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new framework called Disrupt-and-Recover (D&R) for detecting AI-generated text in an efficient, practical, and theoretically grounded manner. The proposed D&R method is based on the principle of posterior concentration, which posits that AI-generated text tends to be more predictable and internally consistent than human-written text. The framework works in three steps: 

1. Disruption: The original text is perturbed through a model-free Within-Chunk Shuffling (WCS) that randomly shuffles words within punctuation-separated chunks while preserving meaning.

2. Recovery: A single black-box LLM call is used to reconstruct (recover) the original text.

3. Similarity Measurement: The similarity between the recovered and original text is computed using both semantic (BERTScore F1) and structural metrics (Kendall’s τ and Spearman’s ρ). High similarity indicates strong concentration, implying AI-generated origin; low similarity suggests human-written text.

The authors provide theoretical analysis proving that recovery similarity acts as a faithful proxy for posterior concentration, forming the foundation of D&R. They also show that D&R has linear computational efficiency, as it only requires one LLM call compared to multiple calls in previous methods.

### Strengths
1. The paper introduces a novel Disrupt-and-Recover (D&R) paradigm based on the posterior concentration principle, offering a new theoretical foundation for distinguishing AI-generated text from human-written text. Unlike heuristic or probabilistic methods, D&R provides a mathematically justified rationale linking recovery similarity to concentration properties of language models.

2. D&R achieves strong detection performance with just one black-box LLM call, making it computationally efficient and cost-effective. In contrast, most prior methods (e.g., RAIDAR, Fast-DetectGPT) require multiple API calls or white-box access to model probabilities, which are impractical in real-world applications.

3. Across diverse benchmarks, D&R achieves state-of-the-art results, with AUROC scores of 0.96 on long texts and 0.87 on short texts, outperforming strong baselines like RAIDAR and Fast-DetectGPT by large margins (+0.08 to +0.14). This demonstrates not only higher accuracy but also lower variance and greater stability across datasets.

### Weaknesses
1. The method proposed in this paper is simple. Therefore, I am a bit concerned about the robustness on texts with adversarial attacks such as content paraphrase and swap since within-chunk shuffling is considered as the main technical contribution. I would suggest testing and comparing this method with baselines on RAID dataset[1], which includes different kinds of adversarial attacks.

2. Some recent baselines are not included such as DALD [2]. Moreover, although this method is a zero-shot methods, it would also be interesting to see how it compares with the training-based methods such as DeTeCtive[3] and OOD-based methods[4] (or at least include some discussion about training-based methods).

3. Ablation of Within-Chunk Shuffling contribution is missing. 

4. All experiments are conducted on English content. The effectiveness on multilingual setting is not validated.

I would be happy to increase my score if the author can address my concerns.


[1] Dugan L, Hwang A, Trhlik F, et al. Raid: A shared benchmark for robust evaluation of machine-generated text detectors[J]. arXiv preprint arXiv:2405.07940, 2024.

[2] Zeng C, Tang S, Yang X, et al. Dald: Improving logits-based detector without logits from black-box llms[J]. Advances in Neural Information Processing Systems, 2024, 37: 54947-54973.

[3] Guo X, He Y, Zhang S, et al. Detective: Detecting ai-generated text via multi-level contrastive learning[J]. Advances in Neural Information Processing Systems, 2024, 37: 88320-88347.

[4] Zeng C, Tang S, Chen Y, et al. Human Texts Are Outliers: Detecting LLM-generated Texts via Out-of-distribution Detection[J]. arXiv preprint arXiv:2510.08602, 2025.

### Questions
See Weakness.

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
3

### Summary
This paper proposes Disrupt-and-Recover (D&R), a novel AI text detection framework that disrupts text through Within-Chunk Shuffling performs a single black-box LLM recovery, and measures semantic-structural similarity between recovered and original text to detect AI-generated content.

### Strengths
1. Proposes a novel recovery-based detection paradigm that achieves efficient detection through Within-Chunk Shuffling and a single black-box recovery
2. Cleverly exploits the posterior concentration property of AI-generated text
3. Clear theoretical motivation with well-explained connection to LLM pretraining biases

### Weaknesses
1. Lacks comparison with the latest detectors (Text Fluoroscopy, Binoculars, ImBD)
2. Lacks consideration of robustness against attacks: e.g., paraphrasing attacks. 
3. The robustness described in the paper is more like generalization capability, such as whether it can effectively generalize to short texts, rather than robustness against adversarial attacks
4. Limited performance on extremely short texts (<50 words)

### Questions
How does the performance compare to other more recent and advanced detectors?

What is the performance when facing adversarial attacks?

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
D&R proposes a framework for AI-generated text detection grounded in a posterior concentration hypothesis.

### Strengths
The novelty is within-Chunk Shuffling and using recovery similarity as a proxy for posterior concentration.

### Weaknesses
Please check the questions

### Questions
The novelty rests on Within-Chunk Shuffling and using recovery similarity as a proxy for posterior concentration.

The method still trains a classifier on recovery features [F1,τ,ρ]. It is supervised calibration on labeled AI and compared with human data. Please define what zero-shot is and what is trained.

 D&R is another transformation-consistency detector. Please clarify what the novelty is. 

Human texts that are locally formulaic can recover with high concentration, and AI texts prompted can recover poorly. Please add counterexamples.

Please add details about how variance over draws affects false decisions.

Please add robustness tests to punctuation-free text, micro-edits inside tokens, and code-mixed text.

LLMs often “fix” grammar, normalize quotes, or expand contractions automatically, even with the prompt. Please add alternative prompts and instruction styles experiments.

Add sensitivity to the alignment algorithm and ties.

Detectors can learn prompt artifacts rather than authorship. Please add prompts and cross-prompt and cross-domain OOD experiments where the calibration classifier never sees those prompts.

Add token-level cost and latency in RAIDAR and Fast-DetectGPT and a throughput plot across lengths. Include results with small local recoverers at realistic speeds.

Add precision/recall@policy thresholds, especially on short texts.

Add an ablation study to justify WCS as the best disruption.

Use BERTScore for embedding-free semantics as an ablation study.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides a new framework for detecting AI-generated texts, called D&R. D&R consists of two distinct functionalities: Disruption, which permutes the given text while preserving the core semantics, and Recovery, which attempts to regenerate the original text using an LLM. The authors argued that such D&R procedure can capture distributional differences between human text and AI-generated texts, due to posterior concentration of reproduced texts. Using a preliminary analysis and a short theoretical analysis, they attempted to support their claims. And with an experiment, they showed that D&R outperforms previous benchmarks and shows robust performance regardless of source models and datasets even if the model calls LLMs just one time.

### Strengths
- Suggest a new method that uses a single LLM call
- Providing a sufficient analysis that strengthen their argument
- Providing a sufficient experimental result showing robustness and high-performance of their model

### Weaknesses
- Need more clarity. It is not clearly stated whether assumptions in Theorem 2 holds theoretically; the assumption "seems" to hold based on the empirical analysis shown in Figure 2. Also, the usage of translation model is not clearly stated before starting the experiment. 
- Another similar work [1] is missing, which calls LLMs just twice. Although the authors made single-LLM-call detector, as the reproduction concept and the underlying assumption is similar, the authors should discuss the similarity between their work and [1].

[1] H. Park, et al., DART: An AIGT Detector using AMR of Rephrased Text, NAACL 2025

### Questions
## Question A. Clarity

A1. Does the assumption of Theorem 2 hold anytime? It seems the assumptions hold based on sanity check results, but I think theoretical analysis should provide clear interpretation when the assumption holds. Though the current mathematical formula is clear and easy to follow, assumptions make me raise some questions about the chance of those cases.

A2. It seems that the authors used DeepSeek-v3 for the main results (before changing the transformation model). Is this correct? I'm asking this because it is not clearly stated before line 377.

## Question B. Similarity to [1]

B1. Two papers (this paper and [1]) both argue that rewriting of given text can reveal the identity of source because AI and human writing is different. Also, both paper attempt to measure semantic similarity. Though I understand there exist several differences between them, could the authors clearly state them in the paper to help the readers understand? Other than the repetitive calls used in [1] and disruption procedure used in this paper, the approach of semantic measurement seems different. How does such difference affect the result?

B2. Compare to [1], it seems that disruption procedure can possibly generate semantically unsimilar text. For example, "A loves B" has different semantics to "B loves A". Therefore the permutation might introduce semantic differences, when I read this paper with the perspective of meaning representation methods. So, can we strictly say that those disruption actually "preserves" the semantics? I understand that the method provides semantically similar texts in general and the example is very special case. Though, I think this should be clearly noted in the paper, as the current version claims that D&R can preserve semantics (without any warning or limitation statements).

B3. Similar to B2, in the context of meaning representation field, researchers have been said that pragmatic similarity (e.g., BERTScore) can be easily affected by external contexts other than the internal semantics (so it might be improper to model the semantic similarity between them; see [2]). For example, "I love you" and "I don't hate you" could be semantically similar in terms of pragmatics but the semantic representation might differ. Although I understand that the disruption technique seldom introduces such changes, as there exists a chance of such unwanted effects, I think this should be warned to the readers. 

[2] K. Ki et al., Inspecting Soundness of AMR Similarity Metrics in terms of Equivalence and Inequivalence, *SEM2024

### Soundness
3

### Presentation
3

### Contribution
3
