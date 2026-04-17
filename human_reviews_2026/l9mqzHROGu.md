# HLD: Approximate Hierarchical Linguistic Distribution Modeling for LLM-Generated Text Detection

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
The widespread deployment of large language models (LLMs) has made the reliable detection of AI-generated text a crucial task. However, existing zero-shot detectors typically rely on proxy models to approximate probability distributions of unknown source models at a single token level. Such approaches limit detection effectiveness and make the results highly sensitive to the choice of proxy models. In contrast, supervised classifiers are often detected as black boxes, sacrificing interpretability in the detection process. To address these limitations, we propose a novel detection framework that identifies LLM-generated text by approximating **H**ierarchical **L**inguistic **D**istributions--**HLD-Detector**. Specifically, we leverage n-grams to capture the feature distribution of human-written and machine-generated text across the word, syntactic, and semantic levels, and perform LLM-generated text detection by comparing these distributions under the Bayesian theory. 
By progressively modeling the linguistic distribution from shallow-level (token/word), then medium-level (syntactic), and ultimately high-level (semantic representations), our method mitigates the shortcomings of previous single feature level detectors, improving both robustness and overall performance. Additionally, HLD-Detector requires only a small amount of offline corpus for distribution estimation, instead of relying on online approximation with large proxy models, resulting in significantly lower computational overhead. Extensive experiments have verified the superiority of our method in detection tasks such as multi-llm and multi-domain scenarios, achieving the current SOTA performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a novel and effective AI text detection method that models the differences between "Human-Written Text (HWT)" and "Machine-Generated Text (MGT)" as hierarchical linguistic distributions across layers (lexical/morphological → syntactic → semantic). At each layer, they compute features using approximate n-grams, and finally use XGBoost to fuse the four-dimensional features for binary classification.

### Strengths
1. Excellent motivation: Zero-shot detection mostly relies on proxy models' token probabilities, which is fragile and computationally expensive; while purely supervised classifiers are accurate, they suffer from weak interpretability and poor generalization. Therefore, the authors cleverly combine perspectives from both approaches.
2. Comprehensive and detailed method comparison
3. Does not rely on source model probabilities, avoiding API access and proxy mismatch issues
4. Provides time efficiency analysis at the end, which is essential in realistic development

### Weaknesses
1. Limited LLMs in comparison, and the GPT-3.5 model is already outdated - should add newer models. I understand that adding new models might require extensive experiments, that's fine - the authors could just conduct small-scale experiments to demonstrate effectiveness would be fine.
2. In Table 1, the authors distinguish between zero-shot and supervised detectors. Table 2 and subsequent tables should also make this distinction, for example, using an asterisk to denote zero-shot methods, etc.

### Questions
How does the method perform on more recent LLMs like GPT-5?

How does the method handle mixed text (partially human-written, partially AI-generated)?

### Soundness
3

### Presentation
4

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
The author proposes HLD-Detector: the novel hierarchical linguistic distribution comparison framework that explicitly models differences across three levels—word, syntax, and semantics—using Bayesian likelihood ratio. This framework simultaneously improves detection accuracy, robustness, interpretability, and efficiency.

### Strengths
The authors systematically and fully support his hypothesis through comprehensive experiments: 
HLD-Detector leads across all benchmarks for DetectRL, including multi-LM and multi-domain.
The SOTA performance of cross-domain/cross-model and attack resistance, validates its robustness; Significantly reduced performance demonstrates the contribution of linguistic feature at each level of its design; In addition, multiple heatmaps allow direct viewing of AI trace phrases/structures.

### Weaknesses
1. With the evolution of generative models, the effectiveness of HLD has not been explored when LLM commonly adopts non-maximum likelihood sampling strategies (such as models after RL).

2. While HLD offers advantages in accuracy and interpretability, there is still room for improvement in computational efficiency compared to RoBERTa-base.

### Questions
1.Based on the experimental results in Table 3, HLD still appears to be "vulnerable" to domain drift.  Are there any online/offline mechanisms that can improve its robustness? Have you explored which of the three levels of linguistic features might be most affected by domain shift?

2.Some more implementation details of HLD are missing, like the hyperparameters smoothing factor $\delta$ , $\tau_{ctx}$ , etc.

### Soundness
3

### Presentation
3

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
This paper proposes a new model for detecting machine generated texts, called HLD-detector. HLD-detector uses word-level differences, syntactic-level differences, and semantic embeddings as its features, and adopts kNN database for estimate density of those features. With XGBoost classifier, the authors claimed that their model achieved highest performance in multi-LLM setting and multi-domain setting. Also, they claimed HLD-detector showed robust performance on cross-domain, cross-model, and diverse attack scenarios. An ablation study showed that those proposed features work well in detecting MGTs. In addition to these experiments, the authors suggested other experimental results such as interpretability, impact of n-gram or text length, impact of data scale, Chinese MGTs, and efficiency of their methods, to emphasize their contribution.

### Strengths
- Kernel density estimation methods for MGT detection, using a simple component of a text, n-gram.
- Multi-dimensional experiments that reveals the actual applicability of HLD-detector.

### Weaknesses
- Though the authors attempted to follow DetectRL benchmark, the source models seem outdated. Because recent models learned more human-like wordings and phrases, recently researchers thought that the syntactic-level features might not suitable to detect MGTs. Refer to [1] and [2]. Also see Question A.

[1] Z. Kim, et al. Threads of Subtlety: Detecting Machine-Generated Texts Through Discourse Motifs. arXiv preprint arXiv:2402.10586 (2024).
[2] H. Park et al., DART : An AIGT Detector using AMR of Rephrased Text, NAACL 2025

### Questions
## Question A. Source models

A1. The naturalness of text generated by source models might affect the result. Recent advances in language modeling proceeds to generate more human-like texts, and this continuously introduces challenges in MGT detection. Although DetectRL can reveal the performance and possibility of detecting MGTs, a reader might ask whether these models can be successfully applicable to recent models. I understand that the authors tried to claim that their model could be applied to recent models with cross-model tests, I recommend adding one more experiments showing whether the proposed model could be successfully detect GPT-4 or Claude 3.5, even if they were trained on previous version of those families.
A1-1. In the same vein, the table might show a kind of ceiling effect: All performance scores were near 100. What if we ran the same experiment on more complex or difficult datasets?

A2. Usually, we don't know source models exactly. When discriminating MGTs from HWTs, a user could not know which source model is a possible source of given text. So, models like RAIDAR or [2] attempted to expand the problem into multi-class classification problem instead of a binary-class problem. Could this model be extended to those cases easily?

## Question B. N-grams and semantics.

B1. As a researcher also working on semantic representations, claiming "N-grams" have "high-level semantics" seems inappropriate. This is because N-gram and its semantic embedding only mirror its pragmatic context, rather than the deep semantics (such as events or participants), and N-gram often cannot model long-distance dependencies. And, this is why [1] and [2] attempted to adopt higher-level of semantic representations in detecting MGTs. Why didn't the authors consider such high-level semantic representations?

B2. Authors noted that the designed features contributed to the performance, in Table 4. But the score seems not much differ, especially on XSum. This makes me raise two following questions.
B2-1. Are the differences statistically significant? Could we claim that the difference was not produced by chance (i.e., Type I error)?
B2-2. Why do the features behave differently across domains? Do the authors have any other explanations on that? Answering this might provide some valuable insights to the readers also.

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
4

### Summary
This paper introduces HLD-Detector, a hierarchical linguistic distribution framework for detecting machine-generated text. Instead of relying on proxy-model probabilities or opaque neural classifiers, the method explicitly models word-level, syntactic, and semantic distribution differences using n-gram statistics and a KDE-based semantic estimator, then combines them via an XGBoost classifier. Experiments on DetectRL show strong performance across multi-LLM, multi-domain, transfer, and adversarial settings, with notable efficiency and interpretability advantages. Ablations confirm the complementary roles of each linguistic layer.

### Strengths
The paper has a good motivation, whose hybrid design combining interpretable distribution modeling with supervised fusion, bridging the gap between zero-shot methods and black-box supervised classifiers.

The paper is also easy to follow.

### Weaknesses
1. Although the authors cited top-tier recent LLMs like DeepSeek-r1, unfortunately empirical evaluations cover early LLMs like Claude,PaLM-2, and LLaMA-2. As a paper in late 2025, I think it is necessary for an AI text detector to include latest LLMs, including: Qwen3, DeepSeek-V3/R1, GPT4, et cetera.

2. Besides, I am curious why Roberta-base, the simplest baseline, outperforms nearly all other methods in Table 3. Is that a fair comparison? I guess the reason is that the training corpus that is used for training Roberta is a strong one, hinting potential unfairness.

3. Unfortunately, the method is confined to English. I think more languages could be included to improve the generality of the proposed method.

### Questions
Now commercial detectors are widely used, is the proposed HLD method comparable to these detectors, e.g. GPT-Zero and ZeroGPT?

### Soundness
3

### Presentation
3

### Contribution
3
