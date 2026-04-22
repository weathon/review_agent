# Token Alignment Heads: Unveiling Attention's Role in LLM Multilingual Translation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6

## Abstract
Recently, large language models (LLMs) have made remarkable progress, with multilingual capability emerging as a core foundational strengths. However, the internal mechanisms by which these models perform translation remain incompletely understood. In this paper, we elucidate the relationship between the attention mechanism in LLMs and their translation abilities. We find that certain attention heads, which we term token alignment heads, are specifically responsible for mapping tokens from the source language to the target language during inference. 
Through a systematic investigation across various models, we confirm that these token alignment heads exhibit several key characteristics: (1) Universality: They are present in all LLMs we studied. (2) Sparsity: They constitute only a small fraction of all attention heads. (3) Consistency: The set of token alignment heads activated by the model shows strong consistency across different language pairs. (4) Causality: Interventionally removing these heads leads to a sharp decline in the model's translation performance, while randomly removing non-token alignment heads has little impact on translation ability. (5) Functional Specificity: Ablating token alignment heads disproportionately harms translation but has a varied impact on other multilingual tasks. We also traced the formation of token alignment heads during pre-training, revealing an evolutionary path of rapid proliferation, stabilization, and eventual pruning. Furthermore we leverage these token alignment heads to filter multilingual training data, and our experiments show that these data could enhance translation capabilities of the models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The presented paper analyzes that role of specific attention heads for the translation task aka translation heads. These are identified with a word alignment approach and thoroughly analyzed for their location, sparsity, language generalization, emergence, as well as impact on multilingual evaluation benchmarks.

### Strengths
- Interesting way of determining translation heads via word alignment properties compared to downstream task performance as done in previous works
- Insightful analysis on the location, sparsity, language generalization, and emergence of translation heads
- Insightful analysis for masking translation heads on downstream tasks including not only translation but also general multilingual benchmarks

### Weaknesses
### Weaknesses

- It would've been interesting to analyze the number of translation heads that are specialized for a specific language / language-pair despite them having a large Jaccard index i.e. `de2ja` seems to have a lower similarity to the other language pairs. Generally, it could be that some language pairs are deemed "harder" and need specialized translation heads. I'm a bit surprised to see that `en2de` and `en2nl`, two very closely related language pairs, have the same overlap ratio as `en2de` and `en2zh`, two fairly distant language pairs. It could also be that the deciding factor here is `en` as the source language, rather than the target language.

### Minor Comments

- The "token mapping" is the task of word alignment and has been around for many years in machine translation literature and probably should be mentioned as such.
- All Figures should likely be included as svg graphics instead of low resolution png images since it makes it more appealing to view details. In Figure 3 and Figure 4 some of the labels are way too small to even view properly. A bit more effort on properly scaling the figures would improve the reading experience.

### Questions
N/A

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
This paper utilizes token-aligned attention scores to detect salient translation heads (TH) in LLMs. The authors find that such attention heads significantly affect multilingual proficiency, as validated by the multilingual dataset FLORES-101. Experiments demonstrate substantial overlapping of salient heads across 10 different language pairs and confirm the deterministic effect of these heads through comprehensive ablation studies.

### Strengths
1. This work presents a token-aligned detection approach with unsupervised LLM labelling that requires inference only. It demonstrates the significance of THs through a masking approach for machine translation.
2. This work illustrates the properties of THs through detailed analysis across different models and languages.
3. This work unveils the learning process for translation heads during model training.

### Weaknesses
1. This work's organization lacks rigorous experimental settings. For instance, Figure 1 fails to adequately demonstrate the function of THs through its single example. Additionally, the selection of models appears arbitrary, lacking consistency in either model size or type.
2. The figures throughout the paper are blurry, making it difficult to discern key numerical values (particularly in Figures 3 and 4).
3. The proposed data augmentation method does not demonstrate significant improvements in Table 1 when compared to the baselines.
4. Despite numerous recent studies exploring the internal mechanisms of multilingual capacity in LLMs, this work lacks a thorough review of related work in this field.

### Questions
1. Given that this is not the first study to explore the multilingual abilities of LLMs, what are the key distinctions between this work and previous research?
2. How do current machine translation metrics demonstrate the significance of translation heads on overall MT performance?
3. Since BLEU and chrF++ metrics cannot inherently recognize languages, is it possible that these translation heads are not fully responsible for multilingual capacity if LLMs cannot function properly without them?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper investigates the role of translation heads in translation and multilingual tasks. Translation heads are identified based on their accuracy in token mapping. The authors identify several features of these translation heads, including i) that they are universal in all LLMs, ii) that they only a few translation heads needed, iii) that the translation heads are consistent across language pairs, iv) translation capability is severely impacted when these heads are masked and v) that they occasionally have an impact of other multilingual tasks. Finally, they trace the development of these heads across cycles of pretraining and show how the attention heads can be used to filter multilingual data for more efficient training.

### Strengths
1. The idea that cross-lingual word-mapping heads exist in multilingual LLMs is interesting, and how they demonstrate impact on translation and some multilingual tasks.
2. The experiments that evaluate key features of the heads are thorough and comprehensive.
3. The writing is clear. All motivations and experimental designs are easy to understand.

### Weaknesses
1. Translation heads are identified solely through cross-lingual one-to-one token mapping. However, translation as a process involves more than token mapping alone, e.g., resolving morphological ambiguity. Current name that is translation heads sounds a little misleading.
2. Previous literature show that transformer-based models tend to process multilingual input in English in the middle layers:
- https://aclanthology.org/2024.acl-long.820/
- https://arxiv.org/abs/2411.04986
- https://arxiv.org/abs/2402.18815
This seems incompatible with your finding in Figure 4 where translation heads are mostly gathered in the middle. What roles exactly do translation heads in the middle layer play in solving multilingual queries?
3. While the role of translation heads is clear on translation task, the results on other multilingual task are less conclusive. There is no clear analysis leading us to understand why this is the case, or the kind of role the attention heads play in solving these tasks.
4. Minor: fonts are too small for all figures.

### Questions
1. Can you reconcile your finding in Figure 4 with prior literature? see Weakness 2.
2. In some cases (Figure 9, xwinograd, xnli), pruning translation heads seem to give better results, why?
3. What is L in Equation 4?

### Soundness
3

### Presentation
3

### Contribution
3
