# Worse Together: Understanding the Brittleness of Multimodal Models on Rare Concept Pairs

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Multimodal models are being deployed in real-world settings where rare or unseen combinations of objects during pretraining are bound to appear at test time. Understanding how these models generalize to rare combinations of concepts is thus an important robustness problem. In this paper, we investigate how the pairwise co-occurrence of concepts in the pretraining dataset impacts CLIP and large multimodal model (LMM) performance on uncommon concept pairs. We measure concept co-occurrence with pointwise mutual information (PMI), which corrects for the correlation between single and paired concept frequencies. We show a strong correlation between PMI in the CLIP pretraining data and zero-shot accuracy in CLIP models trained on LAION-400M ($r=0.97$ and 14\% accuracy gap between images in the top and bottom 5\% of PMI values), and demonstrate that a simple PMI-based image edit can induce an accuracy drop of up to 10\% on images edited to contain low PMI pairs. We additionally find that this behavior in CLIP transfers to LMMs built on top of CLIP ($r=0.70$ for TextVQA, $r=0.62$ for VQAv2). Finally, we demonstrate that fine-tuning CLIP with augmented data covering a broad range of PMI values is a promising strategy to improve robustness on rare concept pairs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper conducts a series of empirical studies on how the co-occurrence of concepts affects the zero-shot performance of CLIP and LLaVA. It uses pointwise mutual information (PMI) to measure how likely two concepts are to be correlated in the pre-training datasets. Based on this, the authors create synthetic image datasets containing both high-PMI and low-PMI samples, and find an almost linear relationship between PMI values and zero-shot performance. This indicates that CLIP's predictions are not robust to the main concepts, but instead depend on this brittle relationship. The authors further use a low-PMI dataset to fine-tune CLIP, which increases zero-shot performance and transfers to other datasets. They propose that PMI-guided data curation can be used to improve robustness.

### Strengths
This paper presents a systematic study on the effects of the pre-training dataset. I find the strengths of this paper in the following aspects:

1. It establishes a linear relationship between PMI and zero-shot performance.

2. It introduces new datasets to investigate the robustness of CLIP models, and shows that fine-tuning on these datasets can transfer to other datasets.

3. It approaches the problem from a data perspective to address the dependency on co-occurring concepts.

### Weaknesses
1. Is this linear relationship between PMI and zero-shot performance dependent on the data curation? I mean, in line 185, does the selection threshold of 10,000 on frequency cause the linear relationship?

2. I am also confused about how Figures 2a and 2b are obtained. What does each dot represent? Do they correspond to a set of (c_accessory, c_ImageNet) concept pairs within a certain PMI range computed on LAION-400M, and then you test zero-shot performance on those concepts generated in GenPairs? Please provide a more detailed method explaining how the plots in your main results are produced.

3. The presentation needs improvement. For example, when I reached line 233, I still had no clear idea what a “key concept” is. Similarly, what is a “non-key concept pair”? These terms are not introduced before being used, which makes it hard for readers to follow.

4. Although I acknowledge that this paper provides a systematic study of the robustness of VLMs from a data perspective, the idea is not groundbreaking. Prior work on spurious correlations and fairness has similarly used counterfactual generation or disadvantaged-group data generation to improve robustness. The low-PMI-guided data construction shares a similar idea with these approaches.

### Questions
1. In lines 184–185, the authors describe how they extract concepts from LAION-400M. How do they avoid including non-noun or non-adjective words such as “a”, “the”, and “be”?

2. Since PMI between two concepts is symmetric, have you tried swapping the zero-shot prediction target to c_accessory? Would that produce a similar trend?

3. Flux.1-dev is a generative model, which seems to be tackling a harder task than zero-shot classification. I am curious why Flux.1-dev is able to generate low-PMI pairs, while CLIP struggles with them even in the zero-shot classification setting. I understand this question is hard to answer. I am asking mainly to discuss your perspective. Any insights or hypotheses would be helpful.

4. How does the fine-tuned model perform on concept pairs with high PMI? Does its performance decrease? What practical insight does this provide about curating training data for robustness? Since we cannot enumerate all low-PMI pairs in a dataset, what is the most efficient strategy?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper highlights a weakness of CLIP and LMMs: the performance is correlated to the Pointwise Mutual Information(PMI) of concept pairs in the training data. This work focusses on the failure of these models for rare combination of concepts, thus extending the current research being done on data-centric findings in vision-language. They also propose two benchmarks, GenPairs and ImageNet-Paste, for testing various concept pairs.

### Strengths
1. The study has a broad scope, which is very important. Data-centric studies should be applied to ALL model families that use them: hence extending the study beyond CLIP models to LMMs and from zero-shot classification to VQA is well-motivated.

2. Various experimental settings have been proposed, some novel. In addition to VQA and zero-shot classification, evaluation is done on GenPairs and ImageNet-Paste for multiple concepts. This is valuable for future research.

### Weaknesses
1. "While visual concepts can be difficult to define and extract from images": [1] has done exactly that. The concept extraction pipeline involves extracting visual concepts using an object tagger and provides a similar but more thorough text concept extraction function than this work. Additionally, [1] releases the concepts and artefacts for LAION-400M which is what the authors use. They could have just taken this metadata to run experiments. Having multimodal concepts allows for more fine-grained analysis, for example by taking the intersection of concepts per sample which ensures the concept exists in both image and text.

2. Here a concept is defined as single lemmatised words extracted from captions. This approach would not be able to distinguish polysemous words ("bat" the animal, "bat" the sports equipment). It also would not take into consideration multi-word concepts ("peanut butter"). 

3. Fine-grained PMI accuracy correlation experiments are essential but missing: it would be good to know how correlation varies when considering caption length, number of concepts in the caption, etc. Also one probe that would be interesting would be to correlate single frequency concept performance with pairwise performance (for example a high frequency concept paired with an extremely rare concept may outperform two low frequency concepts that are not that rare. A small ablation to show this could benefit the work).

4. I believe Eq 2 and 3 are incorrect: while estimating the empirical probability of a concept or pair of concepts in a dataset, it must be normalised by the number of documents not concepts (|D| instead of |C| and $\binom{|C|}{2}$). This affects the PMI calculation and also leads to asymmetric Laplace smoothing.

5. "models struggle to disentangle individual concepts and generalize to new combinations": the paper does not adequately establish this causality. Captions being unrelated/nosiy, synthetic image distribution bias in GenPairs, per-class frequency in pretraining could be factors that lead to that, not necessarily the co-occurence statistic.

6. Similar to point 5, the finding of LMMs being biased towards answering "yes" for high PMI inputs needs to be studied further. Specifically, does the bias come from the vision or text encoder? Recent works have shown that LMMs are over-reliant on the LLM component [2]. Suggested experiment: replace the image with noise and see if the answer is still "yes".

[1] Udandarao et al. No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance, NeurIPS 2024

[2] Vo et al. Vision Language Models are Biased, 2025

### Questions
1. Why did the authors implement a frequency filter as high as 10,000? This would remove long-tailed concepts, which would skew the study of compositional performance, thus biasing the study of brittleness to rare concept pairs. Is it possible to check the number of concept pairs removed when setting a threshold (for example get different numbers for different thresholds)

2. Why have two different Laplace smoothing coefficients for single concept and multi-concept probabilities?

3. Fine-tuning CLIP with concept pairs with a wide range of PMI values shows promising avenues for data curation. Do the authors think this is good practise for pretraining or general tasks?

### Soundness
2

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
- This paper investigates how rare or unseen combinations of visual concepts affect the performance of CLIP and  large multimodal models.
- The author introduces a Pointwise Mutual Information–based framework to measure how often concept pairs appear together in training captions and finds that uncommon pairs lead to much lower accuracy.
- Experiments on both synthetic and real edited images show a strong correlation between concept-pair PMI and zero-shot accuracy of CLIP and MLLMs, suggesting that data composition has a stronger impact than model size.

### Strengths
- The motivation of the paper is clear and important.
- The paper introduces an interpretable, data-driven way to analyze multimodal robustness by framing model brittleness through concept-pair co-occurrence statistics.
- The analysis is supported by clear numerical evidence that convincingly demonstrate the impact of concept-pair rarity on performance.
- The study proposes a lightweight fine-tuning approach guided by PMI balancing, effectively improving generalization without extra data collection.

### Weaknesses
- The study heavily rely on caption word pairs to estimate co-occurrence may miss visual relationships that are not described in text.
- There’s no human analysis to confirm whether low-PMI failures align with human perception of concept rarity.

### Questions
- How can we tell if PMI differences cause performance drops rather than semantic bias?
- The experiment only cover CLIP and few MLLMs. What's the result on some closed-source models like GPT-4?
- Does increasing model size or data diversity reduce the PMI–accuracy correlation, and at what point does this relationship stop holding?
- How sensitive are the findings to the choice of tokenization?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates how the co-occurrence statistics of concept pairs in CLIP’s pretraining data affect model robustness. The authors show a strong correlation between PMI of concept pairs and zero-shot accuracy, revealing that CLIP and LMMs struggle with rare or unseen concept combinations. They further demonstrate that fine-tuning with low-PMI (rare) pairs improves robustness and transfers to datasets. The paper highlights a key source of brittleness in multimodal models and provides a promising remedy.

### Strengths
- The paper presents a compelling study on how the co-occurrence statistics of concept pairs during pretraining affect the robustness of multimodal models such as CLIP and LMMs. The idea of quantifying this relationship through PMI is intuitive.

- The paper makes several interesting and valuable findings, including the strong correlation between PMI and zero-shot accuracy, the observation that fine-tuning with rare concepts can improve robustness and transfer to other datasets, and that LMMs exhibit similar patterns. These findings not only reveal the brittleness of current multimodal systems, but also provide insights into how to potentially mitigate it.

- They conducted many experiments to show strong empirical support for the claims.

### Weaknesses
- Regarding the GenPairs dataset construction, the generated images sometimes contain objects beyond the two intended concepts (e.g., in Figure 6, the image for the pair (coral, chambered nautilus) also includes other elements like a moon, and similarly for other images). Additionally, one could argue that those generated images with lower PMI values tend to have unusual backgrounds or environments as seen from the example images. These introduce potential confounders. The study does not fully control for these factors and therefore does not purely isolate the effect of PMI. So I’m not entirely sure if it is ideal to rely on these generated images for the main analysis.

- The results in Wiedemer et al. (2025) appear to have substantial overlap with this work. How this paper distinguishes itself from that prior study is not sufficiently discussed.

### Questions
Please see the questions raised in the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
3
