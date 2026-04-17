# Can Interpretation Predict Behavior on Unseen Data?

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 2, 6

## Abstract
Interpretability research often aims to predict how a model will respond to targeted interventions on specific mechanisms. However, it rarely predicts how a model will respond to unseen input data. This paper explores the promises and challenges of interpretability as a tool for predicting out-of-distribution (OOD) model behavior.  Specifically, we investigate the correspondence between attention patterns and OOD generalization in hundreds of Transformer models independently trained on a synthetic classification task. These models exhibit several distinct systematic generalization rules OOD, forming a diverse population for correlational analysis. In this setting, we find that simple observational tools from interpretability can predict OOD performance. In particular,  when in-distribution attention exhibits hierarchical patterns, the model is likely to generalize hierarchically on OOD data---even when the rule's implementation does not rely on these hierarchical patterns, according to ablation tests. Our findings offer a proof-of-concept to motivate further interpretability work on predicting unseen model behavior.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper argues that by examining how Transformers allocate attention on standard (in-distribution) inputs, we can predict which rule they’ll apply to unseen (out-of-distribution) cases. Using a toy parentheses-balancing task, the authors train hundreds of small Transformers varying depth, width, regularization, and random seeds, then evaluate which rule each model actually follows OOD. They find that models cluster by their OOD rule, attention patterns can forecast that rule, and correlation does not guarantee causation.

### Strengths
1.	Good and ambitious motivation.
2.	Although it uses a toy setup, it includes many models and experiments, and some findings are interesting.
3.	The paper is well written, with accurate, clear descriptions and excellent details.

### Weaknesses
1. The experimental method relies solely on a simplified parentheses-balancing task. This narrow setup may limit the generality of the conclusions.
2. While the findings (e.g., “independently trained models cluster around systematic generalization rules”) are interesting, the paper would benefit from demonstrating at least one concrete example or use case that shows how such findings could be useful in practical applications or improvements for model designs.
3. The dataset used in evaluation is large, but I'm not entirely sure whether the empirical evidence is sufficient to support the general claims.

### Questions
1. Could the authors elaborate more on the design choice of using the parentheses-balancing task? Why is it an appropriate setting for testing interpretability and generalization?
2. Would the observed findings—such as the clustering of independently trained models around systematic generalization rules—also hold for other tasks or architectures beyond this synthetic setup?
3. How do the identified patterns or correlations facilitate specific downstream applications or improvements in model designs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to utilize interpretability as a tool to predict model outputs on OOD, unseen input data. The problem setting explored here is a synthetic classification task on parentheses strings, where the ID training data is designed to be learnable from either an Equal-Count or Nested rule. However, the OOD test data of most interest for this problem conforms to the Equal-Count rule only; as a result, models that have internalized the Equal-Count (OOD-relevant) rule are expected to perform better on OOD inputs than those which have not. The authors demonstrate that the models indeed show evidence of learning these rules in their attention patterns (identifying several relevant design decisions), as well as other heuristics, and that there is resulting explanatory as well as predictive power from what a model has learned, specifically in predicting its performance on the (carefully defined, for this problem) OOD inputs. They also show that the common practice of ablating proposed explanatory mechanisms is data-dependent, and does not work as conventionally applied, on their OOD inputs. They claim this framework can be useful beyond the specific problem they explore.

### Strengths
I believe the experiments are interesting and may be sound (difficult to evaluate given the poor presentation).

Section 4, with the experiments and results around vestigial circuits and factors in rule-selection, is interesting and readable. Section 5 (also well-named) also reads better than Sections 1-3, but it was a struggle to understand the problem framing and experimental setting, so it is difficult for me to comment on soundness. The observation that the common setting of ablating proposed explanatory mechanisms is data-dependent, i.e. robust to ID data but not OOD data, would definitely be of interest to the community if the rest of the paper were more understandable. In my opinion, this last finding is the most interesting and, currently, the best-substantiated.

The figures are generally helpful in providing the appropriate intuition to understand the paper, but I would urge the authors to still be much clearer in the text.

### Weaknesses
MAJOR: The paper is needlessly hard to follow and feels vague in many places. It makes for a frustrated reader. Many of the questions I have about this paper are probably due to the poor exposition of the motivation, concepts, and experimental setting. The paper flip-flops awkwardly between tedious details and broad, vague conceptual or epistemic statements, making it difficult to follow, evaluate, or build on the work presented.

MAJOR: The paper is missing precise statements to guide the reader through the authors’ motivation and justification for experiments/analyses presented. For example, they should tell the reader upfront that they are first interpreting how Transformers internalize the classification rules needed to separate the training (ID) data, and then using this understanding to model behavior on OOD inputs. The current intro is disjointed and hard to follow in a first read. Without a clear outline, there’s no point in reading the subsequent sections.

MAJOR: The paper uses only a synthetic classification task on parentheses strings for validation and illustration. It is unclear how this extends to data settings where we cannot simply look at the data/task and come up with a good rule that DNNs may or may not internalize when trained. While the authors admit it is a proof-of-concept work, they do not provide any recommendation for how this might be applied in more realistic settings, where we would need to understand what heuristics/rules may apply to the underlying problem. They don’t even state what would hypothetically be necessary to generalize the methods presented. 469-470 “If we identify similar cases in real-world settings…” – how would one begin to do this?
Related to this point: 363-364 The authors state “If we claim to understand a model, we should know its behavior under many unseen conditions” but according to Table 2 on Page 15, they do not even test the very simple opposite setting of OOD data with Equal-Count false and Nested true.

The sections are also disjointed and confusingly repetitive. The two sections (3.1.1. and 3.2.1) both called Experimental Details are confusing (one is under Data and the other under Models, I guess). But the so-called “Details” sections are not actually helpful. Ideally, you have a conceptually clear description with salient details in the main text, and full details in an Appendix or later Methods section. In your case, there is actually not much detail provided, and you could collapse 3.1.1 into 3.1 and 3.2.1 into 3.2 – you’d probably save space. The paper consistently uses vague wording. Examples from the sections mentioned: “models” without specifying DNNs, then “Transformers”, then “a population of classifier models based on the miniGPT architecture with hidden dimension 64” which is still not a full description of the model (“based on”?). 

(not affecting score) Why is Vaswani et al. cited with the year 2023? Everyone knows it as a 2017 paper, and the authors don’t seem to be specifically referring to any concepts from a newer version.

### Questions
What is the data exactly? Four paragraphs in, the authors are describing Equal-Count and Nested but have not properly explained the task setting or what the input data looks like (I see red and green parentheses in Fig 1, but no description anywhere). Even something broad like “classifying on strings” would help guide the reader–the authors can probably do better than that.

Why is the synthetic classification task not trivial? I appreciate the citations, but perhaps add a sentence about the standard settings before lines 148-149. Why Transformers needed to model this? (For example, if Transformers are being used simply for the purpose of attention-based interpretability to see whether a model has learned the logical rules of Equal-Count and/or Nested, this choice should be properly explained and justified).

In the Philosophical Motivation section, how did the authors choose which statements needed citations? No citations are provided for what they say about “other sciences” and “genetics”. In general, this section does not convincingly add value to the paper; though it attempts to situate the authors’ perspective, it is not precise or well-substantiated (in part due to the lack of citations for many of the statements given).

What does it mean for the training dataset to be “compatible” with either of the two rules? the authors repeat this several times, but the meaning doesn’t become clear because they first say:
148-150 “Unlike standard parentheses-balancing settings, our training dataset is compatible with either EQUAL-COUNT, an unordered counting rule, or NESTED, a hierarchical parentheses-balancing rule.”
and then
162-163 “Our training set is compatible with both EQUAL-COUNT and NESTED: every input sequence satisfies either both or neither of Equations 1 and 2.”
which at first seems like a contradiction: XOR vs (both or neither).
However, after re-reading the paper a few times, I think they mean to say that a model which has learned either rule (or both?) should be able to correctly classify the ID data, since all ID data points satisfy either both or neither of the constraints. In contrast, the OOD data is not classifiable from the Nested rule, only the Equal-Count rule (as in Fig 1). This is not evident from how they phrase “compatible” or "ambiguous rule” and should be properly clarified.

Where are the “OOD output probabilities” coming from? Can this be defined better since it’s so important for the clustering in Section 4?

I believe you should be clearer about stating that you train only on ID but most interested in investigating and testing on OOD.

The authors say the method is meant to be correlational and holistic. They also talk a lot about causality, and a broad conclusion of this work seems to be that even what we sometimes consider causal in XAI (interpretability ablations) is actually not strictly causal under OOD data. For a stronger paper, the authors should make more precise claims, and perhaps better separated claims, about 1) what they can interpret from an attention-based model trained in this synthetic setting where it learns hard rules or heuristics, and 2) what the implications are for current interpretability practices, from the finding that ablation analyses intended to demonstrate mechanistic causality are in fact data-dependent.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors aim to explore whether interpretability can be used to predict OOD generalization for unseen data. They introduce EQUAL-COUNT and NESTED rules into a synthetic parentheses dataset to investigate if interpretations of in-distribution data could predict the OOD behavior on the testing set. I think the topic is quite relevant and important for the community, looking at how interpretability can be used to analyze the model behavior. However, the authors only tested the idea on Transformer models trained on a synthetic dataset, which may not be sufficient to validate the idea reliably.

### Strengths
1. Using interpretability to analyze the model behavior could be an interesting topic.
2. The authors provide the dataset and code with detailed experimental settings for good reproducibility.

### Weaknesses
1. The title can be misleading. The authors mainly look at the OOD generalization, and only test the miniGPT type transformer model on a specific synthetic dataset. To me, it is not appropriate to use a general phrase “model behavior”.
2. Also, with a transformer-based architecture with different hyperparameters that can influence the OOD generalization, I think it shouldn’t have been stated “hundreds of models”.
3. All the results presented in the paper rely heavily on specific synthetic dataset configurations and the transformer model architecture with limited hyperparameters. I would assume that this method may not be easily applicable to real-world problems. It is unclear whether the key conclusions of the paper will still remain valid when applied to real-world scenarios across different model architectures.
4. The authors mention that a head is a hierarchical head if it tracks depth on at least 80% of mixed-depth inputs. How sensitive is the method to this threshold? The paper did not justify this threshold or investigate the sensitivity.
5. The authors take space to discuss the philosophical motivation. However, I did not see a tight link based on their empirical results.

### Questions
1. Did the authors try to adapt the proposed method to real-world problems?
2. Can the authors discuss whether the main conclusions will remain valid in more complex cases?
3. Did the authors consider other quantitative evaluations regarding the OOD generalizability?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper investigates whether attention patterns in Transformers can be used to predict how the model will act on Out-of-Distribution (OOD) data. While the experiments show that attention patterns do **correlate** with OOD behavior, they are not necessarily the **cause** of the behavior (attention ablation does not always inhibit the behavior). Finally, the paper cautions that ablation (interventions) on Neural Network architectures should be done on both In-Distribution and OOD data since the results of the ablation can differ considerably.

### Strengths
The paper has a great presentation. The Figures are of high quality, the text is easy to read, and the experiments are well explained.

The paper tackles an important challenge: whether we can use attention heads to predict how a Transformer will act "in the wild". Notably, the fact that attention ablation has no effect In-Distribution, but can have unpredictable behaviors Out-of-Distribution is an important result. It warns explainability researchers that only evaluating explainability methods on In-Distribution data does not give the full picture of the model. This observation might influence research in other areas e.g. explainability in computer vision.

While the paper focuses on a simplified setting (simple data and models), it pushes this experimental setup to the limit. The experiments investigate various model depths, weight decays, number of attention heads (Appx C.2). Transformers are also compared with LSTMs in Appx C.1, showing that LSTMs are unable to learn the "Equal-Count" rule.

### Weaknesses
## Incomplete Related Work

While Section 2 is a great read, I think it is too high-level for ICLR. I would rather motivate the current work by having a Related Work section that discusses in more depth the papers from the introduction. It would be interesting to describe what is activation steering, activation patching, and Sparse Autoencoder (SAE), and their limitations when it comes to OOD data. For instance, the work of (Kisanne et al. 2024, Smith et al. 2025) (line 48 of the manuscript) focus on limitations of SAEs. Is there existing work that shows limitations of other techniques (activation steering and patching) when it comes to OOD data?

## Citations

Some citations are not correct. For example, the paper "Attention is all you Need" is cited with year 2023 while the paper was published in 2017. Other papers are cited using their Arxiv version while they were published e.g. "Extracting Latent Steering Vectors from Pretrained Language Models" was published at ACL 2022. The citations should be corrected in the final manuscript.

## Confusing Terminology

The paper employs a lot of new terminology which can be hard to follow. Appendix A helped me a lot but it is not referenced in the manuscript. I stumbled upon it by chance. Adding a reference to Appx A would help greatly.

### Questions
How sensitive are some of the conclusions to the length of the sequences? Are there some heads that track depth on short sequences but not on large ones?

Are there other ways to perform ablation of attention head? Perhaps forcing a token to only attend to its immediate neighbor is an interesting alternative that inhibits the network from using long-term dependencies.

### Soundness
3

### Presentation
4

### Contribution
3
