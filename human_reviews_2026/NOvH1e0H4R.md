# SADA: Safe and Adaptive Inference with Multiple Black-Box Predictions

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 8, 4, 4

## Abstract
Real-world applications often face scarce labeled data due to the high cost and time requirements of gold-standard experiments, whereas unlabeled data are typically abundant. With the growing adoption of machine learning techniques, it has become increasingly feasible to generate multiple predicted labels using a variety of models and algorithms, including deep learning, large language models, and generative AI. In this paper, we propose a novel approach that safely and adaptively aggregates multiple black-box predictions with unknown quality while preserving valid statistical inference. Our method provides two key guarantees: (i) it never performs worse than using the labeled data alone, regardless of the quality of the predictions; and (ii) if any one of the predictions (without knowing which one) perfectly fits the ground truth, the algorithm adaptively exploits this to achieve either a faster convergence rate or the semiparametric efficiency bound. We demonstrate the effectiveness of the proposed algorithm through experiments on both synthetic and benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SADA (Safe and Adaptive Data Aggregation) — a novel framework for safe and adaptive semi-supervised inference that aggregates predictions from multiple black-box models (e.g., LLMs, deep networks) of unknown quality. The method aims to guarantee:
- Safety: Never performs worse (in mean squared error) than using labeled data alone — even if all predictions are poor.
- Adaptivity: If any one of the black-box predictions is highly accurate (without knowing which), SADA automatically leverages it to achieve semiparametric efficiency or a faster convergence rate.

SADA extends recent prediction-powered inference (PPI) work by Angelopoulos et al. (2023, 2024) to handle multiple predictors simultaneously, offering both theoretical guarantees and empirical validation.
Experiments on synthetic and real data (Wine reviews, Politeness datasets) show that SADA consistently outperforms naive, PPI, and PPI++ estimators — providing stable variance reduction and robust adaptation across scenarios.

### Strengths
- The paper is generally well-written. The intuition behind SADA is well-explained and intuitive. The connection to PPI and PPI++ is made explicit, situating SADA as a generalization.
- Theoretical contribution. Extends prediction-powered inference to multiple prediction sources under semi-supervised learning.
- Experiments on synthetic data and two real-world tasks demonstrates consistent improvement over PPI/PPI++ and robustness to poor prediction quality.

### Weaknesses
- Limited experimental depth. The benchmarks, while well-chosen, are relatively small-scale. There is no demonstration of SADA’s scalability to larger datasets and higher-dimensional parameters.
- Assumption realism.
It assumes $l_\theta(x; y)$ to be a convex loss function. 
The assumption of having multiple predictions with overlapping but uncorrelated noise may not hold when using correlated LLMs (e.g., GPT-4 and GPT-4o-mini). Empirical tests on correlated predictors would strengthen the case.
- The estimation of optimal weight (line 323) depends on the whole set of data points (N), which is not feasible for large N.

### Questions
see above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a method building on prediction-powered inference (PPI) to do unbiased inference when a small set of ground truth training labels and an assortment of synthetic prediction functions are available on unlabeled data. Theoretically and empirically, they demonstrate the strengths of this method as being no worse than a naive estimator, and being adaptively able to converge quickly to a good synthetic estimator if one exists.

### Strengths
- seems like a fairly clear and well-scoped contribution generalizing PPI++ to multiple predictors
- as far as I know it is novel, although someone who has more deeply read all of the PPI literature may disagre
- theory and experiments back up the general points made about the estimator

### Weaknesses
Not a ton of major weaknesses here - not ground breaking work but makes its point pretty cleanly afaict

- it is stated on L186 that predictions Y-hat do not need to have the same form either as each other or Y. The examples of categorical and continuous are given, but it's not clear how broadly this extends. I feels as though somehow they need to be operated on similarly but they can't be so different - for instance, Y-hat can't be a free-text output of an LLM it seems if Y is binary. Could use some clarification here
- In general, one useful baseline to look at here would be averaging the predictors and using that in PPI++ - this is a very natural thing to do when you have many predictors and would give a better sense I think of a strong ensemble-ish baseline
- Fig 2: Visually, we can see that the shape of the result curve in 2c is nice. It would be good to know, potentially in a table, if those results are actually better or worse than the individual PPI++ results
- would be good to run multiple seeds and show confidence bars in Fig 3/4, these experiments look promising but I want to know how significant the improvement is

small notes:
Line 36: should cite the LLMs (GPT, Llama, Deepseek)
L133: I get confused by column vector notations - clarify if this is the inner or outer product (the superscript cross product symbol you define)
Line 270: typo "propose" -> "proposed"
Fig 3: clarify that only the PPI lines differ between these subfigures, is that right? and the SADA lines are the same?

### Questions
What is the theoretical relationship to taking a weighted combination of the predictors as a function, using that in PPI++ and then optimizing the weights? This is not a high priority question but would help with some deeper understanding of the relationship of this method to PPI++

### Soundness
4

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
2

### Summary
This paper proposes an augmented M-estimation framework that leverages multiple auxiliary signals ("black-box predictions", possibly by pretrained general-purpose language models) to reduce asymptotic variance relative to using labeled data alone, and shows adaptivity under idealized conditions.
Technically, the core idea of adding mean-zero augmentation terms built from auxiliary signals to an unbiased estimator and choosing weights to minimize asymptotic variance is classical, with prediction-powered inference (PPI) and its variants as the closest modern instantiation.
The main value appears to be packaging these tools for multi-surrogate, vector-parameter settings and clarifying efficiency guarantees.
The proposed method was evaluated on synthetic data and natural language data using LLMs.

### Strengths
- This paper proposed a general framework with clear recipe for augmenting M-estimators with multiple auxiliary signals, subsuming PPI as a special case.
- The author provided asymptotic variance/MSE dominance over the labeled-only estimator.
- It's proven that the proposed method achieves oracle efficiency if one auxiliary is ideal.
- The projection view makes the weighting understandable and implementable.

### Weaknesses
## Contribution

The author stated that

> First, we consider a semi-supervised setting where multiple sets of predicted labels are available, without making any assumptions about their quality or requiring prior knowledge of which predictions are more accurate. The predicted labels are also not needed to share the same scale or format, either with each other or with the true labels. This bridges the gap between advanced machine learning tools and principled methods for leveraging them to improve the inference results.

However, this statement is true for many other existing papers.
Conceptual novelty is unclear relative to classical augmented estimating equations or GMM weighting.
The authors should precisely articulate what is new (e.g., theory for multi-surrogate vector targets beyond existing results, sharper bounds?)

## Problem framing

> Given multiple predicted labels with unknown quality, how can we aggregate them in a safe and data-adaptive manner to improve estimation and inference?

The problem framing over-emphasizes "black-box predictions" and ignores the equivalence to multi-annotator or measurement-error settings; the difference from **crowdsourcing/ensembling** is not clarified.
Framing the auxiliaries as "black-box predictions" (and hinting they may be from LLMs) rather than "noisy annotators/surrogates" is a marketing choice, not a mathematical distinction.
Please clarify differences from crowdsourced label aggregation and model ensembling.

## "Safe"

> The proposed method is guaranteed to perform **no worse than** (?) the naive estimator (using the labeled data alone) in terms of mean squared error, regardless of the choice of machine learning models or their prediction accuracy.

The term "safe" is overloaded and used here to mean **asymptotic efficiency dominance** rather than robustness.
It seems several existing papers also use the term "safe" this way (https://arxiv.org/abs/2011.14185 PPI and this paper cited, and https://arxiv.org/abs/2502.17741), but in my opinion it's very confusing.
The paper should explain it with more precise language (e.g., asymptotically at least as efficient as labels-only, PSD variance dominance?)

## Writing

- "GPT, Llama, DeepSeek" should have references.
- The introduction makes broad claims ("bridges the gap between advanced machine learning tools and principled methods") without precise, falsifiable statements.

## Future work

> Our method can be extended to the situations under distribution shift. In those settings, developing methods that are robust to distribution shift is essential for enhancing the reliability and practical effectiveness of semi-supervised learning.

This is very generic, and I don't see a connection between the current work (safety?) and the distribution shift literature.
Either drop the generic sentence from the conclusion or make it more concrete (choose a shift model, transport the estimating equations with density-ratio weights, suggest a variance-dominance result, etc.).

### Questions
- The meaning of "gold-standard experiments" is unclear in the machine learning context.
- "Outputs from different models–such as GPT, Llama, or DeepSeek–often differ, sometimes substantial; and the quality of predictions from black-box models can be highly variable." Please provide empirical evidence or references.
- "In particular, low-quality or poorly calibrated predictions can introduce significant noise, increasing variance and leading to unreliable inference." How siginificant? How unreliable?
- What does "perfectly accurate" mean in this context?

### Soundness
3

### Presentation
2

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
This paper looks to leverage the availability of unlabelled data in order to improve statistical inference. Previous works have demonstrated that producing quality synthetic labels for these unlabelled data has become even easier with the proliferation of Large Language Models (LLMs). However, previous works leveraging unlabelled data and synthetic labels make use of only a single predictor. This work uses the predictions from several different models simultaneously to achieve strong variance reduction while maintaining consistency. Asymptotic guarantees are provided, as well as experiments on real and synthetic data to demonstrate the method’s efficacy.

### Strengths
- The problem being addressed is interesting and highly relevant to researchers and practitioners alike, both of which would be interested in better inference methods.
- The mathematical set up is well explained, and the annotations on equations help with building intuition.
- Several relevant baselines are compared with both theoretically and empirically, with the theoretical benefits of the new method being especially well demonstrated.
- In the mean estimation case (as well as others), using the the OLS estimator equivalent to equation (4) is an inspired way of ensuring that a perfectly predictive set of synthetic labels will receive a weight of 1, while other predictions will remain unweighted. 
- The idea of leveraging multiple sets of predictions for the same inference problem is an interesting take on the typical PPI set up.

### Weaknesses
- [W1] The main weakness is that the method does not seem particularly different from PPI, PPI++, and stratified PPI. The same formula is used for finding the optimal weighting over the terms using predictions in the SADA estimator as for PPI++ in the K=1 case. Similarly, the idea of fitting multiple coefficients for multiple pools of data was explored in the stratified PPI paper, although there each of the coefficients are not fit simultaneously and not on data pools of the same size. Similarly, the guarantees around always performing better than the naive estimator are already guaranteed by PPI++, as already stated in this paper. 
- [W2] Despite requiring several more sets of predictions than PPI++, SADA does not seem to be able to exceed the performance of PPI++ on the most correlated set of predictions. This is on display in both Figure 2 and Figure 3. See [Q1]. I would have anticipated that by leveraging multiple pools of predictions simultaneously, we could produce performance better than the best PPI++ estimator.
- [W3] (Minor) There are typos in the work (line 28, or requires -> or require) (line 36, sometimes substantial -> sometimes substantially)

### Questions
- [Q1] It seems as if this method is equivalent to using PPI++ with the most correlated set of data. How would this compare with first testing which set of predictions is most correlated using the labelled dataset, and then running PPI++ with that most correlated set? Estimating this correlation is already part of the process of estimating the correlation coefficient, which is $\lambda$ in the PPI literature and $\omega$ in this paper.

### Soundness
3

### Presentation
3

### Contribution
1
