# RankGen: A Statistically Robust Framework for Ranking Generative Models Using Classifier-Based Metrics

- Avg Score: 3.20
- Decision: Reject
- Scores: 4, 0, 4, 4, 4

## Abstract
Standard metrics for evaluating generative models are brittle, easy to game, and often ignore task relevance. We introduce RankGen, a unified evaluation framework built on four metrics: Quality, Utility, Indistinguishability, and Similarity; each designed to capture a distinct failure mode and supported by PAC-style generalization bounds. RankGen follows a two-stage process: models that violate bounds are discarded, while the rest are ranked using robust, quantile-based summaries. The resulting composite score, Exchangeability, captures both fidelity and task relevance. By exposing hidden pathologies such as memorization, RankGen provides a principled foundation for safer model selection and deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces RankGen, a unified and statistically robust framework for evaluating and ranking generative models, designed to overcome the brittleness and shortcomings of standard, easy-to-game evaluation metrics. RankGen utilizes four PAC-style generalization-bound classifier-based metrics—Quality, Utility, Indistinguishability, and Similarity—to capture distinct failure modes like low fidelity, memorization, distribution shift, and mode collapse.

### Strengths
1. a new method different with the previous FID, IS, Precision&Recall
2. give the new definition for generative quality

### Weaknesses
1. The paper must clearly define the operational and conceptual distinction between Quality, Similarity, and Utility. . All are to discuss the diversity situation.
2. The paper uses "Quality" to assess distribution coverage (fit) rather than the standard meaning of sample fidelity (realism). This non-standard usage must be explicitly stated early on to avoid confusion with existing generative metrics (e.g., FID).
3. The discussion on the metric's diagnostic role, particularly concerning mode collapse (Sec 3.3, 3.6), needs significant expansion. This is a critical phenomenon that requires a deeper analysis, including differentiating between fidelity collapse and diversity collapse. The diagnostic role should be systematically extended to other failure modes (e.g., over-generalization, memorization).
4. similar to 3, the other Diagnostic role from sec 3.5 and 3.6  should also discuss more. 
5. In the paper, the reliance on two specific training sets, ($train_1$ and $train_2$), severely limits applicability.
Unlabeled Data: The authors must propose a methodology to guarantee the distinguishability of  $train_1$ and $train_2$  when working with unlabeled datasets.
Conditional Models: The current Indistinguishability task may be inappropriate for labeled/conditional generative models. The authors must state this limitation or provide an adaptation for such models.
6. Figure 2 is not clear. and figure 3 should give the score for the different models (more clear to see the rank)

### Questions
see the weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes four metrics to evaluate generative model performance on datasets including a classification label. The first measures the quality of generated data comparing the accuracies of classifiers trained on real and synthetic data. The second compares how much adding generated data to real data improves the accuracy of the classifier, compared to adding more real data. The third is how hard it is for a discriminator to distinguish real and generated data. The fourth measures how locally similar real and generated points are. The metrics are computed with multiple train-test splits to estimate uncertainty, and generators are ranked with a Monte-Carlo procedure taking the uncertainty into account. The proposed metrics are evaluated with some sanity checks with Gaussian features + binary label. The paper also evaluates several image and molecule generators with the proposed metrics.

### Strengths
The paper studies an important problem: generative model evaluation metrics are hard to interpret, and estimating their uncertainty is important.

### Weaknesses
The writing of the paper shows signs of heavy LLM use. Some examples:
- Line 159: the definition of $f^{(i)}$ does not make sense, since $i$ appears to be indexing over multiple train-test splits.
- Line 159: not clear why $D^y_{train}$ is a parameter of $f^{(i)}$.
- Symbols for the 4 metrics change in Section 3.7 from what was previously used.
- The paper states that many parameters like underlying classifier and dataset size are swept in the sanity check (Section 4.1), but the results in Table 3 are not given over the whole sweeps, and there is no indication that the numbers in Table 3 are aggregated over results from the sweeps.
- The numbers in Table 3 do not support the stated conclusions in lines 350-356.
- Utility PAC-bound in Section 3.4 does not match the bound that is proven in eq. A.1. The numerators are different.
- The proof of the similarity PAC-bound in Appendix 10 concludes with a different inequality than the one that is supposed to be proven (which appears before the concluding inequality).
- Line 272: formulas for quantile-to-moment conversion rules do not appear in Appendix 5 as stated.
- The last sentence of the paper is "The duplicate metric tables appearing in prior drafts have been **removed** to avoid redundancy." (emphasis from paper).

Besides, the evaluation of the proposed metrics is limited. The only actual evaluation of them is 5 sanity checks of perturbing data with Gaussian features. The rest of the experiments evaluate generative models with the new metrics, but these do not provide any evidence that the proposed metrics are useful since it is not possible to know what values a good metric would have. There are also no comparisons with previous metrics.

In addition, many important details are unclear:
- Line 235: not clear what "same domain" means, which makes the whole definition of the similarity metric impossible to understand.
- PAC bounds for similarity and indistinguishability are stated as a difference between the finite sample value and infinite sample value. It is not clear how one could compute the infinite-sample value to check that the bound is satisfied.
- Classifiers behind the metrics for molecule and image evaluations are not specified.
- Not clear how Rademacher complexities and VC-dimensions for the PAC-bounds are computed for actual classifiers.
- The "mode collapse" test (Appendix 12) doesn't really test mode collapse since replacing generated points with similar ones from real data preserves model. Testing mode collapse with unimodal Gaussian features is not possible in any case.
- The paper states that means and standard deviations of the metric values are not always reliable, and uses medians and interquartile ranges for this reason. But they are immediately converted to means and variances for the Monte-Carlo ranking procedure for some reason.
- Line 158: $D^y\_{train2}$ not defined.
- Line 225: $\mathrm{indist}^*$ is not defined until the Appendix.
- Line 235: $D\_{mix}$ not defined.

### Questions
See weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces four classifier‑based metrics—Quality, Utility, Indistinguishability, and Similarity—each capturing a distinct failure mode. Quality measures how well classifiers trained on synthetic data generalize compared to real data; it is defined as the normalized ratio of classification performance and comes with a PAC‑style lower bound. Utility measures how much synthetic data improves downstream performance beyond real data. Indistinguishability asks how difficult it is for a discriminator to tell real from generated samples. Similarity assesses whether real and synthetic samples share local neighborhoods via entropy of k‑NN domains. These metrics together probe fidelity gaps, redundancy, distributional shifts, and local mixing.

### Strengths
- Clear motivation: The paper points out that existing scalar heuristics (e.g., FID, IS) are brittle, lack statistical guarantees, and conflate fidelity with diversity. It argues convincingly that evaluation should be multi‑dimensional and diagnostic rather than a single score.

- Each metric is accompanied by PAC‑style generalization bounds derived in the appendices. For instance, the quality score bound depends on Rademacher complexity, and the indistinguishability bound depends on VC dimension. This gives the framework a principled way to decide if an empirical score is statistically valid.

- Robust Ranking Procedure: RankGen uses quartiles (median and interquartile range) to summarize heavy‑tailed metric distributions and Monte‑Carlo sampling with pairwise dominance counts to produce uncertainty‑aware rankings. Models that fail PAC bounds are filtered out, and surviving models are compared using robust summaries.

- Diagnostic Interpretation: Instead of just ranking, RankGen explains why a model fails: e.g., a high similarity but low utility score signals memorization, while low indistinguishability reveals distribution shifts. This diagnostic approach can guide safer deployment.

### Weaknesses
- All four metrics rely on a downstream classification task; they require labelled data and a predefined classifier architecture. In many generative settings (e.g., open‑domain image generation, text, audio) labels may be unavailable or the “task” may not be classification. The quality and utility scores hinge on the choice of classifier and evaluation metric, potentially biasing evaluation.

- Computational complexity: RankGen entails multiple stratified splits, training at least five classifiers per generator (for quality, utility, indistinguishability, similarity), computing k‑NN neighborhoods, and Monte‑Carlo sampling for rankings. The method may be computationally heavy, particularly for high‑dimensional data.

- Sensitivity to hyperparameters: Similarity requires selecting k (10–50); the ranking procedure samples from a truncated Gaussian with a ridge variance; the number of splits and δ allocations must be chosen. The paper does not explore sensitivity to these hyperparameters.

- The composite score Emin takes the minimum of the predictive and alignment blocks, which can harshly penalize models that excel in one aspect while slightly underperforming in another. This may discard generators that are strong but specialized (e.g., high‑fidelity but low utility) even if they could be useful for certain applications. Similarly, filtering by PAC bounds may eliminate models that are slightly below threshold despite being practically useful.

- Limited modalities and generators: The experiments focus on relatively small datasets (MNIST‑like) and small‑sized models (e.g., StyleGAN2‑lite, DCGAN). Large‑scale diffusion models (e.g., SDXL, Flux), autoregressive text models, or audio generators are not evaluated, leaving the generality of RankGen uncertain.

- Classifier Dependence: The quality and utility metrics depend on the chosen classifier architecture and metric (accuracy, AUC, etc.), and similarity uses k‑NN on raw features rather than learned embeddings. Different choices could alter results; the paper does not examine robustness to these choices.

- Presentation: The main paper is dense; key derivations, algorithm details, and hyperparameters are relegated to numerous appendices, which may hinder readability. The method introduces many moving parts, which can be daunting for practitioners seeking a simple evaluation protocol.

### Questions
- How does RankGen handle unconditional generative models or generative tasks without obvious classification labels (e.g., open‑domain text generation, image captioning)? Could one use self‑supervised or regression tasks? Are there plans to extend the framework beyond classification?

- Have you investigated the effect of varying the number of resampling splits, the k for similarity, or the δ allocations in the bounds? How should a practitioner choose these values?

- What is the computational cost and generalization of RankGen on large datasets or high‑resolution images like ImageNet 512x512 or text-to-image dataset?

- Did any generators fail the PAC bounds but still perform well empirically? Conversely, did any pass but exhibit poor generalization? An empirical study of bound accuracy would strengthen the theoretical claims.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces RankGen, a statistically grounded text-to-text model for ranking generated text. It reformulates text ranking as a conditional generation probability problem and applies statistical calibration to reduce bias from sequence length and distribution shift. Experiments across multiple NLG tasks show that RankGen aligns more closely with human judgments than existing automatic metrics.

### Strengths
- The paper is well written and easy to follow, with clear organization and presentation.
- The results are interpretable and provide meaningful insights into the model’s behavior.
- The research problem is interesting and relevant to the text generation and evaluation community.

### Weaknesses
- The novelty is relatively limited, as the work mainly reformulates probabilistic ranking rather than introducing a new model architecture.

- The paper lacks stronger comparisons with recent large model–based scoring or preference models, such as GPT-judge or reward models.

- Reproducibility is limited since neither the code nor model weights are released.

- The generalization ability remains uncertain, as the method has not been demonstrated on open-domain generation tasks such as dialogue

### Questions
See in weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper defines and tests some measures that are meant to distinguish real from generated data. 

Quality is meant to measure the difference between accuracy when trained on real versus generated data. Utility is the same but only when part of the data is replaced.

Indistinguishability is meant to measure the ability to tell apart real from generated data. Similarity tests the data with respect to a specific distinguisher (neighborhoods of a given sample).

The main contribution are these definitions. Numerous experiments calculate the requisite statistics on several datasets.

### Strengths
Trying to make sense of the differences between real and generated data is a well-motivated question. This paper fleshes out and test some specific measures for this purpose.

### Weaknesses
There appears to be a conceptual misunderstanding. If the output of a generative model is *indistinguishable* from the training data then no (efficient) test can tell the two apart. Given sufficient data, it is impossible that the quality measure is high but the two are indistinguishable because measuring the quality is a particular way to distinguish between the real and generated data.

It is therefore not sensible that the "indistinguishability rank" can be low but any of the other ones (like quality or utility) are high.  This is merely an indication that the discriminator you use to ascertain indistinguishability is not strong enough to emulate the quality or utility test.

### Questions
In fact in most of the experiments you report the ranks are similar. There are few exceptions. In line 445 you write:

"StyleGAN2-lite delivers high Quality but almost no Utility and weak Similarity, mirroring the synthetic mode-collapse profile: crisp yet
narrow samples. DCGAN lands near chance in Utility while keeping Indistinguishability high, signalling shallow realism that fails to expand the task dataset."

Can you explain what "synthetic mode-collapse profile", "crisp but narrow samples", and "shallow realism" mean and how they are captured by your measures? Some concrete examples (possibly on synthetic data) could go a long way towards justifying the sensibility of your definitions.

### Soundness
3

### Presentation
4

### Contribution
2
