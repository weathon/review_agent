# Privacy \textit{Déjà Vu} Effect: Resurfacing Sensitive Samples in Continual Fine-tuning

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Continual fine-tuning of large pre-trained models is now ubiquitous in industry for adapting a model to freshly collected user data. Existing privacy protection practices assume earlier training data is less sensitive and thus focus on the latest arriving samples. We challenge this assumption by tracking per-sample membership-inference risk across sequential fine-tuning rounds of popular transformer-based models, ViT for image data, and BERT for text data. Our experiments reveal the \emph{Privacy Déjà Vu Effect}: new data can \emph{remind} the model of semantically similar legacy samples, possibly elevating their privacy risk significantly. We further demonstrate that this resurgence is closely correlated with the latent-feature-space similarity between old and new examples. These findings underscore the need for a more comprehensive privacy protection mechanism in continual fine-tuning. We have published our code at \url{https://anonymous.4open.science/r/Privacy-Deja-vu-Effect-F006/README.md}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the Privacy Déjà Vu Effect in continual fine-tuning, showing that adding new data can unexpectedly increase the privacy risk of older training samples. Through membership-inference experiments on vision and language models, the authors find that semantically similar new data amplifies legacy privacy leakage. The results suggest that privacy protection in continual learning must account for both new and historical data.

### Strengths
- The paper reveals a novel phenomenon where new training data can retroactively increase the privacy risk of old samples, extending privacy analysis beyond static settings

- The experiments are well-structured across both vision and language models, demonstrating the consistency of the effect

- The correlation between the effect and NTK similarity provides an interpretable link between data overlap and membership leakage

### Weaknesses
- The rise in LiRA accuracy may partly stem from calibration or confidence-shift rather than genuine memorization changes. Fine-tuning on similar data can inflate model confidence globally, making LiRA more sensitive even without new memorization. The paper does not control for this, so it is unclear whether the reported effect reflects true privacy leakage or attack-side bias.

- The NTK-similarity analysis produces inconsistent and at times contradictory results: high-similarity samples occasionally become safer, and the correlation reverses between vision and language models. The authors ultimately conclude that the privacy déjà vu effect is a “local phenomenon”, driven by a few highly similar samples, but this interpretation offers little predictive or practical value. Without a consistent or generalizable relationship between similarity and risk amplification, the analysis remains descriptive rather than diagnostic, leaving practitioners unable to anticipate which data will experience elevated privacy exposure without direct measurement after fine-tuning.

### Questions
I don't have further questions.

### Soundness
3

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
The paper introduces a new privacy risk phenomenon, termed the “Privacy Déjà Vu Effect,” which provides a quantitative measure of how new data in continual fine-tuning can trigger a model’s memory of old data by examining the TPR/FPR ratio. This finding indicates that old data may still pose privacy risks despite continued private fine-tuning. Experiments and membership inference attacks (MIA) are conducted on the Tiny-ImageNet-200 and IMDB datasets.

### Strengths
1. To the best of my knowledge, the “Déjà Vu” framing is original and intuitively appealing: fine-tuning with semantically related new data can “remind” a model of previously learned samples.

2. This paper provides rigorous MIA analysis and a clear explanation of per-sample TPR/FPR as a proxy for privacy risk, effectively connecting it to differential privacy. Moreover, the violin-plot visualizations make the effect visible and comparable across different conditions.

3. The privacy of continual fine-tuning is a timely and important topic.

### Weaknesses
1. The scope of the experiments is slightly limited. Only two rounds of fine-tuning are tested, and the dynamics over multiple rounds remain unknown. The datasets (Tiny-ImageNet and IMDb) seem small. Using more datasets and conducting additional rounds could make the conclusions more convincing. Moreover, the study only uses the ViT and BERT models to observe the phenomenon in basic tasks such as image classification and text sentiment analysis.

2. The measurement of similarity between features is overly simplistic. The authors only consider SSIM and NTK similarity; however, several more recent works [A, B] provide more detailed measurements of feature similarity. Some discussion of these alternative similarity metrics would be interesting.

[A] Insights on representational similarity in neural networks with canonical correlation. Morcos et al., NeurIPS, 2018.
[B] Similarity of Neural Network Representations Revisited. Kornblith et al., ICML, 2019.

3. It would be more insightful if the experiments were designed to test the similarity of representations across different layers. Intuitively, as depth increases, the representations (or features for image data) may become more similar across different datasets. An extreme case is the last layer in image classification problems, where the representations of different data points collapse to the same point—known as neural collapse [C]—which has also been observed in private fine-tuning of ViT [D]. Related to this work, when neural collapse occurs, the last-layer representations of new and old data may become very close, or even identical, as they collapse to the same feature vector. Thus, if the conclusion of this paper is correct, one may be able to identify old data points more easily based on the new data based only on the last-layer features (please correct me if I am misunderstanding). This could be an interesting direction for future studies. For the current version, some discussion on the similarity between features of specific layers (such as the last layer) and its potential relationship to MIA would be valuable.

[C] Prevalence of neural collapse during the terminal phase of deep learning training. Papyan et al., PNAS, 2020.
[D] Neural Collapse Meets Differential Privacy: Curious Behaviors of NoisyGD with Near-perfect Representation Learning. Wang et al., ICML, 2024.

4. During fine-tuning, the paper uses a very low learning rate ($3 × 10^{-6}$). Intuitively, a low learning rate causes the model parameters to change slowly, thereby preserving information from the old datasets. I suspect that a larger learning rate might lead to less leakage of old data. It would be helpful if the authors provided experiments or at least some discussion on how hyperparameters such as the learning rate affect the conclusions of MIA.

5. Several related works are missing. First, the original paper on differential privacy [E] is omitted. In addition, an important paper on MIA [F] is missing. Furthermore, the TPR/FPR ratio perspective on DP originates from Kairouz et al. (2015). The authors mention that it is related to DP; however, this relationship is formalized through the concept of $f$-DP, which measures privacy levels using the flipped ROC curve, as proposed by [G].


[E] Differential Privacy. Cynthia Dwork, 2006.

[F] Membership Inference Attacks against Machine Learning Models. Shokri et al., 2017.

[G] Gaussian differntial privacy. Dong et al., 2022.

6.Typos: When references appear as the subject or object of a sentence in LaTeX, the \cite command should be used. However, for references such as “Kairouz et al. (2015),” the \citep command should be used to include parentheses.

### Questions
See the weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors demonstrate the existence of the Privacy Déjà Vu Effect during the continual fine-tuning of two representative transformer-based models—ViT for image data and BERT for textual data. To quantify changes in sample-level privacy risk, they employ a canonical class of membership inference attacks, which assess whether a particular data sample was part of the model’s training set. Their findings challenge the prevailing practice of focusing privacy protection solely on newly added data. Instead, the study reveals that such selective safeguarding may inadvertently expose legacy samples to elevated privacy risks due to their semantic similarity with newer data.

### Strengths
1. The authors reveal the Privacy Dej´ a Vu Effect: new data in continual fine-tuning can increase the privacy risk of previously safe samples. 

2. Experiments on two representative foundation models and two benchmark datasets show that the effect might commonly exist.

3. The authors have also experimentally studied the reasons behind this effect and identified the significant factors.

### Weaknesses
1. Absence of Theoretical Support
The paper reports an interesting empirical phenomenon, the Privacy Déjà Vu Effect, but does not provide any theoretical analysis or formal explanation to support the findings. Without a theoretical foundation, the observations remain largely descriptive and lack deeper insight into their underlying causes.

2. Limited Dataset Coverage
The experimental evaluation is conducted on only two datasets: Tiny-ImageNet-200 and IMDb. Given that the contribution is primarily empirical, the limited dataset diversity weakens the validity and generalizability of the conclusions, especially for applications in more complex or realistic settings. Given the empirical nature of the contribution, more comprehensive experimentation is necessary.

3. Narrow Model Selection
The models used in the study are limited to ViT and BERT, which, while representative, are relatively dated. The exclusion of more recent and widely used large-scale models such as GPT, LLaMA, or Qwen raises concerns about the practical relevance of the results to modern continual fine-tuning scenarios.

### Questions
see Weaknesses

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
3

### Summary
This paper proposes the Privacy Déjà Vu Effect — a counterintuitive phenomenon in continual fine-tuning where updating a model on new data can resurface or amplify the privacy risks of previously seen samples. They find that newly added data with high feature-level similarity can increase the privacy sensitivity of a small subset of old samples, even though catastrophic forgetting occurs globally. The study provides an empirical characterization of this effect and raises important implications for privacy protection in continual fine-tuning systems.

### Strengths
1. The paper identifies a previously overlooked privacy vulnerability in continual fine-tuning pipelines.

2. The paper is well written and easy to follow, with clear experimental setups and visualizations.

### Weaknesses
1. The observed phenomenon is currently verified only on two models (ViT and BERT) and two datasets. While the findings are interesting, the evidence is not yet sufficient to claim generality across architectures or modalities.

2. The experiments rely on full-parameter fine-tuning, which may not reflect real-world scenarios where parameter-efficient methods (e.g., LoRA, adapters) are mainly adopted.

3. The phenomenon remains purely empirical and lacks theoretical grounding.

4. The study is limited to only two fine-tuning rounds. It remains unclear whether the effect compounds, diminishes, or stabilizes under longer continual fine-tuning trajectories.

5. From a practical perspective, the work could be strengthened by discussing mitigation strategies or auditing methods that practitioners could use to identify or prevent the Déjà Vu effect in deployed systems.

### Questions
1. Do the authors provide some results about whether the Privacy Déjà Vu Effect manifests in parameter-efficient fine-tuning settings (e.g., LoRA, adapters), where the backbone weights are largely frozen?

2. Could the authors suggest potential mitigation or auditing techniques that might help practitioners detect or reduce this negative effect in practice?

### Soundness
3

### Presentation
3

### Contribution
3
