# Multimodal Classification via Total Correlation Maximization

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Multimodal learning integrates data from diverse sensors to effectively harness information from different modalities. However, recent studies reveal that joint learning often overfits certain modalities while neglecting others, leading to performance inferior to that of unimodal learning. Although previous efforts have sought to balance modal contributions or combine joint and unimodal learning—thereby mitigating the degradation of weaker modalities with promising outcomes—few have examined the relationship between joint and unimodal learning from an information-theoretic perspective.
    In this paper, we theoretically analyze modality competition and propose a method for multimodal classification by maximizing the total correlation between multimodal features and labels. By maximizing this objective, our approach alleviates modality competition while capturing inter-modal interactions via feature alignment. Building on Mutual Information Neural Estimation (MINE), we introduce **T**otal **C**orrelation **N**eural **E**stimation (**TCNE**) to derive a lower bound for total correlation. Subsequently, we present TCMax, a hyperparameter-free loss function that maximizes total correlation through variational bound optimization. Extensive experiments demonstrate that TCMax outperforms state-of-the-art joint and unimodal learning approaches. Our code is available at https://anonymous.4open.science/r/TCMax_Experiments.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
- Addresses an important problem: competition between modalities in training by maximizing total correlation (across modalities + labels) instead of tweaking per-modality losses.
- Derives a lower bound (TCNE) and turns it into a practical, hyperparameter-free loss (TCMax)
- Drop-in training objective; no inference changes or extra heads/schedulers.
- Consistently boosts multimodal accuracy on audio-visual and image–text benchmarks; shows better cross-modal agreement (lower JS-divergence) and more balanced per-modality predictions.

### Strengths
- Clean, well motivated information-theoretic formulation, and nice conversion into a usable loss function with strong results
- Generally strong results across several datasets vs. recent balancing baselines.
- Compute-aware: sampling reduces forward passes
- Analysis beyond accuracy: JS-divergence and negative pair analysis are insightful

### Weaknesses
- Nit: Figure 2: an "illusion" -> illustration?
- See questions

### Questions
- How does this transfer across domains? because you're maximizing joint correlation with the labels on this dataset, it makes sense that you might fit the proper modality contributions here, but how do you know that you're not leading to modality competition in the transfer learning setting (which is common b/c this is probably going to be used for large scale pretraining)?
- What do you lose by sampling instead of doing all the forward passes theoretically required by the method? 
- Do you have some explanation for why TCMax does not outperform baselines on unimodal? How would this fare in cases with high modality imbalance, where joint learning approaches unimodal learning?
- How would this the change in training loss function lead to changes in the realistic downstream uses of the model? e.g., using argmax predictions over logit distributions using distributions as a measure of confidence calibration?

### Soundness
4

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
5

### Summary
This paper proposes TCMax, a multimodal classification objective that consists a lower-bound of Total Correlation. To achieve this lower bound, the authors utilize the MINE of Belghazi and the Donsker-Varadhan representation. The idea of rebasing the expectation of $E_P$ with $Q$ being the factorized $P_{Z_1} P_{Z_2}..P_Y$ is quite interesting for the multimodal training.

### Strengths
The paper has a sound proof to derive the lower bound to TC and an interesting idea to compare the expectation of the joint multimodal distribution with the unimodal ones. The problem and the solution as well motivated.

### Weaknesses
There is one quite major issue. Following the code in the anonymized repo, it seems that the method is using the test set to select the best model during training. This falls under the data leakage between the validation/test set, which are the same in this case. With this major issue, we drive the paper for rejection. Despite that, I will provide some more input on the rest of the paper since the idea is well put.   

The $I(Z_1;Y) + I(Z_2;Y) + I(Z_1, Z_2 | Y) = TC$ ignores that $Z_1, Z_2$ could correlate to predict $Y$, which essentially would be the synergy described by other papers. When you suggest that unimodal training and alignment beyond the task could achieve TC, you ignore this part. I consider this minor since you don't include it somehow in your method, if I understand it well. If you do, please elaborate.

The paper has not included a very important work that is quite close to the method proposed, MCR [1]. There are four parts that are worth discussing. First, they propose a different factorization of the joint mutual information (similarly to TC). Second, they showcase that supervised contrastive learning is lower-bound to some specific MI terms. I will come back to this. Third, they suggest that maximizing the CMI solely is not always the best option since you can still be stuck in a local minimum that you have optimized only one part of this. Finally, what you suggest is also a permutation of modalities and a penalization of their predictive probability which is quite close to permutation importance. Overall, I think comparing with this work and elaborating on improvements/differences will be highly necessary to illustrate the power and novelty of your paper.

About the supervised contrastive learning, what you suggest resembles an InfoNCE that has one positive and many negatives, even the ones with the same label. Including comparison with supervised contrastive or other ways to choose the positive/negative sets shall improve the understanding of the method. Additionally, commenting in terms of MI terms would be extra useful here. 

An intuitive explanation of what the final loss tries to push for will help readers convey the final message. 

Also, please include D\&R [2] for comparison since it has shown to be a strong baseline.

Lastly, I would like to incentivize scaling beyond the typical datasets and models. That should contribute further to our understanding of multimodal competition. 

[1] Kontras, Konstantinos, et al. "Multimodal Fusion Balancing Through Game-Theoretic Regularization." arXiv preprint arXiv:2411.07335 (2024).

[2] Yake Wei, Siwei Li, Ruoxuan Feng, and Di Hu. Diagnosing and re-learning for balanced multimodal learning. In European Conference on Computer Vision, pages 71–86. Springer, 2024.

### Questions
Could you provide results with a separate validation/test set that doesnt include any form of data leakage, and explain this extensively somewhere in your supplementary matterial?

How does you method compare to MCR, D&R and supervised contrastive learning, both conceptually and experimentally?

Could you include an intuitive explanation of your method?

It has been shown that MINE suffers from high variance as an estimator of MI, do you face a similar issue?

### Soundness
1

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
this paper tackles multimodal supervised learning using an information theory perspective. It shows why the modality competition occurs classically when optimizing the cross-entropy loss in a joint-learning framework and advocates for a different strategy: total correlation maximization between modalities and the target. The model is evaluated on six multimodal classification benchmarks and two regression benchmarks, and it demonstrates state-of-the-art results across all datasets.

### Strengths
-	The paper is well written and clearly explains the motivation for their method from an information theory perspective.
-	The mathematical analysis is easy to follow, and the proposed loss is simple, yet effective.
-	The evaluation of the proposed method against baselines is fair and thorough, spanning across different domains and modalities (images, audios, videos).
-	The results are strong in most cases and clearly confirms the hypothesis made by the authors regarding the modeling of cross-modal interactions.

### Weaknesses
-	While I appreciate the completeness in the evaluation protocol in Table 2, I think it neglects the recent emergence of foundation models in the field of unimodal or multimodal representation learning (such as CLIP for vision and language, DINOv3 for vision, Wav2vec 2.0 for speech). I think it is also important to consider these pre-trained models as feature extractors and to apply your method on top of these, as you did afterwards in Table 4 for another dataset. It would clearly demonstrate the benefit of TCMax in a real-life scenario. 
-	Supervised multimodal learning is a bit restrictive in terms of applications. Large-scale multimodal data come usually with very few annotations. I wonder how the proposed model would perform in the case of few-shot learning of semi-supervised learning. 
-	Baseline models and concurrent approaches: since this work is deeply rooted in information theory, I think the works by Paul Pu Liang need to be properly cited and added to the baselines, for instance [1]. His work on self-supervised and supervised multimodal learning (using for instance the Partial Information Decomposition approach [2]) is very close to the one developed in this paper.
-	You mentioned: « without loss of generality, we analyze the scenario with two modalities (audio and visual) »: from an information theory perspective, it restricts the analysis very much. As the authors mentioned, the MI between n=3 variables can be negatives (not the case when n=2), quantifying the interactions generally between n>2 variables is hard (see for instance the Partial Information Decomposition theory, O-Information, gradient of O-Information, etc…) and Total Correlation gives you only a very broad measure of independence between your input variables (without telling anything about  the interactions between paired or triplet of variables in a general system of n variables). I would expect at least a reformulation of this sentence, if not a discussion about it at the end (along with the other limitations of your work). 
-	Typos: Figure 2 “Illusions” -> ”Illustration”

[1] Learning factorized multimodal representations, Tsai et al., ICLR 2019
[2] Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework, Liang et al., NeurIPS 2023

### Questions
-	In equation 6, can you clarify why optimizing I(za, zv |y) is useful in your case since it quantifies the information contained in za and zv irrelevant for y. I would expect this term to actually decrease during training.
-	Implementation details: what is the architecture of the prediction (fusion) head? How did you choose it? Is it similar to other baselines? Does the architecture impact the results? 
-	Are you going to release your code for reproducibility? 
-	You mentioned the computation cost of your method (at least during training). Did you quantify it in practice ?

### Soundness
3

### Presentation
4

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
This paper addresses the problem of modality competition in multimodal learning, where multimodal models tend to overfit dominant modalities and underutilize weaker ones, sometimes performing worse than unimodal baselines. While previous methods have attempted to rebalance or combine joint and unimodal learning empirically, this work takes an information-theoretic approach. The authors analyze modality competition through the lens of total correlation.

The authors propose a new method for multimodal classification that maximizes total correlation between multimodal features and target labels, which naturally promotes more balanced learning and integrates multimodal interactions. They build on Mutual Information Neural Estimation (MINE) and introduce Total Correlation Neural Estimation (TCNE), which estimates a lower bound on total correlation. Using this, they develop TCMax, a novel loss function that optimizes total correlation via a variational bound. Experimental results reportedly show that TCMax outperforms both joint and unimodal baselines, on several datasets.

While the method is currently limited to fully supervised classification tasks, the proposed theoretical framework is both rigorous and well-justified, offering valuable insights that could inspire broader multimodal learning research.

### Strengths
1. Strong theoretical foundation: The paper provides a clear and rigorous information-theoretic formulation for learning from multimodal inputs.

2. Novel objective function: The introduction of Total Correlation Neural Estimation (TCNE) and the TCMax loss.

3. Conceptual clarity: The theoretical motivation is well-grounded and is easily applied into practice.

4. Empirical validation: The experimental results consistently demonstrate performance improvements over multimodal and unimodal baselines.

### Weaknesses
1. **Vague definition of weak and strong modalities.**
Although the paper discusses modality competition, the criteria used to define or quantify “weak” versus “strong” modalities are not clearly specified. Providing a more explicit operational definition or empirical measure would strengthen the theoretical analysis and clarify the interpretation of the results. Additional experiments that explicitly quantify these distinctions would further support the claims.

2. **The claim that the analysis with two modalities holds “without loss of generality” is not justified (l. 141).** In multimodal settings with $M>2$ higher-order dependencies (synergy, redundancy) emerge that are irreducible to pairwise terms. Consequently, the total-correlation objective, identifiability conditions, and estimation behavior differ qualitatively from the bimodal case. Any guarantees or intuitions derived for two modalities therefore cannot be presumed to extend to 
$M>2$ without additional analysis.

3. **In the same vibe as the previous point, scalability to multiple modalities is not clear.**
The current formulation and experiments seem primarily focused on bimodal settings. It is not obvious how the proposed total correlation maximization framework extends to scenarios involving more than two modalities, where inter-modal dependencies become more complex.

4. **Restriction to supervised learning.**
The approach assumes access to fully labeled data, limiting its applicability to self-supervised multimodal scenarios, settings that are highly relevant in practice and that have been addressed in previous works [1, 2].


5. **Missing discussion of relevant related work (FactorCL, CoMM).**
The paper could more clearly articulate how its information-theoretic perspective relates to or differs from recent approaches that explicitly model shared and modality-specific information, such as FactorCL [1] (which uses mutual information–based decomposition) and CoMM [2] (which leverages partial information decomposition). A comparative discussion, both conceptual and empirical, would clarify the novelty and positioning of the proposed framework within this emerging research direction.

[1] Liang, P. P., Deng, Z., Ma, M. Q., Zou, J. Y., Morency, L. P., & Salakhutdinov, R. (2023). Factorized contrastive learning: Going beyond multi-view redundancy. Advances in Neural Information Processing Systems, 36, 32971-32998.

[2] Dufumier, B., Castillo-Navarro, J., Tuia, D., & Thiran, J. P. (2025). What to align in multimodal contrastive learning?. International Conference on Learning Representations.

### Questions
- How do the authors quantitatively define “weak” and “strong” modalities in their analysis?

- The paper claims that analyzing two modalities holds “without loss of generality.” Could the authors clarify the theoretical justification for this claim? How does the proposed framework account for higher-order dependencies (e.g., synergy, redundancy) that arise when $M > 2$ Would the total correlation objective or optimization strategy require modifications?

- Is the proposed total correlation objective compatible with or could be adapted for self-supervised learning setups?

### Soundness
3

### Presentation
3

### Contribution
3
