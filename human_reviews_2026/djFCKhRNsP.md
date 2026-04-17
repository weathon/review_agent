# Representation Invariance and Allocation: When Subgroup Balance Matters

- Decision: Reject
- Scores: 4, 4, 4, 4, 4

## Abstract
Unequal representation of demographic groups in training data poses challenges to model generalisation across populations. Standard practice assumes that balancing subgroup representation optimises performance. However, recent empirical results contradict this assumption: in some cases, imbalanced data distributions actually improve subgroup performance, while in others subgroup performance remains unaffected by the absence of an entire subgroup during training. We conduct a systematic study of subgroup allocation across four vision and language models, varying training data composition to characterise the sensitivity of subgroup performance to data balance. We propose the latent separation hypothesis, which states that a partially fine-tuned model’s dependence on subgroup representation is determined by the degree of separation between subgroup latent space representation of the pre-trained model. We formalise this hypothesis, provide theoretical analysis, and validate it empirically. Finally, we present a practical application to foundation model fine-tuning, demonstrating that quantitative analysis of latent subgroup separation can inform data collection and balancing decisions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates why subgroup accuracy is sometimes drastically affected by data allocation during finetuning, while other times entirely unaffected even by missing an entire subgroup. A property of the learned representation, called the latent separation and defined by the total variation distance between the embedding distribution for data with and without the spurious attribute, is proposed as an explanation. The latent separation and subgroup sensitivity are theoretically linked, and experiments are provided which underscore their strong empirical correlation.

### Strengths
1. The correlation studies in Section 4.2 and 6.3 are exceptionally well-done. The variety of datasets, discussion on least-squares vs power-law fit, exploration of different latent distance metrics, and use of metrics beyond the standard worst-group accuracy all contribute to an informative and precise investigation. Overall, a strong commitment to scientific rigor is obvious.

2. The latent separation hypothesis makes intuitive sense and is justified both theoretically and empirically. The experiments in Section 6.3 and 6.4 show that latent separation is indeed well-correlated with subgroup allocation sensitivity, even in a realistic foundation model finetuning setting. The use of the total variation distance between the embedding distributions is, to my knowledge, quite novel.

### Weaknesses
The main weakness of the paper is the limited scope of its findings.

1. The proposed latent separation hypothesis is not strong enough to answer the key questions presented in the Introduction or the insufficiency of current explanations (such as that in Section 4.3). A good example is the phenomenon that “imbalanced data distributions actually improve subgroup performance” referenced several times in the paper. The proposed theory only explains when certain groups are sensitive to mis-allocation (i.e., when the latent separation is large), and _not_ why the optimal allocation for a known sensitive subgroup may not be a uniform allocation. I am not asking for a method which _predicts_ optimal allocation, as the authors address in Section 6.5, but instead some insight relating subgroup representations and allocation which is more fine-grained than a relative ordering of sensitive subgroups (c.f. Sections 6.3 and 6.4).

2. The operationalization of the proposed latent separation hypothesis is somewhat incomplete, and its findings could be further validated by algorithms leveraging its insights. Here are two ideas:

    a. The latent separation metric could be used as an interpretability device for discovering the most sensitive subgroups within a dataset, which likely correspond to those with substantial robustness concerns. Specifically, given a dataset with unknown group annotations, compute the partition of the data such that the TVD of the embedding distribution between subgroups is maximized. This could help users discover previously unknown biases (an active area of interpretability research, e.g., [1]) and it would validate the latent separation hypothesis if this partition corresponds to a priori known spurious correlations.

    b. The latent separation metric could also be used as a regularizer during training to mitigate sensitivity to certain attributes and encourage the model to learn more generalizable and invariant representations. If regularizing the TVD of known sensitive subgroups leads to a reduction in sensitivity, it would likewise validate the latent separation hypothesis.

3. The theory serves its purpose in connecting the TVD of the embedding distributions to subgroup sensitivity in a simple linear setting. However, it is not too surprising, as it essentially states that if the representation is invariant to a particular attribute, then the proportion of data with that attribute doesn’t matter too much (which follows intuitively from the invariance property). More importantly, similar to point (1), it is not strong enough to provide insight into the key questions in this space. In particular, the question of what training mechanism induces different levels of latent separation for different subgroups is entirely avoided, yet would contribute substantially to empirical and theoretical lines of work in this direction [2, 3, 4].

[1] Moayeri et al. Spuriosity Rankings: Sorting Data to Measure and Mitigate Biases. NeurIPS 2023.

[2] Izmailov et al. On Feature Learning in the Presence of Spurious Correlations. NeurIPS 2022.

[3] Yang et al. Identifying Spurious Biases Early in Training through the Lens of Simplicity Bias. AISTATS 2024.

[4] Qiu et al. Complexity Matters: Dynamics of Feature Learning in the Presence of Spurious Correlations. ICML 2024.

### Questions
1. How is this work contextualized in the broader line of research at the intersection of representation learning and spurious correlations/group robustness? There is a substantial body of work investigating properties of the representation to understand subgroup robustness: see, e.g., [1, 2, 3, 4, 5].

2. Two additional references for the point that perfect subgroup balance may not achieve optimal accuracy which should be discussed: [6, 7]

3. In Section 5, the notation $\mathbb{P}\_\theta$ is a bit confusing. Why is the distribution of $Z$ parameterized by model $\theta$ when it is only $h\_\eta$ which induces $Z$, independent of $g_\theta$? It should also be made clear that $\mathbb{P}\_\theta$ is a distribution/measure and not a probability. Lastly, is it important that both TVDs are bounded by the same $\epsilon$, or does the $4\epsilon$ generalize to $2\epsilon\delta$ if one TVD is bounded by $\epsilon$ and the other by $\delta$?

4. Minor clarity/grammatical improvements:

    a. Missing colon line 83

    b. “Following” misspelled line 268

[1] Izmailov et al. On Feature Learning in the Presence of Spurious Correlations. NeurIPS 2022.

[2] Bahng et al. Learning De-biased Representations with Biased Representations. ICML 2020.

[3] Arjovsky et al. Invariant Risk Minimization. ArXiv 2019.

[4] Zhang et al. Rich Feature Construction for the Optimization-Generalization Dilemma. ICML 2022.

[5] Hermann et al. On the Foundations of Shortcut Learning. ICLR 2024.

[6] Qiao et al. Group-robust Sample Reweighting for Subpopulation Shifts via Influence Functions. ICLR 2025.

[7] Schwartz et al. On the Limitations of Dataset Balancing: The Lost Battle Against Spurious Correlations. NAACL 2022.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a novel and comprehensive study on the effect of latent representation allocation in pre-trained models on the subgroup performance on the later fine-tuning. Both rich experiments and detailed theoretical analysis demonstrate that the sensitivity to subgroup allocation is highly correlated with the difference between the latent representations of two subgroups in the pre-trained model. The findings and discussions in this paper also help explain previously controversial observations regarding the relationship between subgroup performance and their group ratios and re-weighted loss.

### Strengths
(S1) Subgroup/fairness-related problems have been a very important topic in recent years, and this paper investigated the underlying representations of the subgroups to explain model performance instead of focusing on the attributes of the training dataset. I believe the research directions discussed in this paper are very promising and can complement other directions for a better understanding of the problems. Also, it aligns well with the interests of ICLR and positions the paper well among literature. 

(S2) Detailed analysis and rich experiments clearly demonstrate that the representations from the pre-trained model have a big impact on later subgroup performance after the fine-tuning stage. 

(S3) In general, the paper is well written and easy to follow. The problem setup and methods are well described in formal and rigorous language and equations.

### Weaknesses
(W1) The major weakness of this paper is that the scope of the problems and contributions is limited. While the authors discussed the general motivation of understanding how subgroup allocation affects subgroup performance in the introduction, the later problem setup is limited to a pre-training–then–fine-tuning setting. While some existing subgroup works used pre-trained models (e.g, Weng et al, 2023), others did not (e.g, Čevora et al, 2025), or were not specified. In fact, I don’t see that subgroup/fairness works have to rely on pre-trained–fine-tuning settings. As a result, although the key findings help explain some phenomena in those pre-trained-model-based works, in general, I believe readers won’t be surprised by the key findings of this paper that the representations from pre-trained models have a big effect on later fine-tuning subgroup performance. 

(W2) In extension to W1, and more importantly even within this paper’s scope, the important question of how subgroup allocation affects subgroup performance remains unclear without comprehensive evaluations of the changes and patterns of the representations learned before and after fine-tuning. In other words, while we know that the difference of representations in pre-trained models has a big impact, how do the representations change after fine-tuning? Do they change at all? Why can’t fine-tuning correct the biased representations from pre-trained models, even with a larger ratio for the minor groups? What are the patterns of representation changes under different group ratios? The authors are encouraged to conduct comprehensive evaluations of the representations before and after fine-tuning and analyze their changes' relationship with subgroup allocation. Otherwise, the current findings, which mainly focus on the pre-trained model’s representations, are limited.

### Questions
(Q1) The authors are encouraged to respond to my concerns in the Weaknesses section, especially W2. Additional experimental results or simple demonstrations would be greatly appreciated and will influence my future decisions.

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
Past work has proposed models that look at how the allocation of (pre)training data across subgroups influences ML model performance on different subgroups -- often making an intuitive assumption that adding data from certain groups will not decrease performance on those groups (and indeed, will generally increase it). However, ascertaining which groups will be affected most by additional in-group data remained a challenge -- the challenge that this paper takes on. This paper presents an intuitive explanation that the degree to which group allocations will affect group-performance has to do with the relative separation of that group from other groups in the dataset. Formal results and several experiments back this claim up. It seems that the formal results only cover the case of two groups, but I'm guessing could be easily extended to more groups? Overall, this is an important problem and I appreciate that both a formal and experimental treatment is given. I have some concerns and questions over the presentation of the paper that I ask the authors to address during the rebuttal phase.

Notes:
- I have several points of concern and several questions. If this can be addressed during the rebuttal phase, I will consider changing my score.
- The appendix is very long. I did not have time to check it all.

### Strengths
1. This is an important problem and one that I do not think has been addressed yet, to my knowledge.
1. Including both the task-specific and the foundation model experiments adds depth to the experimental results and supports interesting comparisons.
2. I thought the limitations section was insightful and well thought out.
3. Formal statements are relevant and appear to be correct though I have not checked all details in the appendix.
4. Related work is sufficient to the best of my knowledge.

### Weaknesses
*1. Use of linear scaling laws.* The authors use linear fits to describe the relationship between subgroup allocation and group performance in figure 2 and figure 3. As the authors note, this is in deviation to past work (Rolf et al. 2021) -- additionally, this deviates from what one would expect from statistical learning theory. I understand that linear models are easier/less sensitive to fit. I disagree with the authors in their assessments that the trends in Figure 2 look linear. The linear fits seems to be particularly bad at the extremes, x-axis near 0 or 1, which are critical parts of the space for this type of analysis. In my opinion this is a major issue.

*2. Unclear what is being presented in figures 2/3*: The labels on figures 2 and 3 are inadequate. What does "subgroup balanced_acc mean"? I assume blue is group 1 and orange is group 2 in figure 2, but that needs to be states. Also, what does each group (1 or 2) correspond to in each dataset?

*3. Comparison to findings of past work in intro/conclusion:* I find statements like "Unlike standard explanations (e.g., assuming that under-represented or poorly performing-groups always benefit from increased data representation)..." a bit confusing. In my understanding, past work provides parametric forms for understanding how subgroup allocation affects subgroup performance. Those parameters could be such that adding more data from a subgroup does not affect sub-group accuracy, no? So why is this such a focus in comparing to past work.

### Questions
1. The authors mention that their analysis covers more than two groups (as does past work, like Rolf et al. 2021, FYI). But all experiments have two groupings. Could the authors please explain what they mean when they say " The multitude of subgroups we compare within the same dataset allows us to gain more insights than previous studies which usually only consider one or two standard groupings (Rolf et al., 2021; Claucich et al., 2025; Idrissi et al., 2022)."?
2. Why do you need to reduce the dimension of the embeddings with PCA (lines 330) before calculating TV? Why the 70% of the variance (that seems low)?
3. In interpreting the result in theorem 5.1, it's unclear to me how insightful this bound is to understanding model behavior. Is there some kind of trivial upper bound that doesn't require as many assumptions that the authors could compare to, and show that this bound is meaningful in comparison? (perhaps at the expense of additional assumptions?)
4. Could the authors comment on the assumtions part (ii) for theorem 5.1, are these reasonable in practice, and if so, why should we think so?

Suggestions: 
- for Figures 5 and 6, it would be nice to report the actual TV for each condition, along with the arrow that shows TV is increasing left to right

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
3

### Summary
The paper studies when balancing subgroup data affects model performance and proposes the latent separation hypothesis, suggesting that similar subgroup representations lead to less sensitivity to data allocation. It provides a theoretical bound under last-layer fine-tuning and tests the idea across several datasets and pretrained deep learning models. Results show consistent correlations between representation similarity and performance sensitivity, though within simplified and controlled setups.

### Strengths
The paper presents an attempt to explain when balancing subgroup data affects model performance. The main idea to link subgroup performance sensitivity to the similarity of their latent representations is clear. The experiments are consistent across datasets and the analysis is carefully done, though mostly within limited and controlled setups. The paper is clearly written and the results are easy to follow.

### Weaknesses
1. Doesn’t account for subgroup difficulty.
The paper assumes subgroup performance differences come mainly from representation separation, but it doesn’t consider that some subgroups might just be inherently harder to learn. For example, a subgroup with noisier labels or less distinctive features may naturally require more data. Without checking this, the correlation between latent separation and sensitivity could partly reflect difficulty, not just representation overlap.

2. Fixed fine-tuning budget is unrealistic
The experiments always use a fixed total fine-tuning size K, varying only how samples are distributed between subgroups. That makes sense for clean comparisons but doesn’t reflect real scenarios, where we often expand data rather than just reallocate it. The appendix includes a smaller K ablation and shows similar trends, which is nice, but this should be tested on more datasets or at larger budgets. A cost–benefit analysis across varying total sizes would make the setup more practical.

3. Theory limited to last-layer fine-tuning
The main theorem assumes only the classifier head is fine-tuned, keeping the encoder fixed. That’s quite narrow, since full-network fine-tuning is now standard. They do check this empirically and find similar correlations under full fine-tuning, but the theory doesn’t extend there. A relaxed version of the theorem that accounts for small feature drift, or an empirical measure of how stable subgroup embeddings remain during full fine-tuning, would strengthen the contribution.

4. Models are not true foundation models.
The main backbones (small CNN, DenseNet121, ViT-B/16, BERT-base) are standard pretrained networks, not foundation models. The only FM-style test is with CheXagent embeddings, where a linear head is trained on frozen features. While this shows the idea may generalize, it’s still far from real downstream FM use cases. The generalization claim should be softened, or an extra test with end-to-end fine-tuning on a true FM should be added.

### Questions
1. Can you measure or control for inherent subgroup difficulty (e.g. subgroup-level learning curves or probe accuracy) to ensure latent separation isn’t just reflecting hardness?
2. Why assume a fixed fine-tuning budget? Would the same correlation trends hold when the total dataset grows instead of being reallocated?
3. How much does the representation actually move under full-network fine-tuning? Can you quantify feature drift to connect theory and practice?
4. Could you extend the theory to full-network fine-tuning by introducing a small deviation term for representation change?
5. To back up the “foundation model” claim, can you try a second FM or an end-to-end low-rank fine-tuning on CheXagent or another FM?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors study the problem of subgroup performance under group imbalance. Specifically, they theoretically investigate the conditions on which balancing the dataset by the subgroup variable improves the accuracy of each subgroup. The critical quantity is the TV distance between the latent distribution across groups for each class, where lower separability means reduced benefit from data balancing. The authors empirically validate their theory on four datasets.

### Strengths
1. The paper tackles an important problem in algorithmic fairness.

2. The authors derive a simple and intuitive TV upper bound.

### Weaknesses
1. Though Theorem 5.1 provides some good intuition, it does not sufficiently characterize the empirical phenomenon studied in the paper. In particular, the authors should derive a bound on the slope of the per-subgroup balanced acc vs. subgroup proportion (Figure 2), as a function of the TV or some other terms. 

2. The assumption that $P(Y)$ stays the same across the two datasets in L290 seems very strong. In particular, if we have $P(Y | A= 0) \neq P(Y | A= 1)$, it seems like changing the subgroup allocations will necessarily change $P(Y)$.

3. Prior works in spurious correlations have characterized the phenomenon where mitigating one bias will exacerbate another. The authors should characterize their theorem in this setting, e.g. when does balancing wrt one attribute increase the gap in another attribute?

4. The authors state in the remark that $|\Delta Acc|$ is small empirically relative to the subgroup performance change. I am not able to find the empirical validation of this claim.

5. The authors should further motivate the practical utility of the method. It seems like the primary setup is when a practitioner has a large set of possible attributes to balance, and they can compute the TV first and use it as a filter, i.e. only try balancing attributes with high TV.  However, this seems unlikely since we normally know in advance which attributes we must be fair to. In addition, if it is just retraining the last layer, the compute time of calculating TV may not be much less than just finetuning it anyways.

[1] Li, Zhiheng, et al. "A whac-a-mole dilemma: Shortcuts come in multiples where mitigating one amplifies others." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
1. Please provide more details on how the TV is computed in practice using the PCA approach. Is there a way to quantify the estimation error resulting from this?

2. How important is the last-layer retraining assumption here? If I replaced $Z$ with $X$ in Theorem 5.1, and had a way to compute TV in high dimensions, could I then choose $g_{\theta}$ to be my whole network?

### Soundness
3

### Presentation
2

### Contribution
2
