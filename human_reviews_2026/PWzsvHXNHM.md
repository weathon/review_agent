# Select the Key, Then Generate the Rest: Improving Multi-Modal Learning with Limited Data Budget

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 8, 4

## Abstract
Multimodal learning serves as a promising approach for applications with diverse information sources. However, when scaling up multimodal learning data, there are numerous challenges for preparing new data from all modalities due to availability or varied cost of data collection. In this paper, we are the first to demonstrate that multimodal models with only a subset of modalities available for new data could reach and even surpass models continuously trained with full modalities. Our research problem is formulated as: given a limited data collection budget, how to find the appropriate modalities to collect new data and generate for the rest to maximize model performance gain? To answer this, we propose a new paradigm - Select the Key modality, then generate the rest to enable learning with limited data (SK-ll). SK-ll contains two key components: (1) Select the key. We propose a modality importance indicator to find the optimal modalities by assessing their single modal marginal contribution and cross-modal interactions. (2) Generate the rest. For the rest modalities, we substitute with generated embeddings. We conducted extensive experiments across affection computing, healthcare, and various vision-language task with diverse multimodal learning backbones to support the effectiveness of SK-ll. Meanwhile, we present interesting empirical insights such as the data efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Select the Key modality, then generate the rest to enable learning with limited data (SK-ll) as a framework of efficient multimodal learning under a constrained data collection budget. The pipeline first selects the most useful modality using a modality importance indicator that considers both the modality's marginal contribution and its cooperation with other modalities. Then the pipeline has another class-specific flow generation model to generate the missing ones. The proposed pipeline is evaluated on several real-world affection computing tasks and a healthcare downstream task to show its effectiveness over the naive allocation baseline.

### Strengths
- The evaluation study is relatively complete, which evaluates the proposed method on several affect computing tasks, and the experiment setups are well-documented;
- The paper includes a detailed analysis and discussion following the experiment results to highlight the efficiency and stability of the method;

### Weaknesses
- The sentence "multimodal models with only a subset of modalities available for new data could reach and even surpass models continuously trained with full modalities" in the abstract seems to contradict the claim that "The introduction of a new modality usually could bring a non-negative effect" (beginning of section 3.2) and later, "This definition captures the intuition that multimodal input, by
incorporating diverse sources of information, typically reduces prediction loss" (line 228-229), as well as several existing theoretical findings such as [1]. The paper does not provide a solid evidence / experiment to support this claim in the abstract, except for 2 small plots in Figure 1, where there is hardly any outperformance of model trained with subset of modalities over full-modality training (the only noticeable case is the video+text on MOSI, while the margin shown in the plot is hard to interpret and can come from experiment noises);
- Modality selection is the most relevant research question in multimodal learning with constrained resources, which is also listed as one of the primary contributions of the paper. But the paper (1) **misses the relevant discussion in related work**, and (2) **misses comparisons to several existing methods, especially [1], which the reviewer finds highly similar formulation** that uses Shapley value to characterize marginal-contribution-based feature importance scores. The paper should **clearly differentiate itself from those existing methods**, and highlight any additional contribution on top of the overlapping parts;
- **The comparisons between the proposed method and the naive allocation baseline are not fair**. Despite of having the same budget allocating to collect data, the proposed method also requires additional compute to generate the missing modalities, which naturally brings it advantage in terms of the amount of data / compute that the naive allocation baseline does not have. A more proper comparison should be comparing the proposed method with the naive allocation baseline that leverages comparable compute in (at least) training (if generation is not applicable);
- The writing and presentation can be **much further improved** and the reviewer suggests a more careful proofread of the submission. The following listed items are not considered as typos but rather **either significant formatting issues or have caused confusion for understanding**:
1. Incorrect citation format, please different the use of \cite{...} and \citep{...};
2. Misleading use of "Left"/"Right" in Figure 1 caption, which should be "Top"/"Bottom";
3. The three modalities for MOSI and MOSEI (Figure 1) should be Text, Image (instead of Video), and Audio, as Video naturally contains audio and its text transcript (if not explicit). Labeling it as Video causes the confusion whether this modality already contains information of the other two;
3. Inconsistent interchanged notation of $M$ and $\mathcal{M}$: e.g. the domain of a function $f_u$ should be $2^\mathcal{M}$ instead of $2^M$ in the definition of Equation 1, and mixed $M$, $\mathcal{M}$, mixed $\phi$, $\hat{\phi}$ in the definition of Equation 4;
4. Texts starting from line 277 have a smaller font size compared to the rest of main text (before 277);

[1] He, Y., Cheng, R., Balasubramaniam, G., Tsai, Y.-H. H., & Zhao, H. (2024). Efficient modality selection in multimodal learning. Journal of Machine Learning Research, 25(47), 1–39.

### Questions
- Weakness 1: what is the concrete (statistically significant) evidence that supports the claim "multimodal models with only a subset of modalities available for new data could reach and even surpass models continuously trained with full modalities"? And why is it not a contradiction to several existing studies that show the non-negative information advantage from a newly introduced modality?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the practical challenge of multimodal learning (MML) under constrained data collection budgets. To solve this problem, the proposed method, SK-ll, first identifies the most informative modality subset using a modality importance indicator based on marginal contributions and cross-modal interactions, and then generates the remaining modalities via learned embeddings. The framework is validated on affective computing (MOSI, MOSEI) and healthcare datasets (ADNI, MIMIC), achieving average performance improvements of 2–7% over baselines. The results suggest that finding the keys and then generating the rest can yield comparable or even superior performance to models trained on full multimodal data.

### Strengths
- This paper studies an important problem that modalities in multimodal learning are not equal, and their importance could vary. Thus, wisely selecting the important ones for training could help the overall performance.
- The two-step method, select and generate, is quite straightforward and intuitive.
- The performance improvement has been extensively validated over quantitative experiments and qualitative studies.

### Weaknesses
- The proposed method requires a search over all possible combinations, and it computes Shapley values and cooperation scores over all possible modality subsets, which is not computationally friendly.
- The method computes the utility function, which is dependent on the expected reduction in prediction loss. Therefore, it requires multiple rounds of model training, thus could lead to cost increase and being misled by training randomness.
- The method assumes that modalities are independent of each other, which could be invalid in many real-world scenarios. Many modalities could be correlated to each other; thus, the assumptions could not hold, and it needs more justification is needed to support the proposed method.
- The proposed SK-LL is not end-to-end trainable, which requires multistep training. Moreover, the ablation study seems to be missing in understanding which step is more important in contributing to the overall performance.
- Missing several references:
  - Wei et al., Mmpareto: Boosting multimodal learning with innocent unimodal assistance, in ICML 2024.
  - Yang et al., Facilitating multimodal classification via dynamically learning modality gap, in NeurIPS 2024.
  - Huang et al., Towards Out-of-Modal Generalization without Instance-level Modal Correspondence, in ICLR 2025.

### Questions
Please address the questions in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this work, the authors formulates an interesting problem: given some available data in a multimodal setting and some budget to collect additional data from any subset of modalities, how to best select modalities and utilize the budget/additional data to best improve model performance? The proposed method first utilizes Shapley values to determine best modality combinations, then uses a flow-based method to generate the remaining modalities for the newly collected data. The proposed method was evaluated over 6 datasets spanning affective computing and healthcare, and the experiment results demonstrates the proposed method's effectiveness over all of them. Additional analysis was conducted over data allocation balance, accuracy of modality improvement indication, and fairness in healthcare datasets.

### Strengths
1. The formulated problem is very interesting and important: when given a budget to collect additional data from any subset of modalities in a multimodal setting, what is the optimal way to collect them that would benefit the overall model performance the most? Tackling this problem can have an impact on many real world situations, where data collection can incur a cost from a limited total budget.

2. The authors proposed a novel method to cleverly utilize Shapley values to determine the most useful modality combinations that provides the most complementary contribution to overall performance. This approach may also be useful for better understanding modality contribution and balance within various multimodal classification tasks.

3. The method was not only evaluated on affective computing with TVA modalities, but also on healthcare datasets with very different data modalities, which demonstrates the proposed method's generalizability to many application domains.

4. There is additional analysis showing that the proposed method works well with different amount of initially available data as well as varying budget, and also shows that balancing the budget across the selected modalities is better than giving more budget to some modalities than other in the selected subset.

### Weaknesses
While the main experiment is conducted on both affective datasets and ADNI/MIMIC, all the followup analysis about modality importance is on MOSI/MOSEI. The authors should at least provide the selected modalities for ADNI/MIMIC as well as their C(S) values.

The presentation quality of the paper needs improvement. There are many mistakes and inconsistencies within the figures and equations, including the following:

- In figure 1, instead of "left/right" in the caption, it should be "top/bottom" instead

- In Equation 1, {j} should be {$M_j$} instead on two locations (below the summation and at the union on the right), as j is just an index and $M_j$ is the modality element. Also, $\phi_j$ should be $\phi_{M_j}$ to be consistent with later equations.

- In Eq (1), the denominator should be (|M|-1)! instead of |M|!, right? Since S is a subset from |M|-1 total elements.

- In Eq(4), i is undefined. The {$M_i$} is probably a typo and should be {$M_j$} instead.

- Technically, $\hat{\phi}_S$ is not well-defined, as $\phi$ is only defined with single-element subscript in Eq (1) and (3). While I can understand how to extend $\phi$ calculation with a set S as subscript, the current mathematical expression is not very rigorous.

In this work, the authors assumed that the collection cost for each instance from each modality is equal. However, in the real world, the cost for collecting data from each modality will vary. It is unclear whether the proposed method can be extended to work with varying budget cost for different modalities.

### Questions
1. When given a certain budget and a large number of modalities, how do you determine the max number of modalities to sample (the hyperparameter k in Algorithm 1)? How do you balance the tradeoff between having more modalities versus having more data instances on fewer modalities?

2. It is quite strange that, in Table 1, quite a lot of upper bound values are lower than lower bound values. Do you have an explanation for this?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes indicator to decide importance of key modality and generate the rest ones considering circumstances of limited data. Experiments on multiple multimodal downstream datasets are conducted to show effectiveness of the proposed model. Although the motivation makes sense for the paper, the presentation is poor for understanding.

### Strengths
1.	Modality importance indicators are theoretically solid with Shapley value computation. 
2.	Experiments are thorough.

### Weaknesses
1.	The paper seems written in a rush and missing lots of theoretical and experimental details. For instance, the definition and calculation of Z in Equ. 3 and 4 as the core measuring function. Is it dataset-level or modality-level definition?s It’s confused to divide Z as normalization in two times. Besides, architecture of generator has not provided for diverse datasets or tasks. The presentation really need to be improved.
2.	Accuracy of UR-FUNNY and MUSTARD are not reported in Table 1 although the caption has mentioned it.
3.	As described in Section 3.4, the generation and optimization in each epoch contained multiple stage of training, which may increase heavy computation. The increased training efficiency compared with baselines should be reported in Table 10.
4.	Since the motivation is about data limitation, all division ratio from 10% to 100% of data budget should be reported in Table 2 to show consistent improvement of the proposed method.
5.	Ablation study should be remained as controlled variable method instead of the settings such as the selected modalities with diverse budget ratio in Table 4.
6.	Random selection of subset to compute C(S) in Table 11 should be reported with iterative experiments of nonoverlap selection across the whole dataset to ensure consistency.
7.	Visualization of generated and original modality embeddings should be conducted to show the generative effectiveness.
8.	More related paper or baselines should be discussed or compared with incomplete multimodal input. Miss Reference:
[1] 	Devamanyu Hazarika, Yingting Li, Bo Cheng, Shuai Zhao, Roger Zimmermann, and Soujanya Poria. 2022. Analyzing Modality Robustness in Multimodal Sentiment Analysis. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 685–696, Seattle, United States. Association for Computational Linguistics.
[2] Ronghao Lin, Haifeng Hu; MissModal: Increasing Robustness to Missing Modality in Multimodal Sentiment Analysis. Transactions of the Association for Computational Linguistics 2023; 11 1686–1702. doi: https://doi.org/10.1162/tacl_a_00628
[3] Z. Yuan, Y. Liu, H. Xu and K. Gao, "Noise Imitation Based Adversarial Training for Robust Multimodal Sentiment Analysis," in IEEE Transactions on Multimedia, vol. 26, pp. 529-539, 2024, doi: 10.1109/TMM.2023.3267882.
[4] Adapt and explore: Multimodal mixup for representation learning, Information Fusion, 2024
[5] UniMF: A Unified Multimodal Framework for Multimodal Sentiment Analysis in Missing Modalities and Unaligned Multimodal Sequences, IEEE Transactions on Multimedia , 2024

### Questions
1.	Why reported Acc2 on MOSI while Acc7 on MOSEI in Figure 1. Besides, why are V+T+A reported as one edge of the pie while full modalities as a new pie area.

2.	In Equ 2, when there are no modality as input, does it mean to input all-zero vectors or Distribution sampled in normal Gaussian.

3.	Why not compute the S with single modality to better show the contribution of each modality and as supplementary evidences to support results in Table 1 in settings of S with two modalities.

4.	Why does some results in Table 2 show poorer than baselines? 

5.	Why does some results with more data contribute less than other times in Table 2? Such as 20% to 30% compared with 30% to 40% ones on MIMIC.

### Soundness
3

### Presentation
1

### Contribution
3
