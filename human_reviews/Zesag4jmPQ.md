# XAL: EXplainable Active Learning Makes Classifiers Better Low-resource Learners

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5, 3

## Abstract
Active learning aims to construct an effective training set by iteratively curating the most informative unlabeled data for annotation, which is practical in low-resource tasks. Most active learning techniques in classification rely on the model's uncertainty or disagreement to choose unlabeled data. However, previous work indicates that existing models are poor at quantifying predictive uncertainty, which can lead to over-confidence in superficial patterns and a lack of exploration. Inspired by the cognitive processes in which humans deduce and predict through causal information, we propose a novel Explainable Active Learning framework (XAL) for low-resource text classification, which aims to encourage classifiers to justify their inferences and delve into unlabeled data for which they cannot provide reasonable explanations.  Specifically, besides using a pre-trained bi-directional encoder for classification, we employ a pre-trained uni-directional decoder to generate and score the explanation. A ranking loss is proposed to enhance the decoder's capability in scoring explanations. During the selection of unlabeled data, we combine the predictive uncertainty of the encoder and the explanation score of the decoder to acquire informative data for annotation.

As XAL is a general framework for text classification, we test our methods on six different classification tasks. Extensive experiments show that XAL achieves substantial improvement on all six tasks over previous AL methods. Ablation studies demonstrate the effectiveness of each component, and human evaluation shows that the model trained in XAL performs surprisingly well well in explaining its prediction.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces an Explainable Active Learning framework tailored for text classification tasks in settings with limited resources. The key idea is to prompt the model to provide a natural language explanation for its prediction. This explanation is then leveraged both in active learning data selection and model fine-tuning.

### Strengths
The proposed method innovatively combines human-in-the-loop label annotation and LLM-in-the-loop explanation generation to optimize both the model performance and the annotation budget of active learning.

### Weaknesses
Section 3.1 It's inappropriate to use a development set for hyperparameter tuning, as highlighted in references [1][2]. This compromises the integrity of the experimental setup, and is the major reason for rejection. 

Section 4.4: The human evaluation results are perplexing. The main results in Section 4.1 show the model's accuracy on all datasets to be considerably lower than 94%. Given that the predicted labels exhibit an accuracy range of 60-80%, a 94% consistency between the predicted label and its associated explanation appears contradictory.


paper
1. Weaker Than You Think: A Critical Look at Weakly Supervised Learning
2. On the Limitations of Simulating Active Learning

### Questions
Suggestions:
* In Section 2.4, change "golden" to "gold."
* In Figure 2, \pi is introduced without prior definition. Its definition is provided subsequently in Section 2.3.
* In Section 4.1, it would be beneficial to include the zero-shot ChatGPT results and the results from the model "trained on the entire training set" as flat lines in the figures.
* For Figure 2, ensure consistent color coding between the caption and the figure elements (e.g., red and blue arrows).

Questions:
* How about the performance of few-shot LLMs using a random select strategy, in addition to the zero-shot LLM?
* For Figure 4, all other baseline models demonstrate very similar performance across all datasets when limited to 100 data points. Could there be a specific reason underlying this?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The manuscript proposes an Explainable Active Learning (XAL) framework for text classification with a pre-trained uni-directional decoder to generate and score the explanations. XAL proposes a ranking loss to enhance the decoder's capability in scoring explanations and combines the predictive uncertainty of the encoder and the explanation score of the decoder to select the most informative data for annotation. XAL is evaluated on 6 text classification datasets and the results show that the proposed method outperforms existing AL techniques.

### Strengths
XAL generates high-quality explanations for its classification decisions, which can better help users understand and trust the model. This combination makes Active Learning, as human-in-the-loop learning becomes more realistic.

### Weaknesses
- XAL only compares with typical Active Learning methods and does not compare with similar methods like [r1].

- As shown in the ablation study, each component could not provide a stable performance gain on various tasks, e.g., XAL vs. w/o rank.


[r1] Ghai B, Liao Q V, Zhang Y, et al. Explainable active learning (xal) toward ai explanations as interfaces for machine teachers[J]. Proceedings of the ACM on Human-Computer Interaction, 2021, 4(CSCW3): 1-28.

### Questions
- How to set the hyperparameters? Like $\lambda$, $\lambda_1$, $\lambda_2$. 

- In experimental settings, the author only uses 500 labeled samples, as shown in Figure 4, most model performances are far from convergence.

- In figure 4, why the starting point (initial model performance) of XAL are different from other baselines?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Active Learning represents a potent learning paradigm that seeks to minimize the labelled data necessary for training while maximizing the model's performance. Central to active learning research is the design of an effective acquisition function, aimed at selecting the most informative samples with respect to the decision boundaries, thereby enhancing the classifier's learning process. This paper suggests leveraging the predictive explanations generated by Large Language Models (LLMs) such as ChatGPT to complement the predictive uncertainty computed through entropy. The acquisition function combines the weighted sum of entropy and the likelihood of the most probable explanation. In addition to the traditional cross-entropy loss, the authors introduce two additional components: an explanation generation loss and a ranking loss. The experimental results demonstrate the effectiveness of the proposed framework, showcasing its potential to deliver promising results.

### Strengths
* Originality:  The incorporation of explanations generated by LLMs for guiding the selection of informative samples is indeed an interesting idea, even though it is not completely new. This approach is thoughtfully motivated by the principles of explanation-based teaching and learning. Leveraging textual explanations from well-trained and established LLMs, such as ChatGPT, has the potential to provide valuable additional insights that can enhance the classifier's capability to distinguish various class patterns. The experimental results further validate its effectiveness in selecting samples for labelling.
* Clarity: In general, the paper effectively communicates its primary idea, albeit with some minor errors and issues that I will elaborate on below.
* Significance: The presented active learning framework yields promising results when compared to various baselines, such as BALD, CAL, LC, and others. Additionally, the ablation studies effectively demonstrate the contributions of different components within the proposed framework. The qualitative analysis employing T-SNE provides some valuable insights.

### Weaknesses
Despite the comprehensive demonstration of the proposed framework's performance across six different text classification tasks, it is noteworthy that the chosen baseline methods appear to be somewhat outdated. Recent developments in the field of active learning, including ALPS[2], BADGE [1], WMOCU [3], SoftMoCU [4], and BEMPS [5], could have provided more up-to-date benchmarks. Furthermore, the omission of works with similar explanation-based learning concepts mentioned in the related work section raises questions about the completeness of the experimental evaluation. Taken together, these factors lead to concerns regarding the sufficiency of the experimental work in demonstrating the advantages of the proposed framework.

Eq (7) has two running parameters and Eq(8) has one.The authors stated that those parameters were chosen empirically based on the preliminary experiments.  However, the reviewer thinks empirically running those running parameters has an implication in its adaptation, as how sensitive of the performance of the proposed active learning framework is unknown. Thus, it would be good to study the impact of those running parameters. 

Furthermore, the ablation studies show that ME-Exp and w/o rank compare favourably with each other, even with XAL in some data sets. Considering the differences among the three models/variants, the review thought the second loss term in Eq(7) associated with the explanation generation might contribute substation ally to the ultimate performance difference. Adding the ablation studies to that term becomes essential together with the running parameters above. Meanwhile, there is a lack of studies on the acquisition batch size.

Interestingly, the authors have not conducted a comparative analysis of the computational costs associated with evaluating the acquisition functions of different active learning schemes. It would be of practical significance to assess the computational expenses involved in these methods.


References
* [1] J. T. Ash, C. Zhang, A. Krishnamurthy, J. Langford, and A. Agar- wal, “Deep batch active learning by diverse, uncertain gradient lower bounds,” in Proc. 8th Int. Conf. Learn. Representations, 2020. 
* [2] 	M. Yuan, H.-T. Lin, and J. Boyd-Graber, “Cold-start active learning through self-supervised language modeling,” in Proc. 2020 Conf. Empirical Methods Natural Lang. Process. (EMNLP), Nov. 2020, pp. 7935– 7948. 
* [3] G. Zhao, E. Dougherty, B.-J. Yoon, F. Alexander, and X. Qian, “Uncertainty-aware active learning for optimal Bayesian classifier,” in Proc. 9th Int. Conf. Learn. Representations, 2021. 
* [4] G. Zhao, E. Dougherty, B.-J. Yoon, F. J. Alexander, and X. Qian, “Bayesian active learning by soft mean objective cost of uncertainty,” in Proc. 24th Int. Conf. Artif. Intell. Statist., vol. 130, Apr. 2021, pp. 3970–3978. 
* [5] W.Tan, L.Du, and W.Buntine,“Diversityenhancedactivelearningwith strictly proper scoring rules,” in Advances in Neural Information Processing Systems, 2021, pp.10906– 10918. 
* [6] Kuhn, L., Gal, Y. and Farquhar, S., 2022, September. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. In The Eleventh International Conference on Learning Representations.

### Questions
* There are multiple factors that contribute to the performance of the proposed XAL method. The review wondered if the authors could show the convergence analysis of the active learner. In other words, can the learner guarantee to converge to the optimal classifier, as the number of acquired samples goes to infinity?
* The experimental results show that using the explanation score in acquiring samples can contribute to learning. What type of uncertainty does the proposed generation score capture? Is it something to do with semantic uncertainty[6]?
* Regarding Figure 3, Should B in the right column correspond to C in the left column?
* In Section 3.3, Should $D_u$ be $D_l$

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes an explanation-based active learning framework for text classification tasks. The framework selects the most informative samples for annotation, and generates and scores explanations for the classifier's predictions, then it forms a learning objective which combines the classification loss, the explanation generation loss, and the explanation ranking loss. To train the decoder for explanation generation, this paper leverages LLMs to obtain golden explanations. The proposed framework is evaluated on six text classification tasks in a low-resource setting, and the results show that it outperforms basic AL methods and coreset and contrastive AL.

### Strengths
1. This paper injects explanations to uncertainty-based AL process to prevent overconfidence and insufficient exploration.

2. The paper is presented in a coherent manner and easy to follow.

### Weaknesses
1. My major concern is whether generating explanation is the most efficient way to use LLMs in this paper's setting. Based on the examples in Figure 3, a budget of 500 instances corresponds to 1500 explanations. If we task LLMs to generate labels, instead of explanations, within the same API calling times, we could obtain annotated data that is N times greater, where N equals the number of labels (N=3 in this example). Such an increase in annotated data could markedly enhance model performance, particularly in low-resource Active Learning (AL) scenarios.

2. The first weakness also undermines the fairness of the comparison within the given data selection budget. The XAL model uses LLMs to produce high-quality explanations, incurring additional inference costs. A more compelling comparison would involve allocating an equivalent amount of LLM resources to the baseline methods for acquiring more labeled data.

3. The scope of datasets evaluated in this study is somewhat narrow. The number of classes is either 2 or 3, and the training sets do not exceed 8k instances. 

4. This paper does not provide an analysis of the computational complexity of the proposed framework. While the paper mentions that the proposed framework requires more time and computational resources for training than encoder-only classifiers, it does not provide a detailed analysis of the computational requirements of the proposed framework.

### Questions
See above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
