# EAPO: Expert-guided Adaptive Preference Optimization for Recommendation

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
LLM-based recommendation systems have been widely explored due to their extensive world knowledge and powerful reasoning capabilities. However, current approaches fail to fully leverage preference data to optimize for the task, which impedes the performance of LLM-based recommendations. Although Direct Preference Optimization (DPO) has achieved significant success in aligning LLMs with human preferences, its mechanism of treating all rejected items as a homogeneous group fails to effectively capture the users' diverse preferences, resulting in poor performance on fine-grained preference discrimination. Our empirical analysis reveals that nearly half of prediction errors stem from the model's inability to accurately distinguish between chosen items and high-preference rejected items with subtle differences. To address this challenge, we propose an expert-guided adaptive preference optimization (EAPO) framework that pre-trains a lightweight recommendation model as an expert to assign personalized weights to preference sample pairs. Based on theoretical analysis, we design an adaptive $\beta$ strategy: applying smaller $\beta$ values to item pairs with similar preference levels to amplify reward differences, while using larger $\beta$ values for item pairs with significant preference disparities to ensure learning stability. 
Experimental results demonstrate that EAPO not only achieves superior performance in multiple benchmark datasets, but also demonstrates plug-and-play compatibility with a variety of existing preference optimization methods, establishing a new and scalable paradigm in this field.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes EAPO (Expert-Guided Adaptive Preference Optimization), a novel preference alignment framework for LLM-based recommender systems. The key idea is to introduce an expert model  to assign pairwise preference weights, enabling adaptive β adjustment in the DPO loss. Experiments on three Amazon datasets (Movies & TV, Books, Pet Supplies) show consistent gains over S-DPO, β-DPO, and traditional baselines. The method is also shown to generalize to other preference optimization algorithms such as IPO and CPO.

### Strengths
1. The paper identifies a key limitation in existing DPO-based recommender systems, i.e., uniform treatment of negative samples, and provides an intuitive method to address this issue.
2. The derivation of the gradient dynamics and the identification of a critical β threshold seems novel and add theoretical depth to the adaptive mechanism.
3. The plug-and-play integration with CPO/IPO/S-DPO shows that the proposed strategy is not limited to DPO but represents a general optimization enhancement.

### Weaknesses
1. While the expert-guided weighting is novel, the adaptive β mechanism largely builds upon β-DPO (Wu et al., 2024a). The contribution may be seen as an incremental refinement.
2. Although inference cost is said to be unchanged, the training-time overhead of computing expert scores for all sample pairs could be significant.
3. The derivation assumes fixed reward gradients (||δ||² constant), which may not hold in practice for LLMs. Some discussion or ablation on this assumption’s effect would be helpful.

### Questions
1. How sensitive is EAPO to the choice of expert model architecture? Could the same benefits be achieved with smaller MLP-based scorers?
2. How frequently is β updated during training, per batch, per pair, or dynamically via backpropagation?
3. How does EAPO perform on tasks beyond recommendation, such as summarization or dialogue preference tuning?

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
4

### Summary
This paper identifies the problem of treating all negative items as a homogeneous group in DPO training for LLM-based recommenders.

To address this issue, the authors propose EAPO, a novel preference optimization framework to help the model distinguish varying degrees of user preference across data samples. Specifically, EAPO first introduces a small recommendation model (e.g., SASRec) to estimate the preference gap of each chosen-rejected item pair. It then dynamically adjusts the value of $\beta$ according to this preference difference. If the chosen and rejected items exhibit similar preference levels, the value of $\beta$ will be smaller so that the model would learn more information from the item pair, and vice versa. 

Experimental results consistently demonstrate that EAPO outperforms other preference optimization methods like S-DPO and $\beta$-DPO, while maintaining strong generalizability to other preference optimization frameworks.

### Strengths
S1 **Meaningful motivation.** Since the behavioral patterns vary across users,  the preference margins of data samples are also different and should be considered to help the model focus on more challenging data.

S2 **Intuitive and flexible method design.** The design of adaptive $\beta$ is intuitive and implicitly injects the collaborative information into LLMs. The proposed method is also flexible and can be extended to different preference optimization methods.

S3 **Theoretical analysis.** A theoretical analysis is provided on the gradient of EAPO and the concrete determination of the value of $\beta$.

### Weaknesses
W1 **Lack of experiments on different backbone models.** Experiments on more backbone models of different scales can be added to better exhibit the generality of EAPO, similar to those conducted in S-DPO [1] and $\beta$-DPO [2].

W2 **Presentation issues.** There are some issues with the citation format and the paper writing. For example, in line 337, the citation of $\beta$-DPO should be enclosed in parentheses, and in line 344, "we adopted we employed" appears to be a typo.

W3 **Implementation details.** What is the number of negative items used in S-DPO? Besides, since the multiple negative items are included in S-DPO, how is the corresponding reward margin calculated? The inclusion of those details is suggested.

[1] Chen Y, Tan J, Zhang A, et al. On softmax direct preference optimization for recommendation[J]. Advances in Neural Information Processing Systems, 2024, 37: 27463-27489.

[2] Wu J, Xie Y, Yang Z, et al. $\beta $-DPO: Direct Preference Optimization with Dynamic $\beta$[J]. Advances in Neural Information Processing Systems, 2024, 37: 129944-129966.

### Questions
**Q1: The granularity of $\beta$ variation.** $\beta$-DPO [1] reports that instance-level adjustments of $\beta$ can cause training instabilities. Why does EAPO still employ instance-level adaptation instead of a batch-level strategy, such as adjusting $\beta$ according to the average preference gap in each batch?

[1] Wu J, Xie Y, Yang Z, et al. $\beta $-DPO: Direct Preference Optimization with Dynamic $\beta$[J]. Advances in Neural Information Processing Systems, 2024, 37: 129944-129966.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the application of Direct Preference Optimization (DPO) to fine tune the recommender LLM from preferences. Specifically, the paper proposes to take in account the diverse user preferences and enhance upon the baseline S-DPO performance by using a dynamic regularization constant $\beta$. This is done by an additional lightweight recommendation model assignment. The authors conduct empirical study on recommender system task to showcase the effectiveness of the algorithm.

### Strengths
Applying DPO to enhance the LLM recommender system is an important direction to study, the paper conducted pretty comprehensive theorectical analysis and empirical validation on the effectiveness of adaptive beta (guided by pretrained recommender) on various tasks of recommendation.

### Weaknesses
1. The application of DPO based preference tuning tasks for recommender system has been explored in prior works like S-DPO as mentioned by the paper, thus the application itself is less novel. There also seems to lack discussion and comparison to other works applying DPO to LLM based recommenders in follow-up works to S-DPO. 

2. The idea of using dynamic beta to capture the preference diversity has also been explored in e.g. [1],[2] for general LLM preference learning, there lacks proper reference and comparison, and the contribution of the paper to apply similar ideas to recommender system is rather incremental.

3. The choices of dynamic beta requires additional model, which leads to extra complexity of the algorithm.

[1] MallowsPO: Fine-Tune Your LLM with Preference Dispersions, https://arxiv.org/abs/2405.14953

[2] $\beta$-DPO: Direct Preference Optimization with Dynamic $\beta$, https://arxiv.org/abs/2407.08639

### Questions
How will the performance look like if using adaptive beta like heuristic predict entropy in MallowsPO?

### Soundness
2

### Presentation
3

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
EAPO provides a solution of creating/finetuning LLMs for recommendation systems. Authors identify that key weaknesses in DPO type algorithms fail to capture fine-grained use preferences, especially putting equal weight on cases where rejected and chosen items are very close and when they are not. They introduce a solution of using an external trained recsys model to generate fine grained scores for pairs, then use these scores to continue preference optimizations with required changes.

### Strengths
Strengths: 
1. The authors provide theoretical motivation and comprehensive experiments to address the four research questions

2. The algorithm/methodology is tackling a complex real world problem directly

3. The paper is written well and the structure is comprehensive. Overall, the idea is simple: use large $\beta$ for easy pairs and small $\beta$ for hard pairs of rejected and chosen responses.

### Weaknesses
Weaknesses: 

1. It would be worth including stronger baselines. For example LiPO[1] which directly optimizes the list instead of via pairwise preference optimization such as DPO. There are other similar methods. It would be useful to see how these methods can be directly compared with EAPO. 

2. There maybe a potential concern about the log probs of generated answers/responses. So EAPO is mostly just optimizing the reward differences, but not the absolute value of the rewards. This objective can be successfully optimized at the cost of both log probs of chosen and rejected samples going down. This may lead to a model that is good at ranking, but has a massive impact in its generative power. The paper does not talk about this or provide any analysis or intuition around this. maybe some experiments to show that the model’s fundamental generative power is not destroyed. Now the authors could claim that this is a singular purpose model and we dont need it to do well in other tasks. Irrespective, some discussion is merited. 

3. Another important concern is that this is a roundabout way of doing distillation. The performance of EAPO is now capped by the performance of the “reward” model i.e. the recommendation model trained. How do we ascertain that the recommendation model itself is trained well and that the scores it is giving is actually useful and its not confusing hard pairs with easy pairs or that the ranking obtained from it would be good quality? Further, there could be introduction of noise since the rec model is working on feature distribution space of different items, so it could potentially place mobile and usb-c charging cable quite far away as a hard pair, but for the llm its trivially close since it operates in semantic space. 

4. The paper does claim that there is no extra inference cost which is true. But the training process now is quite complex. This includes 1. pretraining one or more expert model(s) to convergence, 2. running on the entire datasets in advance to generate scores for all pairs. This is substantially more expensive than just DPO, which is never acknowledged and should be quantified. 

5. The authors introduce a new hyper param $\gamma$. Now, its not clear how we tune $\gamma$. The algorithm is sensitive on the value of this therefore how sjhould practitioners choose this value?

6. Its not clear from the main results whether they are just one time inference or averaged over multiple seeds of generations with error bars. Such statistical rigor is necessaary. Moreover, the numbers from the LLM based methods are extremely low. Are the authors using zero shot prompting? If yes, that seems artificially constraining, why not few shots? Further, the prompts are not shared in the paper, as we know a lot depends on the structure of the prompts themselves. 

7. The experiments are conducted with llama 3. This is completely fine. But this makes the reader wonder if some of the drastic improvements such as 22% improvements with EAPO could be obtained with a better base model or with a larger base model. Some intuition or quantification here would be nice. 

References: 
1. Liu, Tianqi, Zhen Qin, Junru Wu, Jiaming Shen, Misha Khalman, Rishabh Joshi, Yao Zhao et al. "Lipo: Listwise preference optimization through learning-to-rank." arXiv preprint arXiv:2402.01878 (2024).

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
