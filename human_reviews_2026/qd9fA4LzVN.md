# Label Smoothing Improves Gradient Ascent in LLM Unlearning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
LLM unlearning has emerged as a promising approach, aiming to enable models to forget hazardous/undesired knowledge at low cost while preserving as much model utility as possible. Among existing techniques, the most straightforward method is performing Gradient Ascent (GA) w.r.t. the forget data, thereby forcing the model to unlearn the forget dataset. However, GA suffers from severe instability, as it drives updates in a divergent direction, often resulting in drastically degraded model utility. To address this issue, we propose Smoothed Gradient Ascent (SGA). SGA combines the forget data with multiple constructed normal data through a tunable smoothing rate. Intuitively, this extends GA from learning solely on the forget data to jointly learning across both forget and normal data, enabling more stable unlearning while better preserving model utility. Theoretically, we provide the theoretical guidance on the selection of the optimal smoothing rate. Empirically, we evaluate SGA on three benchmarks: TOFU, Harry Potter, and MUSE-NEWS. Experimental results demonstrate that SGA consistently outperforms the original Gradient Ascent (GA) method across all metrics and achieves top-2 performance among all baseline methods on several key metrics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper identifies the instability of Gradient Ascent (GA) in LLM unlearning and proposes Smoothed Gradient Ascent (SGA)—a simple yet effective modification inspired by generalized label smoothing. SGA mixes forget data with semantically related “normal” samples (generated or selected) through a tunable smoothing rate r, which stabilizes optimization and mitigates GA’s divergence while preserving model utility. Theoretical analysis derives a closed-form estimate for the optimal r, and experiments on TOFU, Harry Potter, and MUSE-NEWS benchmarks show consistent improvements over GA and competitive results against other baselines such as FLAT and NPO.

### Strengths
- SGA is conceptually straightforward, lightweight, and easy to integrate into existing unlearning pipelines.

- Derives an interpretable formula for the optimal smoothing rate and connects it to the geometry of gradients.

- Valuates across three standard unlearning benchmarks with consistent gains in forgetting quality and model stability.

### Weaknesses
There are several weakness:

- Limited novelty in methodology. The use of label smoothing is not new to the unlearning literature. Prior work (e.g., [1]) has discussed how label smoothing can facilitate forgetting, albeit in image classification tasks. The paper should better acknowledge and cite such precedents, clarifying how this work extends those ideas to the LLM unlearning domain rather than presenting them as entirely novel.

- Computational overhead and practicality. The proposed method requires generating multiple synthetic “normal” samples for each forget instance to perform smoothing. This increases both training time and memory cost, as the model must perform roughly K additional forward and backward passes and accumulate corresponding gradients. I think this could be mitigated through online updates, such an approach may still impose a notable computational burden and make comparisons with simpler baselines (like GA) less fair. It would strengthen the work to include a cost analysis or to compare against methods such as DPO trained with the same generated data for a fairer baseline.

- Potential inconsistency in TOFU benchmark results. The reported Forget Quality (FQ) scores on TOFU appear low—around 0.006–0.09, whereas prior works (e.g., NPO’s original results) achieved values close to 1.0 under similar settings (1% forget split). It would be helpful to verify the experimental setup, metrics computation, or evaluation scripts to ensure consistency with benchmark standards.

> [1] Di, Zonglin, et al. "Label smoothing improves machine unlearning." arXiv preprint arXiv:2406.07698 (2024).

### Questions
See the weaknesses section. I will raise my score if the identified issues are resolved.

### Soundness
2

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
4

### Summary
This work applies label smoothing to a batch of normal data and forgetting data, in order to improve the performance of gradient ascent in machine unlearning. Empirical experiments on 7B models and 3 datasets are presented, as well as some theory insights.

### Strengths
The idea is clear and makes sense if one only wants to unlearn some data from the trained model. The results are convincing as far as I can tell, especially the selection of models is sufficiently large and diverse. There is some originality in combining label smoothing, normal data generation, and gradient ascent.

### Weaknesses
Overall, the quality of this work should be enhanced.

1. Retaining is too weak.

Machine unlearning is actually a multi-task problem: it's not only about unlearning, but also about retaining the utility. However, GA (and this SGA) is not good at retaining at all. SGA Table 1 has almost zero FQ that is not remotely comparable to retained model for llama2 and phi, which has 1.0 FQ. Also in Table 3,  KnowMem on Dr (↑) i.e. utility on retaining data is 1.94 whereas retained model is 55, Kl is 48.

2. Missing baselines

Related to the first weakness, I suggest the authors to pay more attention to Gradient Diff methods and compare to it in tables. Also maybe  build a smoothed gradient diff to really improve the retaining.

3. Theoretical analysis is too simple.

I found it obvious and not insightful when reading Section 4.2/4.3. For example, the authors state "a key reason why SGA effectively suppresses the divergence issue of GA is that it alters GA’s gradient ascent direction, preventing the model from updating purely toward maximizing the next-token loss on the forget set." This sentence can be applied to almost any unlearning method like Gradient Diff.

4. Efficiency concern

Using external models to generate K-1 normal data can be expensive, especially when K is large. In TOFU dataset, the normal data is actually the retaining data, so this SGA is similar to gradient diff but requires more retaining data per-step.

### Questions
Is line 187 mistakenly having minus on the first term?

Have the authors considered normalized gradient difference (without automatic learning rate is enough)? 

Is there an ablation study on the cost and effectiveness tradeoff about K? Can K=1 work?

### Soundness
3

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
3

### Summary
The authors introduce a fine-tuning-based unlearning method called Smoothed Gradient Ascent (SGA) that mitigates divergence issues in gradient ascent (GA) by combining the forget example with K–1 “normal” examples through a generalized label-smoothing coefficient (r). Experiments on TOFU, Harry-Potter, and MUSE-NEWS datasets with LLMs show that SGA improves forget quality and reduces divergence compared to GA and related baselines.

### Strengths
- Overall, SGA is novel method that addresses GA’s instability in LLM unlearning. It tackles a key problem in LLM unlearning, improving forgetting-retention balance. The theoretical analysis of the optimal smoothing rate (r*) provides a useful mathematical framework.
- The paper is well-written and structured, with clear explanations and figures.
- Experiments are thorough across three diverse benchmarks with strong baselines, appropriate metrics, and ablation studies that support the method’s effectiveness.
- SGA consistently outperforms baselines, achieving strong Forget Quality without utility collapse, and ranks highly on privacy leakage metrics.

### Weaknesses
- The authors identified the optimal smoothing rate (r*) as dynamic, but in practice it’s fixed during training. Furthermore, the results also show that the r* varies quite a bit across different smoothing rates and models, and some values even cause training to collapse. Have the authors explored methods that computes r* dynamically/periodically during training? Could it improve results?

- SGA relies quite a bit on normal data, which is generated either via embedding similarity or external models like GPT-4o-mini. Could this introduce biases or make the method overly dependent on these models? Also, generating and incorporating this normal data probably adds computational overhead. It would be helpful to quantify the costs involved to assess the method’s scalability. 

- While SGA outperforms the baselines, the paper doesn’t discuss why that's the case, especially compared to methods like KL that also use retain data. Further analysis like gradient dynamics or the contribution of normal data would help in understanding what drives the improvements and strengthen the overall contribution.

### Questions
Please refer to my comments in Weaknesses section.

### Soundness
3

### Presentation
3

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
This paper tackles the divergent collapse behavior of well-known technique 'Gradient Ascent' in fine-tuning based LLM unlearning field. The authors analyse the problem of existing methods and propose simple yet effective technique names as SGA. While doing so, they also provide simple theoretical analysis on why their method can handle those problems and extensive experiments on existing benchmarks.

### Strengths
- Paper is very well written. Especially, the authors provide proper backgrounds for LLM unlearning and also analyse the problem of existing methods well. This helps understanding the motivation of proposed method.

- Although it might be very simple, authors not just justify their arguments by providing empirical results but also by providing some theoretical analysis.

### Weaknesses
- The authors claim that identifying suitable retain set is not feasible when they mention the limitations of the existing methods. I wonder how can we confirm that the generated normal dataset would be the 'suitable' retain set? Especially when we are just relying on the other LLMs which can not be assured that they are properly acting. 

- Looks like (3) and (1) is basically equivalent, but just different realisation of given objective. If so, will they yield same results if 'retain set' is properly selected? Also, (3) would fail as well if the generated data are not proper? This question doesn't necessarily need to be answered with experimental evidence.

- Looks like there are some mistakes in Tables. For example, (Table1, Phi-1.5B, R-RL) has 3 blue rectangles. Also looks like the improvement is quite marginal.

- Also I wonder what is the difference in computational complexity sense compared to existing methods since the given method uses additional LLM models.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
