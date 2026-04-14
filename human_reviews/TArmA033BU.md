# MUSE: Machine Unlearning Six-Way Evaluation for Language Models

- Decision: Accept (Poster)
- Scores: 5, 6, 8, 6, 6

## Abstract
Language models (LMs) are trained on vast amounts of text data, which may include private and copyrighted content. Data owners may request the removal of their data from a trained model due to privacy or copyright concerns. However, exactly unlearning only these datapoints (i.e., retraining with the data removed) is intractable in modern-day models. This has led to the development of many approximate unlearning algorithms. The evaluation of the efficacy of these algorithms has traditionally been narrow in scope, failing to precisely quantify the success and practicality of the algorithm from the perspectives of both the model deployers and the data owners. We address this issue by proposing MUSE, a comprehensive machine unlearning evaluation benchmark that enumerates six diverse desirable properties for unlearned models: (1) no verbatim memorization, (2) no knowledge memorization, (3) no privacy leakage, (4) utility preservation on data not intended for removal, (5) scalability with respect to the size of removal requests, and (6) sustainability over sequential unlearning requests. Using these criteria, we benchmark how effectively eight popular unlearning algorithms on 7B-parameter LMs can unlearn Harry Potter books and news articles. Our results demonstrate that most algorithms can prevent verbatim memorization and knowledge memorization to varying degrees, but only one algorithm does not lead to severe privacy leakage. Furthermore, existing algorithms fail to meet deployer's expectations because they often degrade general model utility and also cannot sustainably accommodate successive unlearning requests or large-scale content removal. Our findings identify key issues with the practicality of existing unlearning algorithms on language models.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a comprehensive benchmark to assess LMs' performance after unlearning according to six diverse desirable properties for unlearned models, including verbatim memorization, QA memorization, MIA performance, utility, scalability, and sequential unlearning requests. The paper considers two Corpus including NEWS and BOOKS, to evaluate the eight existing popular unlearning algorithms. Comprehensive experiments are conducted to reveal the effectiveness and limitations of existing methods.

### Strengths
1. The paper is well-written and easy to follow.

2. Compared to previous works, the paper identifies 6 desirable properties for unlearning and formulates them as evaluation metrics.

3. The paper covers a wide range of existing unlearning methods and conducts comprehensive evaluations with the proposed evaluation metrics.

### Weaknesses
1. My major concern is that the experimental results are not well explained. For example, Task Vector's performance in Table 3 is somehow neglected. The results of Task Vector seem inconsistent with prior works such as [1]. More celebrations should be included to discuss why task vector fails on unlearning.

2. Though the paper proposes six properties. These properties may have already been discussed in related works. From my perspective, only C6 sustainability is the new criteria for evaluation. C3, which compares MIA scores between unlearned models and retrained models, is not sound for under-unlearning.

3. In terms of C3, which states that 'Unlearning leads to privacy leakage,' under-unlearning (PrivLeak < 0) may not be regarded as privacy leakage. I agree that the perfect unlearning should let PrivLeak close to 0. However, in practice, it is not likely for the adversary to compare unlearned models with retrained models. Instead, the adversary only calculates the MIA for the unlearned models to determine membership. Therefore, under-unlearning should be one feasible outcome.






[1] Liu, Zheyuan, et al. "Towards safer large language models through machine unlearning." arXiv preprint arXiv:2402.10058 (2024).

### Questions
Please refer to my weaknesses. Also, I have one more question.


1. Figure 2 seems confusing. Why should the data distribution of 'forget' be close to 'holdout'? This is not reflected in the proposed PrivLeak metric.

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
This paper presents Muse, a benchmark designed to evaluate machine unlearning algorithms. The authors construct the evaluation corpus using content from BBC News articles and popular books, such as Harry Potter. They propose six desired metrics to assess machine unlearning performance, covering aspects such as memorization, utility, privacy, and scalability. The authors evaluate eight different unlearning algorithms using this benchmark. Experimental results indicate that, while current methods achieve effective unlearning, they also lead to substantial drops in utility and privacy.

### Strengths
- This paper overall is well-written and easy to follow.
- The paper offers a more comprehensive benchmark to evaluate machine unlearning, and conduct thorough experimental evaluation and analysis on several existing algorithms on all six metrics.

### Weaknesses
- This work only evaluate four families of machine unlearning methods, which are all training-based. However, training-free approaches, such as those via model editing (e.g. arxiv:2202.05262) or weight pruning (e.g. arxiv:2403.01267), are also relevant and are not covered here. The authors should either incorporate these methods or explain why they were excluded.
- The authors propose six desired properties for machine unlearning, with three reflecting the data owner’s perspective and three for the model deployer. While these criteria are generally reasonable, it remains unclear if they reflect real-world practitioner concerns. If these are indeed practitioner-driven priorities, the authors should provide references or practitioner insights to support why these metrics were chosen over others not included.

### Questions
In Table 5 of Appendix B.4, the authors report GPU times for each unlearning method, with the baseline retrain method requiring 184,320 hours, which seems unusually high. In the current setup, retraining starts with a base model, and the retrain model is finetuned on this base model for 5 epochs. I understand the authors' rationale that, in practical applications, the forget set is integrated into the entire pretrained corpus, making full-corpus retraining necessary rather than just training on the retain set. However, the authors should clarify this approach more explicitly in the experimental setup and justify why finetuning only on the retain set is a valid simulation of full-corpus retraining.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a systematic evaluation framework called MUSE to assess unlearning algorithms for LLMs. They consider 6 dimensions in their evaluation setup, from both the perspectives of data owners and model deployers. The 6 dimensions include verbatim memorization, knowledge memorization, privacy leakage, utility preservation, scalability wrt size of removal requests and sustainability over sequential unlearning requests. They apply MUSE to evaluate eight representative unlearning algorithms on two datasets, finding that current methods struggle to meet the full set of ideal requirements, especially in preventing privacy leakage.

### Strengths
- What the authors propose is very helpful for the community. Plenty of work is focused on developing approximate unlearning methods for LLMs, and evaluation methods employed are all too often ad-hoc rather than comprehensive and rigorous. They setup authors propose is well-thought through and covers all meaningful dimensions (at least that I see). 
- Thorough analysis of unlearning algorithms, for 2 datasets
- I particularly like the realistic setup for utility preservation of Harry Potter.

### Weaknesses
(more see questions)

- Only evaluating one LLM, in one finetuning regime. 
- No justification for the high MIA performances, which is needed to evaluate how realistic the setup and its conclusions are. 
- Limited utility evaluation. 
- Minor clarifications needed

### Questions
- I'd like to understand the MIA a bit more. You mention that the AUC of the MIA against f_target is 0.0. If I understand it correctly, this refers to the very classic MIA where you have to distinguish between members (D_train) and non-members (D_holdout), without any unlearning. Hence two questions, 
	- Why do you not adapt the standard notion of MIA AUC where the 1 label corresponds to membership (which would mean the AUC on f_target would be close to 1.0 rather than 0.0). I think this would make it more clear for the reader. 
	- Why does it perform so well? While the initial work of Shi et al. indeed shows great performances for Min-k% Prob, it is by now well known that the great performance was due to a distribution shift rather than measuring memorization and that typical MIAs against LLMs do not work much better than a random guess [1,2,3]. I don't believe your setup has a distribution shift (as the AUC for f_retrain is 0.47), yet I wonder why the attack works near-perfectly while all MIAs against LLMs in proper setups . Could you add experiments that convincingly show why it works so well in your setup compared to others? My intuition would be that it is due to the fact that you use 5 epochs for finetuning. Could you add ablations to your unlearning evaluation for less epochs? This would help understand how your methods and conclusions apply to (more realistic) setups where LLMs memorize less. 

- As all evaluation for metrics C4-6 are based on utility, I wonder if the metric used for utility should be more elaborate. Right now, you only measure the knowledge memorization on the custom created question-answering pairs. But I think it would be meaningful to also include e.g. perplexity on D_retain or even performances on well known benchmarks (this would also help clarify whether the finetuning represents a realistic setup for a useful f_target). Could you add this as well? 
- Clarifications
	- Could you confirm in Sec 3.2 where the holdout set for books come from? Is it a random split of the full books or is it data from the harry potter wiki? 
	- Maybe nit, but I was quite confused by the notion of f_retrain. The first time you mention f_retrain is in Sec 3.1 if I'm not mistaken. Could you formalize what this refers to in Sec 2? Otherwise it's hard to follow. And it might also be worth it to rename either 'retrain' in f_retrain or 'retain' in D_retain, as I found this to be confusing too; 

**I believe the framework the paper proposes is very valuable, so I'm generally in favor of accepting the paper. However I also believe my concerns are valid and clarifying them (they should be very fixable) would improve the work. Hence, i'm sticking with a 5 right now, but happy to change the score when my concerns are properly addressed during the rebuttal.**

[1] Duan, M., Suri, A., Mireshghallah, N., Min, S., Shi, W., Zettlemoyer, L., ... & Hajishirzi, H. (2024). Do membership inference attacks work on large language models?. arXiv preprint arXiv:2402.07841.

[2] Meeus, M., Shilov, I.,  Jain, S., Faysse, M., Rei, M., & de Montjoye, Y. A. (2024). SoK: Membership Inference Attacks on LLMs are Rushing Nowhere (and How to Fix It). arXiv preprint arXiv:2406.17975.

[3] Das, D., Zhang, J., & Tramèr, F. (2024). Blind baselines beat membership inference attacks for foundation models. arXiv preprint arXiv:2406.16201.

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This article introduces a method that provides an evaluation framework MUSE for machine unlearning. This evaluation framework assesses the machine unlearning capability from different perspectives through six types of evaluations. Experiments conducted on a series of 7B parameter models validate the framework, comparing different methods and their effectiveness.

### Strengths
1: This paper provides a comprehensive and detailed study of methods for machine unlearning, conducting an in-depth evaluation from six perspectives. It offers a thorough assessment framework covering aspects such as semantics, continuity, knowledge, memory, and privacy. Compared to previous evaluation frameworks, this approach has a broader scope, assesses from more perspectives, and utilizes a larger dataset, demonstrating the framework's comprehensiveness and effectiveness.

2: This paper provides a detailed introduction to each method of unlearning, explaining the corresponding implementation processes with comprehensive formulas and theoretical explanations.

3: The experiments on different datasets are presented clearly, with comparisons across various categories.

4: What I appreciate most is the authors’ effort in providing evaluation metrics, giving clarification for each method with straightforward assessments. Although some metrics, such as C5 and C6, are not as clearly defined, the authors’ attempts to evaluate unlearning across various dimensions are commendable.

### Weaknesses
1: Although the authors provide numerous metrics, many of them heavily rely on previous methods. For example, the C3 metric for privacy assessment has already been addressed in a series of earlier approaches. Additionally, it remains unclear whether certain metrics, such as C5 and C6, are truly essential for evaluating machine unlearning, as their explanations in the paper are not entirely clear.

While I appreciate the advantages noted in Strengths, I would like to see more about how these various methods are integrated within this framework and what specific improvements it offers over prior approaches. Metrics like C1, C2, and C3, for instance, have elements originating from previous work. Although Table 1 highlights the advantages of this framework over the previous TOFU metrics, I believe these advantages are not significant and merely provide comparisons with a few additional metrics.

2: The machine unlearning methods presented in Section 4 of this paper all adopt previous approaches, which makes me wonder if this part could be incorporated into the experimental section instead. It occupies a considerable amount of space, and the methods discussed are largely drawn from prior work. The overall structure of the paper could benefit from further refinement to enhance its clarity and conciseness.

3: Although the paper conducts numerous experiments and evaluates them across various metrics, the results intended to be conveyed by these different metrics are somewhat unclear. For example, if we introduce a set of metrics (C1 to C6) to evaluate different machine unlearning approaches, it raises several questions: Why were these metrics chosen? What conclusions do the experiments under these metrics yield? What consistencies or insights do they offer? The paper lacks sufficient commentary in this regard, and its quality could be enhanced by incorporating more examples and providing clearer justifications for the experiments, thus improving the interpretation of the findings.

### Questions
For me, I would like to see a more detailed explanation of the role each metric plays within the framework used, as well as how the experiments impact these metrics across different methods. This section of the discussion could be further refined and deepened to provide clearer insights into the significance of each metric and the implications of the experimental results.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes MUSE as a benchmark that evaluates unlearned models from multiple angles. The authors leverage this benchmark to test 8 popular unlearning algorithms on LLM, exposing the privacy leakage and utility degradation issues in existing unlearning methods.

### Strengths
This paper tackles very important problems with solid efforts to build the benchmark. The six perspectives of the benchmark are impactful and clear. I enjoy reading this work and I am convinced by the experiments, which are sufficiently comprehensive and well-designed.

### Weaknesses
Overall the weaknesses are not significant. I think the scale of data and number of methods may be further extended, for example, the conclusion of a method's effectiveness may change when forget set gets larger.

Also the hyperparameter tuning may be sub-optimal and require more elaboration.

Figure 4 multiple lines are using the same color which is confusing.
The blue curve in Figure 6 seems completely covered.

Minor: Line 414 GA should be GA_GDR?

### Questions
Although my questions do not affect my evaluation, I am curious whether the authors can comment on the method in https://arxiv.org/pdf/2410.22086? In your Figure 5, no methods are recognized as "desired", i.e. high utility and low-to-medium memorization, yet NGDiff seems to be close to desired.

### Soundness
3

### Presentation
4

### Contribution
4
