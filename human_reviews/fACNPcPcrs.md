# Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights

- Avg Score: 5.50
- Decision: Reject
- Scores: 3, 8, 6, 5

## Abstract
Text-based collaborative filtering (TCF) has become the mainstream approach for text and news recommendation, utilizing text encoders,  commonly referred to as language models (LMs), to represent items.
However,  the current landscape of TCF models predominantly revolves around the utilization of small or medium-sized LMs. It remains uncertain what impact replacing the item encoder with one of the largest and most potent LMs, such as the 175-billion parameter GPT-3 model, would have on recommendation performance.  Can we expect unprecedented results?
To this end, we conduct an extensive series of experiments aimed at exploring the performance limits of the TCF paradigm. Specifically, we progressively increase the size of item encoders from one hundred million to one hundred billion, revealing the scaling limits of the TCF paradigm. Furthermore, we investigate whether these extremely large LMs can enable a universal item representation for the recommendation task and revolutionize the traditional ID paradigm, which is considered a significant obstacle to developing transferable “one model fits all” recommendation models. Our study not only demonstrates positive results but also uncovers unexpected negative outcomes, illuminating the current state of the TCF paradigm within the community. These findings will evoke deep reflection and inspire further research on text-based recommendation models. Our code and datasets will be provided upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper explores the performance limits and core issues of text-based collaborative filtering (TCF) recommendation models by systematically increasing the size of the text encoder from 100 million to 175 billion parameters. Experiments are conducted using DSSM and SASRec architectures on three datasets.

The results show TCF performance generally improves with larger text encoders, indicating limits have not been reached even at 175 billion parameters. However, fine-tuning on the target dataset remains necessary for optimal performance despite massive pre-trained encoders. Comparisons to ID-based CF reveal that while frozen 175B encoders can achieve competitive results on some datasets, fine-tuning is required to consistently surpass IDCF, especially with DSSM.

Additional experiments demonstrate TCF exhibits some zero-shot transfer ability, but significant gaps remain compared to models adapted to the target data. Overall, the work provides insights into the limits, competitiveness with IDCF, and transferability of TCF models using extremely large language models.

### Strengths
1. Systematically studies wide range of encoder sizes up to 175B parameters, revealing performance scaling.
2. Compares TCF to strong IDCF baselines, investigating competitiveness for warm-start recommendation.
3. Examines transfer learning potential, important for general recommender systems.

### Weaknesses
1. Only evaluates two basic recommender architectures. More complex models may behave differently.
2. Limited hyperparameter tuning details, so optimal configurations are unclear.
3. Focuses only on random splits, not temporal evaluation protocols.
4. Transfer learning study limited to simple zero-shot approach. More advanced techniques could be explored.

### Questions
For transfer learning, have you tried multi-task or explicit domain adaptation techniques?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the utility of scaling up large language models in the setting of recommendation systems. Recommendation system problems are a fundamentally low-rank and high dimensional problem (the number of missing entries is substantially larger than the number of observed entries) so overfitting is a central issue and integrating neural nets (even simple ones) has not been obvious (as the authors also pointed out, many ideas have been proposed/published but few did fair evaluation). So under the context that building a universal foundational model becomes possible, it is intellectually and practically important to ask how the landscape of building recommender systems is changed. The major questions the authors asked include whether the recommender system also exhibits scaling laws, and whether universal representation is possible. The authors further “partitioned” the questions into smaller, more specific ones that are verifiable/experimentable. The authors’ experiments provide convincing answers to these major questions.

### Strengths
The authors asked many natural and important questions related to the interplay between LLM and recommender system; some of the answers are surprising (scaling laws also exist but universal representation is still hard). The execution and experiments are convincing.

### Weaknesses
It is a very empirical result and very limited effort is made on the theoretical analysis front.

### Questions
Can you comment about the role of overfitting in your work? I noticed people stopped talking about this in neurips/icml/iclr in recent years but the recommender system problems have been closely related to those low rank matrix completion problems, in which significant effort were made to understand variance/bias tradeoff, how the choice on the latent dimensions impact the performance. Is that still relevant when LLM is used for recommender systems (and why/why not relevant)?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the impact of using large language models as text encoders for text-based collaborative filtering on the recommendation performance. It does not propose a new method, but conducts extensive experimental research on the effect of text encoders of different parameter scales on TCF algorithms. The main conclusions are as follows:
1. Larger parameter text encoders can continuously improve the performance of TCF, but even at the scale of the OPT-175B model, it is impossible to achieve the ideal universal representation and is still weaker than the fine-tuned small models.
2. For recommendation scenarios where the main features of the item are text, TCF on the 175B model can achieve a similar performance to the IDCF algorithm and can remove the item ID feature without losing the recommendation effect.
3. Even with the universal item ID feature, due to the possible differences in the matching relationship between users and items in different recommendation applications, the performance of directly transferring the matching model of a certain domain to other domains in a zero-shot manner is still poor.

### Strengths
1. The paper investigates the role of LLM in constructing a universal and transferable recommendation system for text-based collaborative filtering, conducts very extensive experiments, and the work is novel and interesting.
2. The experimental results show that for text-centric recommendation applications, the TCF on the OPT-175B model can achieve comparative performance to standard IDCF algorithm, which is enlightening for the construction of a universal and transferable recommendation system based on LLM.
3. The paper is well-organized, well-written, and easy to understand.

### Weaknesses
1. The experimental datasets use the title as the item feature, and there may be more information that can be utilized but has not been used, leading to the potential of the tested method being underestimated.
2. The paper is mainly experimental and does not propose new solutions.

### Questions
Please address the issues mentioned in the Weaknesses section.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors perform empirical experiments to analyse the upper limits of the text-based collaborative filtering (TCF) recommendation systems. They progressively increase the size of item encoders from one hundred million to one hundred billion to reveal the scaling limits of the TCF paradigm. Moreover, they also study whether the extremely large LLMs can enable a universal item representation for recommendation task. The analysis presented in this work not only demonstrates positive results but also uncovers unexpected negative outcomes, showing the current state of the TCF paradigm within the community.

### Strengths
1. This paper presents an empirical study analysing the performance limit of existing TCF paradigm.

2. The authors study the scaling limits of the TCF paradigm, and also investigate the extremely large LLMs can enable a universal item representation for the recommendation tasks.

3. This study not only shows positive results but also shows unexpected negative results, showing the current state of TCF paradigm within the community. The findings introduced in this paper may inspire further research on the text-based recommender systems.

### Weaknesses
1. The studied problem is interesting and very important for recommender system research, and the experimental results may also inspire future research in recommendation systems. However, this paper may not be suitable to ICLR, it should be better to submit this paper to the IR conferences or journals. 

2. This paper only show the empirical experimental results and present relevant discussions. It does not introduce some novel deep learning techniques.

3. Some details of the experimental settings are not clear. In the experiments, the authors also fine-tune the LLMs with the data in recommendation domains. However, it is not clear what kind of data are used in LLM fine-tuning, and how to fine-tune the LLMs.

### Questions
1. For sequential recommendation, there are some SOTA models that have more complex structure than SASRec and DSSM. Why not using these SOTA sequential recommendation models with more complex structures as backbone models to study the TCF?

2. What kind of data in recommendation domain are used to fine-tune the LLMs and how to fine-tune the LLMs?

3. According to my understanding, the TCF methods studied in this paper use the LLM as the item encoder and use traditional sequential recommendation models to model users' behaviours. Is it possible to directly use LLMs to model the user behaviours?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
