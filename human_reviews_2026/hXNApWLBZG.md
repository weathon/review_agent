# P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling

- Decision: Accept (Oral)
- Scores: 4, 4, 6

## Abstract
Personalized alignment of large language models seeks to adapt responses to individual user preferences, typically via reinforcement learning. A key challenge is obtaining accurate, user-specific reward signals in open-ended scenarios. Existing personalized reward models face two persistent limitations: (1) oversimplifying diverse, scenario-specific preferences into a small, fixed set of evaluation principles, and (2) struggling with generalization to new users with limited feedback. To this end, we propose **P-GenRM**, the first **P**ersonalized **Gen**erative **R**eward **M**odel with test-time user-based scaling. P-GenRM transforms preference signals into structured evaluation chains that derive adaptive personas and scoring rubrics across various scenarios. It further clusters users into User Prototypes and introduces a dual-granularity scaling mechanism: at the individual level, it adaptively scales and aggregates each user’s scoring scheme; at the prototype level, it incorporates preferences from similar users. This design mitigates noise in inferred preferences and enhances generalization to unseen users through prototype-based transfer. Empirical results show that  P-GenRM achieves state-of-the-art results on widely-used personalized reward model benchmarks, with an average improvement of ~2.31\%, and demonstrates strong generalization on an out-of-distribution dataset. Notably, Test-time User-based scaling provides an additional ~3\% boost, demonstrating stronger personalized alignment with test-time scalability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes P-GenRM, a personalised generative reward model to address personalised alignment. In more detail, the authors aim to address the challenge of oversimplifying scenario-specific preferences and generalising new users with limited feedback. P-GenRM is a combination of various methods: (i) transforms preference signals into evaluation chains, (ii) clusters users into prototypes, (iii) various stages of training form SFT, to RL, to hard-negative aware curriculum learning.

### Strengths
This paper has various strengths, particularly I wish to point out:
- The achieved results on first look are indeed promising, especially Table 1, which makes a strong case for their proposed method
- The experiments, especially the ablation experiments, are extensive and cover most of the questions I had while reading this paper.
- Their overview figures (1 and 2) help to understand what is happening in their method.

### Weaknesses
I wish to point out some weaknesses that this paper has, upon which the authors could improve to make a stronger case:

- **missing errorbars**: While Table 1 reports the error bars (I assume this is the standard error?), all other experiments do not report any error bars, which makes it difficult to gauge the statistical significance of the experiments. Especially the results in Table 3 and Figure 3b could have overlapping errors.

- **Composition of many methods**: While I appreciate the work of the authors in explaining and displaying the various parts of their methods, to me (personally), the final method seems a bit over-engineered. This makes it difficult to gauge which part has a more substantial effect on good performance, and I also found it hard for the reader to follow. Table 3 does a good job in slightly mitigating this problem

- **What about Generation?** To me, the reason to have a reward model is to train a policy model which demonstrates personalised alignment characteristics subsequently. While I understand the authors' focus on the accuracy of the reward model, it would also be beneficial to test whether this RM can be used to train a policy model and whether personalisation still works.

- **Is average accuracy the right metric?** If I understand it correctly, the authors use average accuracy as a metric in most tables. This would make sense if we are interested in general preference learning. However, as we are looking into personalised preference learning, it is essential to investigate accuracy across the different types of personas in the dataset. Because if there is a majority class/persona in the dataset, and the RM learns its preferences, the average accuracy will look good, while not actually achieving personalised alignment.

**Misc**
- I did not find any code available to inspect what is happening
- The authors refer to the LLM-as-a-judge output as process reward, despite only giving a single value in return. I would abstain from using this terminology, as process reward models (at least from my understanding) are (semi-) dense reward models that assign intermediate rewards to tokens, rather than just one value at the end.

### Questions
- How would you implement P-GenRM for generation in a policy model?
- Could this potentially even be extended to Direct Preference Optimisation (DPO) as an implicit reward model?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces P-GenRM, a personalized generative reward model that aligns large language models to individual user preferences. This is by generating structured evaluation chains, clustering users into prototypes, and applying a dual-granularity scaling mechanism that adapts scoring both at the individual and group levels. This approach reduces noise, improves generalization to new users, and achieves good performance with additional gains from test-time user-based scaling.

### Strengths
1. This paper address an important problem of user personalization by providing the full pipeline of collecting data, clustering users, refining the personalized reward model and adapts the output. 
2. The experimental evaluation is comprehensive. 
3. The pipeline uses both implicit and explicit preference signals, which fully utilizes the preference dataset.

### Weaknesses
1. The main concern is the limited novelty of the paper. It seems that the main contribution of this paper is proposing the overall pipeline of obtaining personalized outputs, by using existing methods such as generative reward models and clustering users. It is not very clear what the technical contributions are. 

2. Lack of analysis of inference costs. It would be nice if some analysis on the costs of personalization can be done, including analysis of the baselines.

### Questions
What are the practical limitations of this method? I suggest including a limitation section in the paper.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper tackles personalized preference modeling for LLM outputs with a generative reward model, P-GenRM, tailored to each user. The method trains a model that outputs a short persona and a weighted scoring rubric from a user’s history and any stated preferences, enabling clearer, user-aware judgments. The paper proposes a complex three-step training recipe to train their reward model that supports test-time scaling that averages multiple runs for the same user and also brings in signals from similar users to cut noise and handle cold-start. The experiments validate the approach on personalized benchmarks with consistent gains over prior methods.

### Strengths
The paper is generally a solid paper. The strengths include:

1. The motivation is clear and important: personalized preference modeling is a real bottleneck for aligning LLM outputs to individual users. 

2. Conceptually, the method advances online preference handling by turning user evidence into a contextual persona and rubric, and by scaling judgments at test time with both multiple runs for the same user and signals from similar users; this is a novel and well-argued design. 

3. The approach integrates several effective components—supervised imitation, RL with process and outcome signals, hard-negative curriculum, and prototype-based test-time scaling—and the ablations substantiate that each piece contributes.

### Weaknesses
There are still some weaknesses listed as below.

1. The method in the main text is too abstract. Key I/O and losses are not clearly written there or clearly pointed to the appendix. The algorithm, including both training and test-time, is generally complex, and I have some questions to be answered in the question section.

2. Computation cost is underreported. I assume the additional test-time user-based scaling may take much more than the baselines. A computation cost ablation study may be necessary.

3. Minor writing/format issues:
   1. Capitalization after a comma: “Based on this, The personalized generative reward model…”
   2. In table 3, "w/o Cl, RL, SFT" should be w/o CL, RL, SFT

### Questions
1. The paper proposes a three-stage training process. I think the training of generative reward models has been studied in previous works, as mentioned in the related works. What is the relation between your training procedure and previous works exactly? Why do you do so intuitively?
2. For the experimental parts, I have two problems. a) To use your test-time scaling, what is the computation cost to have your results compared to the baselines? b) How many samples are required to generate a reasonable preference for each user?

### Soundness
3

### Presentation
2

### Contribution
2
