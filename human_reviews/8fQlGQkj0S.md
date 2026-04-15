# A Theoretical Analysis of In-context Task Retrieval and Learning

- Decision: Reject
- Scores: 3, 6, 5, 6, 6

## Abstract
In-context learning (ICL) can be used for two different purposes: task retrieval and task learning.
Task retrieval focuses on recalling a pre-trained task using examples from the task that closely approximates the target pre-trained task, while task learning involves learning a task using in-context examples.
To rigorously analyze these two modes, we propose generative models for both pretraining data and in-context samples.
Assuming we use our proposed models and consider the mean squared error as a risk measure, we demonstrate that in-context prediction using a Bayes-optimal next-token predictor equates to the posterior mean of the label, conditioned on in-context samples.
From this equivalence, we derive risk upper bounds for in-context learning.
We reveal a unique phenomenon in task retrieval: as the number of in-context samples increases, the risk upper bound decreases initially and then increases subsequently.
This implies that more in-context examples could potentially worsen task retrieval.
We validate our analysis with numerical computations in various scenarios and validate that our findings are replicable in the actual Transformer model implementation.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
They assume that the pretraining data and the downstream task data is generated from a Gaussian/Linear generation process.  They then theoretically analyze the Bayesian-optimal predictions in the in-context learning setting. They divide in-context learning for downstream tasks into two categories depending on the parameters of their generation processes: (1) task retrieval, and (2) task learning. They derive error bounds for these two categories. Their theoretical results show that having more in-context examples may hurt the performance of the model.

### Strengths
1. A clear/rigorous/mathematical definition of two types of in-context learning (task retrieval and task learning) may be helpful for future works.
2. They derive the error bounds for the given setting that decreases quadratically.

### Weaknesses
Because Xie et al. (2022) has proposed to explain in-context learning with a latent variable model, I would expect new studies, if also adopt a latent variable model, to propose some refinement on the data generation process which should be more realistic. However, in my opinion, the date generation process in this work is not more realistic for the following reasons:

1. Firstly, it’s a gaussian model, so it’s very different from the discrete case of NLP. Because it’s not discrete, it is even less realistic than the HMM model used by Xie et al.
2. Secondly, this work assumes that the generation process of the pretraining data is the same as the downstream task. Again, this is worse than Xie et al., because Xie et al. discuss the distribution mismatch problem between pretraining and downstream tasks at least to some extent.

Additionally, it’s not clear to me what the main takeaway of this paper is. Indeed, the authors derive the error bounds for the two kinds of downstream tasks, however

1. It’s not clear to me how the definition of the two kinds of tasks is relevant in the real-world scenario.
2. Empirically, we do not observe that having more examples hurt the performance.
3. And again, it’s not clear how the generation process is related to the real-world data.

### Questions
I suggest the authors elaborate more on the implications of the bounds they prove.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
NLP tasks like LLMs can be better with in-context examples, and in-context learning (ICL) in the community approaches explanation of the area. In this work, authors summarize two modes of ICL - in-context task learning and in-context task retrieval, which are able to learn a new task or retrieve related tasks during pre-training LMs. In the past, theorists somehow ignore the importance of pre-training distribution. Hence, authors propose a data generative approach based on the distribution to explain those two modes.

### Strengths
1. As a theoretical work, beyond upper bounds for risks of two modes, this work provides evidences based on numerical computations and conducts experiments with Transformer. That would be helpful for practitioners in the future work.
2. Visual illustration like Figure 2 helps to understand the prior distribution.

### Weaknesses
1. I observed that Lemma 1 has been used a lot in the manuscript and is lack of proof. Please at least provide high-level idea of the proofing before using it.
2. Texts and figures are overlapped in Page 8.
3. Similarly, for Theorem 3,4 and Lemma 5, please provide at least few sentences about proofing instead of pointing the appendix.

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors study in context learning (ICL) for task retrieval and task learning. The ICL is modeled using a mixture of linear Gaussian tasks. Using component shifting and component reweighting they derive closed form expressions for posterior distribution given k observatios. This is leveraged to derive risk bounds for the two tasks, retrieval and learning, under squared loss. The theoretical results are backed up by numerical, and experimental results.

### Strengths
- The authors highlight the task retrieval and task learning aspects of ICL. 
- The authors provide risk bounds for both the setup under a mixture of linear Gaussian generative model.

### Weaknesses
- The generative model is simplistic as it is limited to linear Gaussian mixture. In contrast HMM based ICL is already studied in other works already mentioned in the paper. The authors should explain the importance of studying this model. 

- The authors leave out some area of works that are related, e.g. meta-learning and retrieval augmented learning. A few recent examples of the latter are -  'A Statistical Perspective on Retrieval-Based Models' by Basu et al., 'Generalization and stability in in-context learning' Li  et al. 

- The dependence of U shaped risk bound on component reweighting and shifting is not discussed properly. Furtheremore, U shape is also observed through the bias-variance tradeoff in optimization literature. The authors don't connect the U shaped mention here with the bias-variance tradeoff. 

- The novelty in deriving the posterior distributions for the mixture of Gaussian distribution is unclear to me.

### Questions
Please look at the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigates in-context learning for task retrieval and task learning from the lens of generative models for pre-training data and in-context samples. A Gaussian mixture model is proposed for the pre-training data generation process. The paper first demonstrates that in-context prediction with a Bayes-optimal next-token predictor corresponds to the posterior mean of the label given the in-context samples. The paper establishes upper bounds for both task retrieval and learning risks, revealing a quadratic decrease in the risk bound of task learning and a U-shaped pattern for the risk bound of task retrieval. The findings are validated through numerical simulations, and with experiments using Transformers.

### Strengths
- The theoretical analysis using data generative models using a Gaussian mixture model is novel, and presents a new approach to investigating in-context learning. 
- The theoretical results are valid, and are backed by simulations with the data generated from a Gaussian mixture model with Transformers.
- The U-shaped pattern for the risk bound of task retrieval is shown for the first time, which has not been observed in previous works.

### Weaknesses
- The claim regarding the similarity to real-world settings is not quite accurate, as the setting studied in the paper is quite different from in-context learning in NLP tasks with practical LLMs. 
- The assumptions in Section 3.3 are restrictive, and the scope of the theoretical analysis is limited to a very specific setup based on a particular generative model.

### Questions
- "A highly expressive F can be viewed as K separate models F0, . . . , FK−1, where Fk takes exactly 2k + 1 tokens as input. Thus, pretraining can be decomposed into K separate optimization problems." Can the authors provide more explanation regarding this statement?
- Can the authors provide more details about the Transformer? The code seems to suggest the architecture used is GPT-2, and NanoGPT, but these details should be provided in the paper.

Suggestions:
- The experimental results provided in Appendix A with Figures 6,7, and 8 are central results for the paper, and are not quite accessible for readers. It would be beneficial to add more experiments in the main paper, and move some of the analysis to the supplementary material.
- There are typographical issues and formatting issues (e.g. text overlaps in Figure 5) that should be fixed.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a theoretical analysis of in-context learning. This work separates in-context learning into task retrieval and task learning modes. The authors formalise in-context learning as Bayesian inference over pre-trained tasks using proposed generative models. Closed-form posterior distribution and its component re-weighting and shifting mechanisms are presented clearly. They prove task learning risk decreases quadratically as number of examples increases and task retrieval risk follows a U-shaped curve, decreasing then increasing with more examples. They validate analysis theoretically and via simulations on Transformer models.
In summary, the paper provides a formal understanding of in-context learning grounded in Bayesian principles. The analysis reveals unique insights into the distinct behaviors of task retrieval versus learning based on modification of the posterior distribution over pre-trained tasks.

### Strengths
Originality: 
Principled Bayesian framework unifying task retrieval and learning modes of in-context learning.
Interesting generative model provides formal grounding for the analysis.
Derives unique insights such as U-shaped bound revealing limitations of task retrieval.

Quality:
Good mathematical analysis and detailed proofs.
Interesting experiments supporting the theory.
Code provided for reproducibility.

Clarity:
Clearly explains concepts like component re-weighting and shifting.
Delineates assumptions underlying the analysis.

Significance:
Helps advances understanding of in-context learning in large language models.
Formal analysis provides basis for improving real-world in-context performance. 
Insights like U-shaped bound have significant implications for practical model.

### Weaknesses
The generative modeling makes strong simplifying assumptions like Gaussian distributions that limits applicability to complex real-world textual data. Expanding the analysis to more realistic data distributions will strengthen it. Or perhaps even showing that any of their conclusions hold in LLMs, for example does the u shaped phenomena occur in LLMs?

The empirical validation relies heavily on synthetic data simulated from the assumed generative process. Evaluating on real-world NLP tasks would better assess wider applicability.

The criteria for determining the number of examples k for optimal retrieval remains unspecified. Providing more precise theoretical or empirical guidance could improve utility.

Risk is evaluated using squared loss. Evaluating other loss functions (Cross entropy) could expand usefulness across different domains.
As task retrieval becomes more mainstream in LLMs, societal impacts of failures in retrieved tasks can be added. Characterizing the potential downsides and suggesting caution.

### Questions
Please see the weaknesses

### Soundness
4 excellent

### Presentation
2 fair

### Contribution
3 good
