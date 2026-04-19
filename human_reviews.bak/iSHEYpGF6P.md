# Meta-Learning with Task-Environment Interaction

- Decision: Reject
- Scores: 3, 3, 3, 6

## Abstract
The goal of meta-learning is to learn a universal model from various meta-training tasks, enabling rapid adaptation to new tasks with minimal training. Currently, mainstream meta-learning algorithms randomly sample meta-training tasks from a task pool, and the meta-model treats these sampled tasks equally without discrimination, training on them as a whole. However, due to the limitations imposed by training computational power and time constraints, harmful tasks sampled from the imbalanced distribution can have a significant impact on the optimization of the meta-model.Therefore, this paper introduces a form of meta-learning called Task-Environment Interaction Meta-Learning(TIML), which is distinct from reinforcement learning with data preprocessing. In TIML, we create a Task Environment Interaction Mechanism that assesses the interaction between the meta-learning model and the presently sampled task environment. It conducts training differently based on factors such as task difficulty, rewards, harmfulness levels, and others, thereby altering the current practice of uniformly handling multiple tasks.By doing so, we can rapidly enhance the generalization and convergence of meta-learning parameters for unknown tasks. Experimental results demonstrate that the proposed TIML method achieves improvements in model performance while maintaining the same training time complexity. It exhibits faster convergence, greater stability, and can be flexibly combined with other models, showcasing its robust simplicity and universality.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the problem of harmful tasks in meta-model. The authors introduce a Task-Environment Interaction Meta-Learning(TIML) model, which feedback the task environment information to help to select the better task for updating the parameters. To avoid overfitting, they introduce a random parameter to randomly select a task for in-depth learning. Experiments are  conducted on the basis of MAML, and the result shows improvement.

### Strengths
There is a clear structure in the paper. The figures of the paper can present the structure and principles of the model.

### Weaknesses
First of all, there is no sufficient introduction of related work, which should include the recent research on meta-learning. And there is a typo in the title of Section 2 RELATED WORK. 

Second, there is no elaboration of how the task information is passed into the meta-model to calculate the future utility. 

Third, the authors only conducted experiments on the MAML model and its variant, which cannot show the improvement of TIML. There should be some experiments on the recent models to show the improvement of the TIML e.g. SKD, PAL.

### Questions
Can your model be combined with other models besides MAML? Would the experimental result improve in other model?

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a method for prioritizing certain tasks during meta-learning. 

As far as I can tell, each time a batch of tasks is sampled, the method prioritizes tasks that seem to result in the highest intermediate losses during the course of the inner loop.

Various experiments show that this method improves the performance of MAML (and variants of it) somewhat.

### Strengths
- The method seems novel.

- The problem of task selection in meta-learning is important.

### Weaknesses
- The method does not do what it sets out to do. The authors claim that their method has an advantage over previous task-selection methods, in that it can handle the incoming tasks as they are provided, rather than constructing synthetic tasks as in other methods ("In the process of natural selection, we cannot choose whether tomorrow will be sunny or rainy;")

But that is not what is happening. The method requires that a batch of tasks can be sampled, and then that some of these tasks may be prioritized and others ignored during the meta-learning phase, which is equivalent to choosing whether you're seeing rain or sun, and basically similar to synthesizing tasks. Thus it is not clear what advantage is provided over existing task selection methods, including those mentioned by the authors themselves.

- The method is only compared to standard-MAML, and not to any alternative task selection method.

- The method is poorly described, making it difficult to understand (see below). 

- Selecting tasks by difficulty is hardly a novel idea. Juergen Schmidhuber has long emphasized the need to concentrate on tasks that offer optimal difficulty (allowing for maximal learning, neither too hard nor too easy). More recently, the POET algorithm of Wang et al. (arXiv 1901.01753) largely selects new environments by their difficulty for the current generation of learners. (Note: I am not one of these authors)

This past research suggests that the method proposed by the authors (just favor the task with the highest intermediate loss) would fail badly on a very simple case: the presence of impossible or random tasks, which always generate a high loss without allowing for significant learning. This would largely eliminate learning under the method proposed here (but not standard MAML, which would still benefit from the non-impossible tasks).

- The actual algorithm, although confusingly reported, seems to be quite different from standard MAML. In particular, it seems that the gradient updates to the initialization parameters are computed and accumulated over every step of the inner loop ! Then the sum of these computed updates is applied to the initialization parameters in line 20 of Algo 1. This is quite different from MAML, which first goes through an entire inner loop (resulting in a single updated parameter theta_i), then computes a single loss over the "query set" (using theta_i), and takes the gradient of *that* over initialization parameters theta.


This makes the experiments uninformative about the benefit of the approach. If the standard MAML baselines were implemented by this same incremental method, they do not represent actual MAML performance. If, on the other hand, MAML was implemented in the standard way, we do not know whether the tiny improvements result from the task selection or from the novel (and puzzling) scheme of incremental outer-loop gradients.

- Minor: the paper is poorly written, with many typos. The algorithm is unclear (see below), and the discussion of related work is actually spread over sections 1, 2 and 3.

### Questions
- How do you think the algorithm would behave in the presence of impossible or random/uninformative tasks, as mentioned above?

- Confirm whether the outer-loop update (to the initialization parameters) is computed as the sum of intermediate gradients at each time step of the inner loop, which is what lines 14 and 20 in Algo 1 denote in their current form. How were the MAML baselines implemented?

- The text says that At selects the task with the highest *returns*. But the algorithm suggests that it is instead the task with the highest average *loss* (confirmed by the minus sign on the gradient). Which is it?

Minor:

- Section 2: "Related Word" -> Related Work.

- Many cases of citations where the parenthesis are at the wrong place (pro tip: redefine "\cite" as "\citet").

- Why switch from phi to theta in section 3.1?

- What are S and K in Algo 1?

- "Euqal" in the last figure of the appendix.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This study presents a meta-learning approach that employs an epsilon-greedy exploration strategy, actively choosing tasks by utilizing a reward function defined as the running average of each task's training loss. The efficacy of the proposed method is tested through its application to both standard and cross-domain few-shot classification tasks, where it exhibits enhancements over traditional Model-Agnostic Meta-Learning (MAML) techniques.

### Strengths
**Strength 1: Simplicity**

The proposed technique is simple, as it seamlessly integrates with existing meta-learning algorithms without necessitating any alterations to the foundational architecture. Notably, when applied to basic Model-Agnostic Meta-Learning (MAML) frameworks, this method consistently drives marginal performance enhancements.

### Weaknesses
**Weakness 1: Need for Improved Clarity and Precision**

The manuscript would benefit significantly from comprehensive revisions for enhanced clarity and precision in its presentation. It appears that the initial draft may have been produced with the assistance of a large language model, without post-editing.

One area requiring attention is the overall sentence structure and coherence within paragraphs. As it stands, some lengthy paragraphs, for example, the third paragraph of the introduction, are somewhat vague, hindering full comprehension. Additionally, many sentences lack proper spacing after periods.

Furthermore, the mathematical notations and equations present throughout the paper necessitate careful revision. There are instances of typographical errors, possibly arising from formatting issues in LaTeX or direct inputs from language models, which obscure the intended expressions and notations. This lack of precision makes it challenging for readers to grasp the concepts being discussed, particularly in Section 3.1, where most notations appear to be erroneously presented.

Specific examples of confusion include the inconsistent use of $\mathcal{T}\_i$ (collection of tasks or each task?) and the removal of brackets from $\\{ x_i^s, y_i^s\\}\_{s=1}^K$. The symbols $\varphi$ and $\theta$ seem to be used interchangeably, and there's inconsistent subscript formatting, as evidenced with subscript $i$. The notation following $S$ rounds of inner-loop updates also requires correction from $\theta\_{i,\delta}$ to the more appropriate $\theta\_{i,S}$.

Moreover, there is an interchangeable use of certain terms throughout the document, leading to potential confusion. This is observed in the inconsistent use of raw terms and their corresponding mathematical notations, such as $N$ vs. $\mathcal{N}$, $I$ vs. $\mathcal{I}$, $A$ vs. $\mathcal{A}$, and $R$ vs. $\mathcal{R}$. Additionally, the text could be clearer in its use of terms like $\mathcal{T}_i$ and $A_t$, as the current presentation may confuse readers.

In conclusion, the manuscript would substantially benefit from a detailed revision aimed at correcting these issues to improve the clarity and accuracy of both the textual and mathematical content. This process is essential for ensuring the work communicates its valuable insights more effectively to the readership.

**Weakness 2: Insufficient Justification and Elaboration of Central Concept**

A critical aspect of the manuscript that requires enhancement is the depth of rationale and exposition provided for the central idea, particularly concerning the reward function in Equation 4. While the concept as presented appears somewhat intuitive, the narrative lacks a detailed explanation or theoretical background that justifies this specific formulation.

**Weakness 3: Insufficient Novelty and Inadequate Comparative Analysis**
The paper significantly underperforms in establishing its novelty, particularly within the saturated domain of task scheduling algorithms, well-documented by sources [1-6]. A striking omission is the lack of a comprehensive comparison or substantive discussion related to these established works. Both the experimental and related work sections of the paper are noticeably thin on comparative analysis, undermining the paper's credibility and scholarly rigor.

In the experimental design, the authors restrict their focus to MAML-based methods, neglecting a host of other task scheduling approaches documented in [1-6]. This narrow scope fails to justify the proposed method's advantages within the broader context of the field. By not contrasting their results with these established methodologies, the authors miss the opportunity to demonstrate the superiority or the distinctive aspect of their approach.

At its core, the proposed method appears to be an oversimplified version of its predecessors, rather than a groundbreaking innovation. However, this perception may be somewhat influenced by the paper's lack of clarity, making it difficult to ascertain the full scope of the proposed methodology.

[1] SPL: Kumar et al., Self-paced learning for latent variable models, NeurIPS 2010.

[2] FOCAL: Lin et al., Focal loss for dense object detection, CVPR 2017.

[3] DAML: Li et al., Difficulty-aware meta-learning for rare disease diagnosis, MICCAI 2020.

[4] GCP: Liu et al., Adaptive task sampling for meta-learning, ECCV 2020.

[5] PAML: Kaddour et al., Probabilistic active meta-learning, NeurIPS 2020.

[6] ATS: Yao et al., Meta-learning with an adaptive task scheduler, NeurIPS 2021.

### Questions
Question 1: Could the authors provide a more detailed explanation of the rationale behind the reward formulation presented in Equation 4? Additionally, how does this formulation differentiate your approach from existing task scheduling strategies?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces Task-Environment Interactive Meta-Learning (TIML) as a way to optimise gradient-based meta-learning frameworks. During training, TIML selects tasks based on their difficulty and expected future rewards, incorporating a random mechanism to prevent overfitting.

### Strengths
The paper's originality lies in its novel approach to meta-learning, particularly the Task-Environment Interactive Meta-Learning (TIML) method. It introduces a fresh perspective by addressing the challenge of task selection during meta-training, which is an innovative way to enhance few-shot learning.

### Weaknesses
The paper mentions that task selection is based on factors like task difficulty and expected future rewards, but it doesn't provide a detailed discussion or analysis of how these criteria are determined or how they impact the selection process. A more comprehensive exploration of the decision-making process for task selection, along with an analysis of the sensitivity of TIML's performance to variations in these criteria, would enhance the understanding of the method's inner workings.

### Questions
In the context of the TIML algorithm's experiment on cross-domain few-shot classification, why is it significant that TIML exhibits more substantial improvements in the ResNet12 architecture compared to the 4-CONV architecture, and what does this observation suggest about the algorithm's performance in more complex network structures?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
