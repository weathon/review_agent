# Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6

## Abstract
Chain-of-Thought (CoT) training has markedly advanced the reasoning capabilities of large language models (LLMs), yet the mechanisms by which CoT training enhances generalization remain inadequately understood. In this work, we demonstrate that compositional generalization is fundamental: models systematically combine simpler learned skills during CoT training to address novel and more complex problems. Through a theoretical and structural analysis, we formalize this process: 1) Theoretically, the information-theoretic generalization bounds through distributional divergence can be decomposed into in-distribution (ID) and out-of-distribution (OOD) components. Specifically, the non-CoT models fail on OOD tasks due to unseen compositional patterns, whereas CoT-trained models achieve strong generalization by composing previously learned skills. In addition, controlled experiments and real-world validation confirm that CoT training accelerates convergence and enhances generalization from ID to both ID and OOD scenarios while maintaining robust performance even with tolerable noise. 2) Structurally, CoT training internalizes reasoning into a two-stage compositional circuit, where the number of stages corresponds to the explicit reasoning steps during training. Notably, CoT-trained models resolve intermediate results at shallower layers compared to non-CoT counterparts, freeing up deeper layers to specialize in subsequent reasoning steps.  A key insight is that CoT training teaches models how to think—by fostering compositional reasoning—rather than merely what to think, through the provision of correct answers alone. This paper offers valuable insights for designing CoT strategies to enhance LLMs' reasoning robustness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes that Chain-of-Thought (CoT) training enhances out-of-distribution generalization by enabling models to systematically combine learned skills—a process termed compositional reasoning. This claim is buttressed by both theoretical generalization bounds and controlled experiments. Furthermore, a circuit analysis uncovers the internal, multi-stage computational structure that underlies this ability.

### Strengths
1. This paper provides both theoretical analysis and experimental results to support their claims and a mechanism analysis is also completed, which makes the paper comprehensive.
2. The use of a controlled synthetic task allows for a crisp demonstration of the core principle—compositional generalization.
3. The presentation of this paper is clear and easy to understand.

### Weaknesses
1. In my view, the results achieved by the CoT training paradigm proposed in the paper still essentially rely on the model performing single-step reasoning. The model merely learns to recognize the first two tokens in the pattern "e1, r1, r2" and complete the first step of reasoning, and then, given the input "e1, r1, r2, e2", it learns to recognize the last two tokens and complete the second step. What the model has actually learned are two distinct single-step reasoning tasks under different input formats, rather than demonstrating genuine compositional reasoning ability. Moreover, this two-step prediction approach is essentially no different from performing single-step reasoning on atomic data and then explicitly executing two separate single-step inferences. Regarding the OOD generalization capability for compositional tasks, what we would truly like to see is whether the model can accomplish multi-step reasoning directly within a single computational process.  Figure 1 (center right) may also be interpreted by this concern, that there is a trade-off between learning two-hop formations and the atomic reason.
2. There is a discernible disconnect between the theoretical framework and the experimental work. The theoretical results remain without a concrete experimental counterpart for their verification.
3. The robustness analysis feels somewhat detached from the preceding sections. Furthermore, the findings presented in Figure 3 lack a control group for comparison, and the observed trends are intuitively self-evident, which collectively diminishes the insightfulness of this investigation.
4. The related work section discusses latent reasoning, yet I fail to see which part of this paper actually relates to it. If the proposed CoT method is indeed performed in a latent space, the authors should explicitly clarify this point in method part. In that case, I would reconsider my first weakness and the overall contribution of this work.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper 
- theoretically proves that the information-theoretic generalization bounds through distributional divergence can be decomposed into ID and OOD components. 
- empirically proves that CoT training accelerates convergence and enhances generalization from ID to both ID and OOD scenarios. 
- also shows interesting findings that CoT-trained models resolve intermediate results at shallower layers and learns how to think by CoT training.

### Strengths
The papers empirical finding is interesting that CoT training can help OOD generalization. The paper also combines theory with empirical findings, which is a good try.

### Weaknesses
I think the biggest issue is that you cannot directly assume "under suffcient training, the ID generalization error approaches zero" without any justifications. This correlates to what architecture and tasks you are considering. For example, the model can't perform well on the train dataset on the star-graph problem by next-token prediction even if you trained the model suffciently[1]. Thus, in my opinion, it is better to have the view from optimization, which might be hard to analyze. Or you can make more assumptions on the model/task.

[1] Bachmann, G., & Nagarajan, V. (2024). The pitfalls of next-token prediction. arXiv preprint arXiv:2403.06963.

I think the work itself is great, but I think this weakness matters a lot to me. I will increase my score depending on your explanation.

### Questions
Q1: In your experiments, why you split the dataset into 2 parts where one part is to output thought given prompt and the other part is to output final answer given prompt+thought. I think it is equivalent to do SFT on thought+final answer given the prompt as input. Why do you formalize the dataset like that?

Q2: grammar error at line 474, should be "an architectural"

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a theoretical explanation, using generalization bounds to explain why training with CoT data is helpful. The idea is that when the model needs to predict y given x, then the probability can be decomposed as $P(y|x) = \sum_C P(y|X,C)P(C|x)$. If the model has seen the intermediate contexts $C$ then generalization is possible, which does not happen for training without CoT. Experimentally, the authors use a synthetic set-up; the model is trained on 1,2-hop facts 1)without CoT (trained on 1-hop and some 2-hop, evaluated on 2-hop and 2) with CoT (trained on 1-hop and 2-hop together with the intermediate 1-hop facts to perform the 2-hop facts). Their experiments show improved performance and generalization when the model is trained with CoT.

### Strengths
1. The authors provide theoretical results to explain why it can be the case that CoT training leads to better generalization. To my knowledge these results are novel.
2. The paper is clearly written.
3. The experiments also consider noisy data to represent more closely real training data.

### Weaknesses
1. The experimental set-up is not particularly novel. I think it is quite equivalent to other experimental set-ups with synthetic datasets like for example addition. In addition, for example, models are trained with for example one digit addition and some two digit and it has observed that when CoT is also provided the models generalize and perform better (see [1] for example).

### Questions
1. Did the authors tried to perform CoT prompting to the models that were trained without CoT data and see if the performance was improved? The models know each individual component, CoT training basically teaches how to combine these components.

2. Could the authors provide na example from their experimental set-up that matches the theoretical results and shows how CoT training can help predict the correct answer while non-CoT training is unable to do so?

Typo: line 173 the sum should be over C.

### Soundness
3

### Presentation
3

### Contribution
2
