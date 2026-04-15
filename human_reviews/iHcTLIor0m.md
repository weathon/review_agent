# Poly-View Contrastive Learning

- Decision: Accept (poster)
- Scores: 6, 6, 6, 8

## Abstract
Contrastive learning typically matches pairs of related views among a number of unrelated negative views. Views can be generated (e.g. by augmentations) or be observed. We investigate matching when there are more than two related views which we call poly-view tasks,
and derive new representation learning objectives using information maximization and sufficient statistics. We show that with unlimited computation, one should maximize the number of related views, and with a fixed compute budget, it is beneficial to decrease the number of unique samples whilst increasing the number of views of those samples. In particular, poly-view contrastive models trained for 128 epochs with batch size 256 outperform SimCLR trained for 1024 epochs at batch size 4096 on ImageNet1k, challenging the belief that contrastive models require large batch sizes and many training epochs.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies contrastive learning by matching more than two related views, which is called poly-view contrastive learning. Unlike traditional contrastive learning methods that take pairs of tasks, it increases the view multiplicity and investigates the design of SSL tasks that use many views. Experiments show that it is beneficial to decrease the number of unique samples while increasing the number of views of each sample.

### Strengths
The idea of designing contrastive learning methods using poly-view seems novel. It utilizes an observation from the prior works that using multiple positive views improves the performance. The paper is well-written.

### Weaknesses
Although there are prior works showing that multiplicity improves generalization and convergence of neural networks, it lacks rigorous theory on the relation between contrastive learnability and the number of views on each sample.

### Questions
I wonder how strong the theory on multiplicity can be. Is it possible to specify how exactly the number of views on each example improves the algorithmic performance? Would it be essential on the average number of views, or maximal number? Does there exist a threshold on the number of views, such that once it exceeds the threshold, more views do not help?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper investigates the effect when introducing view multiplicity in contrastive learning. Specifically, the paper gives a generic information-theoretic derivation of such multi-view framework and shows that SimCLR loss is a special case of the derived `poly-view' contrastive learning. The paper concluded from the theoretical foundation that higher view multiplicity enables a new contrastive learning where, surprisingly, it is beneficial to reduce the batchsize and increase multiplicity. The paper also associate their  theoretical findings with experiments.

### Strengths
The paper delves into the impact of incorporating multiple perspectives in contrastive learning. The strength include: 

S1: it presents a comprehensive information-theoretic analysis of this multi-view approach and establishes that the SimCLR loss can be considered as a special case of the resulting 'poly-view' contrastive learning. 

S2: Based on the theoretical framework, the research concludes that greater view multiplicity facilitates a novel form of contrastive learning, wherein it proves unexpectedly advantageous to decrease the batch size while augmenting the multiplicity. 

S3: Furthermore, the paper justifies these theoretical discoveries with empirical experiments.

### Weaknesses
However, the paper has several weakness that worths further discussion. 

W1: What is the exact loss function the paper used to define the poly-view contrastive learning? It seems the Eq. (22) is the poly-view contrative loss, whereas it is in a very high level abstract and implicit form, making it hard to interprete how to compute the empirical loss for the terms, and why M=2 links to SimCLR. I recommend to make the loss in a more explicit form of empirical losses and interpretation (e.g., what is the used sufficient statistics for M=2 for SimCLR? ) 

W2: Empirical evidence lacks suitable interpretation and linkage to the significance of the theorems. For significance, I mean how we can use the theorem takeaways to practically improve the SSL algorithms? I expect to see the evidence on larger dataset with mainstream architecture such as ResNet and ViT/transformer with longer epochs. 

W3: It is unclear to me if the multiplicity of views simply benefits from more equivalent of epochs (in the experiments) or whether the exposure to the number of data has been constrained to be exactly same between the poly-view contrastive learning and other baselines. 

W4: There is no comparison between the proposed method and SOTA method, in terms of how the method contributes to and improves the SOTA methods under the theoretical foundations.

### Questions
Please see the 4 weakness above for questions to be addressed.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Although it is possible to design tasks that drawn arbitrary number of views, contrastive works typically focus on pairwise tasks. So, this paper investigates how to match more than two related views in contrastive learning, and derive new learning objectives by information maximization and sufficient statistics. They show that multi-crop reduces the variance of corresponding paired objective but fail to improve bounds on MI; Then they derive new objectives which solve tasks across all views through information theory, and show that the MI Gap is monotonically non-increasing with respect to the number of views. Also, the poly-view contrastive method is beneficial to reduce the batch size and increase multiplicity.

### Strengths
1. Generalizing the information-theoretic foundations to poly-view is an interesting idea, and the One-vs-Rest MI seems to be quite reasonable.

2. Those theoretical results are clear, the derivation process of the One-vs-Rest objective is convincing.

3. The paper is well written and easy to follow.

### Weaknesses
1. The assumption 3 in section 2.4.1 is kind of strong.

2. The experiments do not show the superiority of poly-view contrastive learning. The computation time is not evaluated by real time, and the downstream performance is not displayed.

### Questions
1. The experiments shown in section 3 display the relative compute of algorithms and show One-vs-Rest objectives could beat simCLR with the same relative compute, how about the real training complexity. And how does it perform on real downstream tasks.

2. I cannot fully understand why the One-vs-Rest loss could effectively reduce the training epoch and batch size.

3. The Geometric loss is actually also an extension of simCLR loss, just like Multi-Crop, but Geometric loss could be a tighter bound of MI while Multi-Crop cannot. It seems to be theoretically correct, but how can we understand it empirically.

4. An extension of simCLR loss outperforms the carefully designed SUFFICIENT STATISTICS loss in section 3, does it mean that the poly-view contrastive learning works mainly because the superiority of simCLR loss?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposed a multi-view (against the previous 2-view) contrastive learning, they provide theoretical and empirical evidence that their derived multi-view loss is better than previous multi-crop loss, as it provides a tighter lower bound on the generalized mutual information. They also provide real data evidence showing that their multi-view loss allows more efficient learning compared with previous two-view contrastive learning like SimCLR.

### Strengths
The paper investigated an interesting angle of contrastive learning: instead of increasing batch size, they increase the number of views. They provide a solid theoretical framework for their proposal, linking their proposed multi-view loss with previous SimCLR loss and the InfoMax framework. They also provide detailed analysis for comparing these two losses both theoretically and empirically. Overall it is clearly written and easy to follow, and the theoretical analysis aligns with the empirical findings is another big plus. Overall, they provide a new angle to improve the contrastive learning idea, which I believe might unleash further power of self-supervised learning.

### Weaknesses
I just have one suggestion: maybe you can comment (or leave for future work) about how other self-supervised learning can fit in your framework, or how your idea can be extended to other SSL methods like BYOL etc.

### Questions
I have no questions.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
