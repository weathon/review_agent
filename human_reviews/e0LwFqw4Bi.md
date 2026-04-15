# Towards Unified and Effective Domain Generalization

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 8, 3

## Abstract
We propose \textbf{UniDG}, a novel and \textbf{Uni}fied framework for \textbf{D}omain \textbf{G}eneralization that is capable of significantly enhancing the out-of-distribution performance of foundation models regardless of their architectures. The core idea of UniDG is to finetune models during inference time which saves the cost of iterative training. Specifically, we encourage models to learn the distribution of testing data in an unsupervised manner and impose a penalty regarding the updating step of model parameters. The penalty term can effectively reduce catastrophic forgetting issues as we would like to maximally preserve the valuable knowledge in the original model. Empirically, on up to 12 visual backbones, including CNN-, MLP-, and transformer-based models, ranging from 1.89M to 303M parameters, UniDG shows an average accuracy improvement of 5.4\% on DomainBed. We believe that these performance results are able to manifest the superiority and versatility of UniDG.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper, authors focus on enhancing the out-of-distribution generalization performance of foundation models regardless of their architectures with proposed a novel and Unified framework for Domain Generalization (UniDG). The core idea of UniDG is to finetune models during inference stage which saves the costs of iterative training. Specifically, authors encourage models to learn the distribution of testing data in an unsupervised manner and impose a penalty regarding the updating step of model parameters. The penalty term can effectively reduce the catastrophic forgetting issue via maximally preserving the valuable knowledge in the original model.

### Strengths
In other words, marginal generalization is proposed to update the encoder of Test-Time Adaptation (TTA) and differentiable memory bank is proposed to refine features for DG. Experiments on five datasets such as VLCS, PACS, OfficeHome and so on demonstrate the superiority compared with SOTA methods across 12 different network architectures.

### Weaknesses
The structure of paper is unfitable for most ML reader’s habits, especially, the part of related work should follow the introduction. There will be a better logical relationship for most ML conference paper.

### Questions
However, there are some questions need to be clarified from authors.
1.	Although ablation study is provided in Table4, in appendix D, what is the main reason behind introducing matrix products of learning representations, prototypes and pseudo labels ? How to understand making the memory iteration is learnable and differentiable ? 
2.	Moreover, L_i in formular 25 makes me confused. Please give the detailed explanation of discrepancy between formular 8 and line 9 of Algorithm 1. How about the hyper paremeter lambda in formular 8 ? There should add more ablation experiments.
3.	In Figure 5, please give the detailed explanation why the accuracy of “DomainNeXt” is lower than base when the samples are small. 
4.	There are many typos in manuscript, such as what does “DomainNeXt” mean in Figure 7 of Appendix E ? VLCS and PACS is quoted wrongly in Appendix F. Many formular typos are typed in Appendix B.2
5.     Last but not least, in Table 5 (b) efficiency of UniDG, I'm confused how to get the wall clock time. There is nothing to be analyzed from time complexity. How did you come to learn the conclusion that the proposed UniDG is a online learning scheme ?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper focuses on improving the generalization performance of foundation models finetuned during inference. The authors propose a penalty term that helps to reduce catastrophic forgetting during test-time adaptation. In particular, the authors propose Marginal Generalization - a tradeoff b/w freezing the encoder which would lead to underfitting and updating the decoder which would lead to catastrophic forgetting. The authors demonstrate empirically consistent improvement across different backbones on DomainBed benchmarks.

### Strengths
- Propose a tradeoff b/w freezing the encoder which would lead to underfitting and updating the decoder which would lead to catastrophic forgetting.
- Consistently improved benchmarks.

### Weaknesses
- Theoretical insight why marginal generalization is important for generalization in unseen domains is explained well in Appendix, but is very unclear from the text of the main paper. I think this important aspect should be better discussed in the main text.
- Also motivation for Differentiable Memory Bank should be more clearly written.

### Questions
- With the current formulation of Marginal Generalization, how can you avoid catastrophic forgetting on **source** domains, when even if you impose the distance constraint on the target domain, there are no guarantees it will be still obeyed on the source domain?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on the catastrophic forgetting issue during test-time training (TTA) for domain generalization. Specifically, this paper proposes a Marginal Generalization method to update the encoder for TTA, that is, Marginal Generalization aims to let the encoder learn representations of the target data within a certain distance from the representations obtained by the initial model. To cooperate Marginal Generalization, this paper also proposes Differentiable Memory Bank to facilitate TTA. Experiments on five domain generalization benchmarks demonstrate the effectiveness of the proposed methods.

### Strengths
- The catastrophic forgetting issue during TTA for domain generalization is well motivated.

### Weaknesses
- The discussion about related work is not sufficient. In the section of related work, this paper simply listed many related works, but didnot discusses the relation between the proposed method and the mentioned related works.

- This paper is more likely to be a Test-Time Domain Adaptation work. So I think Test-Time Domain-Adaptation is more suitable in this paper rather than Domain Generalization. 

- I dont believe it is the first time to discuss the catastrophic forgetting issue during TTA for domain generalization. But I do not see any discussions about how to solve the catastrophic forgetting problem in the existing works, such as [1][2], to name a few.

- In the experimental part, such as Table 2, this paper didnot explain why the performances with TTA are inferior to that without TTA. In Table 2, PL, PLClf, SHOT, Tent, TentBN, TentClf are all inferior to None. It is kind of weird, which needs explaination.

- As I can see in this paper, catastrophic forgetting is the main problem to be solve. However, most of experiments are conducted on domain generalization benchmarks to show how well the proposed method performs on the target domains. Only a simple ablation study in Table 5 is conducted to validate that the catastrophic forgetting issue has been mitigated via the proposed method. I think the organization of the experiments is mismatched with the major motivation discussed in this paper.

[1] Continual Source-Free Unsupervised Domain Adaptation.

[2] CoSDA: Continual Source-Free Domain Adaptation.

### Questions
See the weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
