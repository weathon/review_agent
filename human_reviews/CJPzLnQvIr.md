# QuickDrop: Efficient Federated Unlearning by Integrated Dataset Distillation

- Avg Score: 3.75
- Decision: Reject
- Scores: 6, 3, 3, 3

## Abstract
Federated Unlearning (FU) aims to delete specific training data from an ML model trained using Federated Learning (FL).
We introduce QuickDrop, an efficient and original FU method that utilizes dataset distillation (DD) to accelerate unlearning and drastically reduces computational overhead compared to existing approaches. In QuickDrop, each client uses DD to generate a compact dataset representative of the original training dataset, called a *distilled dataset*, and uses this compact dataset during unlearning. To unlearn specific knowledge from the global model, QuickDrop has clients execute Stochastic Gradient Ascent with samples from the distilled datasets, thus significantly reducing computational overhead compared to conventional FU methods. We further increase the efficiency of QuickDrop by ingeniously integrating DD into the FL training process. By reusing the gradient updates produced during FL training for DD, the overhead of creating distilled datasets becomes close to negligible. Evaluations on three standard datasets show that, with comparable accuracy guarantees, QuickDrop reduces the duration of unlearning by 463.8$\times$ compared to model retraining from scratch and 65.1$\times$ compared to existing FU approaches. We also demonstrate the scalability of QuickDrop with 100 clients and show its effectiveness while handling multiple unlearning operations.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work studies how to extract a much smaller dataset for federated unlearning (FU) using dataset distillation (DD), which speeds up the FU time. The method is incremental to (Zhao et al., 2021), adapting it to the federated setting with many clients. This method reuses the gradient updates produced during FL training for DD, so the overhead of creating distilled datasets is very small. However, the result quality is not as good as the more expensive baselines, but finetuning steps on the original dataset can mitigate this issue to achieve comparable accuracy.

### Strengths
The approach is effective especially in speeding up the unlearning request processing, and the idea makes sense and is intuitive.
The experiments are comprehensive.

### Weaknesses
The idea is very incremental to (Zhao et al., 2021). In particular, the main technical sections are Sec 3.2 and 3.3, but Sec 3.2 is basically reviewing (Zhao et al., 2021). Sec 3.3 is short and describes the small changes of (Zhao et al., 2021) in Algorithm 1 to achieve Algorithm 2, where the changes are very standard migration to the FedAvg setting.

The accuracy of QUICKDROP is lower than that of the baselines, and a solution of finetuning is proposed in Figure 4 to be effective. However, finetuning is claimed to be done on the original dataset. Does that mean you still need all clients to collaborate to complete the finetuning?

Finally, while the method is intuitive, it is very simple without much theory.

### Questions
Finetuning is claimed to be done on the original dataset. Does that mean you still need all clients to collaborate to complete the finetuning?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses the problem of federated unlearning and proposes an efficient method based on dataset distillation. It is used to accelerate both the training and unlearning stages The authors conduct unlearning by stochastic gradient ascent. The proposed method is evaluated experiments on three datasets.

### Strengths
1.	Federated unlearning is an interesting research topic. The authors also focus on a general framework of sample-wise unlearning, which is also applicable to class-wise and client-wise.
2.	The paper is clearly written and well organized.

### Weaknesses
1.	The technical contribution is incremental. The proposed method uses a gradient ascent + fine-tuning framework, which has been introduced in [1, 2]. The main contribute of this paper is the use of data distillation (a direct application of existing algorithm), which is similar to the idea of representative data selection in [3]. Thus, this paper seems to be a combination of existing methods.
2.	The proposed method is applicable to sample-wise unlearning, but the experiments are conducted on class-wise and client-wise unlearning. I was expected to see the evaluation of sample-wise unlearning, since it has broader applicability.
3.	Lack of baseline methods. I suggest that the authors consider including some representative ones or at least one method from more different approaches. Additionally, PU-MP is a CNN-specific method. It may not be appropriate to limit the target model to CNN and compare it with the proposed method, which is more general. I list some of the approaches for your consideration:
i.	Efficient retraining: [4, 5]
ii.	Influence function: [6]
iii.	Gradient ascent + regularization: [7, 8] ([8] is compared in the paper)
iv.	Roll back gradient + knowledge distillation: [1]
v.	Scaling: [9]

[1] 2022, Federated Unlearning with Knowledge Distillation
[2] AAAI’21, Amnesiac Machine Learning
[3] IJCAI’22, ARCANE：An Efficient Architecture for Exact Machine Unlearning
[4] IWQOS’21, FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models
[5] WSDM’23, Federated Unlearning for On-Device Recommendation
[6] INFOCOM’22, The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining
[7] ICML’22, Federated Unlearning: How to Efficiently Erase a Client in FL
[8] IEEE Network, Federated Unlearning: Guarantee the Right of Clients to Forget
[9] 2023, VERIFI: Towards Verifiable Federated Unlearning

### Questions
Please refer to Weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes QuickDrop, a new federated unlearning (FU) approach that uses dataset distillation to efficiently remove specific data from a collaboratively trained machine learning model. QuickDrop has each client create a highly condensed "distilled" dataset that preserves the key features of their local training data using dataset distillation. Clients then use this compact distilled dataset during the unlearning and recovery phases rather than their full training datasets. This drastically reduces the computation cost of unlearning. The paper also proposes integrating dataset distillation into the regular federated learning training process by reusing gradients, avoiding extra overhead. Evaluations on MNIST, CIFAR-10, and SVHN show QuickDrop reduces unlearning time by 463.8x compared to full model retraining and 65.1x compared to prior FU techniques.

### Strengths
1. Proposes QuickDrop, which integrates dataset distillation into regular federated learning training by reusing gradients, avoiding extra overhead.
2. Experiments on 3 datasets conclusively show QuickDrop reduces unlearning time by 65-464x over baselines.

### Weaknesses
1. The paper does not discuss communication overhead of exchanging distilled datasets, which is the main concern I have regarding whether the proposed QuickDrop could be useful.
2. Some of the important details are not clear, for example, how non-IID data affects QuickDrop and the impact of distillation hyperparameters.

### Questions
I don't have other questions in addition to the questions in the weaknesses section.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This research is primarily focused on the development of a method designed for the removal of specific training data from a Machine Learning (ML) model that has been trained using Federated Learning (FL). This specialized technique, known as Federated Unlearning (FU), is the central objective of the study.

To achieve this, the authors introduce an innovative and highly efficient FU methodology named QUICKDROP. QUICKDROP leverages the power of Dataset Distillation (DD) to streamline the unlearning process, resulting in a significant reduction in the computational resources required when compared to conventional approaches. The fundamental principle of QUICKDROP involves each client generating a concise and representative dataset referred to as a "distilled dataset" using DD techniques. These distilled datasets play a crucial role in the subsequent unlearning phase.

Furthermore, the authors demonstrate their ingenuity by seamlessly integrating Dataset Distillation (DD) into the Federated Learning (FL) training process. This integration enhances the overall efficiency of QUICKDROP by capitalizing on the reuse of gradient updates generated during FL training for DD purposes. Consequently, the overhead associated with creating distilled datasets is effectively minimized.

The empirical evaluation of QUICKDROP's performance, conducted across three standard datasets, conclusively demonstrates its capacity to deliver remarkable efficiency gains in the context of federated unlearning.

### Strengths
1. The development of QUICKDROP significantly improves the efficiency of the Federated Unlearning (FU) process. By incorporating Dataset Distillation (DD) and Stochastic Gradient Ascent, the method considerably reduces the computational resources and time required for unlearning compared to existing approaches.

2. The research demonstrates the scalability of QUICKDROP by evaluating its performance with a large number of clients (100 clients). This scalability is vital in practical FL scenarios involving numerous participants.

### Weaknesses
The integration of Dataset Distillation (DD) and Federated Learning (FL) is not a novel concept. Previous research presented at conferences such as ICLR and CVPR has explored similar strategies involving the replacement of original data with distilled data for FL training. Applying this strategy to a new application problem, such as federated unlearning, does not represent a sufficiently novel contribution.

The experimental results presented in the research do not demonstrate strong evidence of effectiveness. While the findings indicate a significant improvement in efficiency, there are concerns about the effectiveness of the method, particularly in terms of accuracy. Sacrificing effectiveness for increased efficiency may not be a favorable trade-off in practical applications.

The organization of the paper could benefit from improvement. For instance, the extensive space dedicated to explaining dataset distillation through gradient matching is unnecessary. This concept is derived from existing work and is not a novel contribution of this research.

### Questions
Please refer to "Weaknesses" part.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
