# DeNAV: Decentralized Self-Supervised Learning with a Training Navigator

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3

## Abstract
Current Federated Self-Supervised Learning (FSSL) methods can achieve effective learning on edge devices with unlabeled data. However, in realistic settings, it is not easy to ensure that distributed clients at a large scale can efficiently communicate with a central server. In this work, we study an essential scenario of Decentralized Self-Supervised Learning (DSSL) based on decentralized communications. It is a highly challenging scenario where only unlabeled data is used during the pre-training stage, and the communication between clients involves only model parameters without data sharing. We propose a novel method to tackle the problems, which we refer to as Decentralized Navigator (DeNAV). DeNAV utilizes a lightweight pre-training model, namely the One-Block Masked Autoencoder, with a training navigator to evaluate selection scores for the connected clients and plan the training route based on these scores, eliminating the reliance on server aggregation in federated learning. Comprehensive experimental validation demonstrates that DeNAV surpasses the most advanced FSSL and Gossip Learning methods in terms of accuracy and communication costs.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In realistic FL, it is difficult to ensure that large-scale clients efficiently communicate with a central server. This work studies an essential scenario of Decentralized Self-Supervised Learning (DSSL) based on decentralized communications, in which only unlabeled data is used during the pre-training stage, and the communication between clients involves only model parameters. This paper proposes a method, Decentralized Navigator (DeNAV), utilizing a lightweight pre-training model, namely the One-Block Masked Autoencoder, with a training navigator to evaluate selection scores for the connected clients and plan the training route, eliminating the reliance on server aggregation.

### Strengths
1. The proposed scenario is important for FL. The massive clients may lead to in-efficient communication with central server.
2. This paper is written clearly.

### Weaknesses
1. The motivation is not strong enough, due to the lack of literature review. The Gossip learning does not constrain that every client must communication with all neighbors. Some Gossip federated learning works also propose to let clients only communicate with one or several neighbors [1][2].
2. In section 3.1, it seems that the described scnario looks like the continual learning. Specifically, the trained model is communicated and trained across clients. How to guarantee the convergence of this training scheme?
3. The proposed methods seem to be limited in the context of transformer-based model architectures.
4. The proposed selection score (6) seems to be little heuristic. How is this equation derived? Can such a selection score ensure convergence?
5. Experiment settings are not clear. What is the non-IID degree used, i.e. alpha in dirichlet sampling? For CIFAR-100 and Mini-INAT show that IID accuracy is better than non-IID, which seems to be impossible.

[1] MATCHA: Speeding Up Decentralized SGD via Matching Decomposition Sampling. In ICC 2019.
[2] GossipFL: A Decentralized Federated Learning Framework With Sparsified and Adaptive Communication. In TPDS 2022.

### Questions
See weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a decentralized self-supervised learning approach based on pre-training an auto encoder that can then be extended and fine-tuned on downstream tasks. The pre-training includes a client selection approach with heuristically defined utility functions. Experiment results confirm the advantage of the proposed DeNAV algorithm compared to baselines.

### Strengths
Self-supervised learning is an important and practical topic. This paper considers it in the decentralized/federated scenario, which is good.

### Weaknesses
- Decentralized learning has been widely studied in the literature. The training of auto encoders as in this paper is simply a specific type of decentralized learning, where common decentralized SGD algorithms can be applied. It is not quite clear what is new. 
- In the same way, client selection has been widely studied in the context of federated learning, where different client selection algorithms have been proposed with convergence analysis. This paper presents a heuristic client selection method. Its advantage over other existing methods that have more theoretical rigor is not clear. 
- There is no theoretical analysis of the overall DeNAV algorithm proposed in this paper. Proposition 1 is too informal as a mathematical claim, since any model can be approximated as a linear model if one allows an arbitrarily high approximation error. It is the bound of the approximation error that is more interesting, but such a bound has not been derived. Theorem 1 seems to be simply a least-squares regression result, which is straightforward. In general, it is not quite clear what is the usefulness of the theory presented in Section 3.2, since it is based on possibly inaccurate linear approximation and the main result is straightforward. It does not show the convergence of the overall algorithm, particularly with the client selection mechanism in Section 5.

### Questions
Please clarify the concerns mentioned under weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
