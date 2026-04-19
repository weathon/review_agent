# SubDiff: Subgraph Latent Diffusion Model

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 3

## Abstract
Diffusion models have achieved impressive performances on generative tasks in various domains. While numerous approaches are striving to generate feature-rich graphs to advance foundational science research, there are still challenges hindering generating high-quality graphs. First, the discrete geometric property of graphs gains difficulty in capturing complex node-level dependencies for diffusion model. Second, there is still a gap to simultaneously unify unconditional and conditional generation. In this paper, we propose a subgraph latent diffusion model to jointly address above challenges by inheriting the nice property of subgraph. Subgraphs can adapt diffusion process to discrete geometric data by simplifying the complex dependencies between nodes. Besides, subgraph latent embedding with explicit supervision can bridge the gap between unconditional and conditional generation. To this end, we propose a subgraph latent diffusion model (SubDiff) by taking subgraphs as minimum units. Specifically, a novel Subgraph Equivariant Graph Neural Network is proposed to achieve graph equivariance. Then a Head Alterable Sampling strategy (HAS) is devised to allow different sampling routes along diffusion processes, unifying the conditional and unconditional generative learning. Theoretical analysis demonstrate that our training objective is equivalent to optimizing the variational lower bound of log-likelihood. Extensive experiments show SubDiff achieving better performance in both generative schemes.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a subgraph latent diffusion model to embed the subgraph into the latent space. The explicit supervision used in the subgraph latent diffusion model helps to embed the label information in latent space. A novel subgraph equivariant GNN is also raised to extract the graph representation. A sampling method HES is also devised to unify conditional and unconditional generative learning.

### Strengths
- The paper considers the subgraphs as minimum units instead of separate nodes, which makes sense and has the potential to enhance the substructure perception of GNNs.
- Embeding the condition information to the latent space sounds interesting. 
- The paper proposes a simple method to unify the conditional and unconditional generation via setting different starting Gaussian noise.

### Weaknesses
- The paper claims that they propose a new frequency-based subgraph extractor. However, the method actually used is MiCaM, proposed by (Geng et al., 2023).
- The assumption of the latent embeddings is strong (sec 5.1): the condition must be numerical from 0 to 1 and comparable.     

- The presentation of this paper is not unclear. It misses many important details in the main text, such as model architecture(see minor concerns), and sampling process.
- There exist many approaches that can be used to extract the subgraph, such as BRICS. The ablation study can be added to support the choice of MiCaM.


Minor concerns:
- The explanation of $E_{\theta}$ and $D_{\xi }$ in Eq (5) are missing.
- The explanation of “pooling” in Eq (7) is missing.
- In Eq 12, the specific forms of  $L^2$ and $L^2$ are not given. In this case, how to calculate the element-wise multiplication between $x_{G_s}$ and $L^2$?

 While I think the subgraph diffusion is a promising idea, the presentation of the method and experiments require a substantial amount of work and are not ready for ICLR24.

### Questions
- What is the meaning of Proposition 1? From my understanding, if we get an unconditional generative model, the model can be easily extended to a conditional version. E.g. EDM.
- The input of the denoising network in Alg. 1(training process) is x while z in Alg. (sampling)? Why?

### Soundness
3 good

### Presentation
2 fair

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
The paper proposes a subgraph diffusion model that learns to treat the subgraph as the basic component of the diffusing object. To do so, it overcomes several design challenges, which are demonstrated in the paper thoroughly.

### Strengths
The proposed method is technically solid. (1) it has shown that treating subgraphs as the latent variables can also maintain a lower bound of the graph likelihood; (2) it tackles the Equivariant problem when treating subgraphs as diffusion object; (3) experiment result has shown promising result of the propose method

### Weaknesses
(1) the motivation that drives such an approach may not be sufficient -- it's not very convincing that subgraph-level diffusion will address the problem "graph generative models generate not only the features of each node but also the complex semantic association between nodes."

(2) The claim that the model unifies condition and unconditional generation seems to be irrelevant to the subgraph diffusion. It's not sure why these two components are proposed in one submission

(3) Related works missing -- there is a previous work that has proven that the latent graph diffusion model has a proper lower-bound of the graph likelihood [1].

[1] Chen, Xiaohui, et al. "Nvdiff: Graph generation through the diffusion of node vectors." arXiv preprint arXiv:2211.10794 (2022).

### Questions
See weakness

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
This paper proposes a subgraph latent diffusion model for 3D molecular generation. Its main contributions are: 1. overcoming the dependency between nodes through subgraphs; and 2. proposing a unified model for both unconditional and conditional generation.

### Strengths
1.	A novel subgraph latent diffusion model is proposed in this paper. 
2.	A unified framework is proposed for both unconditional and conditional generation.

### Weaknesses
1.	The authors first propose that the discrete geometric property of graphs makes it difficult to capture complex node-level dependencies for diffusion models. They claim that this problem can be solved by using subgraphs, which they present as the main contribution of this paper. I disagree with this viewpoint. Firstly, the existence of complex node-level dependencies has nothing to do with whether the data is discrete or continuous. Whether it is discrete atomic features or continuous positional features, complex node-level dependencies still exist. Secondly, while abstracting multiple nodes into subgraphs eliminates node dependencies, there can still be dependencies between subgraphs. However, the paper does not propose a solution for this subgraph dependency issue.
2.	The authors' second contribution is the proposal that subgraph latent embedding with explicit supervision can bridge the gap between unconditional and conditional generation. However, the explicit supervision used in the paper is graph-level label, and I do not get the contribution of subgraph latent embedding. In other words, the proposed solution in the paper, such as pooling subgraph latent embedding as in Eq. 7, could be replaced by pooling node latent embedding to obtain graph latent embedding. I doubt the necessity of using subgraph latent embedding to bridge the gap between unconditional and conditional generation.
3.	The description of the methods proposed in the paper is not clear enough. Two methods are proposed in the paper: subgraph-level equivariant architecture (SE-GNN) and head-alterable sampling strategy. Firstly, the paper lacks a clear explanation of how to implement L1 and L2 in SE-GNN. Secondly, in Section 4.2, the authors do not explain why it is called head-alterable, and it is not clear why this is considered a sampling strategy. From the beginning of page 6, this method changes the mean of the Gaussian distribution during the training phase. Additionally, the paper does not explain how to personalize the prior distribution for each property $y_i$, some equations to be presented.
4.	Some recent related works need to be compared, such as MDM [1].

[1] Huang, Lei, et al. "Mdm: Molecular diffusion model for 3d molecule generation." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 4. 2023.

### Questions
Proposed in Weaknesses

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
