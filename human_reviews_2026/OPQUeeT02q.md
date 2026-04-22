# Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 8, 2, 6

## Abstract
Transfer learning aims to facilitate the learning of a target domain by transferring knowledge from a source domain. 
The source domain typically contains
semantically meaningful samples (*e.g.*, images) to facilitate effective knowledge transfer. However, a recent study observes that the noise domain constructed from simple distributions (*e.g.*, Gaussian distributions) can serve as a surrogate source domain in the semi-supervised setting, where only a small proportion of target samples are labeled while most remain unlabeled. 
Based on this surprising observation, we formulate a novel problem termed *Semi-Supervised Noise Adaptation* (SSNA), which aims to leverage a synthetic noise domain to improve the generalization of the target domain. To address this problem, we first establish a generalization bound characterizing the effect of the noise domain on generalization, based on which we propose a Noise Adaptation Framework (NAF). Extensive experiments demonstrate that NAF effectively utilizes the noise domain to tighten the generalization bound of the target domain, thereby achieving improved performance. 
The codes are available at https://anonymous.4open.science/r/SSNA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel solution for Semi-Supervised Noise Adaptation (SSNA), and proposes a new Noise Adaptation Framework (NAF) that leverages a synthetic noise domain, that is, constructed from random distributions to improve generalisation in a semi-supervised target domain. The authors proof a theoretical generalisation bound inspired by domain adaptation theory and design a learning objective combining supervised, noise-domain, and alignment losses. Experiments on CIFAR-10/100, DTD, Caltech-101, and ImageNet-1K show consistent improvements over Empirical Risk Minimisation and compatibility with existing semi-supervised learning methods.

### Strengths
Interesting improvement of existing concept: The idea of transferring structure from a synthetic, non-semantic noise domain is novel and could inspire future research on structure-based generalisation.

Clear theoretical definition and framing: The authors connect the proposed method to domain adaptation bounds, offering a explanation for why structured noise might help generalisation.

Extensive experiments: The empirical section is broad and thorough, showing improvements across multiple benchmarks.
Good writing and reproducibility: The paper is well written, clearly organised, and accompanied by public code, which is appreciated.

### Weaknesses
The proposed method is a extension of prior work "Noise May Contain Transferable Knowledge: Understanding Semi-supervised Heterogeneous Domain Adapta…", with some new experiments and explanation, however the visualisation used for explanation is almost identical (i.e. Figure 1 of this paper vs Figure 4 of the referred paper)

SSNA made a strong assumption on a one-to-one mapping between noise classes and target classes and a known number of target classes. This can be limited in the setting like imbalanced dataset. And the paper didn't not discuss the constraints or experiment with dataset as such.

In most of the experiment results, only the "accuracy" metrics are presented, this can be misleading if the dataset is imbalanced, and does not provide error type. Potentially need to provide other metrics like recall or f1.

Only random noise were discussed and experimented, would other type of noise like Gaussian, uniform, structured noise also applicable?

The paper does not compare to modern self-supervised or contrastive regularisation methods that might yield similar improvements without requiring a noise domain.

### Questions
Overall, I'm still slightly confused on the method is trying to minimised the error on the randomly labelled noise, but there is no clear link to show reduce the error on noise lead to lower error on the real target domain? This can be regularisation by add "noise" to the training instead of really "transferring" learning as it claim?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Unlike conventional transfer learning where the source domain consists of semantically meaningful data, SSNA leverages a noise domain generated from Gaussian distributions as a surrogate source domain to assist target-domain learning. The authors theoretically derive a generalization error bound for the target domain and propose the NAF, which tightens this bound by jointly minimizing the empirical risk on the target domain, the noise-domain risk, and the inter-domain distributional discrepancy. Experiments conducted on multiple benchmark datasets demonstrate that NAF achieves significant performance improvements compared with various existing methods.

### Strengths
- The paper introduces the SSNA problem, which treats the noise domain as a surrogate source domain. This setting holds potential application value in scenarios with privacy constraints or data scarcity.
- In Section 4.1, the authors derive a generalization bound (Theorem 1), explicitly showing that the bound depends on three components: the empirical error on the target domain, the empirical error on the noise domain, and the H-divergence between them.
- The code and parameter configurations are detailed in Appendix B, and an open-source link is provided to enhance reproducibility.

### Weaknesses
- There are concerns regarding the assumptions and some conclusions of NAF. [1] demonstrates that deep neural networks trained on completely random labels or random inputs can fully memorize arbitrary input–output mappings. [2] similarly finds that during optimization, deep networks with structural priors such as convolutional weight sharing, batch normalization, and nonlinear mappings tend to enforce linearly separable clustering in feature space. 
Considering these findings, the reviewer is concerned that the “discriminative structure formed in the noise domain” claimed by NAF may in fact result from pseudo-structural effects induced by the inductive bias of deep models like ResNet, rather than from the noise distribution itself.

- There are also concerns about the way the noise domain is generated. The paper uses a Gaussian distribution, but in real-world settings, natural noise (e.g., Poisson or long-tailed distributions) often exhibits non-Gaussian characteristics. Under such conditions, can the proposed noise projector (g_n) and distribution alignment mechanism still effectively extract discriminative structures? Since non-Gaussian noise tends to have larger intra-class variance, forming compact clusters within each class may be difficult, potentially weakening the effectiveness of knowledge transfer.

- Would the noise distribution parameters (e.g., variance σ, dimensionality d) affect generalization performance?

[1] Understanding deep learning requires rethinking generalization, ICLR'17

[2] Learning to See by Looking at Noise, NIPS'21

### Questions
This paper addresses a highly interesting and novel problem. The writing and presentation are excellent, and the figures and results are consistent with the claims. 

However, some issues may influence the final rating:

- Are there any visualizations, experiments, or theoretical analyses demonstrating that the transferable discriminative structure in the noise domain of NAF does not originate from the network itself?

- Would different types of noise affect the conclusions and experimental results?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work propose a Noise Adaptation Framework (NAF), which introduces random Gaussian clusters as anchors to align unlabeled data in semi-supervised learning.

### Strengths
- the idea is simple and easy to follow

### Weaknesses
__Major Concerns:__
- theoretical justification
  - the proof of Theorem 1 is missing, and the derivation can be problematic.
  - rather than $d_{\mathcal{H}\Delta\mathcal{H}}(P_n,P_t)$, it should be $d_{\mathcal{H}\Delta\mathcal{H}}(D_n,D_t)$ that introduces the uncertainty.
  - the right-hand side should include a source error, i.e., $\epsilon_n(f)$
  - please provide the proof for: $\lambda\leq \hat{\lambda}$ since apparently some terms related to model complexity is missing.
  - $\epsilon_t(f)$ should not limited to labeled target data, which contradicts  line 243. 
  - I strongly suggest the author check [1] proposing a theory for SSDA based on Ben-David (2010), which may help the derivation.
- algorithmic design
  - randomly sampled clusters from $\mathcal{N}(0,I)$ can overlap
  - the conditional discrepancy between randomly initialized Gaussian clusters and target features can be extremely large that cause substantial negative transfer.
  - the assumption that target feature must follow mixture of Gaussian distribution is too strick
- experiment results
  - the performance is far below the SOTA SSL methods such as DST
  - as for an increment to SSL, the comparison should be made between e.g., DST + LERM & DST + NAF


__Minor Concerns:__
- the code is not available from the provided link
- too much subscripts such as $u,l,n,e,t$, which is confusing

***
[1] Learning Invariant Representations and Risks for Semi-supervised Domain Adaptation, CVPR 2021

### Questions
see above

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduced a transfer learning based theoretical framework that learn from the noise domain to help the target domain in a semi-supervised learning manner.

### Strengths
1. The author formulated a novel theoretical framework to upper-bound the semi-supervised domain adaptation, interestingly, the source domain is a noise domain, and this paper is well-motivated.
2. The method origninated from the theretical framework is reasonable.
3. The experiment is good.

### Weaknesses
1. The underlying assumption of this paper is Gaussian distribution, so the author may discuss some other noise distribution settings.
2. I think the success of the knowledge transfer from the source to target is, the class is actually seperable from the source domain, e.g., the author is actually transfering the seperable characteristic of the noise source domain to the target domain (from the Figs of the main paper). Therefore, one concern arises that, for the Gaussian which generates noise in the source domain, will the different sampling strategy would the performance drops significantly? And whether the distance of the sampled Gaussians would affect the performance.
3. Did you experiemnt on the domain adaptation dataset? e.g., VisDA? Since your contribution is the knowledge transfer, why you only employ the CV tasks.
4. From Fig.4, it seems the $\mathcal{L}_n$ is the main contribution to lead to the performance improvement, but how you construct the label of the source noise domains? Could you be more clear on this point?

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
3
