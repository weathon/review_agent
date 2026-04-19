# Neural Collapse meets Differential Privacy:  Curious behaviors of NoisySGD with Near-Perfect Representation Learning

- Decision: Reject
- Scores: 3, 5, 3

## Abstract
In recent studies, it has been demonstrated that large-scale representation learning through pre-training on gigantic datasets significantly enhances differentially private learning for downstream tasks. By training on Google's proprietary JFT dataset, one can achieve an unprecedented 83% Top 1 accuracy on ImageNet with strong privacy parameters $(0.5,8\times 10^{-7})$-DP, even given the high dimensionality of the feature space.
While the exact behaviors of NoisySGD in these scenarios remain theoretically challenging to analyze, we explore an idealized setting using a layer-peeled model for representation learning, which results in interesting phenomena of the learned features known as neural collapse. Under this setting, we have observed several notable behaviors of NoisySGD.
Specifically, we demonstrate that under perfect neural collapse, the misclassification error is unaffected by the dimension of the features. This dimension-independent result holds with any learning rate and even with class imbalance and is not influenced by the nature of the loss functions.
Nevertheless, a dimension dependency emerges when introducing minor perturbations in either the feature or model space. To address this dependency under perturbation, we suggest several strategies, such as pre-processing features or employing principal component analysis to reduce feature dimensions.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
Recent study by De et al. (2022) reports that large-scale representation learning via pre-training on a gigantic dataset significantly enhances differentially private learning in downstream tasks. While the exact behaviors of NoisySGD on these problems remain intractable to analyze
theoretically, the authors consider an idealized setting of a layer-peeled model for representation learning by neural collapse.

The writing is good and the results seem interesting, which attracts me to check their proof. The proofs are very simple. $M_k,k=1,\cdots,K$ form an ETF frame, which separate categories very well, the zero initialization makes $f_0(M_k)=0$ for all $k=1,\cdots,K$, which is very weird. The one-step NoisyGD seems not useful at all.

 I am not sure your results are for NoisySGD or NoisyGD. In introduction section, your statements are all about NoisySGD, but the other parts for NoisyGD. Moreover, there is no definition about NoisySGD at all in the whole paper.

### Strengths
The presentation is good.

### Weaknesses
The results are not meaningful.

### Questions
How about $f_W(x)=Wx+b$?

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies theoretical analysis for differentially private fine-tuning under neural collapse. Specifically, this paper shows that if the neural collapse is assumed, and we only fine-tune last layer, the accuracy bound is indepedent of dimension and only related to privacy parameter. If the neural collapse is not perfect on private dataset, this paper also shows that perturbation on the features, class imbalance would require the accuracy to be depedent on the dimension. This paper also propose data normalization and PCA to mitigate this non-robustness issue.

### Strengths
This paper provides first theoretical understanding of DP fine-tuning reduces the error rate on down-streaming task. The setting of neural collapse is interesting and may be enlightening for potential future research.

### Weaknesses
1. Typos: In theorem 2, is it $\gamma$ accuracy or $1-\gamma$ accuracy?
2. All of the proofs only analyze one-step Noisy-GD algorithm under a very strong neural collapse assumption. This setting is too simple and might not be reflecting what is happening in De at al (2022). If perfect neural collapse holds, then there is no need for further training. For example,  in theorem 2, you can set the clipping threshold G to be very small (near zero) to get near zero error rate. This suggests that you don't have to train on the private data if the neural collapse is assumed. This bound might not be very useful.
3. The proposed tricks are not demonstrated empirically on real datasets.
4. The proof is simple and the technical contribution is limited.

### Questions
1. Is there any empirical improvement by using the proposed data normalization and PCA tricks? I am curious because DP-PCA would also needs privacy-utility trade-off that needs to be accounted.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors use neural collapse theory to analyze the behavior of last-layer fine-tuning with DP. They show that dimension independence emerges in a certain sense under perfect neural collapse and that small perturbations in the train and test data can disturb this independence. They show that data normalization and dimensions reduction can recover the dimension independence in the face of such perturbations.

### Strengths
1. The general phenomenon explored in this paper (i.e. the empirical success of DP deep fine-tuning in high dimensions) is very interesting and timely.
2. The results are presented with a high degree of technical precision and fluency.
3. The result of Theorem 2 seems surprising and interesting to me, although I don't yet have a strong intuitive understanding of the proof.

### Weaknesses
1. **It may be difficult for some readers to understand:** Given the topic, I imagine many readers will be familiar with differential privacy but less familiar with neural collapse. As a result, the third paragraph of the introduction and the corresponding figure 1 will be meaningless to them without more explanation. Some of the introductory material is present in section 2.2, but it is a bit technical and not well-suited to newcomers. I would recommend giving a high level explanation of neural collapse in the introduction to help readers.

### Questions
1. Introduction, paragraph 1, last line: do you mean "no-privacy utility tradeoff"
2. Bottom of page 7, Sigma is missing a backslash.
3. The dimension independence of Theorem 5 requires $\beta_0$ to scale with $p$, but the non-robustness results in 3.2 and 3.3 would also become dimension independent if $\beta$ chosen to vary with $p$ in a similar way. Because of this, it's not clear what we are gaining from dimension reduction in section 4.2.
2. It's not clear to me how this analysis of neural collapse applies to full fine tuning. The success of DP full fine tuning is most surprising because of the many total parameters of the networks (not just in the last layer). Neural collapse may explain some of the dynamics of last layer fine-tuning as presented in this paper, but clearly something interesting must be happening at intermediate layers in the full fine-tuning setting. I think it would be very helpful to mention whether there is any way that these results might shed light on the dynamics of intermediate layers.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
