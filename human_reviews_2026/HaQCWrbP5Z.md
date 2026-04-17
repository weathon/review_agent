# Leave one Expert Out: Robust Uncertainty Quantification via Intrinsic Cross-Validation

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Estimating epistemic uncertainty remains an important challenge in modern Deep Learning (DL). We propose a novel architecture, called Leave one Expert Out (LEO), which is a form of a mixture-of-experts model with latent-space-distance-aware router and a null expert, representing prior belief, to which output of the model collapses if testing datapoint is too different from any of datapoints experts were trained on. This architecture allows to temporarily drop experts from the model, and we utilise this property to train the router to leverage the predictions of remaining experts to make predictions for the datapoints normally assigned to the expert currently removed from the model. We coin this mechanism \textit{intrinsic cross-validation} and show, such a trained router excels at estimating epistemic uncertainty for both in and out of distribution inputs. We demonstrate state-of-art performance on uncertainty quantification in regression benchmarks, such as UCI problems or age prediction on UTK-Face, and CIFAR10 classification benchmark. We also show the proposed method can achieve superior performance in surrogate-based black-box optimization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper introduces LEO, an architecture and training procedure for enabling models that quantify their epistemic uncertainty. It uses a mixture of shallow experts and a router that assigns examples to experts. The method uses a training procedure that simulates OOD scenarios during training to force the router to fall back on a prior distribution for OOD samples. This design ensures the model remains comparable in size to standard architectures, unlike resource-intensive methods like deep ensembles, which require multiple full models.

### Strengths
The paper is interesting and tackles the important problem of epistemic uncertainty quantification in (single) neural networks. The "intrinsic cross-validation" is compelling.

### Weaknesses
- The writing is very procedural and lacks motivation and structure. For example, Section 2.2 does not mention that the purpose of the leave-one-expert-out (actually leave multiple experts out!) is to simulate OOD scenarios during training.
- Why randomly corrupted samples in the CIFAR-10 experiments? Why not use e.g. SVHN as Van Amersfoort et al. 2020 does? Did you only evaluate on covariate shift, not on semantic shift?
- Missing citations for claims in the introduction. Examples:
-- Line 34 "assessing the certainty of that prediction remains a notoriously difficult problem" → it all depends on the context. A well calibrated model is very good at estimating the (aleatoric) uncertainty?
-- Line 45: missing citation for aleatoric and epistemic uncertainty
-- Line 67: "The training process typically does not explicitly encourage the model to output high uncertainty in OoD cases." DeepEnsembles try to enforce diversity between the members (data shuffles, different parameter initialization, ...), resulting in dissagreement between the members for OoD inputs.
-- Line 256: No uses of MAE anywhere
-- Table 1 is not referenced in the text
-- Line 299: Are there more recent references to mixture-of-experts literature also related to uncertainty quantification?

### Questions
- The router is an OOD detector: it is the sole mechanism that pushes the predicted distribution to the prior. Why is there not an experiment that measures how well p_ϕ(t | x) is able to distinguish OOD from ID? How is the performance of the router affected by the choice of prior? What if the prior does not minimize the loss? In that case, the main mechanism behind LEO explained on line 258 saying that the router collapses to the "vague" prior would not stand?
- Unclear experimental details and lacking insights into and discussion of the results. What do you take the NLL of? Of the probability of Eq 1? How should OOD NLL in Table 2 be interpreted? Is the goal to make the likelihood of the OOD samples high? It is not clear how these numbers show that the model's epistemic uncertainty estimation is good.
- The multiple small "experts" are remeniscent of "ShallowEnsembles" proposed by [1] and benchmarked in [2]. What if you just average the experts' predictions? Is that actually what the router does? How different are the predictions of the experts? The only purpose of the experts is to allow the simulated OOD scenarios during training. Would you agree?

[1]: Lee et al. 2015 https://arxiv.org/abs/1511.06314
[2]: Mucsanyi et al. NeurIPS 2024 https://openreview.net/forum?id=x8RgF2xQTj
[3]: Durasov et al. CVPR'21 https://openaccess.thecvf.com/content/CVPR2021/html/Durasov_Masksembles_for_Uncertainty_Estimation_CVPR_2021_paper.html
[4]: Laurent et al. ICLR'23 https://openreview.net/forum?id=XXTyv1zD9zD

### Soundness
3

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
3

### Summary
This paper introduces a method called “Leave on Expert Out” (LEO) which takes a pretrained backbone network $f: \mathcal{X} \to \mathbb{R}^d$ and builds a mixture-of-experts model on-top of it. The paper claims that LEO often produces a mixture-of-experts model that has both good in-distribution prediction performance and relatively good (when compared to other methods) out-of-distribution prediction performance and calibration.

LEO works by training a set of expert prediction heads $h_1, \ldots , h_T: \mathbb{R}^d \to \mathcal{Y}$ on top of $f$, along with a router $r: \mathbb{R}^d \to \Delta^{T + 1}$ that weights the predictions of the experts (with one extra coordinate for a user-defined fallback prediction). Inference on this backbone works in the following way:

1. Given an input $x$, an output prediction $\hat{y}_t = h_t(f(x))$ is generated for each expert $h_t$. These predictions are then converted to output distributions $\delta_1, \ldots, \delta_T$. When doing regression, a scalar prediction $\hat{y}_t$ is converted into a dirac delta function centered at that prediction. When doing classification, predicted logits $\hat{y}_t$ are softmaxed to get a distribution over classes.

2. The final prediction distribution is a mixture of the $T$ expert output distributions along with a single user-specified prior probability distribution $\delta_0$. The weights of the mixture are given by $r(x)$.

   When mixing together dirac delta distributions, in order to preserve differentiability, the output distribution is approximated by a Gaussian whose moments match the true mixture distribution.

LEO training works in three stages:

1. First, each training datapoint is passed through a randomly initialized backbone network to produce a random embedding. These embeddings are then projected onto a random one-dimensional subspace (the same subspace is used for all embeddings), and the datapoints are split up into T contiguous buckets along this subspace, with each contiguous bucket having the same fraction of datapoints assigned to it.

2. T expert heads are created, one for each bucket. Expert-head $t$ is trained to perform well (when attached to the frozen backbone network $f$) at prediction on the $t$th bucket of data. 

3. Finally, the router $r$ is trained on two objectives simultaneously: Firstly, the mixture-of-experts model should perform well at in-distribution prediction. Secondly, the mixture-of-experts model should perform well on predicting datapoint $x$’s label, even when the expert corresponding to $x$’s bucket is dropped along with some other random set of experts. When dropping an expert, its output distribution is replaced by the user-specified prior distribution.

   The paper term the mechanism behind this second loss term “intrinsic cross-validation”.

The paper tests LEO in three settings: regression, classification, and performance at black-box bayesian optimization of the Ackley function when LEO is used as a surrogate model to inform UCB what point to sample next. On regression and classification, LEO is comparable to the best performing baseline methods. On optimization of the Ackley function, LEO outperforms baseline methods when optimizing the Ackley function.

### Strengths
S1: The LEO algorithm is quite interesting and was novel for me (though admittedly I am somewhat unfamiliar with methods that optimize for good calibration / OOD performance beyond the basics like ensembles). In particular, I liked how the paper was able to utilize a mixture of experts' approach, how they designed their router algorithm which naturally places weight on the prior for really anomalous embeddings, and their intrinsic cross-validation scheme.

S2: I liked that the paper linked to a copy of its codebase at https://anonymous.4open.science/r/leave-one-expert-out-DF01/. Reading through this codebase helped me understand the method and experiments better.

### Weaknesses
W1: The evaluations of LEO are all on very toy tasks and none of them involve large scale foundation models. Moreover, the evaluations that are run do not show LEO to be leading other methods by that large of a margin, instead LEO seems just to be among the set of best performing methods.

W2: On the theoretical front, the paper does not really make an attempt to show why LEO should in theory perform better than existing baseline methods. The paper comments on some basic properties of LEO (e.g. lines 166-170), but does not compare its theoretical underpinnings to other methods.

W3: The random-network-embedding-projection type-assignments seem quite arbitrary and not well motivated. I would be curious to compare this approach to some ablations like k-means clustering on trained embeddings, or just completely random assignments.

### Questions
Question #1: Regarding weakness W3, what is the motivation behind this approach? Alternatively, could some ablation experiments be run to provide empirical evidence regarding its effect?

Sugggestion #1: The authors could address weakness W1 by running experiments with LEO on large scale foundation models. I would be most interested to see experiments on making foundation models more calibrated at very hard tasks that foundation models only sometimes get right. If LEO beats out baselines here, I would be convinced it is a very good method.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes “Leave one Expert Out” (LEO) for uncertainty quantification in deep learning. LEO has a mixture of experts architecture, where the router uses the latent space representation of input to understand its distance from the training data to weight the expert predictions. It employs a null expert to collapse the model output to a prior if the input is different from the training data. The model is trained using a new mechanism called intrinsic cross-validation (ICV). ICV drops one or more experts during training, simulating out-of-distribution cases, which results in epistemic uncertainty quantification.

### Strengths
- The paper is well written and organized.
- The proposed method is novel.
- The proposed method is well motivated with connections to well established theory from Gaussian Processes and Bayesian methods.
- Uncertainty quantification is an important field for safety critical applications of deep learning.
- The empirical results are compelling, providing advancements on state-of-the-art in uncertainty quantification.

### Weaknesses
- The empirical results are limited:
   - Ablations on main components like ICV, null expert, different distance metrics are needed to understand the contributions of each.
   - The router has hyperparameters and it would be good to showcase the sensitivity of results to these parameters. Similarly, sensitivity on the number of experts is missing.
   - The initial “type” assessment mentioned in the appendix B1 seems random and might be subject to variability. It would be good to provide experiments showing how choices here affect the results.
- The experiments are conducted at small scales and for certain tasks. Applying the method to text domains and LLMs would strengthen the paper.

### Questions
I mention my main questions and concerns in the weakness section. Some minor points:
- Figure 2b. Most of the methods seems to coincide and it is hard to understand the individual performance of the methods. Is LEO’s CI’s for OOD include ensemble's CI for OOD? If so, I’d love to hear author’s interpretation of the superior performance of ensemble method for both ID and OOD.
- Table 4: 
   - I’d recommend adding the deterministic model’s performance numbers for comparison. 
   - Is it possible to provide a similar table for training requirements?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a single model approach to epistemic uncertainty estimation. The training samples are partitioned into disjoint sets, and during training each set is assigned to a specific expert (linear head); all experts share the same feature extractor. Some key steps are the inclusion of a null/prior expert, the training of a router module which learns the effect of expert exposure to unknown samples, and the overall two-phase training procedure denoted as intrinsic cross-validation (ICV). The system thus learns a supervised OoD signal from the training data alone, while remaining within a single-model framework.

### Strengths
- the ICV mechanism tweaks elegantly the mixture-of-experts paradigm in order to create OoD scenarios within the training set in a supervised manner (i.e., by increasing the probability of the null expert); this contrasts with methods like DUQ which rely on latent distance in an implicit manner

- low memory footprint and good inference speed

- the method is validated on three distinct UQ domains, with a very good performance on the BO experiment which matches/outperforms in high dimensions the GP

### Weaknesses
- the data partitioning strategy is badly justified, and seems arbitrary and plainly wrong with respect to the intended assumption : that the experts are specialists on *distinct* data partitions. The strategy passes samples across a freshly initialized, untrained data extractor and projects embeddings on a random vector. There is no semantic meaning whatsoever, this seems to me like a random vector projection of a random space projection, highly unlikely to create semantically coherent factors. The paper claims this creates a mismatch across experts' training distributions, but it is an arbitrary mismatch. The fact the method works well in these circumstances raises a critical question, namely is the method's strong performance a result of this very high variance partitioning setup, and if yes, why? Since it goes in my opinion against the distinct data partition assumption. Overall, this part is critical and is awfully under studied for the moment.

- an ablation focusing on the number of experts is necessary; the considered number (five) which matches the number of ensemble models / droupout samples is reasonable for comparison purposes, but it is in no way a methodologically sound justification. The study is crucial since the two pitfalls I can foresee are significant : too few experts and the OoD signal becomes weak, too many experts and the method might exhibit underfitting and poor generalization. 

- while convenient, the two step training introduces a limitation; the latent space is good for the expert loss (CE) but it is never trained to produce a latent space useful for the router's task which uses a L2 metric in the aforementioned space.  Why dismiss  end-to-end training completely? The justification that it is this "much faster" is of convenience, and should be underlined rather as a limitation, and justified better following a comparison with end-to-end training

- the claims of performance against SOTA are overstated and should be toned down (e.g. Figure2 Ensemble on ID NLL, ECE and DUQ/EDL on OoD NLL). LEO is good at both which places it conveniently on the Pareto front, but it is a trade-off, it does not blow away the competition.

### Questions
- justify formally the link between the data partitioning strategy and the claimed emergence of distinct data partitions 

- a characterization of the choice of the number of experts which would sustain the emergence of the OoD signal

- a justification for the choice of discarding the option of end-to-end training

### Soundness
2

### Presentation
2

### Contribution
3
