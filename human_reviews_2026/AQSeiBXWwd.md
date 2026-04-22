# Diverging Flows: Detecting Out-of-Distribution Inputs in Conditional Generation

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Flow matching models are able to learn complex conditional distributions from data. Nevertheless, they do not model the distribution of the conditioning itself, which means they can confidently generate samples from conditioning inputs that are not in the training distribution. In this work, we introduce _Diverging Flows_, an approach to train flow matching models that enables a single model to detect OOD conditions, without hindering its generative capabilities. _Diverging Flows_ augments standard flow matching training with a contrastive objective that learns to separate the velocity fields produced by in- and out-of-distribution conditions, effectively modeling the conditions' distribution, and practically enforcing an effective telltale sign during the generation process. At inference time, we combine this signal with conformal prediction to obtain statistically valid OOD decisions. Additionally, _Diverging Flows_ does not require real OOD data, enabling fully self-contained training on the target domain. The results indicate that _Diverging Flows_ is competitive with other OOD detection methods while preserving the predictive quality of the underlying flow model. Ultimately, these results pave the way in adopting generative models as safe and robust predictors in high-stakes domains like weather forecasting, robotics, and medical applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Diverging Flows, a flow-matching approach for conditional generation that can detect OOD conditions. The method augments flow matching with a contrastive objective, ensuring that predicted velocity fields for ID and OOD conditions are separated during training. Experiments include a 2D toy task, image reconstruction OOD benchmarks, and a weather-forecasting setting. Results suggest high OOD detection performance while preserving or slightly improving reconstruction quality.

### Strengths
S1. The direction of calibrating uncertainty in generative models is important for making conditional generation safer and more reliable.

S2. The technical components are clearly described and easy to follow.

S3. The toy example in Section 5.1 clarifies the core concept and how divergence manifests along trajectories.

S4. The paper candidly reports limitations of the proposed approach.

### Weaknesses
W1. The problem setting feels somewhat removed from common practical goals. In many applications, diffusion or flow models are expected to generalize to novel conditions like "an astronaut riding a horse on mars"; this work instead trains a generator that intentionally fails under OOD conditions to flag them. The paper would benefit from concrete scenarios where this behavior is clearly advantageous.

W2. Prior work, DiffPath, has already used diffusion or flow trajectories for OOD detection. This paper can be read as an extension of that idea to flow matching with conditional generation; the incremental novelty feels modest.

W3. Some comparisons appear potentially unfair. In Table 1, the DiffPath baseline uses a single model trained on CelebA for RGB experiments, while the proposed method is trained separately on each dataset.

### Questions
Q1. Why does the triplet-style component in your loss also improve reconstruction quality? An intuitive explanation would help.

Q2. Can you offer any theoretical guarantee or clear conditions under which the contrastive training should improve OOD detection without hurting generation?

Q3. The introduction motivates safety-critical settings such as weather forecasting and robotics. Under OOD inputs, however, the model is designed to produce diverging flows; what prevents these predictions from being harmful in practice, and how is risk actually mitigated at deployment time?

Q4. For Table 2, are reconstruction errors computed on a held-out test set that was not seen during training?

Q5. Compared with DiffPath, what are the pros and cons in terms of the practical usage (e.g. runtime)?

Q6. In Table 1, would it be possible to compare against DiffPath models trained individually on each dataset to ensure a fairer baseline?

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
This work proposed a new method based on (conditional) flow matching to detect out of distribution (OOD) data, using only in-distribution (ID) training data. The main idea is to use the variance of the vector field at multiple time steps (learned through a contrastive variant of flow matching) as a scoring function. Experiments on standard image benchmarks and a weather forecasting dataset seem to confirm the effectiveness of the proposed method.

### Strengths
- a new OOD detection algorithm based on (conditional) flow matching and contrastive learning

- experiment comparison against multiple diffusion based OOD detection baselines on standard image benchmarks

### Weaknesses
- unclear how the context for in-distribution data is generated, on both training and test sets. 

- problem setup is questionable: why insisting on training only on in-distribution data? Anyone who is serious about OOD should at least try to collect a small amount of OOD data, for otherwise we are doing ourselves a disservice and making the problem unnecessarily underdefined and challenging. I recognize many prior works followed the same setup and I am curious to hear the authors' reasons. For some theoretical discussions on this problem setup, see e.g. https://openreview.net/forum?id=sde_7ZzGXOE and https://proceedings.mlr.press/v139/zhang21g.html.

- heavier computational cost for both training and inference: for training, one needs to use a bigger network to account for the context, while for testing, multiple time steps need to be simulated and averaged. Can the authors comment on scalability and training and test time comparisons? Ideally, one should compare to the unconditional flow matching model on the amount of in-distribution data needed to achieve certain performance (e.g., the assumption on Line 052).

### Questions
My main concern is that the authors did not describe how the context c during training and testing is generated and how its choice (including dimension) affects the experimental results. This is a crucial missing piece that will affect my final evaluation. 

I also suggest running this ablation study: during testing, what happens if we disable the context (e.g., provide the same context for all test samples)? Can your method still achieve good performance? In other words, is the context actually doing the heavy lifting?

It is surprising that in Table 1, DF+FGSM achieved AUROC 1 on multiple datasets, implying that those pairs of datasets have no overlap of support at all. If this is the case, many existing generative methods (including those based on likelihood) should perform well too. To the contrary, FM likelihood performed very poorly. Do the authors have any explanation of this observation? How is FM likelihood trained and tested? 

In the second paragraph of Introduction, the authors made a big deal on quantifying uncertainty, but in the end the authors merely employed (split) conformal prediction to address this issue. This is a bit disappointing since the same (split) conformal prediction could literally be applied to any existing OOD detection method.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a framework which simultaneously generates conditional samples and also determine if the conditioning is out of distribution. They do this by creating Gaussian noise or adversarial perturbations of real samples and use a contrastive loss to push OOD conditioned samples away from in-distribution samples. At test time, the instability in the curve of the predicted velocity field grants a Flow Smoothness Score metric which, once surpassing a certain threshold which is obtained through conformal prediction, allows the model to flag whether the sample is in-distribution or OOD. They evaluate on a few standard image datasets which contrast against one another.

### Strengths
Strengths:
Their approach allows them to train a single model for OOD detection without specific OOD data needed within the training procedure.

They show surprisingly strong results in terms of AUROC (see weaknesses section for a caveat here).

Generation quality seems to not be impacted by the inclusion of the OOD detection algorithm.

### Weaknesses
Weaknesses:
The AUROC numbers reported are extremely close to 1 on all the benchmarks. This might be due to the datasets being completely different when trying to direct OOD samples, whereas in the real world the application may not be as clean. In particular, the MNIST vs KMNIST case shows a severe drop in performance which is irregular for a model which should otherwise be performing very well across everything else. This might indicate that the model is not actually robust to OOD detection as claimed.

Some ablation study results regarding the inclusion of the dual-conditioning portion vs the contrastive loss could be helpful in determining the specific contributions of the paper.

Datasets included are somewhat weak; the weather forecasting is done for single step only. Current weather forecasting experiments in particular focus on much longer time horizons and multi-step prediction.

The authors mention in the limitations that the training with the contrastive loss can be unstable, but fail to mention how heavy the impact of the instability is, which can make applying this model particularly hard.

### Questions
See weaknesses.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to detect out-of-distribtuons (OOD) conditions in a conditional generative model (here, in the flow matching framework).

The proposed method, called Diverging Flows, seeks to identify OOD conditionings without relying on the model’s output, but instead by computing scores directly from the velocity fields of the flow matching model.
It relies on *contrastive learning*, i.e. it adapts the standard flow matching loss by adding 2 regularization terms.
These two terms aim at pulling apart positive conditionings from negative ones. 

The paper suggests two strategies to build negative samples: either pure noise or adversarial samples. Thus, it does not rely on an extra OOD dataset.
The training objective encourages the model to produce similar instantaneous velocity fields for in-distribution (ID) conditionings and divergent velocity fields for OOD conditionings.

The OOD detection then relies on an ad-hoc score that measures the variation of the velocity field along the ODE trajectory: the more it varies, the less likely the conditioning is ID.

### Strengths
- The OOD detection problem is interesting.
- The presentation is clear, making the paper easy to follow.
- The method shows empirical effectiveness, and it is applied to real data (weather forecasting).

### Weaknesses
**On the OOD Score**
In Section 3.2, I think there is a confusion between:
- what happens during training where one regresses against the conditional velocity field $u^\mathrm{cond} = x_1 -x_0$ on interpolated $x_t$ 
- what happens during inference, where the goal is to follow the total velocity field which is defined as $u_t = \mathbb E[u^\mathrm{cond}  | x_t]$.

The sentence ''at each step the model should strive to follow this straight trajectory'' is confusing: at inference, the trajectories have no reason to be straight (see e.g. Multisample Flow Matching: Straightening Flows with Minibatch Couplings, Pooladian et al. or Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, Lie et al. for a discussion on straightness).
As the OOD score is based on a straightness hypothesis; this should be clarified in the paper.

**On conformal prediction** The authors use the term conformal prediction at several places in the paper to explain that their method can be used on ''safety-critical'' usages and that it is ''statistically valid''.
From my understanding, the threshold on the scores (that decides if the data is ID or OOD) is set as a quantile on a calibration set made of ID data.
I think this does not bring any guarantee on what happens for OOD data.
Besides, as it is presented as an important asset of the method, I think this part deserves more details both regarding the theory (do we have some guarantess? if yes, they should be described) and in practice (e.g. in the experiments, what is the size of the calibration set?).

**On generation performance** The claim that the modified loss “does not degrade the predictive performance of the underlying FM model” could be more deeply evaluated. For example, Table 2 reports a reconstruction error, but as this is a generative model, computing the FID is also an important metric to assess the quality of generated samples.

### Questions
1. In the experiment made Section 5.2, ID conditionings are exactly samples from the training data. 
As stated in the paper, the method is somehow underemployed in this setting (it's rather used as a first benchmark against other baselines). Yet, it would be interesting to add tasks such as image restoration or style transfer (where the degraded images / the styles are the conditionings), it would be a more relevant demonstration of the method's utility.

2. Does the proposed loss really fit into the standard contrastive learning framework ?
Usually, positive samples are generated by applying well-chosen transformations to true samples, while negative samples correspond to other samples from the train set. Here, there is no positive samples (just the baseline sample) and the negative samples are new conditionings. I think this distinction deserves some clarification.

3. For the weather forecasting experiments, are the same hyperparameters and architecture used as in the RGB image experiments? What are the sizes of the training, calibration, and test sets? This information should be provided in the appendix.

**Minor comments**

- The introduction to Flow Matching (FM) could be slightly clarified: $u_t^\mathrm{cond}= x_1-x_0$ is the conditional velocity field  (conditonned on $(x_0,x_1)$) and what we actually aim to learn is $u_t = \mathbb E[u^\mathrm{cond}  | x_t]$.

- On the beginning of section 3, introducing conditional generation. The fact that it is enough to train a velocity field $u_\theta(x, c)$ to generate $q(x|c)$ using the standard FM loss comes from the hypothesis made in Equation (4): the underlying assumption is that $p_t(x_t|x_1)$ is independant of $c$ conditionnaly on $x_1$, which leads to Equation (3). As it is now stated (with Eq (3) that appears before Eq (4)), it is not so clear.

### Soundness
2

### Presentation
3

### Contribution
2
