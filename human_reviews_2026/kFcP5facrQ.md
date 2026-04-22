# Charting the Frontier: How Optimizing Performance Yields Accurate Scaling Laws on a Shoestring

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Predicting model performance at larger scales enables the design of training strategies and architectures tailored to specific performance targets. Empirical scaling law research identifies functional forms to aid this prediction task. These describe the relationship between loss and compute using a loss-compute frontier defined by learning curves. Due to the empirical nature of this approach, the computational burden is substantial, making strategic resource allocation essential -- yet it remains surprisingly underexplored. In this work, we address this shortcoming by exploring the suitability of Successive Halving (SH) and SH combined with parametric and non-parametric surrogate models. In addition to enabling a more systematic allocation of a given compute budget, our findings show that SH paired with surrogate models yields a set of learning curves that includes one with a lower loss-compute value than what naive uniform allocation or an SH-only approach can obtain. Our experiments demonstrate mean relative improvements of up to $2.84$% and $5.47$% on real-world and synthetic learning curve datasets. This strategic resource allocation enables us to obtain accurate scaling laws at significantly reduced computational costs, saving up to $98.7$% over the traditional exhaustive approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an algorithm inspired by the bandit literature for optimizing compute allocation to fit a scaling law. Experiments show the possibility of extrapolation of scaling laws using less compute than a “uniform allocation” baseline.

### Strengths
1. It is a nice idea to leverage ideas from the bandit literature to optimize compute allocation while fitting scaling laws. 
2. The paper provides good evidence that their particular surrogate decisions work well within the framework of evaluation that they set up. 
3. The added benefit of uncertainty estimates is a nice side benefit.

### Weaknesses
1. The clarity of the paper not good. From figure 1, the curves are not properly labeled. Most of the algorithm details are not clear from the main text, the reader really needs to do a lot of work to figure out what is going on. Line 5 of the main algo is not even properly defined. In general, setting up the bandit problem that motivates the paper is really hard to grok from a quick read.
2. The optimization problem in equation 2 does not make sense as a proxy since the target problem in Eq 1 is maximizing for a full set of compute budgets, but the proxy only selects a single model. This then becomes a top-k in the algorithm, but the motivation here is very unclear.
3. The whole premise of the goal of the scaling law seems to be wrong. In general, the point of fitting a scaling law is to predict what happens when we do a final run with a much larger compute budget. We don’t care about the “goodness of fit” of the curve on lesser compute points per se and we also don’t care about trying to find the minimal loss from a set of models (it will be the largest one). The goals are twofold: (1) predict what will happen when we do a run with C_large compute so we know the ROI of a larger run, and (2) compare two different methods A and B at small scale but extrapolate which one would be better at C_large in the final run. This paper does not seem to attack either goal. 
4. The scaling law that is fit throughout the paper has no irreducible error term. This implicit assumption that infinite compute would give 0 loss seems that it would cause issues scaling the method up as is. 
5. The main “uniform allocation” baseline is a bit unclear and seems that it could be a sort of straw man. Papers like Hoffman et al. are not trying to do efficient prediction of the target loss, they are trying to present the scientific finding of the robust scaling law, so using their methodology exactly as a baseline for efficiency seems to miss the point a bit.

### Questions
1. What is hat L in line 5 of algorithm 1?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates using successive halving (SH) combined with surrogate models to efficiently estimate the compute-optimal scaling laws in training language models. On both synthetic and real experiments, the proposed method outperforms naive uniform allocation and SH alone.

### Strengths
- There is very little published work on efficiently and accurately fitting the scaling law, despite its various challenges such as needing to train models to the compute-optimal frontier while minimizing the overall experiment cost. This work thus touches on an important, open problem.
- The combination of synthetic and real experiments provides good emprical validation to the proposed method.

### Weaknesses
- The noise models do not appear well-motivated by theoretical or empirical study on learning curves [1,2,3], which can provide more sensible models for the correlation than Brownian motion, for example.
- There appear to be no details on how hyperparameters such as learning rate, initialization, and weight decay is optimized. These hyperparameters are crucial for producing reliable scaling laws, and there are principled approaches such as [4] on how their optimal values scale with compute that do not seem to have been adopted in this work.
- The Chinchilla scaling law [5] shows that to reach the compute-optimal frontier, 20 tokens should be trained on per parameter. This means the 1.5B model used in this paper shoud be trained for approximately 3e20 FLOPs, 3x the maximum compute shown in Figure 2. Yet the authors are able to identify the compute-optimal frontier at this scale, which suggests something is off with the experiment.
- Based on the shape of the learning curves, I'm guessing no learning schedule or weight decay is used. Recent work [6] has shown that when using a learning schedule or weight decay, performance rankings can vary dramatically from early to late times and can be hard to predict due to non-intuitive learning curve shapes, potentially significantly reducing the effectiveness of the proposed approach.

[1] Bordelon et al., A Dynamical Model of Neural Scaling Laws

[2] Paquette et al., 4+3 Phases of Compute-Optimal Neural Scaling Laws

[3] Qiu et al., Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks

[4] Yang et al., Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer

[5] Hoffmann et al., Training Compute-Optimal Large Language Models

### Questions
- Does the proposed method work when following best training practice, such as using a cosine or WSD learning rate and weight decay?

### Soundness
3

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
3

### Summary
This paper presents a cost-efficient framework for estimating neural scaling laws under computational constraints. The authors adapt the Successive Halving (SH) algorithm to allocate compute across model families and further enhance it with surrogate models that predict future learning-curve trajectories. Empirical evaluations on both synthetic and real-world nanoGPT datasets demonstrate up to 98.7% compute savings while maintaining accurate scaling-law estimation.

### Strengths
1. This paper tackles an important problem: efficient estimation of scaling laws under limited compute budgets, directly relevant to LLM development and deployment.

2. This paper has demonstrated remarkable computational efficiency, reducing compute usage by up to 98.7%.

3. This paper shows consistent performance across synthetic and real-world datasets.

### Weaknesses
1. The method’s performance strongly depends on the surrogate model’s fidelity in capturing learning-curve dynamics; Deep Ensembles in particular show instability on real data.

2. The framework optimizes only validation loss, without considering multi-objective trade-offs such as downstream generalization.

3. Experiments are limited to nanoGPT, leaving scalability to larger LLMs unverified.

### Questions
1. Could the proposed framework be extended to incorporate additional objectives (e.g., generalization or transfer performance) to better align with real-world model training considerations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a resource-efficient approach to estimating compute-loss scaling laws by applying Successive Halving (SH) with and without surrogate models (Gaussian Processes and Deep Ensembles). The key premise is that traditional methods for deriving scaling laws (e.g., Kaplan et al., 2020) are computationally expensive, requiring full training of many models. The authors reformulate the problem as one of optimal compute allocation across a set of candidate models, then use SH to progressively prune models during training. They further enhance SH by predicting future learning curves using surrogate models conditioned on partial data, allowing for more informed resource allocation.

### Strengths
Originality: Reframes scaling law estimation as a resource allocation problem and bridges hyperparameter optimization with scaling law research. Using SH and learning curve surrogates in this context is novel to my knowledge.

Clarity and quality: Paper is well-structured with visualizations and thorough appendices. The evaluation is done in multiple different settings, including synthetic experiments and nanoGPT data. Surrogates are well-motivated and tested across tasks.

Significance: Large computational savings for establishing scaling laws.

### Weaknesses
The two main discussion points to me seem 1) complexity of the approach and 2) unclear effect on downstream accuracy of the scaling law. 

For 1), the training of multiple models across scales is already a complex endeavor, requiring strategies of how to distribute many parallel training runs across a cluster, budget allocation, checkpointing, learning rate schedules, etc. I am skeptical of the overall introduced complexity of the SH + surrogate approach as a blocker for adoption; e.g., continuously stopping and restarting real model runs in an online fashion. 

For 2), to me, Table 3 doesn't clearly establish whether the learned scaling laws (under budgeted evaluation) are actually useful for predicting downstream model performance beyond the observed region. In a way, the predictive quality of a scaling law is what matters most for practitioners -- I would rather spend more on running small model ablations to arrive at a precise fit for my future large runs, than save compute with SH to get a suboptimal fit. I will rephrase in the question section below.

### Questions
The AbC metric in Table 3 only measures fit error against fully observed curves. How well do the scaling laws—fit using SH+surrogate under a limited budget—predict performance for unseen (larger) models or higher compute budgets? In other words, have you evaluated the extrapolation accuracy of the learned scaling law, or compared it to a baseline scaling law fit on the fully trained models?

Misc.: You simply say you use default hyperparameters for nanoGPT. Could you clarify the exact details, including batch size, learning rate schedule, total tokens, ...etc? Especially the LR schedule will have a strong influence on scaling laws (cf the original Chinchilla observation of the LR decay effect compared to Kaplan), even more so when you perform the model selection in the online fashion to stop and continue training.

### Soundness
3

### Presentation
3

### Contribution
2
