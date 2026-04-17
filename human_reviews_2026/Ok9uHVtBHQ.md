# BLIPs: Bayesian Learned Interatomic Potentials

- Decision: Reject
- Scores: 2, 4, 8, 0

## Abstract
Machine Learning Interatomic Potentials (MLIPs) are becoming a central tool in simulation-based chemistry. However, like most deep learning models, MLIPs struggle to make accurate predictions on out-of-distribution data or when trained in a data-scarce regime, both common scenarios in simulation-based chemistry. Moreover, MLIPs do not provide uncertainty estimates by construction, which are fundamental to guide active learning pipelines and to ensure the accuracy of simulation results compared to quantum calculations. To address this shortcoming, we propose BLIPs: Bayesian Learned Interatomic Potentials. BLIP is a scalable, architecture-agnostic variational Bayesian framework for training or fine-tuning MLIPs, built on an adaptive version of Variational Dropout. BLIP delivers well-calibrated uncertainty estimates and minimal computational overhead for energy and forces prediction at inference time, while integrating seamlessly with (equivariant) message-passing architectures. Empirical results on simulation-based computational chemistry tasks demonstrate improved predictive accuracy with respect to standard MLIPs, and trustworthy uncertainty estimates, especially in data-scarse or heavy out-of-distribution regimes. Moreover, fine-tuning pretrained MLIPs with BLIP yields consistent performance gains and calibrated uncertainties.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose a method for uncertainty quantification based on Bayesian inference that can be used with existing MLIP architectures. They evaluate the method on synthetic data, NH3, and a silica glass system. They evaluate UQ performance and accuracy on energy and force metrics.

### Strengths
- Uncertainty quantification is an important aspect not yet fully addressed by the MLIP community. 
- The proposed method is motivated well and grounded in existing work.
- In general, I agree with the authors that tackling UQ / generalization with MLIPs is an important area, especially since many important scientific applications deal explicitly with extrapolation.

### Weaknesses
While I think the method makes sense and is well presented, the evaluation makes it hard to justify the benefit of the proposed approach:
- The models considered are not the best models. Although the authors mention that the method is not currently amenable to new models like UMA, if the authors are claiming to evaluate generalization capability, then it is important to consider models explicitly trained with broad coverage. While the authors look at Orb, it is hard to evaluate how meaningful any E/F improvements are. The difference seems to go away as data increases and the relative performance benefit it quite small compared to the difference between the PaiNN model trained from scratch and the pre-trained Orb. The UQ performance seems to be worse here than an ensemble.
- In such data-limited regimes, it feels like a deep learning model is not the best choice. A simpler model like sGDML [1] often performs quite well with limited data. How does sGDML perform in terms of UQ and errors?     
- The authors only evaluate F/E regression metrics without evaluating downstream performance (like MD simulations).
- It would strengthen the paper to more clearly address the computational costs during both training and inference. The authors claim multiple times that these models are cheaper than ensembles but don't explicitly quantify the differences. If the cost of having multiple models is an issue, then different readout heads could be evaluated as an alternative. In addition, there are many other cheap ways of evaluating uncertainty, including training a smaller ML model [1]. Bayesian neural networks are also known to be hard to train at scale so it is hard to evaluate how generally applicable this method would be (or if the authors envision this as just a fine-tuning strategy).
- There are a number of existing works that discuss out-of-distribution performance and benchmarks for MLIPs and might be relevant for this work [2,3,4].  


[1] Chmiela, S., Sauceda, H., Poltavsky, I., Müller, K.R., & Tkatchenko, A. (2019). sGDML: Constructing accurate and data efficient molecular force fields using machine learning. Computer Physics Communications, 240, 38–45.

[2] Bowen Deng, Yunyeong Choi, Peichen Zhong, Janosh Riebesell, Shashwat Anand, Zhuohan Li, KyuJung Jun, Kristin A. Persson, & Gerbrand Ceder. (2024). Overcoming systematic softening in universal machine learning interatomic potentials by fine-tuning.

[3] Tobias Kreiman, & Aditi S. Krishnapriyan. (2025). Understanding and Mitigating Distribution Shifts For Machine Learning Force Fields.

[4] Chanussot, L., Das, A., Goyal, S., Lavril, T., Shuaibi, M., Riviere, M., Tran, K., Heras-Domingo, J., Ho, C., Hu, W., Palizhati, A., Sriram, A., Wood, B., Yoon, J., Parikh, D., Zitnick, C., & Ulissi, Z. (2021). Open Catalyst 2020 (OC20) Dataset and Community Challenges. ACS Catalysis, 11(10), 6059–6072.

### Questions
- What are the energy errors in Fig. 3?
- Why is performance worse with 128 samples vs. 32 samples in Fig. 3?
- What is the inference slowdown from using this type of model?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper describes how a particular formulation of variational inference can be applied to the setting of a machine learning interatomic potential. The method uses local reparametrization and involves input dependent variance scaling factors: the variance of each weight is specified as the squared mean value multiplied by an input dependent scale (adaptive variance term). It is discussed how the method retains equivariance (in distribution) when used with an equivariant GNN architecture. The method is demonstrated on a simple N-body system, a small ammonia dataset with training data near equilibrium an out of equilibrium test data, and finetuning a pretrained ORB model on a SiO2 dataset. The results show favorable peformance in comparison with deep ensembles.

### Strengths
The paper is well written and easy to follow.

The proposed modeling approach is reasonable and seems to work well. It is very general and can be applied to a long range of problems, also outside MLIPs.

### Weaknesses
Deep ensembles are commonly used for UQ in MLIPs, and there are several papers that focus on different ways to do this, however this literature is not cited sufficiently. In particular it would be beneficial with a discussion on how posthoc calibration is beneficial. 

All examples are in the small data regime and for the most part concerned with out of distribution uncertainty. Typically this means high sensitivity to hyperparameter choices. Often a slightly over-regularized model can trade in distribution performance for better out-of-distribution performance. However, in-distribution performance is not reported, and it is not described in sufficient detail how baselines including deep ensembles were tuned. 

The paper is positioned (and titled) as a Bayesian method, however when using a data dependent approximate posterior, the Bayesian interpretation is a stretch.

The discussion around equivariance and invariance in distribution in section 3.3 is very interesting and important in my opinion. I would have liked to see this extended to a more formal analysis, showing exactly when and how invariant coefficients (alpha and beta) is enough to guarantee equivariance in distribution for an equivariant base architecture. Furthermore, the argument that this does not hold for architectures based on irreducible representations of SO(3) would be very interesting to unfold more.

### Questions
I did not find a reference to Figure 1 in the text - is it missing?

If I understand correctly, eq. (8) is not actual objective used? There is a scaling factor on the KL term as described in A.1. This should be stated more clearly in the main text. 

Why is the prior in eq. (9) written in terms of it sampling procedure rather than directly as its distribution?

Eq. (9): I assume this variance matches the variance under weight dropout with the weights rescaled by 1/(1-p)? If I have understood correctly, consider writing this more clearly in the text.

What exactly were hyperparameters for the Ammonia data? Did you use the same hyperparameters for fitting the ensemble? If so, this would likely not be optimal.

How exactly are the ORB models without BLIP finetuned? This could be done in many different ways (full finetuning, last layer retraining, last layer finetuning, adapters, etc.) Are you certain a well chosen finetuning strategy would not be competitive?

In C.1 what exactly does this mean: "Dropout probabilities p are predicted and (...)"? I am missing some detail regarding this.

In C.3 it is mentioned how the KL scale and prior (dropout probability), but details are missing regarding how these were chosen. 

How exactly were the train/validation splits created? 

How exactly were validation data used for BLIP, for training ensembles and for finetuning?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a Bayesian framework for learning potential energy surfaces (PESs) via equivariant message passing neural networks. Compared to the non-Bayesian setting, this approach in principle allows for uncertainty quantification. It is also claimed that predictive accuracy is improved through the Bayesian approach. 

In contrast to previous approaches which rely on (compute and memory intensive) ensemble methods, the presented method only introduced a light computational overhead for a given model. It is also flexible and can be incorporated into existing neural network potentials. 

This seems like a valuable contribution to the field.

### Strengths
The paper significantly contributes to an important problem and provides a novel approach to Bayesian modeling in computational chemistry.

### Weaknesses
The paper claims that the Bayesian approach improves predictive accuracy for out of distribution configurations, specifically out of equilibrium geometries. For out of equilibrium geometries (in case they have multireference character) even DFT will not provide accurate energies or forces, which makes this a quite fundamental issue. I have a difficult time believing that a Bayesian model should overcome this and would appreciate a more detailed elaboration, as well as an explanation of what precisely causes the gain in accuracy.

### Questions
I would appreciate a more detailed explanation of Variational Adaptive Dropout (maybe in the Appendix).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The authors apply a variational Bayesian framework for machine learning interatomic potentials, used for molecular modeling and simulation. The goal of this framework is to better calibrate the model and quantify uncertainty when encountering new atomistic structures. The authors apply this method to training a model from scratch on ammonia data, and fine-tuning the Orbv3 model on a new dataset. The error is quantified based on the energy and force MAE.

### Strengths
- There is a need for MLIP uncertainty quantification methods, and the variational framework applied here has been less common for MLIPs.

### Weaknesses
A number of my concerns around the paper are understanding what utility this approach has given the progress in the field. 

- The MLIP field now has models trained on large, broad datasets. The experiment of training a network from scratch feels unrealistic now, such as the first example the authors look at. They are also using an architecture (PaiNN) that is outdated, with many architectures having improved on this architecture. It’s not clear if this method can actually help the best models trained on large datasets, which is where the utility of the method would be.

- Error is only quantified by energy and force MAE, vs. other evaluation tasks. The BLIP error in Table 3 and Figure 3 is not very statistically significant to show that the method actually improves, even in the settings that the authors picked which don’t seem very relevant to where modern-day MLIPs are at now. The bar chart in Figure 3 is also highly misleading in the y-axis units.

- The models used and comparisons are also not the best ones. It is unclear if the method can actually make the best models better.

- This paper generally needs a lot of work, and it is very unclear if this method would actually be useful to use for MLIPs. This includes using the best models trained on a lot of data, more rigorously exploring out-of-distribution cases, and more rigorously quantifying the utility of the uncertainty estimates and performance improvements on downstream tasks. An example place to start would be models trained on the Open Molecules 2025 dataset.

### Questions
- What does it look like if the method is used with an MLIP trained on larger datasets, such as eSEN or UMA? 

- Building on the above, can you show examples where the uncertainty quantification provided by this method is noticeably beneficial in some way?

### Soundness
2

### Presentation
2

### Contribution
1
