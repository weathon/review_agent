# ASIDE: Adaptive and Separable Interventional Dynamics  via Progressive Meta-Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
To decide how to change the future trajectories of a dynamics system, it is important to predict not only the intrinsic dynamics of the system but also its response to external interventions. 
While notable progress has been made in learning intervention effects over time, existing research has prioritized the challenge of time-varying confounding in observational data. 
Significant challenges however remain in aspects related to the modeling and inference of latent dynamics. 
A first and foremost challenge lies in the need to separate, from a composite observation, the natural temporal evolution of intrinsic dynamics from its response to external interventions. 
This challenge is further exacerbated by the need to integrate rich history information into these latent dynamics. In this paper, we present a novel framework of adaptive and separable interventional dynamics (ASIDE) to overcome these challenges. 
First, we decompose the latent dynamics into separate components of intrinsic dynamics and its responses to external interventions at the latent space. This is in contrast to existing approaches that model and infer the composite dynamics as a black box. 
Second, we leverage meta-learning to enable these components to separately adapt to their relevant context examples in past history, addressing both inter- and intra-subject variabilities. 
This is in contrast to existing approaches that use history only to initialize a \textit{one-size-fit-all} forecasting function. 
On synthetic and real benchmarks, we demonstrate the advantage of ASIDE in improving forecasting accuracy for both intrinsic and interventional dynamics, in settings with or without time-varying confounding.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses time‑series forecasting under external interventions. It proposes ASIDE, which decomposes predictions into baseline (intrinsic) and response (intervention‑driven) components implemented by separate networks whose outputs are combined additively. Each network takes a context embedding of the history as input, and the model is optimized for future predictive accuracy. Training proceeds in three stages: (i) intrinsic dynamics only, (ii) intervention dynamics only, and (iii) joint training. Evaluation is conducted on a synthetic dataset aligned with the method’s assumptions and on the MIMIC dataset.

### Strengths
* The paper targets an important problem in the healthcare domain: estimating treatment effects in medical time-series.
* On the synthetic experiment, the method substantially outperforms the baselines, however, the scenario is not realistic (see Weaknesses).

### Weaknesses
The paper should be rejected due to (i) limited novelty, (ii) unclear conceptual differences from prior work, (iii) focus on a special‑case problem with no clear real‑world use case, and (iv) issues with the empirical evaluation.

**(i) Limited Novelty**

The method comprises: (a) an additive decomposition with separate networks for baseline and response dynamics; (b) conditioning of these networks on history via a context embedding, and (c) a three‑stage training procedure. Here, components (a) and (b) are not novel ideas, which limits the originality of the contribution.

**(ii) Conceptual differences with prior work**

The paper does not discuss prior work relevant to (c) the three‑stage adaptive training, making it impossible to assess novelty. 

In addition, several claims reference prior literature without citations, e.g.:
* Ln. 97-98: All existing intervention-effect models … - no reference.
* Ln. 99-101: … typically achieved in a two-stage encoding-decoding framework… - no reference
* Ln. 184-185: While different adaptation mechanisms exist, … - no reference
* Ln. 186-187: … differ from existing works … - no reference
* Ln. 192-194: …, existing works mostly focus on … - no reference

**(iii) Special‑case problem with no clear real‑world connection**

The paper models intervention effects without addressing time‑varying confounding, thereby tackling a special case of counterfactual outcome estimation studied in prior work [Bica+20; Melnychuk+22]. In real medical time‑series, interventions are typically assigned based on patient history. Additionally, the manuscript does not connect its restricted setting to a concrete real‑world use case.

**(iv) Issues with empirical results**

Because the method only estimates treatment effects in the absence of time‑varying confounding, the synthetic experiment uses randomly assigned treatments. This setup is unrealistic and limits the significance of the observed gains. In the real‑world MIMIC experiment, the method does not show substantial improvements over the main baseline, the causal transformer (CT) [Melnychuk+22].

**Minor comments**
* Figures and captions are not yet polished and could be improved.
* Several repeatedly used phrases are unclear and should be defined at first use:
  * “one-size-fit-all” forecasting function
  * leverage inductive bias

### Questions
There are several points missing in the main text:

* The training prediction horizon for the proposed method and each baseline is not reported. Please clarify the horizon used for each method in each experiment.
* Parameter counts for ASIDE and for all baselines are not provided. Please report them.
* What is the training‑time overhead of the three‑stage procedure relative to single‑stage training? Is it $3 \times$?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ASIDE (Adaptive and Separable Interventional Dynamics), a framework that disentangles latent intrinsic dynamics from interventional responses in time-series data. The method leverages (1) explicit decomposition of latent ODE components into intrinsic (with own dynamic) and intervention-dependent (influenced by action) (2) a progressive meta-learning scheme that first learns intrinsic dynamics from intervention-free segments, then adapts interventional components via counterfactual generation. Context embeddings extracted from past history enable adaptation to inter- and intra-subject heterogeneity. Experiments on synthetic and real world dataset show improved forecasting accuracy over RMSN, CRN, and CT baselines.

### Strengths
1. The explicit separation of intrinsic and interventional dynamics is intuitive and provides interpretability benefits over existing black-box time-series models.

2. The two staged optimization that first learns intrinsic and then interventional dynamics is a neat idea that mitigates entanglement issues common in neural ODE frameworks. (although not sure why this is framed as meta- learning but okay)

3. Both synthetic and real-world data are evaluated, with ablation studies that show each component’s contribution. In general, the empirical results are fare.

### Weaknesses
1. The paper is dense and difficult to follow. Key intuitions behind the meta-learning and counterfactual embedding extraction could be explained more clearly, intuitively and visually. Can you try to add more explanations?

2. While the work uses causal terminology and very much close to causal literature, it does not formally define or justify causal assumptions (especially around identifiability). Can you provide some justification or insight here?

3. Maybe some baselines are missing, e.g. causal time series models, or recent counterfactual CDE or transformer-ODE baselines. Maybe there are some other baselines to compare to?

4. Not sure how much novelty this idea have, decomposable models for time series separate default condition and conditions with intervention is not new. The literature should not just limited to neural ODE?

### Questions
Please see my weakness section.

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
This paper proposes a new framework, ASIDE, for modeling time-series data under external interventions. ASIDE is designed to explicitly separate the intrinsic system dynamics and responses to external interventions at the latent space. To address inter- and intra-subject variability, ASIDE leverages a meta-learning approach, extracting context embeddings from historical data to adapt both the intrinsic and intervention dynamics. This design allows the model to handle heterogeneity across and within subjects, thereby improving forecasting accuracy, especially over longer prediction horizons and with increasing heterogeneity. The framework is evaluated on both synthetic and real-world datasets to demonstrate its effectiveness.

### Strengths
1. The motivations for disentangling and adapting the intrinsic and intervention-driven dynamics in time-series data are clearly described, and corresponding solutions are proposed.
2. Experiments on both synthetic and real-world datasets, including ablation studies, are conducted to demonstrate ASIDE’s advantages.
3. The paper is generally well-written and easy to follow.

### Weaknesses
1. The baselines are too few (only three) and outdated (the newest is from 2022). The authors should consider more advanced baselines. In addition, classical non-neural-network-based models are lacking.
2. The proposed model appears to require more computational complexity due to its more complicated structure and training process. A complexity comparison should be provided and discussed.
3. While the model is claimed to be more interpretable, the paper does not provide qualitative or quantitative analyses of interpretability benefits.

### Questions
Same as the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a framework called Adaptive and Separable Interventional Dynamics (ASIDE) to address challenges in modeling dynamic systems and time series forecasting. Using both synthetic and real-world benchmarks, they demonstrate that ASIDE improves forecasting accuracy for both intrinsic and interventional dynamics, under settings with or without time-varying confounders.

### Strengths
The paper is well written, and the proposed method is conceptually clear and easy to follow.
The authors conduct experiments on both synthetic and real-world datasets, providing empirical evidence for the effectiveness of ASIDE.

### Weaknesses
Several concerns should be addressed to strengthen the paper:

a. Outdated baselines:
Most baseline methods used for comparison are from before 2022. More recent baselines, such as [1] and [2], should be included to better demonstrate the advantages of ASIDE.

b. Lack of theoretical justification:
The theoretical support for the method is insufficient. The authors should provide a more rigorous theoretical analysis or justification to explain why ASIDE can outperform existing methods.

c. Limited datasets:
Only two datasets are used in the experiments. Additional datasets, especially those focused on out-of-distribution (OOD) forecasting, should be included to provide a more comprehensive evaluation.

d. Minor contribution of meta-learning:
The discussion of meta-learning in the paper appears somewhat trivial and should not be emphasized as a major contribution. This component represents a natural approach for representation learning in time series forecasting rather than a novel idea.

References

[1] Liu, Haoxin, et al. "Time-series forecasting for out-of-distribution generalization using invariant learning." arXiv preprint arXiv:2406.09130 (2024).

[2] Wang, Yuxuan, et al. "Timexer: Empowering transformers for time series forecasting with exogenous variables." Advances in Neural Information Processing Systems 37 (2024): 469–498.

### Questions
Please refer to the weaknesses above. Addressing these issues would significantly strengthen the paper and clarify the contribution of the proposed method.

### Soundness
3

### Presentation
3

### Contribution
2
