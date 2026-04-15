# FedEve: On Bridging the Client Drift and Period Drift for Cross-device Federated Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3

## Abstract
Federated learning (FL) is a machine learning paradigm that allows multiple clients to collaboratively train a shared model without exposing their private data. Data heterogeneity is a fundamental challenge in FL, which can result in poor convergence and performance degradation. \textit{Client drift} has been recognized as one of the factors contributing to this issue resulting from the multiple local updates in \fedavg. However, in cross-device FL, a different form of drift arises due to the partial client participation, but it has not been studied well. This drift, we referred as \textit{period drift},  occurs as participating clients at each communication round may exhibit distinct data distribution that deviates from that of all clients. It could be more harmful than client drift since the optimization objective shifts with every round. 
In this paper, we investigate the interaction between period drift and client drift, finding that period drift can have a particularly detrimental effect on cross-device FL as the degree of data heterogeneity increases. To tackle these issues, we propose a predict-observe framework and present an instantiated method, \fedeve, where these two types of drift can counteract each other to mitigate their overall impact. We provide theoretical evidence that our approach can reduce the variance of model updates. Extensive experiments demonstrate that our method outperforms alternatives on non-iid data in cross-device settings.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work investigates the decoupling of classic client drift which is widely studied in previous works, and proposed "period drift" which is less explored. Authors propose a simple method FedEVE based on their predict-observe framework, and extensive experiments support their proposed method's performance.

### Strengths
- This work is well organized and written.
- This work proposes a very simple and effective method based on the Bayesian filter (or Kalman filter). The experimental results support their claim.
- The proposed "period drift" concept is good for federatede learning community to  further study. Authors are encouraged to open-source their source codes for FedEvE and other compared methods which helps to broaden the influence of this work.
- Personally I like the analysis of Kalman Gain a lot : )
- One less studied area discusses how to perform FL under noisy labels [1], future studies can explore this area with the light of  authors' proposed framework.

[1] Jiang X, Sun S, Wang Y, et al. Towards federated learning against noisy labels via local self-regularization[C]//Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022: 862-873.

### Weaknesses
- Since this work is tightly related to the client selection, so the random seeds to conduct experiments on their proposed method and baseline methods should be given to increase the reproducibility.
- The total client number for other datasets (CIFAR10/100) seems not given.

### Questions
See above.

### Soundness
3 good

### Presentation
3 good

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
This paper proposes the period drift, which means that, participating clients at each communication round may exhibit distinct data distribution. Authors claim that it could be more harmful than client drift since the optimization objective shifts with every round. To this end, this paper investigates the interaction between period drift and client drift, finding that period drift can have a particularly detrimental effect on cross-device FL as the degree of data heterogeneity increases. Then, a predict-observe framework and an instantiated method, FEDEVE is proposed, where these two types of drift can compensate each other to mitigate their overall impact.

### Strengths
1. Using Bayesian filter to compensate two sources of drift is novel.
2. The connection between server momentum and Kalman Filter is interesting.
3. The paper is written clearly.

### Weaknesses
1. The so called ``period drift'' comes from the stochastic sampling of clients. If we see sampling clients as sampling data in SGD, such a period drift also happens during SGD -- each batch of data has distinct data distribution from other batches. Authors should provide a more rigorous definition of period drift and show that how the period drift harms training.
2. The Figure 3 shows the period drift that the sampled data on one client varies across different rounds. This may still be similar to [1], as indicated in related work. Moreover, to address this varying effect, the clients can traverse the whole local dataset using sampling without replacement. 
3. Experiment result show little improvements than baselines.

[1] Diurnal or nocturnal? federated learning of multi-branch networks from periodically shifting distributions.

### Questions
1. See weakness 2. When clients traverse the whole local dataset using sampling without replacement, does the period drift still happen?
2. As shown in Figure 3, how fedavg_perod_drift_only is drawn? Specifically, how to guarantee that only period drift happens, but client drift not happens?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the effect of various "drifts" in Federated learning settings. In particular, the paper focuses on period drift, which arises due to partial participation of clients in FL settings. The paper proposes a predict-observe framework and provides an instantiation of the framework, FedEve, to handle these drifts. Experiments are provided to demonstrate the effectiveness of these approaches. While the paper has interesting elements, I have the following primary concerns:

(1) The so-called "period drift" arises in almost all stochastic optimization methods. Of course, this could be severe in FL settings due to higher data heterogeneity but the presentation of the paper is misleading since it is presented as if it is a new concept.

(2) Missing mathematical rigor: It felt like the paper was missing mathematical rigor. For instance, period drift was not defined in the whole paper. The exact definition of it is missing. Furthermore, at places, terms were introduced without proper mathematical definition (e.g. w_server in Assumption 3.1).

(3) Assumptions in the paper are very strong. While the authors tried to provide some vague justification, this does not represent any realistic scenario.  Assumption 3.2 especially looks very strong and I do not believe it happens in practice. Are there any empirical evidence provided to support these Assumptions (which I may have missed)?

(4) The empirical analysis looks fairly weak. The improvement on most datasets seems somewhat small and experiments do not provide any justification for the assumptions made in the paper.

Overall, while the paper has interesting elements, I believe there are severe shortcomings need to be addressed before publication.

### Strengths
Refer to summary

### Weaknesses
Refer to summary

### Questions
Refer to summary

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
