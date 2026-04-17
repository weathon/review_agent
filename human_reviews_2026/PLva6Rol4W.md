# Increasing Information Extraction in Low-Signal Regimes via Multiple Instance Learning

- Decision: Reject
- Scores: 4, 2, 8

## Abstract
In this work, we introduce a new information-theoretic perspective on Multiple Instance Learning (MIL) for parameter estimation with i.i.d. data, and show that MIL can outperform single-instance learners in low-signal regimes. Prior work \citep{nachman_learning_2021} argued that per-instance methods are often sufficient, but this conclusion presumes enough per-instance signal to train near-optimal classifiers. We demonstrate that even state-of-the-art per-instance models can fail to reach optimal classifier performance in challenging low-signal regimes, whereas MIL can mitigate this sub-optimality. As a concrete application, we constrain Wilson coefficients of the Standard Model Effective Field Theory (SMEFT) using kinematic information from subatomic particle collision events at the Large Hadron Collider (LHC). In experiments, we observe that under specific modeling and weak signal conditions, pooling instances can increase the effective Fisher information compared to single-instance approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose using a Multiple Instance Learning framework to improve parameter estimation in scientific analyses in low signal regimes. They provide an information-theoretic motivation, arguing that by aggregating multiple independent instances into a 'bag', the SNR of the learning task is effectively increased. This does allow a machine learning model to extract more Fisher Information than it could from processing each instance individually. The method is demonstrated by applying it to a high-energy physics problem.

### Strengths
The paper is well-written and the information-theoretic motivation is sound and clearly described in the Reviewers' opinion. The empirical demonstration of performance degradation in single-instance models versus the MIL approach in the presence of background contamination is also a well-executed and easy to follow experiment. 
Additionally also the discovery that models violate the second Bartlett identity and the suggested fix are interesting results by itself.

### Weaknesses
- The paper focuses on a single application only. While the application problem seems important, given that the authors introduce a general framework, the Reviewer would expect at least one other examples to show the applicability of the approach. Especially as the authors claim a general-purpose framework, this should also be reflected in the experiment section.
- The technical contribution of the paper seems to be limited, the core architecture is a simple MLP, and the multiple instance learning aggregation is a standard global average pooling of embeddings. There are currently no novel architectures, loss functions or training procedures proposed although the authors mention this as an important next step. From the Reviewers' perspective, the paper seems to be an exploration of a known model limitation rather than the introduction of a novel learning paradigm.

### Questions
See weaknesses and:
-The multi-class results were achieved by creating a large ensemble of 20 independently trained models. What would happen for a lower number of models and does this suggest a reduced robustness of a single model ?

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
This paper discusses a hypothesis testing scenario, where evaluating the likelihood function is intractable and hence the standard likelihood ratio test (LRT) becomes infeasible. The paper considers an alternative approach, where a neural classifier is trained based on simulated data under different hypotheses. In particular, it considers a solution where multiple instances are assigned to one common label (called MIL). The paper examines this idea in particle physics, for detecting deviations from Standard Model in collision experiments. By experimentation on synthetic data, it is shown that the proposed method is superior to the combination of decisions on individual instances (ensemble methods). This observation is further theoretically justified by arguments involving Fisher information and Cramer-Rao bound.

### Strengths
I am not familiar with physics literature and hence cannot assess the significance of the paper within this field. From a general statistics perspective, especially in the context of data fusion, the contribution of the paper is a lightweight method, based on LRT, which fuses multiple instances at a feature level rather than at a decision level. Feature-level fusion is known to be superior to decision-level fusion, especially in low-SNR regimes, but is generally considered a complex task.

### Weaknesses
I wonder how novel or substantial contribution is. As already mentioned, the fact that feature-level fusion is superior to the decision level is well-known and intuitive. The use of NNs for estimating likelihood ratios is not entirely new and is extensively discussed in the context of neural ratio estimation (NRE). From the perspective of MIL, the paper considers a simplified scenario, which to me is a repetition of the
standard point estimation theory with multiple observations. A major part of the theoretical discussions, e.g. the vanishing of ML error and growth of FI with O(\sqrt{N}), can be found in multiple classical sources.

Another drawback of the suggested approach is that for deployment, it requires an ensemble of independent observations of similar size to the ones used for training. This can be a limitation in practice.

The presentation of the paper can also be improved. It is sometimes difficult to understand the motivation behind the concepts introduced. For example, I am not familiar with the notion of effective Fisher information, and it is not clear to me what it implies. Some notations remain unexplained too. For example, in line 198 e_ij seems to refer to the elements of e_i, but this is not defined. Moreover, \theta_SM and \theta_SMEFT are not properly introduced.

### Questions
After understanding the problem of interest, I am surprised about the use of LRT, in this context. The reason is that one of the alternatives is presented as a composite hypothesis (theta\neq 0). A consequence of applying LRT is that the alternative values of theta (e.g. \theta_1) must be selected beforehand. How can this be done? And are not tests such as GLRT more suitable for this scenario?

If I understand it correctly, the training procedure does not explicitly bias the logits toward the individual likelihood ratios. Is it possible to guarantee that they are estimates of LRs? Indeed, Appendix C shows that they are biased in nature. And if biased, how is the presented theory based on CRB relevant to them?

As a minor comment in (11), do you mean by “+ 0” a higher order term?

### Soundness
3

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
4

### Summary
This paper demonstrates and theoretically justifies that multiple instance learning (MIL) for parameter estimation with i.i.d. data can outperform single instance learning (SIL) at low signal-to-noise ratios. Prior work demonstrated that single-instance learning is sufficient, assuming the training is optimal, but for low signal-to-noise, this may not be satisfied. They demonstrate the effectiveness of MIL in constraining Wilson coefficients of the standard model effective field theory (SMEFT) using kinematic information from subatomic particle collision events at the CERN LHC, observing that, for low signal-to-noise ratios, pooling instances can increase the effective Fisher information compared to single-instance approaches.

### Strengths
* Investigation of multiple instance learning in a new setting (low signal-to-noise ratios), which is of practical importance at the CERN LHC. 
* Theoretical justification using effective Fisher information to explain why MIL practically improves on SIL for low SNRs.
* Comparison to multiple baselines, including parameterized neural networks, and in multiple settings, including binary and multi-class classification.
* Code is made public.

### Weaknesses
* Unclear if the data has been or will be made public.
* Some details on the training procedures are missing, e.g. how large is the training data set? How many epochs was each algorithm trained for? Were training hyperparameters, e.g., learning rate, optimized? Etc. Since part of the claims depend on (non)optimality of the models, these are important considerations.

### Questions
* Was min/max pooling studied in addition to the average pooling of rht embedding vectors in a given bag? Min/max pooling seems like a more suitable choice if the goal is to classify if there is *any* signal instance in the bag.
* Can the studied datasets be made public?
* Could you define the effective Fisher information or clarify how/why it differs?
* Fig. 1: How large is the training data set? Are all of these models trained with the same size data set? How many epochs was each algorithm trained for? Were training hyperparameters, e.g., learning rate, optimized?
* Would the single-instance learning eventually “catch up” to the multiple instance learning if provided a large enough dataset even in the low SNR regime? If so, then another way to cast these results are that MIL is more data efficient.
* Fig. 2: Fix typo “Ensamble”

### Soundness
3

### Presentation
3

### Contribution
3
