# Learning Continuous and Discrete Dynamics for Time Series Anomaly Detection via Probabilistic Modeling

- Decision: Reject
- Scores: 2, 8, 8

## Abstract
Anomaly detection for multivariate time series plays an important role in many applications, enabling, e.g., risk monitoring in cyber-physical systems. While existing methods achieve good results on continuous variates, they struggle when having to learn both continuous and discrete dynamics across continuous time. Further, existing methods simply sum up reconstruction or contrastive errors from each variate to obtain final anomaly scores without recognizing differences in importance of variates with different measurement units. To overcome these limitations, we propose TAD-UP that learns both continuous and discrete dynamics for Time series Anomaly Detection via Unified Probabilistic modeling. First, we propose two co-dependent branches of efficient neural ordinary differential equations with the compound Poisson process to learn both continuous and discrete dynamics for different variates. We also propose a gate mechanism to learn correlations among different dynamics. Second, we propose to model a joint probability distribution for anomaly detection. The resulting model is optimized using Maximum Likelihood Estimation on joint variates, instead of using reconstruction or contrastive losses on each variate. We detect anomalies using joint probabilities, which take the marginal probabilities of different variates into account. Experiments on nine real-world datasets from different domains offer evidence that TAD-UP is capable of state-of-the-art accuracy and better efficiency tradeoff.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a probabilistic framework for multivariate time-series anomaly detection that separately learns continuous and discrete dynamics through two co-dependent neural ODE branches. Experiments on nine datasets show competitive or state-of-the-art performance.

### Strengths
The idea of separately learning continuous and discrete dynamics and integrating them through co-dependent neural ODEs is interesting and offers a new perspective for handling heterogeneous multivariate time series.

### Weaknesses
1. Lack of design motivation. The paper introduces several components, continuous co-ODE, discrete co-ODE, and gated TCN, but does not sufficiently explain why these particular designs are chosen.
2. Limited novelty in model architecture. The proposed framework appears to be a direct combination of existing elements: standard embeddings for continuous/discrete variables, TCNs for temporal modeling, and ODE formulations. The overall contribution lies more in integration than in new methodological innovation.
3. Modeling of discrete dynamics. The discrete branch relies on a Poisson-process-based formulation (Eq. (5)), which inherently models binary “jump/no-jump” events. It is unclear how this approach generalizes to multi-valued discrete states, which are common in categorical time-series data.
4. Expressiveness of Gaussian mixture assumption. The joint probability of all variables is modeled by a (mixture) Gaussian distribution, whose representational capacity is limited for complex, high-dimensional time series.
5. Loss formulation ambiguity. In Eq. (11), the latter term of L_c seems to include  the L_{MLE} term already present in the first half, raising doubts about whether both parts are necessary or if redundancy exists.
6. Weak theoretical contribution. Theorems 1 and 2 in the Appendix are standard linear-algebra results (positive semi-definiteness of covariance matrices) and do not provide theoretical insight specific to the proposed model.
7. Unclear inference mechanism. The paper emphasizes that marginal probabilities help identify important variables in anomaly scoring, but the inference section does not explain how this importance is actually utilized or interpreted during detection.
8. Missing baselines. The experimental comparison omits recent diffusion-based methods such as ImDiffusion, which are relevant and often strong baselines for time-series anomaly detection.

### Questions
See the weaknesses section.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a method that learns both continuous and discrete dynamics for time-series anomaly detection and uses these dynamics to estimate correlations among continuous and discrete variables. Results across nine benchmarks are promising.

### Strengths
This is a well-written document, and the proposed scheme appears sound. The authors report results on nine benchmarks and compare against 21 baselines.

The paper employs Neural Ordinary Differential Equations (NODEs) to learn temporal dynamics. Prior work (e.g., Jin et al., 2023) has shown that NODEs can model continuous dynamics in continuous time. This paper further demonstrates how NODEs can model dynamics for discrete events and proposes a way to learn correlations among variables.

The ablation study is well designed.

The supplementary material includes important details that help one understand the work carried out in this document.

### Weaknesses
It is not clear how the trained model is used at inference time. Section 3 frames the task as Time Series Anomaly Detection, but it is silent on how the learned models—which output correlations between continuous and discrete variables—are applied to detect anomalies within a given window. A brief, explicit description of the inference procedure would help.

Relatedly, does the method require an anomaly threshold? If not, why is it unnecessary? If it does, how is the threshold chosen?

The paper will benefit from a limitations sections.  Are there any?

### Questions
I can see how one might construct dense features at each time step t_i for continuous variables. How is this done for discrete variables? (Ln. 224)

I did not fully understand Eq. (4). How does the gating work? Is there no sum-to-one constraint? The same question applies to Eq. (7). Also, in Eq. (7), what happened to \lambda(t)?

In Section 4.5, why is it not appropriate to model correlation among discrete variables with a Gaussian distribution? Recall that discrete events are converted into dense features at each time t_i, so one would imagine that it is possible to compute correlations between discrete variables using Gaussina distribution.  And why does this correlation depend on the horizon T?  

My understanding is that the losses in Section 4.5 are used to train the model. Could you expand on how the ground-truth targets were defined to compute these losses?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes TAD-UP that learns both continuous and discrete dynamics for Time series Anomaly Detection via Unified Probabilistic modeling. The contributions of this paper are as follows:
1. Proposes the first method capable of learning both continuous and discrete dynamics for multivariate time series anomaly detection.
2. Proposes gated co-dependent NODEs with compound Poisson process to learn correlated continuous and discrete dynamics.
3. Models joint probability distribution across different dynamics and optimize the model with MLE.
Results show that TAD-UP is capable of state-of-the-art accuracy on multiple real-world datasets and better efficiency tradeoff.

### Strengths
Originality and Significance: This is the first work trying to learn both continuous and discrete dynamics for multivariate time series anomaly detection. The proposed gated co-dependent NODEs with compound Poisson process is novel for fusion continuous and discrete variates. This modeling approach enables the demonstrated state-of-the-art performance, providing a more accurate and efficient tool for multivariate time series anomaly detection.
Quality: Overall the solution is clearly motivated and reasonably implemented, and corresponding evaluations are comprehensive.
Clarity: The paper is well written and easy to follow.

### Weaknesses
1. Although the proposed method combining continuous and discrete dynamics is intuitive and yields convincing experimental results, it appears to lack sufficient theoretical analysis or further experimental interpretation of how continuous and discrete dynamics mutually reinforce each other in multivariate time series anomaly detection. 
2. For datasets containing only continuous variables, such as Creditcard, there is no explanation of how each module in the model are simplified to handle only continuous variables. Please clarify.

### Questions
1. I wonder if it's possible to provide some visualizations of ablation results and analyses to explain how continuous and discrete dynamics mutually enhance multivariate time series anomaly detection.
2. Taking the continuous co-ODE as an example, I understand that the author employs a gated TCN approach to introduce the discrete dynamics Z(0) during the process of dH(t)/dt. However, could this notation be optimized to better represent Z(0)'s involvement? For instance, dH(t)/dt|Z(0). Since this aspect constitutes the core contribution of the paper, highlighting it in the notation would be preferable. This suggestion also includes the representation of ODESolver(.) in Figure 2.

### Soundness
3

### Presentation
4

### Contribution
4
