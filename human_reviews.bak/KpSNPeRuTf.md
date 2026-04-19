# A Novel Autoencoder Based Approach for Counterfactual Estimation Using Sparsity Constraints

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 1, 3

## Abstract
Building upon the abduction-action-step scheme and the structural causal model framework, this paper introduces the Conditional Sparse Autoencoder (CSAE), a novel approach for time series counterfactual estimation using encoder-decoder based architectures with a sparsity constraint to disentangle the roles of the inputs in the expected outputs. We benchmark CSAE with Conditional Variational Autoencoder (CVAE), the most widely adopted encoder-decoder architecture for counterfactual estimation, showing that CSAE clearly outperforms CVAE in this domain. Furthermore, we demonstrate the versatility of CSAE by extending it to image-based counterfactual scenarios, obtaining promising results. This work has important implications for a wide range of applications across various domains including finance, healthcare, and transportation, where being able to perform accurate counterfactual estimations is critical for decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a conditional sparse autoencoder to perform counterfactual inference on time series and images

### Strengths
- The paper is well written in terms of language and organisation
- The method attempts to tackle an important issue , that is counterfactuals in time series

### Weaknesses
- The authors have missed a lot of related literature in counterfactuals of timeseries that they should be comparing, contrasting and benchmarking against. For example 3: 

    - Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations Seedat et al ICML 2022
    - Causal Transformer for Estimating Counterfactual Outcomes, Melnychuk et al ICML 2022
   - Non-parametric identifiability and sensitivity analysis of synthetic control models, Zeitler et al CLeaR 2023
 As a matter of fact the authors completely ignore the entirety of Synthetic Control literature and Epidemiology and Bio-Signals literature that has at its crux the estimation of counterfactual timeseries. 

- It is unclear how the proposed method performs the abduction and action step. The only information given is that the representation is sparse. Does this mean that the new counterfactual sample is just dictated by the conditioning factor? Does it include traversing the latent space ? 

- It is unclear what kind of causal guarantees the method offers. It appears that the only contribution is a sparsity constraint that is neither novel nor clear how it gives us any causal properties. 

- The image task is not properly motivated nor explained. The figure of different color hues is not clear what is the factual part and what is the counterfactual. 

- The introduction of sparsity to an autoencoder is not novel as it is well known in the community

### Questions
- How does the proposed method compare in theory and practice with synthetic control, and other Neural network and transformer based methods for prediction of time series counterfactuals ? 
- How does the method actually guarantee any causal insights ? 
- How is the abduction and action performed in this method ? 



Overall I dont think this paper is ready yet for publication. Its contributions are not novel, lacking any clear and sound causal guarantees. The evaluation and comparison is insufficient

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper adds an L1/L2 regularization term over the bottleneck latent variables to the regular Autoencoder loss, Eq. (2).

### Strengths
This paper conducts a comprehensive review of Generative Models for counterfactual prediction, and adds an L1/L2 regularization term over the bottleneck latent variables to the regular Autoencoder loss, Eq. (2).

### Weaknesses
- The structure of this paper is confusing, leaving me uncertain about the author's intended message and purpose. 

- Novelty: The loss function of CSAE is $\mathcal{L}_{\mathrm{CSAE}}(\mathbf{x})=|x-\hat{x}|-\lambda \sum_i\left|z_i\right|$. Is that all? An L1/L2 regularization term? How does it perform counterfactual estimation and what is the counterfactual prediction objective?

- Is Time Series Counterfactual the focus of this manuscript? Why did the authors spend a significant amount of space discussing content that is not directly related to it? It was not until the fourth paragraph that the topic was introduced.

- The motivation of this paper is a bit confusing. What are the challenges in conducting Time Series counterfactuals? How do traditional Time Series methods approach this and what are their limitations? Why can an L1/L2 regularization term over the bottleneck latent variables implement Time Series Counterfactuals? What is the motivation behind this? The presentation should focus more on the core issues addressed in this paper. 

- Typos: “… by jointly training and encoder … and a decoder …” → “… by jointly training an encoder … and a decoder …”

### Questions
See weakness

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a conditional auto-encoder with sparsity constraints for counterfactual estimation and generation. The method is motivated for time-series data, but tested on both time-series and image data. Experimental comparisons were made to conditional VAE and, in the setting of time-series data, a LSTM.

### Strengths
Counterfactual estimates and generation are relatively under-explored in time-series data. The paper is thus tackling an important and worthy research question.

### Weaknesses
This work can be improved in several major areas:

1. While the method is heavily situated within the context of counterfactual estimation and generation, the methodology itself is very marginally tied to causal inference. In fact, it is more related to disentanglement itself. Even for disentangling purpose — the use of sparsity constraint to minimize the information in the bottleneck is heuristic without any theoretical guarantee that z will not attempt to encode information about the conditioning/parent variable (and the success of which should largely depends on the regularization strength). Nothing in this seems to be addressing causal modeling (other than the conditioning), or addresses causal disentanglement at the presence of correlation.

2. In all experimental settings, it is stated that there is no “confounding” in the data and that the two factors (e.g., digit and hue) are independent. This is quite confusing — if two factors have causal relations, there is a high likelihood that they will appear correlated in the data, thus making naive disentanglement (assuming independent generative factors) difficult — In fact, this is the key challenge for most causal inference and generation work to address such “correlation” from observational data. If this correlation does not exist (which I interpret as the confounding as mentioned in the paper), such as estimating intervention effect from randonmized trial data, the the key challenge is gone. 

3. Similarly, while the paper was using time-series as a main motivation, pointing out that existing works that deal with images cannot be directly applied to time-series data, it was not pinpointed what exactly are the challenges associated with time-series counterfactuals, and how the presented method addresses them.

4. The work is missing a large number of necessary baselines for comparison, including in time series data (such as RMSN [1], CRN [2], CausalTransformer [3]) and in static image data (causalGAN, causal-VAE, SCM-VAE, ICM-VAE, etc)

[1] Forecasting Treatment Responses Over Time Using Recurrent Marginal Structural Networks
[2] ESTIMATING COUNTERFACTUAL TREATMENT OUTCOMES OVER TIME THROUGH ADVERSARIALLY BALANCED REPRESENTATIONS
[3] Causal Transformer for Estimating Counterfactual Outcomes

### Questions
The contribution and rigor of the presented work are overall unclear to me. It’d be helpful if the authors can address my major comments above.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes to use conditional sparse autoencoders (CSAE) instead of  conditional variational autoencoders (CVAE) for counterfactual estimation in the DSCM framework focused on timeseries problems. The authors compare the two methods on a synthetic, semi-synthetic and a proprietary timeseries dataset,   as well as coloured MNIST. The experiments indicate superior performance of the CSAE compared to the CSAE.

### Strengths
The paper tackles the important problem of counterfactual estimation for time-series problems and identifies performance drawbacks in currently used models.

### Weaknesses
The paper is a simple combination of the deep SCM framework with sparse autoencoders. The writing of the paper could use some editing as it's riddled with errors. Furthermore, section 3.1 very closely follows [1] while some of the metrics in 4.2 very closely follow [2], almost being a citation. Even thought the results look promising, the novelty is very limited and the experimental setup is too narrow to provide evidence of this method being suitable for general settings.

[1] Pawlowski, Nick, Daniel Coelho de Castro, and Ben Glocker. "Deep structural causal models for tractable counterfactual inference." Advances in Neural Information Processing Systems 33 (2020): 857-869.
[2] Monteiro, Miguel, et al. "Measuring axiomatic soundness of counterfactual image models." The Eleventh International Conference on Learning Representations. 2022.

### Questions
- The paper mentioned that methods are deterministically decoded, and as such does not use well defined probabilities as section 3 suggests. Is this wanted?
- Why is the precision required for timeseries higher than for other counterfactuals?
- The problems mentioned in the paper are already brought up in [2] (see the confounded data experiments) and have been tackled in e.g. [3]. How does this method compare?
- Why does the probabilistic nature of CVAEs introduce additional errors?
- The explanations of the metrics are hard to follow. It would be helpful to add equations here, especially for the "Added variations"
- Whats the assumed causal graph for the experiments?

[3] Kumar, Amar, et al. "Debiasing Counterfactuals in the Presence of Spurious Correlations." Workshop on Clinical Image-Based Procedures. Cham: Springer Nature Switzerland, 2023.

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor
