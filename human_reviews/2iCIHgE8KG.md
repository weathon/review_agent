# Discovering Temporally Compositional Neural Manifolds with Switching Infinite GPFA

- Decision: Accept (Spotlight)
- Scores: 8, 8, 8, 6

## Abstract
Gaussian Process Factor Analysis (GPFA) is a powerful latent variable model for extracting low-dimensional manifolds underlying population neural activities. However, one limitation of standard GPFA models is that the number of latent factors needs to be pre-specified or selected through heuristic-based processes, and that all factors contribute at all times. We propose the infinite GPFA model, a fully Bayesian non-parametric extension of the classical GPFA by incorporating an Indian Buffet Process (IBP) prior over the factor loading process, such that it is possible to infer a potentially infinite set of latent factors, and the identity of those factors that contribute to neural firings in a compositional manner at \textit{each} time point. Learning and inference in the infinite GPFA model is performed through variational expectation-maximisation, and we additionally propose scalable extensions based on sparse variational Gaussian Process methods. We empirically demonstrate that the infinite GPFA model correctly infers dynamically changing activations of latent factors on a synthetic dataset. By fitting the infinite GPFA model to population activities of hippocampal place cells during spatial tasks with alternating random foraging and spatial memory phases, we identify novel non-trivial and behaviourally meaningful dynamics in the neural encoding process.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
the authors introduce switching infinite gpfa — an extension of the classical gpfa model to account for the possibly time varying dependence of observed neural activity on different latent factors.  the authors make this possible by using an indian buffet process as the generative model for a (infinite) binary mask that selects the latent features read out  to the observation space at each point in time.  they outline how to perform tractable variational EM inference/learning for this model class.  the authors then validate their model and inference/learning procedure on synthetically generated data, and then show how their method can be used to extract behaviorally meaningful latent features from rats performing a spatial navigation task.

### Strengths
the paper is very well written. the background section is clear and in my opinion and succesfully takes the reader from the original gpfa to their new generative model that incorporates an indian buffet process prior over the binary masking matrix, Z.   since approximate inference in this model is highly non-trivial, the authors developed an approximate variational EM procedure for inference/learning.  i appreciated the extensive discussion covering what terms are and are not tractable in the variational bound and how the authors deal with the intractable terms in a practical manner;  important details that would clutter the main text were referenced often and helped with further clarifications. their synthetic data example validates their inference approach and reassuringly shows the infinite gpfa model can match standard gpfa inference quality even when there is no encoding variability.  in their last experiment, they apply their method to neurophysiological recordings taken from a rat performing a spatial navigation task;  they demonstrate how their method can reveal the compositional structure of the latent space by identifying different latent factors being loaded onto the neural population at different locations or trial events.

### Weaknesses
more comparisons could be helpful.  for example, it could have been interesting to see how bGPFA also compares to infinite gpfa with and without model mismatch similar to the synthetic example presented.  

from fig 2b and fig 3b, it does appear that infinite gpfa takes substantially longer to reach convergence.  do the authors expect this difference gets substantially worse with higher latent state dimensionality? it could be helpful to see convergence plots for a dataset that requires higher latent state dimensionalities.

### Questions
for fig.2c what expected level of masking for Z was used?  on that topic, it could also be interesting to show how the gap between gpfa and infinite gpfa inference performance varies with the expected value of alpha.  additionally, an inline figure or addition to figure 1 helping to build intuition between numerical values of alpha and the expected number of features could be useful.

is the runtime comparison in Fig.2h for a single EM iteration, or only inference time?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a novel model, an extension to GPFA, that incorporates stochastic activation of latent factors in the loading process via IBP prior. This results in dynamically switching expression of latent factors for each neuron across different timepoints, hence incorporating the dynamic shifts in internal states of the animal. They apply their model, infinite GPFA, to two datasets, one synthetic and one real world  neuroscience dataset (hippocampal place cells during spatial navigation).

### Strengths
- **Novelty**: The proposed model, infinite GPFA, has a robust mechanism that allows for  estimation of both the number of latent factors and their time-varying activations without requiring manual tuning. In addition, the sparsity allows for learning of more interpretable latent factors, which is helpful for interpreting neural computations.
- This framework opens up new avenues in neuroscience for exploratory investigations of experimental data.
- Presentation is clear.

### Weaknesses
- More comparison to other methods could have strengthened the utility and performance of infinite GPFA, specifically, using some of the previously established methods like GPFA with ARD prior. Although GPFA with ARD prior is not designed to capture latent factors across time, it would be useful to show it quantitatively.

Minor points
- l060 ‘An as example,’ → ‘As an example,’
- Figure2.a the axis labels are illegible .
- In general figure 2 gets rendered very slowly, I am not sure the exact cause but it might be worth investigating because if it’s simple like rasterization or high resolution graphics, it can be easy to fix.

### Questions
- How does the infinite GPFA handle cases where it identifies overlapping/slightly differing  latent factors ?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present an extension to GPFA, a widely used latent variable model in neuroscience, that uses an Indian Buffet process as a nonparametric extension to automatically select latent dimensions at each time point. This avoids the need for a priori latent dimensionality choice in GPFA, a well-known limitation to the method, and allows for a sparse selection of latent activations at each time point, which can identify transitions in the latent representation, enhancing the models usefulness in the identification of behavioral states. The authors show strong validation on synthetic datasets as well as real spiking data. The theory is clear and model development and implementation is clear and sound.

### Strengths
The switching GPFA and switching infinite GPFA models effectively tackle a significant limitation commonly encountered in many latent variable models in neuroscience, particularly within GPFA: the a priori selection of latent dimensionality. Additionally, these models enhance the approach by allowing for unequal contributions of latent variables at different time points, addressing another critical shortcoming of traditional GPFA. This advancement represents a noteworthy contribution to latent variable modeling in neuroscience. The authors also incorporate inducing points for improved scalability, a practical and well-established extension from the existing GP literature.

### Weaknesses
The weakest part of the manuscript is the lack of evaluations to any competing approach. The authors appear only compare to variants of their own model. In particular, because the authors emphasize the advantage of not needing to pre-select latent dimensionality, some evaluation against the ARD approach in Jensen et al would be appreciated. The authors claim the ARD is inferior due to requiring marginalizing over all of the data to determine latent dimensionality, and this is sound reasoning, however, I am curious as to how exactly different the models fits and latent posteriors would be. It might be possible, for example, for the ARD GPFA model to learn an appropriate number of latent dimensions and have periods of time where different groups of latents are minimally or highly variable. I think it would help a reader get a sense of how svGPFA compares Bayesian GPFA, as the latter is a model that was motivated in a very similar way.

Note also that the manuscript "Uncovering motifs of concurrent signaling across multiple neuronal populations", Gokcen et al. also uses an ARD prior in a similar GPFA-style model - might be worth citing

One small point -- Figure 2 is difficult to render in the browser and this specific page lags. I suspect the figure size is too large, maybe due to panel d. Downsampling this figure before adding it to the latex might help.

### Questions
None

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the infinite GPFA model, which is Bayesian non-parametric extension of the classic GPFA by combining GPFA with an Indian Buffet Process Prior. This model can potentially infer infinite set of latent factors from data. A variational EM algorithm is proposed to perform the inference. The authors demonstrate the effectiveness of this model through analysis on simulated and real datasets.

### Strengths
- The paper is clearly written. The model formulation and related works are clearly introduced. 
- The authors have done extensive experiments on real neural data and synthetic data, and results seem good.

### Weaknesses
- The idea of combining GPFA with IBP prior is not revolutionary. 
- I listed some questions in the section below.

### Questions
- unclear sentence line 366: "Moreover, in an SLDS, only the latent dynamics changes following context switching, hence requiring a non-negligible number of timesteps (depending on the spectral radius of transition operator) for the reflection of context changes in the observation space. in contrast, the compositional nature of factor loading process in the infinite GPFA model allows immediate differential expression of latent processes into neural activities."

Can you clarify this a bit? infinite GPFA model seems to also have the factor loading process in latent space, why it allows immediate expression into neural activities than SLDS? 

- How is the number of features D selected for svGPFA in the experiments section for synthetic data and real data?

- What's the future research direction for this paper?

### Soundness
3

### Presentation
3

### Contribution
2
