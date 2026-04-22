# Hyper Hawkes Processes: Interpretable Models of Marked Temporal Point Processes

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Foundational marked temporal point process (MTPP) models, such as the Hawkes process, often use inexpressive model families in order to offer interpretable parameterizations of event data.  On the other hand, neural MTPPs models forego this interpretability in favor of absolute predictive performance. In this work, we present a new family MTPP models: the _hyper Hawkes process_ (HHP), which aims to be as flexible and performant as neural MTPPs, while retaining interpretable aspects. To achieve this, the HHP extends the classical Hawkes process to increase its expressivity by first expanding the dimension of the process into a latent space, and then introducing a hypernetwork to allow time and data dynamics. These extensions define a highly performant MTPP family, achieving state-of-the-art performance across a range of benchmark tasks and metrics. Furthermore, by retaining the now-conditionally linear recurrence, the HHP also retains much of the structure of the original Hawkes process, which we exploit to create direct probes into _how_ the model creates predictions. HHP models therefore offer both state-of-the-art predictions, while also providing an opportunity to ``open the box'' and inspect how predictions were generated.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Hyper Hawkes Process (HHP), a novel marked temporal point process model that combines the interpretability of classical Hawkes processes with the expressivity of neural MTPPs. The key innovations include: (1) lifting Hawkes dynamics into a latent space to decouple from mark dimensionality, (2) employing a hypernetwork to generate time-varying dynamics while maintaining conditionally linear recurrence, and (3) developing event-level attribution methods for interpretability. The paper also demonstrates competitive performance across seven real-world benchmarks and provide interpretability analysis on synthetic tasks.

### Strengths
(1). The idea of combining a hypernetwork with a linear Hawkes recurrence is well-motivated.

(2). The model in the paper is also rigorously developed, with clear motivations for each extension (latent space, time-varying dynamics, etc.).

(3). The work contributes to both the performance and interpretability of MTPPs, which is important for real-world applications (e.g., healthcare, finance)

### Weaknesses
(1). The paper should better highlight how HHP differs from and improves upon these models.

(2). The interpretability tools are only demonstrated on synthetic data. It's better to also show their utility on real-world datasets.

(3). The choice of a GRU-based hypernetwork is not justified against alternatives such as transformers, SSMs and so on.

### Questions
(1). Why was a GRU chosen over other sequence models? Was attention or Mamba considered?

(2). Can the authors discuss how HHP could be used in a real-world setting (e.g., healthcare) to provide insights beyond prediction?

### Soundness
2

### Presentation
2

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
While neural network–based adaptations of the classical Hawkes process show strong performance, they often lack interpretability.
This paper introduces an interpretable TPP model called the Hyper Hawkes Process (HHP), where interpretability is achieved through the model’s architecture design. Here, interpretability means that the model can identify the attribution or influence of each event.
The main contribution of this work is addressing the typical trade-off between interpretability and performance. The proposed method strengthens the model’s capability by increasing the latent dimension. Quantitatively, it performs competitively with baseline methods while uniquely offering interpretability.

### Strengths
- Adapting neural networks to classical methods improves expressiveness but can make the model more of a “black box.” The authors address this by using eigenvector decomposition to keep the model interpretable, offering a clear particle-based view and attribution.
- They also overcome the common trade-off between performance and interpretability. By increasing the latent dimension, the model becomes more expressive and reduces this trade-off.
- One interesting observation in this work is that the conventional Hawkes process ties the latent dimensions to the mark space, which opens up a promising direction for future research.

### Weaknesses
- Minor. The visualization of particle attribution in Figure 2 is difficult to interpret. Distinguishing sample lines using a dotted style or alternative markers could improve clarity.
- The model increases the latent dimension and adds more architectural components, which may lead to longer runtime. It is unclear whether the runtime is comparable to baseline methods.

### Questions
- In Figure 2, could the particle attribution be visualized more clearly, for example by using dotted lines or alternative markers to distinguish sample lines?
- Given that the model increases the latent dimension and adds more architectural components, how does its runtime compare to the baseline methods?
- Other than visualization, are there any practical applications of the particle attribution?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper models the dynamics of the Hawkes process with exponential decay kernel in a single-layer latent space and applies an affine transformation to obtain the intensity function. Further, the paper leverages a hypernetwork to infer a time-varying matrix for the exponential decay rate parameter given historical events. Model parameters are learned via maximum likelihood estimation derived from the projected intensity function. Experimental results on seven datasets for next-event prediction demonstrate log-likelihood performance competitive with baseline methods. Additionally, the paper explores the interpretability of the proposed approach by adopting event-level attribution-based techniques.

### Strengths
- The paper is well-written and easy to follow.
- Results on seven datasets for next-event prediction demonstrate log-likelihood performance that is competitive with baseline methods.
- The paper explores the interpretability of the proposed approach by adopting event-level attribution techniques.

### Weaknesses
- The paper appears to be a straightforward extension of the previously proposed DLHP approach with minimal modifications, namely, using a single-layer latent space and a hypernetwork to estimate event-specific decay rates.

- Given the limited technical contributions, the experimental results are underwhelming:

1) Although the paper claims state-of-the-art performance, the reported log-likelihood appears comparable to baseline methods.
2) Table 1: It is unclear why raw event accuracy metrics are omitted in favor of rank-based evaluations.

-  The paper asserts that the proposed approach is more interpretable than prior methods. However, by modeling dynamics in the latent space, the event-triggering kernel becomes less interpretable. It is unclear how the proposed attribution method enables recovery of the ground-truth triggering kernel. I encourage the authors to discuss this further and include qualitative results, as well as comparisons with other interpretable approaches such as [1,2]. Unfortunately, the results presented in Figure 2 are unconvincing without direct comparisons between predicted and empirical ground-truth kernels.

- The proposed approach also appears computationally expensive relative to alternatives. I encourage the authors to provide an analysis of computational complexity, including the number of parameters and training/inference time.


- Given that the ablation study shows removing the latent space and hypernetwork does not result in a significant performance drop, it is unclear why the added complexity is necessary, especially since a simpler model would be more interpretable and computationally efficient than the proposed approach.

**References**
- [1] Isik Yamac et al., "Hawkes Process with Flexible Triggering Kernels", MLHC 2023.
- [2] Pan Zhimeng et al., "Self-adaptable point processes with nonparametric time decays", NeurIPS 2021.

### Questions
- Table 1: Could you provide raw accuracy metrics instead of ranks?
- Tables 1–2: Could you include standard errors for the predictions?
- Could you benchmark the proposed approach against baselines in terms of computational efficiency?
- Table 1: Could you also include the ablation models?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose hyper Hawkes process (HHP), which uses a hypernetwork to adapt dynamics over time and to decouple its latent space from the dimensionality of the marks. Experiments demonstrate that HHP outperforms state-of-the-art models and enable insight into the model mechanics for interpretability purposes.

### Strengths
The proposed model is well justified and described in enough detail in the main paper to give a reader a sense of the mechanism driving the model, and unlike existing models that use a representation model for the history of events, the proposed model does it in a manner that is interpretable, i.e., the contribution of individual marks to the estimated intensities can be also estimated. Moreover, as the authors point out, the "transition" operator is both expressive and efficient due to the use of an spectral decomposition.

The leave-one-out estimator in (9), which stems from (8) constitutes a natural way to quantify the influence of an event in the history on the estimated intensity, which aids interpretability.

The experiments are, in general terms, extensive. The authors consider seven datasets, event and mark metrics spanning model likelihood, error (RMSE and accuracy) and calibration (PCE and ECE), and ablation experiments to demonstrate the contribution of each component of the proposed model. There is also an experiment illustrating the interpretation capabilities of the model.

### Weaknesses
The main weakness of the proposed model lies on the experimental evaluation of the proposed model. Specifically, the proposed model has the overall best rank (Table 1), but is only better than the competing methods in half of the metrics. However, the bigger issue is that the metrics do not account for variation, thus it is very difficult to assess the significance of the results, so in that sense it may be possible that the difference between HHP and DHLP is not at all significant. Moreover, the poor calibration performance of the model is concerning in the sense that it greatly impacts interpretability, which is a core value of the proposed model.

Although presented as an advantage, the results indicate that the expressivity of a time-dependent transition operator may not be as useful at least in terms of performance.

The interpretability experiment is a welcome addition, however, it will be better if the authors also presented interpretability experiments with real data. On the same line, although the interpretability of the model is a desirable property, one wonders if interpretation at the particle level is realistic in practical scenarios.

### Questions
- Figure 1 needs either a better caption or needs to be described in more detail.
- Have the authors consider an ablation where beta_t = D_i? which seems related to the ablation study where V_i = V.
- How to reconcile the poor calibration with the better likelihood metrics?

### Soundness
3

### Presentation
3

### Contribution
3
