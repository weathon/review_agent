# Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion

- Decision: Accept (Poster)
- Scores: 4, 4, 8

## Abstract
Generating tabular data under conditions is critical to applications requiring precise control over the generative process. Existing methods rely on training-time strategies that do not generalise to unseen constraints during inference, and struggle to handle conditional tasks beyond tabular imputation. While manifold theory offers a principled way to guide generation, current formulations are tied to specific inference-time objectives and are limited to continuous domains. We extend manifold theory to tabular data and expand its scope to handle diverse inference-time objectives. On this foundation, we introduce Harpoon, a  tabular diffusion method that guides unconstrained samples along the manifold geometry to satisfy diverse tabular conditions at inference. We validate our theoretical contributions empirically on tasks such as imputation and enforcing inequality constraints, demonstrating Harpoon's strong performance across diverse datasets and the practical benefits of manifold-aware guidance for tabular data. Code URL: https://github.com/adis98/Harpoon

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper generalizes previous ideas from manifold theory to diffusion models for tabular data. This allows the usage of differentiable losses at inference time to guide samples along the data manifold at a given diffusion time $t$. Depending on the objective, this produces samples that satisfy constraints based on partially observed data and specified inequality constraints without the need for re-training the diffusion model.

### Strengths
- The authors generalize previous results, extending the usefulness of manifold-based insights to tabular data. In particular, they remove the necessity of squared losses, which extends the effective conditioning capabilities.
- The ability to condition on certain features or constraints at inference time without the need for training the model on that specific conditional generation task is very valuable, in particular for tabular data.
- The illustrations mostly paint an intuitive picture of the underlying mechanisms.
- I like the effort of making the model more comparable to other diffusion-based approaches by adjusting the backend architecture. 
- The presented results show competitive or better performance in practice. Great improvements can be seen in scenarios that impose inequality constraints.

### Weaknesses
- The methodological background on diffusion is basically non-existent and the overview of the tabular diffusion models is severely underdeveloped. Since TabDDPM, many models have been proposed that 1) often considerably outperform TabDDPM and 2) do not rely on multinomial diffusion. In fact, models that treat categorical data in some continuous space exist (e.g., TabSyn [1] or CDTD [2]). It is unclear how the results extend to such models.
- The missingness rates of 0.5 and 0.75 in the main results seem very extreme. The cited DiffPuter paper uses 30%. In reality, it is questionable why we would want to impute data when 75% of the data is actually missing. For more moderate missing rates, the performance advantage of the proposed method is not clear but it remains mostly competitive.
-  Only using a single metric to evaluate sample quality of tabular data (in the case of inequality constraints) is not enough. Metrics like the detection score (see, e.g., metrics used in [1] and [2]) which evaluate the joint distribution more holistically would be interesting to see and give more insight into how the manifold guidance impacts samples.
- The examined constraints for the conditional generation tasks are rather simple and focus on a maximum of two features (one categorical, one continuous), so the proportion that training samples are valid is still quite high. It is unclear how the framework performs under more constraints, which make the guidance more difficult. An ablation study should investigate what happens if the number of constraints is increased, such that the number of valid training samples shrinks. In the extreme case, is the method able to recover a single valid observation?
- The paper makes the claim in line 112 that extending the results from Chung et al. to tabular diffusion is not trivial. One reason is that tabular data contains discrete features. This, however, is then solved by the rather trivial solution of simply one-hot encoding discrete features and treating them as continuous. Besides this encoding, there is no tabular-data-specific modeling in the paper.
- Figure 2a) does not actually illustrate the behavior of a "spotlight" that becomes sharper near $\mathcal{M}_0$.  In the Figure the position of $x_t$ relative to $\mathcal{M}_0$ never changes. When the spotlight becomes sharper, I would expect an $x_s$ closer to $\mathcal{M}_0$ where $s < t$. Note also that the orthogonal projection of $x_t$ and $x_s$ need not be the same.
- Making the assumption in line 253 that $Q_t(x_t)$ is approximately orthogonal for all $t$ based on only a single empirical observation seems not well-founded. Could this assumption not be weakened by making $\eta$ time-dependent, such that the guidance term has more impact when it is also more likely to hold, i.e., as $t\rightarrow 0$?
- It is not clear what tabular diffusion framework is assumed. It appears to be very similar to to StaSy model [3], which also one-hot encodes categorical data. The authors should be more specific about the framework they are using. If it can be used on any or most tabular diffusion frameworks (maybe under certain conditions) this should be stated as a strength of the framework. Coming up with an entirely new generative model does not highlight the strength of the potentially more general approach.
- Getting the tangential gradient requires backpropagating through $\epsilon_\theta$. It should be clarified how costly the method is, in particular at inference time and relative to the other baselines.
- Based on the inference time losses and the results, it should be highlighted by the authors that it is not guaranteed that samples satisfy the specified constraints. The approach is more similar to penalization than imposing hard constraints.
- The authors do not discuss how their approach differs from classifier-guidance when using similar losses.

---


[1] Zhang, et al. (2024) Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space. ICLR.

[2] Mueller, et al. (2025) Continuous Diffusion for Mixed-Type Tabular Data. ICLR.

[3] Kim, et al. (2023) STaSy: Score-based Tabular data Synthesis. ICLR.

### Questions
In addition to the suggestions and questions stated in Weaknesses, I kindly ask the following questions:

- Could this manifold guidance lead to a skewed (conditional) data distribution?
- Your guidance mechanism leads to *soft* constraints but not cannot impose *hard* constraints. This should be made clear in the text. Does that lead to issues in the imputation task? Since the non-missing values are not strictly fixed but still updated during sampling, could it lead to the final samples actually deviating from the partially observed information?
- Does the guidance also work for latent diffusion models (TabSyn, CDTD, etc.) or is a data-space model necessarily needed?
Why is it a problem if models push samples orthogonally towards $\mathcal{M}_0$ (line 104/105)? Even if it skips a shell, in the next update step, the model, which is conditioned on $t$, should have no problem continuing the path if trained properly. Is this an artifact from a model that is not capable / complex enough?
- How does your guidance generalize to high dimensions, in particular when increasing the number of features or categories in a dataset? One-hot encoding is known to be very inefficient and will blow up the dimensions when applied to a categorical feature with 100s or 1000s of categories. 
- Can relative or logical constraints be accommodated? For instance, consider a data table of order and delivery dates. Naturally, we would expect that order date < delivery date.
- Figure 2 b) and c) give a good intuition of moving a sample along a shell. Can there be situations in practice where this fails? For example, if at a particular level the manifolds are no longer connected, e.g., $\mathcal{M}_t$ is not a single piece but two separate ones?
- Are you using the QuantileTransformer to transform the continuous features for the TabDDPM model? From own experience this appears to be a crucial step for performance and is ubiquitous in tabular diffusion models.
- In line 420: "not just the constrained ones [...]", do you actually mean the *un*constrained features? 
- Figure 3 is not well-designed. It does not does not include the red or green lines due to unfortunate overlaps. It would yield a more consistent argument if the x-axis would indicate $t \in [0,1]$ and not the denoising step.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a method that can perform conditional tabular diffusion named HARPOON. It can generate new tabular data with specific conditions. In the theory contribution section, the authors show: 1. diffusion de-noiser acts as orthogonal projectors to data manifold 2. any differentiable inference-time loss has gradient in the tangent space of the manifold and therefore updates using this gradient can preserve realism. HARPOON is a method of combining unconditional de-noising step and the constraint-guided tangent gradient step. The experiments show that HARPOON outperforms other baseline methods.

### Strengths
1. The paper is well written and provides both theoretical and experimental contributions
2. The theory is novel and is the first to use diffusion model's orthogonal projection to manifolds in tabular setting
3. HARPOON can handle mixed data types and has much lower constraint violation rate when doing conditional generation

### Weaknesses
1. Computation time might be an issue but this is not discussed in the experiments
2. The utility, fidelity and privacy aspects of the generated tabular data is not discussed. It would be great to see where HARPOON stands among these 3 aspects.

### Questions
1. Does the tabular data generated from HARPOON improve the downstream model?
2. What does the run time of HARPOON look like against other methods?

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
3

### Summary
The paper proposes a new way to guide diffusion models for tabular generation at inference time. In particular, they show how at every time step $t$ most of the datapoints lie on a shell of dimension $d-1$ ($d$ being the original dimensionality of the datapoints) and that while the denoising process creates an orthogonal projection onto the shell of the datapoint $x_t$, the gradient of any loss function defined  starting from a condition $c$ is tangent to such shell. Exploiting these geometric results, they are able to create a new procedure for guiding the diffusion process at inference time, where essentially they interleave the diffusion process which (as they nicely put it) acts as a "compass" towards the manifold of the original datapoints and the update wrt the loss function that encodes the constraints which moves along the shell to guide the model towards the right region of the manifold.

### Strengths
The paper is very well written and provides a lot of intuitions about the results they propose. I really appreciated the visualisations of the gradients and the updates. 

The experimental analysis is extensive and with very positive results. 

The authors give a very nice geometric explanation of why their method works.

**Note:** It is difficult for me to assess the novelty of this work wrt the previous works on diffusion models as I am not familiar with them.

### Weaknesses
1.  The authors do not report the sampling generation time. As this is an important metric for tabular data generation, it would be nice to have it

2. In theorem 3.2 $\mathcal{C}$ is not defined. Also it is not clear which conditions we have on $\mathcal{C}$. Can it really be any arbitrary information? For example (Stoian & Giunchiglia, 2025) has extended the work cited in your paper to constraints expressed as disjunctions over linear inequalities. This defines non-convex and disconnected spaces hence violating your assumption 1. Would this time of conditioning be allowed? What about polynomials? 

3. A better definition of the alpha metric is needed. 



Minor things: 

1. Citation for Borisov et al. is missing the year 
2. Sometimes equation is written with capitol E and sometimes with lower case e 

References: 

Stoian & Giunchiglia. Beyond the convexity assumption: Realistic tabular data generation under quantifier-free real linear constraints, ICLR, 2025.

### Questions
1. In order to impose the linear inequality constraints you had to devise an ad-hoc loss function. Do you see this as a possible bottleneck for the widespread application of your solution? 

2. Aside from categorical constraints, it is very reasonable to assume that tabular data have disconnected support. This makes me wonder how realistic is assumption 1 and also how important it is that assumption 1 is met in practice. Would it be possible to have an ablation study where a multiple datasets are created with each of them violating the assumption in different degrees and then studying how the method performs on them?

### Soundness
3

### Presentation
3

### Contribution
3
