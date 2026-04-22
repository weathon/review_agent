# ADiff4TPP: Asynchronous Diffusion Models for Temporal Point Processes

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
This work introduces a novel approach to modeling temporal point processes using diffusion models with an asynchronous noise schedule. At each step of the diffusion process, the noise schedule injects noise of varying scales into different parts of the data. With a careful design of the noise schedules, earlier events are generated faster than later ones, thus providing stronger conditioning for forecasting the more distant future. Our method models the joint distribution of the latent representation of events in a sequence and achieves state-of-the-art results in predicting both the next inter-event time and event type on benchmark datasets. Additionally, it flexibly accommodates varying lengths of observation and prediction windows in different forecasting settings by adjusting the starting and ending points of the generation process. Finally, our method shows superior performance in long-horizon prediction tasks, outperforming existing baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
ADiff4TPP is a flow-matching model for event sequences with a per-event piecewise linear noise schedule. The noise schedule allows for partial denoising to forecast event sequences with fewer events.

### Strengths
1. Strong experimental results
1. Asynchronous schedule mixes autoregressive forecasting and full-sequence generation.

### Weaknesses
1. The paper mixes up flow matching and diffusion in multiple places. These are two different methods and cannot be used interchangeably. The title says diffusion model but my impression is that this is actually a flow matching model.
1. Figure 1 has a label Event Duration, but events in TPPs do not have a duration.
1. The paper talks about windows, e.g. in Section 3.4, but this is misleading as it implies that the model forecasts events in a window $[t, t']$ when it actually generates a fixed number of events regardless of their times.
1. In my opinion, the variable-length generation is not really variable-length, because it has a fixed upper bound.

### Questions
1. Why is $A(s)$ a matrix instead of a vector?
1. How many steps did you use?
1. Does your asynchronous schedule require the number of steps to be larger than the number of events to be generated?
1. How are the results if you solve from 1 to 0 instead of $s_{start}$ to $s_{end}$?
1. Have you tried any smooth alternatives to your piecewise linear asynchronous schedule?
1. How do other diffusion-like models perform in the long horizon prediction task such as Add-Thin?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new diffusion-based model for TPPs. It first learns the latent representation of the events using an autoencoder. Then, it learns the diffusion with an asynchronous noise schedule which adds noise to events at different speeds, based on their temporal ordering. Authors theoretically motivate and justify the choices. The paper is clear and easy to follow. The results are good, model is outperforming established competitors on common benchmarks.

### Strengths
It makes sense to map an event which consists of a real time and categorical mark into a single vector. This transforms everything into a nice latent space and one can use any kind of generative model.

The proposed noise schedule is sound and authors show in section 4 some theoretical results.

The empirical results are good, showing improvement over previous models. Ablations studies justify the choices made in the paper.

### Weaknesses
Converting the events to a latent space is a convenient solution that avoids dealing with the specifics of TPP data. After this step is done, any model can be fit to capture the distribution of the latent space. Using diffusion after VAE step is not thoroughly tested. Both using asynchronous noise diffusion without VAE, and some other generative model on latent representations is a possible ablation study.

There are additional hyperparameters for beta-VAE and asynchronous diffusion that have to be picked and tuned. The method is limited by the maximum length of a sequence N. Sampling is slow for shorter sequences. The method is a combination of known techniques so novelty is limited.

### Questions
- Can you comment on the performance of non-diffusion models on latent representations?
- Why didn't you encode the full sequence to a latent representation using an encoder-decoder?
- Do you expect that beta-VAE will work for any TPP dataset?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
ADIFF4TPP proposes an asynchronous noise schedule for diffusion (flow matching to be correct) models for forecasting the next N events of a TPP. Furthermore, they propose to use a VAE to better encode Events to then model them in a latent space with the flow model.

### Strengths
- Derive asynchronous noise schedule for Flow matching.
- Asynchronous time schedule is interesting conceptually.

### Weaknesses
Even though proposing an asynchronous noise schedule could be interesting, the motivating claims are at times factually incorrect and the claims are not sufficiently justified by theory or experiment.

Motivation:
- Actual motivation for the asynchronous noise schedule is not backed-up. There is no theoretic reason why even when presented with noisier (teacher forced) past, the model should be limited.
- Shorter solving of ode for forecasting does not incorporate that later events also get less update steps, i.e. same trade-off between number of steps and expressivity. Furthermore, it could be that one actually needs more steps overall, since events are not updated at each step.


Method:
- TPPs are generative models, but this method only learns to forecast the next fixed number of points. 
- Usage of VAE compared to other common encodings is unmotivated and not properly showed beyond limited ablation. Why would we want to model in the latent space?

Experiments:
- Limited evaluation: It proposes a new noise schedule, but only presents limited ablation of the schedule. 
- Claiming efficiency, but do not provide wall clock comparisons.
- Limited number of benchmark datasets.
- Do not evaluate against AddThin, EventFlow, Ludke et al. 2024 for long-lange forecasting, even though they have shown Sota results. I know that none of them model marks, but could still be compared against, especially within their evaluation metrics.

Minor:
- Lacking command of related work, e.g., AddThin does not model the intensity, but instead directly the joint distribution; Ludke et al. (2024) does not allow direct likelihood evaluation; EventFlow is not a diffusion model and a lot of related work on TPPs is missing.
- Flow matching is claimed to be a variant of DMs, which is not true. 




Overall, proposing an asynchronous noise schedule is interesting but is improperly motivated and supported. Furthermore, to me the contribution of an asynchronous noise schedule is limited and in my honest opinion only warrants a workshop paper, i.e., is not a main track contribution. Furthermore, the "generative" model does not capture one defining and essential property of TPPs, the distribution over the number of events.

### Questions
- Why would one use a VAE plus a Diffusion/Flow model?
- Why didn't you show the usage of the asynchronous schedule on existing models, but also propose an all together new architecture?
- Why do you use a Flow model instead of just applying your schedule to a diffusion models for which you would not have to restrict the schedule as much, since there are no invertibility constraints?
- Why don't you model the number of events within the diffusion/flow-matching framework?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ADIFF4TPP, a novel diffusion model framework for temporal point processes that employs an asynchronous noise schedule to handle event sequence generation. Main contributions include: (1) A matrix-valued noise schedule A(s) enabling different diffusion speeds for different events; (2) A $\beta$-VAE-based latent space for handling heterogeneous event data; (3) A conditional flow matching objective extended to asynchronous schedules. The method supports both next-event and long-horizon prediction by adjusting the observation and prediction windows in the denoising process and achieves superior results on multiple TPP benchmarks.

### Strengths
(1). The work addresses some challenges in TPP modeling (e.g., data heterogeneity, variable sequence length, long-horizon prediction) and demonstrates the potential of asynchronous diffusion in sequential data modeling. The asynchronous diffusion concept for TPPs is genuinely innovative.

(2). The extension of flow matching to matrix-valued noise schedules is well-grounded.

### Weaknesses
(1). The specific parameterization of  A(s) in Equation (5) lacks theoretical or empirical justification.

(2). Add analysis about the trade-off between reconstruction quality and generative performance

### Questions
(1). How does the method handle very long sequences where L≫N? Algorithm 3 uses traditional sliding windows, but is there an evaluation of its effectiveness for capturing long-range temporal dependencies?

(2). How sensitive are the results to the choice of ODE solver and step size? Is there numerical error accumulation in long-horizon prediction?

### Soundness
2

### Presentation
3

### Contribution
2
