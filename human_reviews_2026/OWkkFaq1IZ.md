# From Observations to Events: Event-Aware World Models for Reinforcement Learning

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 2, 8, 4

## Abstract
While model-based reinforcement learning (MBRL) improves sample efficiency by learning world models from raw observations, existing methods struggle to generalize across structurally similar scenes and remain vulnerable to spurious variations such as textures or color shifts. From a cognitive science perspective, humans segment continuous sensory streams into discrete events and rely on these key events for decision-making. Motivated by this principle, we propose the Event-Aware World Model (EAWM), a general framework that learns event-aware representations to streamline policy learning without requiring handcrafted labels. EAWM employs an automated event generator to derive events from raw observations and introduces a Generic Event Segmentor (GES) to identify event boundaries, which mark the start and end time of event segments. Through event prediction, the representation space is shaped to capture meaningful spatio-temporal transitions. Beyond this, we present a unified formulation of seemingly distinct world model architectures and show the broad applicability of our methods. Experiments on Atari 100K, Craftax 1M, and DeepMind Control 500K, DMC-GB2 500K demonstrate that EAWM consistently boosts the performance of strong MBRL baselines by 10\%–45\%, setting new state-of-the-art results across benchmarks. Our code is released at [https://github.com/MarquisDarwin/EAWM](https://github.com/MarquisDarwin/EAWM).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper adds an "event-aware" layer to world models so that agents learn from meaningful changes (events) rather than raw frames alone. It defines events across modalities, predicts them with a dedicated head, and uses a Generic Event Segmentor (GES) to weight losses. Plugging this into DreamerV3 and Simulus yields strong empirical gains across Atari 100K, Craftax, DMC, and DMC-GB2.

### Strengths
- The paper proposes a novel extension that can be attached to existing world models.
- Strong empirical performance across benchmarks.

### Weaknesses
- Event definition in Section 2.2 feels rushed:
  - Eq. (1): the image intensity $I_t$ is not explicitly introduced in the text.
  - Line 151: "direction of the event" is unclear, does this mean the sign of the brightness change?
  - For ordinal data, the event is defined as $p_i$. For consistency with visual data it should likely be a tuple e_{t,i}.
  - Eq. (2): $\Delta o_i$ is not defined.
  - This carries into Eqs. (3)-(4): wwhat exactly are the events $e$? In places it seems $p$ might be intended instead of $e$.
- Discrepancies in Figure 3 vs. Eq. (3):
  - The sequence model is labeled $F_\theta$ in the figure but $f_\theta$ in the equation.
  - In Eq. (3), the representation model is conditioned on $y_t$, in the figure it is conditioned on $h_t$.
  - In Eq. (3), the event predictor conditions on $y_{t-1}, y_t$, in the figure it appears to use $y_t, y_{t+1}$.
  - Why is the observation predictor categorized under "EA" rather than "WM"?
- The sentence "We employ Adaptive Gaussian Mixture Models to automatically detect such events from video" needs concrete detail. What features are modeled, how thresholds are chosen/updated, and how detections map to the event tuple. This is central to the method but currently vague.
- The description of the Geneirc Event Segmentor is hard to follow:
  - How is $\alpha_t$ computed exactly, do indices $i$ range over pixels or batches? This should be made precise.
  - Line 272: please define "event boundaries."
  - The role of GES (as a weighting coefficient for the event/observation losses) emerges only from the equations. The text says "reallocate attention" but should explicitly tie to Eqs. (7)-(8).
  - The function $g$ in Eq. (6) appears without explanation in the main text, only the appendix clarifies it.
  - From Eq. (6), GES seems independent of $\theta$. If it has no learned parameters, why is it part of the world model rather than a fixed weighting step?
- Reported scores show discrepancies vs prior work:
  - DreamerV3 numbers on Atari 100K are lower than the original paper (mean 1.15 reported here vs 1.25), which makes the gap to EADream's 1.29 smaller than suggested. Please clarify whether these were re-runs and why they differ.
  - For Simulus, the reported mean and median (1.61 and 0.74) differ from the Simulus paper (1.65 and 0.98). Please explain the deviation.
- Ablations are unclear:
  - "No Event Predictor": What exactly is disabled? If event prediction is off, shoudn't the model reduce to the original world model? Why do "EADream w/o Event Predictor" and DreamerV3 differ? Is this related to the RSSM-OP change, but if so, why is there the same effect for Simulus?
  - "No GES": If GES is off, is $\omega$ also set to $0$? If so, consider an additional ablation where GES is on but $\omega = 0$ to disentangle effects.
  - "Without Observation Prediction": RSSM-OP seems to matter a lot. An ablation "Dreamer + RSSM-OP" would isolate its contribution.
- Minor: Missing citations to Craftax and DMC in Section 4.1.

I'm willing to raise my score if the presentation is improved and these points are addressed, as the method and results look very promising. My current score is low because, in its present form, the paper should not be accepted due to several presentation issues, but if these are fixed I would lean toward acceptance.

### Questions
- The method adds several hyperparameters on top of already hyperparameter-heavy world models. How sensitive are the new hyperparameters in practice, and how difficult was tuning?
- Please address the clarification requests in the weaknesses above.

### Soundness
2

### Presentation
1

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
The paper suggest to extend a typically world model for reinforcement learning (e.g. Dreamer) by an event predictor.  The paper claims that that addition leads to better latent states.  Experiments show that those latent states indeed achieve better performance on various benchmarks.

### Strengths
- well written introduction

- excellent results across different benchmark datasets, e.g. Atari 100K

- the framework can be hooked to existing methods, e.g, Simulus and Dreamer, and consistently improves results

- the additional events are auto-labeled using an "Automated Event Generator" (based on Adaptive GMMs)

### Weaknesses
- not much to criticize!

### Questions
- Eq. (1): $p_i$ doesn't appear in the formula!  when is $p_i$ positive or negative for visual inputs?

- How sensitive is the method to the choice of $C_I$ in Eq. (1)?

- Automated Event Generator:  what are events e.g., in the game of Pong?

- I didn't really understand, what is the point of the "Generic Event Segmentator"?  What do you mean with "reallocate attention from events to raw observations"?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Event-Aware World Model (EAWM), a framework that enhances model-based reinforcement learning by predicting events rather than raw observations. The method introduces three key components: an automated event generator that extracts events from multi-modal observations without manual labels, an event predictor that shapes representations through information bottleneck optimization, and a Generic Event Segmentor (GES) that identifies event boundaries to stabilize training.

### Strengths
* Well-motivated approach.
* The paper demonstrates consistent and substantial improvements across diverse benchmarks.
* The paper successfully demonstrates applicability across different architectures.

### Weaknesses
* The core idea of integrating dynamics between frames as an additional learning signal for world models has already been explored by DyMoDreamer [1]. Moreover, DyMoDreamer achieves better performance than EADreamer on both Atari 100K and DeepMind Control Vision benchmarks with a simpler method.
* While the authors claim to learn task-relevant environmental dynamics, the EAWM primarily relies on inter-frame pixel differences. This approach may still filter in many task-irrelevant events.
* Adding the event detector likely increases the training time for world models. The paper does not report wall-clock training time comparisons or computational cost analysis.

[1] Dymodreamer: World Modeling with Dynamic Modulation

### Questions
* Why do EASimulus and EADream have different event predictor inputs? EASimulus conditions on $(y_{t-1}, y_t)$ while EADream conditions on $(h_t, \hat{z}_t, z_t)$. 
* Can the authors provide qualitative imagination results on DMC-GB2? Given that DMC-GB2 tests generalization to visually noisy observations (randomized colors and video backgrounds), it would be valuable to visualize the world model's imagined trajectories and corresponding event predictions. This would help verify whether EAWM truly learns to attend to task-relevant objects while filtering out background distractors, or if the performance gains come from other factors.
* What is the wall-clock time comparison with baselines?

I am willing to raise my score if the authors can address these concerns.

### Soundness
3

### Presentation
3

### Contribution
2
