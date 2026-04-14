# Continual Slow-and-Fast Adaptation of Latent Neural Dynamics (CoSFan): Meta-Learning What-How & When to Adapt

- Decision: Accept (Poster)
- Scores: 10, 6, 5, 6

## Abstract
An increasing interest in learning to forecast for time-series of high-dimensional observations is the ability to adapt to systems with diverse underlying dynamics. Access to observations that define a stationary distribution of these systems is often unattainable, as the underlying dynamics may change over time. Naively training or retraining models at each shift may lead to catastrophic forgetting about previously-seen systems. We present a new continual meta-learning (CML) framework to realize continual slow-and fast adaptation of latent dynamics (CoSFan). We leverage a feed-forward meta-model to infer *what* the current system is and *how* to adapt a latent dynamics function to it, enabling *fast adaptation* to specific dynamics. We then develop novel strategies to automatically detect *when* a shift of data distribution occurs, with which to identify its underlying dynamics and its relation with previously-seen dynamics. In combination with fixed-memory experience replay mechanisms, this enables continual *slow update* of the *what-how* meta-model. Empirical studies demonstrated that both the meta- and continual-learning component was critical for learning to forecast across non-stationary distributions of diverse dynamics systems, and the feed-forward meta-model combined with task-aware/-relational continual learning strategies significantly outperformed existing CML alternatives.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper introduces CoSFan, a continual meta-learning framework that adapts to changing dynamics in high-dimensional time-series data. By using a novel "what-how & when" model, it detects task shifts and quickly adjusts to new tasks, minimizing catastrophic forgetting. The claim is that CoSFan outperforms existing methods in accuracy and adaptability across non-stationary data distributions.

### Contributions
- Novel Continual Meta-Learning (CML) Framework: CoSFan, designed for continual adaptation of latent dynamics in time-series forecasting. It combines slow and fast adaptation mechanisms.

- What-How & When Framework:
  - What-How Meta-Model: Quickly adapts to specific tasks by identifying system dynamics (what) and generating task-specific parameters (how) using a feed-forward hyper-network.
  - Automatic Task Shift Detection: Identifies task boundaries to update the model for non-stationary distributions.

- Experience Replay Mechanisms:
  - Task-Aware Reservoir Sampling: Uses boundary detection to pseudo-label tasks for better context-query pairing.
  - Task-Relational Experience Replay: Clusters tasks using Bayesian Gaussian Mixture Models to handle frequent transitions and maintain rare tasks.

### Strengths
- Well written paper.
- Effectively combines feed-forward meta-learning with task-aware adaptation, addressing limitations in prior CML approaches reliant on gradient-based updates.
- Uses task-relational experience replay and GMM clustering to manage task transitions, a creative extension for handling non-stationary data.
- Robust empirical results validate CoSFan’s advantages, demonstrating reduced catastrophic forgetting and better adaptability than existing methods.
- Methodology is rigorous with well-defined metrics and detailed evaluation, covering key aspects of adaptation speed, memory usage, and performance retention.
- Generally clear presentation of the framework and methodology, with a structured breakdown of components ("what-how & when") aiding comprehension.
- Potential to influence further work in both CML frameworks and high-dimensional latent dynamics forecasting, with broader implications for adaptive AI in dynamic settings.
- The use of 5 seeds with average and standard deviation for all experimental results is greatly appreciated and confirms that these results are not simply a lucky seed; although the std remains very high on certain experiments.

### Weaknesses
- There is work to be done on the presentation of the figures (too small, need better explanations/more exhaustive captions, not clear/to many overlapping lines...)
---
- Limited to synthetic datasets (e.g., bouncing balls, Hamiltonian systems); adding real-world datasets (e.g., financial time series, climate data) would strengthen claims of applicability.
(Weakness acknowledged by the authors in limitations)
---
- The comparison with prior CML methods, especially MAML-based and task-agnostic approaches, lacks deeper discussion on specific performance trade-offs (e.g., speed vs. accuracy in adaptation).
- The added computational cost of GMM clustering and task-relational replay is not well-documented; detailing memory and processing requirements would help assess scalability.
---
- Assumes detectable task shifts and local stationarity; the model’s robustness to gradual or overlapping task shifts is not explored, limiting the generalizability to more fluid non-stationary environments.
- A quantification of how slow is to slow for the task shift to be picked up by the model would be interesting.
- Suggested improvements: include experiments with blurred task boundaries or transitions to assess flexibility in ambiguous scenarios.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a continual meta-learning framework (CoSFan) for forecasting high-dimensional time-series data generated by non-stationary distributions of dynamic systems. CoSFan addresses the limitations of traditional meta-learning approaches that assume stationary task distributions and known task identifiers. It proposes a feed-forward "what-how" meta-model to quickly infer the current dynamic system and adapt a latent dynamics function accordingly. Furthermore, it introduces mechanisms to detect task boundaries and employs task-aware and task-relational experience replay for continual adaptation of the model, mitigating catastrophic forgetting. Experiments on simulated physical systems demonstrate CoSFan's ability to learn and adapt to new tasks while retaining performance on previously seen ones, outperforming existing CML alternatives and standard latent dynamic models with continual extensions.

### Strengths
The paper is clearly written. Different components of the system and how they work together to form the final solution are clearly described. 

The experiment section provides lots of insight, with studies of individual components in the framework and comparisons with other alternative solutions. These ablations are especially important for a complex system.

### Weaknesses
The paper opts for using hypernet as the meta-learner. Although it ablates the hypernet solution to alternatives such as MAML and Bayesian gradient descent, it lacks the comparison with a sequence leaner, such as Transformer.

### Questions
1. Line 215, what is s in $T_j^s$? Could you add an explaination of what s is in the text?
2. Line 216, what's z? Could you add an explaination of what z is in the text?
3. Eq. 3, what's $l$? Is it summing over $l$?
4. Line 367, typo "withou".
5. Figure 2, why is it some method task aware replay performs better than full ER?
6. Table 1, could you add a description to what number represent in the caption? Could you add the unit in the table?
7. For task-relational buffers, why is the wrong cluster assigned? Is it because of the context embedding is not well-represented? If so, is it due to the meta-learning objective? What could be done to make it better?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposed a new method for continual meta-learning (CML). Different from previous methods (e.g. MAML variants), the proposed method uses a single feed-forward network. Using the contex encoder and hyper network, the support samples are encoded to context vector, and the context vector is used to produce the parameters for encoded query samples using the hyper network. Furthermore, the authors also proposed task-aware reservoir sampling approach by using the gaussian mixture model to identify the tasks. In the experiment section, the proposed methods outperforms the gradient-based meta learners, and also show the effectiveness of using task-aware sampling strategy.

### Strengths
Strengths

1. Different from the gradient-based methods which needs a number of gradient steps to adapt, the proposed methods can adapt to novel tasks using a sinlge feed-forward network.

### Weaknesses
Weaknesses

1. I think the motivation behind each component is weak. For example, for adapting to the query samples, the CosFan uses the hyper network to produce the parameters. However, why we should use the hyper network framework in this feed-forward network? Isn't it possible to use other trainalbe embedding networks for the query samples?

2. The mechanism for detecting the task boundary is not novel. The detection mechanism simply comes from the methods in [1]. However, though the authors show that the advantage of CosFan over other baselines lies on the ability for detecting the task boundary without any assumptions on the task identifiers, I think other baselines can adopt the task boundary detection mechanism used in this paper with simple modification. Therefore, I don't think the ability on detecting the task boundary in CosFan is advantage compared to other methods

3. The overall notation is confusing. For example, in line 217~220, what is the measing of superscript s in $T_j$? Is it support set? I think it is not clear. Furthermore, in the definition of $T_j$, the internal sequences are not dependent on $j$. I think the authors should clarify all the notations to prevent the confusion.


[1] Caccia et. al., Online Fast Adaptation and Knowledge Accumulation (osaka): A New Approach to Continual Learning, NeurIPS, 2020

### Questions
Already mentioned in weaknesses section

### Soundness
3

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
This paper introduces CoSFan, a method for meta continual learning that the paper evaluates on sequences of dynamical systems tasks. The approach consists of a hyper network based approach where the parameters of the current task being adapted to are generated using a forward pass of the hypernetwork on a context vector created using the average encoding of the samples in a context set. The generated parameters are used in a dynamics model that predicts the latent of the next element in the sequence and is then decoded. The meta-parameters and the encoder/decoder are optimized using the MSE between predicted and ground truth query sequences. The paper also explores how to detect task boundaries to manage/balance the number of samples in a replay buffer being used to rehearse on previous tasks in the sequence. It proposes two different mechanisms: one where it looks at any spike in loss as a task change, and one where it use a gaussian mixture model to cluster examples and detect whether any new clusters have formed.

The paper evaluates on a series of image based dynamical systems tasks where the physics/gravity constants are changed in each task, and the model must predict the next state. The paper shows an improvement in a slight improvement in learning performance over other meta learning approaches, but shows a clear improvement in forgetting over other approaches when using both their meta learning approach and the task detection based replay.

### Strengths
- Figure 5 shows a very clear use case for the task relational replay over even a task aware (and presumably task agnostic) replay.
- Figure 4 shows that the meta learning approach proposed by the paper helps mitigate forgetting when task detection is used.
- The method proposed can adapt to new sequences/new samples much faster than gradient based adaptation approaches, since they don’t need to take any gradient steps.

### Weaknesses
- The way Figures 2 and 4 are presented is a bit confusing. For one, they use different metrics, and it’s unclear what the insight from using one over the other is. For figure 2, the red bar is the only meta-learned method, so the other 3 methods were presumably evaluated in a different setting. Were they given the “context” information? It seems the more direct comparison with baselines is in Figure 4. 
- It’s unclear if the hypernetwork based approach can scale better than an adaptation based approach when the number of heterogeneous tasks increases. 
- As the authors do mention, the task shift detection seems to rely on abrupt task shifts, and would likely fail on gradual task shifts. In the case where the task shift detection fails, and the default is to task agnostic reservoir sampling, this method seems to do slightly worse than other meta learning based approaches.

### Questions
- For the experiments depicted in Figures 2 and 4, it seems each task is trained for the same number of samples. In this setting, why is there a big difference in the retained performance of specifically the meta learned methods between task agnostic and task aware replay? Wouldn’t the number of samples of each task still be approximately the same given that the replay size per task is still in the hundreds? Is the difference of a few samples really making that big of a difference?
- How does the hypernetwork based method scale with the number of heterogeneous tasks?
- I am a bit unclear about the setting. Is there a different context set sampled for each query example? Or is it just that the previous k sequences is treated as the context set?
- It would be interesting to see an adaptation based baseline which was also given the context vector as input.
- Do you have a version of Figure 3 comparing the Task aware replay with the task relational replay?

### Soundness
2

### Presentation
3

### Contribution
2
