# Autonomy-Aware Clustering: When Local Decisions Supersede Global Prescriptions

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 6, 0

## Abstract
Clustering arises in a wide range of problem formulations, yet most existing approaches assume that the entities under clustering are passive and strictly conform to their assigned groups. In reality, entities often exhibit local autonomy, overriding prescribed associations in ways not fully captured by feature representations. Such autonomy can substantially reshape clustering outcomes—altering cluster compositions, geometry, and cardinality—with significant downstream effects on inference and decision-making. We introduce autonomy-aware clustering, a reinforcement (RL) learning framework that learns and accounts for the influence of local autonomy without requiring prior knowledge of its form. Our approach integrates RL with a deterministic annealing (DA) procedure, where, to determine underlying clusters, DA naturally promotes exploration in early stages of annealing and transitions to exploitation later. We also show that the annealing procedure exhibits phase transitions that enable design of efficient annealing schedules. To further enhance adaptability, we propose the Adaptive Distance Estimation Network (ADEN), a transformer-based attention model that learns dependencies between entities and cluster representatives within the RL loop, accommodates variable-sized inputs and outputs, and enables knowledge transfer across diverse problem instances. Empirical results show that our framework closely aligns with underlying data dynamics: even without explicit autonomy models, it achieves solutions close (within $\sim$3–4\% gap) to the ground truth (where autonomy is known explicitly), whereas ignoring autonomy leads to substantially larger gaps ($\sim$35–40\%).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents the idea of autonomy-aware clustering to tackle real-world scenarios where entities deviate from the prescribed cluster assignments. By integrating reinforcement learning with deterministic annealing, the proposed method captures these autonomy effects, leading to satisfactory clustering results on the London Traffic dataset.

### Strengths
1. This paper proposes the entity autonomy problem in the existing passive clustering frameworks, which seems to be a practical research direction.
2. The proposed method is theoretically grounded, with detailed mathematical derivations and proofs.

### Weaknesses
1. In Fig. 1, it is not clear how such autonomy would occur in existing clustering methods. For example, for the given data, the classic K-means algorithm would not achieve such degraded results as illustrated in Figures c and d. I feel that a more intuitive example and explanation would help the readers to understand that the proposed autonomy problem is realistic and commonly encountered.
2. How could the proposed Adaptive Distance Estimation Network estimate the behavior of instances? In the fully supervised case, how can such estimations reflect real autonomy rather than random guesses? According to the experiments, the methods with or without prior autonomy knowledge only have a performance gap of 3-4%, which is a bit surprising. 
3. In fact, such a result may indicate that the dataset used for evaluation is not representative and complicated enough. It is not convincing whether the method could generalize to other scenarios, such as social networks and recommender systems, especially considering the hyperparameters required.
4. In Eq. P1, $\rho(i)$ is not explained.

### Questions
My major concerns lie in two aspects: i) more examples and discussions are needed to prove that such an autonomy problem widely exists in real-world applications; ii) it is a bit unreasonable that the model could produce results close to ground truth even without any priors on the entity autonomy. In other words, one could change the "ground truth" and the model would produce the same but "much worse" results. These concerns need to be addressed.

### Soundness
2

### Presentation
2

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
The paper introduces the novel concept of Autonomy-Aware Clustering, addressing the critical limitation in traditional clustering where entities are assumed passive and strictly conform to assignments. In reality, entities exhibit local autonomy ($p(k|j,i)$), probabilistically overriding prescribed cluster memberships, which can significantly alter cluster centers and composition. The framework tackles this challenge in two stages. First, for the case where autonomy models are known, it adapts Deterministic Annealing (DA), leveraging its advantages in handling non-convexity and initialization sensitivity. Second, for the practical, unknown-autonomy case, it formulates the problem as a unit-horizon Markov Decision Process (MDP) and proposes a Reinforcement Learning (RL) framework to jointly learn the assignment policy and cluster representatives. A key component is the Adaptive Distance Estimation Network (ADEN), a Transformer-based attention model used within the RL loop for model-free learning of dependencies. Empirical results demonstrate that the autonomy-aware framework produces effective results.

### Strengths
The most notable originality lies in the formal definition and robust solution for autonomy-aware clustering, explicitly accounting for stochastic entity behavior conditional on the policy. The quality is high due to the integrated methodological approach, which uses the strong theoretical basis of Deterministic Annealing (DA) and augments it with a practical Reinforcement Learning (RL) framework incorporating a Transformer-based ADEN for model-free learning. The work's significance is demonstrated by compelling results showing the method is highly accurate and, interestingly, can sometimes use the RL exploration to achieve up to a 10% improvement over the explicitly known-model solution by escaping local minima. The paper's clarity is excellent, making the sophisticated methodology accessible.

### Weaknesses
The framework's primary reliance on the DA formulation limits its direct generality to resource allocation or other clustering problems that can be represented by a distance to a cluster representative, excluding non-centroid-based clustering methods. The implementation of the ADEN to estimate the average cost within the RL loop may introduce complexity and potential instability in training that is not fully analyzed, compared to simpler clustering cost functions. While the empirical results are strong, the paper lacks a direct comparison of the ADEN/RL method against modern deep clustering baselines (e.g., autoencoder-based clustering) on the same dataset, making it difficult to isolate the contribution of the autonomy modeling aspect from the benefits of using a deep Transformer model generally.

### Questions
What is the practical stability and computational overhead of training the Transformer-based ADEN within the RL loop, especially for large $K$ clusters or very high-dimensional data, and did the authors observe sensitivity to ADEN initialization or hyperparameter choices? Is there a theoretical path to extend this framework to non-centroid-based clustering methods, such as those relying on density or connectivity, given the current formulation's dependence on cluster representatives $y_j$? Is the reported 10% performance gain over the known-model solution an isolated case, or can the authors provide statistics on how often the learning-based approach effectively escapes local minima across a broader range of problem instances?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces autonomy-aware clustering, a new formulation of the clustering problem that explicitly accounts for cases where individual entities may override their prescribed cluster assignments. For example, sensors sending data to a non-assigned base station or users acting outside expected preference groups. The authors formalize this idea by introducing a local autonomy model  that captures the probability of an entity originally assigned to cluster jj actually joining cluster kk. They first extend Deterministic Annealing clustering to this setting when the autonomy model is known, and then develop a reinforcement-learning-based approach (with a Transformer-based Adaptive Distance Estimation Network, ADEN) to learn the autonomy dynamics when unknown. Experiments on synthetic data and a decentralized sensing application (UAV placement over traffic sensors) show that the proposed method achieves near-optimal solutions even without explicit autonomy models, outperforming baselines that ignore autonomy.

### Strengths
* The paper raises a genuinely interesting and underexplored issue — that entities in clustering may act autonomously rather than passively following their assigned clusters. This perspective is both conceptually fresh and practically relevant, especially for systems with distributed agents (sensors, users, robots, etc.).
    
* Casting clustering with unknown autonomy as a one-step MDP is a clever move, allowing the use of RL techniques without explicit autonomy models.
    
* The synthetic and real-world experiments convincingly show that ignoring autonomy leads to significant degradation in performance, while the proposed model maintains small optimality gaps.

* The paper manages to combine analytical results (e.g., β-annealing behavior) with practical deep-learning components (ADEN) in a coherent framework.

### Weaknesses
- While the framing is fresh, it remains unclear how far this differs from existing probabilistic or soft-clustering models (e.g., mixture models, stochastic EM, or clustering with noisy assignments). In those methods, points also have probabilistic memberships, which may implicitly capture similar uncertainty. The paper could better clarify how “local autonomy” goes beyond mere stochasticity in assignments.
    
- The autonomy term is elegant mathematically but may be hard to interpret or estimate in real data. If autonomy stems from unobserved confounders, does the model risk overfitting to noise rather than uncovering meaningful autonomy?
    
- In many practical domains, deviations from assigned clusters could also be viewed as data noise, mislabeling, or temporary network failures, phenomena often handled by preprocessing rather than by modifying the clustering objective. The paper could better articulate when modelling autonomy is essential rather than optional.
    
* The RL + transformer-based approach (ADEN) seems quite heavy relative to the conceptual simplicity of the problem. The added learning complexity may limit adoption in standard clustering pipelines.

### Questions
1. How does this framework differ fundamentally from soft or probabilistic clustering methods (e.g., GMMs, fuzzy c-means) that already allow stochastic assignments?  
    What does the autonomy layer model that those do not?
    
2. In practice, how can one tell whether observed deviations reflect genuine _autonomy_ versus noisy or corrupted data?  
    
3. Does the method require retraining for every new dataset, or can the learned ADEN generalise across settings with similar autonomy patterns?
    
4. How sensitive is the method to the choice of annealing schedule or β-step size?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
## Summary

The paper presents autonomy-aware clustering, which allows entities (data points) to probabilistically deviate from their assigned clusters according to a probability distribution. 
It extends Maximum Entropy Principle–based Deterministic Annealing (MEP-DA) by local autonomy distributions into the objective function. When the autonomy distribution is known, the authors provide analytical update rules and analyze convergence behavior. When it is unknown, they propose a neural network (ADEN) to estimate average costs through supervised learning, alternating between model training and cluster updates until an annealing threshold is reached.  The authors evaluate their method on synthetic and real-world traffic data. The results indicate that accounting for local autonomy improves clustering performance compared to ignoring it.

## Contribution

While I have reservations about whether local autonomy is a good model for decentralized sensing, I acknowledge that it could be an interesting problem to study. The theoretical contribution seems solid, albeit incremental.

### Strengths
- Local entity autonomy could be an interesting problem, and decentralized sensing is a plausible scenario.
- The theoretical contribution seems solid, albeit incremental.

### Weaknesses
## Soundness

The paper makes several unsubstantiated claims and has major methodological flaws:

- The authors do not justify why local autonomy is a better model for decentralized sensing than other approaches.
- The authors claim to use Reinforcement Learning but this is not justified, in my opinion (see detailed discussion below).
- The MDP the authors claim to solve is never formalized.
- The experimental setup has major flaws: the problems are contrived and artificial, the baselines are not clearly defined.
- Some numerical results are missing (e.g., UDT19 dataset) and the empirical evaluation has major flaws. How can an algorithm achieve "10% improvement over ground truth"?
- The claim "These gaps can be further reduced through standard hyperparameter tuning and extended training" is not supported by any evidence.
- The problem formulation is presented as major contribution, but it is a straightforward extension of MEP-DA.

## Presentation

The presentation is poor and needs significant improvement:

- The writing is convoluted and hard to follow. The paper is full of jargon and undefined terms (e.g., "entity").
- The figures/tables are low quality and hard to read.
- The related work section is weak and does not contextualize the work well. Yes, other works do not use local autonomy, but how do they methodologically differ from this work? Why is local autonomy a better model?
- The conclusion is completely missing.
- Limitations are completely missing.

I want to discuss the Reinforcement Learning aspect in more detail, as it is a major red flag for me. From my understanding, the first loop in Algorithm 2 performs the following steps:

1. Assign new distances to all data points based on the current cluster centroids.
2. Sample assignments using the currently learned soft assignment distribution $\pi_\theta$.
3. Observe actual assignment outcomes using the autonomy distribution and update empirical costs.
4. Update an exponential moving average of the costs.
5. Update the neural network parameters $\theta$ using supervised learning to minimize the squared error between the predicted costs and the empirical costs.

This is not Reinforcement Learning. It does not use an MDP formulation. There is no action space, state space, transition dynamics, or reward function. There is also no temporal component: While the training loop is run repeatedly, there is no temporal difference learning despite the claim of a "straightforward Q-learning–style stochastic iterative update". Instead, Algorithm 2 is supervised learning with an alternating optimization scheme. The authors should clarify this, or correct my misunderstanding by formally defining the MDP they are solving and the Bellman equation they are using.

### Questions
- Why would I model the decentralized sensing problem with local autonomy instead of modeling interference/noise directly?  
- Can the authors clarify why Algorithm 2 is using Reinforcement Learning?  
- What MDP is Algorithm 2 solving? The authors should be able to formalize the action space, state space, transition dynamics, and reward function.  
- What baseline is used in the results? MEP-DA w/o the local autonomy distribution?
- In the results, the paper claims that "Notably, when $\kappa=0.1$ , $T = 0.1$, ADEN matches the performance of the model-based baseline (ground truth), and for $\kappa=0.5$, T = 0.1 it achieves approximately a 10% improvement over the ground truth, despite the absence of an explicit autonomy model".  
  - What is the model-based baseline (ground truth)? Algorithm 1 with the true autonomy distribution?  
  - How is it possible to outperform the ground truth?
- Where are the numerical results for the UDT19 dataset?  
- What are limitations of the proposed method? When would I use it over other methods and vice versa?

### Soundness
1

### Presentation
1

### Contribution
2
