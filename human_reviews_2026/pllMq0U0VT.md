# Evolutionary Emergence of Neurodynamic Networks for Robust Control: A Simple Excitatory-Inhibitory Network

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Fine-grained network models based on differential equations,  and neurodynamic synapses and neurons provide a realistic description of biological neuronal networks, compared with mainstream artificial neural networks.  They nevertheless have not been widely explored, mainly due to the lack of effective parameter training methods.  We propose a neurodynamic model training method that combines an efficient neurodynamic simulation architecture and an evolutionary algorithm.  Based on a simple Excitatory-Inhibitory network, a neurodynamic model with task control capabilities is successfully obtained via parallel dynamic simulation, and network selection methods under evolutionary pressure.  Compared with the state-of-the-art reinforcement learning methods, the resulting neurodynamic network can achieve comparable task control performance for Mojoco tasks in a significantly smaller network scale within fewer training steps.  Our work provides an alternative path to functional networks alongside mainstream reinforcement learning frameworks, and prove the feasibility of the evolutionary approach toward biological intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes training small "neurodynamic" models, i.e., rate-based neurons governed by systems of differential equations, obeying Dale's Law with an evolutionary framework to solve MuJoCo tasks. The results are compared to spiking neural networks and multi-layer perceptrons.

### Strengths
1. The distributed training scheme takes good advantage of the combination of small networks and evolutionary framework.
2. The networks can obtain improved reward scores on the MuJoCo tasks and are sometimes competitive with, e.g., multi-layer perceptrons.

### Weaknesses
1. Both MuJoCo simple and complex tasks are, in my view, rather unsophisticated by modern standards. Although I fully appreciate the purpose of this paper being to investigate new techniques, I would feel more confident in the potential promise of these techniques if they could be demonstrated on more naturalistic and complex tasks, e.g., including perception in simulation environments like IsaacSim.
2. The network sizes are only experimented with up to 55 neurons. Given the relative computational simplicity of rate-based networks, I am curious why the tested number wasn't higher. Related to this, I could not find an analysis comparing the FLOPs of the different models compared, which would aid in making comparative claims.
3. There are many small typos, e.g., L284, and instances where in-text citations are not formatted neatly, e.g., L360.

### Questions
1. How does your method compare to similar bio-evolutionary strategies like presented in Najarro E, Risi S. Proc 33rd Conf Neural Inf Process Systems (NeurIPS 2020). 2020: 20719–20731, 2020?
2. Related to 1, is the motivation to use a "neurodynamic" model or impose Dale's Law similar to the suggestions made in Burns, T. F. (2021). Classic Hebbian learning endows feed-forward networks with sufficient adaptability in challenging reinforcement learning tasks. Journal of Neurophysiology, 125(6), 2034-2037?
3. Figure 2B seems to 'assert' the fitness landscape without direct measurement. Is this correct? If so, could you replace the illustration with empirical or analytical results?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an excitatory-inhibitory BNN combined with an evolutionary algorithm model. For specific tasks, it achieves performance comparable to state-of-the-art reinforcement learning algorithms using fewer neurons and fewer training steps. Furthermore, by comparing with RNNs of the same size, it demonstrates that this model can evolve certain network structures during the evolutionary process. This work provides a biologically inspired, interpretable approach for functional networks and mainstream reinforcement learning frameworks.

### Strengths
* Originality: This paper innovatively combines BNNs with evolutionary algorithms, introduces a novel noise input current term in LIF neuron inputs to simulate random perturbations in real biological neurons, and incorporates synaptic delay as a new parameter.
* Quality: The paper compares the performance of six models across eight tasks to support its conclusions, conducts correlation analyses to substantiate subsequent findings, and references extensive additional literature.
* Clarity: The paper is clearly written and logically structured, with effective use of figures and tables.
* Significance: By introducing novel inputs and network parameters, the paper enhances biological interoperability in detail, potentially offering insights for research in explainable artificial intelligence.

### Weaknesses
* The BNN's performance on the InvertedDoublePendulum and Swimmer tasks was significantly inferior to that of the MLP, yet the paper offers no explanation for this discrepancy, which appears to against the conclusion of "comparable performance."
* The article conducted correlation analysis only on the CartPole task, rendering its conclusions somewhat arbitrary.

### Questions
1. Why did the BNN perform significantly worse than the MLP on the Inverted Double Pendulum and Swimmer tasks?
2. Does the conclusion that the BNN develops correlations between neurons still hold true for tasks more complex than CartPole?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors proposes a biological neurodynamic network (BNN) trained via evolutionary algorithms (CMA-ES and NES), demonstrating its performance on standard MuJoCo reinforcement learning tasks. The authors argue that such networks can achieve comparable performance to gradient-based reinforcement learning methods with smaller network sizes and fewer optimization steps. They frame this work as a proof of principle, emphasizing the biological interpretability and potential for scaling toward more complex brain-like networks.

### Strengths
- The combination of neurodynamic modeling and evolutionary optimization is conceptually interesting and biologically inspired.
- The inclusion of excitatory-inhibitory (E-I) balance, noisy inputs, and synaptic delays is a meaningful attempt to increase biological realism.
- The description of the distributed EA training setup and simulation framework is detailed and technically sound.
-  The discussion on evolved structure and network correlation provides an interesting link between learning dynamics and biological structure formation.

### Weaknesses
1. There are multiple formatting errors, especially missing whitespace before citations (e.g., “activities,Hodgkin & Huxley (1952)” )
2. The authors repeatedly claim to have achieved “state-of-the-art performance”  despite the results showing significantly lower returns than standard baselines. For example, in the Ant task, the BNN achieves ~950–973 reward while SOTA PPO results are much higher.  This discrepancy undermines the “comparable performance” claim.
3. The evaluation is confined to small networks (20 neurons) and limited environments. The authors acknowledge this, but the assertion that scaling up will yield better results is speculative and untested.

### Questions
- Does the robot learn to move effectively in the ant domain? 
- How sensitive are the results to network size and hyperparameters (e.g., number of neurons, mutation rates)?
- Did the authors compare the same computational budget (wall-clock or FLOPs) with gradient-based methods?
- How reproducible are the results given stochastic processes? How many independent runs were performed?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose an approach to overcome the limitations of training biologically realistic neural network models. These models posses properties that may make them more interesting than standard models trained via back propagation, but their non-differentiable nature makes them hard to fit in RL settings. Their approach combines several ingredients, which is a critical element of the paper:

1. Evoluationary algorithms as the optimization method: They combine standard methods (CMA-ES, NES) with an island + migration method (i.e. one where subpopulations are maintained and only allowed to share individual at certain intervals).
2. A Leaky-Integrate-and-Fire (LIF) model of neuron with a noisy inputs.
3. A synaptic model which not only tracks its weight but also a delay value.
4. Excitatory-Inhibitory balance: neurons in a network are assigned to either an excitatory or inhibitory set, a common property in biological neural networks.

The approach is evaluated on a subset of the standard continuous control tasks in OpenAI Gym. The results show that the authors' approach is competitive against the proposed baselines.

### Strengths
I believe the motivation is good. Making biologically plausible network appealing to researchers in AI, and even to computational neuroscientists who would like to scale them up to more demanding settings.

### Weaknesses
There are a few, critical weaknesses that make it hard for me to recommend this paper:

1. While I am not opposed to researchers just combining different ingredients and showing that a particular combination solves a longstanding issue, it is also imperative that said researchers show how the different components contribute to their results. In the case of the present work, the authors need to provide ablations to justify their decisions.
2. Failing that, is there some question that can be answered with this framework that couldn't be answered before? If so, what is this question and what have we learned? Right now, this feels more like "we did it because we can", which is just not enough.
3. There is a substantial literature that studies how to combine biologically plausible models of neural networks with evolutionary algorithms which they authors need to cite and explain how their approach is different.
    * For example, [1] is very similar to the current approach and [2] explores the potential role of delay constants, while [3] also explores other benefits like adaptation when recovering from damage.
    * NOTE: these are probably not the only ones that are relevant. There is whole body of literature on using EA + biologically plausible ANNs in different settings. The authors need to perform a better review of the existing literature
4. Other issues with wording. For example the authors say they introduce at several points (e.g. when taking about the ENLARGE framework and the EAs) yet they are not introducing these approaches, they are applying them. It confused me at one point because I thought they were going to present a variant of these but this was not the case.

References:

1. G. Shen, D. Zhao, Y. Dong, & Y. Zeng, Brain-inspired neural circuit evolution for spiking neural networks, Proc. Natl. Acad. Sci. U.S.A. 120 (39) e2218173120, https://doi.org/10.1073/pnas.2218173120 (2023).
2. Habashy, K. G., Evans, B. D., Goodman, D. F., & Bowers, J. S. (2024). Adapting to time: Why nature may have evolved a diverse set of neurons. PLOS Computational Biology, 20(12), e1012673.
3. Najarro, E., & Risi, S. (2020). Meta-learning through hebbian plasticity in random networks. Advances in Neural Information Processing Systems, 33, 20719-20731.

### Questions
Some questions related to the points above:
* E-I balance is realistic, but why is it needed to perform in some AI task?
* Why is a delay constant interesting?
* What happens if they don't use an island + migration setting?
* How does the approach build on previous work which the authors have failed to mention?

### Soundness
2

### Presentation
2

### Contribution
1
