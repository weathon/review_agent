# Towards a Unified View of Neuron Interpretation and Behavior Control in Large Language Models

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
Existing works in neuron interpretations and behavior control in Large Language Models are largely developed independently of each other. On one hand, the pioneering works in neuron interpretation rely on training sparse autoencoders (SAE) to extract interpretable concepts. However, interventions on these concepts are shown to be less effective in model behavior control. On the other hand, dedicated behavior control approaches rely on adding a steering vector to the neurons during the model inference, while ignoring the aspect of interpretation. In this work, we present a unified framework that establishes connections between them, which is crucial to truly understand the model behavior via interpretable internal representations. Compared to existing SAE based interpretation frameworks, the unified framework not only enables effective behavior control, but also uniquely allows flexible user-friendly concept specification and maintains the model performance. Compared to dedicated behavior control approaches, we guarantee the steering effect in behavior control while additionally explaining which concept has how much contribution to the steering process and the roles of them in explaining the to-be-steered neurons. Our work sheds light on designing better interpretation frameworks that explicitly consider the aspect of control during the interpretation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a linear framework intended to unify interpretability and controllability in large language models. It collects hidden activations from model responses to selected example prompts and treats each collected activation vector as a “concept,” forming a concept matrix. It then learns a linear decoder that maps any new hidden state at a chosen layer into coefficients over these concepts.  Those coefficients indicate which behaviours are active in that state, and that modifying the coefficients and reconstructing the activation lets them steer the model’s behaviour at inference time.

### Strengths
The paper offers a single linear pipeline that links interpretation and control, which is conceptually clean.

The reconstructed activations can be swapped in with little or no perplexity increase and extremely low reconstruction error, suggesting the method can intervene without obviously breaking the model.

The approach exposes a more human facing control surface than typical steering methods: instead of adjusting opaque neuron groups or unlabeled latent directions, the user is effectively turning specific behaviours up or down using concepts derived from natural-language examples

### Weaknesses
The notion of a “concept” is weakly grounded. Instead of learning disentangled latent factors from activations (as in dictionary learning / SAE work), the method simply takes raw activation vectors from prompted examples and declares each one to be a concept. The sentences are likely to be polysemantic bundles of behaviour, not clean features. Calling this “concept discovery” or “interpretability” stretches the standard meaning of those terms in the cited literature.

The reported errors seem high for modern SAEs. Since the overall objective is different (see previous point); therefore, it does not make sense to lean into it.
	

The experiments regarding steering are narrow and should be expanded to evaluate a broader set of behaviours in benchmarks such as AXBENCH, which the authors have cited.

### Questions
Please clarify how the SAE reconstruction error is computed in Table 1? Are these normalised per-dimension/per-token average?

### Soundness
2

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
This paper introduces a unified framework to integrate the interpretation of neurons with the control of behavior in LLMs. The goal is to overcome the shortcomings of current Sparse Autoencoder-based interpretability frameworks (e.g., imprecise reconstruction, uninterpretable bias terms and ineffective control) while providing the explanatory power often absent in pure behavioral control techniques. Experimental results demonstrate that the SVR features significantly outperform baseline methods such as Sparse Autoencoders (SAE) and Attention Steering (AS) across multiple behavioral control tasks.

### Strengths
1.Clear and important motivation: This paper clearly articulates a crucial gap in current LLM neuron interpretability and controllability research. The proposed vision of a unified framework holds significant academic merit.
2.Methodological Novelty: The unified framework is novel and insightful. By changing the autoencoder's objective function—from reconstructing raw activations (SAE) to reconstructing task-specific steering vectors—it successfully integrates the supervisory signal of the control task into the feature learning process, which is the key to achieving highly controllable features.
3.Comprehensive Interpretability Analysis: The paper extends beyond control efficacy by providing robust interpretability analyses, including attribution analysis of SVR features and the identification of "Steerable Neurons" linked to specific behaviors (e.g., safety refusal), thereby fulfilling its promise of a "unified view."

### Weaknesses
1.This optimization problem solves for the L2-minimum norm solution for $W_{dec}$. L2 minimization tends to produce many small weights, not sparse weights. The paper claims this makes coefficients more selective and justifies this with only a single case study (0.1734 $\rightarrow$ -0.0028) in Section 5.3. This is weak. Why would the L2 solution systematically produce more interpretable coefficients than other solutions? The authors do not provide sufficient theoretical or experimental justification.
2.Infinity of solutions: As the paper acknowledges (Sec 4.1 and Appendix G), when the number of concepts $c > n$ , there are infinitely many combinations of $C$ and $W_{dec}$ that satisfy $C W_{dec} = \mathbb{I}$. The paper's choice of $W_{dec} = C^{T}(CC^{T})^{-1}$ is just one of these infinite solutions. This means the coefficients $W_{dec}x$ is entirely dependent on the user's choice of $C$ and this specific $W_{dec}$ solution. This seems more like a projection, projecting the neuron's activation into a user-specified concept basis, rather than truly discovering the model's internal computational mechanism. How do we know this explanation is what the model is really doing, versus a mathematically equivalent form we have imposed on it? This may conflict with the goal of neuron interpretability.
3.The framework's precise reconstruction relies on the number of concepts $c$ being greater than or equal to the neuron dimension $n$. For LLMs (e.g., Llama 3 8B, $n=4096$), this implies the user must provide at least 4096 (linearly independent) concept sentences. This is completely unrealistic in practice. The advantage of user-friendly and flexible specification becomes meaningless under the $c \ge n$ requirement.

### Questions
1.Can you provide broader evidence (except the case in Sec 5.3) that this L2 solution systematically yields more interpretable results than the other (infinitely many) solutions to $C W_{dec} = \mathbb{I}$?
2.The framework's guarantees rely on $c \ge n$. However, in a practical application, a user might only care about a small number of concepts (e.g., $c=10 \sim 20$), which is far less than $n=4096$. In this $c \ll n$ case (where $C W_{dec} = \mathbb{I}$ cannot be satisfied), how does the framework perform?
3.How sensitive is the entire interpretation to the specific choice of concepts in $C$? If I swap one "random" concept sentence in $C$ for another, do the interpretation coefficients change drastically? Does this imply the explanations themselves are unstable?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a unified framework aimed at bridging neuron interpretation and behavior control in Large Language Models (LLMs). The framework ensures that the decomposition of input neurons can be reconstructed precisely while achieving control effects comparable to existing behavior-control approaches. Empirical results demonstrate that the proposed method achieves lower reconstruction error and maintains equivalent behavior-control performance relative to prior work.

### Strengths
1. The paper is clearly presented and well-structured.
2. The idea of unifying neuron interpretation and behavior control within a single framework is appealing and potentially impactful.
3. The proposed approach is compatible with existing sparse SAE-based methods.

### Weaknesses
1. The rank requirement of the concept matrix implies that the number of concepts must exceed the number of intermediate activations. This condition may limit the practical scalability of the method in real-world scenarios.
2. The evaluation is conducted on a relatively small LLM (8B parameters), making it unclear whether the approach scales to larger foundation models commonly used in practice.
3. Experiments are limited to some specific LLM layer, and the rationale for this choice is not sufficiently discussed. It remains uncertain whether the findings generalize across different layers.

### Questions
1. Does the proposed framework introduce additional computational overhead compared with prior neuron-interpretation or behavior-control methods?
2. What are the key limitations of the proposed approach, particularly with respect to scalability and practical deployment?

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
2

### Summary
This paper proposes a unified framework that bridges two previously separate research directions in large language models (LLMs): neuron interpretation and behavior control. Traditional neuron interpretation methods use sparse autoencoders (SAEs) to extract interpretable concepts but fail to effectively influence model behavior, while behavior control techniques steer model outputs through vector interventions without interpretability. The proposed framework connects these two perspectives, enabling both interpretable understanding and effective control of internal neuron representations. It allows flexible, user-friendly specification of semantic concepts, preserves model performance, and quantifies the contribution of each concept to the steering process. Overall, the work provides a principled foundation for designing interpretation methods that inherently incorporate controllability in LLMs.

### Strengths
The proposed framework appears novel, and its motivation is clearly articulated. The paper is generally well written, and Figure 2 provides a helpful visualization of the overall design, aiding readers' understanding of the main idea.

### Weaknesses
Some components of the proposed design could be explained in greater detail for clarity. Additionally, the paper lacks a discussion or comparison of computational complexity and runtime efficiency, which would help assess the practical feasibility of the approach.

### Questions
1. As mentioned in Appendix E, the concept matrix $C$ can be quite large (e.g., $16384 \times 512$). Computing its pseudo-inverse to obtain $W_{dec}$ (Eq. (9)) may be computationally expensive. Could this become a bottleneck in practice? If so, how to address or mitigate this issue?  
2. In the lower half of Figure 2, the user-specified descriptions are passed through a language model to construct the concept matrix. Does this language model require specific training or fine-tuning, or can any pre-trained language model be used directly here?  
3. Related to the above two questions, it would be helpful to provide more detail on how users specify the concept matrix $C$. The current description references prior works but does not sufficiently explain the step-by-step process of constructing $C$, which makes it difficult for readers unfamiliar with those references to follow.

### Soundness
3

### Presentation
3

### Contribution
3
