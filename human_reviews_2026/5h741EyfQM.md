# A Brain-Inspired Gating Mechanism Unlocks Robust Computation in Spiking Neural Networks

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 8, 4, 2, 4

## Abstract
While spiking neural networks (SNNs) provide a biologically inspired and energy-efficient computational framework, their robustness and the dynamic advantages inherent to biological neurons remain significantly underutilized owing to oversimplified neuron models. In particular, conventional leaky integrate-and-fire (LIF) neurons often omit the dynamic conductance mechanisms inherent in biological neurons, thereby limiting their capacity to cope with noise and temporal variability. In this work, we revisit dynamic conductance from a functional perspective and uncover its intrinsic role as a bio-inspired gating mechanism that modulates information flow. Building on this insight, we introduce the Dynamic Gated Neuron~(DGN), a novel spiking unit in which membrane conductance evolves in response to neuronal activity, enabling selective input filtering and adaptive noise suppression. We provide a theoretical analysis showing that DGN possess enhanced stochastic stability compared to standard LIF models, with dynamic conductance intriguingly acting as a disturbance rejection mechanism. DGN-based SNNs demonstrate superior performance across extensive evaluations on anti-noise tasks and temporal-related benchmarks such as TIDIGITS and SHD, consistently exhibiting excellent robustness. To the best of our knowledge, for the first time, our results establish bio-inspired dynamic gating as a key mechanism for robust spike-based computation, providing not only theoretical guarantees but also strong empirical validations. This work thus paves the way for more resilient, efficient, and biologically inspired spiking neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces the Dynamic Gated Neuron (DGN), a novel spiking neuron model that incorporates dynamic membrane conductance as a biologically-motivated gating mechanism. DGN modulates membrane decay via activity-dependent conductance, enabling selective information retention and improved robustness to noise. The authors connect DGN behavior to gating in LSTMs, analyze stochastic stability, and present empirical results on many temporal datasets, showing improved accuracy and robustness under noise and adversarial attacks.

### Strengths
1. The work is significant, as it draws clear inspiration from neuroscience, building on established literature on activity-dependent ion-channel plasticity, to introduce an input-dependent gating mechanism into spiking neuron models. Different from LSTM in conventional deep learning, the proposed DGN model is biologically grounded and novel. It may have a significant multi-disciplinary impact.

2. This paper is well-written and easy to follow. It presents a well-structured bridge among conductance-based neurons in computational neurosience, LSTM in deep learning, and LIF neurons in SNNs, and clearly articulates the differences and connections among DGN, LSTM, and LIF, offering valuable multi-disciplinary insight.

3. The work is supported by rigorous theoretical analysis demonstrating how dynamic conductance mechanisms enable adaptive leakage and noise suppression. 

4. Experiments are extensive in this work, with strong results supporting the claims and demonstrating the strong temporal processing capability and robustness of DGN. 

5. The proposed model is also advantageous in parameter efficiency, which is particularly valuable for SNNs operating in edge applications.

### Weaknesses
1. The authors seem not to discuss the feasibility of deploying the DGN model on neuromorphic chips. Since the computational advantages of SNNs are primarily achieved on such hardware, it would be valuable to discuss this aspect.

2. The DGN model includes a numerical truncation function. However, it is unclear which specific type of truncation function is used in the experiments. What influence does the choice of truncation function have on performance? Are these functions hardware-friendly?

### Questions
It is not very clear how the DGN model is used to build both feedforward and recurrent networks. From Table 1, recurrent DGNs appear to outperform their feedforward counterparts. Could the authors clarify how the recurrent version is constructed and trained? Additionally, is the recurrent DGN more stable or easier to train compared to the feedforward architecture, and if so, why?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Dynamic Gated Neuron (DGN), a novel spiking neuron model that incorporates dynamic conductance mechanisms inspired by biological neurons. The authors propose that this mechanism acts as an intrinsic gating function, similar to forget gates in LSTMs, enabling adaptive control of information flow and memory retention. The model is rigorously evaluated on multiple speech and neuromorphic benchmarks, demonstrating state-of-the-art performance and superior robustness against noise and adversarial attacks. Theoretical analysis using stochastic differential equations further supports the noise resilience of DGN. The work is well-motivated, methodologically sound, and empirically thorough.

### Strengths
1. The idea of using dynamic conductance as a gating mechanism in spiking neurons is novel and well-grounded in neurophysiology. The connection to LSTM-like gating is insightful and bridges bio-inspired models with artificial neural networks.

2. The paper provides a solid theoretical analysis of the model’s stability under noise, using stochastic differential equations and variance comparisons with LIF neurons.

3. Extensive experiments on multiple datasets (TIDIGITS, SHD, SSC, etc.) across both feedforward and recurrent architectures demonstrate the model’s effectiveness in accuracy and robustness.

4. The paper includes a wide range of noise and adversarial attack scenarios, showing consistent superiority over existing SNN models and even LSTMs in some cases.

5. The appendix includes detailed derivations, training procedures, hyperparameters, and noise generation algorithms, which facilitate reproducibility.

### Weaknesses
1. Although the model is compared with several recent SNN variants, it would be beneficial to include more baselines  (e.g., spiking-based Transformers).

2. The DGN model introduces additional parameters and computational overhead compared to LIF. A more detailed analysis of energy efficiency or inference speed on neuromorphic hardware would strengthen the practical contribution.

3. Lack of pseudocode for model computation.

4. Why does DGN perform relatively poorly under the recurrent architecture on the SHD dataset?

### Questions
1. Could the dynamic conductance mechanism be integrated with other advanced SNN architectures (e.g., attention-based or transformer-like SNNs)?

2. Have you considered evaluating the model on neuromorphic hardware to assess its real-world efficiency and latency?

3. The paper mentions a "simplified DGN" (s-DGN) in the appendix. Could this be discussed more in the main text to highlight trade-offs between performance and complexity?

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
5

### Summary
This paper provides the Dynamic Gated Neuron (DGN), a novel spiking unit in which membrane conductance evolves in response to neuronal activity. DGN appears enhanced stochastic stability compared to standard LIF models.

### Strengths
This paper proposes an improved model. Intuitively, this model introduces more computation but does not increase the number of parameters, and experiments show improved performance. These findings suggest that the research presented in this paper represents a worthwhile improvement.

### Weaknesses
1. This paper is hard to follow.

2. Theoretical analysis is limited. There is a great deal of analysis work on SNNs that combats randomness and noise.

3. This paper proposes only one improved model. To my understanding, this model does not appear to have a special training mechanism; it simply uses the traditional global training method for SNNs. Based on past experience, this approach makes it difficult to achieve new functionalities. The authors' claimed intrinsic adjustment mechanism is merely a claim and cannot be verified experimentally or proven theoretically.

4. The experiments in this paper are insufficient; comparative experiments with the same architecture or parameter set are needed. Furthermore, I don't understand why it's necessary to compare it with artificial neural network models like LSTM that take real-number sequences as input.

### Questions
1. Does the improved model have a special or proprietary training mechanism that can find a solution with specific functions?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes the Dynamic Gated Neuron (DGN) model that integrates a bio-inspired dynamic conductance mechanism to enhance the robustness and performance of spiking neural networks (SNNs), outperforming traditional LIF models and other baselines in anti-noise tasks and temporal benchmarks.

### Strengths
1.  This paper introduces a biologically grounded Dynamic Gated Neuron (DGN) model with a dynamic conductance mechanism, which enables adaptive input filtering and noise suppression

2. The DGN-based spiking neural networks (SNNs) demonstrate superior performance and robustness across multiple benchmarks (e.g., TIDIGITS, SHD) under various noise types and adversarial attacks.

3. The comparison in the figures is very intuitive, allowing the innovative points of the authors' architecture to be clearly identified.

### Weaknesses
1.  The authors claim that the DGN is a generalized spiking neuron model, yet there is no evaluation on relevant computer vision datasets (NMNIST, CIFAR, N-CALTECH).

2.  This paper lacks an evaluation of energy consumption, and it remains unaddressed whether the newly introduced unit (DGN) will significantly increase power consumption, as reflected in Equations 5-8.

### Questions
The low power consumption of SNNs lies in simplifying multiplication to addition; however, in my view, D (as denoted in the DGN model) behaves like a floating-point number, which causes SNNs to lose this characteristic.

### Soundness
3

### Presentation
3

### Contribution
2
