# Variational Quantum Hypernetworks for Generalized Quantum Circuit Initialization

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 2

## Abstract
To the Program Chairs,

We are withdrawing our submission (ID: 23591) to incorporate significant revisions based on the reviewers' valuable feedback.

Since we plan to resubmit this work to another double-blind venue shortly, we are concerned that leaving the abstract publicly visible on OpenReview may compromise the anonymity of our future submission. Additionally, given the recent community discussions regarding the protection of pre-publication ideas, we explicitly request that the abstract text be removed or hidden from the public view.

We believe this measure is necessary to ensure fair review in our subsequent submission. Thank you for your support.

Sincerely, The Authors

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the trainability issues of variational quantum circuits, focusing on both barren plateaus (regions in the parameter landscape where gradients vanish exponentially with circuit depth or qubit number) and gradient explosion (gradients temporarily grow uncontrollably during early optimization). To address these challenges through better parameter initialization, the authors propose VQ-Hypernetwork, a meta-learning framework that generates depth- and problem-aware initializations for variational quantum algorithms. The method employs a GRU-based hypernetwork to produce well-conditioned parameters, aiming to stabilize gradient norms and improve convergence. Experiments on MaxCut and Heisenberg models demonstrate faster optimization and more stable training dynamics compared with standard Gaussian or uniform initializations.

### Strengths
1. The paper introduces the concept of gradient explosion in variational quantum algorithms, which is rarely discussed compared to the well-known barren plateau issue. This fresh angle adds a new and underexplored dimension to the landscape of quantum optimization challenges.

2. The experiments are thorough and thoughtfully designed, comparing multiple initialization strategies across different problem types. Particularly interesting is the visualization of optimization trajectories, which provides intuitive insights into how different initializations influence training dynamics.

### Weaknesses
1. While the paper identifies transient gradient growth at the beginning of training, it remains unclear whether this phenomenon truly constitutes an explosion analogous to that in classical deep networks. In variational quantum algorithms, gradients are inherently bounded by the parameter-shift rule, and their magnitudes typically scale polynomially with circuit size. With a suitably small learning rate, which can be chosen to scale inversely with the circuit size or measurement variance, such temporary spikes may not pose a genuine optimization problem. The authors should clarify why these transient increases lead to meaningful instability, rather than being benign artifacts of initialization or optimizer dynamics.

2. According to line 124, Zhang et al. (2022) defines a depth-aware Gaussian initialization by scaling the variance inversely with circuit depth. However, the released code provided in line 232 shows that the authors instead scaled the standard deviation as 1/depth. This implies that the actual variance scales as 1/depth^2, which excessively compresses parameter amplitudes for deeper circuits. As a result, the Gaussian baseline is incorrectly implemented, and all related experimental comparisons are potentially invalidated.

3. Several numerical values in Table 2 (Heisenberg results) are identical to those in Table 1 (MaxCut). This repetition is too systematic to be random yet too obvious to imply intentional manipulation. It most likely results from a copy-paste oversight during manuscript preparation. Nevertheless, it undermines confidence in the experimental reporting and suggests insufficient validation or proofreading of results.



4. The proposed VQ-Hypernetwork extends the idea of classical hypernetworks (Ha et al., 2016) to the quantum setting by generating initialization parameters for variational circuits conditioned on depth and task. While this adaptation is well-motivated, it remains conceptually close to existing meta-learning paradigms. The paper would benefit from clarifying what specific quantum inductive biases or architectural constraints make this formulation unique, beyond being a straightforward application of an existing idea to a new domain.

### Questions
1. Can the authors provide quantitative evidence that the transient gradient growth they observe actually harms optimization performance? For example, do learning curves diverge or oscillate without stabilization? Since gradients in VQAs are bounded by the parameter-shift rule, it would be useful to show whether the phenomenon persists under smaller learning rates or alternative optimizers.

2. Given the mismatch between the theoretical definition of Gaussian initialization (1/depth scaled variance) and the implemented version (1/depth scaled std), the authors should re-run the affected experiments using the correct scaling and update the corresponding figures and tables. It would be important to clarify in the revision whether the claimed “gradient explosion” remains observable after this correction.

3. The duplicated numerical values between Table 1 and Table 2 suggest a likely copy-paste oversight. The authors should re-report the corrected numerical results in the main text in the updated version.

4. How does the proposed VQ-Hypernetwork differ, in principle or in effect, from a standard hypernetwork? In other words, what inductive biases, if any, arise from the quantum structure itself rather than the classical meta-learning component?

5. Since the VQ-Hypernetwork is trained on specific depths and problem instances, can it generalize to unseen circuit sizes or Hamiltonians? Reporting such transfer or extrapolation results would help assess the broader applicability of the approach.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper identifies that both vanishing and exploding gradients hinder convergence of variational quantum algorithms and proposes VQ-Hypernetworks, a meta-learning framework that generates initial parameters via classical pre-training, taking into account the quantum circuit architecture which utilizes these parameters. The approach is meta-trained on 450 problem instances and demonstrates benefits on MaxCut and XYZ Heisenberg models across systems of 8-20 qubits and circuit depths of 4-20 layers, showing improved gradient stability and convergence compared to Gaussian and Uniform initialization on 5 validation samples. As a foundational enabling layer for a broader VQA ecosystem this VQ-Hypernetwork is claimed to condition the optimization landscape for more effective training.

### Strengths
The paper points out an often overlooked component for successful training in QML: gradient explosion and identifies parameter initialization as a potential way to improve the convergence of the quantum training phase. It utilizes figures extensively to visualize the benefits of the proposed classical pre-training.

### Weaknesses
The assessment of the proposed strategy is purely empirical and the claims regarding the benefits are based on 2 problem types, resulting in a narrow experimental scope. Theoretical justification for why such a strategy might be beneficial in general, along with more examples on real-world datasets and rigorous statistical analysis, would increase the credibility of the claimed contributions. The comparison baseline consists of only two simple standard initialisation strategies, but doesn't include strategies such as the mentioned Q-MAML, problem-specific warm-starts, or other more advanced heuristics. The cost analysis does not account for the classical pre-training phase. The discussion does not include topics such as applicability to larger systems with more than 20 qubits, the effects of noise in the NISQ era, or the performance of the parameter initialisation on datasets outside of the domain and distribution of the training dataset.

### Questions
How does this strategy perform across data domains?
What is the computational cost of the classical pre-training phase in terms of quantum circuit evaluations, and how many new problem instances must be solved to amortise this cost?
Does the need for classical pre-training limit this approach to problem sizes where classical simulation is still feasible, thereby undermining its applicability in the quantum advantage regime?
How do realistic NISQ noise models (gate errors, decoherence, measurement noise) affect the benefits of the pre-training, given that all experiments were performed on noiseless simulators?
Does the approach generalize to Ansätze beyond hardware-efficient circuits?
What is the minimum training set size required for effective meta-learning?

### Soundness
2

### Presentation
2

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
This paper introduces a meta-learning based method to mitigate trainability isses of VQA algorithms; specifically a classical meta-learner is introduced, called VQ-hypernetwork, which generates problem-specific intital parameters.
The network does so given problem descriptions and the circuit depth on input. 

The authors analyze the performance numerically for a couple of task families (MaxCut, Heisenberg model), and compare vanilla initializations (Gaussian, Uniform) performance versus their own, at various qubit numbers and depths (spaced in increments of 4 from 4 to 20). The performance of their method seems t avoid some of the concentration issues better. The idea is good, but I have serious concerns about the underlying theory and prospects when large scaling is needed.

### Strengths
In this work, the authors tried to address the issue of transient gradient explosion in training a variantion quantum circuits.This is an important question.
In a novel step, instead of focusing on addressing BP, they are more interested in this often-overlooked problem of "transient gradient explosion" (perhaps this is what they believe the most novel aspect is). They propose a so-called VQ-Hypernetwork to initialize some problem-aware and depth-aware parameters for VQAs, and with their numerics they showcases the advantage of this framwork over other parameter initialization methods.

### Weaknesses
Lack of clarity: after inspecting this manuscript, I haven’t found any detailed mathematical framework or expressions for the followingsthe structure of the GRU, the structure of the hyper network (in Eq.2, what is the funtion \theta(\Phi)?), the gradient of the meta-objective, the anstza and training method used for the VQA, etc.

The authors in part base their novelty on "gradient explosion" yet no definition is given. Barren plateaus are a mathematically precisely defined concept. I am very confused about this as due to unitarity I don't see how gradients could explode at all (see also neural nets, versus orthogonal nets).

Another issue is that the experiments that are provided are in my view not sufficient to convincingly demonstate the effectiveness of their VQ-Hypernetwork. More experiments of other model could be done to validate their methods. Also, one example but not at least, more experiments with different qubit number should be done when they want to show the training instability is systematically exacerbated by increases in the number of qubits.
Allow me to defend this last point: training models at 20 or so qubits can be done as although gradients may vanish, they are still not negligible at these sizes. I am very concerned how any training and any of this would work at relevant sizes where the models become intractable, and the gradients actually vanish.

### Questions
1. What is the structure of the VQ-Hypernetwork and how do you train it? 
2. The author said the gradient of the meta-objective is computed via chain rule, do you have a mathematical expression? And what is the complexity of calculating your gradient? Can your method scale to 100 qubit size? How much time does the meta-training take (and scale with n)? 
3. In Figure 3 and Table1&2, it seems that, on average, the VQA with depth20 have a better approximation of ground state energy at the end (200 step) with Gaussian initialization method than VQ-Hypernetwork. Why and does this suggest your VQ-Hypernetwork possibly not work for large systems?
4. What is the definition of transient gradient explosion

### Soundness
2

### Presentation
2

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
The paper tackles an often-overlooked trainability pathology in VQAs, gradient explosions. It proposes a VQ-Hypernetwork, a classical GRU-based meta-learner that, conditioned on problem features and target circuit depth, one-shot generates initial parameters for the variational circuit. The hypernetwork is trained end-to-end (PennyLane + PyTorch) to minimize initial expected energy over batches of tasks; at inference, it serves as a learned initializer after which a standard VQA optimizer updates only θ.

### Strengths
1. Focusing on gradient explosion complements the community’s emphasis on vanishing gradients and broadens the trainability discussion for deep VQAs.
2. A one-shot initializer that conditions on both task structure and circuit depth is practically appealing and compatible with existing VQA workflows.

### Weaknesses
1. The paper claims to mitigate gradient explosion, yet the quantitative baselines are primarily initialization schemes (Gaussian, uniform/reduced-domain) historically motivated by vanishing-gradient mitigation. To substantiate the “explosion” claim, comparisons should include explosion-oriented controls such as gradient clipping, norm-based rescaling, adaptive per-layer step sizes, trust-region / line search, and Quantum Natural Gradient (QNG) steps for the first few iterations. Without these, the evidence for specifically taming the explosion (versus merely “better seeding”) is less convincing.
2. The paper stops short of explaining why the learned initializer suppresses spikes. Given the modest system sizes in simulation, some formal guarantees or stylized analysis would meaningfully strengthen the claim.
3. Evaluation is confined to MaxCut and 1D XYZ with a single hypernetwork design (GRU). It would help to (i) test additional ansatz families, (ii) add architecture ablations to verify that the depth-aware, sequential generation is the key ingredient.

### Questions
1. How does the initializer perform on out-of-distribution instances?
2. How essential is the GRU and the explicit depth conditioning? What happens with a depth-agnostic MLP or a Transformer hypernetwork?
3. Did you use PennyLane’s noisy backends in any experiments? If yes, which noise channels/parameters and shot counts? How do results compare to the noiseless/statevector setting?
4. Please specify the gate templates per layer and discuss portability to other ansatze.

### Soundness
1

### Presentation
2

### Contribution
2
