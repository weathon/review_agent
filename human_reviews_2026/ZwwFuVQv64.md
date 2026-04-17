# Layerwise Federated Learning for Heterogeneous Quantum Clients using Quorus

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Quantum machine learning (QML) holds the promise to solve classically intractable problems, but, as critical data can be fragmented across private clients, there is a need for distributed QML in a quantum federated learning (QFL) format. However, the quantum computers that different clients have access to can be error-prone and have heterogeneous error properties, requiring them to run circuits of different depths. We propose a novel solution to this QFL problem, Quorus, that utilizes a layerwise loss function for effective training of varying-depth quantum models, which allows clients to choose models for high-fidelity output based on their individual capacity. Quorus also presents various model designs based on client needs that optimize for shot budget, qubit count, midcircuit measurement, and optimization space. Our simulation and real-hardware results show the promise of Quorus: it increases the magnitude of gradients of higher depth clients and improves testing accuracy by 12.4% on average over the state-of-the-art.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a method to address the problem of different architectures and behaviors of quantum computers in a quantum federated learning scheme. They use layerwise loss to aggregate the parameters of local QNNs with specific approaches, gaining performance improvements. The proposed method is examined on real quantum computers.

### Strengths
1. Using layerwise losses for QFL is a novel approach. 
2. The paper is well-written overall.

### Weaknesses
1. The proposed method appears limited to binary classification problems. 
2. Line 291 states "we design an ansatz where it is possible to obtain the outputs from all layers in a single shot," but it's unclear how the ansatz enables single-shot behavior.
3. The baselines compared are only QML models, no classical baseline is included. It is plausible that classical models could easily outperform QML models on binary classification problems.

### Questions
1. In line 265, "Passing the state unchanged thus requires you to prepare another copy of it, which induces additional shot overhead that is linear in the number of layers and is a nontrivial cost" → does this mean state tomography is required in general? And how is this linear in the number of layers? 
2. Following W2, how does the ansatz enable the single-shot behavior? Is there any proof? From line 296, "we evaluate each layer's outputs by computing the marginal distribution on its ancilla" → how can you calculate the "distribution" with a single shot?
3. Does the parameter aggregation process, which aligns parameters for models with different layers, affect the expressiveness? For example, a quantum state produces output, becoming an intermediate state of another model. And this "another model," after additional layers, will produce the output of the same task. Then why not just use the shallower model if a shallow circuit is enough for such a task?

### Soundness
3

### Presentation
2

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
The paper presents a federated learning framework to train quantum clients with different circuit depths by applying a layerwise loss (pairwise KL coupling and circular aggregation of rotation parameters). The authors claim up to a 12.4 percent accuracy gain over “Q-HeteroFL” on binary classification of MNIST and Fashion-MNIST datasets. The manuscript also presents results across multiple IBM QPUs.

### Strengths
The problem formulation and framing are clear, focusing on depth heterogeneity in QFL. The concrete FL procedure with circular angle averaging, parameter slicing by depth, and layerwise loss is an impressive methodology.

Circuit engineering is thoughtful. The designs cover orthogonal resource constraints such as shots, ancillae, and mid-circuit measurement, and they are practical on current hardware. The Funnel variant is particularly pragmatic. 

Validation on real quantum hardware is a core strength.

### Weaknesses
Major Weaknesses: 
The evaluation scope is narrow. Only binary classification is used, and only on MNIST and Fashion-MNIST (both grayscale, 28×28 toy image datasets). No multiclass tasks, no quantum-native datasets, and missing diversity (especially in a paper that lacks theoretical proofs) weaken the authors’ claims. These datasets are fairly simple and do not require deep quantum circuits to perform well. Under this narrow experimental setting, a 12 percent improvement over a baseline does not establish general scalability, robustness, or utility of the method.

When a paper offers only empirical support, it is expected to include evaluation across multiple types of problems, including at least multiclass classification and a different dataset, such as CIFAR-10 or CIFAR-100. None are present.

Because the experiments are so limited in difficulty and diversity, the work must instead offer compensation with solid theoretical justification that proves why the method should work beyond toy datasets. However, convergence theory is missing, there is no stationary-point guarantee, and no rigorous mathematical explanation of how KL coupling affects the optimization landscape.

Minor (for maximum information transfer) Issues: 
The data encoding pipeline is under-specified. Feature preprocessing, feature-to-quantum-state mapping, and the reupload schedule (mentioned briefly in the paper) are extremely important. These affect the number of qubits required as well as circuit depth. Upon checking the appendix and supplementary materials, I found that the authors used PCA to reduce the dimensionality. Overall, reproducibility and completeness of the paper are negatively impacted by the lack of inclusion of these details in the main manuscript. I hope the authors include this.

Baseline selection seems to inflate the claimed gains. Vanilla QFL forces all clients to use the shallowest model, giving predictably weak results. Standalone training is a trivial lower bound. Q-HeteroFL is an extension of HeteroFL, implemented by the authors without external validation. It is not clear whether these baselines are suitable enough to be regarded as “state of the art.”
On the one hand, the engineering around layerwise circuit execution is very creative, and the hardware study is a solid strength. I would love to score this paper very highly if theoretical guarantees were provided or if the experiments were significantly more diverse.

### Questions
1. How does the proposed KL coupling affect the optimization landscape for deeper clients? Can the authors provide a theoretical or empirical justification that this does not restrict the expressive power of deeper PQCs?

2. Do the authors expect the claimed gains to hold on multiclass or quantum-native datasets? It would be very interesting to see.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this work, the authors propose a structured Federated Learning framework (Quorus) that adaptively selects and synchronizes subsets of model layers within the framework of federated learning optimization. Instead of transmitting the complete parameter set or using uniform layer selection, LFL introduces a layer-wise importance estimator and communication scheduler that determines which layers to aggregate in each round based on gradient variance and update magnitude. 

The experiments on binary classification on the MNIST and Fashion MNIST datasets demonstrate that the proposed approaches consistently outperform other quantum federated learning methods, such as Q-HeteroFL, and achieve better performance on IBM quantum platforms.

### Strengths
1. The work firstly proposes the Quorus framework that is tailored for heterogeneous-depth clients, providing an alternative way to realize quantum machine learning on realistic quantum processors without QEC techniques. 

2. The proposed layer-importance metric based on normalized gradient variance and the communication schedule can be plugged into existing federated learning systems, demonstrating the advantages of Quorus in this work. 

3. The experiments are scalable to 50-200 clients, making the authors' claimed methods reasonable. 

4. The appendix includes detailed hyperparameters, dataset information, and pseudo code.

### Weaknesses
1. Although the work claims to propose that the Quorus is effective in both simulation and realistic quantum processors, the approach is largely heuristic. In particular, the gradient-variance-based importance metric is intuitive, and it does not involve convergence or bias-variance analysis of partial aggregation. 

2. The work exacerbates the fairness gaps across clients if some layers are seldom updated. It looks like the authors mention this in passing, but do not quantify or mitigate it. 

3. The paper omits comparison with communication-efficient Federated Learning methods such as FedDrop, Sparse Ternary Compression, or Selective-FedAvg, which are conceptually closest. 

4. Ablations on importance metric choices or thresholds are limited. It is unclear how sensitive the system is to hyperparameters (e.g., top-k fraction per round).

### Questions
1. How does your Layerwise Federated Learning (LFL) framework differ conceptually and algorithmically from prior “layerwise parameter sharing” or “selective aggregation” methods?

2. Have you compared with or at least cited existing selective-layer FL methods in the classical setting (e.g., FedPer, FedRep, Selective-FedAvg, FedDrop, FedPAQ)? If not, please justify why those are not directly comparable or include a baseline comparison.

3. Could your algorithm be interpreted as a special case of layer-dropout or structured sparsification techniques? If so, why does it deserve to be treated as a new paradigm rather than an instance of that family?

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
This paper proposes a quantum federated learning algorithm that can account for errors in quantum circuits. The authors use extensive simulations to show how this approach can improve test accuracy.

### Strengths
+ The idea of using varying-depth quantum circuits is both interesting and timely.
+ The authors provide extensive experiments to validate their approach.
+ The solution can reduce barren plateau effects.
+ There are clear relevance to emerging quantum platforms.

### Weaknesses
- The gains from the experiments seem very limited.
- The scalability of this solution is not studied.
- The results seem limited to basic classification tasks.

### Questions
- How does your algorithm scale with the number of layers and number of agents? It would be useful to provide scalability experiments.
- Can you handle heterogeneous datasets? Can you run experiments with such settings?
- How do you justify such a complex design for a very small (around 10%) gain?
- What type of quantum hardware is needed for this solution to work?
- Can you study the effect of error correction or mitigation?

### Soundness
3

### Presentation
3

### Contribution
3
