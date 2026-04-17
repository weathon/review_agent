# Mitigating Barren Plateaus in Quantum Neural Networks via an AI-Driven Submartingale-Based Framework

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 2

## Abstract
In the era of noisy intermediate-scale quantum (NISQ) computing, Quantum Neural Networks (QNNs) have emerged as a promising approach for various applications, yet their training is often hindered by barren plateaus (BPs), where gradient variance vanishes exponentially in terms of the qubit size. Most existing initialization-based mitigation strategies rely heavily on pre-designed static parameter distributions, thereby lacking adaptability to diverse model sizes or data conditions. To address these limitations, we propose AdaInit, a foundational framework that leverages generative models with the submartingale property to iteratively synthesize initial parameters for QNNs that yield non-negligible gradient variance, thereby mitigating BPs. Unlike conventional one-shot initialization methods, AdaInit adaptively explores the parameter space by incorporating dataset characteristics and gradient feedback, with theoretical guarantees of convergence to finding a set of effective initial parameters for QNNs. We provide rigorous theoretical analyses of the submartingale-based process and empirically validate that AdaInit consistently outperforms existing initialization methods in maintaining higher gradient variance across various QNN scales. We believe this work may initiate a new avenue to mitigate BPs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a framework based on LLMs to mitigate the BP phenomenon in QNNs. The authors provide a theoretical basis and first results to showcase their approach.

The paper has significant structural issues, in particular lacks a reading flow. The theorems and lemmas are stated as a long list without transitions and explanations, making the paper hard to follow. There is a substantial amount of literature on BPs missing and, considering that it is the main objective of the paper, needs to be added. Further, the related works are given as a list of papers, again, making it hard for a reader to actually grasp what the state-of-the-art is. 

The experimental setup is not clearly described, and the experiments are not extensive enough to conclude about the performance of the method. In particular, starting initialization only at five distinct points in time does not result in any sort of statistical significance. It is for this reason that I do not think that the manuscript is in a publishable format.

Further, I do not think that this paper provides any contribution to the broader research community. Using LLMs to aid in optimizing a QNN, whose task the LLMs could solve by themselves without any problem, seems to be a fundamental overkill. There is a significant waste of computational resources while prompting state-of-the-art (!) LLMs to optimize a QNN to solve iris, wine, Titanic and MNIST. These datasets are further not reasonable datasets for validating the approach, since even the simplest regression can solve them up to high accuracy. I am more than aware that it is not possible yet to really solve classically hard ML problems with QML, but at least moving beyond trivially solvable datasets would be appropriate.

I would urge the authors to rethink the objective of this work or provide a convincing reason why employing LLMs would be a valuable approach to training QNNs.

### Strengths
- The authors test with a variety of different LLMs to evaluate differences in specific model used
- The authors put a lot of effort into providing the figures, which are as a result very intuitive to read

### Weaknesses
- "such as BPs, referring to a kind of..". This is really not a scientific description of a phenomenon, and in particular it is also not accurate towards the severity of the issues.
 - [Comment] please use \citep when citing within a sentence, this makes the paper more readable.
 - There is a substantial amount of literature on BPs missing. Given that the whole works aims at mitigating them, a clearer picture of state-of-the-art is necessary in the related work section. I would also expect more explanations of the approaches that are listed, s.t., the interested reader actually gets an overview of state-of-the-art, rather than just a collection of papers.
 - Line 110: "A quantum algorithm works by applying a sequence of unitary operators to an initial state". This is a description of a quantum circuit, not an algorithm. This is way too simplistic - it misses measurement, classical post-processing etc., which are crucial in quantum algorithms.
- Line 109: A state does not collapse into $\ket{i}$ for all $i$ after measurement, but rather into one of them. 
- Line 112: VQCs do not play a core role in QNNs - QNNs are a subcategory of VQCs. If your definition is different, please elaborate.
- From Definition 1. Please provide some connection between the theorems and some elaboration on them. Making the paper a pure list of theorems makes it unreadable. The same holds for Definition 2 onwards. I would also suggest providing at least proof sketches in the main part of the paper.
- Line 306: ansatz-induced BPs are not a thing (ansatz-induced could also mean entanglement induced, etc.). The referenced paper talks about the connection of expressivity (or more concretely approximate 2-designs) to BPs.
- Missing quantitative results. How much better are the approaches in percent, how significant are the results?
- The authors state that they repeat the results five times to get reliable results, however, this is not backed by statistical evidence. To get anything close to reliable, one would (1) need to incorporate significance tests and (2) run a much larger number of random initializations.
- The authors provide a trade-off analysis in the Appendix, to compare the computational costs and performance benefits. I am missing the discussion on usage of LLMs here.

### Questions
- Line 118: Where is this definition of QNNs coming from? What kind of neural network layers are you talking about (repeated applications of U?). Where is the reference?
- Why does your architecture use a classical fully connected NN in the end? This is not standard to the best of my knowledge (otherwise provide reference). In any case, this should be stated in the main part of the paper.
- In light of the choice of adding a FCNN at the end of the quantum circuit, I would be curious to see how the performance of the models is if only the FCNN is used for prediction with the classical data only as input. As I said, I am questioning the whole setup, but even having QNN + classical FCNN to predict the simple datasets used seems already an overkill before employing the LLMs.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes AdaInit, an AI-driven framework that mitigates barren plateaus in quantum neural networks. The method iteratively generates effective initialization parameters using large language models (LLMs) guided by a submartingale-based process to ensure non-vanishing gradient variance. The authors provide theoretical guarantees for convergence and boundedness, and conduct experiments on various benchmark datasets, including Iris, Wine, Titanic, and MNIST.

### Strengths
The paper introduces a novel intersection of LLMs and quantum optimization, supported by rigorous theoretical proofs, but I have not checked them. The submartingale formulation is mathematically grounded, and experiments are clearly organized.

### Weaknesses
1. The use of a general-purpose LLM for updating parameter probabilities appears unjustified. It is unclear why an LLM not trained on quantum circuit data can reliably identify parameters that avoid barren plateaus.
2. The authors claimed the time complexity for QNN training is O(T_{tr} \cdot |\theta_0|), while overlooking the cost of calculating gradients in each step. In particular, evaluating gradients accurately may require exponentially many measurements, making the method potentially infeasible at larger scales.
3. The proposed algorithm also relies on computing the gradient variance $Var(\partial E^{(t)})$ at each iteration to guide parameter updates. Similarly, this calculation again demands exponentially many measurements under barren plateau conditions, raising concerns about its scalability and practicality on current NISQ hardware.
4. The manuscript includes numerous lemmas and theorems, some of which are peripheral to the main idea. This dense theoretical presentation obscures the core contributions. The paper would benefit from presenting only key theoretical results in the main text and moving technical proofs and auxiliary lemmas to the appendix for clarity.
5. The experimental setup is somewhat limited for studying barren plateaus, as the authors only explore configurations with either many qubits and shallow depth or few qubits and deep depth. Barren plateaus are theoretically less likely to occur in these regimes. To convincingly demonstrate the framework’s effectiveness, the authors should evaluate it on larger quantum circuits with both high qubit counts and significant depth, where barren plateaus are more probable and challenging to mitigate.

### Questions
The questions are included in the Weakness.

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
3

### Summary
The paper introduces AdaInit, designed to address the barren plateaus problem in QNN training. Unlike traditional methods that use static, preset distributions to initialize parameters, Adainit adopted an iterative strategy: it leverages a generative AI model to dynamically generate new initial parameters that are more likely to escape the BP problem. The paper theoretically models this iterative process as a sub martingale and proves its convergence. Experiments show that this method outperforms existing methods in maintaining high gradient variance, especially as the QNN scales.

### Strengths
1. Casts initialization as an iterative search with a simple progress statistic and connects it to sub martingale tools.
2. Empirical sweep spans qubits and layers across four datasets with ablations on prompts and llm hyperparams.

### Weaknesses
1.The core mechanism's assumptions are too strong. The paper's core argument rests on the assumption that large language models can intelligently explore the high-dimensional parameter space of QNNs, thereby finding regions of high gradient variance. This is an extremely strong assumption. The framework provides only a scalar feedback signal (gradient variance) from an extremely complex space. However, to date, there is no theoretical or empirical evidence demonstrating how LLMs can perform such high-dimensional, geometry-aware optimization.
2. Inconsistency between the theoretical framework and the proposed method: While the application of submartingale theory is ingenious, its underlying assumptions appear to contradict the very nature of the AdaInit framework. The generation of the new parameter θ(t) explicitly depends on historical data (θ(t-1) and S(t-1)), violating the independence assumption crucial to many theoretical assertions.
3. Critical omission of experimental noise: This paper aims to address the back propagation problem, a particularly acute problem in the NISQ era. However, the experiments in this paper are conducted in an idealized, noise-free simulation environment. This is a critical omission, as noise can create its own "barren plateaus" and fundamentally alter the gradient distribution. Methods that work in a noise-free environment are not guaranteed to work on real quantum hardware. The lack of experiments under realistic noise models severely limits the practical significance and impact of our results.

### Questions
1.Could you provide a more rigorous justification for why an LLM is a suitable tool for this task? What inductive biases or capabilities of modern LLMs make them superior to classical optimization algorithms like Bayesian Optimization or Evolutionary Strategies for exploring the QNN parameter space based on scalar feedback? 
2.How do you reconcile the apparent contradiction between the i.i.d. assumption used in your submartingale proofs and the history-dependent, iterative nature of the AdaInit algorithm? Could you clarify how the theoretical guarantees hold under the more realistic condition of temporal dependency in the parameter generation process?

### Soundness
2

### Presentation
3

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
The authors propose a foundational framework called AdaInit, which leverages generative models with a submartingale property to iteratively synthesize initial parameters for quantum neural networks (QNNs) that maintain non-negligible gradient variance, thereby effectively mitigating the Barren Plateau problem.

### Strengths
1.Utilizes generative models for parameter initialization, effectively ensuring non-negligible gradient variance in the early stage of training, thereby mitigating Barren Plateau (BP) issues.
2.Extensive experiments validate the framework’s effectiveness across different QNNs scales, demonstrating good scalability and adaptability.
3.The framework is adaptive, dynamically adjusting parameter generation based on dataset characteristics and gradient feedback, enhancing the intelligence of initialization.
4.Introduces a theoretically analyzable submartingale-based iterative process, providing formal guarantees for the convergence of generated parameters.

### Weaknesses
1.Limited experimental scale: Currently validated only on small-scale QNNs with few qubits.Suggestion: Future work could explore larger-scale QNNs or real quantum hardware to evaluate scalability in high-dimensional quantum systems.
2.Focuses only on initialization: In practical applications, the key metrics are final training performance, including accuracy, convergence speed, and robustness/generalization. AdaInit only addresses the ability to start training by avoiding Barren Plateaus. Suggestion: To address this limitation, future work could extend AdaInit to not only ensure non-negligible initial gradient variance but also consider downstream training performance
3.Restricted dataset types: Only binary classification tasks were used, limiting generality.Suggestion: Test on multi-class or higher-dimensional datasets to verify robustness and applicability.
4.Dependency on generative model quality: The effectiveness of initialization relies on the output of the LLM or generative model.Suggestion: Introduce validation mechanisms or enhance the feedback loop to ensure generated parameters are effective.
5.High computational cost of iterative generation: Multiple iterations for parameter generation and gradient variance evaluation can be time-consuming.

### Questions
1.The experiments are limited to small-scale QNNs (≤20 qubits) and binary classification datasets. How would AdaInit perform on larger QNNs or multi-class/high-dimensional datasets?
2. AdaInit mainly ensures non-negligible initial gradient variance, but does it improve final training performance (accuracy, convergence speed, robustness/generalization)?Is higher initial gradient variance always correlated with better downstream training outcomes? Could the model still converge slowly or achieve suboptimal final accuracy?
3. How would AdaInit perform in noisy intermediate-scale quantum (NISQ) devices with measurement errors?
4.AdaInit relies on generative models (e.g., LLMs) to produce parameters. How sensitive is the method to different LLMs, model sizes, or prompt designs?Could errors or instability in the generated parameters negatively affect QNN training?

### Soundness
3

### Presentation
2

### Contribution
1
