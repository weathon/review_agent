# Towards the Three-Phase Dynamics of Generalization Power of a DNN

- Decision: Reject
- Scores: 4, 8, 6, 6

## Abstract
This paper addresses the core challenge in the field of symbolic generalization, i.e., how to define, quantify, and track the dynamics of generalizable and non-generalizable interactions encoded by a DNN throughout the training process. Specifically, this work builds upon the recent theoretical achievement in explainable AI, which proves that the detailed inference patterns of DNNs can be strictly rewritten as a small number of AND-OR interaction patterns. Based on this, we propose an efficient method to quantify the generalization power of each interaction, and we discover a distinct three-phase dynamics of the generalization power of interactions during training. In particular, the early phase of training typically removes noisy and non-generalizable interactions and learns simple and generalizable interactions. The second and the third phases tend to capture increasingly complex interactions that are harder to generalize. Experimental results verify that the learning of non-generalizable interactions is the direct cause for the gap between the training and testing losses.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes measuring DNN’s generalization power by decomposing its predictions into AND–OR interactions. It also proposes a three-phase dynamics of the generalization power of interactions.

### Strengths
The proposed three-stage dynamics sounds interesting and reasonable. The experiments results span multiple architectures and domains.

### Weaknesses
The non-generalizable interactions can only be identified by training the model on the test data, which is not realistic.

### Questions
Where did you define non-generalizable interactions? How do you identify them? Are those interactions data-specific or they are invariant across different datasets?

Training on test data sounds weird. Is it possible to identify the non-generalizable interactions only using train and evaluation data, and the identified interactions can be transferred to test data?

Can the framework of identifying non-generalizable interactions be transferred to transformers?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Recent advances in symbolic generalization show that, given a DNN and an input, there exists an AND-OR logical model (a sum of AND and OR interactions over input components) that faithfully replicates the network’s predictions for every possible input masking. Under three common conditions, the conciseness requirement is naturally satisfied by the construction of the AND-OR model, meaning that only a small subset of salient interactions contributes most to the logical model’s total output.

The authors assume that if the interactions in the AND-OR model generalize well to the test set (i.e. appear in the AND-OR model of base model trained on the test samples), the neural network itself also generalizes well to the test set. Building on this claim, one can infer the model’s generalization capability (also during training) from that of an AND-OR model’s interactions. To compute a sufficient rigorous lower bound on the generalization capacity of the logical model, only salient interactions need to be considered, making the generalization test (for both the AND-OR model and the neural network) computationally tractable.

The paper provides a comprehensive evaluation across multiple models, dividing the training process into three stages, where generalization first improves and then declines. The decline corresponds to overfitting - when learned interactions become specific to the training set and fail to generalize to the test set - resulting in decreasing training loss but increasing testing loss. Finally, the authors propose a training method designed to prevent the model from learning non-generalizable interactions, showing that while training loss increases, testing loss remains stable or even improves.

### Strengths
1. Novelty: The paper presents a new and non-trivial approach to analyzing the training process and offers an interesting perspective on representing and handling overfitting.

2. Comprehensive Evaluation: The experiments are conducted on six medium+ models trained on standard benchmarks across both NLP and vision domains.

3. Soundness: The experimental results (Fig. 3, Fig. 4) clearly support the main claims of the paper.

4. Scalability / Extendability: The method appears to scale well and can be applied in multiple settings.

### Weaknesses
1. Readability: While perhaps inevitable given the topic’s complexity, the paper is difficult to follow. Some provided illustrations (e.g. Fig. 1, Fig. 2) and examples (Fig. 7) are overly complex. Furthermore, the absence of an Evaluation section results in an excessively long and dense Methodology section. The paper also lacks details on how the AND-OR logical model is constructed, which appears to be a very relevant background for this work.

2. Limitations: The proposed method cannot quantify the model’s generalization power with respect to unseen input samples - only for those within the test set.

3. Runtime: The suggested training process to avoid overfitting appears to be time-consuming. According to Eq. (11) and Eq. (12), an AND-OR logical model must be generated for each input sample in the training set, and the generalizability of each salient interaction must be tested against the test data. What is the computational complexity and runtime of this process?

### Questions
1. Splitting the paper into separate Methodology and Evaluation sections is strongly recommended.
2. Why do the authors divide the training process into three phases? A more natural split would be into two parts - Phase 1 for increasing generalization power and Phase 2 for decreasing generalization power. 
3. Are the three assumptions of the conciseness property really common (all at the same time) in practice?
4. The assumption that the model’s generalization can be derived from the generalization of an AND-OR model's interactions is unclear to me: the latter holds for a specific sample and the power set of its masking options, but not necessarily for all inputs.
5. Adding a simple running example would improve readability. I assume it may be too complicated with current space limitations, so it is only a nice-to-have addition.

Typos:
- Line 431: (dose not exert -> does).
- Line 655: missing $\sigma_S$ in the equation of $o^{or}_S$ (after the minus).
- Line 764: red color for testing loss, blue color for training loss.

### Soundness
4

### Presentation
2

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
This paper investigates the generalization dynamics of deep neural networks (DNNs) through the lens of symbolic interactions. Building on prior work showing that DNN inference can be decomposed into AND-OR interaction patterns, the authors propose a method to quantify the generalization power of individual interactions by comparing them against baseline DNNs trained on test data. They discover a three-phase training dynamic: (1) early phase removes non-generalizable interactions and learns simple patterns, (2) middle phase learns increasingly complex interactions with decreasing generalization, and (3) late phase predominantly learns non-generalizable interactions leading to overfitting. Experiments on vision and NLP datasets validate these findings.

### Strengths
**Comprehensive theoretical foundation**
The paper builds systematically on the AND-OR interaction framework from prior work, providing clear mathematical formulations and proofs.

**Well-articulated three-phase dynamics**
The identification of distinct training phases characterized by interaction generalization power provides new insights into DNN training dynamics and offers a principled explanation for the train-test gap.

**Extensive empirical validation**
Experiments span multiple architectures and datasets, demonstrating consistency of findings across domains.

### Weaknesses
**Unclear causality**
While the paper shows correlation between non-generalizable interactions and overfitting, the causal direction is unclear. Are non-generalizable interactions the cause of overfitting, or merely a symptom of models memorizing training data?

**Phase transitions not rigorously defined.** The three phases are identified qualitatively from figures, but there are no precise criteria or automatic methods to detect phase boundaries. This makes the framework difficult to apply systematically.

**Minor presentation issues.**
There’s a typo (“non-gneralizable”) in Fig. 4.

### Questions
1. How can the method scale beyond 10 input variables? Can this framework be extended to realistic input dimensions?

2. Why is training on test data a valid approach for identifying generalizable interactions? Could alternative methods (e.g., cross-validation set) achieve similar results without test data access?

3. How do findings extend to attention-based models, where interactions might be explicitly modeled through attention mechanisms?

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
1

### Summary
This submission proposes a framework to define, quantify, and track the generalization power of primitive inference patterns—formalized as AND-OR “interactions” among input variables—encoded by deep neural networks (DNNs) over the course of training.  The paper shows that removing non-generalizable interactions from the network’s output narrows the train–test loss gap, suggesting these interactions are a proximate cause of overfitting.

### Strengths
1. The paper is clearly written with helpful figures and a step-by-step methodology.
2. The framework is anchored in prior formal results ensuring exact matching on masked inputs and sparsity of salient interactions under common conditions.
3. The three-phase dynamic is intuitively appealing and coheres with known overfitting behavior, offering a clear diagnostic narrative for training curves.

### Weaknesses
1. The binary label G may be too coarse for nuanced interactions.
2. Claims of causality are suggestive but not definitive: removing non-generalizable terms is consistent with overfitting, yet residual confounds may remain.
3. The baseline DNN is trained on the test set, which could bias labels and inflate H if the baseline overfits
4. The masking protocol and the restriction to exactly 10 variables per input may affect interaction discovery and comparability across tasks

### Questions
For LLMs, can you extend analyses from masked classification settings to generation (e.g., next-token distributions) with interaction tracking over context length? What changes in the three-phase dynamics do you expect?

Is there evidence of rare but truly generalizable high-order interactions (e.g., compositional patterns)? How would the current proxy distinguish them from non-generalizable ones?

### Soundness
3

### Presentation
3

### Contribution
3
