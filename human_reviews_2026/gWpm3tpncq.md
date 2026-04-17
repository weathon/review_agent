# From Offline to Online Memory-Free and Task-Free Continual Learning via Fine-Grained Hypergradients

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Continual Learning (CL) aims to learn from a non-stationary data stream where the underlying distribution changes over time. While recent advances have produced efficient memory-free methods in the offline CL (offCL) setting online CL (onCL) remains dominated by memory-based approaches. The transition from offCL to onCL is challenging, as many offline methods rely on (1) prior knowledge of task boundaries and (2) sophisticated scheduling or optimization schemes, both of which are unavailable when data arrives sequentially and can be seen only once. In this paper, we investigate the adaptation of state-of-the-art memory-free offCL methods to the online setting. We first show that augmenting these methods with lightweight prototypes significantly improves performance, albeit at the cost of increased Gradient Imbalance, resulting in a biased learning towards earlier tasks. To address this issue, we introduce Fine-Grained Hypergradients, an online mechanism for rebalancing gradient updates during training. Our experiments demonstrate that the synergy between prototype memory and hypergradient reweighting substantially allows for improved performances of memory-free methods in onCL. Code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The core objective of this paper is to address a challenging problem in the field of Continual Learning (CL): how to successfully adapt effective memory-free algorithms from idealized offline (offCL) settings to more realistic and difficult online, task-free (onCL) environments.

To solve the problem of gradient imbalance, the paper proposes its core innovation, Fine-Grained Hypergradients (FGH). This is a novel optimization technique based on the key idea of:

+ Learning an independent, dynamic gradient weight for each parameter within the model.
+ Leveraging the gradient directions from two consecutive iterations to assess learning stability: if the directions are aligned, the update step is amplified; conversely, if they are opposed (indicating oscillation), the update is suppressed.

### Strengths
1. The problem addressed by the paper online, memory-free, task-free continual learning is indeed a highly challenging and practically significant direction in the current field.
2. The combined framework proposed in this paper achieves outstanding performance in experiments, especially under the 'multi-learning-rate evaluation' paradigm designed by the authors, showcasing the robustness of their method.

### Weaknesses
1. The entire work can be viewed as an effective combination of two known techniques (prototypes and hypergradient descent), making the contribution more empirical than conceptual. The performance improvement from FGH largely stems from enhancing plasticity in the online setting; Equation (7) progressively increases the intra-task learning rate to boost plasticity, a mechanism that has been explored in prior work [1]. 
2. Regarding catastrophic forgetting, the method essentially relies on prototype replay, which is also a common technique in previous literature. For a venue like ICLR, which seeks fundamental innovations, the weight of this contribution is insufficient.
3. The authors use ADAM in their experiments. From a learning rate perspective, could ADAM and FGH conflict? Is it possible for a situation to arise where ADAM suggests a large learning rate while FGH suggests a small one? In other words, do FGH and ADAM work synergistically, or is there a functional redundancy? Given the prevalence of ADAM, the authors should have included a discussion on this.
4. The authors should provide a comparative experiment between a "global FGH" and the proposed "fine-grained FGH" to demonstrate the necessity of the fine-grained design.

[1] Online Learning Rate Adaptation with Hypergradient Descent ，ICLR2018

### Questions
See the weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper leverages hyper-gradients for continual learning. The key idea is to use prototypes as memories and use hyper-gradients for adaptive learning rate selection.

### Strengths
The paper writing is clear.
Experiments show the effectiveness of the proposed method under the proposed setting.

### Weaknesses
Limited novelty: both the hyper-gradient and prototype based memory are not new, they have been widely used in previous works for adapting learning rates (https://arxiv.org/pdf/1703.04782) and prevent forgetting (https://arxiv.org/pdf/2308.00301) already. 

Experiment setting and claims: This work claims to be online and memory free, however, it uses cached prototypes which is also just a form of memory, without having other methods using the same compute, memory and storage, it is not fair to claim the performance gain.

Also, even with complex method implementation, the method is just a little bit better than simple ER, while uses heavy hyper-parameter tuning, which is prohibitive in the online CL scenario. This makes the setup and method both far from practical.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the migration of offline memory-free continual learning (CL) methods to online, memory-free, and task-free CL scenarios. It introduces a prototype-based auxiliary memory module (P) and a fine-grained hypergradient mechanism (FGH) that dynamically balances gradient imbalance and learningrate sensitivity. Experiments on CIFAR100, CUB, and ImageNet-R show consistent gains across multiple
baselines under multi-learning-rate evaluation. The work is practically motivated and conceptually coherent, offering a bridge between offline and online CL paradigms.

### Strengths
1) The topic is timely and relevant, targeting the underexplored Offline→Online transition in CL with
clear theoretical and practical significance. 

2) The proposed P+FGH framework effectively addresses two core challenges of online CL —
catastrophic forgetting and gradient imbalance — through a minimal-intrusive and generalizable
design. 

3) Experiments are comprehensive, covering diverse datasets and learning rate settings, demonstrating
the method’s robustness and transferability.

### Weaknesses
1) The online scenario remains quasi-online, relying on pre-defined task splits rather than fully
stream-based settings, limiting realism. 

2) The novelty of both P and FGH is moderate: the prototype update mirrors CoPE (2021), and FGH
lacks formal convergence or stability analysis and clear differentiation from prior hypergradient
methods. 

3) Recent baselines (e.g., PROL 2025, PMLR 2025) are missing, and parameter details (γ, β₁/β₂, Si- Blurry settings) are insufficiently reported, affecting reproducibility and fairness.

### Questions
1) How would the proposed FGH behave under fully stream-based or class reappearance settings?

2) Has γ been systematically tuned or theoretically analyzed for robustness across datasets?

3) Can the authors quantify FGH’s computational overhead compared to existing hypergradient or adaptive LR
optimizers?

### Soundness
3

### Presentation
3

### Contribution
3
