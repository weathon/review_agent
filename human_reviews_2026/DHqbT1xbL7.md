# Anon: Exploring the Adaptivity of Optimizers and Beyond

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Adaptive optimizers such as Adam have achieved great success in training large-scale models like large language models and diffusion models. However, they often generalize worse than non-adaptive methods, such as SGD on classical architectures like CNNs. We identify a key cause of this performance gap: adaptivity in pre-conditioners, which limits the optimizer's ability to adapt to diverse optimization landscapes. To address this, we propose **Anon** (**A**daptivity **N**on-restricted **O**ptimizer with **N**ovel convergence technique), a novel optimizer with **continuously tunable adaptivity** $\gamma\in \mathbb{R}$
, allowing it to interpolate between SGD-like and Adam-like behaviors and even extrapolate beyond both. To ensure convergence across the entire adaptivity spectrum, we introduce *incremental delay update (IDU)*, a novel mechanism that is more flexible than AMSGrad's hard max-tracking strategy and enhances robustness to gradient noise. We theoretically establish convergence guarantees under both convex and non-convex settings. Empirically, Anon consistently outperforms state-of-the-art optimizers on representative image classification, diffusion, and language modeling tasks. These results demonstrate that adaptivity can serve as a valuable tunable design principle, and Anon provides the first unified and reliable framework capable of bridging the gap between classical and modern optimizers and surpassing their advantageous properties.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Anon (Adaptivity Non-restricted Optimizer with Novel convergence technique), an optimizer with a tunable adaptivity hyperparameter γ that allows it to interpolate between SGD-like and Adam-like behavior — and even go beyond both. They formalize adaptivity as a knob that changes how the optimizer scales the loss landscape: higher adaptivity flattens sharp valleys and amplifies flat regions, while negative adaptivity does the reverse. To ensure stability across this expanded spectrum, they introduce Incremental Delay Update (IDU), a soft accumulation mechanism that replaces AMSGrad's max-tracking and provides better convergence and noise robustness. Experiments across image classification, diffusion, and language modeling tasks show that Anon consistently outperforms baseline optimizers, highlighting adaptivity as a key tunable property for optimization.

### Strengths
1) The paper is well written
2) Provide intuition on the main adaptivity hyperparameter
3) Provide synthetic experiments to elucidate the capability of Anon to discover solutions that traditional optimizers will not uncover
4) Extensive evaluation comparing to various standard optimizers on various tasks across different modalities
5) Consistent improvements across a variety of settings

### Weaknesses
1) Outside of synthetic examples (Fig 1) they dont show experiments to support the claim that Anon learns solutions (network parameters) that differ from the other standard optimizers. One experiment could be to see if you ensemble the networks learned by standard optimzer with Anon if there is a improved performance.

2) Lacks comparisons to optmizers like Lion and Muon.

### Questions
1) Suggestion for showing diversity of learnt weights:  Even though norms of the network weights may seem different, it would be more interesting if some variant of cosine similarity on the weights of the network trained with Anon vs other optimizers would be quite interesting. Ex: compare the similarity of weights learnt with a certain optimizer  with weights learnt by other optimizers and do it for all the optimizers and see how different Anon is from other optimizers. 

Perhaps such an experiment would help showcase the different type of networks that Anon is capable of converging to?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new optimizer Anon. The authors claim that the adaptivity measure (provided in the paper) of existing optimizers always falls in the segment [0,1]. Based on toy problems, they demonstrate that the adaptivity beyond the segment [0,1] might be beneficial, which serves as a motivation for new algorithm design with such adaptivity. The authors provide theoretical under restrictive assumptions to demonstrate some convergence rates for the Anon algorithm. Finally, the authors demonstrate that the algorithm improves over many existing algorithms on several

### Strengths
- The authors provide a measure of adaptivity based on the function $\psi$, which constructs the preconditioning. They claim that the adaptivity measure $A$ always remains in the range $[0,1]$ for most existing algorithms, which might be seen as their limitation (e.g., the authors suggest that it may be the reason why Adam underperforms when training on ResNet-like models). Therefore, the authors motivate the study of algorithms whose adaptivity goes beyond the range $[0,1]$ and introduce Anon

- The authors provide some sort of convergence guarantees in the online convex and stochastic non-convex cases and obtain standard convergence rates.

- Empirical results demonstrate that Anon outperforms existing algorithms on different tasks: Resnet-18 and -50 on Imagenet, GPT2-like models on OpenWebText, and image generation on CIFAR10. In all the cases, Anon achieves better performance metrics than baselines.

### Weaknesses
- In my view, the paper is badly written; it is hard to follow and understand the main claims. I encourage the authors to either provide toy examples and simplify their claims, including theorems, so that the paper becomes more readable. In the current form, it is impossible to evaluate the contributions of the paper. Below, I provide more details on why I find this paper hard to read:

    - First, I don't understand why Definition 1 serves as an adaptivity measure. Providing some simple examples, toy experiments, or similar to demonstrate the intuition behind such a choice of adaptivity measure would improve the paper. 
    - Undefined notation that is used in various places throughout the paper. To mention a few: what is EMA$^{\gamma}$?; why do existing convergence proofs require the assumption (4) (many papers don't need such a monotonicity of the effective lr); what are $\beta_{1t+1}$? and $\beta_{11}$?
    - Many typos and inaccuracies in the writing.
    - Section 3.1 is completely unclear, which should serve to provide the intuition. I don't understand why such a choice of preconditioning will fulfill all the requirements mentioned before in the text. Why are $\beta_3$ and $a_n$ chosen in such a way? Is there any intuition for that?
    - Theorems 3 and 4 are unreadable. There are so many constants, sums, etc, that it is hard to parse the rates.

- The convergence guarantees are provided under very restrictive assumptions: a bounded domain and a bounded gradient norm, which significantly limit the theoretical applicability of the proposed algorithm. The rates are implicit; I don't see directly what the convergence speed is. I am pretty sure that the dependencies on the problem constants are from the optimal. Therefore, I do not find a value in such convergence guarantees, as any algorithm can converge if a sufficient number of assumptions is used. 

- Unclear constants in Theorem 4:

    - Does the choice of constants $C_l$ and $C_u$ serve as the assumption, or can we ensure the existence of such constants by properly setting the hyperparameter?
    - The constant $J$ is defined as the sum of two terms. The second term involves the sum for $k=1$ to $\hat{a}_t$. Why does this sum depend on $t$? Does it mean that $J$ depends on $t$? Can we show that $J$ is bounded uniformly?
    
- Although Anon performs well in all the cases, I don't find any explanation for the choice of $\gamma$. Choosing $\gamma=0.1$ or $\gamma=1.01$ looks like a very careful tuning, making the comparison unfair. Therefore, the gap in performance between Anon and other baselines can be due to the untuned hyperparameters of the baselines. The algorithm is not robust to the choice of $\gamma$, which limits the applicability, as we basically obtain one more hyperparameter to tune. I encourage the authors to provide guidelines to set $\gamma$ in practice.

- The authors overuse "SOTA" in the paper. For example, accuracy $92.47$ is not SOTA and can be further improved by adding some tricks to the training. 

- Sophia paper is known for its bad experimental part and unreliable results. Therefore, using their experimental setup and implementation is not the best thing to do. This issue questions the reliability of the results in this work as well.


To summarize, I believe the presentation of this work has to be improved significantly.

### Questions
- I find the results in Figure 3 weird because the performance of Anon is not smooth w.r.t hyperparameters. Why is it so?

- Please, add a link to Table 6 in the main part. It is hard to understand without directly computing $A$ why the adaptivity is in $[0,1]$ in most cases.

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
5

### Summary
The authors propose Anon, a novel optimizer stemming from considerations on adaptivity of gradient methods -- measured as a new quantity $A$, which for standard adaptive methods is in the range $[0,1$. The authors generalize this interval and show performance of their method on a set of tasks (from toy to LM), as well as provide convergence rate.

### Strengths
0) The paper is nicely written, formal, and nicely packaged. I appreciate the authors efforts in delivering a readable high-quality product.

1) Studying preconditioners is interesting, and specifically, new measures of adaptivity are needed and can shed new light on poorly understood phenomena. I like the approach.

2) The paper has a big set of experimental results, together with a comprehensive analysis. I briefly checked and all seems OK here. The rate is in some way not surprising but is good to report. Adds value to the paper

### Weaknesses
There are unfortunately a few weaknesses that prevent me from placing a positive score, though I like the approach and the style.

0) The measure A is a simple function (yet, written in a complicated way in the general setting): it does correlate with adaptivity strength, but given the premises and the large framework, I was expecting something more insightful: the quantity still contains EMAs in Adam, and is trivial or not directly computed in most other settings.

1) The algorithm Anon, proposed by the authors, looks a little overengineered, with no clear intuition. Potentially, other methods can also achieve the same effect. This is not a straightforward method, and its performance is not stellar -- see the next point.

2) At the price of a bit of complication of the method, the authors were not able to convince me: their improvements in language modeling are tiny. They compare all methods with the same hyperparameters -- yet we know well that those might be suboptimal, and it is easy to be best at one hyperparameter value. Note here that you instead chose your algorithm form. This is unfair. I agree, this requires massive computing to evaluate fairly, but the stakes are high. I could see, for example, Adam with a higher beta2 working better (similar effect), or this being an effect of the noise in the learning rate.

### Questions
0) Thm 1 is a bit confusing. You did not define the equivalence class. This looks more like a definition

1) Imagenet experiments: I cannot find Anon?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Anon - a novel optimization algorithm designed to address the generalization gap between adaptive optimizers (like Adam) and non-adaptive methods (like SGD) across diverse model architectures. The core idea is to formalize "adaptivity" as a continuously tunable hyperparameter, which allows Anon to smoothly interpolate between SGD-like (low adaptivity) and Adam-like (high adaptivity) behaviors, and even extrapolate beyond these traditional boundaries. To ensure convergence across this broad adaptivity spectrum, the authors propose Incremental Delay Update (IDU), a new stabilization mechanism more flexible and robust to gradient noise than AMSGrad's max-tracking.

Theoretical guarantees for the method are provided and a thorough empirical study is conducted over state-of-the-art optimizers on image classification, diffusion, and language modeling tasks.

### Strengths
I find the below strengths in this work:
- Most significant strength is the formalization of adaptivity as a continuous control variable and the introduction of Anon, which allows for continuously tunable adaptivity via a hyper-parameter
- Proposed method seems to  work well empirically. Anon is shown to outperforms a wide range of strong baseline optimizers across diverse tasks and architectures, including ResNet, diffusion models, and GPT2. 
- Incremental Delay Update (IDU) mechanism is a good idea and contribution, effectively tackling the inherent instability risks associated with extending adaptivity beyond the conventional range.
- Potentially IDU can even lead to faster training times compared to Adam by transforming expensive vector power operations into negligible costs, enhancing Anon's practical applicability.

### Weaknesses
I also find some issues in the work:
- While Anon allows for adaptivity beyond $[0, 1]$, the empirical results primarily showcase $\gamma$ values relatively close to this range, it would be good to investigate more thoroughly what happens with extreme values of the hyper-parameter. The paper mentions that extreme adaptivity risks instability but doesn't extensively explore what happens at very large positive or negative values 
- The mathematical formulation of the Anon pre-conditioner in Equation 5 and its implementation in Algorithm 2 appear somewhat complex with several intermediate variables and conditions. I understand that this may be necessary for theoretical guarantees, but a more intuitive explanation or a simplified interpretation of how IDU practically operates could improve understanding.

### Questions
SOme questions for the authors:
- How sensitive is Anon's performance and convergence to these specific choices of hyper-parameter for IDU?
- Can the authors provide a more intuitive explanation of how the incremental delay update (IDU) specifically enhances robustness to gradient noise?
- How does one extend the concept of adaptivity to optimizers that do not strictly conform to the "Generic Optimizer Method Frame" outlined in Algorithm 1?

### Soundness
3

### Presentation
3

### Contribution
3
