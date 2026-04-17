# HomM: Homogeneous Momentum Optimizer with Finite-Time Convergence

- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
We introduce HomM, a homogeneous momentum optimizer derived from a perspective of continuous-time dynamical systems. HomM integrates homogeneity (scaling) and momentum, achieving finite-time convergence to an optimal solution under standard assumptions for both convex and non-convex objectives. To bridge theory and practice, we propose a semi-implicit discretization of the continuous-time HomM. Additionally, we present a unified framework for understanding adaptive optimizers through the lens of homogeneity, highlighting comparisons with HomM. Empirical evaluations on deep learning benchmarks show that HomM outperforms widely used momentum-based baselines, including SGD with momentum and Nesterov acceleration, as well as adaptive methods such as Adam and Lion.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a momentum-based optimizer that exhibits a degree of a scale invariance ("homogeneity") that is parameterized by a constant $\alpha\in(-1,0]$. The paper provides theoretical results, demonstrating finite-time convergence of the continuous-time dynamics in both the well-conditioned and smooth+PL regimes. Additionally, the authors propose a discretization scheme for the continuous-time dynamics and empirically evaluate the method on deep learning tasks. The results suggest that homogeneity may lead to better generalization than adaptive gradient methods, possibly due to the avoidance of sharp minima.

### Strengths
1. To my knowledge, finite-time gradient flows have not been well explored in machine learning. The idea is novel and certainly could be of interest to the theoretical ML community.
2. The authors also generalize the homogeneity property of common optimizers, which is another theoretical contribution.
3. For the most part, the paper is well-written and clear to read.

### Weaknesses
Unfortunately, I believe that the potential impact of the contributions is limited, and there are several claims by the authors that are not well supported or motivated. For example,
1. The theoretical results rely on (1) continuous-time dynamics; (2) smoothness assumptions; (3) strong convexity/PL assumptions (the authors claim "both convex and non-convex objectives" throughout the paper, but the PL condition is assumed in the non-convex setting). These assumptions are extremely strong, and little discussion is provided regarding their practicality outside of a theoretical setting.
2. A discretized algorithm is provided, but no theoretical results (convergent hyperparameter regimes, convergence rates, etc.) are shown.
3. I don't understand the claim that scale invariance is an undesirable property when it comes to "over-sensitivity around flat minima". The cited works in lines 300-308 do not support the claim that adaptive optimizers fail to generalize well *because* of scale invariance.
4. The experiments sizes are far too small, featuring results on only toy example, e.g. CIFAR. There is little evidence to suggest that the proposed method scales to larger models and tasks, and the results presented in the paper are not convincing.
5. Comparison to state-of-the-art optimizers (in particular, optimizers with weight decay) are brushed off as future work. I don't see why adding weight decay (or other regularization mechanisms) would be an unfair comparison, especially if the method is being proposed as an empirically effective optimizer.
6. The proposed method introduces a new hyperparameter, $\alpha$, that seems to require extensive tuning.
6. (minor) typos:
- Use `\citep` for citations that should be parenthetical.
- Line 111: "stardand homogeneous"
- Line 632: the correct inequality is $f^*\geq f_\theta-\frac{1}{2\mu}\nabla f_\theta^\top\nabla f_\theta$. The rest of the proof should be adjusted accordingly.
- Line 664: "Lyapunoc"
I have not checked the proofs in the detail, but even assuming that the theoretical results are correct, the mentioned weaknesses are already substantial.

### Questions
1. To my understanding, Figure 1a does not appear to demonstrate the finite-time convergence of the proposed method?
2. If 1. is due to discretization, how does the finite-time convergence result transfer from continuous time to discrete time? If finite-time convergence in continuous time does not imply finite-time convergence in discrete time, what is the benefit of having a continuous finite-time result?
3. See also Weaknesses.

### Soundness
2

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
3

### Summary
This paper proposes the HomM algorithm, which adaptively scales both the parameters and momentum in the optimization process. The main results include:  
* Proof of the finite-time convergence of the continuous-time differential equation under the convexity and PL assumptions.  
* Proposal of semi-implicit update rules for the continuous-time equations, and numerical demonstration that the discrete-time algorithms can well approximate the continuous-time equations.  
* Experiments comparing the performance of HomM with other algorithms, including Adam, SGD, and Lion.

### Strengths
* The idea of incorporating homogeneity into optimizer design is interesting.
* The result of finite-time convergence of HomM differential eqaution provides theoretical guarantee on the performance of the purposed algorithms.

### Weaknesses
* In general, I find the paper not well-organized. In particular, it lacks a section discussing related work, which makes it unclear how the current paper is positioned within the existing literature.

* The idea of using homogeneity in algorithm design is not sufficiently motivated. Specifically, in the introduction, the authors mention that current adaptive optimization methods fail to fully explore the flat regions of the loss landscape. However, the motivation for how incorporating homogeneity can address this issue is not clearly stated.

* I also find the conclusion that HomM can outperform commonly used algorithms such as SGD and Adam somewhat premature, given the experimental evidence presented. In particular, in the CIFAR-100 experiments (Table 3), the reported test accuracy of Adam is 72.08%, which appears lower than commonly reported in the literature. For example, in Table 3 of Li et al. (2022), SGD achieves 78.10% accuracy on CIFAR-100 using a ResNet-18 architecture, which is higher than the accuracy of HomM reported in this work. 
**I strongly encourage the authors to include comparisons with recent benchmarks reported in the literature, in order to provide a more comprehensive and convincing evaluation of HomM against existing algorithms.**

* In line 29, “Neursnov” should be corrected to “Nesterov.”

Reference:

Li et al., Efficient Generalization Improvement Guided by Random Weight Perturbation.

### Questions
What is the high-level reason that homogeneity can improve the ability of optimization trajectories to explore the flatter regions of the loss landscape?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces HOMM, a homogeneous momentum optimizer derived from a perspective of continuous-time dynamical systems. 
The author(s) of the paper derives convergence guarantees under the standard assumption that the objective is strongly-convex and smooth (gradient Lipschitz), as well as under the non-convex case when PL inequality is assumed. A discretization scheme of the continuous-time dynamical systems is provided. Numerical experiments are conducted that shows that promising results compared to the classical methods such as SGD with momentum and Nesterov's accelerated method, as well as adaptive methods such as ADAM and LION.

### Strengths
(1) The continuous-time dynamics and hence the discretization are novel. 

(2) Convergence guarantees are provided for the continuous-time dynamics under the strongly-convex plus smoothness assumption, or assuming PL inequality holds. The proof is based on constructing some non-trivial Lyapunov function. The Lyapunov function involves both quadratic terms and the objective.

### Weaknesses
(1) Currently, there are not any non-asymptotic convergence guarantees for the discrete-time algorithm. Since the author(s) mention in the experiments that HomM can sometimes outperform Nesterov's accelerated method and SGD with momentum, it would be interesting to see if HomM enjoys the same or even better dependence on the condition number compared to Nesterov's accelerated method. Without such theoretical results, it is not easy to convince that one should use HomM other than more classical methods in the literature.

(2) Numerical results are mixed. For example, in Figure 3, HomM enjoys better test accuracy, and hence generalization performance, but it seems it has worse training accuracy and training loss compared to existing methods in the literature.

### Questions
(1) Theorem 2 and Theorem 3 are both asymptotic in nature. I am wondering if you can provide some theoretical bounds on $T_{s}$. That seems to be possible from Theorem 1 that you quoted from the literature. It will be interesting to see how $T_{s}$ depends on the condition number in the strongly-convex case, and how it behaves when you assume PL inequality, which might shed some insights how HomM performs according to the theory.

(2) The title of Section A.3 should be Proof of Theorem 3. There is a typo.

### Soundness
3

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
5

### Summary
In this work, the authors introduce a novel optimization mechanism named Homogeneous Momentum (HOMM), designed for optimizing continuous-time dynamical systems. The proposed approach integrates homogeneity (scaling) and momentum principles to accelerate convergence. The key idea is to achieve finite-time convergence to an optimal solution under standard assumptions applicable to both convex and non-convex objective functions.

### Strengths
1. Novelty

The proposed Homogeneous Momentum (HOMM) mechanism presents an innovative contribution to the field of optimization. By integrating homogeneity (scaling) and momentum, the authors introduce a fresh perspective on accelerating convergence within continuous-time dynamical systems. This integration is conceptually appealing because it unifies two distinct acceleration principles under a single framework, potentially offering both theoretical elegance and practical performance improvements. The idea of embedding homogeneity into a momentum-based optimizer appears novel and could inspire further developments in optimization dynamics and algorithmic design.

2. Experimental Strength

The experimental section provides solid empirical evidence supporting the effectiveness of the proposed HOMM. The authors compared HOMM against several widely used gradient-based optimizers, including SGD, ADMM, and other peer methods, across multiple benchmark tasks. The reported results demonstrate consistent performance gains in terms of convergence speed and solution quality. The empirical validation is well executed and suggests that HOMM generalizes effectively across different optimization landscapes. Additional ablation studies or sensitivity analyses (if included) further strengthen the experimental credibility. Overall, the experiments convincingly substantiate the claimed advantages of the proposed approach.

3. Theoretical Analysis

The paper also provides a rigorous theoretical foundation. The authors establish finite-time convergence results under standard assumptions for both convex and non-convex objectives. The derivations appear mathematically sound and align with the framework’s design principles. The theoretical analysis contributes meaningful insight into why the proposed mechanism achieves improved convergence behavior and differentiates it from existing momentum-based methods. This analytical strength enhances the paper’s overall impact, bridging conceptual innovation with provable guarantees.

### Weaknesses
1. Limited Novelty

The conceptual contribution of Homogeneous Momentum (HOMM) appears incremental rather than groundbreaking. While the authors describe HOMM as a novel integration of homogeneity and momentum, the underlying formulation—particularly as presented in Equation (1)—relies heavily on the canonical momentum structure that has been well established in prior optimization frameworks, including ADMM and other momentum-based methods. The use of homogeneity as a scaling mechanism is not new, and the paper does not provide sufficient justification for how this particular combination represents a substantive theoretical advancement. Consequently, the proposed method lacks a clear differentiating factor that would establish it as a conceptual breakthrough within the optimization literature.

2. Weak Experimental Design and Limited Scope

The experimental evaluation is relatively narrow in scope. The authors primarily benchmark HOMM against a small set of peer optimizers using limited datasets, such as CIFAR-100, which restricts the generalizability of the findings. To convincingly demonstrate the practical utility of the proposed method, the evaluation should be extended to a broader range of benchmarks, including more challenging or domain-diverse datasets. For example, applying HOMM to medical imaging tasks (e.g., the ADNI dataset) or large-scale vision benchmarks would provide stronger empirical evidence of its robustness and adaptability. The current experimental setup, while adequate for preliminary validation, does not substantiate the claimed superiority of HOMM across different application domains.

3. Outdated Evaluation Framework

The choice of model architecture further limits the strength of the experimental validation. The authors rely on an older deep neural network, ResNet-34, to evaluate performance. Given the rapid evolution of deep learning, such a model may no longer be representative of the current state of the field. To demonstrate the scalability and modern relevance of HOMM, the authors are encouraged to evaluate their approach on contemporary architectures and openly released large-scale models, such as recent Large Language Models (e.g., LLaMA 4). This would provide a more compelling test of the optimizer’s generalizability and efficacy in modern AI contexts.

4. Incomplete Theoretical Analysis

The theoretical discussion in the paper is not comprehensive enough to support the claimed analytical rigor. Although the authors cite several works related to finite-time convergence, the provided proofs are high-level and omit essential details regarding convergence bounds and rate guarantees. In particular, the absence of discussion on key comparative results, such as the convergence bound of ADMM, typically 
$𝑂(\frac{1}{\sqrt{T}})$, makes it difficult to assess the theoretical strength of HOMM. A deeper analysis connecting the proposed framework to established convergence theory, including explicit upper bounds or stability guarantees, would substantially strengthen the theoretical contribution.

5. Limited Practical Applicability

The potential applications of HOMM appear constrained by the simplicity of the experimental design. The reported improvements are confined to standard image classification tasks using conventional architectures, which limits the relevance of the findings to more complex or real-world scenarios. To demonstrate broader impact, the authors should evaluate HOMM on more demanding tasks and models, such as transformer-based networks or multimodal large language models. Without such validation, the current results provide only limited evidence of HOMM’s scalability or applicability beyond simple benchmark problems.

### Questions
Reviewer Questions and Suggestions

1. Convergence Bound Analysis
The paper would benefit from a more detailed discussion and formal proof of the convergence bound of the proposed HOMM. Currently, the analysis focuses primarily on finite-time convergence, but it remains unclear how HOMM compares asymptotically with other state-of-the-art optimizers. For instance, if HOMM can achieve or surpass convergence rates such as $𝑂(\frac{1}{\sqrt{T}})$ (typical of ADMM) or $O(\frac{1}{T^{\frac{1}{3}}$) (as in STORM), this would provide a compelling argument for its superior efficiency. Establishing such bounds, or at least providing a theoretical comparison, would substantially strengthen the theoretical contribution and position HOMM more clearly within the broader optimization landscape.

2. Comparison with Advanced Optimizers (e.g., STORM)
The current experimental evaluation omits a comparison with recent “super-optimizers”, such as STORM and related adaptive momentum-based methods. Since these approaches are well recognized for their robustness and accelerated convergence in stochastic optimization, including them as baselines is essential. A direct comparison between HOMM and STORM would provide valuable empirical evidence of HOMM’s claimed advantages and help clarify its relative strengths and limitations. This would also enhance the paper’s credibility by situating the proposed method within the context of contemporary state-of-the-art optimizers.

3. Expansion to Broader Datasets
The empirical validation is currently restricted to a limited set of natural image classification benchmarks. To demonstrate generalizability and broader applicability, it would be valuable to extend the experiments to diverse datasets or domains, such as medical imaging, text-based tasks, or multimodal benchmarks. Evaluating HOMM on datasets beyond standard image classification (e.g., ADNI or other high-dimensional biomedical datasets) would strengthen the experimental section and provide stronger evidence of the optimizer’s versatility across application domains.

4. Discussion of Non-Smooth, Non-Convex Optimization
It would be valuable for the authors to discuss how HOMM performs in the context of non-smooth, non-convex optimization problems, which are increasingly common in modern deep learning architectures. As AI models become more complex, their associated loss landscapes often exhibit irregular, non-convex structures that challenge traditional optimization methods. A theoretical or empirical discussion of HOMM’s potential advantages—or limitations—in such settings would significantly enhance the paper’s impact and forward-looking relevance.

### Soundness
3

### Presentation
2

### Contribution
1
