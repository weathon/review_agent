# Q-Learning with Fine-Grained Gap-Dependent Regret

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 8

## Abstract
We study fine-grained gap-dependent regret bounds for model-free reinforcement learning in episodic tabular Markov Decision Processes. Existing model-free algorithms achieve minimax worst-case regret, but their gap-dependent bounds remain coarse and fail to fully capture the structure of suboptimality gaps. To address this limitation, we establish fine-grained gap-dependent regret guarantees for both UCB-based and non-UCB-based algorithms. In the UCB-based setting, we develop a novel analytical framework that explicitly separates the analysis of optimal and suboptimal state-action pairs, yielding the first fine-grained regret upper bound for UCB-Hoeffding (Jin et al., 2018). In the non-UCB-based setting, we revisit the only existing algorithm, AMB (Xu et al., 2021), and identify two issues in its design and analysis: improper truncation in the $Q$-updates and violation of the martingale difference condition in the concentration argument. To resolve these issues, we propose two refinements of AMB: the UCB-based ULCB-Hoeffding and the non-UCB-based Refined AMB. For ULCB-Hoeffding, we establish the same fine-grained regret bound as UCB-Hoeffding by applying our fine-grained framework, highlighting its broad applicability. For Refined AMB, we derive a rigorous fine-grained gap-dependent regret bound in the non-UCB setting and demonstrate consistent empirical improvements over the original AMB.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents the  fine-grained, gap-dependent regret bounds for model-free reinforcement learning in episodic tabular Markov Decision Processes. While previous algorithms achieved minimax worst-case guarantees, their gap-dependent analyses were coarse, depending on the smallest suboptimality gap rather than individual ones. The authors introduce a new analytical framework that explicitly separates optimal and suboptimal state-action pairs, enabling fine-grained regret analysis. They apply this framework to the well-known UCB-Hoeffding algorithm, deriving a tighter bound that matches known lower bounds up to polynomial factors, and propose a simplified variant, ULCB-Hoeffding, which achieves similar theoretical guarantees with improved empirical performance. The paper also revisits the non-UCB-based AMB algorithm, identifying key theoretical flaws and proposing a refined version that restores correctness, ensures valid concentration analysis, and achieves a rigorous fine-grained regret bound in this class. Experimental results on synthetic MDPs confirm the theoretical findings, showing that the refined algorithms outperform prior methods. Overall, the work provides a unified framework that advances theoretical understanding of model-free reinforcement learning by bridging worst-case and instance-dependent analyses.

### Strengths
1. Introduces a fine-grained decomposition that separately analyzes optimal and suboptimal pairs, tightening gap-dependent bounds.
2. Framework applies to both UCB-based and non-UCB-based algorithms.
3. Provides the first fine-grained regret bound for model-free RL; matches known lower bounds up to polynomial factors.
4.  Identifies and corrects subtle theoretical flaws in prior work (AMB).
5. Experiments confirm theoretical improvements and show scalability across MDP sizes.

### Weaknesses
1. Results do not extend to function approximation or continuous-state RL.

2.  The theoretical derivations are mathematically heavy and might be difficult for practitioners to follow or generalize.

3.  Experiments are conducted on randomly generated MDPs rather than benchmark environments.

4.  The bounds, while asymptotically tight, may not yield practical improvements in all regimes.

5. The experimental comparison includes few competing algorithms beyond AMB and its variants.

### Questions
1.  Can the proposed fine-grained framework extend to function approximation (e.g., linear or neural models)?
2.  How do fine-grained regret improvements translate to real-world tasks (e.g., navigation, games)?
3.   Are the polynomial factors in H (e.g., H^5 or H^6) necessary, or could further refinement reduce them?
4.  Does the fine-grained analysis provide insight into exploration dynamics, beyond theoretical improvement?
5.  Could this framework be combined with variance-dependent or adaptive analyses to yield sharper or adaptive regret bounds?
6. Could ULCB-Hoeffding’s structure be adapted to other algorithms (e.g., Q-learning with linear approximation)?

### Soundness
3

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
4

### Summary
This paper aims to establish fine-grained, gap-dependent regret bounds for model-free algorithms in episodic tabular MDPs. While such bounds exist for model-based methods, model-free approaches have been limited to coarse bounds dependent on the global minimum gap, $\Delta_{min}$. The authors provide a two-part affirmative answer:

1.  For UCB-based algorithms, they develop an analytical framework that distinguishes between optimal and suboptimal state-action pairs. Using this, they derive the first fine-grained, gap-dependent regret bound for the classic UCB-Hoeffding algorithm.
2.  For non-UCB-based algorithms, they revisit the AMB algorithm (Xu et al., 2021), identifying and correcting significant algorithmic and analytical flaws to propose a Refined AMB. They then provide the first rigorous fine-grained bound for a non-UCB-based method.

Empirical results on synthetic MDPs validate the theory, showing that the refined algorithms outperform the original AMB.

### Strengths
1.  A key strength is the identification and correction of critical flaws in the AMB algorithm (Xu et al., 2021). The identification of improper truncation and violation of martingale difference conditions is a valuable and clear-cut contribution.
2.  The paper successfully applies a fine-grained analytical framework, separating optimal/suboptimal pairs,  to the model-free setting, yielding the first fine-grained, gap-dependent regret bound for the widely-known UCB-Hoeffding algorithm.
3.  The experiments clearly support the theory. The flawed AMB algorithm performs poorly, while the Refined AMB and UCB-Hoeffding all perform well and exhibit the logarithmic regret behavior predicted by the new theory.

### Weaknesses
1.  The paper's primary weakness is its failure to position its analytical framework relative to recent model-based works that also achieve fine-grained bounds, e.g., Dann et al., (2021); Chen et al., (2025). The paper cites these works but does not discuss the technical challenges of adapting their analysis to the model-free setting. Without this comparison, the core technical contribution in Section 3.3 appears to be an incremental adaptation rather than a novel framework.
2.  The derived bounds include $H^5$ and $H^6$ terms. While the focus is on the gap-dependence, this looseness in $H$ is a significant limitation of the current analysis and makes the bounds less tight, even if they match the lower bound except for the factors in $H$.
3.  The inclusion of the ULCB-Hoeffding algorithm feels unnecessary. It achieves the same theoretical bound as UCB-Hoeffding in Theorem 3.3, but performs noticeably worse in the experiments shown in Figure 1, distracting from the two stronger, clearer contributions.

### Questions
1. Please explicitly compare your analytical framework in Section 3.3 to the techniques used in model-based papers like Chen et al. (2025). What are the key technical novelties required to make this style of fine-grained analysis work for model-free Q-learning? What new challenges arise from this difference that your analysis overcomes?
2.  Can you elaborate on the source of the large polynomial dependence on $H$? Is this an artifact of the analysis, e.g., from recursively applying bounds, and do you see a path to tightening it?
3.  Could you provide a direct comparison between your final bound for Refined AMB (Eq 10, depending on $|Z_{mul}|$) and the bound for UCB-Hoeffding (depending on $|Z_{opt}|$)? Which bound is tighter, and under what conditions?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the online tabular RL problem.
The authors show that UCB-Hoeffding (Jin et al., 2018) achieves a fine-grained, gap-dependent logarithmic regret bound.
Such a result is the first among model-free algorithms.
While there is an exception of AMB (Xu et al., 2021), the authors point out errors in the analysis of AMB and propose how to fix it, providing a correct fine-grained regret bound and its proof for the algorithm.

### Strengths
The paper proposes a general framework for obtaining fine-grained, gap-dependent logarithmic regret bounds of model-free algorithms.

The raised issues for AMB and the corrected version seem valid.

### Weaknesses
Some core definitions for the analysis are not properly given.

- Is $\eta_ i^{N}$, introduced in line 310, $\eta_ i \Pi_ {j=i+1}^N (1 - \eta_ j)$ or the $N$-th power of $\eta_ i$? It seems like it's the former but I could not find the definition.

- What is the input state-action pair of $N_ h^k$ when written without one? It appears multiple times including in the definition of $\omega_ {h'+1}^k(h)$, but it is not clear from the context.

- It seems like the definition of $\omega_ {h'+1}^k(h)$ requires a state-action pair as $k^i$ requires a state-action pair in its definition. However, the notation does not reflect the fact. Also, there is a problem of using $N_ h^k$ without definition, so I have no idea what $\omega_ {h'}^k(h)$ is supposed to represent.
I hope there is a description about what $\omega_ {h'+1}^k(h)$ represents as it is hard to understand it intuitively.  
Partially due to these ambiguities, Lemma C.3 is not trivial. Also, it is really hard to see why the equation in line 372 is true. Could the authors explain how these equations are established?  
In addition, as I see that $\omega_ h^k = \mathbb{I}\lbrace (s_ h^k, a_ h^k) \in Z_ {\text{sub}, h}\rbrace$ in the end, which is a simple function, wouldn't plugging in this value from the beginning simplify the analysis a lot without needing to define multiple series of $\omega$?  

I will be happy to raise my score once these points are clarified.

### Questions
1. While a general framework for fine-grained, gap-dependent bounds for model-based algorithms is known (Simchowitz & Jamieson, 2019; Dann et al., 2021; Chen et al., 2025), I see that the analysis in this paper is different from the one in Simchowitz & Jamieson (2019). What challenges are there in applying the techniques in these papers to the model-free setting?

2. I don't understand why UCLB is proposed when it is no better than UCB-Hoeffding both theoretically and empirically. If the goal is to show the generality of the framework, couldn't any other model-free algorithm be used, for instance, UCB-Bernstein or UCB-Advantage?

3. In Section 4, it is mentioned that the analysis of Xu et al. (2021) violates the martingale property. It is due to the fact that $h'$ in their work is also random? It is not clear what this part is trying to claim. For instance, what are the values of the expectation and conditional expectation described in this paper?

4. Under what MDP was the experiment conducted?

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work establishes gap-dependent regret bounds for UCB-based model-free RL regarding individual suboptimality gaps $\Delta_h(s,a)$ instead of global one $\Delta_{\min}$. Besides, this work identifies the issues in analyzing the AMB algorithm, which is non-UCB-based model-free algorithm, on incorrectly applying concentration inequalities, and refines the algorithm and the analysis.

### Strengths
1 This work establishes the first and tight individual-gap-dependent regret bounds for model--free RL, which improves coarse global-gap-dependent coarse bound in the prior work. The core technical contribution lies in separating the analysis of of optimal and suboptimal state-action pairs.

2 This work refines and fixed the issues in the AMB algorithm and associated analysis, which provides a rigorous fine-grained regret bound for non-UCB-based algorithm.

### Weaknesses
1 The analysis is based on episodic tabular MDP. The theoretical guarantees do not extend to complex settings such as linear MDP, MDP with function approximation. 

2 Model-based algorithms has shown to achieve fine-grained gap-dependent regret bound. This work addresses a theoretical gap where model-free algorithms were lagging behind model-based algorithms in this setting. The contribution may not be significant.

3 The regret bound has high dependency on the horizon $H$.

### Questions
1 There lacks comparison between this work and model-based RL. For example: a) What is the technical difficulty in adapting techniques in model-based algorithms for deriving fine-grained gap-dependent regret bound? b) Comparison on the dependency on $H$, and potential improvement on $H$.

2 What is the advantage of ULCB-Hoeffding over UCB-Hoeffding? They achieve the same fine-grained gap-dependent regret bound and their analysis are also similar. Given that ULCB is considerably more complex, I don't see the necessity of introducing ULCB algorithm in the main context.

3 Comparing the last term in Eqn. (10) and Eqn (2), the non-UCB-based algorithm AMB achieves a better sample complexity in $H$. Is it due to the sharper analysis of AMB? 

Minor issues:

1* Line 226 "line 15 in Algorithm 2": line 15 should be replaced.

2* Line 310 Eqn. (5): inside indicator function, should $k$ be $k'$? The recursive definition is hard follow. It would be great if the authors could come up with intuitive way to explain the intuition if exists.

3* Line 951 "(1),." -> "(1)."

### Soundness
3

### Presentation
2

### Contribution
3
