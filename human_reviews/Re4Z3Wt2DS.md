# Variational Mirror Descent for Robust Learning in Schrödinger Bridge

- Avg Score: 6.80
- Decision: Reject
- Scores: 5, 8, 5, 8, 8

## Abstract
Schrödinger bridge (SB) has evolved into a universal class of probabilistic generative models. Recent studies regarding the Sinkhorn algorithm through mirror descent (MD) have gained attention, revealing geometric insights into solution acquisition of the SB problems. In this paper, we propose a variational online MD framework for the SB problems, which provides further stability to SB solvers. We formally prove convergence and a regret bound $\mathcal{O}(\textrm{\small$\sqrt{T}$})$ of online mirror descent under mild assumptions. As a result of analysis, we propose a simulation-free SB algorithm called Variational Mirrored Schrödinger Bridge (VMSB) by utilizing the Wasserstein-Fisher-Rao geometry of the Gaussian mixture parameterization for Schrödinger potentials. Based on the Wasserstein gradient flow theory, our variational MD framework offers tractable gradient-based learning dynamics that precisely approximate a subsequent update. We demonstrate the performance of the proposed VMSB algorithm in an extensive suite of benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a new numerical solver for learning Schrodinger bridge, a new class of generative models that generalizes denoising diffusions. Specifically, the authors built upon the _known_ the connection between Sinkhorn/SB method and Mirror Descent, extending it to solving SB, while presenting strong theoretical results. An algorithm based on Gaussian mixture approximation was proposed and validated on 2D synthetic datasets, single-cell dataset, and latent-space image translation.

### Strengths
- Theoretical contributions are solid. I think many of their results in Sec 5 may as well be of interested for readers from other domains.

### Weaknesses
- On the theoretical side, most of the results in Sec 4 seem to be adopted from Karimi et al., 2024 [1]. Can the author clarify what's newly proposed in Sec 4?

- To me, the main contribution lies in Sec 5, where the author proposed an approximate algorithm for the Wasserstein flow using mixture of Gaussians. This line of "mixture-of-Gaussian approximation" was initially proposed as a (1) fast and (2) non-NN-based alternative to solving SB problems, at the cost of scalability to high-dimensional applications. However, neither advantages were highlighted in empirical evaluation. If the proposed method still requires NNs (for constructing $\pi_\phi$, I suppose, and surely in adv training), it's unclear why we should still consider mixture of Gaussian models. 

- Following the previous point, I suspect this scalability of mixture Gaussian models is the reason why the empirical evaluation has been limited to benchmarks on 2D synthetic dataset and single-cell dataset, which, in my opinion, are no longer suitable for benchmarking SB methods (most methods yield visually indistinguishable learning results ...), and image translation yet on low-dim latent space or adversarial learning. Table 4 indicates the proposed method does not surpass DSBM-IMF. If a discriminator is required for training, it's unclear why we should still consider non-NN SB solver. 

[1] Sinkhorn Flow as Mirror Flow: A Continuous-Time Framework for Generalizing the Sinkhorn Algorithm

### Questions
- What exactly is the difference between VMSB and VMSB-M algorithmically, the paragraph starting L445 is very unclear. Given their marginal performance differences, if not worse, I see no points of introducing VMSB-M except creating additional reading burden for reviewers.

- How exactly is $\phi$ being optimized? That one sentence in L394 seems overly simplified to me.

### Soundness
2

### Presentation
2

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
This paper proposed a novel variational online mirror descent algorithm for Schrodinger Bridge motivated by learning theory and justified by numerical experiments. Authors first provide a new perspective which formulates the MD as Wasserstein gradient flow using the SB cost. Based on this, authors further proposed a variational OMD for the SB problem.

### Strengths
This paper proposed a simulation-free variational OMD for solving SB problems based the Wasserstein-Fisher-Rao geometries of Gaussian Mixture. Rigorous theoretical results together with numerical results show this new method performs more robust with reduced uncertainties compared to previous methods.

### Weaknesses
I think the theoretical results, numerical results and presentations are all good.

### Questions
One advantage of this method is improved robustness and reduced uncertainty compare to other methods. From theoretical perspective, this is justified by the regret bound. Intuitively, this is due to the convexity of SB cost used for VOMD, compared to the non-convex nature of the other methods. I wonder if it's possible to show directly that the proposed method is better than other method under some situations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors present a practical implementation of the recent generalization [1] of Iterative Proportional Fitting (IPF) within the Mirror Descent (MD) framework for Schrödinger bridges (SB). This method utilizes a newly introduced Gaussian Mixture Model (GMM) parametrization for SB [2], enabling closed-form iterations in the MD process. The paper also includes a convergence analysis of the proposed algorithm.

**Post-rebuttal**. I keep my score as "borderline reject". The last authors' response to me answered my question 3, but I still have serious concerns regarding question 1. In fact, the newly provided results by the authors (nearest neightbors to generated samples in the train data), in my view, clearly indicate an overfit to data. Therefore, I really doubt the generative capabilities of the proposed GMM-based model. I feel a little bit like the authors try to sell something here which does not really work. I believe that this section in the paper may be misleading to the ML community and probably should be removed. If so, the paper should be judged by the rest results. The rest results are theoretically interesting, but it seems like in practice they lead to very minor improvements at the cost of increase of the computational time, so I think my score is fair.

### Strengths
The authors improve the convergence stability of the LightSB solver [2] by incorporating nonstationary estimates of conditional plans into the training objective, showcasing this enhancement through extensive experimental results. Additionally, they conduct some theoretical analysis of the proposed algorithm (provide a regret bound).

### Weaknesses
- *Clarity should be improved.* The paper’s overall structure is difficult to follow, and the notation is particularly challenging—especially compared to the clarity in the original works [1], [2], [5]. For instance, the authors use the sequence notation ${a_t}_{t=1}^\infty = {0, 1, \dots}$ to mean ${0, 1, 0, 1, \dots}$ instead of the standard sequence ${0, 1, 2, 3, \dots}$. Additionally, they reference equations that have not yet been introduced in the text. The manuscript also contains spelling and typographical errors, such as “OT problems with different volatility and .,” “impotence” instead of “idempotence,” and “bye the following arguments” instead of “by the following arguments.” Moreover, the definition of Maximum Mean Discrepancy (line 460) is incorrect: the formula introduces just the L2 distance between probability density functions. The part with adversarial training in the main text looks alien and it is not particularly clear how it fits to the proposed framework.

- *EMA baseline is omitted.* The authors appear to use smoothing via Mirror Descent (MD) in a way that resembles standard Exponential Moving Average (EMA) approach, which is a common trick in deep learning [6]. Given this, the paper lacks a comparison with EMA smoothing.

- *Theory is a combination of prior works.* One may argue that the contribution lacks substantial originality, as it primarily combines methods from three existing works [1], [2], [5]. Additionally, while the original work [1] suggests various adaptations for large-scale problems (e.g., image processing) in Appendix C, this paper opts for Gaussian Mixture Models, which, as highlighted in [2], face limitations when applied to large-scale scenarios. Furthermore, the connection between the theory of gradient flows and this natural smoothing concept is not clearly established.

- *Practical gain is not convincing.* While the authors claim applicability to high-dimensional problems, their demonstration is limited to images in a 512-dimensional latent space. Moreover, the improvement on the EOT benchmark [4] as well as in many other cases is minimal. At the same time, the convergence time as stated in Appendix D is about 10-30 minutes on GPU instead of few minutes on CPU as it was originally stated for LightSB [2]. Authors do not address exploring convergence time on different applications, which could be slower in real-life applications compared to carefully tuned optimizers like Adam [3].

### Questions
Please consider providing answers to the following questions:

- How could the authors justify the significance of improving the metrics in the considered experimental setups taking into account that this is achieved by a significantly increased computational time?
- Could you please provide a detailed experimental comparison with the standard EMA approach for smoothing the model training? Does the proposed clever smoothing really outperform the standard approach?
- Any models based on Gaussian mixtures are known to suffer from the mode collapse in the sense that only one component in the mixture is actually active. Does your method address this problem somehow?

I think clarifying these questions is of high importance because your method risks being overshadowed by baseline [2], which features an out-of-the-box minimization objective, operates with simplicity, and offers significantly faster computational speed.

**References**

[1] Karimi, Mohammad Reza, Ya-Ping Hsieh, and Andreas Krause. "Sinkhorn Flow as Mirror Flow: A Continuous-Time Framework for Generalizing the Sinkhorn Algorithm." In International Conference on Artificial Intelligence and Statistics, pp. 4186-4194. PMLR, 2024.

[2] Korotin, Alexander, Nikita Gushchin, and Evgeny Burnaev. "Light Schrödinger Bridge." In The Twelfth International Conference on Learning Representations.

[3] Kingma, Diederik P. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).

[4] Gushchin, Nikita, Alexander Kolesov, Petr Mokrov, Polina Karpikova, Andrei Spiridonov, Evgeny Burnaev, and Alexander Korotin. "Building the bridge of schrödinger: A continuous entropic optimal transport benchmark." Advances in Neural Information Processing Systems 36 (2023): 18932-18963.

[5] Lambert, Marc, Sinho Chewi, Francis Bach, Silvère Bonnabel, and Philippe Rigollet. "Variational inference via Wasserstein gradient flows." Advances in Neural Information Processing Systems 35 (2022): 14434-14447.

[6] Morales-Brotons, Daniel, Thijs Vogels, and Hadrien Hendrikx. "Exponential moving average of weights in deep learning: Dynamics and benefits." Transactions on Machine Learning Research (2024).

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a novel approach to solving Schrödinger bridge (SB) problems in probabilistic generative modeling by proposing the Variational Mirrored Schrödinger Bridge (VMSB) algorithm. The experimental results show the VMSB algorithm consistently outperforms prior SB solvers in terms of stability and convergence across multiple generative modeling benchmarks.

### Strengths
1.It is a novel framework which reformulates SB problems using variational principles and online mirror descent (OMD). This results in a robust, simulation-free algorithm with theoretical guarantees. 
2.The theoretical is solid. The paper establishes a convergence and regret bound for their OMD method, showing that it attains a regret bound, signifying efficient learning dynamics.
3.The experiments empirically validate VMSB’s performance in various tasks, including synthetic data generation and high-dimensional datasets, highlighting its robustness over existing methods.

### Weaknesses
1.The paper is heavily focused on theoretical descriptions, which can make it challenging to follow.
2.The WFR gradient decent algorithm is relatively simple; incorporating higher-order methods could enhance the efficiency of inner iterations.

### Questions
I am open to raising my rating if the authors can address the following question:
1. Could you conduct experiments to demonstrate the efficiency of this simulation-free model compared to traditional models? For instance, showing a comparison of time versus performance metrics would be helpful.
2. Could you give more detailed descriptions of the models used for each experiment?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work builds on the recent connection between numerical algorithms for solving Schrodinger Bridge (SB) / EOT problems (Sinkhorn / IPF) and Mirrror Descent (MD).

The focus is on the online setting, where data becomes available as a stream, hence the extension to Online Mirror Descent.

Theoretical results regarding the proposal, under an idealized and more realistic settings, are provided in Section 4.2, while the computational and algorithmic implementation aspects are covered in Sections 4.3, and 5.1-5.2.

As noted, the confidence in my assignment is low (i.e. it is difficult for me to accurately judge the significance of the contributions), due to my limited familiarity with various prerequisites.

### Strengths
I enjoyed the focus on MD, i.e. equation (6), as a mean to derive existing and novel algorithms, which complements the explorations of Karimi et al, and further advances the field in this research direction.
The experimental results shows improvements across the tested scenarios, and are complemented by theoretical convergence results.
The implementation of the proposed algorithm rests on connections with Wasserstein gradient flow (specifically, in the setting of Gaussian measures), which are of interest in their own.
The writing style is formal and for the most part everything is clearly and precisely defined.

### Weaknesses
My main criticism of this work concerns the fact that it consists of a very dense presentation.
The paper covers a lot of ground, with topics ranging from the classical static and dynamics formulation of SB, to mirror descent (and related concepts for the infinite-dimensional version considered), to connections between SB and mirror descent, to Wasserstein flows and Otto's calculus.

I am well aware that the 10-pages limit makes it difficult to properly introduce all these concepts.
At the same time, unless the reader already has a (more than passing) familiarity with all the mentioned concepts, the paper and its contributions are quite difficult to parse, as it will become evident from my questions in the following section.
In particular, I found it difficult to identify the fundamental components making up the proposed methodology and its concrete implementation (i.e. what absolutely should be understood), and what main assumptions / approximations stand behind them.

My personal suggestion, would be to delegate (or shorten) some parts currently present in the manuscript to the appendices (on a side note: the appendices already contain quite a lot of interesting "introductory" material, a fact that is not so evident while reading the main text), that would not excessively impact the presentation, such as (these are just some possibilities):
- (1) and the part surrounding (1), which remains too abstract
- The results of 4.2 could maybe be stated as a single  informal (in the sense not detailing all the required conditions, with the current version moved to the appendix) Theorem
- The LightSB specific formulation (in practice it is a Gaussian mixture parametrization of the conditional static transport map)
- Proposition 2 as well (the specific form that enters (14) does not seem strictly necessary for understanding)

And use this space to better highlight key aspects of the proposal, and required concepts necessary to understand it.
I am not advocating turning the main text to an "Executive Summary", but I feel some improved balance can be found with the goal of expanding the intended target audience.

### Questions
To start with, I think I managed to follow, with some questions, the developments up to Lemma 2. Regarding this part:

1. I could not completely make sense of Figure 1. Is it meant to convey that the IPF iterations match the marginal distributions one at a time, while the proposed method lives somewhere else in the space 𝒞, while still converging to the EOT solution? (In this case, why is π_0 for IPF not on one of the red/blue curves)? 

2. Relatedly: I was surprised that in Lemma2 the dynamics for π_t are expressed in terms of the conditional distribution π_t(•|x), should the LHS of the stated limit also be the conditional version? Does the stated result also result in an ODE flow limit as in (Karimi et al. ,eq (13))?

3. Equation (6) plays a central role, with different F_t resulting in different algorithms. In all cases, the proximity term remains fixed. Is there a fundamental intuitive reason for this? Can the authors envision further applications of (6) that would be relevant for SB problems?

4. (Minor): I got tricked for a moment into a completing the sequence α_t as {0, 1, 2, ...} around line 209.]

5. If I understood correctly, there are two challenges into applying Lemma2: how to implement the Wasserstein Gradient flow, and we do not have access to π*. Is this correct?

6. While I agree that "However, learning methods of SB remain somewhat atypical, each requiring a sophisticated approach to derive solutions.", some progress in that direction has been achieved in [1], [2]. While these works do not target the streaming setting specifically, they seem more suitable than iterative methods such as IPF, so they might be a reasonable baseline to show improvements upon?

Section 4.2 onward was more difficult to parse.

7. In Section 4.2, the idea is to substitute for π* with an (evolving) estimate π˚_t, and in this section assumptions are given regarding this approximating process such that, with some choice of step-sizes, convergence still holds....can the authors expand a bit on the high-level idea of the approach and on the result? What is the reference estimation fitted using a Monte Carlo method, is it π˚_t? To what extent can the assumptions be verified when applying the proposed method?

8. In Section 4.3, what is ρ_t? Is ε_t() given by f() in Lemma2 with π˚_t instead of π*? So the relevance of (8) is that, applied to Lemma2 with π˚_t  allows us to leverage Theorem 3's characterization, that then yields Algorithm 1 by leveraging the explicit form of (14) that is due to the use LightSB?

9. Is the approximation in (14) due exclusively to the use of the Mixture of Gaussian coupling parametrization from LightSB?

10. The method, like LightSB, centers around the static setting. Can the results of this work also yield some insights for methods employing neural networks to solve the dynamic version of SB, or is it possible to envision extensions of the present work in that direction?

11. What is the author's intuition on the superior performance of the proposed method in the non-streaming EOT benchmark? Is it because the proposed algorithm results in some kind of smoothing in the descent iterates? Would employing EWMA (exponentially weighted moving average scheme) for the parameters' updates in LightSB yield a similar effect?

[1] Schrödinger Bridge Flow for Unpaired Data Translation (https://www.arxiv.org/abs/2409.09347)

[2] BM2: Coupled Schrödinger Bridge Matching (https://www.arxiv.org/abs/2409.09376)

### Soundness
3

### Presentation
3

### Contribution
3
