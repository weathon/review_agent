# Distributionally Robust Optimization via Generative Ambiguity Modeling

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 2

## Abstract
This paper studies Distributionally Robust Optimization (DRO), a fundamental framework for enhancing the robustness and generalization of statistical learning and optimization. An effective ambiguity set for DRO must involve distributions that remain consistent to the nominal distribution while being diverse enough to account for a variety of potential scenarios. Moreover, it should lead to tractable DRO solutions. To this end, we propose generative model-based ambiguity sets that capture various adversarial distributions beyond the nominal support space while maintaining consistency with the nominal distribution. Building on this generative ambiguity modeling, we propose DRO with Generative Ambiguity Set (GAS-DRO), a tractable DRO algorithm that solves the inner maximization over the parameterized generative model space. We formally establish the stationary convergence performance of GAS-DRO. We implement GAS-DRO with a diffusion model and empirically demonstrate its superior Out-of-Distribution (OOD) generalization performance in ML tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the DRO problems with generative model-based ambiguity sets. For the problems with new constraints, a new framework is proposed to solve the problems and numerical results are provided.

### Strengths
The paper provides both theoretical and numerical results for the proposed methods.

### Weaknesses
The writing can be improved. For example, the $s_\theta$ used in line 204 is not previously defined.  The definitions of $J_{VAE}, R_{VAE}$ are important and should be highlighted under equation (6). The smooth and Lipschitz assumptions used in the Theorems should be defined.

### Questions
While the presentation contains some minor issues, my main concern lies in the motivation and conceptual clarity of this paper.

1. The paper argues that for general DRO problems, in the case of $\varphi$-divergence DRO, the target distribution must be absolutely continuous with respect to the nominal distribution, and that Wasserstein-DRO formulations are computationally complex. The authors then propose an upper bound on the KL divergence in Lemma 1. However, I find this result puzzling: the left-hand side is expressed as a general function of $P_\theta$, while the right-hand side depends only on $\theta$. This mismatch suggests that the authors implicitly treat $P_\theta$  as a parameterized distribution determined by $\theta$. Consequently, the results in this paper seem to apply only to specific parameterized models, e.g.,  VAEs or diffusion models, rather than to general optimization problems in the DRO framework.

2. Even within the context of VAEs and diffusion models, I am not convinced of the necessity of this approach. Lemma 1 shows that the upper bound on the KL divergence is linear in $J$ and, given $P_0$,  the gap between KL and J is fixed. Therefore, the results produced by the proposed method may not differ substantially from those obtained through traditional KL-based methods, especially since the constraint functions are linear and the KL divergence is typically easier to compute.

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
The paper addresses the problem of distributionally robust optimization for enhancing robustness in statistical learning under out-of-distribution scenarios. The main idea is to construct a novel ambiguity set using likelihood-based generative models, such as diffusion models or VAEs, which allows for distributions that are consistent with the nominal distribution while enabling support shifts and tractable solutions. Key results include the GAS-DRO algorithm (Algorithm 1), which solves the minimax problem via dual learning and policy optimization. Theoretical contributions comprise Theorem 1, establishing convergence of the inner maximization to the optimal oracle with bounded KL divergence to the nominal distribution, and Theorem 2, proving stationary convergence of GAS-DRO under smoothness and Lipschitz assumptions. Empirical results on Electricity Maps datasets show superior OOD generalization compared to baselines like W-DRO, KL-DRO, and DRAGEN.

## Clarity

The paper is very well-written and clearly structured. The introduction provides excellent motivation by clearly outlining the limitations of existing DRO frameworks. The proposed method (GAS) is developed logically, building from the problem setup (Sec 3) to the generative model formulationand the algorithm.

## Originality / Novelty

This work is highly original. The central idea of defining a DRO ambiguity set using a generative model's reconstruction loss is, to my knowledge, new.

### Strengths
- The core idea of formulating the ambiguity set over the parameters of a generative model and, crucially, using the *reconstruction loss* $J(\theta, P_0) \le \epsilon$ as the constraint (Eq. 7) is a highly novel and elegant contribution. It reframes the problem from an intractable search over distributions to a tractable optimization over model parameters.
  
- The method provides a compelling solution to the fundamental tension between $\phi$-divergence DRO (no support shift) and Wasserstein-DRO (intractable/conservative). The use of generative models inherently allows for sampling beyond the nominal support space, while the constraint $J(\theta, P_0)$ ensures consistency with the nominal distribution.
  
- The paper proposes a concrete, end-to-end algorithm (Algorithm 1). The application of dual learning and policy optimization to solve the complex inner loop is a clever technical contribution that makes the framework practical.
  
- The authors provide non-trivial convergence guarantees for their proposed algorithm. Theorem 1 establishes the convergence of the inner-loop maximization oracle, and Theorem 2 proves stationary convergence for the full min-max procedure, lending strong theoretical weight to the method.
  
- The experiments, summarized in Table 1, show a clear and substantial performance improvement for GAS-DRO over all baselines. It achieves the lowest average error (0.0163) by a significant margin over the next-best baseline, DRAGEN (0.0230), and standard ML (0.0450). The ablation studies in Appendix D.2 are also thorough.

### Weaknesses
- The theoretical link (Lemma 1) between the proposed constraint $J(\theta, P_0) \le \epsilon$ and traditional divergence-based ambiguity sets is a one-way bound ($J \le \epsilon \implies D_{KL}(P_0 || P_\theta)$ is bounded). This ensures distributions in GAS are "consistent," but it's not clear if the set is sufficiently expressive. It's possible that a "worst-case" distribution $P^*$ that is close to $P_0$ (in a standard metric) might not be representable by any $P_\theta$ in the GAS, even with a powerful generative model. The paper relies on the "strong capability" of generative models, but this is not formally characterized.
  
- The analysis assumes bounded reconstruction loss and specific conditions on objectives (e.g., $\beta$-smooth, L-Lipschitz), which may not hold in all deep learning settings; this is a theoretical concern (Sec. 6, Thm. 2).
  
- The paper doesn't explore the connection between the adversary budget $\epsilon$ with regular ambiguity radius in DRO, like as $\mathcal{B}_\delta(P_0)$.

## Minor:

1. correct below typos:
  
  - “support **shit** testing cases”
    
  - “…restrict **the the** search of worst-case distributions…”
    
  - “The convergence of the Moreau **envelop** gradient norm…”
    

2. Repeated references like Staib & Jegelka (2017a)/(2017b) without disambiguation in prose.
  
3. "Ma et al. Ma et al. (2024)" — The citation "Ma et al." is repeated. Also "Wang et al. Wang et al. (2025)" — The citation "Wang et al." is repeated.

### Questions
Lemma 1 is doing a lot of work: it’s the bridge from small reconstruction loss” to bounded $D_{\mathrm{KL}}(P_0 | P_\theta)$.
But in the main text, the statement is fairly high-level:

    - It asserts $D_{\mathrm{KL}}(P_0 | P_\theta) \le J(\theta,P_0)+R(p',P_0)+C_1$, where $R(p',P_0)$ depends on the prior and $C_1$ depends on $-H(P_0)$.

    - It then argues that this allows support shift (since it’s the *reverse* KL, which only needs $P_0 \ll P_\theta)$. 
However, several details are pushed to Appendix C.1 and are not visible in the main body we have:

- Are there assumptions on model capacity (i.e., is $P_\theta$ assumed expressive enough to approximate $P_0$ arbitrarily well)?
- Are there bounds on $R(p',P_0)$ in practice, especially for diffusion, where $p'$ is a fixed Gaussian prior?
- Is $C_1$ finite when $P_0$ is empirical with finite support (so $-H(P_0)$ is well-defined)?
  These questions matter because if $R(p',P_0)$ is large or uncontrolled, then “bounded reconstruction loss ⇒ bounded KL” may not actually constrain the adversary meaningfully. Right now, the story is intuitive but not fully airtight in the visible text.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel approach to Distributionally Robust Optimization (DRO) that leverages generative models to define the ambiguity set of distributions. Instead of using a divergence or Wasserstein ball, the ambiguity set is parameterized by a generative model (e.g. a diffusion model or VAE) which can produce adversarial distributions beyond the support of the nominal data while still remaining similar to it. The authors introduce an algorithm called GAS-DRO (Generative Ambiguity Set DRO) that alternates between an inner loop (updating the generative model to find a worst-case distribution) and an outer loop (updating the primary model/decision variable on data sampled from that worst-case distribution). They enforce a constraint (via a dual Lagrange multiplier) to keep the generative distribution within a certain “distance” (ambiguity budget *ε*) of the nominal distribution. The paper provides theoretical guarantees – proving that the inner maximization converges to the optimal worst-case distribution and that the overall iterative procedure converges to a stationary point. Empirically, GAS-DRO is implemented with a diffusion model as the adversary and tested on a time-series forecasting task under distribution shift (electricity grid emissions data). It achieves significant improvements in out-of-distribution (OOD) generalization, outperforming several baselines like standard DRO with KL or Wasserstein ambiguity sets and a recent generative DRO method (DRAGEN). In summary, the paper’s main contributions are the introduction of generative-model-based ambiguity sets for DRO, a corresponding optimization algorithm with convergence analysis, and demonstration of improved robustness on real-world data.

### Strengths
1. The paper introduces a generative-model-based ambiguity set for DRO, which allows considering distributions outside the original support while maintaining similarity to the nominal distribution. The approach does not require too much prior condition and addresses the limitations of KL-divergence ambiguity (no support shift) and improves over Wasserstein ambiguity by providing a tractable, parameterized search space. 
2. Empirical OOD Performance Gains: GAS-DRO demonstrates state-of-the-art OOD generalization performance on the Electricity Maps time-series datasets.
3. The writing is straightforward and easy to understand.

### Weaknesses
1. This paper investigates DRO, which is widespread considered as an approach for discriminative tasks. The method in the paper instead uses a generative method to address the classification problem but determining ambiguity set itself needs the sampling operation. Therefore, I wonder how much extra compuational cost this method has introduced compared to normal plug-and-play baselines? For instance, how much longer (in terms of training time or iterations) does GAS-DRO take compared to standard DRO or ERM on the same task?
2. How sensitive is GAS-DRO to the choice and quality of the generative model? For example, if one used a smaller diffusion model or a different type of generator (like a Variational Autoencoder, as mentioned, or even a GAN), would the method still perform as well? Any insight into how approximation errors in the generative model impact the robustness of the learned model would be valuable.
3. The empirical evaluation, while thorough for the chosen task, is focused on a single domain (renewable energy time-series forecasting on Electricity Maps data). It would strengthen the paper to see results in other domains to confirm that GAS-DRO’s benefits hold broadly. The current evidence of superior performance is strong for the scenario presented, but the approach's generality would be more convincing with a wider range of experiments.
4. Maybe adding more discussions on the use of generative modeling in common classification or OOD problems will be better. For example, a diffusion classifier [1] is adopted for image classification. Although authors have presented related work on IID settings, the description of OOD is somewhat lacking [2, 3].

[1] Li, Alexander C., et al. "Your diffusion model is secretly a zero-shot classifier." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2023.
[2] Tong, Yunze, et al. "Latent Score-Based Reweighting for Robust Classification on Imbalanced Tabular Data." *Forty-second International Conference on Machine Learning*.
[3] Zhang, Hengrui, et al. "Mixed-type tabular data synthesis with score-based diffusion in latent space." *arXiv preprint arXiv:2310.09656* (2023).

### Questions
Please see Weaknesses.

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
3

### Summary
This paper proposes GAS-DRO. It uses diffusion models and VAEs to build ambiguity sets for DRO. The method constrains the reconstruction loss of generative models instead of using KL divergence or Wasserstein distance. The authors provide convergence guarantees and test on time series forecasting tasks.

### Strengths
1. The paper identifies real problems with existing DRO methods. φ-divergence restricts support. Wasserstein distance is hard to optimize.

2. Theorems 1 and 2 provide convergence guarantees. Lemma 1 connects reconstruction loss to KL divergence.

3. GAS-DRO achieves 63.7% improvement over baseline ML. It outperforms existing DRO methods by significant margins.

### Weaknesses
1. Paper claims "first to model ambiguity sets in parameterized space of likelihood-based generative models" (page 2). This is incorrect. Michel et al. (2021) "Modeling the Second Player in Distributionally Robust Optimization" (ICLR 2021) already proposed Parametric DRO (P-DRO) that: (a) uses neural generative models q_ψ for adversary, (b) parameterizes uncertainty set with model weights ψ, (c) uses likelihood-based Transformers (evaluates log q_ψ(x,y) in Equation 8), (d) solves same min-max game structure, (e) tests on NLP tasks. The paper completely omits this citation. The distinction between "implicit" (GAN) and "explicit" (diffusion) generative models is insufficient—Michel et al. already used likelihood-based models. After removing false claims, only incremental contributions remain: using diffusion instead of Transformers, reconstruction loss instead of KL constraint, and time series application. No direct comparison with P-DRO is provided.

2. Limited experimental scope. Only one domain (time series forecasting). Michel et al. tested sentiment classification and toxicity detection. Need broader evaluation across vision or NLP tasks.

3. No computational cost analysis. Diffusion models are expensive. No runtime, memory, or scalability comparisons provided.

### Questions
See the weakness

### Soundness
3

### Presentation
2

### Contribution
2
