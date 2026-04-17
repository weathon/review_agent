# Training-Free Stein Diffusion Guidance: Posterior Correction for Sampling Beyond High-Density Regions

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Training-free diffusion guidance provides a flexible way to leverage off-the-shelf classifiers without additional training. Yet, current approaches hinge on posterior approximations via Tweedie’s formula, which often yield unreliable guidance, particularly in low-density regions. Stochastic optimal control (SOC), in contrast, provides principled posterior simulation but is prohibitively expensive for fast sampling. In this work, we reconcile the strengths of these paradigms by introducing Stein Diffusion Guidance (SDG), a novel training-free framework grounded in a surrogate SOC objective. We establish a theoretical bound on the value function, demonstrating the necessity of correcting approximate posteriors to faithfully reflect true diffusion dynamics. Leveraging Stein variational inference, SDG identifies the steepest descent direction that minimizes the Kullback-Leibler divergence between approximate and true posteriors. By incorporating a principled Stein correction mechanism and a novel running cost functional, SDG enables effective guidance in low-density regions. Experiments on molecular low-density sampling tasks suggest that SDG consistently surpasses standard training-free guidance methods, highlighting its potential for broader diffusion-based sampling beyond high-density regions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles a core limitation of training-free diffusion guidance: many methods rely on Tweedie-based posterior approximations, which become unreliable in low-density regions. The authors propose Stein Diffusion Guidance (SDG), a training-free sampler that frames guidance via a surrogate SOC objective, corrects the Tweedie posterior toward the desired posterior using a Stein variational descent step. The method reports consistent gains on molecular low-density sampling benchmarks over standard training-free baselines.

### Strengths
* Clear exposition of the problem and its trade-offs. The authors articulate precisely why methods based solely on the Tweedie’s formula can misguide sampling especially when exploring low-density regions of the data manifold and why the alternative of a full Stochastic Optimal Control (SOC)-based approach, although theoretically sound, is prohibitively expensive for fast sampling. The proposed method neatly targets this gap: improving generative guidance while remaining efficient.

* Strong presentation and readability. The paper is well written, with clear definitions, motivating examples, and a logically structured.

* Well-motivated use of Stein Variational Gradient Descent (SVGD). Integrating SVGD for correcting the approximate posterior is not only technically sound but also conceptually elegant: it directly addresses the KL divergence gap between the approximate posterior (via Tweedie) and the true posterior, offering a principled mechanism to refine guidance.

### Weaknesses
* Motivation. The paper’s central premise is that targeted discovery that is, generating samples in low-density regions is a realistic and important use case. This assumption motivates their critique of the Tweedie’s formula as suboptimal in such regimes. However, the practical relevance of this assumption is debatable: in many real-world tasks, exploration into extremely low-density areas may not be desirable or even well-defined. Note that the authors introduce a KLD term to enforce that the learnd distribution doesn't deviate from the target. Balancing these tworequirements is not trivial.

* Novelty: The key technical contribution is the introduction of an SVGD-based correction to adjust the Tweedie posterior. While the idea is elegant and theoretically grounded, it represents an incremental conceptual advance rather than a major leap

* The SVGD density in Equation (4) can, in fact, be implemented more efficiently using only vector dot products and first-order derivatives, as demonstrated by Messaoud et al. (2024) [1]. This might alleviate the need for the proposed back and forth correction. 

* SVGD is known not to scale to high density region. Also setting the kernel bandwidth is not trivial. The authors only bring up the effect of particle size.

* Some symbols weren't define properly: p^{u}_t in proposition 3.1.

[1] Messaoud, Safa, et al. "S $^ 2$ AC: Energy-Based Reinforcement Learning with Stein Soft Actor Critic."ICLR (2024).

### Questions
* Kernel specification. Could you elaborate on the specific kernel functions and bandwidth selection criteria used in the Stein update? Additionally, how sensitive are the results to these choices — in particular, does performance degrade noticeably under different kernel families (e.g., RBF vs. IMQ) or bandwidth scaling strategies?

* Alternative KL computation. How would your derivation change if the KL divergence were computed explicitly using the closed-form expression for 𝑞 provided by Messaoud et al. (2024) [1]?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a training-free diffusion guidance algorithm to sample from a tilted distribution. To this end, the authors consider a stochastic optimal control formulation of the fine-tuning problem. By plugging in a special state cost and system dynamics, they can derive a variational bound of the value function splitting the control in to the traditional diffusion guidance (which does not yield the correct result) and some correction term. For numerical tractability they approximate the correction term, which is related to Stein Variation Gradient Descent. The paper shows a couple of examples for imaging applications and molecule sampling.

### Strengths
- The viewpoint of defining an upper bound of the value function in prop 3.2 and the splitting of the correspondin g control in (4) are quite nice. In particular, we get as an intermediate consequence of prop 3.2 that $V(x,t)=\min_q \bar V(x,t,q)$.

- The method remains training-free and includes SVGD into diffusion sampling.

### Weaknesses
The paper has serious issues including undefined notations, poor grammar, lack of formalism and technical errors. A couple of examples:

- Lemma 2.1: The corresponding text is not a sentence (please fix the grammar).

- Lemma 2.2: The corresponding text is not a sentence (please fix the grammar). The assumptions on the statement are missing. In particular, Nüsken & Richter require the function $f$ to be in $C^1$.

- What is $\delta$ (appears only in eqt (3) of the main text and the appendix)? From reading the proof of Prop 3.1 I guess the authors mean the delta distribution defined by $\delta(g) = g(0)$ for admissible test functions $g$.

- Proposition 3.1: Lemma 2.2 can't be applied here as it relies on the assumption that $f\in C^1$ (which the authors omitted for some reason). The $f$ from the authors is not even a function but rather a distribution.

- The statement of Tweedie formula in line 203 is weird. Tweedie says that the right hand side corresponds to $\mathbb{E}[x_T|x_t]$. The approximation happens by writeing $x_T\approx \mathbb{E}[x_T|x_t]$, which is obviously far off in most cases.

- $\mathcal Q$ is undefined

- Lemma 3.3: What means $\approx$ here? Can you show that $\text{LHS}-\text{RHS}<\text{something}$? Sorry, but this is not a formal statement of a lemma.

In summary, I think that the paper contains some promising ideas. However, the presentation is far below everything which is acceptable and there are plenty of informalities or technical errors. Therefore I suggest that the authors rewrite it, formalise the loose ends and resubmit it to the next conference.

Other comments:

- SDG without Stein correction is an obscure name (Stein without Stein)...

- The comparisons in Table 1 are choosen by cherry-picking. Why don't the authors include TFG or the other comparisons in from Table 3 in Ye et al., 2024? (Just because the usually perform better than the proposed method?)

- Despite the small batch size of 256 (which is really small for SVGD), the computational cost (14 min runtime + >18GB GPU memory according to Table 7) is quite expansive considering that the experiments are conducted on CIFAR10 which is comparibly small.

- It is not completely true that minmizing the SOC loss requires to differentiate the generation SDE. Serveral methods propose differentiation-free approaches which minimize the SOC loss indirectly, see e.g. https://arxiv.org/abs/2409.08861, https://arxiv.org/abs/2502.04468, https://arxiv.org/abs/2405.20971 (I noted that the first one is already cited; however the statement that it requires "backpropagating reward signals through entire neural-SDE sampling trajectories" is not true). The related work should be adjusted accordingly.

- It would be beneficial to visualize the method and its effects on a tractable 2D toy example, where distributions can actually be compared visually.

### Questions
- The ablation with the number of particles is nice. However, I am missing small batch sizes here. The authors correctly observe that using 3000 particles does not lead to better results than 512. What happens if you take less particles? From which point does it start to break? Is the story the same for the imaging examples?

- For the SVGD updates, which kernel do you use? How do you select the bandwidth? How sensitive are the results towards the bandwidth? Generally, this seems to be a critical point to me. Kernel-methods are often sensitive to the dimension of the underlying space. In particular for the image results this seems to be rather limiting.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Stein Diffusion Guidance (SDG), a novel training-free method for guiding diffusion models towards low-density, high-reward regions. The authors identify a limitation in existing training-free guidance methods, which rely on a biased posterior approximation via Tweedie's formula, leading to unreliable sampling, especially in low-density areas. To address this, they formulate the guidance problem within a surrogate Stochastic Optimal Control (SOC) framework. The key innovation is a "Stein correction" mechanism, derived from Stein Variational Gradient Descent (SVGD), which is applied in a "back-and-forth" manner to refine the approximate posterior samples, bringing them closer to the true diffusion posterior before applying reward-based guidance. Experiments on molecular design and standard image tasks demonstrate that SDG outperforms baselines, particularly in discovering novel, high-affinity drug candidates.

### Strengths
- The work presents a creative and novel combination of ideas from SOC and SVGD to address training-free diffusion guidance. The formulation of a surrogate SOC objective with a principled KL-divergence term and its optimization via a back-and-forth Stein correction is a non-obvious and original contribution.

- SDG provides a plug-and-play solution that can significantly enhance the reliability of training-free diffusion guidance when exploring low-density, high-value samples. Given that many scientific discovery tasks (such as drug and material design) rely on this capability, this work holds important practical implications.

- The experimental evaluation is thorough, spanning both a primary application (molecular generation) and general tasks (image guidance), with comprehensive ablation studies that validate the importance of each component (Stein correction, low-density annealing).

### Weaknesses
1. The motivation is not clear. My understanding is that the authors aim to sample from the joint distribution of $\log p_t$ and $r$, which is essentially a diffusion with guidance problem. Why one cannot directly sample based on the unregularized score $\nabla\log p_t + \lambda \nabla r$, where $\lambda$ is a coefficient controlling the guidance weight? Why is it necessary to devise a complicated SVGD + SOC method to solve this problem? Could the authors provide the intuition behind the design of this algorithm?

2. Noise Conditional Score Networks (NCSN) [1] were proposed to address low-density sampling issues, and SDE-based methods [2] further reinforced this. What are the key differences and advantages of the proposed method (SDG) compared to these existing methods?

3. The method inherits the known limitations of standard SVGD in high-dimensional spaces. The performance saturation with increasing particle size (Table 3) and the inherent instability of kernel-based updates in high dimensions are valid concerns.

4. I find Section 2 (Preliminaries) to be quite disjointed; the roles of SOC and SVGD are not clearly connected. Could the authors clearly explain in the introduction or summary of Section 2 the specific role of SVGD within the SOC framework, thus preparing the reader for the content of Section 3?

References:

[1] Generative Modeling by Estimating Gradients of the Data Distribution. NeurIPS 2019.

[2] Score-Based Generative Modeling through Stochastic Differential Equations. ICLR 2021.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies training‑free diffusion guidance in settings where the target samples lie in low‑density regions. The authors argue that widely used training‑free guidance based on Tweedie’s formula can mislead sampling off the true diffusion posterior, especially in low density, whereas SOC‑based posterior simulation is principled but slow.

### Strengths
- The paper addresses discovery of low‑density regions, a regime where off‑the‑shelf training‑free guidance tends to fail.
- SDG derives a surrogate SOC control and makes the case that approximate posteriors must be corrected to reflect true diffusion dynamics. The proposed Stein correction gives a principled way to correct Tweedie‑based posteriors.
- The results on molecular synthesis are compelling. The authors ablate on alpha and epsilon and on the low-density region tasks. 
- The method section is well structure, and the figures are informative.

### Weaknesses
- The main novelty seems to lie in how the posterior is corrected. The paper’s corollary explicitly shows SDG subsumes Langevin correction, which can make SDG feel like refined and more stable rather than fundamentally new. 
- SDG uses particles and reports increased runtime. For large N or high‑dimensional guidance, the method may be costly.
- If the camera‑ready added a strong large‑scale inverse problem in conditional image, text, or other domain study, I could see my ranking moving to an 8.

### Questions
- from your ablations, is the Stein correction or the running‑cost guidance the dominant contributor in low‑density discovery?
- What kernels did you use in the Stein operator and how sensitive are results to the kernel choice and particle count? 
- As continuous-state diffusion methods are more and more adapted to text generation, do you expect similar stability and benefits for classifier‑free text conditioning or for complex inverse problems?

### Soundness
3

### Presentation
4

### Contribution
2
