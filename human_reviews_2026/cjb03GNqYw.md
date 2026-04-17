# SoFlow: Solution Flow Models for One-Step Generative Modeling

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
The multi-step denoising process in diffusion and Flow Matching models causes major efficiency issues, which motivates research on few-step generation. We present Solution Flow Models (SoFlow), a framework for one-step generation from scratch. By analyzing the relationship between the velocity function and the solution function of the velocity ordinary differential equation (ODE), we propose a Flow Matching loss and a solution consistency loss to train our models. The Flow Matching loss allows our models to provide estimated velocity fields for Classifier-Free Guidance (CFG) during training, which improves generation performance. Notably, our consistency loss does not require the calculation of the Jacobian-vector product (JVP), a common requirement in recent works that is not well-optimized in deep learning frameworks like PyTorch. Experimental results indicate that, when trained from scratch using the same Diffusion Transformer (DiT) architecture and an equal number of training epochs, our models achieve better FID-50K scores than MeanFlow models on the ImageNet 256x256 dataset.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
### **Motivation**
- The paper is motivated by the computational inefficiency of recent Consistency Models, which still rely on local supervision using Jacobian–Vector Products (JVPs) to enforce consistency of velocity fields at intermediate timesteps.  
- Such methods remain local approximations of ODE dynamics — they match instantaneous velocities rather than learning the global solution function that defines the full trajectory.  
- To overcome this, the authors propose directly modeling the ODE solution itself with a neural network, aiming to bypass both the high JVP cost and the reliance on multi-step numerical integration or teacher distillation.


### **Methods**
- Approximates the true ODE solution using a neural network \( f_\theta(x_t, t, s) \) under sufficient conditions ensuring equivalence.  
- Introduces **Flow Matching Loss** (for boundary alignment at \( s=t \)) and **Solution Consistency Loss** (for temporal consistency across \( s<t \)).  
- Removes the need for JVP computation and integrates Classifier-Free Guidance (CFG) directly into training for guided one-step generation.

### **Contributions**
- Presents a solution-function–based training approach as an alternative to velocity-field supervision in existing flow and consistency models.  
- Describes a training-time CFG procedure (including variance-reducing velocity mixing) compatible with one-step inference. 
- Reports improved FID-50K on ImageNet 256×256 (3.35 vs. 3.43 for MeanFlow) using the same DiT-based backbone and cifar-10

### Strengths
- **[S1] Clear motivation and formulation:** The paper articulates a coherent motivation for moving from locally supervised velocity-field models toward a solution-function approach, addressing computational inefficiency in consistency-based methods.

- **[S2] Consistent empirical improvements:** SoFlow demonstrates FID gains across all MeanFlow model sizes on ImageNet-256×256, while maintaining stable one-step generation, indicating practical viability of the proposed approach

### Weaknesses
**[W1] Lack of theoretical grounding in global solution learning** 
 
SoFlow claims to learn the global ODE solution operator (Eq. (6)) directly, but its training relies only on local consistency objectives (eq. (14)). While this condition theoretically implies $ f_\theta = f $, the actual supervision—based on first-order Taylor approximations and stochastic sampling of $(t, l, s)$—remains inherently local, creating a gap between the theoretical formulation and practical optimization.

Traditional Flow Matching constructs a global velocity field via CFM, where local supervision is provably consistent with the true global dynamics. Similarly, Consistency Models enforce locally consistent trajectories using Jacobian-based constraints to form a globally coherent velocity field. In contrast, SoFlow extends this idea to the solution trajectory itself, but without a comparable theoretical foundation.

From a functional approximation perspective, to the best of my knowledge, there is no established theoretical evidence that a neural parametrization of the ODE solution trajectory can recover global behavior solely through such locally defined objectives. As this forms the paper’s main conceptual contribution, providing additional theoretical analysis or empirical validation would help substantiate the claim.

**[W2] Ambiguity in guidance supervision**

Unlike MeanFlow, which learns a local velocity field
$v_\theta(x, t) \approx \mathbb{E}_{p(x_0, x_1|x_t)}[\alpha_t' x_0 + \beta_t' x_1]$ and guarantees global trajectory consistency via explicit numerical integration,

SoFlow directly parameterizes a solution operator
$f_\theta(x_t, t, s) \approx x_t + \int_t^s v(f(x_t, t, u), u)du.$

This global mapping is nonlocal—it depends on the entire integral path of $v$—and therefore the training problem is fundamentally different in nature.

The proposed solution consistency loss (Eq. 14) is based on a first-order Taylor approximation:
$f_\theta(x_t, t, s) - f_\theta(x_t + v(x_t, t)(l - t), l, s) \propto \partial_1 f_\theta(x_t, t, s) v(x_t, t) + \partial_2 f_\theta(x_t, t, s),$
which supervises local temporal consistency between nearby timepoints $(t, l, s)$.
However, this formulation does not explicitly enforce
$\partial_s f_\theta(x_t, t, s) = v(f_\theta(x_t, t, s), s),$
the defining property that would make $f_\theta$ an integral curve of the true ODE.
Thus, even if Eq. (6) provides sufficient conditions for $f_\theta = f$,
those conditions are not directly constrained in practice, leaving a theoretical gap between local training objectives and the desired global solution behavior.

Moreover, under Classifier-Free Guidance (CFG), SoFlow supervises a guided trajectory
$f_\theta(x_t, t, s; c, w),$
implicitly assuming that
$\partial_s f_\theta(x_t, t, s; c, w) =v_g\big(f_\theta(x_t, t, s; c, w), s \mid c\big), \quad v_g(x, t \mid c) = wv(x, t \mid c) + (1-w)v(x, t),$
where the unconditional component $v(x, t)$ is itself an approximation.
In MeanFlow, this guided field $v_g$ remains a local and well-defined quantity.
In contrast, SoFlow assumes that the integrated, extrapolated trajectory is itself consistent with some valid conditional ODE solution—an assumption that is neither enforced by the objective nor theoretically guaranteed.

From a mathematical standpoint, the learning problem shifts from a well-posed local field approximation
to an ill-posed operator regression problem over nonlinear solution trajectories.
The paper would benefit from clarifying whether the learned $f_\theta$
is empirically observed to satisfy
$
\partial_s f_\theta(x_t, t, s; c, w)
\approx
v_g(f_\theta(x_t, t, s; c, w), s \mid c),
$
and whether such trajectories can be interpreted as genuine integral curves of a consistent guided flow field,
rather than heuristic extrapolations fitted by the loss.


**[W3] Limited evaluation of model performance and validity:**  

  The paper reports only FID scores, which primarily reflect perceptual quality but not distributional coverage or faithfulness to the learned dynamics. To properly assess whether the proposed solution operator $f_\theta$  approximates the true ODE solution and preserves the target data distribution, additional quantitative evaluations would be valuable — for instance:  
  - Inception Score (IS) or precision/recall to measure coverage and diversity.
  - Deviation or residual norms (e.g., $\| \partial_s f_\theta - v(f_\theta, s)\|$) to measure how closely the learned operator satisfies the underlying ODE.  

 Such analyses would better demonstrate whether SoFlow learns a genuinely faithful solution operator rather than merely optimizing perceptual similarity.

**[W4] Explanation for scheduling and mixing effects:**
 
  While Table 1(a)–(f) includes results for the $k$ scheduling (Eq. 16) and the velocity mix ratio $ m $, the paper provides only empirical trends without clarifying the underlying mechanism.  
  For instance, the paper states that a smaller $ m $ improves performance by reducing the randomness of the velocity term, but it is unclear whether this effect arises from variance reduction, regularization, or an implicit bias toward model-predicted guidance.  
  Similarly, the scheduling function $ r(k, K) $ in Eq. (16) is varied (exponential, cosine, linear), yet the text does not explain how the schedule theoretically affects optimization stability or convergence. 
 
Could the authors elaborate on why these heuristics influence performance — and whether the model remains stable and consistent under different schedules or $ m $ values?

**[W5] Comparsion table:**  
  While the proposed method focuses on one-step generation and demonstrates consistent improvements over MeanFlow across model sizes, the experimental comparison remains somewhat narrow.  
  For a fairer assessment of one-step efficiency and quality, it would be beneficial to include either (i) 2-step results of recent consistency-based models or (ii) one-step flow distillation models such as [1~5].
  Including these comparisons would allow readers to more accurately contextualize SoFlow’s efficiency–quality trade-off against both direct training and distilled few-step baselines.

[1] sCT, Simplifying, Stabilizing, and Scaling Continuous-Time Consistency Models., arxiv, 2024\
[2] CTM, Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion, ICLR, 2024\
[3] IMM, Inductive Moment Matching, ICML 2025.\
[4] Align Your Flow: Scaling Continuous-Time Flow Map Distillation, arxiv, 2025\
[5] Progressive Distillation, Progressive Distillation for Fast Sampling of Diffusion Models, Neurips, 2023

### Questions
Please refer to the “Weaknesses” section for detailed points.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed the SoFlow, a one-step generation paradigm for flow-matching diffusion models (FM). The main motivation for this paper is to directly approximate the unique solution of the ODE of the FM. The experimental results show that SoFlow could successfully achieve the one-step generation.

### Strengths
1. This paper proposes a novel method for one-step generation. This paper demonstrates how to approximate the unique solution of the ODE for the FM using sound theoretical proofs. Then, they build a learn target to train the model. In this way, one-step generation could be regarded as the solution from T to 0, thereby being directly solved via SoFlow. To the best of my knowledge, this method is the first one to achieve this and will truly benefit the development of FM. 

2. The experimental results clearly show that SoFlow works well.

### Weaknesses
1. I have to say that the writing of this paper should undergo a significant revision. Firstly, in Eq. 1, there is no explanation for $\alpha^{'}_t$ and $b^{'}_t$. Eq 5 should at least be split into two equations, not integrated into one. I highly recommend that the author add the integrated formulation of a unique solution. In this way, the reader could clearly know why $f(x_t, t, t)$ contains two time-related variables, one for the start and one for the end. Meanwhile, it may be better to expand Eq. 6 - Eq. 12 in the Appendix. This section is one of the most crucial parts of this paper and warrants a more detailed explanation. Meanwhile, this could simplify the audience's budget. Then, why is there no equation number in lines 192 and 202 lines.  The equation in 192 lines should also explain how it is derived. This is also important and directly related to your loss. Then, $\alpha$ and $b$ are redefined twice in 123 lines and again in Eq. 10, which may mislead the reader. In the end, the CFG section should reorder the logic. Currently, it is challenging for the reader to understand the rationale behind introducing a condition. The solution of the ODE itself does not contain the condition $c$. Meanwhile, the dataset itself cannot recognize the condition. This will lead the reader to question the validity of the CFG section.

2. I think the author should add a section to illustrate how to achieve distillation. Well, training from scratch is necessary, but distillation is also essential and can further enhance the contribution of this paper (can refer to the consistency models).

3. Experimental results in the Table. 1 has some inconsistency in the Table. 2. For example, the noise scheduler part shows the best FID is 59.97. But the true best FID is 3.35.

### Questions
1. How much influence does using the expectation of the velocity to replacing the velocity term? Is there any way to quantify the bias after replacing it with expectation?

To summarize, I believe this paper is novel. Approximating the ODE solution directly is an interesting approach. The experimental results prove its validity. But the writing should be clearer. The primary concern for me is the writing of this paper, which I believe does not meet the standards of ICLR. During the rebuttal, I recommend that the author improve the quality of their writing. Therefore, I rate it as marginally above the acceptance threshold, temporarily.

### Soundness
3

### Presentation
1

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This manuscript proposes a variant of consistency models, which can be trained from scratch without JVP and achieve better 1-NFE FID than MeanFlow on ImageNet 256x256. The method involves a combination of flow matching and consistency training loss with carefully designed scheduling, loss weighting, and CFG handling. The results are promising, showing that JVP is not the only way to achieve comparable results to MeanFlow on ImageNet.

### Strengths
- The ImageNet 1-NFE results of the proposed method is very strong, beating MeanFlow without using the inefficient JVP operation. This could imply a very positive message to the community: JVP may not be really necessary for strong performance in consistency training/distillation. Rather, a well-designed training objective with careful CFG handling could also achieve something very competitive. 
- All technical details (theories, design choices, ablation results) look very sound and the presentation quality is good. Notably, the introduction of velocity mix ratio is an interesting design choice, which clearly makes sense in terms of reducing variance and accelerating convergence.

### Weaknesses
- While I'm very positive on based on the evaluation results, this manuscript does not really introduce novel fundamental approaches that clearly distinguish it from prior art. For example:

  - The model architecture is similar to prior CM extensions that learns mappings from arbitrary t to s (e.g., CTM, MeanFlow)
  - Combining FM loss with consistency loss is also seen in prior work (e.g., CTM)
  - The CFG handling method can be broadly seen as an online guidance distillation approach (where the unconditional velocity is modeled by an unconditioned network prediction [1, 2]). MeanFlow also uses similar CFG handling (which is very important to FID).
  - For the consistency loss, the velocity mixing design is essentially mixing the consistency distillation objective (with $v_\text{guided}$ as the teacher, essentially it's online self distillation) with the consistency training objective (using real data).
  
  While this work does a very good job integrating all these techniques together, the limited novelty makes it difficult for a clear accept in my opinion.

- No multi-step results are presented in the paper. Multi-step generation is closer to applications of large-scale models. MeanFlow demonstrated very strong 2-NFE results. Beating it on 2-NFE could have further strengthened this work.

[1] Chen et al. Visual Generation Without Guidance. ICML 2025

[2] Tang et al. Diffusion Models without Classifier-free Guidance

### Questions
A few questions about design choices:
- Is a randomly sampled $s$ really necessary? For 1-NFE sampling, theoretically $s$ can just be zero. I wonder if there is a performance gain when using a random $s$ compared to setting $s=0$.
- Regarding the velocity mix ratio, $m=0$ is not tested in the ablation studies. Theoretically $m=0$ could work as long as there is a FM loss, basically the FM loss learns the velocity from data, and then the consistency loss at $m=0$ reduces to consistency distillation from the learned velocity (basically it's online self distillation).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper derive a loss for learning the flow map $f(x_t, t,s)=x_s$ following the supervision of flow matching. The authors closely follow the derivation of Align Your Flow [1] and MeanFlow [2]. The main contribution of the paper is showing that applying the linear approximation of a small step (i.e., $t\rightarrow s$) as in equation 13 is not needed and the flow map can be learn with a finite difference approximation instead as in equation 14.

[1] Sabour, Amirmojtaba, Sanja Fidler, and Karsten Kreis. "Align Your Flow: Scaling Continuous-Time Flow Map Distillation." arXiv preprint arXiv:2506.14603 (2025).

[2] Geng, Zhengyang, et al. "Mean flows for one-step generative modeling." arXiv preprint arXiv:2505.13447 (2025).

### Strengths
1. The paper is well written (with slightly confusing notation).

2. The proposed method is backed by theory.

3. The proposed method achieve state-of-the-results (though comparable with baselines).

4. Proposed method suggest that in practice using JVP in [1,2] is not needed, potentially saving training cost.

### Weaknesses
1. the proposed method is of low novelty:

    a. The presented derivation is very close to Align Your Flow [1].

    b. The proposed preconditioning (referred as "Euler parametrization" ) of the flow map $f_{\theta}(x_t,t,s) = x_t + (s-t)F_{\theta}(x_t,t,s)$ was already suggested by [1] and essentially results in making the actual network parameters $F_{\theta}$ learn the mean field as in [1].

    c. Equation 13 was already used by [1], and composed with the Euler parametrization returns the mean flow objective [2].

    d. Approximating $v_t(x_t,t)=\mathbb{E}[\alpha_t' x_0 + \beta_t' x_1]$ with $\alpha_t' x_0 + \beta_t' x_1$ inside the flow map as in equation 14 is justified only for small $t-s$ where marginalization trick [3] can be applied as done in [2].

2. The statement that the proposed method "consistently outperforms Meanflow" (line 425) is somewhat limited given that in Table 2 for DiT-L and DiT-XL the difference between SoFlow and MeanFlow is 0.12 and 0.08 FID (resp.). 

3. Though is seems the two method, SoFlow and MeanFlow, achieve comparable results, since SoFlow do not use JVP it can potentially reduce training cost which can be significant in large scale training. However, no empirical comparison or analysis of this is found in the paper.

4. No empirical validation of the method on the multistep setting. In particular the range 2-4 steps since previous work [1,2] reported state-of-the-art results in this range.

[3] Lipman, Yaron, et al. "Flow matching guide and code." arXiv preprint arXiv:2412.06264 (2024).

### Questions
Could the authors provide empirical comparison for training costs of SoFlow vs. MeanFlow ?

### Soundness
3

### Presentation
3

### Contribution
2
