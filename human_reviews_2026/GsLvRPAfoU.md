# Improving Fine-Grained Control via Aggregation of Multiple Diffusion Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
While many diffusion models perform well when controlling particular aspects such as style, character, and interaction, they struggle with fine-grained control due to dataset limitations and intricate model architecture design. This paper introduces a novel training-free algorithm for fine-grained generation, called Aggregation of Multiple Diffusion Models (AMDM). The algorithm integrates features in the latent data space from multiple diffusion models within the same ecosystem into a specified model, thereby activating particular features and enabling fine-grained control. Experimental results demonstrate that AMDM significantly improves fine-grained control without training, validating its effectiveness. Additionally, it reveals that diffusion models initially focus on features such as position, attributes, and style, with later stages improving generation quality and consistency. AMDM offers a new perspective for tackling the challenges of fine-grained conditional generation in diffusion models. Specifically, it allows us to fully utilize existing or develop new conditional diffusion models that control specific aspects, and then aggregate them using the AMDM algorithm. This eliminates the need for constructing complex datasets, designing intricate model architectures, and incurring high training costs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a two-step process for combining multiple diffusion models. In the first step, the authors use Spherical Linear Interpolation (SLERP) to combine the models, and in the second step, they perform a descent procedure to move the combined output closer to the mean ( increasing probability of the sample ).

### Strengths
Propose novel method of Spherical Linear Interpolation for combining multiple diffusion models.

### Weaknesses
### Need Proofs for Claims / Additional Assumptions

**Lines 130–146:**  
The assumption \(p_{\theta_1}, p_{\theta_2} \subset \mathcal{M}_0 \) implies that \( p_{\theta_1} \) and \( p_{\theta_2} \) are trained on the same data, or equivalently, that they are generative models of the same underlying data distribution.  
However, this assumption holds **only if** both models are trained on data drawn from the same distribution. For example, if model A is trained on real images while model B is trained on paintings, the underlying data distributions differ, and hence the assumption \( p_{\theta_1}, p_{\theta_2} \subset \mathcal{M}_0 \) becomes invalid.

---

Can you provide a proof supporting the claim that additive or modified architectures only enhance control features?  
This assertion appears **baseless** and **lacks supporting theoretical or empirical justification**. Consequently, the argument for a “same diffusion-model ecosystem” becomes **ill-defined and conceptually inconsistent**.

---

**Line 164:**  
What do you mean by *“same task”*?

---

### Clarifications Needed

The motivation behind using *spherical interpolation* and *deviation optimization* is unclear.  
It is also not evident how the parameters \( q_1, w_2, n_\theta \) are chosen.

There exist several approaches to aggregate diffusion models, many of which are derived from a probabilistic perspective under certain assumptions of independence (broadly referred to as *compositionality*).

### Comparison to Relevant Works

1. **Reduce, Reuse, Recycle:** [arXiv:2302.11552](https://arxiv.org/pdf/2302.11552)  
2. **Compositionally:** [arXiv:2206.01714](https://arxiv.org/pdf/2206.01714)  
3. **Conditional Independence Assumed:** [arXiv:2503.01145](https://arxiv.org/abs/2503.01145) — enforcing conditional independence  
4. **Without Conditional Independence (Controllability):** [arXiv:2302.14368](https://arxiv.org/pdf/2302.14368)  
5. [arXiv:2505.13213](https://arxiv.org/pdf/2505.13213)  
6. **Algebraic Perspective:** [arXiv:2502.04549](https://arxiv.org/abs/2502.04549)

All these works rely on certain **underlying assumptions**, particularly regarding independence.  
When such independence does not hold, an additional **weighting or hyperparameter term** is typically introduced to account for the violation.

---

### Example: Independence vs. Dependence

If conditional independence is assumed:

\[
p_{\theta}(x \mid y_1, y_2)
  = p_{\theta}(x \mid y_1)
  + p_{\theta}(x \mid y_2)
  - p_{\theta}(x \mid \varnothing)
\]

If independence **is not** assumed:

\[
p_{\theta}(x \mid y_1, y_2)
  = p_{\theta}(x \mid y_1)
  + w_1\, p_{\theta}(x \mid y_2)
  - w_1\, p_{\theta}(x \mid \varnothing),
\]

where \( w_1 \) controls the degree of independence between \( y_1 \) and \( y_2 \).

---

### Connecting to the Current Work

In line with the assumptions of the current work, suppose \( z_t \) is drawn from the distribution \( p(z_t) = p(z_t \mid \varnothing) \), and assume that \( z_t \subset \mathcal{M}_t \) for all \( t \in [0, T] \).  
The objective can then be viewed as **linearly combining distributions** to maximize the probability of lying on the data manifold:

\[
\max_{a,b,c}\; p(a z^{(1)}_{t-1} + b z^{(2)}_{t-1} + c \mid \varnothing, z_t),
\]

such that the combination lies on the manifold of \( p(z_t) \).

Since the distribution is Gaussian, the maximum corresponds to its **mean**, reducing the sampling process to:

\[
a\, p_{\theta}(x \mid y_1, y_2)
  = p_{\theta}(x \mid y_1)
  + b\, p_{\theta}(x \mid y_2)
  + (1 - a - b)\, p_{\theta}(x \mid \varnothing).
\]

---

### Interpretation

Methods such as **ADAM** or **spherical interpolation** represent alternative interpolation strategies—implicitly introducing weighting factors rather than explicitly modeling independence.  
In the current work, this combination is obtained via **spherical interpolation**, without direct access to \( p(z_t \mid \varnothing) \) (as in the classifier-free guidance formulation).  
However, it remains unclear **why any \( s < T \)** should necessarily lie on a sphere.

---

### Broader Concern

For methods that do not assume any structural property of the data distribution, the introduction of a **hyperparameter** becomes unavoidable.  
It is also unclear **how these hyperparameters should be selected**.

---

### Results and Evaluation

The reported results appear potentially misleading.  
As the authors themselves claim, combining any two methods tends to improve performance over both individual methods.  
Since the hyperparameters are tuned by the authors, the worst possible case corresponds to \( w_2 = 0 \).  
Therefore, unless a **principled approach** for hyperparameter selection or a **comparison with existing aggregation methods** is provided, the results section offers **limited insight**.

### Questions
Addressing all the weaknesses will answer all my questions

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the problem of fine-grained conditional control in diffusion models. The proposed method, AMDM, aggregates multiple diffusion model scores via spherical interpolation, and employs a deviation optimization step to stabilize the aggregated score function. The framework enables users to combine the strengths of multiple fine-tuned diffusion models without requiring complex multi-capability datasets or multi-stage finetuning. Experiments on several conditional generation tasks demonstrate that AMDM can effectively integrate diverse capabilities from different models at inference time.

### Strengths
- The method is grounded in solid theoretical analysis, particularly regarding the confidence and reliability of aggregated diffusion scores.
- The empirical evaluation is thorough, and the comparisons clearly demonstrate how AMDM improves over baselines in multiple tasks.
- The motivation is practical and relevant: enabling reuse and integration of existing specialized diffusion models without retraining.

### Weaknesses
- Although the theoretical analysis is detailed, the paper may benefit from a simple controlled or toy example to help intuitively illustrate the effect of score aggregation and deviation optimization.
- The evaluation primarily focuses on three types of models (MIGC, InteractDiffusion, and IP-Adapter). While the results are encouraging, a broader range of conditional diffusion methods or application settings would better support the generality of the method.
- One of the key claims — that diffusion models initially prioritize feature generation before later refining image quality and consistency — is mainly supported by a single ablation (Table 3). Additional analysis or diagnostic visualization would help strengthen this conclusion.

### Questions
1. Regarding spherical interpolation and Equation (6):

    Could the authors clarify the geometric intuition or assumption behind applying spherical aggregation to score vectors? Is the assumption that these score fields locally reside on a shared spherical manifold, or is the spherical constraint introduced primarily for normalization and stability?
    
2. Computation overhead:

    What is the computational cost of AMDM compared to running a single diffusion model?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the Aggregation of Multiple Diffusion Models (AMDM) algorithm, a training-free method for improving fine-grained conditional control in image generation. AMDM works by aggregating features from multiple diffusion models within the same ecosystem to integrate their respective strengths. The key components are 1) Spherical aggregation 2)Deviation optimization. The authors demonstrate that AMDM significantly improves fine-grained control capabilities by integrating different models' strengths.

### Strengths
- The authors identify a genuine limitation in current models that they excel in specific aspects but struggle with others, and provide a solution without requiring retraining.
- The approach is backed by mathematical analysis of why aggregation works for models in the same diffusion ecosystem.
- Extensive experiments demonstrate clear improvements in both qualitative results and quantitative metrics.
- Unlike many compositional methods that introduce significant computational overhead, AMDM has minimal additional cost.

### Weaknesses
- While the authors provide theoretical justification, the assumptions (functional proximity, conditional proximity) are somewhat heuristic and don't offer global guarantees.
- AMDM only works for models within the same diffusion ecosystem, limiting its general applicability.
- While some comparisons with compositional methods are provided, a more comprehensive comparison with other training-free approaches would strengthen the paper.
- When aggregating models, there might be unintended interactions between different aspects that aren't fully explored.
- The optimization step size is selected empirically, and the authors acknowledge this as a limitation requiring future work.

### Questions
- How sensitive is AMDM to the choice of models within an ecosystem? Are there certain model combinations that work particularly well or poorly?
- (minor) What happens to the result if the positional information is not given? Would it still struggle to generate as specified from the text prompt or could this fine-grained control be coming from additional information, poisition.
- (minor) This is more like a question and just curiosity on my side. Could this fine-grained problem only exist within the open-source models? In other words, would the problem still exist in the models such as Sora?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles the problem of fine-grained control in diffusion-based generative models that fail at nuanced, multi-conditional control ,e.g., spatial arrangement and style preservation. The authors propose a training-free algorithm called AMDM that aggregates latent representations from multiple diffusion models.  During inference, AMDM merge latent variables geometrically using spherical aggregation and a deviation optimization step. The authors conducted several empirical experiments to show improvements in attribute, style, and interaction controllability with AMDM.

### Strengths
- The perspective of leveraging existing diffusion models to address fine-grained, controllable generation in a training-free manner is an interesting direction.
- The paper is well-written and easy to follow; empirical performance and results demonstrate the promise of the proposed method.

### Weaknesses
- "Diffusion ecosystem" is mentioned across the paper as a prerequisite of AMDM but somewhat loosely defined, for example, how to quantitatively verify whether any two models belong to the same ecosystem is unclear. 
- Evaluation scope is limited to certain derivatives of Stable Diffusion models. Throughout the experiments, the authors only picked a few classic SD1.4/1.5 and SDXL models, but did not examine whether the findings generalize to more recent model architectures based on DiT instead of U-Net, such as as SD3 or Flux. 
- Comparing to other model composition methods in the literature, the proposed method only applies to high-dimensional Gaussian and fails to general distributions (as shown in Table 7)

### Questions
- Besides the comparisions between SD1.5 and SDXL, how sensitive is AMDM to broader model heterogeneity (schedulers or attention designs)? For example, you may conduct some experiments on models based on DiT to show the generality of your method?
- The weighting factor $w$ in Slerp is treated as hyperparameter, but lacks a principled selection method. Empirical experiments are conducted based on two/three model aggregation where $w$ is determined with ablations. Can you discuss more on how to systematically select these parameters, and especially with a set of models (beyond three typical models you selected), how your propose method work empirically?

### Soundness
3

### Presentation
3

### Contribution
2
