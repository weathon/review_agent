# Understanding the Mixture-of-Experts with Nadaraya-Watson Kernel

- Decision: Accept (Poster)
- Scores: 6, 2, 2, 6

## Abstract
Mixture-of-Experts (MoE) has become a cornerstone in recent state-of-the-art large language models (LLMs). Traditionally, MoE relies on $\mathrm{Softmax}$ as the router score function to aggregate expert output, a designed choice that has persisted from the earliest MoE models to modern LLMs, and is now widely regarded as standard practice. However, the necessity of using $\mathrm{Softmax}$ to project router weights into a probability simplex remains an unchallenged assumption rather than a principled design choice. In this work, we first revisit the classical Nadaraya–Watson regression and observe that MoE shares the same mathematical formulation as Nadaraya–Watson regression. Furthermore, we show that both feed-forward neural network (FFN) and Mixture-of-Experts (MoE) can be interpreted as a special case of Nadaraya–Watson regression, where the kernel function corresponds to the input neurons of the output layer. Motivated by these insights, we propose the **zero-additional-cost** Kernel Inspired Router with Normalization ($\mathrm{KERN}$), an FFN-style router function, as an alternative to $\mathrm{Softmax}$. We demonstrate that this router generalizes both $\mathrm{Sigmoid}$- and $\mathrm{Softmax}$-based routers. **Based on empirical observations and established practices in FFN implementation, we recommend the use of $\mathrm{ReLU}$ activation and $\ell_2$-normalization in $\mathrm{KERN}$ router function.** Comprehensive experiments in MoE and LLM validate the effectiveness of the proposed FFN-style router function $\mathrm{KERN}$.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper revisits the design of the Mixture-of-Experts (MoE) routing mechanism through the lens of Nadaraya–Watson regression, a classical kernel-based estimator. The authors identify a structural similarity between MoE routers, feedforward networks (FFNs), and kernel smoothers, and use this observation to reinterpret the MoE routing process as a form of parametric kernel regression. Building on this perspective, the authors propose a new router function called KERN (Kernel Inspired Router with Normalization), which replaces the traditional Softmax gating with a lightweight $l_2$-normalization and ReLU activation. KERN eliminates the need for projecting gating scores onto a probability simplex, simplifying training and improving numerical stability. Empirical evaluations show that KERN achieves balanced expert utilization and improved stability in large-scale MoE and LLM setups, without adding computational or parameter overhead. The approach generalizes both Softmax- and Sigmoid-based routers while maintaining the inductive biases of standard FFN layers.

### Strengths
1. The paper offers a principled reinterpretation of Mixture-of-Experts (MoE) routing through the lens of Nadaraya–Watson regression, providing theoretical clarity and unifying MoE with FFN and kernel regression models.
2. The paper is well-organized, progressing clearly from Nadaraya–Watson regression to FFN interpretation, MoE formulation, and the final KERN methodology.
3. The authors conduct sufficient experiments and ablation studies on multiple MoE setups, validating both the theoretical claims and practical benefits of KERN.

### Weaknesses
1. The paper appears to change the order of operations between normalization and activation layers. Specifically, Equation (4) applies the activation function before normalization, which is inconsistent with Equation (3), where normalization precedes activation. This discrepancy may affect the theoretical consistency of the proposed analogy between FFN and Nadaraya–Watson regression.
2. The main contribution focuses on designing an alternative router function for MoE. While the proposed KERN router is elegant, its generality appears limited compared to broader architectural components like the Softmax function, which is widely used beyond MoE.
3. The paper would benefit from a figure that clearly presents the computational flow of the proposed router function. Although the mathematical formulation is clear, a visual depiction would enhance understanding and accessibility, especially for readers less familiar with MoE internals.

### Questions
1. How does changing the order between normalization and activation (as seen in Eq. (4) vs. Eq. (3)) influence the theoretical interpretation or empirical behavior of the model? Has this difference been empirically tested or justified in the paper?
2. Can the proposed KERN router be extended beyond MoE architectures to serve as a general routing or attention mechanism in neural networks, similar to how Softmax is used broadly across architectures?

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
5

### Summary
This paper reinterprets MoE routing through the lens of the Nadaraya–Watson (NW) estimator, arguing that both FFNs and MoE can be seen as parametric NW regressors. Building on this view, the authors propose KERN (Kernel-Inspired Router with Normalization): a “FFN-style” router that replaces Softmax with a linear projection followed by L2-normalization, ReLU, and a learnable global scale, combined with standard Top-k selection. The motivation is to avoid exponential activations’ saturation, keep output scale stable, and improve expert utilization. Experiments from 520M→6.9B (active 125M→1.3B) on Books3/ArXiv and a 50B-token FineWeb-Edu pretrain report consistent loss gains and better zero-shot downstream accuracy versus Softmax/Sigmoid/Tanh routers and dense baselines.

### Strengths
- **Clear, unified view.** The NW formulation offers a simple mathematical template that maps FFN and MoE into the same “kernel+normalization” structure, making the proposed design choices easy to state and implement.
- **Low engineering overhead.** KERN is a drop-in change (ReLU+L2norm+γ) and the paper claims no substantive time cost relative to Softmax routing.
- **Consistent wins in the reported setup.** Across multiple model sizes and datasets, KERN slightly but repeatedly outperforms Softmax/Sigmoid/Tanh, including at longer contexts.

### Weaknesses
- **Limited novelty.** The proposed router is essentially a straightforward substitution of the probabilistic Softmax with a normalized ReLU gate, i.e., $\gamma \cdot ReLU(norm(xW)))$, in replace of $Softmax(xW)$. While the Nadaraya–Watson (NW) perspective is conceptually interesting, the paper mainly **instantiates** an NW-like form without analyzing whether NW’s assumptions (kernel choice, normalization, bandwidth/scale selection) align with MoE routing objectives (load balance, capacity control, calibration). The stated motivation—“Softmax/Sigmoid suffer gradient saturation/vanishing”—is asserted rather than demonstrated (e.g., no gradient-magnitude, activation-scale, or saturation diagnostics).
- **Insufficient comparison to closely related work.** Most components of the final design already exist in the literature, yet the paper primarily compares against vanilla Softmax/Sigmoid/Tanh routers, omitting stronger, conceptually adjacent baselines and ablations.
	- **ReLU routing:** \[1\] replaces Softmax with ReLU to obtain fully differentiable, dynamic routing.
    - **Logit normalization:** \[2\] normalizes routing logits to decouple expert-centroid magnitude from assignment.
    - **Output scaling:** Industrial implements like \[3,4\] adjusts the routing output scale to preserve training dynamics comparable to dense models.  

- **Questionable empirical setup and reporting.** The reported gaps show Sigmoid performing markedly worse than Softmax—sometimes **larger** than the gap between Softmax-MoE and its dense counterpart—which is atypical relative to prior results \[2,5\] and suggests possible tuning or configuration mismatches. Critical details for effective MoE training are missing: the load-balancing objective (form and coefficient), capacity factor/dropless vs padded routing, post-Top-(k) renormalization (if any), and per-expert utilization/entropy or drop-rate statistics. Without these, it is difficult to assess stability, fairness of comparisons, or reproducibility of the claimed gains. 

\[1\] ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing https://arxiv.org/pdf/2412.14711
\[2\] On the Representation Collapse of Sparse Mixture of Experts https://arxiv.org/pdf/2204.09179
\[3\] DeepSeek-V3 Technical Report https://arxiv.org/abs/2412.19437
\[4\] Kimi K2: Open Agentic Intelligence https://arxiv.org/abs/2507.20534
\[5\] Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts https://arxiv.org/pdf/2408.15664

### Questions
- How does the scale parameter $\gamma$ evolve during training (per layer and across settings)?
- Do gradient vanishing and saturation actually occur with Softmax and Sigmoid routers, and does KERN empirically mitigate them?
- Does KERN improve expert load balancing?
- How does KERN behave in MoE architectures with shared experts?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper revisits the routing mechanism of Mixture-of-Experts (MoE) from a statistical perspective. The authors draw an analogy between the MoE router and the Nadaraya–Watson (NW) kernel regression, showing that both can be viewed as weighted aggregations over functions. Motivated by this connection, the paper introduces KERN (Kernel-Inspired Router with Normalization) — an FFN-style router that replaces Softmax with ReLU activation and ℓ₂ normalization.
Extensive experiments across model sizes (125M–6.9B parameters), context lengths, and datasets (Arxiv, Books3, FineWeb-Edu, etc.) demonstrate consistent gains of KERN over Softmax, Sigmoid, and Tanh routers. The method adds no parameters or computational overhead, making it a practically attractive alternative for large-scale MoE systems.

### Strengths
This paper revisits the routing mechanism of Mixture-of-Experts (MoE) from a statistical perspective. The authors draw an analogy between the MoE router and the Nadaraya–Watson (NW) kernel regression, showing that both can be viewed as weighted aggregations over functions. Motivated by this connection, the paper introduces KERN (Kernel-Inspired Router with Normalization) — an FFN-style router that replaces Softmax with ReLU activation and ℓ₂ normalization.
Extensive experiments across model sizes (125M–6.9B parameters), context lengths, and datasets (Arxiv, Books3, FineWeb-Edu, etc.) demonstrate consistent gains of KERN over Softmax, Sigmoid, and Tanh routers. The method adds no parameters or computational overhead, making it a practically attractive alternative for large-scale MoE systems.

### Weaknesses
The central theoretical claim---that Mixture-of-Experts (MoE) routing is *equivalent* to Nadaraya–Watson (NW) kernel regression---is conceptually appealing but lacks mathematical rigor. 

The presented argument only shows that the softmax weighting can *approximate* a kernel-weighted aggregation under certain parameterizations, not that the two models are formally or probabilistically equivalent. 

In particular, NW regression requires a symmetric and positive-definite kernel function with explicit density normalization, while the softmax router relies on asymmetric learned logits and gradient-based optimization dynamics. 

Hence, the connection is heuristic rather than theoretically grounded. 

The paper would benefit from a clear statement of the conditions (e.g., scaling limits, normalization assumptions) under which this analogy approximately holds. 

Overall, the claimed equivalence is oversold, weakening the theoretical soundness of the contribution.

### Questions
1. The paper claims a formal equivalence between softmax routing and Nadaraya–Watson kernel regression. Could the authors clarify whether this relationship is a *functional approximation* (similar weighting structure) or a *true probabilistic equivalence* under specific assumptions?  

2. Under what mathematical conditions---such as temperature scaling, large-dimension limits, or specific kernel choices---can softmax routing converge to the NW estimator?  

3. Given that the softmax router’s logits are learned through gradient descent rather than fixed distances, how does this affect the validity of interpreting it as a kernel regression model?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper connects MoE routers to Nadaraya-Watson regression. The authors then propose KERN, a zero-cost FFN-style router using ReLU activation and ℓ2-normalization. Experiments on language modelling and NLP tasks demonstrate KERN's effectiveness.

### Strengths
- The authors motivates various design decisions thoroughly.
-  This work connects kernel method perspective to MoE router design.

### Weaknesses
- Improvement is usually marginal. 
- It's a pity that the connection to Nadaraya-Watson estimator isn't well explored. It's not obvious why the router design is inspired by this perspective, they don't have much shared element, and somehow we don't see any attempt at adopting the traditional formula as a router. I think the paper will be a lot more interesting if it shows variants like Gaussian kernel.

### Questions
- In "Advantages of the KERN router function", why does KERN help with gradient vanishing? Sigmoid and Softmax may still have some gradient even if the input is at small value, but RELU function in KERN directly cutoff the gradient.

### Soundness
3

### Presentation
3

### Contribution
2
