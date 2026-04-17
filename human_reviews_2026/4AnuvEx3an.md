# $\alpha$-LoRA: Effective Fine-Tuning via Base Model Rescaling

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Fine-tuning has proven to be highly effective in adapting pre-trained models to perform better on new desired tasks with minimal data samples. Among the most widely used approaches are reparameterization methods, which update a target module by augmenting its frozen weight matrix with an additional trainable weight matrix. The most prominent example is Low Rank Adaption (LoRA), which gained significant attention in recent years. In this paper, we introduce a new class of reparametrization methods for transfer learning, designed to enhance the generalization ability of fine-tuned models. We establish the effectiveness of our approach in a high-dimensional binary classification setting using tools from Random Matrix Theory, and further validate our theoretical findings through more realistic experiments, such as fine-tuning large language models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes $\alpha$-LoRA, a reparameterization family that rescales each row of a frozen base weight matrix by a learned factor α when applying a trainable (optionally low-rank) update. The authors analyze this idea in a high-dimensional linear binary classification model using Random Matrix Theory (RMT) and derive a  closed-form optimal scaling $\alpha$ that maximizes asymptotic accuracy. They validate the theory on Amazon Reviews and demonstrate consistent improvements over vanilla LoRA on GLUE tasks (RoBERTa-base).

### Strengths
1. **Clear theoretical contribution with closed-form solution**.  The  paper gives a unique closed-form $ \alpha^*$ , giving guidance beyond “set $\alpha=1$” in fune-tuning setting.
2. **Practical instantiation for LLMs.** Extending $\alpha$ to a per-row vector aligns well with layered architectures; the formulation is simple to implement and novel in PEFT.

### Weaknesses
I am not an expert in theoretical analysis, so I am unable to rigorously verify the correctness of the authors’ proofs. The following comments are therefore made from a more practical and empirical perspective.

1. **Scope of theory.** The RMT analysis is confined to one-layer linear (ridge) classification under a Gaussian mixture with identity covariance. While the paper is transparent about this, the gap to two-layer (or multi-layer) nonlinear networks remains large, so the theory’s prescriptive power for LLMs is still indirect.
2. **Connection to low-rank structure is not formalized.** The core theory is agnostic to LoRA’s low-rank reparameterization. $\alpha$ -rescaling could be  applied to many PEFT variants. The paper positions $\alpha$ -scaling as complementary, but does not analyze interactions with rank choice or decomposition schemes, which would strengthen the story.
3. The idea of rescaling pre-trained channels is established in prior literature. Notably, IA³ fine-tunes LLMs by learning multiplicative factors on intermediate activations within linear transformations—an operation that is effectively equivalent to scaling the output channels of the associated weight matrices  𝑊. This close correspondence substantially diminishes the methodological novelty.
4. **Sample efficiency** The $\alpha$-update uses fresh mini-batches separate from those used to update the adapters. This design may trade sample efficiency for generalization. 
4. **Experimental concerns**:

- **Baselines.** Since the paper mentions complementary methods (e.g., DoRA), including such baselines would better position $\alpha$-LoRA beyond vanilla LoRA
- **Hyperparameters.** The appendix shows different adapter LRs and LoRA vs. $ \alpha$-LoRA on several tasks, which can favor $\alpha$-LoRA. The authors even uses different $T$ or ``val split`` for different seed on RTE, SST2 and QQP. While I appreciate the authors’ transparency, the need for such delicate hyperparameter tuning raises concerns about the method’s robustness and practical effectiveness.
- **Limited scale of empirical evaluation**. The experiments are conducted on relatively small datasets and modest-sized models. The experiments would be more convincing if the authors could include evaluations on larger-scale benchmarks (e.g., AlpacaEval) and more capable models such as LLaMA 2-7B/13B.

### Questions
See weakness.

### Soundness
3

### Presentation
2

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
This paper proposes $\alpha$-LoRA, a PEFT variant that rescales the frozen base weights row-wise by a vector $\alpha$ before adding a (low-rank) adapter, i.e., for a layer with frozen $W^\star \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ the update is
$$
W_{\text{new}} = \alpha \odot W^\star + W,\quad W =  AB \text{ (LoRA) },
$$
where $\alpha\in\mathbb{R}^{d_{\text{out}}}$ and $\odot$ is row-wise scaling. Most of the paper analyzes a **high-dimensional linear binary classifier with squared loss** under a Gaussian mixture: pre-training data are drawn from $\tfrac12\mathcal{N}(\mu,I_p)+\tfrac12\mathcal{N}(-\mu,I_p)$, and **fine-tuning** targets a shifted mixture with class means $\pm\mu_\beta$, where $\mu_\beta=\beta\mu+\mu_\perp$ and $\mu_\perp\perp\mu$ (identity covariance in both phases). Denoting the ridge solution on target data by $w=\tfrac1nQXy$ and the pre-trained weights by $\tilde w$, the **$\alpha$-fine-tuned** classifier has the closed form
$$
w_\alpha = w + \alpha \gamma Q\tilde{w},
$$
and the authors derive an **asymptotically optimal scalar $\alpha^\star\neq 1$** (closed-form via deterministic equivalents) that maximizes target accuracy in this GMM setting. They then **extend from a scalar to a per-row vector $\alpha$ for neural nets** and give a practical heuristic: treat each $\alpha$ as trainable and update it periodically (every $T$ steps) with a separate optimizer/batch while training LoRA adapters. Empirically, they report that choosing/learning $\alpha$ improves transfer on Amazon Reviews category shifts (where the theory’s $\alpha^\star$ is competitive) and yields consistent gains over vanilla LoRA on GLUE with **roberta-base**. Overall, the work argues that **decoupling magnitude via base-weight rescaling** is a simple, general lever that improves generalization in fine-tuning.

Generally, the paper has a good theoretical contribution, but the experimental evaluation is limited, and there is a gap between the theoretical setting and the LLM fine-tuning setting, as well as a gap between the theoretically analyzed method and the method used in practice.

### Strengths
- **(S1)** In the linear GMM setting, the analysis is careful and culminates in a closed-form $\alpha^*$ that depends on data-dependent scalars (via RMT). Figures show how the best α varies with alignment β and dimension, matching intuition and theoretical findings. 
- **(S2)** Row-wise rescaling of the frozen weights is architecture-agnostic and easy to add to existing LoRA pipelines; parameter overhead is negligible. 
- **(S3)** The method demonstrates consistent empirical gains:
  - On Amazon Reviews transfers, $\alpha$-LoRA (or $\alpha^*$ from theory) beats both no-FT and standard FT ($\alpha = 1$) across all source-to-target pairs reported. 
  - On GLUE, $\alpha$-LoRA improves over LoRA on all six tasks.
- **(S4)**The paper is well presented and clearly conveys complicated theoretical analysis *in the main text*, which is notable.

### Weaknesses
- **(W1)** The theoretical model used to analyze the proposed method makes several *strong* assumptions, including **(i)** a *linear* binary classifier trained with **squared-loss ridge regression**, **(ii)** data drawn from **spherical Gaussian mixtures** with identity covariance, **(iii)** a highly constrained source-to-target shift of the form $\mu_\beta=\beta \mu+\mu_\perp$ with a single alignment parameter and an orthogonal residual, and **(iv)** reliance on **high-dimensional asymptotics** $(p,n, N\to\infty)$ with fixed ratios. While I understand these choices enable clean RMT analysis, they limit the external validity and impact of the theoretical results for modern, non-linear, multi-task/generative fine-tuning scenarios. 
- **(W2)** Even under these assumptions, at a high level, the paper introduces a reparametrization scheme for finetuning binary linear classifiers $w_\alpha = \alpha w + \Delta w$ and then finds the $\alpha$ leading to optimal generalization. In other words, within the suggested framework, the paper can derive an optimal $\alpha$ for generalization. However, there is a lack of motivation for why this reparametrization scheme is a good idea in the first place. E.g, why not have $w_A = A w + \Delta w$ for $A \in \mathbb{R}^{d \times d}$? Why not use other reparametrization schemes?
- **(W3)** The paper takes a **very large** leap from fine-tuning binary linear classifiers to fine-tuning **language models**. The method is modified heavily: **(i)** row-wise rescaling of the base weights instead of a single scalar, **(ii)** parameter delta is parameterized by a low-rank adapter instead of a full weight matrix, and **(iii)** the optimal $\alpha^*$, is no longer closed form (which is the main contribution of the paper) and is optimized with the parameters (unclear how this effects generalization). While the empirical results are positive, I don't believe that this can be considered the same method, and in terms of a pure empirical contribution, a comparison to the broad literature on LM fine-tuning and PEFT more broadly is lacking. I believe a middle ground, considering a practical setup that doesn't deviate as much from the original proposed method, perhaps on simpler tasks, would be informative.

### Questions
- **(Q1)** Is the assupmtion that $\frac{p}{n}, \frac{p}{N} \to \mathrm{const}$ justifiable/common in the literature? It doesn't seem intuitive to me that data dimensionality $p$ grows with the number of samples.
- **(Q2)** How sensitive are results to $T$ (update period) and to learning rate for $\alpha$? Any stability issues if $\alpha$ is updated every step? 
- **(Q3)** Do you share $\alpha$ across Q/K/V/O projections or learn separate vectors? You mention sharing reduces overhead. How does it affect accuracy? 
- **(Q4)** Any results on instruction-tuning or reasoning datasets where alignment shifts are larger (the regime where your theory suggests $\alpha$ matters most)?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents an interesting insight: the frozen weights in LoRA can be rescaled, introducing an additional degree of freedom that yields better performance. The authors provide theoretical analyses supported by experiments on real-world tasks.

### Strengths
1. This paper raises an interesting point: investigating the scale factor of pretrained weights in LoRA fine-tuning.
2. Several theoretical analyses are provided to support this argument.
3. The paper is quite well-written.

### Weaknesses
1. The majority of the theoretical analysis focuses on binary classification problems, which differs from the actual training of large language models (LLMs).
2. The proposed idea is somewhat narrow, specifically concerning the scaling factor of pretrained weights.
3. The experiments are primarily conducted on GLUE; large-scale experiments on large LLMs are therefore needed.

### Questions
I find that the optimal alpha is larger than 1. Could the authors provide some analysis on this?

The optimal alpha is coupled with the low-rank adaptation matrix BA; that is, increasing the magnitude of BA also affects the optimal alpha. Additional ablation studies are therefore needed to explore this relationship.

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
This paper proposes α-LoRA, a novel parameter-efficient fine-tuning method that introduces a scaling parameter α applied row-wise to frozen base model weights before adding trainable adapters. The authors provide theoretical analysis using Random Matrix Theory (RMT) for high-dimensional binary classification under a Gaussian Mixture Model, proving the existence of an optimal α* ≠ 1 and deriving its closed-form expression. Experiments on Amazon Review (linear classification) and GLUE benchmarks (RoBERTa fine-tuning) demonstrate consistent improvements over standard LoRA, with the optimal α learned via a separate optimization procedure using held-out batches.

### Strengths
- The RMT analysis is rigorous and provides closed-form expressions for optimal α* in the theoretical setting
- The deterministic equivalent framework is well-established and appropriately applied
- Proof structure is systematic
- The connection between α and task alignment β is intuitive and theoretically justified
- Novel theoretical insight: The optimal α* ≠ 1 result challenges the implicit assumption in LoRA that base weights should be preserved at their original scale
- Practical algorithm: Algorithm 1 provides a trainable approach when closed-form α* is unavailable
- Consistent improvements: Table 1 and Table 2 validates theory on linear models
- Generalizable framework: Extension to multi-source fine-tuning and arbitrary source classifiers shows broader applicability

### Weaknesses
- Limited novelty over prior work: The idea of scaling frozen weights is simple; the main contribution is showing α* ≠ 1 theoretically. However: DoRA already rescales weights (magnitude vs direction decomposition). The row-wise scaling in Eq. 10 is reminiscent of adapter biases. Learning α via separate optimization is similar to meta-learning approaches

- Modest empirical gains: Table 2: Most improvements are <2%. No significance tests or confidence intervals provided
Three seeds is minimal for statistical validity, Some tasks show negligible improvement (QQP: 0.06%)

- Incomplete experimental validation: No comparison with DoRA, AdaLoRA variants, or recent LoRA improvements (LoRA+, etc.)
Only one base model tested (RoBERTa-base); no evaluation on larger LLMs despite claiming applicability. Missing low-data regime experiments despite motivation ("minimal data samples" in abstract). No experiments on catastrophic forgetting or out-of-domain robustness

- Why do query and value matrices get similar α values (mentioned in Section 5.2 overhead discussion) but not analyzed?
What determines when α* > 1 vs α* < 1? Figure 2 shows α* increases with β, but why does dimension p amplify this effect?

- Algorithm 1 requires sampling fresh batches and separate optimization, increasing training time. No wall-clock time comparisons provided. Memory overhead during training (storing two optimizers) not discussed

- Only classification tasks tested; no generation, QA, or reasoning tasks.

### Questions
- Can you provide wall-clock training time comparisons? How much overhead does Algorithm 1's separate α optimization introduce?
- How sensitive are results to T? Table 3 uses T=1, Table 4 uses T=20. Why? What's the ablation?
- Why different learning rates for LoRA vs α-LoRA in Table 4? (1e-4 vs 2e-4) Doesn't this confound the comparison?
- Can you test on larger models? The paper claims applicability to LLMs but only tests RoBERTa-base (125M). What about Llama-7B, Mistral-7B?
- How is β estimated for GLUE tasks? The Amazon Review experiments explicitly compute β, but this isn't mentioned for RoBERTa experiments.
- What happens when Assumption 4.1 is violated? Can you characterize robustness to violations of ‖μ‖ = O(1)?
- Why do query/value get similar α values? You mention sharing α across attention could reduce overhead, does this hurt performance?
- Can you provide significance tests? 
- How does α-LoRA compare to DoRA? DoRA also rescales weights. Is α-LoRA complementary or redundant?

### Soundness
3

### Presentation
3

### Contribution
3
