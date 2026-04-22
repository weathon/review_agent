# JEPA-Reasoner: Generative Latent Space Reasoner

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
While Joint-Embedding Predictive Architecture (JEPA) has emerged as a powerful architecture for learning rich latent representations, it fundamentally lacks generative abilities. Meanwhile, latent space reasoning attempts for Transformer models like COCONUT do improve performance, but they ultimately rely on token-by-token generation, which still accumulates compounding error and relies on context information to gain reasoning insights. To address these limitations, we propose JEPA-Reasoner, a novel JEPA model enhanced with generative ability that reasons in latent space. We augment it with a separate action-taker model, Talker, to produce human-readable sentences. Our approach demonstrates that decoupling latent space reasoning and token generation enables JEPA-Reasoner to produce mixed latent vectors that might lay the foundation for multi-threaded reasoning, while performing autoregressive generation with superior robustness to compounding error.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces JEPA-Reasoner, a novel architecture designed to equip Joint-Embedding Predictive Architectures (JEPAs) with generative capabilities while mitigating the compounding error problem inherent in standard token-by-token autoregressive models. The framework consists of two main components: JEPA-Reasoner and Talker. One is for the latent reasoning, and the other translates these latent representations into human-readable outputs. The paper validates this approach on two synthetic tasks: a tree-search task to probe the model's ability to represent uncertainty via "mixed latent vectors" and a Context-Free Grammar (CFG) generation task to measure robustness against noise and error propagation.

### Strengths
* The core idea of decoupling latent-space reasoning from token-space generation is intuitive and well-motivated.
* The tree-search experiment (Sec 5.1) and the corresponding PCA visualization (Appendix E, Fig 8) provide strong evidence for the "mixed latent vector" hypothesis. The results show the model generates continuous latent representations that lie between discrete vocabulary vectors, effectively representing uncertainty without collapsing to a single choice.
* The probabilistic factorization $P(R,X) = P(R) \cdot P(X|R)$ explains how error propagation can be contained.

### Weaknesses
* Although Table 4 attempts to show robustness advantages of the decoupled architecture under Gaussian noise perturbations, the reported gains are numerically small and inconsistent across scales. For instance, the large JEPA-Reasoner (R_large) achieves only ~0.46 vs 0.37 for C_large—an improvement of about 0.09 absolute accuracy. Meanwhile, the middle- and small-scale variants even underperform the COCONUT models.
* The paper's evaluation is confined entirely to two synthetic tasks: tree-search and CFG generation. There is no comparison to existing standard reasoning datasets (e.g., GSM8K). This limits the paper’s generality.
* The paper lacks an ablation analysis. There is no systematic examination of the proposed components. In particular, the authors do not isolate the contributions of key design choices such as: the hybrid normalization, the EMA target encoder strategy, or the scaled cosine distance loss with parameter k.

### Questions
* Can you provide a more rigorous explanation for the poor performance of the $R_{middle}$ and $R_{small}$ models beyond the "parameter efficiency" argument in Appendix B.4? 
* Can you provide more evaluations on standard reasoning benchmarks?
* The experiments seem to be minimal, as the only baseline is a single COCONUT model. How does JEPA-Reasoner's performance compare against other existing latent reasoning models or other lines of CoT models?

### Soundness
2

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
The paper extends the JEPA into a generative model capable of performing autoregressive reasoning in latent space. And they propose decoupling the reasoning and generation processes: a reasoner predicts future latent representations, while a separate Talker reconstructs readable tokens from those latents. Empirical experiments on synthetic tree-search and context-free-grammar tasks show that JEPA reasoner achieves strong robustness to noise.

### Strengths
1. Its decoupled reasoner and talker design effectively separates reasoning from surface generation, helping to reduce compounding errors and improving robustness and experiments on synthetic reasoning tasks demonstrate strong stability and noise tolerance, empirically validating the model’s ability to maintain consistent latent dynamics. 
2. The model shows potential for multi-hypothesis representation, with mixed latent vectors indicating that the system can encode several possible reasoning paths simultaneously which is a promising step toward more structured reasoning behaviors.

### Weaknesses
1. The model’s training objective focuses on latent similarity (cosine distance) rather than logical correctness, meaning it optimizes for smooth representation transitions rather than true reasoning accuracy. 
2. All experiments are conducted on synthetic tasks (tree-search and CFG generation), leaving its effectiveness on real-world reasoning or language tasks unverified.

### Questions
I understand that the use of an EMA based target encoder is standard in traditional JEPA models to provide stable self-supervised learning. However, in the context of reasoning, this choice seems less intuitive. Could you clarify why the EMA objective is still appropriate for reasoning tasks, given that reasoning typically requires directional or causal transitions rather than representational smoothing?

### Soundness
2

### Presentation
3

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
This work proposes extending the JEPA family of models to text generation by training to predict next latent representations without decoding them. They incorporating a "Talker" module trained independently, which handles the token generation, decoupling the latent reasoning space from the latent token generation space. They the JEPA-Reasoner in two stages and then evaluate against the typical transformer and Coconut in two synthetic settings.

### Strengths
* Separating the reasoning representation from the token generation is a super interesting idea.
* Novel training setup for JEPA for text.
* Promising performance on synthetic tasks.

### Weaknesses
* While the synthetic analysis is certainly interesting, only having two synthetic settings is a serious limitation of the work. 
* Further the relevance of the synthetic settings is questionable. It is unclear why Gaussian noise perturbations to the latent reasoning representations is an interesting setting. As far as I know Gaussian noise isn't a realistic noise model for language model latents representations. In the CFG complete task I assume token level errors are single token drops, replacements, or additions (its not made explicitly in the paper). While I agree this is a more reasonable noise model for natural language, it is still relatively unclear what the significance of doing better in this setting means practically for language models.
* The discussion of results is minimal, it would be helpful to explain in some depth the significance of your findings and why doing well in these settings has practical implications for language models.
* Related work section on latent reasoning is limited to Coconut. However there are a number of other works on reasoning in the latent space that are probably worth mentioning (e.g. Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Singh, S., Bartoldson, B. R., ... & Goldstein, T. (2025). Scaling up test-time compute with latent reasoning: A recurrent depth approach. arXiv preprint arXiv:2502.05171).
* The writing is hard to follow in a number of places. The work would benefit greatly from a revision for clarity.
* This seems like promising initial work on a novel method for language modeling, however, without more thorough benchmarking in real settings its challenging to understand the significance of the contribution.
* It seems there are a number of specifically architectural details required to make JEPA Reasoner work. Some analysis or ablation of these details would be an interesting contribution.

### Questions
None.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
the paper introduce JEPA-Reasoner. Extending JEPA to enable autoregressive generation by decoupling latent reasoning from token generation. I find the core idea of separating reasoning in latent space from token production interesting, and the theoretical framework around error propagation is well motivated.

### Strengths
I think the  proposed decoupled architecture is genuinely novel and the theoretical motivation for separating high-level reasoning from token generation makes sense. I particularly liked the analysis showing mixed latent vectors can represent uncertainty between choices, suggesting potential for multi-threaded reasoning. The ablation studies convincingly demonstrate that the Talker depends on the Reasoner's semantic content.

### Weaknesses
My main concern is the limited scope of evaluation on only synthetic tasks without any natural language benchmarks. I'm also unclear about computational costs compared to baselines, especially given the two-model architecture. The paper doesn't discuss how this approach would scale to real-world tasks or what the training complexity looks like in practice.

### Questions
How does training time and inference speed compare to COCONUT and standard transformers? Have you tried this on any standard NLP benchmarks beyond the synthetic tasks? What happens if you want to do few-shot learning or prompting with this architecture?

### Soundness
2

### Presentation
2

### Contribution
2
