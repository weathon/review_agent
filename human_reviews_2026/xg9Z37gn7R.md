# Provable Low-Frequency Bias of In-Context Learning of Representations

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
In-context learning (ICL) enables large language models (LLMs) to acquire new behaviors from the input sequence alone without any parameter updates. Recent studies have shown that ICL can surpass the original meaning learned in the pretraining stage through internalizing the structure of the data-generating process (DGP) of the prompt into the hidden representations. However, the mechanisms by which LLMs achieve this ability are left open. In this paper, we present the first rigorous explanation of such phenomena by introducing a unified framework of double convergence, where hidden representations converge both over context and across layers. This double convergence process leads to an implicit bias towards smooth (low-frequency) representations, which we prove analytically and verify empirically. Our theory explains several open empirical observations, including why learned representations exhibit globally structured but locally distorted geometry, and why their total energy decays without vanishing. Moreover, our theory predicts that ICL has an intrinsic robustness towards high-frequency noise, which we empirically confirm. These results provide new insights into the underlying mechanisms of ICL, and a theoretical foundation to study it that hopefully extends to more general data distributions and settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper theoretically studies the effect of attention maps on data generated from a graph random walk. Under balancedness and stability assumptions, it is shown that token representations converge over both context and across layers. This shows that tokens become biased towards smooth representations, explaining an in-context learning phenomenon empirically observed in a previous paper.

### Strengths
The paper shows that in-context learning of token representations arises a consequence of an intrinsic bias towards low-frequency hidden representations in certain attention models on Markovian data. The theory also suggests an intrinsic robustness towards high-frequency noise, which is confirmed via toy experiments.

### Weaknesses
* "Convergence" of representations along the context direction (Theorem 1) is not really interesting. This is simply a consequence of assuming the convergence of weight fractions and representations (Definitions 1,3). Basically, the result is saying that if token $x$ attends to each token type $k\in f(x)$ with relative weight $\pi_k$, then the result of attention is the normalized sum of $\pi_k z_k$ (combined with straightforward error/concentration analysis, e.g., the relative empirical frequency converges to $\pi_k$).

* This analysis also relies on unrealistic assumptions to enable perturbation bounds. Why would the nonzero weights converge to the proportion of tokens (Def.3)? This assumption reduces attention to merely doing a counting operation. While induction heads are one example of such an operation as the authors point out, there is no point to studying its "convergence."

* Layerwise convergence also requires very unnatural assumptions. Although the paper claims their layerwise analysis can cover "FFNs, normalization layers, and other non-linearities" (Section 2.2), the "great mapping" assumption constrains $\sigma$ to be bounded above and below by linear functions, and moreover the only example given (Lemma 12) is a linear function. The only explanation given is "This allows us to insert FFNs or other neuron-wise transformations without disrupting convergence" which feels like this was added merely for the sake of enabling analysis. The spectral gap assumption given in Theorem 2 with no explanation is also very strange, see the last bullet point.

* The assumption that the attention map can be decomposed into A,B,O,T types is justified in Appendix G by checking the proportion of attention weights which fit each type. However, the condition that the weights for each token is a specific type, is much weaker than the condition that the entire attention map is decomposable into a linear combination of the 4 matrices $A^{(A)},A^{(B)},A^{(O)},A^{(T)}$ (which has only 3 degrees of freedom).

* The proof of Theorem 2 is wrong. When applying Corollary 11 in (62), the constant terms should be inversed. Otherwise the condition on line 326 can always be satisfied by choosing $\gamma_1=0$ and becomes vacuous. If this is fixed, the condition becomes that $\lambda_q/\lambda_{q+1}$ is bounded below by some large constant for all $q$, it is unclear why this would hold (moreover, this should not be called the spectral gap, which has a different meaning. Also, it should be made clear that this assumption should hold for all $q$).

### Questions
See Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a rigorous theoretical framework to explain the in-context learning of representations (ICLR) phenomenon in Transformers, where the characteristics of the arbitrarily designated token sequence–generating process outweigh the semantic content of the tokens in shaping their representational geometry. By introducing the concept of **Double Convergence**, the authors show analytically that representations first converge over the context to a semantics-independent limiting distribution and then, across layers, exhibit an implicit low-frequency concentration. Using a simplified single-head Transformer with structured attention, they prove that hidden states align with the low-frequency eigenvectors of a graph Laplacian defined by the data-generating process. Through analysis based on this theoretical framework and supporting empirical validations, the paper rigorously formalizes several heuristic observations from the original ICLR study and demonstrates that the ICLR process is robust to noise injection.

### Strengths
1. The paper’s theoretical foundation is both conceptually original and mathematically rigorous. The **double convergence**—one with respect to sequence length and another with respect to layer depth—neatly characterizes the representation learning process observed in the original ICLR paper and integrates it (at least up to some simplifications) into the actual forward computation of the model. The authors astutely identify the natural connection between the stationary distribution and transition kernel and the way the attention mechanism operates over embeddings across the context. The decomposition of attention operators into four basic types appears novel and insightful, aligning (to a certain degree) with empirical interpretability findings on the typical modus operandi of attention heads. This formulation also clearly facilitates the spectral analysis of how attention influences model representations.

2. The authors make remarkable efforts to connect their theoretical framework with the anecdotal and heuristic observations in the original ICLR paper. The framework demonstrates strong explanatory power by successfully characterizing where empirical findings align with—or deviate from—the theoretical results, based on well-understood principles from spectral graph theory. It also corrects several ad hoc claims from the ICLR paper through empirical experiments that closely match the a priori theoretical predictions, thereby reinforcing the validity and precision of the proposed theory.

### Weaknesses
1. While I acknowledge the authors’ justification in lines 170–178, the settings and assumptions underlying this paper still seem overly restrictive. To obtain context-wise convergence independent of token semantics, the authors assume a single-head attention layer and fixed attention weights—conditions that are far removed from the actual configurations of modern attention modules. Although they empirically validate the factorization of attention weights into four basic forms in Appendix G, which partly supports this assumption, it is highly likely that such simple structures occur only in the highly synthetic ICLR task and do not generalize to more realistic ICL settings involving richer semantics and the complex collective behavior of multiple attention heads (e.g., [1]). Beyond the fixed-weight assumption, the paper further introduces constraints such as balanced attention maps and specific token-mapping behaviors, which make certain theoretical results (particularly Theorem 1) relatively straightforward.

2. The authors restrict their analysis of convergence and low-frequency concentration in ICL hidden states exclusively to the highly constrained ICLR setting. This raises questions about whether their results can generalize to broader ICL scenarios involving representation analysis. For example, the convergence proof requires a context length exceeding ten times the vocabulary size—a reasonable condition for the ICLR task with its small, limited vocabulary but unrealistic for typical ICL tasks using large language model vocabularies on the order of \(10^{4}\) tokens. Moreover, the assumption of semantic-independent convergence diverges from empirical findings in more general ICL contexts, where representational constellations are demonstrably semantic-aware (see [2], [3]). 

[1] Singh, Aaditya K., et al. *“What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation.”*  
[2] Han, Seungwook, et al. *“Emergence and Effectiveness of Task Vectors in In-Context Learning: An Encoder–Decoder Perspective.”*  
[3] Kirsanov, Artem, et al. *“The Geometry of Prompting: Unveiling Distinct Mechanisms of Task Adaptation in Language Models.”*

### Questions
1, How do you obtain the limiting representations needed to compute the normalized energy scores in Fig 3?

2, In Definition 4, since $Z \in R^{d \times c}, D \in R^{c \times c}, U \in R^{d \times r}$, there seems to be a dimension mismatch in the matrix multiplication

3, In Deifnition 5 you arranged $\lambda_1,...,\lambda_c$ in non-decreasing order, but then in theorem 2 you arranged them instead in a non-increasing order. Consider fixing one of them to ensure consistency.

### Soundness
4

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
2

### Summary
This paper provides a theoretical analysis of in-context learning (ICL) focusing on the phenomenon of In-Context Learning of Representations (ICLR) recently observed by Park et al. (2024), where large language models (LLMs) internalize the data-generating process (DGP) of input sequences (e.g., random walks on graphs) within their hidden representations.

The authors introduce a new theoretical framework called double convergence, comprising two intertwined processes:
Context-wise convergence — within each layer, token representations converge to limiting representations determined by token identity.
Layer-wise convergence — across layers, these limiting representations further evolve to capture global structural properties of the DGP.
Together, these processes produce an implicit low-frequency bias in the hidden representations, leading to smooth, globally consistent embeddings that suppress high-frequency (noisy) components.

The paper formalizes these ideas through Theorem 1 (context-wise convergence) and Theorem 2 (layer-wise convergence). This explains several empirical findings in prior work—e.g., why representations align with global graph structure, why energy decays without vanishing, and why models show robustness to high-frequency noise.

Empirical validations (e.g., Figure 2–4) using pre-trained Transformers (Llama-3.1-8B) confirm that model embeddings align with theoretical predictions, exhibiting low-frequency structure and robustness to injected noise.

### Strengths
Originality: The paper provides the first rigorous theoretical framework linking ICL dynamics to low-frequency bias and graph-spectral smoothness, a novel perspective that unifies several previously disconnected empirical phenomena (representation alignment, energy decay, noise robustness). The “double convergence” view is both conceptually clean and mathematically tractable.

Quality and clarity:  The work offers formal proofs (Theorems 1–2, Lemmas 6–8) establishing convergence guarantees under balanced attention assumptions, with explicit probabilistic bounds and dependencies on spectral gap and Lipschitz constants. These results extend existing analyses of ICL (e.g., Lu et al. 2024; Li et al. 2025) to deep, nonlinear Transformers—beyond linear attention or single-layer cases. By showing that the limiting representations converge to the low-frequency eigenspace of the Laplacian, the paper provides a principled link between ICL dynamics and graph learning theory. This analogy is insightful and offers a unified way to interpret the “geometry” of learned representations. The alignment between theory and empirical trends strengthens credibility.

Significance: The results deepen theoretical understanding of why LLMs can generalize and remain robust without explicit optimization. The proposed low-frequency bias may become a fundamental explanatory principle for future ICL studies, bridging analysis of Transformers, graph spectral learning, and inductive biases in representation learning.

### Weaknesses
1. The analysis assumes that attention maps are “balanced” and externally fixed, depending only on token identity rather than learned representations. While this enables tractability, it significantly limits realism—modern Transformer attention depends on contextual interactions. The authors partially justify this with empirical evidence (“covers 70% of connections”), but the assumption still weakens the generality of the claims.

2. The main theorems are proved under a specific DGP—a random walk on a connected undirected graph with fixed stationary distribution. While extensions are mentioned in Appendix E, the central results remain tailored to this highly idealized setting. The relevance to natural language or general Markov data distributions remains unclear.

3. The empirical validation focuses on graph-based random walks and low-dimensional synthetic setups. No real-world data (e.g., text sequences or few-shot reasoning tasks) are evaluated. This limits the practical relevance of the theoretical insights.

4. The related work discussion is not comprehensive enough. For example, for the mechanism of ICL, it only includes theoretical work  and doesn't mention empirical progress on understanding ICL mechanism.

### Questions
How does this framework relate to recent work showing that Transformers perform implicit gradient descent (e.g., Ahn et al., 2023; Von Oswald et al., 2023)? Could low-frequency bias arise as a by-product of gradient-based meta-learning dynamics?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces the concept of Double Convergence as the underlying mechanism behind the  structured representations observed in in-context learning (ICL). Double convergence comprises two parts - context-wise and layer-wise convergence. The paper provides a theoretical framework for these two processes based on some assumptions on the attention maps. It extends the findings of Park et al. (2024) by providing a theoretical framework explaining the empirical observations report by Park et al like the structured hidden representation that capture the structure of the data generating process. It makes reasonable modification to the data generating process from Park et al. by fixing the first set of token in the input sequence. It also verifies the theoretical framework based on simplified transformer model for analysis, where it removed dynamic attention maps and residual connections. Overall, they show that, within the proposed framework, ICL representations converge in two axis resulting in low-frequency bias.

### Strengths
1. This studied topic of research is quite relevant for the community. The paper provides a sound theoretical framework to explain the ICLR phenomenon proposing the Double convergence process. Following the defined theoretical framework, it replicates empirical evidence from Park et al. 2024 using the simplified transformer model and further provides new insightful empirical analysis.

2. The detailed energy-decay analysis in hidden representations across layers is novel and support the low-frequency bias hypothesis. Robustness experiment in Section 5.5  adds more evidence for the hypothesis. They also justify the reason for the energy decay using the proposed double convergence framework.

3. The assumptions on simplified transformer model are well supported with empirical verification in Appendix G, showing a good overlap of 72% attention weights.

4. Paper is well-structured and easy to follow. Overall, the theoretical framework is well defined, and empirically validated.

### Weaknesses
1. The empirical work shown in this paper is limited to a single DGP process, which is same as Park et al. 2024. Adding experiments on additional DGP process would strengthen evidence for the proposed framework.

2. It would be insightful to see how the model behaves to higher high-frequency noise in input sequence than just 1% in Figure 4/Section 5.5 experiment.

### Questions
1. Since this work built on the work by Park et al. 2024. Can you draw parallels with Park et al. 2024 including exact improvements and new insights. 

2. Details of the final model architecture used for empirical validation is missing. Can authors  please add more details about the model architecture including number of layers, parameters and also attention pattern visualization (in the next version). Releasing details about the empirical experiments would improve reproducibility.

Minor remarks
- There are a bunch of grammatical errors in the text. Please check the text carefully again. 
- Some notations in Algorithm 1 are not defined.

### Soundness
3

### Presentation
4

### Contribution
3
