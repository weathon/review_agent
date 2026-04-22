# Secret-Protected Evolution for Differentially Private Synthetic Text Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Text data has become extremely valuable on large language models (LLMs) and even lead to general artificial intelligence (AGI).
A lot of high-quality text in the real world is private and cannot be freely used due to privacy concerns. Therefore, differentially private (DP) synthetic text generation has been proposed, aiming to produce high-utility synthetic data while protecting sensitive information.
However, existing DP synthetic text generation imposes uniform guarantees that often overprotect non-sensitive content, resulting in substantial utility loss and computational overhead. Therefore, we propose Secret-Protected Evolution (SecPE), a novel framework that extends private evolution with secret-aware protection. 
Theoretically, we show that SecPE satisfies $(\vp, \vr)$-secret protection, constituting a relaxation of Gaussian DP that enables tighter utility–privacy trade-offs, while also substantially reducing computational complexity relative to baseline methods.
Empirically, across the OpenReview, PubMed, and Yelp benchmarks, SecPE consistently achieves lower Fréchet Inception Distance (FID) and higher downstream task accuracy than GDP-based Aug-PE baselines, while requiring less noise to attain the same level of protection. 
Our results highlight that secret-aware guarantees can unlock more practical and effective privacy-preserving synthetic text generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes **Secret-Protected Evolution (SecPE)**, a novel framework for generating high-fidelity synthetic text under formal privacy constraints. Unlike traditional Differential Privacy (DP), which enforces uniform guarantees across all data points, SecPE introduces the concept of **Secret Protection**, focusing on safeguarding specific sensitive content (“secrets”) rather than entire records. The framework extends the Private Evolution (PE) paradigm and integrates a new **Secret Clustering** module, which leverages public data and limited noisy private updates to build secret-aware cluster centers for efficient selection. In the **Protected Evolution** phase, candidate synthetic texts are iteratively generated and selected based on similarity to these noisy representatives. Theoretical analysis shows that SecPE satisfies $(p, r)$-secret protection—a relaxation of Gaussian Differential Privacy (GDP)—and yields tighter utility–privacy trade-offs. Empirical results on **OpenReview**, **PubMed**, and **Yelp** datasets demonstrate that SecPE achieves lower FID, higher downstream accuracy, and significantly reduced computational cost compared to µ-GDP–based Aug-PE baselines.

### Strengths
### Strengths
1. The paper introduces a **new privacy notion, (p, r)-secret protection**, that generalizes and relaxes Gaussian DP, offering a theoretically grounded yet more flexible privacy-utility balance.
2. The proposed **SecPE algorithm** effectively combines secret-aware protection with the efficiency of K-means–based clustering, reducing computational complexity from $O(MN_{syn})$ to $O(KN_{syn})$.
3. **Empirical results** on multiple datasets show consistent performance improvements, with SecPE achieving higher downstream accuracy and lower FID values compared to Aug-PE.
4. The framework provides a **clear theoretical linkage** between secret protection and existing DP frameworks (DP, GDP), thereby establishing conceptual coherence.
5. The experimental section is thorough, including analyses on efficiency, downstream accuracy, FID, and ablations over LLM size and clustering hyperparameters.

### Quality
The paper is technically solid and theoretically well-founded. The connection between secret protection and Gaussian DP is clearly derived and mathematically consistent. The experimental evaluation is extensive and replicable. However, the lack of a real-world deployment or qualitative human evaluation of the generated text slightly limits the practical validation.

## Clarity
The paper is **clearly written and well structured**, especially in the methodology section. Figures and algorithms (notably Algorithms 1–3) provide a good overview of the pipeline. However, some mathematical symbols (e.g., blow-up function and trade-off function) could be briefly explained when first introduced to enhance readability for non-privacy experts.

### Significance
SecPE represents a **notable conceptual advance** in privacy-preserving text generation, shifting from record-level to secret-level guarantees. Its relaxation of GDP could inspire a new line of research in secret-aware privacy mechanisms, potentially extending beyond text to multimodal settings. The method’s computational efficiency also makes it practical for real-world applications where DP-finetuning is infeasible.

### Weaknesses
### Weaknesses
1. The paper would be strengthened by **a more comprehensive comparison with DP-finetuned LLMs**, which, while computationally costly, have been shown to perform competitively in private text synthesis.
2. Although the paper reports improvements in synthetic quality, **SecPE’s performance on personally identifiable information (PII)** tasks shows only marginal gains. The method’s real-world effectiveness in dense secret scenarios therefore remains somewhat inconclusive.
3. The definition of "secret" remains **qualitative and application-dependent**. The paper acknowledges this limitation but does not explore strategies for adaptive or data-driven secret detection beyond keyword-level identification.
4. The **sensitivity of the algorithm to different clustering sizes (K)** and to varying prior parameters $(p, r)$ could be analyzed more rigorously, as these hyperparameters directly influence both privacy guarantees and utility.

### Questions
1. How sensitive is SecPE’s performance to the choice of secret prior probabilities \(p_j\)? Would a mis-specified prior significantly degrade the protection or utility?
2. Could the authors discuss how SecPE scales when the number of secrets grows very large (e.g., thousands of potential sensitive attributes)?
3. The authors mention that clustering abstracts away fine-grained details. Could this abstraction lead to mode collapse or reduced diversity in the generated synthetic data?
4. In *user-level DP*, the protection unit is an **entire user’s data** (i.e., all records belonging to a user), whereas SecPE protects **individual secrets** regardless of user identity. Could SecPE be extended to user-level protection, or combined with it to achieve multi-granular privacy?

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
3

### Summary
This paper proposes Secret-Protected Evolution (SecPE), a novel framework for generating synthetic text that addresses key limitations in existing privacy-preserving methods. The authors identify that standard Differential Privacy (DP), imposes a uniform privacy guarantee that over-protects non-sensitive data, leading to unnecessary utility loss. The paper provides a strong theoretical foundation, proving the pipeline satisfies (p, r)-secret protection. Empirically, evaluations on OpenReview, PubMed, and Yelp benchmarks show SecPE consistently outperforms a GDP-based baseline.

### Strengths
Strengths include the novel and practical privacy formulation, some computational efficiency gains, comprehensive evaluation across multiple models and datasets

### Weaknesses
(1) The framework also relies on a clear definition of "secrets" and their prior probabilities (p), which is left somewhat ambiguous and could be a practical hurdle. 

(2) The presentation needs some improvement.   For example, the sentence on LIne 75-77 is not understandable.   The definitions needs some background knowledge, which is absent. 

(3) Finally, performance improvements are more modest when secrets are dense (as in a PII task), and the method's dependence on representative public data for clustering is not fully explored.

### Questions
(1) How can secrets be systematically defined and their priors quantified in real-world scenarios?

(2) Could adaptive clustering mitigate the minor utility loss in non-private settings?

(3) How does the framework compose over multiple data releases

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel Secret Protection Evolutionary framework (SecPE) for differentially private (DP) synthetic text generation. The work aims to address the issue of significant utility loss and high computational overhead associated with traditional DP methods in text generation. By introducing the concept of Secret-Protected Evolution	as an alternative to uniform DP guarantees,  it achieves a better utility-privacy trade-off. In the theoretical part, the paper defines a (p, r)-secret protection criterion and establishes its connection to Gaussian DP. The experimental section validates the advantages of SecPE across several benchmark datasets (OpenReview, PubMed, Yelp), demonstrating reductions in Fréchet Inception Distance (FID), improved accuracy on downstream tasks, and decreased runtime.

### Strengths
1.This work proposes a (p,r)-secret protection framework and establishes its theoretical connection to Gaussian Differential Privacy (GDP), offering a new perspective for privacy-preserving research.
2.The experimental design is comprehensive, and the validation is thorough.

### Weaknesses
1.The paper operates on the assumption that "secrets" can be predefined; however, in real-world scenarios, sensitive content is often dynamic and non-enumerable.
2.The paper fails to analyze the impact of the cluster number K on the effectiveness (it only mentions "insensitive" but provides no ablation studies).
3.The improvement on the PII task is relatively limited (Table 7), potentially due to high secret density diminishing the protection benefits. Further analysis on the impact of secret sparsity is required.

### Questions
The manuscript employs the term "operative point" on multiple occasions without providing a formal definition.

### Soundness
4

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
3

### Summary
The authors adapt private evolution to offer guarantees under the “secret-protection” framework, which is a recent DP relaxation proposed by (Ganesh et al., 2025). They introduce a clustering step with public data to improve efficiency of PE. They conduct experiments by instantiating the secret-protection framework by declaring random words as “secrets” and examples containing them to betray the secret, showing utility improvements over regular PE that provides uniform protection.

### Strengths
- The topic is of high importance to the community. Secret protection is a promising DP relaxation that can potentially address many “type mismatches” between DP and the kind of sensitive data that companies own and want to extract value out of. Synthetic data via private evolution is one of the least resource- and expertise-intensive ways to get something out of your sensitive data. For these two reasons, finding a good design here will unlock a lot of useful practical applications.

- The experiments demonstrating utility improvement of SecPE compared to PE, in the setting where the secret protection definition is instantiated by declaring PII or random words as “secrets” (with examples containing them considered to betray the secret), are well-executed and highly interesting.

### Weaknesses
- As secret protection is a relatively nascent privacy framework, I believe the authors could do more in terms of exposition in Section 3.1, as well as (1) give the specific instantiation of (S,E,T) (in the notation of (Ganesh et al., 2025)) that defines the neighboring relationship fundamental to applications of secret protection, and (2) describe how secret clustering relates to secret protection.

  - Specifically, the authors give a relatively terse statement of the secret-protection definition. Lemma 3.2 proves GDP implies a particular instantiation of secret protection. Rather than focusing on this point, the main text would be better served to describe further the algorithmic changes to PE enabled by relaxation of the DP definition.

- The core algorithmic tool for enabling the secret protection guarantee is from (Ganesh et al., 2025). Nothing particular PE under secret protection is introduced.

- While authors report computational efficiency gains from SecPE, these gains feel misattributed. Reported speedup is in terms of improving pairwise similarity calculations. First, it is relatively unintuitive to me that the bottleneck is similarity computation, rather than iterative rewrites with a possibly expensive API model. Perhaps a FLOPs or memory analysis of the operation, compared to inference FLOPs and memory would shed some light here. Second, avoiding pairwise similarity computations seems orthogonal to SecPE, and there exist many simple and practical solutions to the problem for regular PE. For example:
  - Using approximate nearest neighbours search over the index of synthetic examples (https://github.com/google-research/google-research/tree/master/scann).
  - Clustering the synthetic data, and voting on clusters. Although not used in PE, this is employed in the highly-related work on private postprocessing (https://arxiv.org/abs/2402.13659), which can be viewed as half of one step of private evolution.

### Questions
- Secret protection is instantated by a (S,E,T) tuple (in the notation of Ganesh et al, 2025). In experiments, what is T?

- Any intuitions about how the number of secrets and their choice affect the utility improvement of SecPE compared to PE? Is this entirely predicted by the computed noise and sampling parameters?

### Soundness
4

### Presentation
2

### Contribution
2
