# On the Reasoning Abilities of Masked Diffusion Language Models

- Avg Score: 7.00
- Decision: Accept (Oral)
- Scores: 6, 8, 8, 6

## Abstract
Masked diffusion models (MDMs) for text offer a compelling alternative to traditional autoregressive language models. Parallel generation makes them efficient, but their computational capabilities and the limitations inherent in their parallelism remain largely unexplored. To this end, we characterize what types of reasoning problems MDMs can provably solve and how efficiently. We do this by connecting MDMs to the well-understood reasoning frameworks of chain of thought (CoT) and padded looped transformers (PLTs) in the finite-precision log-width setting: We show that MDMs and polynomially-padded PLTs are, in fact, equivalent in this setting, and that MDMs can solve all problems that CoT-augmented transformers can. Moreover, we showcase classes of problems (including regular languages) for which MDMs are inherently more efficient than CoT transformers, where parallel generation allows for substantially faster reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper analyzes the computational capability of MDMs, which is composed of a planner and a predictor, both implemented by transformers. One core finding is the equivalence between MDMs and PLTs under the discussed setting. Based on this equivalence, this paper characterizes the computational capability of MDMs and provides a comparison between MDMs and CoT Transformers.

### Strengths
1. The paper is well-written and provides a detailed discussion of the problem setting, related work, and its theoretical assumptions.
2. The target question, the computational capability of MDMs, is practically relevant and important.
3. The proposed "planner-predictor" formulation for MDMs is reasonable.
4. The approach of linking MDMs to the PLTs is natural and allows the use of well-studied conclusions about PLTs.

### Weaknesses
1. Positional encodings. The main theoretical results appear to rely on very strong assumptions about the positional encodings. As discussed in Appendix D.1, the PEs are constructed to carry complex algorithmic information (e.g., the results of division and modulo operations). This assumption seems to offload a large part of the required computation from the Transformer's computation mechanism onto the input representation.
2. Idealized planner.  In the proofs, the planner is a Transformer designed to perfectly execute a specific algorithm. It is unclear if the practical confidence-based unmasking planner could replicate this perfect capacity.
3. Empirical study. While the work is theoretical, the results would be strengthened by even simple empirical studies to see the practical performance of the predicted capabilities (e.g., the efficiency of MDMs on parallelizable tasks versus CoT).

### Questions
1. Positional encodings. How much of the computational capability attributed to MDMs (e.g., in Theorem 3.2) is actually due to the pre-computed information in the PEs rather than the Transformer’s computation mechanism? This should be more clearly discussed.
2. Idealized planner. How could practical planners approximate the performance of the idealized planners? A more detailed explanation or empirical investigation of this gap would strengthen the paper's real-world implications.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper explores the reasoning capabilities of MDMs from the standard complexity theory perspective. The main message of this paper is that MDMs excel on highly parallelizable or ambiguous-generation tasks, achieving strong results with few denoising steps and easy steering via subtask learning. The authors also provide rigorous theoretical statements to support this claim.

### Strengths
This paper works on a topic that previous work didn't explore at all: theoretically exploring the MDM's reasoning ability. Although the community believes that MDM can achieve better performance than a causal model in tasks that require non-causal thinking, formalizing that intuition would've been needed for the community. Last but not least, the paper is clearly well-written, even people who aren't familiar with complexity theory can easily follow.

### Weaknesses
This paper doesn't provide any empirical results, which is not actually in the scope, though. Moreover, although the introduction is clearly well-written, Figure 1 gives too much information and is even a bit hard to follow. Also (as far as I understand), this paper's theoretical analysis is based on a remasking scheme (where we remask the unmasked tokens again), which people don't really use in their large-scale Masked Diffusions. Although the authors provide two MDM references that use this scheme, my thought is that the remasking strategy is actually not the mainstream sampling approach, at least by far.

### Questions
- How does the theory result affect under without remasking strategy?
- Are there (at least some prior work's) empirical results that can support the author's claim?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides a formal theoretical analysis of the reasoning capabilities of Masked Diffusion Models (MDMs) for language generation. It establishes equivalence between MDMs and Padded Looped Transformers (PLTs) under finite-precision, log-width settings, and compares their expressivity to Chain-of-Thought (CoT) transformers. The authors show that MDMs can simulate CoT reasoning (with some overhead), and are provably more efficient on parallelizable problems. They also identify a "sequentiality bottleneck" in CoT transformers, which MDMs can overcome due to their parallel nature. The paper concludes that MDMs are better suited for parallelizable reasoning tasks, while CoT is more efficient for inherently sequential ones.

### Strengths
1. The paper provides formal proofs and complexity-theoretic characterizations of MDMs, grounding their reasoning capabilities in well-understood models like PLTs and CoT transformers.
2. It introduces the idea of a "sequentiality bottleneck" in CoT and shows how MDMs can leverage parallelism, offering a clear separation in expressive efficiency for certain problem classes (e.g., regular languages, NC1).
3. The idealized MDM model is motivated by practical implementations, and the authors show how their theoretical framework aligns with real-world MDM behaviors, such as confidence-based unmasking and resampling.

### Weaknesses
1. The paper is purely theoretical and lacks experimental validation of the claims. While theoretical depth is valuable, even synthetic experiments could help illustrate the practical implications of the findings.
2. The assumptions may be too strong. The analysis relies on idealized assumptions (e.g., finite-precision, log-width transformers, perfect planners/predictors), which may not reflect real-world limitations of MDMs or PLTs.
3. The reasoning tasks considered (e.g., regular languages, circuit complexity classes) are formal and abstract. It’s unclear how the results translate to more complex, open-domain reasoning tasks commonly faced by LLMs.

### Questions
1. Do you plan to validate your theoretical findings empirically? For example, can you design controlled experiments to show that MDMs outperform CoT on parallelizable tasks like expression evaluation or state tracking?
2. How do your assumptions affect the realism of the model? What are the implications of relaxing assumptions like perfect approximation or uniform unmasking? How would your results change under noisy or learned planners?
3. Can your framework be extended to other reasoning paradigms? For instance, could similar analyses be applied to latent diffusion models, state-space models, or hybrid autoregressive-diffusion architectures?

### Soundness
3

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
This paper provides a formal analysis of the reasoning and computational capabilities of MDM. Under a finite-precision, logarithmic-width transformer setting, the authors prove that MDMs are theoretically equivalent to PLTs and can simulate CoT reasoning. They further show that MDMs are provably more efficient only for parallelizable problems (e.g., regular languages), where parallel denoising enables faster reasoning, while for inherently sequential tasks (e.g., P-complete problems) MDMs offer no efficiency gain and may even be less practical due to architectural overhead.

### Strengths
1. The paper offers a novel and rigorous theoretical framework for analyzing the reasoning capability of masked diffusion models, a topic that has been largely unexplored.

2. It provides clear conceptual connections between MDMs, chain-of-thought transformers, and padded looped transformers, helping to unify different reasoning paradigms under one formulation.

3. The analysis yields meaningful theoretical insights into when MDMs can achieve efficiency gains through parallelism, offering guidance for future research on diffusion-based reasoning models.

### Weaknesses
1. The theoretical framework relies on strong assumptions about positional encodings, which are constructed to include arithmetic information (e.g., division and modulo) that real transformers cannot compute, potentially overstating MDMs’ practical capability.

2. The analysis is purely theoretical, without even minimal empirical validation or illustrative experiments to verify whether the predicted efficiency gains appear in practice.

### Questions
1. The paper makes strong assumptions about positional encodings, requiring them to include arithmetic information such as division and modulo. Do widely used schemes like RoPE [1] satisfy these assumptions?

2. The authors argue that for P-complete problems, MDMs cannot benefit from parallelism and that the absence of KV-cache makes autoregressive models preferable. Would this conclusion still hold if we consider recent MDM variants that incorporate KV-cache [2]?

[1] Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding.

[2] Wu et al. Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding.

### Soundness
3

### Presentation
3

### Contribution
3
