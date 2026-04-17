# Intrinsic Entropy of Context Length Scaling in LLMs

- Decision: Accept (Oral)
- Scores: 6, 10, 2, 4

## Abstract
Long Context Language Models have drawn great attention in the past few years. There has been work discussing the impact of long context on Language Model performance: some find that long irrelevant context could harm performance, while some experimentally summarize loss reduction by relevant long context as Scaling Laws. This calls for a more thorough understanding of how long context impacts Language Modeling. In this work, we (1) propose to use `Intrinsic Entropy' for explaining the impact of context length on language modeling; and (2) conduct experiments on natural language and synthetic data, validating our proposed theoretical assumptions and deductions. Our theoretical framework can provide practical insights such as establishing that training dataset size dictates an optimal context length and bounds context length scaling for certain cases.
We hope our work may inspire new long context Language Models, as well as future work studying the physics of Language Models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a theoretical and empirical framework for
understanding how context length affects the performance and scaling
behavior of large language models. The authors propose the concept of
Intrinsic Entropy, which quantifies the information a model can encode
within its internal (“intrinsic”) representation space given a
certain context length. They decompose cross-entropy loss into two
parts — the Bayes Risk, representing the irreducible uncertainty from
limited context, and the Approximation Loss, representing imperfect
model learning. From this perspective, the Bayes Risk decreases
monotonically with longer context since more information is available,
whereas the Approximation Loss tends to increase because longer
contexts enlarge the model’s effective input dimension and complicate
optimization. The trade-off between these opposing effects gives rise
to an optimal context length that minimizes total loss for a given
dataset size.

Empirically, experiments using LLaMA-3.1 (8B and 70B) on OpenWebText
subsets and a synthetic parity dataset confirm the theory. The
measured relationship between Intrinsic Entropy and Cross-Entropy Loss
is nearly linear (correlation > 0.97), and the validation loss across
context lengths follows a scaling law of the form H=C0 + C / l^gamma.
. Moreover, when varying training-data size, the optimal context
length shifts upward — indicating that larger datasets justify
proportionally longer contexts. The findings imply that excessively
extending context windows can degrade performance when data are
insufficient, emphasizing the need for joint optimization of context
length, data size, and model capacity. This framework bridges
statistical physics–style scaling laws with information-theoretic
reasoning, offering a principled explanation for when and why long
contexts help or harm LLMs.

### Strengths
1. The paper introduces a novel concept — Intrinsic Entropy — that
links information-theoretic measures to model-internal representations
(“intrinsic space”). This provides a principled bridge between
entropy, Bayes risk, and approximation loss, enabling a quantitative
explanation for the existence of an optimal context length. By
deriving analytical relationships the work extends traditional
scaling-law theory (e.g., Kaplan et al. 2020) to the underexplored
dimension of context length. This is conceptually powerful: it
reframes the intuition that “longer is better” into a mathematically
grounded trade-off between information gain and representational
complexity.




2. Experimentally, the paper validates its theory using LLaMA 3.1 models
and synthetic data, showing a near-perfect linear correlation (R^2 >
0.97) between Intrinsic Entropy and Cross-Entropy loss, across varying
context lengths. The authors further identify a data-dependent optimal
context length, which scales predictably with dataset size, and
reproduce this result across different datasets. This dual
theoretical–empirical consistency gives the framework strong
credibility and practical relevance: it offers a diagnostic tool for
optimizing context length in training regimes and provides insight
into the physics-like scaling behavior of LLMs.  In short, the paper’s strength lies in its integration of
first-principles reasoning, strong empirical validation, and broad
applicability — offering both a new theoretical lens and actionable
insight into how LLMs handle long contexts.

### Weaknesses
1. The paper’s theory hinges on several idealized assumptions that may
not hold in practice: The notion of a stable “Intrinsic Space” — a well-behaved latent
manifold shared across models — is assumed rather than
demonstrated. In reality, representations in transformer layers are
non-stationary and architecture-dependent, making it difficult to
assert that a single intrinsic space can be defined across context
lengths. The linearity assumptions — e.g., that Bayes Risk is linearly
proportional to Intrinsic Entropy or Intrinsic Dimension — simplify
the analysis but are only loosely supported empirically. These
relationships might hold approximately in narrow regimes but could
break under different architectures, data modalities, or tokenization
schemes. While mathematically elegant, the framework risks being too abstract
to describe the heterogeneous behaviors of real LLMs under diverse
conditions.



2. The experiments, though carefully designed, rely on a small number of
datasets and model families: Only LLaMA 3.1 (8B and 70B) and OpenWebText subsets are tested for
natural language, with one synthetic dataset for controlled
validation. There is no downstream evaluation (e.g., reasoning or QA tasks) to
verify whether the proposed “optimal context length” correlates with
functional performance. The paper also lacks cross-model validation — results are not shown
for alternative architectures like Mistral or Gemini-style
mixture-of-experts, which could challenge the universality of the
scaling law. The findings, while compelling within the tested settings, remain
tentative as a general principle across model types and data regimes.



3. Although the “Intrinsic Entropy” construct is theoretically neat, it
is not directly measurable without heavy computational overhead (e.g.,
eigenvalue decompositions of hidden representations across
contexts). Moreover, its connection to optimization dynamics and
architecture-level design remains abstract — the paper suggests
guidelines for tuning context length but provides no concrete
algorithm or scaling heuristic to implement this in real-world
training. The work’s insights are conceptually illuminating but operationally
difficult to apply for practitioners designing next-generation
long-context LLMs.

### Questions
1. The entire theory rests on the existence of a stable Intrinsic Space
where internal representations of data reside and where entropy can be
measured. However, as the authors themselves admit, this space may
depend not only on the data but also on the neural network
architecture and training dynamics.  Can Intrinsic Entropy be defined independently of specific
architectures or layer representations, making it a universal property
of language modeling itself?


2. While the framework explains cross-entropy scaling, it is unclear
whether higher or lower Intrinsic Entropy predicts better reasoning,
retrieval, or long-document comprehension. The authors validate on
synthetic and language data but do not link the proposed measure to
functional metrics. Does Intrinsic Entropy correlate with actual task-level performance,
or does it capture only internal statistical regularities?

3. Can the optimal context-length principle generalize across models,
data types, and modalities? The observed law that the optimal context length increases with
dataset size is based on a limited set of models (LLaMA-3.1) and
corpora (OpenWebText, synthetic parity). Whether this scaling
relationship extends to multimodal LLMs, retrieval-augmented
architectures, or dialogue models remains untested.  Is the “optimal context length” phenomenon universal across model
families and data modalities, or an artifact of specific transformer
training regimes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The authors propose a new scaling law that takes context length into account. It has the form of
$$L = C_0 + C/l^\gamma + A(l)/D^{\alpha(l)}$$
Where l is the context length. This form is a natural generalization of the scaling law suggested by Kaplan et. al. to the case of a variable context length. It captures the trade-off between reduced training sample efficiency and increased generation confidence that LLM experiences as the context length grows.

The authors study the form of $\alpha(l)$ and show that it can be represented as $\alpha(l) = c\dim(l)$ with $dim(l)$ being the dimensionality of the latent space necessary to represent text of length l. Since larger text naturally may contain more information, the latent dimensionality grows with l, which means that $\alpha(l)$ decreases. Thus, the suggested scaling law consists of two terms, one of which increases with l while the other one decreases, which implies the existence of an optimal value of l.

The results summarized here are shown rigorously in the presented work. It is worth noticing that the relation between the coefficient $\alpha$ of the scaling law and the intrinsic dimension of the data was known in the literature. The authors push it further by leveraging the relationship between the intrinsic dimension and the context length.

### Strengths
The paper presents an interesting result, consisting of a non-trivial observation and a sound and consistent theory.

The authors provide a convincing set of evidence that supports their theoretical assumptions.

### Weaknesses
The paper claims that the theoretical framework presented can provide practical insights, such as establishing that the size of the training dataset dictates an optimal context length and bounds context-length scaling. It seems to miss that, in practice, the context length is determined not to minimize the loss function but rather for other considerations, such as the typical context length in the target use cases and the capabilities of the computational hardware. If the prescribed “optimal” context length is larger than the typical context length in applications, the context length will still be kept within the application's demands due to computational constraints. If the prescribed “optimal” context length is shorter than the typical context length in applications, the context length will still be kept up to the application's demand, or the system would not be deployable.

Another argument against the practicality of the delivered result is that the real objective behind LLM training is not the cross-entropy values, but the evaluation results on downstream metrics. To prove the practicality of the results, the authors would need to conduct experiments that hold compute/memory constant and study task metrics (e.g., RAG accuracy, long-form QA F1, code completion pass@k) against context length

### Questions
NA

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes to investigates into LLM context length from model entropy perspective. The paper leverages cross entropy (and treated as Bayes Risk in the paper, for example in Figure 2 and explained in 2.2.2) and show that it empirically bears a linear relationship with intrisic entropy which is calculated as log of the eigenvalues. Furthermore, by approxmimating the model approximation error in section 2.3, the paper shows theoretically that there exists an optimal context length in section 3 and also demonstrate it empirically.

### Strengths
The direction which contructs a decomposition and shows that optimally there exists an optimal context length as shown in section 3 is very interesting. Eq6/7 effectively shows that D would imply an optimal L. There are several empirical results that demonstrate this; besides, the linear relationship between cross entropy and intrinsic entropy seems novel and insightful.

### Weaknesses
I personally think that the mathematical construction lacks rigor. For example, there is no justification of equaling states with volume divided by dimensions. Although eq(3) is understandable, it is not an acceptable exposition since l is discrete. It is also problematic to directly treat crossentropy as Bayes Risk particularly there is no justification provided throughout the paper.

The paper is badly presented. The figures are not very readable most of the time with non informative titles (e.g. Figure 1, part1/part2/part3) and equations through 381-386. All these make papers more read as a draft instead of a formal conference submission. All the above significantly makes the reading process harder. For example, I am not able to understand the synthetic task presented in Figure 5 right part (with carefully reading its descriptions 372-377)/

### Questions
Figure 1 and Figure 7 shows the correlation between cross entropy and N. How are samples obtained for a particular N please? More in detail, how is correlation calculated for a particular (N, cross entropy) pair?

### Soundness
1

### Presentation
1

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
This paper investigates the impact of context length on Language Model (LM) performance through a novel theoretical lens termed "Intrinsic Entropy." The authors posit that the cross-entropy loss can be decomposed into Bayes Risk (the irreducible error of an optimal model) and Approximation Loss (the error from the trained model failing to match the optimal model). The core contribution is the proposal that the Bayes Risk is linearly related to the Intrinsic Entropy of the data manifold, which itself is a function of context length. They further argue that while longer contexts reduce Bayes Risk by providing more information, they simultaneously increase the intrinsic dimension, making the Approximation Loss harder to minimize, especially with limited data. This trade-off leads to an "optimal context length" for a given training dataset size, beyond which validation loss increases. The paper supports its claims with experiments on both natural language (OpenWebText with LLaMa and GPT-2) and a carefully constructed synthetic dataset.

### Strengths
1. While the decomposition of loss into Bayes Risk and Approximation Error is standard, and the use of an "intrinsic space" is inspired by prior work, the formulation of "Intrinsic Entropy" as a central concept to bridge context length and loss is novel. 
2. The idea that there is a fundamental trade-off leading to an optimal context length is a significant and non-obvious insight that challenges the simplistic view that "more context is always better." 
3. The paper is generally well-structured and clear.

### Weaknesses
My main concern is that existing work [1] has pointed out that PPL cannot serve as a standard for evaluating long-text performance. Therefore, experimental results on more diverse long-text benchmarks are necessary, such as RULER, HELMET, and Longbench v2.

[1]Fang L, Wang Y, Liu Z, et al. What is Wrong with Perplexity for Long-context Language Modeling?[J]. arXiv preprint arXiv:2410.23771, 2024.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
