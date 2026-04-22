# Provable Separations between Memorization and Generalization in Diffusion Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Diffusion models have achieved remarkable success across diverse domains, but they remain vulnerable to memorization---reproducing training data rather than generating novel outputs. This not only limits their creative potential but also raises concerns about privacy and safety. While empirical studies have explored mitigation strategies, theoretical understanding of memorization remains limited. We address this gap through developing a dual-separation result via two complementary perspectives: statistical estimation and network approximation. From the estimation side, we show that the ground-truth score function does not minimize the empirical denoising loss, creating a separation that drives memorization. From the approximation side, we prove that implementing the empirical score function requires network size to scale with sample size, spelling a separation compared to the more compact network representation of the ground-truth score function. Guided by these insights, we develop a pruning-based method that reduces memorization while maintaining generation quality in diffusion transformers.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a theoretical analysis of the memorization phenomenon in diffusion models. The analysis offers two perspectives: a statistical and an architectural one. The statistical perspective distinguishes between the ground-truth score function, which reflects the underlying data distribution, and the empirical score function that minimizes the loss on the (sampled) training data. Optimizing the diffusion model pushes the model to learn the empirical score function, which minimizes the training loss, instead of approximating the ground-truth score function, thereby leading to memorization. The subsequent analysis of the network complexity (number of parameters) demonstrates that over-parameterized diffusion models learn the complex empirical score function, also leading to memorization. Based on these theoretical insights, the paper explores mitigation strategies by reducing the model complexity by pruning attention heads or weight decay. These experiments are conducted on CIFAR-10 and Gaussian mixtures.

### Strengths
- The paper provides fascinating insights into the training dynamics and influencing factors favoring memorization in diffusion models. Thorough mathematical foundations support the proposed framework. All necessary proofs are provided in the main paper or the appendix.
- The theoretical analysis of the loss gap between the ground-truth and the empirical score function offers an intuitive explanation of why diffusion models can inadvertently memorize and replicate parts of their training data.
- While, to some extent, this is to be expected, the pruning- and regularization-based mitigation strategies support model developers in their design decisions for building diffusion models and setting the training parameters.

### Weaknesses
- While the paper provides an interesting perspective on diffusion memorization, the core insight that a model tends to minimize the empirical loss instead of the ground-truth loss feels like the traditional bias-variance trade-off and overfitting phenomena in statistical machine learning. I do not want to reduce the paper’s contribution, but want to emphasize that the high-level message of the paper might have limited novelty. However, it might be the case that I overlooked some crucial novelty of the paper here.
- The paper focuses on unconditional and low-resolution diffusion models trained on comparably small datasets (a couple of thousand training samples). Whereas I think this is appropriate for the provided analysis, it remains unclear to what extent the results transfer to more complex distributions, such as LAION data, and sample counts in the billions. Yet, I acknowledge that running such experiments is not feasible for small research labs, which is why I do not request such experiments. However, a more thorough discussion on how the authors expect their findings to scale or behave in such regimes would benefit the paper.
- The results of the proposed pruning mitigation support an improvement in memorization mitigation. However, the reduction is relatively small (about 5 percentage points compared to the baseline, still reporting a 68% memorization ratio). Therefore, it feels less suitable for practical application.

### Questions
- Do the findings also apply to large-scale diffusion models, e.g., Stable Diffusion trained on billions of data samples?
- As a follow-up question: We know that Stable Diffusion has memorized some of its training samples. Does this mean the capacity of Stable Diffusion is too large? Or can we assume, since only a small share of duplicated training samples has been memorized, that Stable Diffusion already has a sufficient size, and the issue is the duplicated dataset?

### Soundness
4

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
This paper theoretically proves a fundamental separation between memorization and generalization in diffusion models. It shows that finite-sample Fisher divergence and network capacity jointly cause models to fit empirical rather than true score functions. Experiments demonstrate that pruning and regularization can reduce memorization while preserving generation quality.

### Strengths
1. The paper provides a rigorous theoretical analysis of diffusion models by quantifying the loss gap between the empirical and ground-truth scores under sub-Gaussian, Hölder-smooth data distributions, and further establishes an architectural separation showing that approximating the empirical score requires higher model complexity.
2. The paper proposes a practical one-shot pruning method for Diffusion Transformers that removes low-importance heads in the small-t regime, effectively reducing memorization while maintaining or even improving generation quality, as supported by CIFAR-10 experiments.

### Weaknesses
1. The analysis mainly focuses on the small-t regime, but the paper does not empirically verify whether memorization indeed concentrates in this phase during real generation — an experiment comparing large-t and small-t generations could strengthen the claims.
2. I think The theoretical results rely on strong assumptions such as sub-Gaussianity, which limit their applicability to real-world data.
3. The proposed pruning method is tested on a single dataset, its generalization to other datasets or more complex diffusion models is uncertain, and its ability to substantially reduce memorization appears limited.

### Questions
See Weakness. Further:
    1. The results rely on sub-Gaussianity and Hölder smoothness assumptions. How sensitive are your main theorems to these conditions?

### Soundness
3

### Presentation
4

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
This paper develops a theoretical framework for explaining memorization and generalization in diffusion models. First, it shows that the denoising score matching loss with respect to an empirical distribution is not minimized by the ground-truth score function, due to a statistical lower bound.
Second, it proves that under the universal approximation perspective, representing the empirical score requires larger networks than the case of ground-truth score.
These results establish quantitative separations between the regimes where diffusion models memorize versus generalize.
The authors further propose a pruning scheme for mitigating memorization while preserving generation quality.

### Strengths
The paper provides a solid theoretical analysis necessary for understanding the phenomenon of memorization and generalization in diffusion models. It rigorously establishes, in a statistical sense, the loss separation between the true and empirical score functions on empirical data, and quantitatively characterizes the network complexity required to learn each by leveraging results from universal approximation theory, which are appreciable theoretical contributions. Moreover, the authors effectively convey the high-level intuition behind their theorems, enabling readers to grasp the core ideas without following every technical detail.

### Weaknesses
- **Unclear theoretical hypothesis and motivation.** It is not entirely clear what specific hypothesis the theory aims to formalize or explain. Judging from the experiments, the paper seems to intend to relate sample size, network size, and weight decay to memorization and generalization in diffusion models, but the central claim or insight remains vague. The analysis does show that limited sample size can cause the empirical loss to favor memorization over generalization, but this is well-known and qualitatively unsurprising. It is unclear whether the authors' goal is to quantify the transition (e.g., identifying a regime such as $\log n = O(d)$) or to argue its relevance to large-scale diffusion models. The theoretical results are not explicitly connected to the empirical observations in Section 6.1, making it difficult to understand what the experiments are meant to test or demonstrate.

- **Weak empirical validation.**
The experiments do not concretely demonstrate the quantitative connection to the theory, e.g., the right balance between sample and network size is not discussed. Some claims, such as Lines 402–403 (“With sufficient sample size, network width increase promotes generalization”), depend heavily on narrow parameter ranges and is not properly substantiated (network width much greater than 1024 may eventually lead to memorization even for n=10K samples). The discussion of weight decay remains tautological (“proper network width and weight decay prevent memorization”) without clarifying what constitutes a proper scaling or whether such relationships can be derived from the theory.


- **Ambiguous role and limited contribution of the pruning method.**
The pruning scheme introduced in Section 6 seems disconnected from the main theoretical narrative. Reducing model size to mitigate memorization seems to be a straightforward idea and is not clearly tied to the presented theoretical framework. Its empirical impact appears modest, and many existing fine-tuning or guidance-based unlearning approaches are likely more effective. Its inclusion blurs the paper’s focus between theoretical analysis and engineering heuristics.


Overall, the paper seems uncertain about its main contribution: as a purely theoretical paper, the results are technically sound but conceptually expected; as a scientific study linking theory and empirical phenomena, it lacks a clear hypothesis-testing structure; and as an applied work, the proposed pruning method does not add substantial novelty.

### Questions
1. What is the central hypothesis that the experiments are designed to test? How are the theoretical results meant to explain the observed effects of sample size, network size, and weight decay on memorization and generalization?

2. Does the theory predict a quantitative regime (e.g., for sample size, $\log n = O(d)$ or larger) in which generalization can emerge, and are the chosen sample sizes in experiments intended to probe that regime?

3. Regarding the statement that “proper network width and weight decay prevent memorization,” can your theory specify how network size and weight decay should quantitatively relate to achieve generalization?

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
4

### Summary
The paper proposes a theoretical approach to understand and mitigate memorization in diffusion models. The paper shows that there is (even with a polynomial number of training samples) a non-negligible loss gap for smaller t, which leads to the empirical loss being minimized, which in turn leads to memorization. As a solution, a pruning-based approach is presented, pruning the attention heads with the lowest importance score.

### Strengths
- The paper presents a strong theoretical understanding of why memorization happens in diffusion models.
- The paper shows that even with more training data, the loss gap will be present, showing that adding more training data to prevent memorization is not sufficient.

### Weaknesses
- In the experimental section, it is not mentioned which model architecture was used.
- The experiments are only conducted on small datasets such as the CIFAR-10 dataset and a synthetic Gaussian mixture dataset
- The pruning-based method is not evaluated against other SOTA pruning-based methods.

Misc:
- In line 95 there is a typo in "correspnding"
- Line 103 "an" -> "a"

### Questions
Q1: How does this theoretical approach support the empirical observation that deduplication seems to help with memorization mitigation?  
Q2: Why is the fine-tuning after pruning the attention heads necessary? And more importantly, how do you make sure that no new samples in the fine-tuning set are memorized?  
Q3: How are recall and precision calculated in Table 1?  
Q4: Does this only hold for unconditioned text-to-image diffusion models? Does this also hold for other noise schedulers?

### Soundness
3

### Presentation
3

### Contribution
3
