# What Scales in Cross-Entropy Scaling Law?

- Decision: Accept (Poster)
- Scores: 2, 8, 6, 4

## Abstract
The cross-entropy scaling law has long served as a key tool for guiding the development of large language models. It shows that cross-entropy loss decreases in a predictable power-law rate as the model size increases. However, recent evidence indicates that this law breaks down at very large scales: the loss decreases more slowly than expected, which causes significant trouble for developing large language models. In this paper, we hypothesize that the root cause lies in the fact that cross-entropy itself does not truly scale; instead, only one of its hidden components does. To investigate this, we introduce a novel decomposition of cross-entropy into three parts: Error-Entropy, Self-Alignment, and Confidence. We show both theoretically and empirically that this decomposition precisely captures the training dynamics and optimization objectives. Through extensive experiments on multiple datasets and 32 models spanning five orders of magnitude in size, we find that only error-entropy follows a robust power-law scaling, while the other two terms remain largely invariant. Moreover, error-entropy constitutes the dominant share of cross-entropy in small models but diminishes in proportion as models grow larger. This explains why the cross-entropy scaling law appears accurate at small scales but fails at very large ones. Our findings establish the error-entropy scaling law as a more accurate description of model behavior. We believe it will have wide applications in the training, understanding, and future development of large language models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates why the widely used cross-entropy scaling law—which predicts that cross-entropy loss decreases as a power law with model size—begins to fail for very large language models. The authors hypothesize that cross-entropy itself does not truly scale, but rather only a hidden component within it does. To uncover this, they introduce a novel mathematical decomposition of cross-entropy into three distinct parts: Error-Entropy, Self-Alignment, and Confidence. Using a new metric called Rank-Based Error (RBE)—the rank position of the correct token—the paper expresses cross-entropy as the sum of the entropy of the RBE distribution (Error-Entropy), the KL divergence between that distribution and the model’s score distribution (Self-Alignment), and a confidence term related to score normalization. This decomposition offers a more interpretable view of what cross-entropy measures in language model training.

Through extensive experiments across multiple datasets (Wikipedia, C4, and GitHub) and 32 models spanning five orders of magnitude in size, the authors find that only Error-Entropy consistently follows a clean power-law scaling with model size, while the other two components remain mostly invariant. They therefore propose an Error-Entropy Scaling Law as a more fundamental description of model behavior. The paper argues that this new law explains why the classical cross-entropy scaling law appears valid for small models but breaks down for large ones—because Error-Entropy dominates at small scales but shrinks in relative importance as models grow. The authors conclude that Error-Entropy may provide a more accurate and theoretically meaningful measure for understanding and predicting large model performance.

### Strengths
1. The paper’s most significant strength is its conceptual novelty.  By
decomposing cross-entropy into Error-Entropy, Self-Alignment, and
Confidence, it provides a new interpretive framework for understanding
what cross-entropy actually measures in language model training. This
move reframes the long-standing “scaling law” discussion away from
raw loss values toward a more structured, information-theoretic
understanding of model behavior.



2. The work offers a coherent explanation for an empirical puzzle that
has troubled practitioners: why the cross-entropy scaling law fits
well for small models but fails at very large scales.  By showing that
only Error-Entropy continues to follow a power law, the paper proposes
a simple and intuitive reason — as models grow, non-scaling
components (Self-Alignment and Confidence) dominate.  This insight
gives practical and theoretical clarity to the field’s observed
irregularities.



3. The decomposition is mathematically clean and transparent.  Starting
from the standard definition of cross-entropy, the authors derive the
decomposition step by step (Eq. 5–8, p.5), connecting it to
measurable quantities like rank distributions and score norms.  This
clear derivation enhances reproducibility and provides a framework
that can be extended or critiqued rigorously by others.



4. The paper demonstrates solid empirical design: it tests the hypothesis
across 32 language models (Pythia, GPT-2, Qwen, LLaMA, OPT, Mistral,
etc.) and three distinct corpora (Wikipedia, C4, GitHub) over five
orders of magnitude in scale.  Although limited in token sampling,
this breadth of architectures strengthens the empirical robustness and
helps generalize the scaling claim within the text domain.


5. Finally, the findings have broad implications.  Practically,
recognizing that only Error-Entropy scales could inform training
objectives and compute allocation for large models.  Theoretically,
the decomposition opens a potential bridge between rank-based
evaluation and information theory, offering a new direction for
research on learning dynamics and interpretability in large-scale
neural systems.

### Weaknesses
1. Although the decomposition is mathematically valid (Eq. 8, p. 5), it
is fundamentally algebraic rather than theoretical.  The decomposition
introduces new quantities (RBE), but this does not derive them from a
principled model of language or probability theory. In other words, “Error-Entropy” is justified empirically, not from first principles
— there is no information-theoretic or Bayesian derivation explaining
why the entropy of rank distribution should obey a power law. Consequently, the argument is more descriptive than explanatory: it simply
re-expresses cross-entropy in new coordinates rather than uncovering
causal structure. In short: the authors show that “error-entropy scales,” but not why
it should do so.


2. The central variable, Rank-Based Error (RBE), depends on sorting model
outputs by rank (p. 4 – 5).  While this makes intuitive sense (rank as semantic accuracy), it has drawbacks.  Primarily,  ranking operations are non-differentiable. Hence, RBE and its entropy cannot directly serve as a training objective; the authors
mention this only implicitly. Rank distributions depend on vocabulary size.  This can distort entropy values when models differ in
tokenization or vocabulary. Thus, “error-entropy” may not generalize well across architectures
or token sets.



3. The paper shows strong empirical correlations (R^2 being 0.97 ),
but: it does not test alternative decompositions — e.g., whether
other monotonic functions of cross-entropy could produce similar
scaling. It provides no ablation or causal test: does directly minimizing
error-entropy improve model scaling?  (They hint at this as “future
work.”)


4. Despite spanning “five orders of magnitude,” the experiments rely
mostly on open-weight transformer families (Pythia, Qwen, GPT-2,
LLaMA, etc.) trained on text-only datasets (Wikipedia, C4, GitHub;
p. 7 – 8). All datasets share similar linguistic distributions.  The sample size
per dataset is small (≈ 1 000 tokens per model; App. A.2 p. 12),
meaning statistical noise could bias scaling exponents. Hence, the “Error-Entropy Scaling Law” may not generalize beyond
their curated setup.



5. In Table 1 and 2 (p. 7 – 8), the differences in R^2 and |\delta| between
cross-entropy and error-entropy are numerically modest (e.g., 0.973 →
0.975).  Yet the paper interprets these small gains as evidence of a
fundamentally new law.  Given log–log regression’s sensitivity to
outliers and limited sample size, the improvement could easily arise
from noise or fitting artefacts.

### Questions
1. Error-Entropy is defined using rank statistics, which are inherently
non-differentiable. This raises an essential question:
Can Error-Entropy (or a differentiable surrogate) be used as a
practical training objective or evaluation metric?  If not, what is
its functional utility beyond post-hoc analysis? The paper hints (p.9) that Error-Entropy could inspire new loss
functions, but provides no method for computing gradients or
approximations.  Bridging this gap between theoretical definition and
practical optimization would determine whether this concept can
actually influence model development.



2. The paper’s key claim is that Error-Entropy—the Shannon entropy of
the rank-based error (RBE) distribution—is the true quantity that
scales.  However, is Error-Entropy revealing a genuinely new
theoretical property of learning dynamics, or is it simply a
re-expression of cross-entropy under a rank-based transformation? If Error-Entropy’s scaling behavior arises purely from the algebraic
decomposition of cross-entropy, it may not have independent
explanatory power.  Clarifying whether this metric corresponds to a
deeper principle (e.g., a conserved information quantity or
energy-like invariant) is crucial for establishing its theoretical
legitimacy.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the widely observed phenomenon that the cross-entropy scaling law, which predicts that model loss decreases as a power-law of model size, begins to break down at larger scales. The authors hypothesize that the root cause is that cross-entropy loss itself is not the quantity that truly scales. To test this, they introduce a novel and exact mathematical decomposition of the cross-entropy loss into three interpretable components: Error-Entropy, Self-Alignment, and Confidence. Through extensive experiments on 32 models spanning five orders of magnitude (up to 70B parameters), they find that only Error-Entropy, which measures the entropy of the rank of the correct token, exhibits a robust and predictable power-law scaling. The other two components remain largely invariant or noisy. The paper argues that the apparent scaling of cross-entropy in smaller models is an illusion caused by Error-Entropy being the dominant component. As models grow larger, its relative contribution diminishes, allowing the non-scaling components to disrupt the overall power-law trend. The authors propose the "Error-Entropy Scaling Law" as a more fundamental principle for understanding and predicting model performance.

### Strengths
1. The core idea of decomposing the cross-entropy loss is highly original. The three proposed components are not only mathematically sound but also intuitively interpretable. They map to distinct aspects of a model's learning process: getting the answer right (Error-Entropy), calibrating its output probabilities to its own error patterns (Self-Alignment), and expressing certainty by assigning low scores to incorrect tokens (Confidence). This provides a powerful new analytical tool.

2. The paper offers a simple and elegant explanation for a complex and important problem—the breakdown of cross-entropy scaling laws. The finding that only one component truly scales, and that its proportion changes with model size (Figure 8), compellingly explains why the law holds for smaller models but falters for larger ones.

3. The discovery of the Error-Entropy Scaling Law could have broad implications. It suggests a more reliable metric for extrapolating model performance and could inspire new training objectives focused directly on minimizing Error-Entropy, potentially leading to more efficient training or better-calibrated models.

### Weaknesses
1. The paper is motivated by the failure of scaling laws in the very large model regime, yet the analysis stops at 70B parameters. The most dramatic and puzzling scaling behaviors are often observed in models exceeding 100B parameters. Without data from these frontier models, the claim that the Error-Entropy law holds true at the absolute largest scales remains an extrapolation.

2. The analysis of how the components evolve during training (Figures 3-6) is based on the Pythia model family, with the largest model being 1B parameters. While insightful, it's unclear if these same dynamics (e.g., Error-Entropy being optimized first) hold true when training much larger models (e.g., 70B), where training dynamics can be qualitatively different.

3. The study exclusively analyzes existing pre-trained checkpoints. It does not explore how different training configurations (e.g., optimizers, data quality, learning rate schedules) might influence the behavior and proportions of the three components. It's possible that certain training strategies could alter the dynamics of Self-Alignment or Confidence, affecting the overall cross-entropy behavior.

### Questions
How many tokens are used to train the models in appendix A

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper decomposes cross-entropy into three components—Error Entropy (EE), Self-Alignment, and Confidence. It experimentally shows that the part that actually scales with model size is Error Entropy. Based on this, it proposes an explanation for why the cross-entropy scaling law fails at very large model sizes: as the model gets larger, the proportion of cross-entropy contributed by Error Entropy becomes smaller.

### Strengths
1. The overall structure is clear, the figures are appropriate, and readers can easily grasp the core ideas.
2. The mathematical derivations are clear and rigorous.
3. It offers a new perspective on the cross-entropy scaling law, which is thought-provoking.

### Weaknesses
1. Experimental results are not strong enough.
   The authors aim to use Error Entropy to explain the failure of the cross-entropy scaling law at large parameter sizes. However, possibly due to the experimental setup or model size, the experiments in the paper do not clearly demonstrate the failure of the scaling law. In this situation, relying only on EE having a slightly higher \(R^2\) than CE (and not even for all model x dataset combinations) to argue that EE is the part that actually scales is not very convincing.  
   If the authors could change the experimental setting (e.g. try different datasets, context lengths, or model sizes) to actually reproduce the failure of the CE scaling law, and at the same time show that EE still fits well, then the results would be much more persuasive.

2. Confusing phrasing. 
  While the overall argument is clear and the main findings are well laid out, a few parts are awkwardly expressed. For instance, in Section 3.3 on Self-Alignment, lines 296–303 already state clearly that Self-Alignment is about aligning \(p_e\) and \(q_e\). However, the two sentences that follow (lines 304–308) are hard to follow. - “Since different models have different error distributions, they also produce different probability scores.” — this seems to claim a causal link between error distributions and probability scores, but the paper does not substantiate that claim (correlation does not imply causation). - “If, by contrast, probabilities truly represented the single real language distribution as suggested by cross-entropy, we would not expect variation in probability scores across models, which deviates from empirical observations.” — this is also unclear, because a model’s probability distribution can vary for many ordinary reasons (initialization, training dynamics, local optima, etc.), so variation across models is not in itself evidence against cross-entropy. If I am misunderstanding the authors’ intent here, clarification would help — the current wording leaves too much room for interpretation.

### Questions
1. In Table 1, GPT-2’s CE \(R^2\) on the GitHub dataset is only 0.0717, which is far from the other results. Is this a typo?
2. Can you provide a possible explanation for why the share of EE in CE decreases as model size increases?
3. Why are the models used in the qualitative analysis different from those in the quantitative analysis? In other words, since the quantitative analysis already covers 30 models, why does the qualitative analysis only show 16?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an alternative rank-based metric, error-entropy, and argues that it has a more robust scaling law than CE. It shows that CE can be decomposed into three components: error-entropy, self-alignment, and confidence, and through experiments, authors demonstrate that only error-entropy scales consistently with model size, while the other components remain random. Since error-entropy becomes a smaller part of CE as model size increases, this can potentially explain why scaling laws fail for larger models.

### Strengths
1. Novelty: the rank-based metric, error-entropy, as well as the CE decomposition are novel as far as I know. The error entropy has nice properties, in particular robustness to post-processing such as temperature scaling. 

2. Extensive experiments: this paper includes many experiments, covering the training dynamics of smaller models and the scaling of the three components in larger models. There is a variety of models and datasets, including text and coding tasks. 

3. Clarity: the paper is well-written and the exposition is clear.

### Weaknesses
My main concern is the significance of the findings:
- Is there evidence that using EE, one can predict the EE of a larger model better than using CE to predict CE?
- Is EE as indicative of downstream performance as CE? 
- Regarding self-alignment and confidence: from section 3, self-alignment decreases during training and confidence increases, so I would expect self-alignment to be smaller and confidence to be larger as model size increases. However, self-alignment has a clear upward trend as models get larger, and confidence doesn't have a clear trend at all. Are these metrics meaningful in practice? Why would self-alignment increase for larger models that have more capacity for learning?
- In section 5, the paper argues that self-alignment and confidence might be the reason for the CE scaling law to break down at larger model sizes. However, in the scaling plot, it's not clear that there's a CE scaling breakdown and EE has more robust scaling. Can the authors provide evidence that EE doesn't suffer from the same issue?

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
