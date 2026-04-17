# Geometric and Information Compression of Representations in Deep Learning

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Deep neural networks transform input data into latent representations that support a
wide range of downstream tasks. These representations can be characterized along
information-theoretic and geometric dimensions, but their relationship remains
poorly understood. A central open question is whether low mutual information (MI)
between inputs and representations necessarily implies geometrically compressed
latent spaces and vice versa. We investigate this question using neural collapse
as a measure of geometric compression and theoretically sound MI estimation in
conditional entropy bottleneck (CEB) networks and continuous dropout networks.
We evaluate the interplay between MI, geometric compression, and generalization
on classification tasks under controlled noise injection schemes. Our findings show
that low MI does not reliably correspond to geometric compression, and that the
connection between the two is more nuanced than often assumed. We conjecture
that generalization acts as a potential confounder in this connection rather than
being a direct consequence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work disputes the connection established in previous literature (e.g., by Goldfeld et al., 2019) between the geometric compression of latent representations $Z$ and mutual information $I(X;Z)$, where $X$ are the inputs of a neural network. The authors argue that a decrease in $I(X;Z)$ does not imply a more clustered $Z$; on the contrary, their experimental results suggest that compression correlates with higher mutual information.

The authors also provide a supplementary theoretical result that justifies the Information Bottleneck (IB) analysis of DNNs with analytic activation functions and continuous dropout.

### Strengths
The paper's key strengths are its main theoretical result (Theorem 3.1) and the sheer scale of its experimental evaluation. While Theorem 3.1 is an extension of the result from Adilova et al. (2023), I consider it to be important for proving the non-vacuousness of IB analysis for a wider class of neural networks. Furthermore, the experimental results are quite insightful, highlighting the intricate interplay between $I(X;Z∣Y)$, Neural Collapse, various hyperparameters ($\beta$ in CEB and $\lambda$ in the Gaussian dropout framework) and accuracy+generalization.

### Weaknesses
I have two major concerns in regard to the methodology:
1. **How MI is measured**.
    - The authors are rather inconsistent with their choice of MI estimators. For Figure 1, they use NPEET (which is, by the way, not referenced in the main text); for CEB, the variational bound is employed; for Gaussian dropout (GD), they use DoE estimator.
    - The paper lacks concrete justification for the choice of estimators. Specifically, the use of NPEET is unexplained, the claim of a "practically tight" variational bound (line 248) is unsubstantiated, and the superior performance of DoE over other SOTA estimators (lines 250-254) is not demonstrated. I kindly ask the authors to elaborate on these decisions and provide the experimental results that support their claims.
2. **When MI is measured.** As stated in lines 128-130, this study focuses on the connection between MI and clustering at the end of training. While this is an interesting direction, I find it only loosely connected to the original works on the IB principle.

    For instance, Shwartz-Ziv & Tishby (2017) measure MI throughout the training. They identified a distinct compression *phase*, where $I(X;Z)$ begins to decrease after a certain epoch. Goldfeld et al. (2019) later connected this *phase* to geometric compression (under normal conditions). Therefore, an anti-correlation between MI and NC observed only at the end of training does not preclude the occurrence of such a compression phase, nor does it rule out geometric compression as its driver (for example, geometric compression can drive MI to the minimal value *throughout the training*, but the minimum itself can still be anti-correlated with NC).

   Moreover, recent studies suggest that compression phases can be transient and may not result in ultimate MI compression (e.g., $I(X;Z)$ might exhibit an overall steady growth punctuated by rapid drops correlated with improvements in training loss). Please, refer to Figure 5 in [1] or non-ReLU IB experiments in [2].

I also encourage the authors to include the full proof of Theorem 3.1, since there is no limit on the length of the Appendix.

[1] Butakov et al. "Information Bottleneck Analysis of Deep Neural Networks via Lossy Compression". Proc. of ICLR 2024.

[2] Anonymous Authors. "A Generalized Information Bottleneck Theory of Deep Learning". ICLR 2025 submission: [https://openreview.net/forum?id=reOA4r0FGL](https://openreview.net/forum?id=reOA4r0FGL).

**Minor issues:**

1. In lines 167-170, the joint distribution of $X$ and $Y$ is said to be "typically continuous", while $Y$ is clearly discrete since the task is classification.
2. The $\parallel$ symbol in Tables 1-2 is not visually appealing due to misalignment. The missing values are also a bit confusing. I understand that they suppose to mean that "gen" requires evaluation on both train and tests subsets. Perhaps, a viable option is placing it in-between the columns using `\makecell` or a similar macro. Finally, it is not immediately obvious what "Perf." stands for. Overall, I suggest an overhaul of these tables.
3. Perhaps, Figure 2 might benefit from log-scaling the `y` axis.
4. The equivalence in line 406 requires additional explanation. As I understand,
   $$
   I(X;Z \mid Y) = I(X;Z) - I(X;Y;Z) = I(X;Z) - I(Z;Y) + \underbrace{I(Z;Y \mid X)}_0,
   $$
   where $I(Z;Y \mid X) = 0$ since $Y \to X \to Z$ is a Markov chain. Please, elaborate on this in the main text.
5. In line 716, a backslash before `log` is missing. I also kindly suggest using `\text` for `dist`, `vol` and `finite` in the subsequent derivations.

**Conclusion:**

Overall, the paper appears rather unpolished. The methodology also needs stronger justification. For these reasons, I recommend a major revision.

### Questions
1. Why did you use NPEET instead of DoE for Figure 1?
2. The original implementation of the DoE estimator relies on rather weak approximations of the distributions (e.g., Gaussian). Are they good enough for your complex task?
3. Do you have any intuition behind the anti-correlation between MI and NC? For me, a positive correlation is quite intuitive (clustered representations are "degenerate" and typically encode less information), but I still can not explain the opposite behavior that you observe.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses an open question in representation learning: Does low mutual information (MI) between inputs and learned representations imply geometric compression of those representations, and vice versa? The authors probe this through experiments on classification networks with continuous dropout (injecting noise) and with the Conditional Entropy Bottleneck (CEB) objective. They also attempted to examine the role of generalization

### Strengths
1) Theoretically sound MI estimation

2) The authors present evidence that one can observe low mutual information without strong within class variation collapse, and variation collapse can occur even when mutual information remains high (as was known for deterministic networks).

3) they also measured that the relationship between generalization and compression is not causal.

### Weaknesses
While the experimental design is solid and the question is important, the theoretical framing is not as rigorous. 

1) The paper repeatedly refers to “Neural Collapse,” but only measures NC1 (within-class variance). The co-occurrence of NC1-NC2 are critical for a geometry to be called neural collapse (Thm 1 and 2 in papyan 2020). Only NC1 can include degenerative solutions. 

2) Neural collapse also refers to when training accuracy goes to 100% (or plateau nearby 100%), did you observe that in your experiments? If not (low beta in the CEB objective, which may lead to compressing away even classification relevant information), it is hard to even say your model attained neural collapse.

### Questions
1) Would you please clarify your definition of compression: whether it’s informative compression (e.g., late-phase IB) or trivial compression (e.g., untrained/noisy encoding)? 

2) Is it possible to control for test accuracy to demonstrate that generalization is a confounder of compression and low MI between input and latent representations.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper attempts to better elucidate the relationship between geometric quantities, such as the neural collapse of a neural network, and the information compression that network is capable of. In doing so this paper aims to compare between conditional entropy bottleneck models and models trained with "continuous dropout", and track to what degree neural collapse happens at the same times as information compression.  They find that there are several differences between these different metrics but there is typically a negative relationship between information compression and neural collapse.

### Strengths
The proof presented here is to my knowledge novel, and the proof is mathematically interesting and nontrivial. I think that the use of dropout as a way to introduce Stochasticity to allow for the information bottleneck theory to become sensible is also interesting.

The paper also contains a detailed and clear related works section, which can help in reading the literature in this area.

### Weaknesses
1. The primary weakness with this paper is that it is not clear, from the empirical results provided, what the takeaway is. Is the takeaway intended to be that neural collapse and information compression are not very strongly or obviously related, as Fig. 3 seems to display? In that case is the purpose of the paper to display a null result (which I think is not an issue, but it should be stated as such)?

### Questions
1. From Figure 1, what are we supposed to take from this? My impression is that the neural collapse is somehow orthogonal to the mutual information, is this the right way to interpret this?
2. Are there other ways to measure the mutual information that could provide with more stable estimates for the continuous dropout model?
3. Can you provide some more information as to what the data in Table 1 and 2 are coming from? Are these the combinations of the four considered setups here? Are there differences between them?

### Soundness
2

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
3

### Summary
This work examines whether information-theoretic compression, measured by the mutual information I(X; Z), implies geometric compression, quantified by Neural Collapse (NC). The authors find that the relationship is not reliable. Their theoretical and experimental results show that a decrease in mutual information does not necessarily lead to a more collapsed geometric structure.

### Strengths
This work establishes the finiteness of mutual information in dropout networks employing analytic activation functions. It presents experiments across different architectures and datasets. The identification of generalization as a potential confounder in the relationship between compression and generalization. The theoretical toy model in Appendix illustrates why mutual information and neural collapse measures can diverge in practice.

### Weaknesses
1) The paper relies heavily on the accuracy of MI estimates, yet the justification for the chosen methods is somewhat brief. For CEB, the claim that the variational bound is "practically tight" is asserted but not thoroughly validated. The gap $\mathbb{E} [D_{KL}(p_{Z \mid(|) Y} \mid(|) q_{Z \mid(|) Y})]$ is assumed to be small due to co-training (line 247), but no evidence is provided to quantify this gap.

2) While the DoE estimator might be reasonable choice in some situations, its sensitivity and potential biases in the high-dimensional regimes of state-of-the-art models are not deeply discussed or ablated as far as I know. Thus, it would be more convincing to compare results against a wider suite of MI estimators (line 252).

3) I agree that I(X; Z \mid(|) Y) = I(X; Z) - I(Z; Y), but the claimed I(X; Z \mid(|) Y) \approx I(X; Z) in line 406 should be justified more rigorously.

4) By "geometric compression" the paper means the NC metric. While this is a well-established measure for class-separation geometry and used recently (lines 39-40), it is not the only possible measure. I think the work would be more insightful if it were expanded to include other geometric measures, such as intrinsic dimension, which is mentioned in passing (line 37), or others.

### Questions
To further validate the findings, it would be helpful to see results with other MI estimators and to also compute other geometric measures (e.g., intrinsic dimension). This would test whether the correlation with MI holds for geometric properties beyond Neural Collapse.

### Soundness
3

### Presentation
3

### Contribution
2
