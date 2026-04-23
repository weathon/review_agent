Now I have enough information to write the final review. Let me compile everything.

## Summary

The paper proposes two novel loss functions for test-time adaptation (TTA) with pseudo-labels: sparse cross-logit (sparse-CL), which operates directly in logit space and yields a constant L1 gradient norm with respect to the logits, and k-hardness negative learning (k-NL), which penalizes k complementary labels to enlarge decision boundaries. The central thesis is that logit-space losses enable stable training with high learning rates, allowing the model to adapt "quickly" and thereby mitigating memorization of noisy pseudo-labels and confirmation bias.

## Strengths

- **Substantial and consistent empirical improvements over SAR**: Across three settings (normal, imbalance, batch size 1), the proposed method (SAR* + L_final) improves over SAR by 5.2%–8.1% on ImageNet-C severity level 5 (Tables 1–3). Notably, sparse-CL resolves collapse cases where SAR fails entirely (e.g., Snow: 53.6%→66.1%, Frost: 46.2%→64.4% in Table 1).

- **Compelling demonstration of the core mechanism (Figure 2)**: The figure directly shows that scaling the learning rate in SAR causes accuracy to crash from ~55% to ~30% after batch 10, while the proposed method steadily reaches ~57%. This is the key empirical validation that logit-space losses enable stable high-LR training.

- **Valuable insight about logit-space gradient behavior**: The observation that operating in logit space can decouple gradient magnitude from prediction confidence is genuinely useful. In cross-entropy, the gradient for the pseudo-class vanishes as confidence increases (Eq. 6: gradient norm = 2(1-p_k)), while sparse-CL maintains a constant gradient (Eq. 11), ensuring continued learning even on confident pseudo-labels.

- **Simple, plug-and-play design**: The method is essentially a loss function replacement in the SAR framework, requiring no architectural changes. Algorithm 1 provides clear pseudocode, making reproduction straightforward.

- **Figure 3 provides empirical gradient stability evidence**: The six subplots show dramatically reduced gradient norm fluctuation for the proposed method compared to entropy minimization and cross-entropy during actual training, empirically supporting the stability claim.

## Weaknesses

### Fatal
None.

### Major

- **Invalid mathematical step in deriving sparse-CL from cross-entropy (Equation 8)**: The paper motivates sparse-CL by replacing $p_i = \frac{\exp(h_i)}{\sum_j \exp(h_j)}$ with $p_i \approx \exp(h_i)$ (Eq. 8), dropping the partition function. This is not a valid approximation during normal training — it only approximately holds when all logits are very negative (so $\sum_j \exp(h_j) \approx 1$), which is precisely the regime that does not hold in practice. The paper presents this as a principled derivation ("motivated by cross-entropy"), but the resulting loss $-\sum_i \hat{y}_i h_i$ is not a legitimate surrogate derived from cross-entropy through valid approximation. This matters because the paper frames its contribution as theoretically motivated; the loss could instead be cleanly motivated as simply maximizing the logit of the pseudo-class (a margin-maximizing objective), which is a valid and honest justification.

- **Imprecise "zero gradient variance" claim conflates logit-space and parameter-space**: The paper states (Section 3.2): "the variance of the gradient norm respect to $h$ when learning with sparse-CL is equal to 0" and concludes "this loss will yield a smaller gradient variance during updating." While the logit-space gradient norm is indeed constant (=1), the actual parameter-space gradient is $\nabla_\theta \mathcal{L} = (\nabla_h \mathcal{L})^T (\partial h / \partial \theta)$, and $\partial h / \partial \theta$ varies substantially across samples. The total parameter-gradient variance is therefore not zero. Having $\nabla_h \mathcal{L}$ constant does reduce variance compared to losses where both factors vary, which is a fair argument, but the paper's claim as stated ("zero gradient variance") overclaims and conflates two different quantities. This matters because the entire logical chain (stable learning → high LR → fast adaptation → reduced memorization/confirmation bias) rests on this stability argument.

- **Missing learning rate ablation despite LR being the paper's central mechanism**: The paper's core thesis is that sparse-CL enables high learning rates. While Figure 2 shows one comparison (SAR at default LR, SAR at scaled LR, and the proposed method at scaled LR), there is no systematic learning rate sweep comparing sparse-CL vs. entropy minimization vs. cross-entropy across multiple LR values. Without this, it is difficult to assess how much improvement comes from the loss function itself versus simply using a larger LR. This is the single most important missing experiment given the paper's framing.

### Minor

- **No direct experimental verification of reduced memorization/confirmation bias**: The paper motivates sparse-CL through memorization effect and confirmation bias (Section 1), but never measures whether the method actually reduces noisy-label memorization or error accumulation over time. Tracking pseudo-label accuracy across adaptation steps would directly test these hypotheses and significantly strengthen the paper.

- **Limited experimental benchmark diversity**: All experiments are on ImageNet-C at severity level 5 with a single ViT architecture. While ImageNet-C is the standard TTA benchmark, additional datasets (CIFAR-10-C, CIFAR-100-C) or architectures would strengthen generalizability claims. The abstract claims "diverse set of TTA experiments," but the actual scope is narrow.

- **Learning rate values not reported in main text**: Given that the learning rate is the paper's central mechanism, not reporting its specific value (nor those of $k$, $s$, $\alpha$) in the main text is a notable omission. These values are presumably in the appendix.

### Trivial
None.

## Nice-to-Haves

- Reframe sparse-CL as a margin-maximizing objective on the pseudo-class logit rather than deriving it through the invalid $p_i \approx \exp(h_i)$ approximation. This would provide an honest and arguably more insightful theoretical motivation.
- A systematic LR sweep comparing sparse-CL, EM, and CE across multiple learning rates to quantify the loss-vs-LR contribution.
- Testing sparse-CL as a drop-in replacement in other TTA frameworks (EATA, CoTTA) beyond SAR to demonstrate generalizability.
- Adding standard deviations across runs, as TTA methods can be sensitive to data ordering.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Critic: "unbounded noise ratio obstacle is mentioned but never addressed"** — The paper hypothesizes in Section 1 that "quick" adaptation addresses all four challenges including unbounded noise ratio (line 40: "the generated pseudo-label procedure becomes more accurate with a lower noise ratio... as labels are produced at each inference step, benefiting from improved convergence speed"). While the connection is indirect, the paper does not simply drop the obstacle.

- **Critic: "cross-entropy discarding pseudo-classes is standard and desirable"** — The paper argues this is undesirable under pseudo-label learning where even confident predictions should continue being reinforced. This is a reasonable design choice for TTA (where models degrade over time and need continued adaptation), not an obvious error.

- **Critic: "no analysis or ablation of $s$"** — This is essentially a hyperparameter sensitivity concern, which falls under minor reproducibility nitpick. The skip mechanism is well-motivated by the need to avoid noisy negatives from nearby classes.

- **Critic: "SAR† vs SAR* not clearly defined"** — The notation is used in the tables; SAR† appears to be the original SAR, SAR* is SAR with the proposed loss replacing entropy minimization. While more explicit definitions would help, this is a presentation nitpick.

- **Critic: "DDA results are notably poor; unclear if DDA was given a fair evaluation"** — The paper uses standard benchmark results from prior work. This is not a valid criticism of the current paper.

- **Strength Finder: "Figure 4 shows clearer separation between true and noisy pseudo-labels"** — While interesting, this visualization-based claim is informal and not directly validated quantitatively. Kept as supporting evidence but not elevated to a core strength.

- **Critic: "Equation 2 has an apparent sign error"** — The critic themselves note this may be a parser issue. Removed.

## Novel Insights

The paper identifies an underappreciated connection between logit-space loss functions and gradient stability in TTA: by bypassing the softmax normalization, the gradient with respect to the logits becomes independent of the prediction confidence, which prevents the "gradient vanishing on confident predictions" problem inherent in cross-entropy. While the paper's specific derivation via $p_i \approx \exp(h_i)$ is invalid, the underlying insight — that the softmax's partition function is the source of confidence-dependent gradient scaling — is correct and valuable. Reframing sparse-CL as directly optimizing the logit margin rather than the softmax probability would be a more honest and potentially more generalizable theoretical contribution.

## Suggestions

- Replace the invalid $p_i \approx \exp(h_i)$ derivation with a direct motivation: "We propose to maximize the logit of the pseudo-class directly, which yields a constant gradient and is equivalent to a margin-maximizing objective in logit space." This is honest, clean, and arguably more insightful than the current framing.
- Add a learning rate sweep (e.g., $\eta \in \{0.001, 0.005, 0.01, 0.05, 0.1\}$) comparing sparse-CL, EM, and CE to isolate the effect of the loss function from the effect of LR scaling.
- Track and report pseudo-label accuracy over adaptation steps to directly verify the claimed reduction in memorization and confirmation bias.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| "Entropy is not Enough" (DeYO) | 9w3iw8wDuE | 7.0 | Similar topic (critiquing entropy-based TTA), but much stronger theoretical grounding and broader experiments. Our paper is clearly weaker. |
| AEA for TTA | sEMJ1PLSZR | 6.25 | Similar contribution type (new loss replacing EM for TTA). Solid theory, moderate improvements. Our paper has larger empirical gains but weaker theory. |
| PASLE for TTA | 3Z2flzXzBY | 6.40 | TTA with selective label enhancement. Accepted with moderate contributions. |
| DART for TTA | xqxG5WogN6 | 5.67 | TTA pseudo-label refinement. Weaker empirical improvements, rejected. Our paper has stronger empirical results. |
| MDAA for MM-CTTA | UhKkWHkvfg | 5.0 | Multi-modal TTA. Decent results, various weaknesses, rejected. Our paper has stronger results. |
| Active Test-Time Prompt | pdzHpQbGrn | 2.50 | Trivial engineering extension. Our paper is clearly above this. |
| Dynamic Smoothing | 85Eej2kUHQ | 2.33 | Incorrect theorem. Our paper's theoretical issues are presentation flaws, not fundamentally incorrect methods. |

Our paper's empirical contributions (5–8% improvement) exceed those of the medium-scoring rejected TTA papers (DART at 5.67, MDAA at 5.0), and the core insight about logit-space gradient stability is genuinely valuable. However, the theoretical presentation has two significant flaws (invalid derivation, imprecise variance claim) that undermine the paper's stated theoretical contributions. Compared to AEA (6.25, accepted) which also proposes a new TTA loss function, our paper has stronger empirical results but weaker theoretical justification. The lack of LR ablation is a notable gap given the paper's central mechanism.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>