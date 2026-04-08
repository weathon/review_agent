## Human Reviewer 1

### Summary
The paper proposes a bootstrapping framework for LLM unlearning that tackles the “squeezing effect,” where probability mass shifts to semantically similar outputs instead of true forgetting. Two variants are introduced: BS-T, which suppresses high-likelihood tokens, and BS-S, which augments the forget set with high-confidence generations. The authors provide theoretical analysis and experiments across TOFU, MUSE, and WMDP, showing improved balance between forgetting and retention.

### Strengths
1. The paper is very readable, with a logical flow from motivation → analysis → method → theory → experiments. Figures and appendices are well-organized, and pseudocode makes the algorithms easy to reproduce.
2. The authors make a thoughtful observation about the squeezing effect and systematically demonstrate its existence through both qualitative and quantitative analysis. The proposed bootstrapping strategy is a creative extension of this insight, and the experiments convincingly show that BS-T and BS-S outperform existing unlearning baselines under various settings.
3. The work is conceptually motivated by an intuitive yet underexplored idea—connecting unlearning failures with the model’s own belief distribution. This is a fresh perspective on unlearning that moves beyond purely loss-based formulations, and the motivation is clearly justified both intuitively and empirically.

### Weaknesses
1. While the paper focuses on redistributing likelihood as the core cause of spurious unlearning, the explanation still feels surface-level from a semantic standpoint. The essence of the problem may not lie solely in likelihood shifts, but rather in the fact that current unlearning methods attempt to correct predictions without accounting for semantic relatedness. Unlearning should arguably target semantic classes of knowledge, rather than isolated outputs or sequences. A more principled formulation in semantic embedding space (e.g., clustering or alignment-based unlearning) might provide a deeper understanding of what “forgetting” really entails.
2. Although BS-T and BS-S generally achieve the best average scores, in several tasks their performance margins over strong baselines like NPO or RMU are modest. The results could be strengthened with additional analysis.

### Questions
1. In BS-S, high-confidence generations are added to the forget set, but such sequences may still contain unrelated or benign information.
How does the method ensure that these “bootstrapped” samples do not lead to accidental forgetting of non-target knowledge?
Is there any filtering mechanism beyond temperature-controlled decoding?
2. The theoretical part (Section 4.2) discusses the dynamics of probability redistribution under BS-T versus GA, but it would be very valuable to show empirical probability dynamics for BS-S as well—similar to Figures 2(b)–(c) for GA and NPO.
This would help demonstrate whether BS-S effectively flattens or redistributes probability mass in the way the theory predicts.

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
2

---

## Human Reviewer 2

### Summary
This paper identifies a critical failure mode in previous gradient-based LLM unlearning methods (like Gradient Ascent and NPO), terming it the "squeezing effect". The authors demonstrate that these methods, while successfully reducing the probability of the exact target response, could redistribute this probability mass onto semantically related rephrasings. This leads to "spurious unlearning", where the sensitive knowledge persists, a failure often masked by standard automated metrics like ROUGE.  To address this, the paper proposes a bootstrapping (BS) framework that explicitly targets the model's own high-confidence generations (its "model beliefs") as additional unlearning signals. In practice, the method utilizes token-level BS and sequence-level BS.

### Strengths
1. The identification and mechanistic analysis of the "squeezing effect" is a novel and significant contribution. It provides a clear diagnosis for a subtle but critical flaw in widely-used unlearning methods. This finding is highly significant for the field, as it suggests many existing methods may offer a false sense of security regarding privacy and safety.

2. The core claim of the "squeezing effect" is not just asserted but convincingly demonstrated through empirical analysis of probability dynamics (Fig. 2), tracking how probability mass shifts from the target to high-likelihood alternatives. The paper also provides a theoretical analysis for BS-T (Thm. 4.2) within the learning dynamics framework, explaining why suppressing the model's beliefs helps reshape the gradient to mitigate squeezing.

3. The empirical evaluation is comprehensive, covering three diverse benchmarks (TOFU, MUSE, WMDP), multiple model families (LLaMA-2, LLaMA-3, Zephyr), and various model scales (1B, 3B, 7B, 8B), demonstrating the robustness of the findings.

### Weaknesses
1. One small weakness is the practical cost of BS-S. Algorithm 2 implies sampling $N$ high-confidence sequences for every sample in a batch during training. This requires $N$ inference passes for each training step, which seems computationally prohibitive and scales poorly. Figure 6 shows BS-S is ~2x slower than NPO, and it might be even worse as $N$ grows. The paper also notes OOM issues when set $N=5$. It would be better if adding an ablation on the frequency of belief sampling (e.g., once per epoch vs. once per batch).

2. As noted in the ablations (Fig. 5) and limitations (Sec. G), the methods are sensitive to the bootstrapping coefficients ($\lambda_{BST}$, $\lambda_{BSS}$). Performance appears to drop off significantly if these are not tuned correctly. This could be a major barrier to adoption, as it may require expensive, model- and dataset-specific tuning. The paper would be stronger if it provided more intuition or heuristics for setting these crucial parameters.

3. The theoretical analysis is a key strength for BS-T, but it is missing for BS-S. As BS-S is the more complex and often better-performing method, it would be better to add a theoretical standpoint why sampling $N$ full sequences is superior to the more efficient token-level suppression of BS-T.

### Questions
1.  In BS-S, what is the nature of the $N$ sampled sequences? Are they $N$ semantically distinct paraphrases, or are they minor lexical variations of the same core "belief"? If the diversity is low, would a smaller $N$ (e.g., $N=1$ or $N=2$) enough, thereby mitigating the cost?

2. The main experiments (e.g., Table 1) appear to combine the BS methods with retention regularization (i.e., using $\mathcal{D}_r$). How much of the utility preservation is attributable to the BS method itself versus this external regularization? What does the performance of "pure" BS-T/BS-S (without $\mathcal{D}_r$) look like compared to "pure" NPO? This would help isolate the true impact of your method on the forget-retain balance.

3. In Line 302 to 303, GA will increase mass on $H_k^{(i)}$, but according to Figure 2 (b), it is not very obvious that GA shifts the probability mass to high-likelihood regions. Would it be better to clarify more on this?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper clearly proposed and defined the "Squeezing Effect" for the first time: the existing gradient ascent forgetting method only reduces the probability of the target response, causing the probability mass to be "squeezed" into a semantically similar high-confidence area, thereby causing spurious unlearning. To this end, the Bootstrapping framework proposed in this paper uses the model beliefs to guide the forgetting process. Extensive experiments on multiple benchmarks confirm the effectiveness of this approach.

### Strengths
- The paper is easy to follow.
- The content of the paper is substantial, with both a summary of existing work and sufficient theoretical evidence.

### Weaknesses
- The paper acknowledges in Appendix G that this method is very sensitive to the settings of hyperparameters such as the bootstrapping coefficient, and often requires extensive tuning for specific datasets and models. This seriously affects the method's application in practical scenarios.
- Lack of comparison of computational overhead between various baseline methods.
- The Bootstrapping framework relies on the high-confidence results generated by the model itself to guide forgetting. However, the model's confidence often does not necessarily reflect the required forgetting content correctly. For example, the model's confidence may be high, but its answer may be wrong, or it may have low confidence but the answer may be correct.
- There is a lack of experiments on more diverse model structures to prove the effectiveness of the proposed method.

### Questions
Since I am completely unfamiliar with this area, I will adjust my score based on the suggestions of other reviewers.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
2

---

## Human Reviewer 4

### Summary
This paper investigates the "spurious unlearning" problem that occurs when LLMs perform unlearning tasks. The authors point out that while existing methods (such as gradient ascent and NPO) can reduce the probability of a target response, they also redistribute the probability mass toward semantically similar regions, resulting in a "squeezing effect."
To address this issue, the authors propose a Bootstrapping (BS) framework that incorporates the model's own high-confidence generation (model belief) into the forgetting objective.
BS-T suppresses high-probability tokens at the word level;
BS-S suppresses high-confidence generation of entire segments at the sequence level.
The authors validate their approach on benchmarks such as TOFU, MUSE, and WMDP, and provide a theoretical analysis of BS-T to explain its mechanism for mitigating the squeezing effect.

### Strengths
- Interesting and Important Discovery: The paper reveals the "squeezing effect" and the resulting "spurious unlearning", suggesting that current unlearning methods only achieve superficial unlearning and further analyzing the reasons.

- Simple Yet Effective Design: The Bootstrapping framework directly targets areas of high probability of false forgetting by jointly suppressing the target response and the model's own high-confidence output. It requires no additional models or external data and is logically self-consistent.

- Comprehensive Experimental Results: Systematic experiments on multiple datasets and models, compared with strong baselines such as NPO, WGA, and RMU, show consistent improvement. Qualitative examples and gradient dynamics analysis are also provided to further strengthen the demonstration.

### Weaknesses
- BS-S has high computational overhead: Sequence-level bootstrapping requires generating multiple belief sequences for each sample, significantly increasing computational costs. The paper does not provide clear time or resource costs, nor does it discuss scalability for large-scale applications.

- BS-S lacks theoretical support: The authors provide a gradient analysis for BS-T, but BS-S is validated solely by experimental results and lacks formal explanations or convergence guarantees.

- High-confidence suppression strategies carry risks: High confidence does not necessarily indicate content that should be forgotten. The top-k outputs of the model may contain semantically related but harmless tokens; ranking by probability alone may lead to excessive forgetting.

### Questions
This paper identifies and addresses flaws in existing LLM unlearning methods, using a sound approach and robust results. 

To further enhance the paper's contributions, I have the following comments:

1. Quantitatively demonstrate the effectiveness of mitigating the "squeezing effect," such as the change in semantic similarity between generated samples before and after forgetting;

2. Report the computational overhead of BS-T and BS-S, and discuss their applicability in large-scale scenarios.

3. Supplement sensitivity and ablation analysis of the hyperparameters λ, k, and N;

4. Clarify the model belief sampling strategy and evaluate the impact of different parameter settings on the results;

5. Explore dynamic k or entropy-based adaptive strategies to mitigate the over- or under-forgetting issues associated with a fixed k.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3