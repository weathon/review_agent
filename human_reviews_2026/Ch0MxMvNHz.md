# Multiple Token Divergence: Measuring and Steering In-Context Computation Density

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Measuring the in-context computational effort of language models is a key challenge, as metrics like next-token loss fail to capture reasoning complexity. Prior methods based on latent state compressibility can be invasive and unstable. We propose Multiple Token Divergence (MTD), a simple measure of computational effort defined as the KL divergence between a model's full output distribution and that of a shallow, auxiliary prediction head. MTD can be computed directly from pre-trained models with multiple prediction heads, requiring no additional training. Building on this, we introduce Divergence Steering, a novel decoding method to control the computational character of generated text. We empirically show that MTD is more effective than prior methods at distinguishing complex tasks from simple ones. On mathematical reasoning benchmarks, MTD correlates positively with problem difficulty. Lower MTD is associated with more accurate reasoning. MTD provides a practical, lightweight tool for analyzing and steering the computational dynamics of language models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Multiple Token Divergence (MTD), a novel and lightweight metric for measuring the in-context computational effort of language models. MTD is defined as the KL divergence between the output distribution of the full model and that of a shallow, auxiliary prediction head. The core intuition is that a large divergence implies the full model is performing complex computations that cannot be approximated by a simple shortcut. The authors demonstrate that MTD can be computed from existing pre-trained models with Multiple Token Prediction (MTP) heads without requiring any new training. They also propose Divergence Steering, a decoding method that uses MTD to control the "computational character" of generated text by interpolating between the full and shallow output distributions. Empirical results show MTD is a more effective measure of task complexity than prior methods like Prediction of Hidden States (PHi). On mathematical reasoning, MTD correlates positively with problem difficulty, and lower MTD values are associated with more accurate reasoning.

### Strengths
**Simplicity and Practicality:** The primary strength of MTD is its simplicity compared to prior methods like PHi. It avoids invasive architectural changes, complex loss functions, and unstable training. The ability to compute it post-hoc on models already equipped with MTP heads (like MiMo-7B) makes it a highly practical tool for analysis.

**Novel Decoding Method:** Divergence Steering is a genuinely new mechanism for controlling generation. It introduces a steering parameter, α, that is conceptually orthogonal to temperature—controlling "computational character" rather than just entropy. The use of geodesic interpolation for this process is principled and elegant.

**Strong Empirical Analysis:** The paper provides a robust empirical validation of MTD as a metric. The head-to-head comparison with PHi on synthetic tasks clearly demonstrates that MTD (with latest embedding access) provides a cleaner signal of computational complexity . The findings on the MATH dataset—that MTD correlates positively with difficulty while NLL correlates negatively—are insightful and effectively decouple the concepts of model surprise and computational effort.

**Isolating Computational Effort:** The paper makes a key contribution by showing that allowing the auxiliary head access to the latest token embedding enables the MTD metric to specifically isolate information gain attributable to non-trivial computation, rather than just novel information in the current token.

### Weaknesses
**Limited Impact of Steering on Reasoning:** The most significant weakness is the acknowledged failure of Divergence Steering to improve performance on the core reasoning tasks analyzed. While the method shows interesting, task-dependent effects on toy creative problems, its inability to enhance mathematical reasoning in pre-trained models severely limits the practical impact of the paper's second main contribution.

**Ambiguous Interpretation of the MTD Signal:** The paper finds that lower MTD is associated with correct reasoning on MATH and GSM-8k . This is in direct contrast to prior work where higher PHi loss was linked to correctness. The authors' hypothesis that this is model-dependent is plausible but leaves the interpretation of the MTD signal ambiguous. It is unclear whether high MTD should be interpreted as "deep, productive thinking" or "inefficient, confused thrashing." This ambiguity complicates its use as a straightforward optimization target.

**Conditional "No-Training" Claim:** The claim that MTD requires no additional training is conditional on using a model that already has an MTP head. Most widely-used open-source models do not. The proposed workaround—training a head via distillation —reintroduces the training complexity and cost that MTD is positioned as an alternative to, weakening one of its main selling points.

**Post-Hoc Steering Direction:** In the creative tasks, the optimal steering direction (positive vs. negative α) depends on whether the task is "discovery" or "construction". This is an interesting finding, but the paper offers no method to determine the correct direction for a new, arbitrary task a priori, making the steering method difficult to apply in practice.

### Questions
1. The failure of Divergence Steering on reasoning tasks is a critical point. Could you elaborate on your hypothesis for why this occurs? Does it suggest that post-training (e.g., SFT, RLHF) makes a model's generation process too brittle for this kind of decoding intervention, or is MTD fundamentally measuring a property that is not directly aligned with generating a correct reasoning path?

2. Your finding that lower MTD correlates with correctness is intriguing and contrasts with prior PHi results. Beyond model-dependency, could it be that MTD is capturing a different aspect of computation, such as "processing efficiency" or "convergence on a solution," where lower divergence indicates a more confident, streamlined path?

3. For models without a pre-existing MTP head, what is the practical cost (in data and compute) of training a new head via distillation? How does this compare to the cost of implementing and training a full PHi model, and does it maintain a significant advantage in terms of simplicity and stability?

4. Given that the optimal steering direction (α) is task-dependent, how might a practitioner apply Divergence Steering to a novel task where this direction is unknown? Have you explored any methods for dynamically adapting α during generation?

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
This paper introduces Multiple Token Divergence (MTD), a metric designed to measure the in-context computational effort of a language model at each token. The authors propose that standard metrics like next-token loss fail to capture the complexity of the reasoning required to produce a token. Their method addresses the shortcomings of prior work (like the PHi loss).

MTD is defined as the KL divergence between the full model's next-token output distribution and the distribution of a shallow, auxiliary prediction head (an MTP module). The core intuition is that if a "shortcut" shallow model can easily predict the same token as the full, deep model, the full model isn't performing complex computation. A large divergence, however, implies the deeper layers are performing non-trivial work. A key refinement is to provide the shallow MTP module with access to the current token embedding, which helps MTD isolate computational effort from the simple information gain of the new token.

The authors also introduce Divergence Steering, a decoding technique that uses the MTD signal to control the "computational character" of the generated text by interpolating between the full and shallow models' distributions. This can bias generation to be more "anti-speculative" (favoring computationally intensive tokens).

### Strengths
- The paper's most fascinating result (sec 4.2) is the decoupling of computational effort (MTD) from predictive plausibility (NLL). The finding that MTD correlates positively with MATH problem difficulty while NLL correlates negatively is a significant contribution.
- The discovery that MTD and NLL are anti-correlated with respect to problem difficulty is very interesting. It provides a new, orthogonal axis for analyzing model behavior. NLL measures "plausibility" or "surprise," while MTD measures "effort". 
- The use of geodesic interpolation (via the Fisher-Rao metric) to navigate between the two output distributions ($\pi$ and $\pi_{MTP}$) is a principled and technically sophisticated approach, avoiding the pitfalls of a naive linear or log-linear interpolation.

### Weaknesses
- The paper's discussion briefly notes that MTD may "entangle genuine computational effort with memorization". This is a problem because the shallow MTP head is likely trained to be very good at predicting common, high-frequency (like, memorized) n-grams. A high MTD might simply signal that the full model is generating a novel or rare sequence (for example, a specific fact or a unique turn of phrase) that the shallow head couldn't possibly predict, which is not necessarily the same as in-context computation.
- In section 4.2. MTD is explained as a measure of "sophisticated in-context computation", but the experiments find that lower MTD correlates with correct reasoning. This seems contradictory: if high MTD means high computational effort, but low MTD means correctness, it suggests that high MTD might actually be a signal of the model "struggling" or engaging in inefficient/failed computation, rather than successful, deep reasoning.
- The MTD metric is strongly relative, because it measures the divergence between a full model and a specific shallow MTP head. The paper's results are contingent on the architecture of the MTP head used. If the MTP head were slightly more powerful or weaker, how would the MTD values and their correlations change?
- The main LLM experiments are carried out on a single 7B-parameter model (MiMo-7B). While this model is well-suited for the study, it remains unclear if these findings (especially the crucial MTD-difficulty and MTD-correctness correlations) generalize to other model architectures.
-

### Questions
The finding that lower MTD correlates with correct reasoning (sec 4.2, fig 7c) is one of the most interesting results of the paper. It seems to contrast with the initial motivation of MTD as a measure of "sophisticated" or "non-trivial" computation, which you might associate with successful reasoning.
Could you elaborate on this relationship? Does this finding imply that high MTD is a signal of inefficient or failed computation (like, the model is "struggling" or "thrashing" on its way to a wrong answer), rather than a signal of successful, deep reasoning? How do you handle this with the fact that high MTD also correlates with overall problem difficulty?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Multiple Token Divergence (MTD), an alternative to PHi for measuring computational effort by computing the KL divergence between a model's output distribution and that of a shallow auxiliary prediction head. Based on this, the authors propose Divergence Steering, a decoding method that uses MTD to control the "computational character" of generated text.

### Strengths
1. The paper is clear and well-motivated: the paper addresses limitations of PHi with a simpler, non-invasive method.
2. The evaluation is fairly comprehensive, covering both pre-trained models as well as those trained from scratch. While the model may require some adaptation (if it doesn't have an MTP head), it's less invasive than PHi.
3. MTD shows good correlation with task complexity and difficulty (e.g., on MATH). Furthermore, it's effective for CoT rationale selection when combined with NLL, giving MTD practical use.

### Weaknesses
1. The paper relies heavily on informal notions of concepts like "complexity", as well as "interesting" and "boring" tasks, instead of formal definitions.
2. The authors only focus on one "interesting" task in Section 3.1. It would be interesting to see if these results generalize to other "interesting" tasks.
3. Section 4.2 is missing a direction comparison to PHi for pre-trained models. Looking at the PHi paper, they report lower correlation with reasoning difficulty, but this could be due to other confounding factors (e.g., different pre-trained model). A direct comparison would be valuable here.
4. Likewise, more work is needed to understand the role of Divergence Steering with pre-trained models. Without such analysis, the impact of Divergence Steering is diminished.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
