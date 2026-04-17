# Learning Shrinks the Hard Tail: Training‑Dependent Inference Scaling in a Solvable Linear Model

- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
We analyze neural scaling laws in a solvable model of last-layer fine-tuning where targets have intrinsic, instance-heterogeneous difficulty. In our Latent Instance Difficulty (LID) model, each input's target variance is governed by a latent ''precision'' drawn from a heavy-tailed distribution. While generalization loss recovers standard scaling laws, our main contribution connects this to inference. The pass@$k$ failure rate exhibits a power-law decay, $k^{-\beta_\mathrm{eff}}$, but the observed exponent $\beta_\mathrm{eff}$ is training-dependent. It grows with sample size $N$ before saturating at an intrinsic limit $\beta$ set by the difficulty distribution's tail. This coupling reveals that learning shrinks the ''hard tail'' of the error distribution: improvements in the model's generalization error steepen the pass@$k$ curve until irreducible target variance dominates. The LID model yields testable, closed-form predictions for this behavior, including a compute-allocation rule that favors training before saturation and inference attempts after. 
We validate these predictions in simulations and in two real‑data proxies: CIFAR‑10H (human‑label variance) and a maths teacher–student distillation task.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes the LID model, a simple analytical framework linking training-time and inference-time scaling laws. It shows that as training data scales, inference performance (pass@k) follows a steeper power-law w.r.t. k, with the effective exponent increasing and saturating at a limit—capturing how learning “shrinks the hard tail” of errors. The authors further derive a compute-allocation rule balancing training and inference effort and validate the theory through experiments on CIFAR-10H experiments.

### Strengths
- The paper introduces the LID model — a simple yet analytically tractable setup that connects training-time generalization scaling and inference-time scaling laws. This formalization is elegant and fills a clear theoretical gap between two rapidly developing empirical phenomena: training scaling laws and inference-time compute scaling.

- The derived theory provides clear insights into how the scale of training data affects inference scaling by effectively shrinking the hard tails of the error distribution. These are valuable conceptual contributions.

- The compute-allocation analysis (Section 4.3) offers interpretable results on when to invest in training versus inference compute, providing a practical takeaway.

- The paper also presents solid empirical results on CIFAR-10H that support the theoretical conclusions.

- The framework and analysis open up promising directions for future work, offering valuable foundations for studying the interaction between training-time scaling and test-time scaling.

### Weaknesses
The experiments are conducted on CIFAR-10, while inference-time scaling is a more relevant and currently critical topic for language models. It would be nice if some experiments could be conducted in that setting to better connect the work to practice.

### Questions
Please see the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the inference-time neural scaling laws specifically for the pass@$k$ metric. The analysis is grounded in a simple theoretical framework called the Latent Instance Difficulty (LID) model. Using this framework, the authors show that pass@$k$ follows a power law, where the exponent is shown to depend on both the generalization error of the trained model and the difficulty of the instance.

### Strengths
* This work provides a valuable theoretical analysis of inference-time neural scaling laws. A key contribution is establishing a novel connection, demonstrating how the scaling exponent is influenced by training-time dynamics.
* The paper introduces the Latent Instance Difficulty (LID) model, a simple yet valuable theoretical framework. This model is a notable contribution in its own right and shows potential for broader applicability beyond the immediate scope of this work.
* The theoretical contributions are well-supported by empirical validation on the Real-World CIFAR-10H dataset. This effectively bridges the gap between the proposed LID framework and real-world phenomena, demonstrating the practical relevance of the findings.

### Weaknesses
My main concern regards the presentation and rigor of the theoretical results. The current draft often discusses findings in a narrative, line-by-line fashion rather than consolidating them into complete, formal theorem statements. This fragmented presentation makes the paper's theoretical contributions difficult to follow and hard to assess.

In addition, the frequent usage of the approximation symbol ($\approx$) throughout the theoretical components, without a formal definition of the approximation level (e.g., in terms of asymptotic notation or explicit error bounds), makes the theoretical results feel incomplete and lack rigor.

Consolidating the key results (e.g., bounds, scaling exponents) into clearly stated theorems or propositions, along with rigorous definitions for all approximations used, would significantly improve the paper's clarity and trustworthiness.

I am open to increase my score if the presentation of this work can be improved during rebuttal period.

### Minors
* Please use LaTeX-style double quotes `` '' instead of " " (e.g., line 15, 20, 89).
* The metric 'pass@k' should be formatted in math mode pass@$k$ (e.g., line 16, 21).
* In line 134,137, 243, the authors use $Y_x^* \sim N(...)$ but $Y_x^* = N(...)$ would be more precise.
* It appears that every block equation has an equation number, even when not referenced. Please consider removing unnecessary equation numbers.

### Questions
* Could the authors elaborate on how these findings can be stated in a more rigorous manner? (See 'Weaknesses')
* Could the authors discuss on how the LID model might be applied in broader scenarios?

### Soundness
3

### Presentation
1

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
The paper derives a training-dependent inference exponent and shows that the pass@k failure rate, decays as a power law, with an exponent that is small for poorly-trained models and grows with the number of training samples N, eventually saturating at an intrinsic limit determined by the tail of the task’s true difficulty distribution. This implies that when the model is undertrained, the marginal benefit of
acquiring more training data is high. Once the model is well-trained and the inference exponent has saturated, further gains are best sought by investing in more inference-time compute.

### Strengths
- Investigating the effect of the training data and its size on pass@k is an important and timely topic
- The conclusions are backed up with theoretical analysis
- The analysis yield concrete empirical guidance for training vs test-time compute

### Weaknesses
- The discussion of the analysis is deferred to the last page of the paper (and very briefly in page 7), this makes it hard to follow the derivatives and the conclusions. I suggest the authors discuss the conclusions after every derivation to clarify the messages throughout the paper. Besides, I strongly suggest the authors to revise the introduction and discuss the conclusions as it is done in the last page. This is very important for conveying the message to the readers, currently all the important messages are hidden in notations and equations. As someone who feels comfortable reading math, I had a hard time to understand the "importance" of the conclusions even after reading the intro for a couple of times.

- I liked the paper and its message and I have no problem with analysis in the simplified linear setting. However, to confirm the validity of and importance the conclusions in practice (after all, pass@k is mainly used for LLMs), why don't the authors add experiments with LLMs? This is probably the largest weakness of the paper.

### Questions
- Can you add experiments with LLMs to confirm the conclusions? I saw that you mentioned "For complex reasoning, a more powerful model might not only learn the mean better but also fundamentally simplify the problem, an effect our current model does not capture". Does it mean that you expect the conclusions to not hold in practice for LLMs? or you still believe the conclusions would hold? Confirming this in practice would add a big value to the paper. 

ps: while experiments with LLMs are not cheap, fine-tuning smaller models is still doable in most academic labs.

### Soundness
3

### Presentation
3

### Contribution
3
