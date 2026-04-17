# Don't Ignore the Tail: Decoupling top-$K$ Probabilities for Efficient Language Model Distillation

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
The core learning signal used in language model distillation is the standard Kullback-Leibler (KL) divergence between the distribution of the student and the teacher. Traditional KL divergence tends to be dominated by the teacher’s highest-probability modes, thus diminishing the influence of less probable yet potentially informative components of the output distribution. We propose a new tail-aware divergence that decouples the contribution of the teacher model's top-$K$ predicted probabilities from those with lower probabilities, while maintaining the same computational profile as the KL Divergence. Our decoupled approach reduces the impact of the teacher modes and, consequently, increases the contribution of the tail of the distribution. Experimental results demonstrate that our modified distillation method yields competitive performance in both pre-training and supervised distillation of decoder models across various datasets. Furthermore, the distillation process is efficient and can be performed using a modest academic budget for large datasets, eliminating the need for industry-scale computing capabilities.\footnote{We used LLMs like Grammarly and ChatGPT-Plus to check grammar and spelling and to polish our work.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a tail-aware knowledge distillation (KD) method for language models, termed TAD. The approach decouples the top-$K$ probabilities of the teacher distribution from the tail (remaining probabilities) in the KL divergence loss, thereby increasing the gradient signal for lower-probability outputs (the "tail") during distillation. This decoupling allows the student model to better capture information carried by less probable teacher outputs, overcoming the well-known issue where vanilla KD is dominated by teacher models. The authors present mathematical derivations, a thorough gradient analysis, and comprehensive empirical results across several pretraining and supervised distillation tasks. Notably, their approach achieves competitive or superior accuracy on multiple benchmarks, including mathematical reasoning tasks, all within modest computational constraints.

### Strengths
1. This paper clearly points out a key weakness of standard knowledge distillation -- that the loss is mainly affected by the teacher's top tokens. It then suggests a clear and logical way to fix this problem by dividing the teacher's output distribution based on token ranks. The mathematical explanations show a solid understanding of their arguments.

2. The paper addresses compute constraints and provides FLOP cost comparisons, establishing that TAD is practical for academic-scale training and accessible without industry-level GPUs.

### Weaknesses
1. The paper's treatment of the tail amplification factor $\beta$ is largely empirical and lacks theoretical grounding. Section 2 does not provide formal analysis of convergence, stability, or statistical effects from scaling the tail term, even though $\beta$ is described as sensitive. Without clearer justification, it remains uncertain when amplifying low-probability tokens improves generalization versus merely amplifying noise.

2. Although the evaluations are broad among LLMs and mathematical domains, most experiments are focused on either standard few-shot evaluation tasks or mathematical reasoning datasets. The absence of tasks in other domains -- such as dialogue modeling, summarization after supervised training.

3. The codebase is not released at time of review. While the paper claims code will be made available upon acceptance, currently, replication depends on implementation from scratch.

### Questions
See weaknesses.

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
4

### Summary
This paper introduces Tail-Aware Distillation (TAD), a new method to improve language model distillation. It solves a key problem in standard distillation (using KL divergence), where student models over-focus on the teacher's most likely predictions (the "head") and ignore the informative, low-probability "tail."

TAD uses a new loss function that decouples the head and tail, amplifying the learning signal from the tail. This forces the student to learn the teacher's full knowledge distribution.

Experiments show TAD outperforms other methods in both pre-training and supervised distillation (especially for math) while remaining highly efficient, adding almost no computational cost over standard distillation.

### Strengths
1. This paper is clearly written and easy to follow.
2. TAD is more efficient and more effective than the baselines.

### Weaknesses
The paper's primary weaknesses are in its experimental design, specifically regarding fair cost comparison and the scale of pre-training.

1. Unfair Computational Budget: The comparison between TAD and the CE (no-KD) baseline is misleading. It ignores TAD's massive upfront computation, namely the teacher inference cost (to generate logits) and the full logit storage cost (making Top-K optimization impossible). A fair comparison would fix the total computational budget, which would require training TAD for significantly fewer steps than the CE baseline.

2. Insufficient Pre-training Scale: Pre-training on only 5B tokens is too small to draw reliable conclusions. The paper would be much stronger if it demonstrated the method's effectiveness at a larger scale, ideally by plotting the scaling law curves for each method (TAD, Vanilla KD, and CE) to show how performance scales with increased compute.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
For Distillation of LLMs, this paper proposes a modification of KL Divergence loss that decouples and upweighing the loss on the tail of the distribution from that of the head. The proposed loss modification does not require any overhead, and is simple to implement. Across a range of student model sizes (0.5B to 2B) and teacher model sizes (1.5B to 9B), the authors demonstrate consistent improvement across multiple datasets for pre-training, continued pre-training, and fine-tuning when training for short horizons (2B tokens).

### Strengths
1. The proposed method is simple and intuitive, and the proposed modification to KLD divergence can be implemented with negligible overhead.
1. The paper is well written and easy to follow, the derivations and proofs are detailed.
1. The authors compare a range of model families and sizes, for both teacher and student, including randomly initialized, pre-trained, etc.
1. The proposed method shows consistent improvement in pre-training, in continued pre-training, and in fine-tuning, particularly for fine-tuning.

### Weaknesses
1. The sequence-level normalization in Equation 2 seems somewhat ad-hoc and less principled.
1. The pretraining experiments are extremely extremely small - eg 2B tokens for 1.2B model. This is extremely, extremely far from model convergence (significantly smaller than even chinchilla's "optimal" of ~20x tokens to params), and performance may converge/fall below other methods in longer horizons.
1. Results on most datasets in Table 1/Table 4 are almost at the random value of these datasets (due to the extremely small pre-training duration) - this will somewhat limit the reliability of these scores.
1. The authors method specifically upweighs the tail of a distribution, which has been shown to negatively impact model calibration and downstream fine-tuning.
Given the compute limitations, the paper would perhaps be better positioned if its focus was primarily on the supervised fine-tuning domain, with more extensive SFT experiments. My rating for this paper is closer to 5, but ICLR does not allow that as a score.

### Questions
1. Line 87 the authors state "This normalization makes the loss stable for nominal values of $\beta$ such as 1 or 2". But $\beta=1$ corresponds to vanilla KLD - did the authors observe some instability without this modification even for $\beta=1$ ?
1. For Vanilla KD results, I assume the authors used only the KLD loss, without adding the CLM loss? Adding a CLM loss has been shown in multiple prior works to increase KD performance. Can the authors report results with adding the same weightage of CLM loss to KLD as used in their own method, specifically perhaps for Table 6? This will help better isolate effects of the proposed method to that of simply adding CLM loss.
1. Similar to Anshumann et al. (2025), can the authors compare the Expected Calibration Error of the models trained with their method to that of Vanilla KD and to that of just CLM? Anshumann et al. (2025) observed that up-weighing the tail of the distribution improved pre-training scores, but worsens model performance post fine-tuning due to worse calibration of the student - something I conjecture may also happen with the author's method. Specifically, the total probability mass in the student head may be less than the accuracy of the head tokens, and in the tail may be greater than accuracy.
1. This calibration error can also been from the gradient analysis in Equation 5. In general, the teacher will have larger capacity than student, and hence higher accuracy. As LLMs are often well-calibrated, this means $ \sum(p^T_k) > \sum(p^S_k)$ i.e., the teacher will have more probability mass in the head. Let us assume $\beta(X)=2, \sum(p^T_k)=0.8,  \sum(p^S_k)=0.6$ .Then Equation 5 becomes $ \frac{\partial{L}}{\partial{z_i}}= 1.5p^S_i - 2p^T_i $. So even if $p^S_i$ becomes equal to $p^T_i$, this gradient will cause $p^S_i$ to still increase further. Taking any of their trained model, can the authors compare the gradient direction for the tail tokens compared to that of a vanilla KLD?
1. Line 344 the authors state "Adding the cosine loss on hidden states improved both Vanilla KD and TAD". Do the "Vanilla KD" rows in Table 4 include the cosine loss? Can the authors also share results without this cosine loss? 
1. (Minor) The authors state "The code is currently not released to preserve anonymity" - the authors could anonymously share the code.
1. (Minor Typo) Line 852 $p^T$ should be $p^S$

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Tail-Aware Distillation (TAD), a novel knowledge distillation method for training smaller causal language models from larger teachers. Its key contribution is decoupling the KL divergence loss into a top-K term and a tail term, amplifying the gradient contribution of the tail probabilities to prevent the student from over-focusing on the most likely tokens. The method is computationally efficient, requiring similar FLOPs to vanilla KD, and enables effective large-scale pretraining and supervised distillation even with limited resources. Extensive experiments show TAD produces competitive or superior students across various model sizes and tasks, including mathematical reasoning, outperforming recent methods like MiniPLM.

### Strengths
This paper's primary strengths are threefold. 

First, it presents a novel and principled adaptation of Decoupled Knowledge Distillation, transitioning it from supervised image classification to the label-free domain of language model pre-training via a new rank-based (Top-K vs. tail) decoupling. 

Second, it is supported by rigorous theoretical foundations, including a detailed derivation of the proposed loss and an insightful gradient analysis that explains the method's dynamics. 

Finally, the work is substantiated by comprehensive experiments demonstrating the algorithm's effectiveness and versatility across diverse settings—including pre-training distillation from scratch, domain adaptation for specialized skills like mathematical reasoning, and supervised fine-tuning—while consistently maintaining a computational efficiency comparable to the highly practical Vanilla Knowledge Distillation.

### Weaknesses
1. The introduction fails to clearly articulate the motivation behind the proposed method. The rationale for developing Tail-Aware Distillation (TAD) only becomes apparent later in Section 2, which disrupts the logical flow and makes it difficult for readers to grasp the core contribution early on.

The formalization of the method is not self-contained. For instance, the notation \mathcal{P} is used without a clear definition, and the formulation of vanilla knowledge distillation is not properly introduced. This lack of clarity hinders understanding, especially for readers unfamiliar with the baseline approaches.

2. The authors explicitly state that TAD is not a variant of Decoupled Knowledge Distillation (DKD), arguing that DKD is label-anchored while TAD is rank-anchored. However, when K = 1 and the teacher's top-1 token is treated as a pseudo-label, the two methods exhibit substantial similarities. This suggests that TAD can be viewed as an adaptation of DKD to the language modeling domain, rather than a fundamentally new approach. The distinction drawn by the authors appears overstated and may misrepresent the true nature of their contribution.

3. The method introduces two new critical hyperparameters: the Top-K value and the tail weight coefficient \beta. Although a sensitivity analysis is provided, showing robustness around K = 5-10 and \beta = 2, this adds non-trivial complexity and tuning cost compared to Vanilla KD, which has no such parameters. For practitioners, this necessitates extra work to determine the optimal configuration for different teacher-student pairs and data domains, potentially hindering the method's ease of adoption.

4. The experimental comparisons are primarily limited to Vanilla KD, Sequence-KD, MiniLLM, and MiniPLM. The evaluation would be more comprehensive and convincing if it included other relevant and advanced distillation techniques. For instance, comparisons with methods using Reverse KL divergence (On-policy KD) are absent. Including these baselines would better situate TAD's performance within the broader landscape of knowledge distillation research and provide a clearer understanding of its relative advantages.

### Questions
Given that the core idea of decoupling the KL divergence into target vs. non-target (or top-K vs. tail) components was originally proposed in Decoupled Knowledge Distillation (DKD, Zhao et al., 2022) for vision tasks, could the authors more explicitly clarify the novelty of their method beyond a direct adaptation of DKD to the language modeling domain? 

While the shift from image classification to causal LM pretraining is non-trivial, the fundamental decomposition and re-weighting strategy appear conceptually similar. What are the key algorithmic or theoretical innovations in TAD that distinguish it from a straightforward application or extension of DKD to language?

### Soundness
3

### Presentation
2

### Contribution
2
