# Utility-Diversity Aware Online Batch Selection for LLM Supervised Fine-tuning

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Supervised fine-tuning (SFT) is a commonly used technique to adapt large language models (LLMs) to downstream tasks. In practice, SFT on a full dataset is computationally expensive and sometimes suffers from overfitting or bias amplification. This facilitates the rise of data curation in SFT, which prioritizes the most valuable data to optimze. This work studies the online batch selection family that dynamically scores and filters samples during the training process. However, existing popular methods often (i) rely merely on the utility of data to select a subset while neglecting other crucial factors like diversity, (ii) rely on external resources such as reference models or validation sets, and (iii) incur extra training time over full-dataset training. To address these limitations, this work develops UDS (Utility-Diversity Sampling), a framework for efficient online batch selection in SFT. UDS leverages the nuclear norm of the logits matrix to capture both data utility and intra-sample diversity, while estimating inter-sample diversity through efficient low-dimensional embedding comparisons with a lightweight memory buffer of historical samples. Such a design eliminates the need for external resources and unnecessary backpropagation, securing computational efficiency. Experiments on multiple benchmarks demonstrate that UDS consistently outperforms state-of-the-art online batch selection methods under varying data budgets, and significantly reduces training time compared to full-dataset fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduced a novel online batch selection framework for supervised fine-tuning in LLMs. The crucial idea of this framework was to design the sample importance score by taking utility and diversity into consideration. Experimental results showed that the proposed framework provided improved fine-tuning performance and reduced training time.

### Strengths
**Originality:** This work presented a novel online batch selection framework by jointly considering data utility, intra-sample diversity, and inter-sample diversity. It connected the nuclear norm of the logits matrix with both data utility and intra-sample diversity. It designed a structured bilinear random projection of logits for compact embedding when estimating the inter-sample diversity.

**Quality:** The correlation between the loss reduction and the nuclear norm of the logits matrix was empirically justified. Experiments demonstrated the superior performance of the proposed online batch selection framework over baselines. The ablation studies also confirmed the impact of the data utility/intra-sample diversity and inter-sample diversity components.

**Clarity:** The notions of data utility, intra-sample diversity, and inter-sample diversity were clearly presented. The key theoretical analysis was highlighted.

**Significance:** The proposed framework shows the impact of data selection in supervised fine-tuning in LLMs. It would largely advance the data valuation areas in LLMs by automatically selecting valuable samples.

### Weaknesses
(1) The theoretical analysis of the intra-sample importance score via the nuclear norm is unconvincing.
- It is unclear why the nuclear norm is used, instead of the Frobenius norm discussed in Lemma 3.1. Given the connections of these two norms, what are the unique advantages of the nuclear norm to characterize the intra-sample importance score?
- It is highlighted that **High Nuclear Norm Indicates High rank and orthogonal rows (high diversity)**. It is confusing why the high nuclear norm can guarantee the high rank of the logits matrix. Here is a possible counterexample: there are two matrices $A = [[10, 0], [ 0, 0]]$ and $A = [[1, 0], [ 0, 1]]$, it holds that $rank(A) =1< rank(B)=2$, but $\Vert A \Vert_*=10 > \Vert B \Vert_*=2$.
- The analysis for the connection between the nuclear norm and the optimization utility is also unclear. It is confusing how Eq. (8) provides insights into understanding the correlation between the nuclear/Frobenius norm and the optimization utility. Their correlation is only empirically analyzed in Figure 2. It is unclear whether this correlation holds theoretically.

(2) It seems that the diversity distance of Eq. (9) focuses only on the distinction from recent data $Q$. It doesn't measure the inter-sample diversity of samples in the current batch $\mathcal{B}_t$. If so, it is possible to choose repeated samples from $\mathcal{B}_t$. Besides, a larger buffer size $M$ might lead to better measurement of the inter-sample diversity, but will also produce high memory requirements. It would be better to show the impact of the value of $M$ in balancing the inter-sample diversity measurement and the memory requirement.

(3) Some experimental settings are missing, e.g., the value of the buffer size $M$, the trade-off factor $\alpha$, etc.

### Questions
(1) Why does it require considering the optional history $\mathcal{H}_t$ in the optimization problem of Eq. (1)?

(2) How would the discrete Fourier transform (DFT) matrices $F_1, F_2$ be selected?

(3) What is the value of the trade-off factor $\alpha$ in Eq. (11)? How will it affect the online batch selection performance? 

(4) What is the value of $M$ in the experiments?

(5) How is the representation $z$ obtained in subsection 3.2, given the autoregressive language models in subsection 2.1?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce a Utility-Diversity optimized Sampling (UDS) method for SFT, focusing on 3 main axioms for scoring batches live to be chosen for fine-tuning: 1) jointly considering utility, and diversity for both inter-sample and intra-sample inclusion, 2) preventing external access or data leakage, 3) efficient reduction in training time compared to full SFT. 

To capture intra-sample value they use the norms of the logits matrix (equation 4),  and for inter-sample diversity the do a bilinear projection of logits for compact embeddings and show matching against historical data reduces redundancy (equation 9). Using the combined score they introduce an algorithm that picks the best batches to be used for SFT. Their technical depth in their proof relies on utilizing a sandwiching argument for relating nuclear norm and Frobenius norm of the logits matrix. They prove a theorem that they can indeed use this sandwiching argument to combine the intra and inter scores they have derived in additive form, while avoiding the storage of an explicit NV × d projection matrix and reduces the computational complexity from O(NV d) to O((N + V ) d log(NV )).

In their experimental setup they use Llama-3.1-8B and Qwen-2.5-7B models as backbone, and test against random sampling, regular (full data), MaxLoss (a scheme prioritizing samples with highest training loss), MaxGrad (a scheme prioritizing samples with largest gradient), RHO-Loss, and GREATS (they claim it as SOTA). Their results show victory over SOTA on all fronts.

One key observable gap in this work is the lack of extensive literature review on this topic. There are a ton of other smart sampling methods for SFT out there, including ones using Fisher Information gain, Optimal Design, and even model based estimations, where other models are used for inferring sample importance. Just one line of work I'm aware of that is heavily relevant is "Data-Efficient Supervised Fine-Tuning of Language Models Using Optimal Design" by Deb et. al. The authors could've done a much more detailed and rigorous literature study on this before claiming win over SOTA. However, the claimed computational gain is not evident from their presented results.

The mathematical rigor of their proof in Appendix C seems reasonable and correct to me, because mainly there is not much complication added other than using the Horn & Johnson Lemma (Lemma 3.1) the right way.

### Strengths
The paper is well written, the simplicity they have achieved in conveying their idea is exceptional, and their experimental setup seems to be comprehensive. The work seems original, inspired by their own contribution in ideas. The novelty of combining two aspects of intra and inter diversity is appealing and rare, and significance of their work is in pushing the boundaries of more efficient SFT via introducing more novel ideas.

### Weaknesses
Their literature review is not comprehensive and jumps to conclusion on winning over SOTA on all fronts without a deep dive into other methodologies in the literature. It could also benefit from a more comprehensive benchmarking of computation cost gain, since after all this is one of the key aspects where their work, i.e. efficient SFT, is expected to show value. The novelty in their proof technique doesn't seem to be exceptional. It seems to be a modified re-application of Johnson-Lindenstrauss lemma.

### Questions
1- Is there a complication in their proof technique that I'm missing? Would love to know more about the challenges they faced in making their proof work.
2- Is there a reason in their limited literature review that I'm not following? Maybe they have clustered the literature review into the best performing ones already, meaning that they already have concluded one of the benchmarks they compare agains is already known to be better than a ton of other works in the literature. In that case it would be great to make that statement clear and elaborate on it a bit more, or otherwise, make sure they include other existing literature as well.

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
4

### Summary
The paper proposes a online data selection method for fine-tuning LLMs. It operates without a validation set, instead using the model's own signals to identify valuable training instances. The core idea is to select data that promises a large loss reduction (utility) and is different from recently seen samples (diversity). To achieve this, UDS calculates a composite score for each data sample based on two components: intra and inter sample importance scores. The authors evaluated UDS on several benchmarks, including MMLU, ScienceQA, GSM8K, and HumanEval, using Llama-3.1-8B and Qwen-2.5-7B as base models.

### Strengths
+ The approach is strongly motivated by the idea that achieving the highest possible model performance is the primary goal.

+ The method is self-contained ("self trace") and does not require a held-out validation set. This makes it more versatile and applicable in scenarios where creating a representative validation set is difficult or impractical.

+ The authors have shown strong performance.

### Weaknesses
- The fundamental assumption that a large loss reduction is always beneficial is questionable. Such samples could be outliers, noisy data, or adversarial examples. By prioritizing them, the model might be reinforcing incorrect patterns or learning "bad regions" of the data distribution that are better left under-learned.

- A sample that yields a large loss reduction is not necessarily more informative; it could simply be an "easy" point for the model to learn quickly. The method may not distinguish between truly valuable, complex examples and those that just offer a steep but potentially trivial learning gradient.

- The inter-sample diversity score only compares a candidate sample to a buffer of historical samples. It does not ensure diversity within the newly formed batch. This could lead to selecting a batch of samples that are all novel compared to the past but highly redundant with each other, limiting the learning signal in that update step.

- The method's reliance on in-distribution data for both training and testing means its robustness is unknown. The lack of a validation set is a significant risk when facing domain shift, as the model's internal signals may no longer correlate with generalization performance on unseen distributions.

### Questions
Could you elaborate on why reducing repetition within a single training instance is critical for improving model performance?

How does the method distinguish between a valuable, under-learned sample and a noisy or misleading one that also produces a large loss reduction? How do you prevent the model from reinforcing learning in a "bad direction"?

The inter-sample score ensures diversity against past data. What is the motivation for not also considering diversity among the samples selected for the current batch to prevent redundancy in a single gradient step?

Have you evaluated this method when fine-tuning models that have already undergone instruction tuning?

How does your method perform on datasets that require long-context reasoning, where both utility and diversity might be harder to capture?

How does this selection strategy perform when the test set is from a different distribution than the training data?

How much total data was used for the training experiments, and were the different datasets combined or trained on separately?

In Figure 4, you state that K=8 corresponds to full-dataset fine-tuning. Could you explain this, as K usually represents the number of selected samples from a larger batch?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes **UDS (Utility-Diversity Sampling)**, an online batch selection framework for SFT of LLMs. For each candidate example, UDS computes (i) an **intra-sample score**: the **nuclear norm** of the sequence-by-vocab logits matrix to capture both optimization utility and within-sequence diversity; and (ii) an **inter-sample score**: a **diversity distance** computed in a low-dimensional embedding against a FIFO memory of recently selected samples. The two scores are combined to select the top-K examples per batch without extra backprop or external resources. Experiments on MMLU, ScienceQA, GSM8K, and HumanEval with Llama-3.1-8B and Qwen-2.5-7B report consistent gains and, in some settings, higher throughput than full-dataset SFT.

### Strengths
- **Clear, simple selection signal**: Using the logits nuclear norm avoids gradient computation and external models/val sets, aligning with the forward pass that already occurs each step. The paper also motivates the nuclear norm via bounds with the Frobenius norm.
- UDS consistently outperforms baselines (MaxLoss/MaxGrad/RHO-Loss/GREATS) on accuracy and often improves throughput vs. full-data SFT.

### Weaknesses
- All fine-tuning uses LoRA with small batch sizes; it would help to see results under full-parameter SFT or larger batches to ensure the gains persist when per-step signal-to-noise changes.
- ablations on buffer update policies (e.g., reservoir sampling, class-aware sampling) are not shown.
- Some missing recent papers on data efficiency for LLMs for related works discussion:

    1. Sachdeva, N., Coleman, B., Kang, W.-C., Ni, J., Hong, L., Chi, E. H., Caverlee, J., and Cheng, D. Z. How to train data-efficient llms. (AskLLM, Density sampling)

    2. Axiotis, K., Cohen-Addad, V., Henzinger, M., Jerome, S., Mirrokni, V., Saulpic, D., Woodruff, D. P., and Wunder, M. Data-efficient learning via clustering-based sensitivity sampling: Foundation models and beyond. In Proceedings of the 41st International Conference on Machine Learning. PMLR, 2024. (Clustered Sampling)

    3. Deb et al., FisherSFT: Data-Efficient Supervised Fine-Tuning of Language Models Using Information Gain. In Proceedings of the Forty-Second International Conference on Machine Learning (ICML), 2025.

### Questions
- Have you tried alternative **buffer policies** (reservoir sampling; clustering-based prototypes) and distance aggregations (e.g., farthest-k, soft-min) to reduce sensitivity to buffer composition?
- The first-order analysis is written for SGD. Is the loss-reduction correlation under Adam with LoRA adapters?
- The SRFT-style factorization - did you compare to smaller random features (e.g., sparse JL, CountSketch-style tensor sketches)

### Soundness
3

### Presentation
4

### Contribution
3
