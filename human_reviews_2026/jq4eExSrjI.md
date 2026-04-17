# Bayesian Data Reweighting Improves Retrieval in Knowledge-Based VQA

- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
Knowledge-based Visual Question Answering (VQA) requires retrievers to incorporate external knowledge, e.g., documents, to answer questions. Existing retrievers are typically optimized with standard contrastive learning, which treats all non-positive pairs as equally informative, leading to false negative bias and difficulties in hard negative mining. To overcome these issues, we propose \textbf{Bayesian Data Reweighting (BDR)}, a probabilistic framework that assigns learnable importance weights to query-document pairs and performs Bayesian inference over these weights. We derive closed-form posterior updates under conjugate priors and develop an efficient EM algorithm for weight estimation. This approach adaptively emphasizes informative pairs without explicit hard negative mining. Experiments on two representative multimodal retrievers demonstrate consistent improvements, with BDR achieving gains of up to $8.6$ points on individual datasets and an average recall of $68.6$ across all M2KR datasets, surpassing the previous state-of-the-art.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Bayesian Data Reweighting (BDR) — a probabilistic framework for improving multimodal retrievers in knowledge-based visual question answering (KB-VQA). Instead of treating all non-positive samples equally (as in standard contrastive learning), BDR introduces learnable importance weights for each query-document pair and performs Bayesian inference to adaptively emphasize informative negatives and down-weight false ones.

The authors derive closed-form posterior updates under conjugate priors using an auxiliary variable augmentation scheme, enabling efficient inference through a stochastic Expectation-Maximization (EM) algorithm. They prove theoretical guarantees for asymptotic consistency and finite-sample convergence, and empirically demonstrate that BDR improves retrieval performance across both CLIP-based retrievers (PreFLMR) and LLM-based retrievers (VLM2Vec).

Experiments on multiple KB-VQA datasets (e.g., OKVQA, InfoSeek, EVQA) show consistent gains. Integrating BDR into retrieval-augmented generation also improves downstream VQA accuracy and BLEU scores, sometimes surpassing even large LLMs like GPT-4V in end-task performance.

### Strengths
1. BDR introduces a Bayesian probabilistic treatment of sample weighting within contrastive learning, which is an elegant and theoretically grounded approach to mitigate false and hard negatives. The authors provide formal results (conditional conjugacy, consistency, and finite-sample bounds), giving BDR mathematical credibility beyond empirical heuristics.
2. The stochastic EM algorithm allows practical application to large-scale datasets, overcoming scalability concerns typical of Bayesian inference.
3. Evaluation spans multiple retrievers, architectures (CLIP, LLM-based), and datasets, showing consistent and substantial improvements. The study also assesses efficiency, retrieval quality, and VQA accuracy, providing a full picture of impact. BDR improves both retrieval and downstream VQA generation, outperforming prior state-of-the-art retrievers and even some much larger LLM systems.
4. The presentation is good and clear.

### Weaknesses
1. Although the stochastic EM is efficient, the paper does not fully quantify its additional training-time cost relative to vanilla contrastive learning or other reweighting baselines.
2. Sections 3.2–4 are mathematically heavy; the presentation could be streamlined. Should be improved to avoid overwhelming readers unfamiliar with Bayesian inference.
3. The study lacks direct comparison with other hard-negative mining or debiased contrastive learning methods under the same multimodal retrieval setting, which would help isolate BDR’s contribution.

### Questions
1. How does the proposed approach's runtime compare with standard InfoNCE training (e.g., training time per epoch or GPU-hours)?
2. How are the importance weights initialized during training? Do they converge to stable distributions, or require annealing / regularization to prevent collapse?

### Soundness
4

### Presentation
4

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
This paper introduces Bayesian Data Reweighting (BDR), a novel framework designed to improve multimodal retrievers for knowledge-based visual question answering (VQA). BDR addresses the limitations of standard contrastive learning by assigning learnable, adaptive importance weights to positive and negative samples through a principled Bayesian inference procedure.

### Strengths
- The method demonstrates performance gains across multiple model architectures (CLIP-based and LLM-based)
- Theoretically, it provides proofs for inference via conjugate priors and establishes the statistical consistency of its objective.

### Weaknesses
- The empirical improvements attributed to BDR appear marginal in several key scenarios. For instance, on the EVQA dataset in Table 1, the gains over the InfoNCE baseline are minimal. Furthermore, the average improvement reported in Table 3 is modest. These results raise questions about the practical significance and consistent advantage of BDR over strong baselines.
- The experimental evaluation primarily uses the standard InfoNCE loss as the baseline. However, to properly situate BDR's contribution, it is crucial to compare against more advanced methods that also address false and hard negatives, such as the debiased contrastive loss and hardness-aware weighting schemes, which are discussed in the related work. Without such comparisons, the relative merit of the proposed Bayesian approach remains unclear.
- The theoretical analysis relies on the assumption that negative samples are i.i.d. This is a strong and often unrealistic assumption in contrastive learning, where negatives are typically sampled from a shared batch, introducing structured correlations. 
- The paper positions BDR as a general framework for contrastive learning. However, its effectiveness is demonstrated solely within the domain of knowledge-based VQA. Evaluation on other canonical contrastive learning tasks may be helpful to substantiate the claim of generality.

### Questions
The theoretical analysis proves global properties like consistency but does not formally link the Bayesian weighting mechanism to the core concepts of false and hard negatives. Could the authors provide further insight, either theoretically or empirically, into how the inferred weights correlate with the semantic "hardness" or "falseness" of negatives? A deeper analysis showing that the framework reliably suppresses false negatives (assigning near-zero weights) and up-weights informative hard negatives would significantly strengthen the causal claim behind its success.

### Soundness
2

### Presentation
2

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
This paper focuses on the Knowledge-based Visual Question Answering (KB-VQA) task. The main motivation of this paper is that standard contrastive learning of KB-VQA ignores the potential hierarchical structure of negative pairs including true, false, and hard negatives. These negative pairs need to be handled carefully in the training phase of contrastive learning, therefore a Bayesian reweighting method is proposed by the authors.

### Strengths
-	The topic of this paper looks reasonable.
-	The paper is well-written.
-	The experimental results are promising.

### Weaknesses
-	The main motivation of this paper is that standard contrastive learning of KB-VQA ignores the potential hierarchical structure of negative pairs including true, false, and hard negatives. Although the authors cite related paper to justify this argument, it lacks obvious quantitative (e.g., the statistics of these hierarchical negative pairs in datasets) or visualized feature embeddings among different pairs to demonstrate the reasonability of this motivation in the introduction or experiment section. 
-	For the Efficient Inference with Stochastic Expectation Maximization, it is necessary to formulate the specific complexity using stochastic Expectation-Maximization
(EM) algorithm.
-	For the Augmented Likelihood and Conditional Conjugacy, it is confused that how the data- augmentation is introduced. To me, the random variable $\mu$ is the alternative of explicit sample reweighting and how the random variable links the data-augmentation method? It is a parameter-augmentation method rather than the data-augmentation one? This section is quite mess and it is hard to follow easily.

### Questions
NA

### Soundness
2

### Presentation
2

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
The authors identify that treating all negative pairs as equally informative can lead to false negative bias, making hard negative mining particularly challenging. To address this issue, the paper introduces a novel Bayesian data reweighting approach that calibrates the contributions of positive and negative samples to improve knowledge retrieval for KB-VQA. An efficient EM algorithm is proposed to estimate the optimal weights for both positive and negative examples. The proposed method achieves superior performance compared to previous approaches on the M2KR benchmark.

### Strengths
- Proposes a novel Bayesian data reweighting method for KB-VQA retrieval.

- Demonstrates consistent improvements over baseline retrieval setups across various backbones and two architectures.

### Weaknesses
- Please use the correct citation style in LaTeX for ICLR.
- In some results (e.g., on OVEN), the baseline scores are higher and should be bolded in Table 1 instead of the proposed method.
- Regarding the VQA performance on E-VQA, it is unusual that the VQA accuracy is much lower than EM, as they should theoretically be on par. Moreover, previous work typically reports the BEM score, which is the standard metric used in the original E-VQA paper. For fair comparison with prior work, the authors should report BEM instead of accuracy for E-VQA. Additionally, the oracle results reported in the original E-VQA paper are substantially higher under the BEM metric. It would be valuable if the authors could reproduce and report the corresponding BEM scores with Oracle for comparison.
- The different configurations of the Gamma prior appear to yield exact identical performance. Could the authors elaborate on how these hyperparameters influence the final results and provide more details behind their selection?
- Similarly, please provide more detailed insights on the three types of priors, beyond the brief intuition mentioned in the paragraph starting at Line 206.

### Questions
- Line 424: What does BRCL stand for?

- How does the training cost (computation/time) compare with the InfoNCE baseline?

- Since the method assigns weights to all examples, could these weights be used to rank examples in the corpus for positive/negative selection or data pruning?

### Soundness
2

### Presentation
2

### Contribution
3
