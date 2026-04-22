# Parameter-Efficient Reinforcement Learning using Prefix Optimization

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 8, 4, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) is a leading approach for tuning language models on mathematical reasoning tasks. However, it remains unclear whether RLVR's gains stem from genuine reasoning improvements or simply from steering the model toward answer formats that already appear in the reference distribution. Inspired by recent evidence \citep{zhao2025echo,yue2025does}, we study this question by optimizing only the first $k$ tokens (e.g. $k=32$) of each solution, generating the remainder of the response from the reference model. We study two methods for prefix optimization, using a naive algorithm that clusters prefixes and selects the best prefix (Prefix Clustering), and a method that optimizes the prefix by finetuning a lightweight adapter model with RL (Prefix-RL). We show that tuning only the first $k$ tokens can significantly improve the accuracy on math, suggesting that at least some of the gains from RL are due to upweighting a preferable solution strategy. Our results suggest that simple prefix optimization methods can provide an efficient alternative to RL, delivering substantial improvements across different models and benchmarks for a tiny fraction of the compute required for standard RL, and that these gains are robust across prefix lengths and random seeds.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates parameter-efficient reinforcement learning (RL) for math reasoning by optimizing only the first k generated tokens (the “prefix”) and letting a frozen, larger target model complete the solution. 

Two approaches are explored: (1) Prefix-Clustering, which selects a single fixed prefix by clustering candidate prefixes from the reference model and choosing the best on a training set; and (2) Prefix-RL, which RL-finetunes a small 1B adapter model to generate the first k tokens while the large target model remains frozen. 

Using verifiable rewards (answer correctness), the authors show consistent gains on math benchmarks (e.g., MATH-500, AIME, Minerva) across Qwen and Llama families, including FP8-quantized ones, with significantly lower training compute than full-model RL. The empirical results suggest that a substantial share of RL gains arises from steering toward effective formats/strategies rather than improving token-by-token reasoning across the entire sequence.

### Strengths
- This paper introduces a compute-lean adapter-based RL setup where only k initial tokens are learned, separating strategy choice from long-horizon generation.
- The pipeline is well-illustrated (adapter emits prefix; target completes; reward computed on final answer).

### Weaknesses
1. The paper implicitly assumes early tokens determine the solution strategy that the model will keep following. This may not hold for reflective/iterative solvers (e.g., o3-like, DeepSeek-R1, Qwen-Thinking) that backtrack, revise, or branch mid-solution. The generality of prefix steering under multi-pass reflection remains untested.

2. Prefix-Clustering protocol seems train-set-dependent and of unclear inference value. The method traverses MATH-train to choose a single best fixed prefix, which may not be practically meaningful at inference time (and risks train-set over-selection).

3. To support the efficiency claim, add a direct baseline: “1B Prefix-RL + Large Target” vs. “Full RL on the Large Target” under matched or budget-normalized compute and matched data. Without this, the efficiency–performance trade curve is hard to judge.

4. Some figure narratives (e.g., the 1B self-completion plot analogous to Fig. 3) could better articulate what hypothesis each figure specifically tests (e.g., how much of full-RL gain is recovered by prefix control?)

5. The paper states this is the first demonstration of RL finetuning applied to quantized models yet the method does not RL-update the quantized target’s weights—only the small adapter is updated while the FP8 model is used for inference-only completion. As worded, this can be misleading.

### Questions
See weakness.

### Soundness
2

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
3

### Summary
This paper investigates whether the performance gains observed in RLVR for mathematical reasoning are due to genuine improvements in reasoning ability, or primarily from shifting the model toward high-accuracy solution strategies already present in the base distribution. To answer this, the authors propose prefix optimization: only the first k tokens of a generated solution are optimized, while the remainder is completed by a frozen reference model.

Two methods are evaluated:
1. Prefix Clustering — selects a fixed prefix via k-means clustering of sampled candidate prefixes and uses it for all inputs.
2. Prefix-RL — trains a small adapter using PPO to generate an input-conditional prefix conditioned on the question.

Despite modifying only a tiny fraction (first 16–64 tokens) of the sequence, both methods yield substantial accuracy improvements on math benchmarks such as MATH-500, AIME, AMC, Minerva, OlympiadBench, often recovering a large share of full RL gains. Prefix-RL is compute-efficient, works with quantized models, avoids catastrophic forgetting, and requires inference-only access to the main model. Improvements are most pronounced when the adapter and target share a model family.

Overall, the work argues that strategy selection and formatting, not deep reasoning skill, may explain a substantial portion of RL gains.

### Strengths
1. A simple enough method, especially Prefix Clustering, not only brings significant improvements on downstream tasks, but also unveils that the high-quality solution already learned in the pre-training distribution. It offers another profound insight into the origin.
2. Highly compute-efficient;
3. Could work with closed-weight models;

### Weaknesses
1. Lack of direct comparison to full RL at a large scale
2. Lack of comparison to other parameter-efficient RL methods.
3. Generalization beyond math remains uncertain;
4. Prefix clustering harms Qwen but helps Llama, suggesting architectural or data-distribution differences worth deeper investigation. I think more analysis on why Qwen behaves differently needs to be conducted.

### Questions
Same to Weakness.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method for performing RL with low computational resources. The approach optimizes a small model to generate the beginning portion of responses, after which a large model completes the remaining decoding. The authors experiment with Llama3.1 and Qwen2.5 series models on several mathematical reasoning tasks. Results show that prefix-RL can achieve most of the performance gains of standard RL using relatively little computational resources.

### Strengths
1. This paper proposes a method for RL under low computational resources that can achieve most of the performance gains with far less computational cost than conventional RL.
2. The proposed method does not require full access to the target model; it only needs inference access. Therefore, it is applicable not only to open-source models but also to closed-source models.
3. The paper designs a Prefix Clustering experiment to verify the importance of the beginning portion of the response for performance gains, and further proposes the main method of this work, Prefix-RL.

### Weaknesses
1. The method in this paper is limited by the need to use models from the same family, which may cause it to perform poorly or even fail in settings involving closed-source models.
2. Experiments in this paper were conducted only on mathematical reasoning tasks, so the method's applicability to other RLVR tasks remains unknown.

### Questions
1. The upper-right subplot of Figure 4 shows an anomalous behavior of Prefix Clustering on the Qwen model; in L375–L376 the paper explains this as "Qwen’s preferred openings are more input-dependent." Could a clearer example be provided to substantiate this point?
2. If the target model were used directly as the adapter model, what kind of performance could be expected? This approach seems to potentially eliminate the need for an additional model and avoid the restriction that the method requires a smaller model within the same family. Theoretically, such performance should lie between the current method and Full-RL, and it would also allow a clearer comparison of the stylistic and performance differences between the "prefixes" obtained by this method and those obtained in the paper.
3. Has Prefix-RL shown gains on OOD tasks? For example, can we observe that the adapter model produces more guiding responses in tasks other than mathematical reasoning?
4. In L460–L461 it is mentioned that "cross-family configurations lead to performance degradation." Is there any data that can visually show the extent of these performance drops? Also, note that open-source models versus closed-source models would also count as "cross-family."

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
4

### Summary
This paper introduces a methodology called _Prefix-RL_ to algorithmically identify ways for an LLM to start its response given a user input so that it is more likely to correctly answer math questions. The work uses this methodology to fine-tune both LLama and Qwen models on the mathematical reasoning benchmarks MATH, AIME, AMC23, and Minerva. The results show similar gains to direct RL finetuning.

### Strengths
* The paper's primary strength is the ~3000x reduction in FLOPs and the 4x reduction in GPU requirements during training. This approach to RL fine-tuning is much more accessible to research labs than tuning the full model.

* The core idea is based on an interesting insight and is also practical to implement. The insight that prefixes index into parts of the training data that are useful for answering certain math questions is potentially a fruitful idea for inspiring more works.

* This method is demonstrated on FP8-quantized models, which, as noted in Section 3.2, was previously difficult to do. It seems to make progress on the performance gap between quantized Llama-8B and its full-precision counterpart, which is impressive and useful.

### Weaknesses
*Weaknesses*
* The work is posed as a form of parameter-efficient RL, but only compares against a standard RL baselines. A more fair comparison would consider other techniques for parameter-efficient RL such as LoRA [Hu et al., 2022] (or QLoRA [Dettmers et al., 2023] for quantized models) or Adapters [Houlsby et al., 2019]. It would also be nice to compare against prefix-tuning, as mentioned in the related work, given that it can be directly trained on the same labels generated for RL as a supervised signal. 

* It is unclear why the baseline method of prefix clustering was selected with k=16, whereas the experimental method was tested at k=32 and k=64. I believe this is a bit of an unfair comparison, as it is possible that k=16 is simply not enough tokens to meaningfully index into the parts of the LLMs training data that are “good” for solving math questions, which is the core insight of this work. It would be better to have a comparison that has equal numbers of k values across all approaches.

* Additionally, selecting k doesn’t seem to be very clear-cut. In Table 1, it appears that having k=32 seems to work well for some models/benchmarks (e.g., the Qwen-72B model or the Minerva Benchmark), whereas k=64 works better for others. It seems difficult to know ahead of time what value should be selected for k to achieve good performance. This mitigates some of the benefits of efficiency, because implementing this approach now requires an engineer to search over the k-values that work the best.

* The paper appears to report results from a single training run for each experiment. The training curves (e.g., in Figure 4) seem to be quite noisy, with high variance between steps. Selecting the "best checkpoint" from a single, noisy run is not super robust to initial conditions. The paper should report the mean and standard deviation over multiple runs (e.g., 3-5) with different random seeds to establish statistical significance, as is the standard in the RL literature.

* This approach seems to rely on having some objective way of calculating “correctness” of an answer. It is not clear how well this does under different kinds of label noise that is common in RLHF. Having a “correct answer” is a large assumption for problems that LLMs are typically used for, such as creative writing, open-ended dialogue, or exploratory information retrieval. 

*Minor Edits*
* The section on Prefix clustering is a bit confusing. Is it one prefix for all evaluation examples, or is it the nearest cluster’s prefix? It seems like the former, but a more reasonable baseline would be the latter.
* It is not immediately clear why setting g_theta to the same size and architecture of the reference model implies that “improvement from RL upweights existing strategies” as claimed in Sec 2 under the subheading “Prefix-RL”. This could be better elaborated on.
* In Section 3.1, the authors state the MATH training split has 7,500 examples. This dataset is appears to have 12,500 examples. Your subsequent filtered number of 8,888 examples also suggests the starting number was larger than 7.5k. Please double-check and clarify this dataset statistic.

References:
Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. "Lora: Low-rank adaptation of large language models." ICLR 1, no. 2 (2022): 3.

Dettmers, Tim, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. "Qlora: Efficient finetuning of quantized llms." Advances in neural information processing systems 36 (2023): 10088-10115.

Houlsby, Neil, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. "Parameter-efficient transfer learning for NLP." In International conference on machine learning, pp. 2790-2799. PMLR, 2019.

### Questions
The questions below correspond to the number of bullet points of the weaknesses.

* 1.1: How does this technique compare to other techniques for parameter-efficient RL such as LoRA [Hu et al., 2022] (or QLoRA [Dettmers et al., 2023] for quantized models) or Adapters [Houlsby et al., 2019]?
* 1.2: What are the benefits of this approach of using RL with automatically calculated labels vs. supervised fine-tuning with automatic labels?

* 2.1: How does prefix clustering at k=32 and k=64 compare to the proposed approach?

* 3.1: how can one determine a k-value for their problem?
* 3.2: what is the worst-case number of evaluations to make for k?

* 4.1: how consistent are these results across different training runs?

* 5.1: how sensitive is this approach to noise in labels?
* 5.2: how applicable is this approach to other problems that are common uses of LLMs?

### Soundness
3

### Presentation
2

### Contribution
2
