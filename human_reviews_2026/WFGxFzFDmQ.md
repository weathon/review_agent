# Detecting Benchmark Contamination Through Watermarking

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Benchmark contamination undermines LLM evaluations, and existing post-hoc detection methods are heuristics and thus lack verifiable guarantees.  We propose a proactive solution: embedding cryptographic watermarks into benchmarks \emph{before} their release through question reformulation with a language model, and introduce a detection algorithm that overcomes tokenizer mismatches by aligning text prefixes to reliably identify the watermark signal in the suspect model. To validate our method, we pre-train 1B-parameter models on 10B tokens with controlled contamination of MMLU and ARC. The watermarking process preserves benchmark utility, while our test detects contamination with high confidence, achieving e.g. a $p$-value $< 10^{-5}$ for a mere 5\% performance gain on 5000 MMLU questions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method for detecting whether LLMs have been trained on benchmark data (benchmark contamination). Before benchmark release, the authors rephrase the benchmark samples using keyed semantic-preserving transformations, introducing a slight probabilistic bias toward a “greenlist” of tokens. If a model has encountered these samples during training, its conditional distribution will exhibit a statistically significant preference for the greenlist tokens. Detection is based on a binomial significance test, providing verifiable $p$-values. Controlled contamination experiments show a consistent relationship between performance inflation and statistical significance. The paper also provides tokenizer-aligned implementations and multiple benchmark experiments, proposing a framework that can be adopted for evaluation governance.

### Strengths
1. The paper tackles an important and timely problem: benchmark contamination directly affects reported leaderboard results and evaluation integrity.

2. The engineering implementation is practical, offering a “pre-release watermark → post-release verification” workflow with tokenizer alignment, which could serve as a foundation for evaluation governance.

### Weaknesses
The paper’s core strategy of keyed rephrasing combined with statistical testing is closely related to several existing lines of research across benchmark, pretraining, and fine-tuning contexts, making the boundary of novelty somewhat blurred.  

(Benchmark scenario) — STAMP (Rastogi et al., 2025) introduces multi-keyed rephrasings for the same original text and conducts paired statistical tests between public and private variants to prove whether a benchmark or test set was included in training. This process is conceptually identical to this paper’s approach of keyed rephrasing and post-hoc significance testing, differing mainly in the form of the test statistic (paired likelihood versus binomial counting) and artifact management. Similarly, Oren et al. (2023) present a black-box proof of contamination using order-based likelihood tests, which, while not using watermarks, share the same objective of providing verifiable contamination evidence without access to training data or weights.

(Pretraining scenario) — STAMP’s main goal is dataset membership proof for pretraining corpora. Its two-stage pipeline (keyed rephrasing plus paired test) and threat model closely mirror those of the present paper if we treat a benchmark as a subset of the pretraining corpus. To stand out, the current paper needs to emphasize its incremental advances in statistical calibration, detection power, or governance workflow.

(Fine-tuning scenario) — The causal chain (insert signal during training → detect significance after training) is the same as in the current work, differing mainly in target domain (fine-tuning versus benchmark). Can You Finetune Your Binoculars? (Elhassan et al., 2025) embeds watermarks into model weights using dual LoRAs for generator and observer optimization, focusing on model-level traceability but following a similar training-stage embedding and post-hoc verification logic. Towards Watermarking of Open-Source LLMs (Gloaguen et al., 2025) systematically examines watermark durability under model merging, quantization, and further fine-tuning, discussing post-training detectability in open-weight models. These efforts share the conceptual foundation of encoding verifiable biases during training and statistically detecting them afterward.  
   
   Thus, although this paper applies the framework to benchmark evaluation, the conceptual path—training-time signal embedding followed by statistical testing—has been explored across these adjacent settings. I appreciate the authors with further discussion on the following references. Besides, this paper did not discuss the usage of LLMs, which is required by ICLR this year.

---

### References

- Elhassan, F., Ajroldi, N., Orvieto, A., & Geiping, J. (2025). Can you finetune your binoculars? Embedding text watermarks into the weights of large language models. arXiv. https://arxiv.org/abs/2504.06446  
- Gloaguen, T., Jovanović, N., Staab, R., & Vechev, M. (2025). Towards watermarking of open-source LLMs. arXiv. https://arxiv.org/abs/2502.10525  
- Oren, Y., Meister, N., Chatterji, N., Ladhak, F., & Hashimoto, T. B. (2023). Proving test set contamination in black box language models. arXiv. https://arxiv.org/abs/2310.17623  
- Rastogi, S., Maini, P., & Pruthi, D. (2025). STAMP your content: Proving dataset membership via watermarked rephrasings. ICML 2025 / arXiv. https://arxiv.org/abs/2504.13416  
- Yang, S., Chiang, W.-L., Zheng, L., Gonzalez, J. E., & Stoica, I. (2023). Rethinking benchmark and contamination for language models with rephrased samples. arXiv. https://arxiv.org/abs/2311.04850

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a method that embeds watermarks into questions in LLM benchmark datasets to enable benchmark contamination detection. The watermarks are added by rephrasing the questions with increasing the probabilities of a list of randomly selected tokens (called the green list) being chosen as the next token in the decoding process. The list being randomly generated, a model that has never seen the dataset will predict the next token independently of the green list, conditionally on the previous tokens (the null hypothesis). On the other hand, if a model were trained on a benchmark dataset with proposed watermarks, its predictions would have statistical dependence on the watermarks (the alternative hypothesis). The proposed method uses this fact to construct a statistical test for the detection. Proposition 1 shows that this test correctly controls the false positive error rate.

Then, the authors present several experiments to confirm that
the rephrasing does not largely change the performance evaluation from the original one and that the p-values are sufficiently small when there is contamination. Moreover, they provide more numerical analyses on the impacts of the model size, window size, benchmark size, rephrasing model, tokenizer, and so on.

### Strengths
- The paper tackles a relevant and important issue concerning the modern research area.

- The proposed ideas of generating the pseudo-random green list is interesting and effective.

- It presents extensive empirical results for supporting the authors' claims and for analyzing the impact of the hyper parameters.

- There is a theoretical guarantee (Proposition 1) for the proposed detection method.

### Weaknesses
- The proposed method seems to focus on models that are trained and make inference based on next token prediction, but the paper does not mention this limitation.

- The proposed method is effective to detecting contamination, but it does not prevent it.

- Some details about the proposed method and the experiments are missing.

- Some part of the writing is not clear to me.

### Questions
- What is deduplication?

- If the next token is almost deterministically identified based on the previous token, artificially increasing the probabilities of the green tokens as proposed may make the rephrased question invalid. Does this never happen?

- Lines 270--271:
> We train 1B transformer models (Vaswani, 2017)

  Do the authors use the exactly the same architecture and the training objective as in Vaswani et al. (2017)? (Also, please correct this bibliography item. The other authors are missing.)

- In the experiment of Table 4, why do the authors limit the $k$ so small? I find the proposed $k=2$ quite small. Why don't you test larger ones?

**Other minor comments**:

- Lines 286--287: What are "the different choices" and why do we want to choose "the one with the smallest loss" for the evaluation?

- Lines 279--280:
> Between steps 2500 and 7500, every 5000/#contaminations, we take a batch from the shuffled concatenation of the three benchmarks instead of the batch from DCLM.

  This sentence is unclear. I believe it requires rewriting.

- "Inferential evidence": I think I understand the nuance, but I would like to suggest avoiding the word "inferential" in such an informal sense, especially in research on artificial intelligence. Do the authors believe these words are appropriate here?

- Table 1 (and other similar experiments): What is "the number of condemnations"? (What is the unit?)

- Lines 285--305: what is the difference between the setups in calculating the p-values in Table 1 (and Table 6) and Figure 3b (and Figure 8)? I suppose the radioactivity score $S$ was only used in the Contamination detection experiment.

- Lines 350--352:
> Table 1 links the contamination detection to the actual cheating on the benchmarks when $\delta = 4$ is used. We see that for the three benchmarks, whenever the cheat is greater than 10%, detection is extremely confident.

  How do I see this from Table 1?

- "Figure 4.4": Figure 4?

- Line 411--412: "larger models require fewer contaminated batches to achieve the same gain on the benchmark". Is this something we can see from Figure 4?

- In Figure 5:
> Question: “The rate of acceleration of an object is determined by the mass of the object and”

  Is this question complete?

- In Table 4: "dedup" --> deduplication?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a proactive framework to detect benchmark contamination in LLMs. Instead of trying to find evidence of contamination after a model is trained, the authors suggest embedding a cryptographic watermark into a benchmark before its public release. This is achieved by using an LLM to rephrase the benchmark's questions, embedding a secret statistical signal (the watermark) during the text generation process.
To test a suspect model, a benchmark provider can then check for the presence of this watermark's statistical bias in the model's predictions. The method requires white-box (logit) access to the suspect model and includes a novel algorithm to handle mismatches between the tokenizers used for watermarking and by the suspect model. Through extensive experiments involving pre-training 1B-parameter models with controlled contamination on benchmarks like MMLU and ARC, the authors demonstrate that their method preserves the benchmark's utility while detecting contamination with very high statistical confidence.

### Strengths
- Proactive and Verifiable: The core strength is its shift from post-hoc, inferential detection methods to a proactive approach that provides verifiable, statistical proof (a p-value) of contamination. Meanwhile, the experiments show that even with a strong watermark, the rephrased benchmarks remain effective for evaluating and ranking models, with performance being very similar to the original versions.
- Practicality: The paper addresses the practical challenge of distinct tokenizers by introducing a robust cross-tokenizer detection algorithm, making the solution more broadly applicable.

### Weaknesses
- White-Box Access Requirement: The detection test requires full logit access to the suspect model. This limits its use to open-source models and cannot be used by external parties to audit closed, API-only models.
- Vulnerability to Intentional Evasion: The framework is primarily designed to detect unintentional contamination from sources like web scraping. A determined, malicious actor could potentially devise strategies to circumvent detection, such as by rephrasing the questions again to remove the watermark or by training only on the answers.

### Questions
- The experiments focus on question-answering benchmarks. How do you envision this watermarking approach being applied to other types of evaluation, such as code generation, mathematical reasoning (where rephrasing is brittle), or creative writing tasks?
- The method focuses on watermarking the questions. Have you considered the possibility of watermarking the provided answers for open-ended questions?

### Soundness
3

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
The paper introduces a method that offers provable evidence of benchmarks for LLMs. The key idea involves watermarking benchmarks before public release, achieved through rephrasing LLMs. Contamination in downstream models is then reliably identified using a hypothesis test that leverages the "radioactivity" inherent in watermarks. The authors successfully validate this approach, demonstrating its efficacy by detecting contamination in an LLM pre-trained on a contaminated corpus. Furthermore, they confirm that the rephrasing process maintains the original utility of the benchmarks.

### Strengths
* The paper is well-written, easy to follow, and features comprehensive, well-executed experiments.
* The proposed technique is demonstrably effective. To confirm its reliability and understand how the repetition factor influences the strength of the statistical test, the authors trained multiple LLMs from scratch on a pre-contaminated corpus.
* The authors propose a novel detection algorithm to detect radioactivity in case of tokenizer mismatch between the rephrasing LLMs and the suspect LLMs. This enhances the practical applicability of the algorithm.

### Weaknesses
* The method overlaps significantly with past works that were not used as baselines or mentioned in related works [1,2,3,4]. For example, [1] also proposes a similar idea and proposes a hypothesis test that leverages watermarked rephrases. [2] proposes a dataset inference method that can also be used for detecting contamination. Works such as these should be compared in related works and evaluated against as benchmarks. 


* The current demonstration, while confirming the utility of the rephrased benchmarks across various LLM sizes, is limited because all tested LLMs belong to the same model family as the one used for the original rephrasing. It is essential that the authors extend their evaluation to include LLMs from different model families.


* For reliable detection at a statistically significant strength (requiring approximately 16 repetitions as shown in Figure 3b), a substantial amount of repetition is necessary. Furthermore, it would be beneficial to include a demonstration of how the size of the total training corpus influences the test's strength. The detection also requires benchmarks of significant sample size (around 2k-4k) samples.


* The authors should also empirically verify that their method is robust against false positives using off the shelf LLMs. For a more rigorous test of false positives, the detection strength should also be evaluated against keys that are distinct from the contaminated keys. 


I would be willing to reconsider my overall assessment based on authors' responses to the above points. For instance, based on favorable comparisons with highly relevant prior work.

References:

[1] STAMP Your Content: Proving Dataset Membership via Watermarked Rephrasings. ICML 25

[2] LLM Dataset Inference: Did you train on my dataset?

[3] DE-COP: Detecting Copyrighted Content in Language Models Training Data

[4] Proving Test Set Contamination in Black Box Language Models

### Questions
* Will the proposed method be applicable to other general forms of dataset?

* Could the authors include the % percentage of scored tokens in Table 5?

* The method does not seem to be applicable on existing benchmarks, can I suggest that the paper acknowledges this limitation in their work?

### Soundness
3

### Presentation
3

### Contribution
3
