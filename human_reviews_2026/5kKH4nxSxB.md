# Evidence-Free Claim Verification via Large Language Models

- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Hallucination detection is essential for reliable LLMs. Most existing fact-checking systems retrieve external knowledge to verify hallucinations. While effective, these methods are computationally heavy, sensitive to retriever quality, and reveal little about an LLM inherent fact-checking ability.
We propose an evidence-free claim verification task: identifying factual inaccuracies without external retrieval. To study this setting, we introduce a comprehensive evaluation framework covering 9 datasets and 16 methods, testing robustness to long-tail knowledge, claim source variation, multilinguality, and long-form generation.
Our experiments show that traditional uncertainty quantification methods often lags behind detectors based on internal model representations. Building on this, we develop a probe-based approach that achieves state-of-the-art results. 
To sum up, our setting establishes a new path for hallucination research: enabling lightweight, scalable, and model-intrinsic detection that can facilitate broader fact-checking, provide reward signals for training, and be integrated into the generation process.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper argues that RAG-centric fact verification is noisy, unscalable, and lacking in robustness. They propose the task of "evidence-free" claim verification to tackle these concerns. The authors benchmark 18 RAG-free verification methods on 9 datasets, and propose a probing approach, INTRA, which outperforms these existing methods.

### Strengths
- Exploring the ability of models to detect and correct factual inaccuracies while relying solely on their parametric knowledge is an important benchmark for foundation models.

- The emphasis on uncertainty calibration is appreciated, and serves well in the context of the RAG-free claim verification task.

- The INTRA approach is interesting and well-formulated. It seems like a reasonable design that builds nicely on prior methods.

- The experimental results are moderately informative.

- The multilinguality is nice: It would have been interesting to focus more on the challenges of extending fact verification benchmarks and methods to other languages.

### Weaknesses
While I appreciate INTRA and the provided experimental results, I find that, to me, these are the main exciting points of the paper. In particular, there are a few presentation and theory-centric concerns that I have:

- While I agree that models should be evaluated on their ability to identify hallucinations/inaccuracies with parametric knowledge alone, the task presented in the paper itself doesn't seem like a particularly noteworthy contribution. Theoretically, any fact verification benchmark could be trivially converted into a RAG-free benchmark.

- In the introduction, the authors could better position the role of evidence-free retrieval in the broader landscape of LLM evaluations, and how it complements RAG-based verification. As presented, the evidence-free paradigm is presented as an explicitly better replacement for RAG verification, when to me the two tasks seem largely complementary.

- The paper doesn't fully address the issue of context, or the fact that many statements are neither completely true or completely false. In the context of RAG verification, this is partially addressed by grounding claims against existing documents. Without this grounding, the veracity of many natural language statements becomes far less clear. This is partially addressed in Appendix C, but a larger discussion is warranted, specifically w.r.t. the relationship between this task and RAG-based benchmarks, and also especially w.r.t. applicability on real-world data.

### Questions
- How would the authors describe the purpose of this task, situated in the broader scope of fact verification in research and in practice?

- What challenges are involved in multilingual fact verification?

- How should ambiguous claims that are neither fully correct or incorrect be handled? For example, subjective claims?

### Soundness
3

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
2

### Summary
This paper introduces the task of evidence-free claim verification, where the goal is to detect factual inaccuracies in LLM outputs without relying on external retrieval systems. The authors conduct a comprehensive evaluation across 9 datasets and 18 methods, testing robustness across multiple dimensions including long-tail knowledge, multilingual claims, and cross-model generalization. They propose INTRA (Intrinsic Truthfulness Assessment), a probe-based approach that achieves state-of-the-art results by aggregating layer-wise predictions from middle layers using learned attention pooling and quantile normalization.  The work considers hallucination detection as a lightweight, model-intrinsic capability that could serve as a reward signal for training or be integrated into generation pipelines.

### Strengths
The evidence-free setting addresses real limitations of RAG-based approaches (latency, retrieval quality sensitivity, computational cost) while leveraging LLMs' parametric knowledge. he evaluation spans 9 diverse datasets covering important generalization dimensions (long-tail knowledge, multilinguality, cross-model, human-made vs. synthetic claims). Baselines have 18 methods across supervised and unsupervised categories. The paper also provides valuable analyses on layer-wise performance, popularity stratification, multilingual performance, and claim position effects that advance understanding of hallucination detection.

### Weaknesses
The evaluation is restricted to LLaMA 3.1-8B-Instruct. This do limit the generalizability of findings. While the paper argues for "lightweight" detection, no actual runtime, memory, or FLOPs comparisons are provided. Given that verbalized approaches are competitive and require no training, a more comprehensive comparison (including GPT-4, Claude, etc.) would be better. There is also no confidence intervals or significance tests are reported for the main results.

### Questions
How would INTRA be used as a reward model during training? Would it require frozen base model representations?
How does INTRA compare to verbalized approaches in terms of latency?
Figure 2: The claim position analysis is interesting but would benefit from error bars.
For inference-time use, how would the detector handle claims that require reasoning chains or multi-hop inference?
The quantile normalization step seems important but is under-explained?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Authors investigate the automatic fact checking task under the conditions that an LLM is used and no external knowledge is allowed. Their model is the continuation of the idea that the internal states of an LLM is a good indicator for verification of claims. They train an embedding layer on the representations of the LLM layers, then train a classifier to predict the veracity given the embedding vector, and then train a final classifier on the predictions of the layer-wise classifiers.

 The method is evaluated in numerous datasets and compared to various baselines and shows some improvements.

### Strengths
They paper reads well, and the experiments are detailed.

### Weaknesses
- Authors claim that they formalize the evidence free claim verification task. In my opinion, this is not something that they can be expecting to receive a credit for. Separating methods that rely on external data and those that don't, is not the formalization of a new research problem and is as trivial as showing two separate tables in the results section. It is insignificant and has been done by others [1].
- The proposed model is not significantly different from previous methods, such as [2]. The core idea is that the internal states of LLMs to some extent can reveal the veracity of claims, which we already knew for almost three years. Now, can we train more classifiers on each layer, and then train another classifier on the outputs of the previous classifiers to squeeze a few more percentage improvement?  Yes, probably. Which is basically what authors have done. But is this significant? In my opinion, no.
- The improvements are not good either. Out of 9 datasets, the method only works comparatively well in 2 datasets (CC and CMP). Interestingly these two datasets are synthetic datasets (created either by LLMs or a random process). Which shows the narrow scope of the applicability of the methods that only rely on the internal LLM states.



[1] Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models, 2023

[2] The internal state of an LLM knows when it’s lying, 2023

### Questions
None

### Soundness
1

### Presentation
3

### Contribution
1
