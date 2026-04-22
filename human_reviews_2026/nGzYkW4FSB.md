# Synchronizing Probabilities in Model-Driven Lossless Compression

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 8

## Abstract
It is well-known in the field of lossless data compression that probabilistic next-symbol prediction can be used to compress sequences of symbols. Deep neural networks are able to capture rich dependencies in data, offering a powerful means of estimating these probabilities and hence an avenue towards more effective compression algorithms. However, both compressor and decompressor must have exactly matching predictions; even small differences from non-determinism (which often happen with learned models due to hardware, software, or computation order) can lead to cascading decoding failures. In this paper, we formalize the problem of prediction mismatch in model-driven compression, and introduce Probability Matching Interval Coding (PMATIC), a model-agnostic algorithm that tolerates bounded prediction mismatch with low overhead. PMATIC works with the predicted probabilities, making it compatible as a drop-in replacement for the arithmetic encoder in model-driven compression tools. We show theoretical correctness and performance bounds for PMATIC, and validate these results on text data. These results confirm that, when paired an advanced prediction model, PMATIC is robust to prediction mismatch while achieving compression rates that out-perform standard modern compression tools.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles a practically critical but underexplored problem in model-driven lossless compression: the instability caused by prediction mismatch between encoder and decoder due to LLM non-determinism. The authors formalize this as the probability matching problem and propose PMATIC (Probability-Matched Interval Coding), a theoretically grounded and model-agnostic alternative to arithmetic coding that tolerates bounded prediction mismatch. Theoretical guarantees on decodability and compression efficiency are presented, and experiments on Wikipedia and Enwik8 datasets validate correctness and reasonable efficiency loss under synthetic noise.

### Strengths
- The work highlights a key yet overlooked obstacle in deploying LLM-based compression systems: non-deterministic inference, bridging the gap between theory and practical system reliability.

- The formalization of bounded prediction mismatch and the introduction of a matching interval coding mechanism are conceptually clean and mathematically sound. The paper also provides provable correctness and upper bounds on the additional code length, demonstrating an informed trade-off between robustness and compression efficiency.

- PMATIC is model-agnostic and can be seamlessly integrated as a drop-in replacement for arithmetic coding, showing potential broad applicability beyond text compression.

### Weaknesses
- Experiments are confined to synthetic perturbations on a single model (Llama-3.1) and small text corpora. This does not convincingly capture the stochastic, architecture- or library-dependent non-determinism that motivates the problem. I'd like to see more experiments. 

- There already exist benchmarks for model-driven or LLM-assisted compression [1,2,3]. These could provide a valuable performance baseline.

- Runtime, helper-bit statistics, and computational overhead are not reported. It is unclear how PMATIC scales when applied to larger models or real-time compression.

- Although the paper derives an analytical bound on the extra bit cost, it does not verify whether the empirical losses follow this bound.

- Some related works are missing [4,5,6]

- In all, I think this paper addresses an important and practical problem, but the experimental section is insufficient. If the authors can provide comprehensive and convincing experimental results during the rebuttal period, I will raising my score.

[1] Valmeekam C S K, Narayanan K, Kalathil D, et al. Llmzip: Lossless text compression using large language models[J]. arXiv preprint arXiv:2306.04050, 2023.
[2] Mao Y, Pirk H, Xue C J. Lossless Compression of Large Language Model-Generated Text via Next-Token Prediction[J]. arXiv preprint arXiv:2505.06297, 2025.
[3] Mittu F, Bu Y, Gupta A, et al. Finezip: Pushing the limits of large language models for practical lossless text compression[J]. arXiv preprint arXiv:2409.17141, 2024.
[4] Mao Y, Li J, Cui Y, et al. Faster and stronger lossless compression with optimized autoregressive framework[C]//2023 60th ACM/IEEE Design Automation Conference (DAC). IEEE, 2023: 1-6.
[5] Mao Y, Cui Y, Kuo T W, et al. Accelerating general-purpose lossless compression via simple and scalable parameterization[C]//Proceedings of the 30th ACM International Conference on Multimedia. 2022: 3205-3213.
[6] Goyal M, Tatwawadi K, Chandak S, et al. DZip: Improved general-purpose loss less compression based on novel neural network modeling[C]//2021 data compression conference (DCC). IEEE, 2021: 153-162.

### Questions
- How large is the helper-bit overhead in practice, and how does it vary with δ?

- What is the computational impact, such as encoding/decoding latency compared to standard arithmetic coding?

- The paper mentions the potential extension to stochastically bounded mismatch. Could the authors elaborate on whether PMATIC can be adapted to probabilistic mismatch distributions observed in real inference pipelines?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
We theoretically formulated the problem of "non-determinism" that arises in lossless compression using large-scale language models (LLMs), and proposed a new compression coding method, PMATIC (Probability-Matched Interval Coding), to overcome this problem.
LLM outputs a probability distribution $P(x_t|x_{<t})$, but even for the same input, it may produce slightly different probabilities due to different GPUs, different libraries, or differences in parallel order. Conventional arithmetic coding breaks down if the probability distributions of the encoder and decoder do not match perfectly. However, if the probability mismatch is sufficiently small (bounded mismatch), accurate decoding is possible if both agree on an "intermediate common distribution", which is the Probability Matching problem.
PMATIC is a method that extends existing arithmetic coding with probability quantization, ensuring consistent coding even if the encoder and decoder make slightly different probability predictions.
For PMATIC, this paper theoretically evaluates its correctness and compression loss.

### Strengths
Originality: This paper is the first to mathematically formalize the problem of "probabilistic model-driven compression" $\times$ "LLM nondeterminism".
Quality: Not only is there a mathematical discussion of performance analysis, but the usefulness of the proposed method is also verified through numerical experiments, making the paper of high quality.
Clarity: The problem to be addressed is clearly stated, the algorithm is given in detail, and the argument is clear enough.
Significance:  This paper is one of the first to formulate the problems that are encountered when actually applying LLM as lossless compression, and is expected to make an important contribution to related research.

### Weaknesses
As mentioned in 6. Future work, if LLM is actually used for lossless compression, the size of the models required to implement the compressor and decoder will be a major problem. This issue has not been discussed in this paper.

### Questions
There has been discussion about static performance, such as the achievable compression performance. However, how much better is it compared to existing methods in terms of the amount of computation required for compression and decoding?

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
4

### Summary
The authors propose an algorithm to quantize / bin the probability distributions to account for possible deviations of the predicted CDFs between the encoder and decoder. The motivation is due to the non-determinism of LLMs, which can result in small differences in floating point numbers even when run on the same hardware.

### Strengths
- The method is reasonably simple to apply to existing LLMs, without needing to retrain.
- The authors provide some guarantees on the added bit length due to binning.
- This is a very important, and realistic problem, that needs to be solved to have next-gen AI codecs, and I appreciate the authors pushing on real world problems.

### Weaknesses
- The experiments are significantly lacking in breadth. If the method is general, the authors could provide further experiments with different data modalities.

- The following is hard, but would significantly improve the paper: can the authors estimate what are typical deviations present in relevant scenarios where AI codecs could be applied? For example, take any open source model, and apply the encoder and decoder using 1) a different version of CUDA, 2) different models, and other variables that might vary in practice. This would significantly improve the contribution and ground the paper in real world applications.

- Adding figures explaining the binning procedure would significantly improve the exposition of the algorithm.

### Questions
- Do the authors have a good sense of how large the mismatches are in practice (see box above)?
- Can the authors provide more experiments across data modalities, and possibly varying other variables that could impact the degree of mismatch (see box above)?
- Can the authors add figures to explain the binning better?

### Soundness
3

### Presentation
1

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a method to address the problem of practical usage of LLMs in data compression. A novel algorithm is presented for compression with uncertainty in probability predictions based on arithmetic coding. Furthermore mathematical analysis is presented showing compression bounds under a given uncertainty. Experiments denote successful demonstration with synthetically generated noise.

### Strengths
1. This is a very relevant problem, I believe this would allow the community to actually make practical compressors with the proposed algorithm. 
2. Article is original and a novel algorithm is proposed. Paper is quite easy to read.

### Weaknesses
1. The only weakness I would say is some analysis on what delta's would we expect by changing hardware or going from GPUs to CPUs. Also there is relevant research addressing uncertainty in LLM prediction which should be added to related work (https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
2. Please add comparison with some more compressors, especially CMIX. Also include latency because that is where traditional compressors win by a huge margin.

### Questions
Please see weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3
