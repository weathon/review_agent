# Extending the Context of Pretrained LLMs by Dropping Their Positional Embedding

- Decision: Accept (Poster)
- Scores: 6, 2, 4, 2

## Abstract
So far, expensive finetuning beyond the pretraining sequence length has been a prerequisite to effectively extend the context of language models (LM). In this work, we break this key bottleneck by ***Dro**pping the **P**ositional **E**mbeddings of LMs after training (DroPE)*. Our simple method is motivated by three key theoretical and empirical observations. First, positional embeddings serve a crucial role during pretraining, providing an important inductive bias that significantly facilitates convergence. Second, over-reliance on this explicit positional information is also precisely what prevents test-time generalization to sequences of unseen length. Third, positional embeddings are not an inherent requirement of effective language modeling and can be safely *removed after pretraining* following a short recalibration phase. Empirically, DroPE yields seamless *zero-shot* context extension *without any long-context finetuning*, quickly adapting pretrained LMs without compromising their capabilities in the original training context. Our findings hold across different models and dataset sizes, far outperforming previous specialized architectures and established rotary position embedding scaling methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper challenges the conventional methodology by bravely removing positional encoding at serving time (NoPE), but the model had been initially trained using RoPE and recalibrated with continuous pretraining in a later phase without positional encoding. 

For the background, it is well-known that NoPE
1. can learn positions of tokens due to causal mechanisms in attention;
2. but underperforms in terms of negative log-likelihood when comparing with most existing positional encodings.
Although NoPE is known to have good generalization properties in terms of context length, it is generally dismissed as the widespreading intuition is that the positional information is "hard to learn" or even "lost". 

The paper provides an explanation of why NoPE has difficulties in learning positional information. On the other hand, it also provides some observations to explain why RoPE, though learning the positional information fast, has trouble with length generalizations.

### Strengths
1. Simplicity.
2. The idea is fairly novel.
3. Results are fairly compelling with good theoretic analysis to back them up.
4. Potential impact. I personally think this inspires the search of better positional encoding by not only looking at architecture manipulations, but also the training process.

### Weaknesses
- One of the biggest concern to widely adopt this in LLM training is a rigorous study to confirm potential risks. There are two aspects that I'm particular concerned, despite the good numbers on academic benchmarks
  1. The loss spike and cooling down, does it hurt further learning dynamics? What does the spike do to other layers' parameters, parameter/grad norms, etc. Would it make further SFT different?
  2. This is less important as to a "weakness", but how does it fare on other more up-to-date LLM evaluations? This might be relevant if we want to understand if it is genuinely without regression on all tasks, or whether there are some issues that we need to understand.
- There seems to be a lack of experimental rigorousness. For this I still very much want to understand all these comparison setups, what are held as control variables and how the confounders are ruled out. Also there seems to be a sneaky modification of architecture in Appendix C.1, even though they "look minor", introducing unwanted variable makes experiments shaky. See my questions below.

### Questions
It's possible that I miss certain things already in the appendix, but below are the questions I want to understand better.

- I don't understand Figure 9 and Table 4: are you comparing the original SmolLM with SmolLM + extra 30/60/120B token's finetuning? Or did you match an equi-token setup and further finetune SmolLM for the same amount of tokens? If the former, it is perhaps not entirely fair and it still bears the question as to how much the original model would trend higher given the additional training. Also, do you have certain measurement of confidence interval on those eval accuracies? 
- In Table 2, what are the exact setup for SmolLM+PI/NTK/YaRN, and which token budget for the DroPE did you use?
- In Appendix C.1, you mentioned a "simple QKNorm". Does that mean you modified the architecture slightly after dropping the positional encoding? What happened without? Are we able to see a detailed analysis on ablating all these variables?

I will read more later and write more questions if something arises.

I am tentatively giving it a 6 as I think it is very intriguing work and the authors have very good math foundations and taste. But experimental rigorousness is crucial. **I may revise the score higher or even lower** as I understand more about the details of the work.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces DroPE, a simple and effective method for extending the context length of pretrained language models without expensive long-context finetuning. The core idea is to remove the positional embeddings (PEs) from a fully trained model and then perform a short "recalibration" training phase. The authors argue that PEs are a crucial scaffold for efficient pretraining but inherently limit generalization to longer sequences. By dropping them post-training, DroPE achieves significant zero-shot context extension. Empirical results on a 0.5B parameter model and the SmolLM model show that DroPE outperforms established RoPE scaling methods on long-context tasks while preserving performance within the original context window.

### Strengths
1. The central idea of dropping positional embeddings to achieve context length extrapolation is interesting. It reframes PEs as a temporary training aid rather than a permanent architectural component.

2. The proposed DroPE method is simple to implement and demonstrates strong empirical performance, simply but effectively alleviating the OOD issue of RoPE.

### Weaknesses
1. The paper's central claims are not validated at a scale that reflects the current state of long-context LLMs. The experiments are limited to relatively small models (under 0.5B parameters) and a modest 2x context extension (e.g., 2048 to 4096 tokens). This is a significant limitation, as the most pressing need for context extension exists in much larger models (7B+) and for vastly longer sequences (e.g., 32k, 128k, and beyond). It is unclear if the method's effectiveness and training stability would hold when extending context by 10x or 100x, where challenges like attention dilution and loss of positional signal become far more severe. The strong claims of the paper require evidence at a more demanding scale to be fully convincing.

2. While the paper provides extensive analysis, a notable portion of it lacks novelty and feels disconnected from the proposed method. The analysis in Section 4, which details the failure of RoPE-scaling methods due to the compression of low frequencies, largely reiterates well-established findings from the original papers on YaRN, NTK-RoPE, and LongRoPE. Furthermore, these theoretical insights do not directly inform the specific design of the DroPE recalibration process, such as the required duration or optimal hyperparameters. The theory explains why a problem exists but offers little guidance on how to best implement the proposed solution.

3. The theoretical justifications in Section 3, intended to prove the necessity of PEs during training, rely on arguments that feel trivial. For example, Proposition 3.2 proves that a NoPE transformer's gradients vanish on an artificial sequence of identical tokens. This is an unrealistic edge case that has little bearing on training with diverse, real-world data. Spending significant space on such formalisms seems to over-justify a widely accepted premise (that PEs help training) and detracts from what could have been a more focused empirical investigation into the DroPE method itself.

### Questions
1. How do you expect DroPE to perform when scaling to much larger models (e.g., 7B+) and extending the context by a much larger factor (e.g., from 4k to 128k)? Do you anticipate any new optimization challenges?

2. Have you evaluated DroPE on tasks that are highly sensitive to precise token positions (e.g., Passage reranking in HELMET)? Is there a trade-off where long-context generalization comes at the cost of fine-grained positional awareness?

[1] HELMET: How to Evaluate Long-context Language Models Effectively and Thoroughly

3. What is the intuition for what the model learns during recalibration? Is it primarily learning to infer relative positions from the causal mask, and have you observed corresponding changes in attention patterns?

### Soundness
2

### Presentation
2

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
This paper proposes a novel method to extend the context length of pretrained language models by removing RoPE after pretraining and conducting a short recalibration phase. The authors argue that positional embeddings are crucial for training convergence but harms zero-shot generalization to longer contexts.

### Strengths
1. Novel and counterintuitive hypothesis where positional embeddings might not be necesssary throughout a model's lifecycle.
2. Treating PEs (Positional Embeddings) as "training scaffolds" that can be removed is elegant and very good for downstream use cases such as long context fine-tuning and kv-cached inference performance.
3. Good theoretical contributions on why PEs are necessary during training and why they can be removed after.
4. Strong potential for practical impact because of how simple this method is. Furthermore, reduces the inductive biases of LLMs after removing RoPE.

### Weaknesses
1. Experiments on small models might not scale well to larger models. Could be possible to take existing large LMs and recalibrate them with DroPE.
2. No ablations or thorough experimentation on the recalibration phase. Naively removing PEs might work but given the theory there should be better ways. Furthermore, recalibration cost is not explored either.
3. Dubious claim that YaRN cannot extrapolate to longer contexes, directly contradicting the original paper and results from the industry (DeepSeek R1, Qwen3, GPT-OSS).
4. Lack of comparisons against more recent length generalization work (LongRope 2, sparse attention, etc.)
5. Only test at 2k context lengths, most methods now test at 128k+.
6. Specific solution to autoregressive LLMs, does not generalize to diffusion models.

### Questions
1. Why was YaRN not able to extrapolate to 2x context length? This might be a mistake since Qwen3 uses YaRN without finetuning and has a perfect score in the needle-in-a-haystack benchmark.
2. Do you know if DroPE is better than YaRN because it has more training time during the recalibration phase? Can you compare it against YaRN but with the same recalibration phase?
3. Do you think that gradually removing the RoPE during the recalibration phase be better than outright removing it in a single step? It's highly likely that DroPE is equivalent to setting a RoPE Linear scaling to infinity.
4. What happens with DroPE's performance at higher context scaling lengths? 4x, 8x, etc?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces DroPE, a method that removes RoPE after pretraining to address its limitations. The authors demonstrate that while RoPE provides crucial inductive bias for rapid convergence during training, its explicit positional encoding hinders zero-shot generalization to longer sequences. They analyze why existing RoPE-scaling techniques fail in zero-shot scenarios by showing how they distort low-frequency attention heads essential for long-range semantic understanding. The core innovation is DroPE, which eliminates RoPE and employs a brief recalibration phase, compelling the model to rely on implicit positional cues from the causal mask and data patterns. In experiments, this approach achieves robust zero-shot extrapolation when extending the context length to twice the training length, significantly outperforming complex RoPE-scaling methods.

### Strengths
1. The paper shows that RoPE trained from scratch outperforms NoPE: NoPE has higher perplexity and exhibits gradient-vanishing issues during training.
2. The authors propose a new way to obtain a NoPE base model by “dropping” RoPE from a well-trained RoPE model (Drop RoPE).
3. They evaluate on selected datasets (NIAH and four LongBench subsets) and empirically demonstrate that RoPE→NoPE bases achieve length-generalization benefits for limited extrapolation ranges (mainly around 2×).

### Weaknesses
1. To me, the manuscript reads more like a blog post than a formal paper. The central claim—that DropRoPE provides a generalization advantage—feels weak across several experimental dimensions: the variety of base models tested, the extrapolation distances considered, the number and comprehensiveness of benchmarks, and the choice of baselines.
2. The observation that performance can be restored by continued training after dropping positional encodings has been reported elsewhere. For example, Table 5 in [1] shows smollm-135M can recover performance via continued training even when positional encodings are fully or partially removed.

[1] Towards Economical Inference: Enabling DeepSeek’s Multi-Head Latent Attention in Any Transformer-based LLMs (https://arxiv.org/pdf/2502.14837v1)

### Questions
1. Can NoPE extrapolation methods such as [2] be integrated with DropRoPE?
2. A 2× extrapolation advantage is likely of limited practical significance. Can you evaluate generalization over a wider range of lengths (for example, from 2× up to 8×, as in [2])?
3. Why were only four LongBench subsets selected rather than the full LongBench suite?
4. Can the method be validated on widely used LLMs (e.g., LLaMA, Qwen)?

[2] Length Generalization of Causal Transformers without Position Encoding (https://arxiv.org/pdf/2404.12224)

### Soundness
1

### Presentation
2

### Contribution
2
