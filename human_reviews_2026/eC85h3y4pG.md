# Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Recent advances in depth-recurrent language models show that recurrence can decouple train-time compute and parameter count from test-time compute.
In this work, we study how to retrofit existing pretrained non-recurrent language models into depth-recurrent models.
We find that using a curriculum of recurrences to increase the effective depth of the model over the course of training preserves performance while reducing total computational cost.
In our experiments on grade-school math, we train on math data from Common Crawl and observe that retrofitting pretrained models to be depth-recurrent results in better performance at a given training compute budget than simply post-training the original non-recurrent language model. Further, we train our retrofitted recurrent models on a mixture of FineWeb-Edu and high-quality Nemotron general and math data. We observe that retrofitting can yield performant, general-purpose depth-recurrent language models that improve with test-time compute via scaled recurrence, outperforming static-depth post-trained baselines on a range of common benchmarks including: GSM8K, ARC and PIQA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates recurrent language models. The main contribution is
- It showed copying weights from pretrained models helps in optimization
- It showed a curriculum over recurrent depths can improve training speed
- It showed TinyLlama and Llama can be converted into recurrent models that have better GSM8K performance compared to vanilla models and can recover basic language modeling performance by a "healing" period.

### Strengths
The paper showed recurrent models can have better param efficiency, achieve better results on GSM8K and recover basic language abilities. The experiments is supportive and the claim is clear.

### Weaknesses
W1: The biggest one is I do not know what is the paper/your method's position on test-time scaling. I am not saying that recurrent models must surpass other methods like RL-based methods, you can even prove that it is sub-optimal. The paper just lacks this part.

W2: The second weakness is that I do not get enough insights from the paper. The fact that copying weights from pretrained LLMs can have advantage compared to training from scratch is also happening in diffusion language models (Dream). Your second contribution is also in the same case, I think the method is already existing for a long time, or it is an engineering trick...

I think the interesting point/insights in my opinion may be why the layers should be chosen in that way for example. The paper now looks like a technical report to me.

The weaknesses below are minor.

W3: How do you compare recurrent models initialized from pretrained-weights to LLMs fairly, for example you should also take flops that used to train LLMs into account? Also, I think you actually need a larger dataset to pretrain the LLM or get a comparable results if training recurrent models from scratch, which is also a hidden cost.

### Questions
Q1: I am curious that what on earth is the difference between: CoT, continuous CoT (Coconut, you do not discrete the vector to tokens), recurrent models (if viewing intermediate recurrent results as CoT) and looped transformers. Are some concepts the same?

Q2: What is the support for the flops calculation. I do not find any experiments or theory that support it.

Q3: It is weird that you trained on fewer recurrence but have better extrapolation (TinyLlama) and I am not satisfied by your explanation. Also, I am curious about the behavior of intermediate choice of #recurrence.

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
3

### Summary
This paper shows that introducing latent recurrence into pretrained language models can improve the reasoning performance. The main contribution is a set of techniques that translate a feed-forward transformer architecture (from the Llama family) to the latent recurrent model by Geiping et al. (2025). The techniques include weight transfer from pretrained feed-forward transformer, recurrence scheduling during training, optimizer selection, and language capability through “healing” (i.e., train on natural language modeling dataset). Empirically, the paper shows that initializing the recurrent model with pretrained feed-forward weights yields better accuracy compared to random initialization. Moreover, the resulting recurrent architecture is also more performant in math compared to an equally post-trained feed-forward model.

### Strengths
This is clearly an empirical paper, and the authors did a reasonable job in describing and conducting the experiments. While not particularly strong on the methodological side, the main insight that reuse of pretrained feed-forward weights for latent recurrent networks is practically useful.

### Weaknesses
1. The paper shows that it is beneficial to initialize the weights in a latent recurrent model with pretrained feed-forward weights. However, it is not clear if this approach is overall compute optimal, i.e., FLOPS(pretrain feed-forward)+FLOPS(post-train recurrent) > FLOPS(only pretrain recurrent (maybe for longer)). Hence, we’re still missing a clear compute-optimal recipe for training latent recurrent models.
2. The reasoning results show the feed-forward performances without test-time scaling. It would be beneficial to consider test-time scaling also for these architectures. 
3. The comparison to the baseline (only pretrain recurrent) from Geiping et al. (2025) should be included in the main Table 1, and not hidden in the appendix.

### Questions
I would appreciate it if the rebuttal could address the individual weaknesses.

### Soundness
2

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
This paper investigates how to leverage pretrained LLMs when developing recurrent-depth language models. To do so, the authors use the recurrent architecture from Geiping et al. (2025) and take transformer blocks from a pretrained (fixed-depth) Llama to initalize its transformer blocks, rather than random initialization. In order to make this work, the paper presents a two-phase training regime, where the language model capabilities are "recovered" first before math reasoning fine-tuning. At the start of this training, recurrence depth is progressively scaled up. Results on the target task, GSM8K, show that this recurrent-depth language model outperforms random initialization but more importantly, the original language model from which the transformer blocks were taken. In addition, performance on other tasks is maintained or improved.

### Strengths
1. clear motivation (expensive training of depth-recurrent models) and a practical idea (leveraging heavily trained Llama models)
1. intuitive method for re-purposing transformer blocks from pretrained fixed-depth LLMs.
1.  effective training regime strategy that notably incorporates a recurrence-scheduling curriculum, adapted from recent works.
1. significant amount of ablations (architectural configuration, layer selection, initialization, optimizer, training phases, data mixtures, etc.) that provide justification for the different components of the methodology
1. the presented end-to-end method both (1) improves upon the fixed-depth model from which the transformer blocks are taken and (2) proves more efficient/better than random initialization.

### Weaknesses
Presentation/Paper Organization
1.  The abstract is very insufficient; while concise, it lacks important details that help explain the paper.
1.  The terminology used creates quite a bit of confusion. Notably, the terms "surgery", initialize", "convert", and "retrofit" seem to be used to describe overlapping concepts. For example, retrofit is used to describe the method altogether, but also specifically the retraining part. It took me a long while to understand what was going on because of this. 
1. In general, I would settle a single-framing: either this method can be framed as an efficient way to initialize recurrent-depth language models or as a way to convert fixed-depth language models to recurrent-depth ones. It seems as though the authors try to frame this method both ways concurrently.
1. The organization of Section 3 & 4 can be greatly improved. Section 3 vs. 4 should either be split architecture & initialization vs. training or description of experiments/ablations vs. results.
1. the abstract, intro, and discussion emphasize GSM8K, why? The non-GSM8K results (Table 1) are arguably more impressive. I argue the benchmark evaluations should be presented very differently, and perhaps in their own section.
1. Discussion section is poorly organized and doesn't sufficiently bring together findings/contributions of paper and contextualize them in current landscape

In general, the presentation & framing required numerous reads to fully understand.

*Suggestions/Other:*
1. L51-58 these two sentences are not very clear.
1. Related Works should mention some early-exit and speculative decoding literature.
1. L137 I think the empirical experiments of the layer selection should be emphasized more, as if it feels like a critical piece. I would bring up from the appendix the empirical results & provide some intuition (grounded in literature perhaps?)
1. Figure 6 presents single-phase and two-phase side by side with respect to train step, but this is a bit weird since the two-phase would have been trained for 26 billion tokens prior to 0.

### Questions
1. L47 this seems like a very arbitrary checkpoint? what is the explanation?
1. Even when training on SFT data, it is treated as unsupervised data for CPT, correct?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies more efficient training of depth-recurrent models (Geipeng et al. 2025). Depth-recurrent models are based on the intuition of test time compute but instead of the model 'thinking' in discrete tokens it 'thinks' in continuous space. 

The authors study how to use a pre-initialized non-recurrent model to initialize training for a depth recurrent model showing:

- that it gives significant computational advantages compared to starting from scratch.

-using a curriculum to increase the recurrence depth over training is beneficial

-Finding that Muon is a more effective than AdamW in the authors' use case.

-A two stage training pipeline (where model first trains on FineWeb-Edu before the general mixture), which the authors find benefits the depth recurrent model training (probably due to the 'healing' required after the model surgery when initializing)

Overall I found the paper well written and practically useful.

### Strengths
Nice practical study on an important problem given the excitement around depth recurrence and test-time compute in the community. 

A solid set of experiments and ablations are provided and I think this paper will be a useful reference to practitioners in the field.

### Weaknesses
-I found the terminology, 'Tiny Llama' and 'Llama' to be confusing. I think it would be clearer if 'Llama' also had a prefix to make it clear there are two different models.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
