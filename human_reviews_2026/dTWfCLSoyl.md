# In-Place Test-Time Training

- Avg Score: 7.33
- Decision: Accept (Oral)
- Scores: 6, 8, 8

## Abstract
The static "train then deploy" paradigm fundamentally limits Large Language Models (LLMs) from dynamically adapting their weights in response to continuous streams of new information inherent in real-world tasks. Test-Time Training (TTT) offers a compelling alternative by updating a subset of model parameters (fast weights) at inference time, yet its potential in the current LLM ecosystem is hindered by critical barriers including architectural incompatibility, computational inefficiency and misaligned fast weight objectives for language modeling. In this work, we introduce **In-Place Test-Time Training (In-Place TTT)**, a framework that seamlessly endows LLMs with Test-Time Training ability. In-Place TTT treats the final projection matrix of the ubiquitous MLP blocks as its adaptable fast weights, enabling a ``drop-in" enhancement for LLMs without costly retraining from scratch. Furthermore, we replace TTT's generic reconstruction objective with a tailored, theoretically-grounded objective explicitly aligned with the Next-Token-Prediction task governing autoregressive language modeling. This principled objective, combined with an efficient chunk-wise update mechanism, results in a highly scalable algorithm compatible with context parallelism. Extensive experiments validate our framework's effectiveness: as an in-place enhancement, it enables a 4B-parameter model to achieve superior performance on tasks with contexts up to 128k, and when pretrained from scratch, it consistently outperforms competitive TTT-related approaches. Ablation study results further provide deeper insights on our design choices. Collectively, our results establish In-Place TTT as a promising step towards a paradigm of continual learning in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors propose a new method that allows small LLMs to adapt and learn from new information while they are being used, rather than only during initial training. The approach, called In-Place Test-Time Training, updates specific internal components of the model efficiently without needing full retraining. Concretely, it uses the last layer of MLPs in transformer blocks. Further, it also introduces a new learning goal aligned with the model’s main prediction task and uses an optimized update process for speed and scalability. The authors show some experiments in small to moderate size LLMs showing that this method improves performance on long and complex tasks while remaining computationally efficient. Finally, they present a set of nice ablation studies.

### Strengths
This is a very good paper. In particular, I liked the following aspects of the paper:

1) The idea of the paper makes perfect sense, and the problem it tackles is important. Having LLMs that can be adapted at test time, improving their performance, is definitely a step in the right direction towards building strong models.

2) The authors solve the method using two simple key ideas: (1) using only the last linear layers in MLPs of Transformer blocks, (2) using a novel objective that somehow aligns with the task of next-token predictions. In both cases, these idea make complete sense, and are shown to be effective, despite being quite simple.

3) Because, the authors are using only a small subset of parameters, I assume that the method is quite computationally efficient.

4) As shown in several experiments, the method seems to improve over the baseline.

5) The paper is excellently written. I had a good time reading and understanding the paper.

### Weaknesses
1) The main weakness of the paper is that it has been evaluated in a relatively limited setting. Qwen3-4B, while a strong baseline, is still a very small LLM for modern standards, thus it is unclear if the method shows the same effectiveness when used with larger, or even significantly larger LLMs, which tend to be state-of-the-art.

2) Furthermore, it would have been cool if the authors would have checked the effectiveness of their method in other families of LLMs (even if having the same size), for example LLama, Mistral, Intern etc.

3) Finally, would the results hold if going in the domain of VLMs. Such results be it on Qwen3-VL or some version of InternVL would make the contribution of the paper even more solid.

### Questions
It would be nice if the authors are able to show those experiments. I would increase my score in such a case.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a Test-Time Training framework for LLMs which does not require any additional layers / parameters and can be introduced in existing pre-trained LLMS with some training as well as when a model is being trained from scratch. The main idea is to repurpose the down projection component of a GLU styled MLP into fast weights. Compared to previous fast-weight and TTT implementations, they also introduce a separate objective for the fast-weights which is more in tune with the overall LLM task of next-token prediction and is also theoretically motivated. Based on these two core ideas, the authors perform extensive evaluation of their technique, called In-Place TTT, on pretrained LLMs using continual learning as well as training LLMs of different sizes from scratch. They compare IPTTT to full attention, sliding window based attention methods and demonstrate the effectiveness of their scheme.

### Strengths
In my opinion, the core strength of this paper lies in the idea of repurposing the existing weight matrices in a neural network and using them as fast weights. This means their technique can be used to improve context management in existing LLMs as well as train large scale models using TTT. Moreover, the technique uses TTT not at the attention level but at the MLP block level which allows for efficient parallelization of the IPTTT framework using prefix sums and chunking sequences into token blocks. The theoretical analysis seems grounded and correct although I did not verify it fully. The experimental section is also done well and I don’t have any major concerns. The performance difference between their technique and the baseline, especially on long context tasks shows the promise of this technique as well. I believe this work will be useful to the community to build upon.

### Weaknesses
The main issue I have with the paper is certain sections confused me a little bit. There were some language issues, typos, and a few instances of confusion caused by notation. Here are some typos / language improvements I would suggest.

Line 161 -  also viewed  → also be viewed

Line 192 -  onece → once

Line 175 - we adapt only the MLP blocks → we only adapt the MLP blocks 

Also, in the paragraphs on Efficient Adaptation with Chunk-Wise Updates, the target is referred to as V but then later it becomes V^{hat}. Is there a reason for that?

Also, I believe the authors have made bit of an overstatement in the explanation of  barrier (ii) "the canonical per-token update mechanism of TTT is inherently sequential, severely bottlenecking the parallel processing capabilities of modern accelerators. 
And also here : "This chunk-wise update strategy is designed for modern hardware. By processing large blocks of
tokens at onece, it highly leverages parallelism and utilizes the computational power of GPUs or
TPUs, thus resolving the efficiency bottleneck that hinders prior research."

While it is true that Fast-weight like implementations are inherently sequential, chunk-wise training is not a new approach and was already explored / discussed in several fast-weight like papers e.g [1], [2]. I suggest the authors adjust for this.


[1] Titans: Learning to Memorize at Test Time 

[2] : Fast weight programming and linear transformers: from machine learning to neurobiology

### Questions
The major confusion I have about the methodology is understanding how it works during inference. In regular fast weights, the goal is to make sure the fast weights compress the context (the reconstruction objective). But here, it is to have them predict info about the future tokens using V. However, V for future tokens is not available during inference, so how do the fast weights even updated?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a novel in place test time training framework that enables LLMs to dynamically adapt a set of ‘fast’ weights during inference time with new information. The authors provide a way to repurpose the existing final projection matrix of the MLP blocks of the decoder transformers and avoid costly retraining. They also provide an efficient chunk wise update rule and an objective that aligns with the next token prediction task. Lastly they also provide a context parallel algorithm. Experiments on a 4B qwen show that the approach achieves improvements over baseline. Good practical paper, just a few questions below.

### Strengths
1. The method is practical and drop in design that is novel and straightforward to integrate in practical deployments
2. The authors have made the learning objective aligned with next token prediction from a generic reconstruction objective and show ablations to prove their point
3. The authors also additionally provide context parallel algorithms for maximal computation efficiency

### Weaknesses
1. The experimental section is based on a single family of model qwen. What if selecting that specific layer is suboptimal in some other family of models like llama. What will the authors do with a mixture of experts paradigm?

2. The authors provide limited justification for selecting the MLP block’s final projection matrix. There are other MLP blocks with large matrices inside the decoder transformer as well, perhaps the attention layers output projection matrix etc. This seems like a heuristic and an ablation could make it clear which layer is the optimal layer to choose for this purpose. What if a combination of layers provides the best performance. 

3. On that note isn't it possible to do a modified Lora TTT[1]? For example, instead of changing the weights of the model itself directly as fast weights, we keep a set of lora weights that we keep updating at inference time, and in lieu of its small flops footprint, instead of changing the full matrix of only the projection matrix, one can now change a few different matrices across the transformer block in a flop matched regime. Isn't that a reasonable baseline to consider?

References: 
1. Kojima, Yuto, Jiarui Xu, Xueyan Zou, and Xiaolong Wang. "Lora-ttt: Low-rank test-time training for vision-language models." arXiv preprint arXiv:2502.02069 (2025).


3. Some figures like figure 3 are missing legends.

### Questions
see weaknesses

### Soundness
3

### Presentation
4

### Contribution
3
