# DeCAL Tokenwise Compression

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
This paper introduces DeCAL, a new method for tokenwise compression.  DeCAL uses a native encoder-decoder language model pretrained with denoising to learn to produce high-quality compressed representations from the encoder.  DeCAL applies small modifications to the encoder, with the emphasis on maximizing compression quality, even at the expense of compute.  We show that DeCAL at 2x compression can match uncompressed, with usually only a minor dropoff in metrics up to 8x compression, among question-answering, summarization, and multi-vector retrieval tasks.  DeCAL distinguishes itself from prior works by supporting larger chunks and significantly more compressed output per chunk.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces DeCAL, a tokenwise compression framework that leverages encoder–decoder pretraining with denoising to produce compressed representations without sacrificing downstream performance.DeCAL embeds a learnable latent sequence within the encoder that cross-attends to the evolving input tokens layer by layer, enabling gradual, information-preserving compression. Through systematic experiments across summarization, question answering, and multi-vector retrieval, the authors demonstrate DeCAL's performance.

### Strengths
1. The usage of average pooling as the starting point for compressed tokens is interesting.

### Weaknesses
1. The work has questionable experiment settings. What are the "baseline" in most of the tables? I assume it's T5, so why don't the authors provide comparison with other token compression works (which can be easily extended to the encoder-decoder setting) such as [1]?
2. L317 - L326, the author mention briefly three token compression methods and claim to use them as baselines, yet no further information is provided to show how they are extended to the paper's experiment setting and only 1 single "baseline" is referenced in the result table. This raises concerns about the clarity on how experiments are conducted.
3. At the end, the goal of most token compression techniques nowadays is to handling long-context tasks, could the authors provide more experiments results under this setting?
4. Why does the work limit the scope of the method under encoder-decoder baselines? Can we extend it to decoder-only architecture by training an additional encoder like [1]? If not, please provide a clear justification.

[1] LLoCO: Learning Long Contexts Offline, Tan et. al., EMNLP 2024.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new tokenwise compression technique called DeCAL. It follows a T5 architecture, prepending a sequence of “pooling tokens” (summed with average pooling of some input tokens) alongside the input, to extract a compressed version of the input via bidirectional self-attention. This compressed input sequence is fed to a decoder. It follows pre-training on a span denoising task, and finetunes on several QA, summarisation and retrieval tasks. It shows good downstream performance on those tasks with compression, with a performance with 2x compression matching the performance of the “uncompressed” baseline.

### Strengths
- The paper summarizes well the different related works on compression of token sequences.
- The paper is easy to follow.
- The paper present a significant number of experiments: from pre-training with different compression ratios, experiments with others methods like AttnPool or the custom CCProxy approaches, to a number of finetuning experiments on several tasks including Document-based QA, Summarization and Retrieval. 
- The paper presents a few ablations on modelling choices in section 4.4
- The paper shows good results in comparison with the baseline, while using high compression ratios, on many tasks

### Weaknesses
The paper method lacks of novelty:
 - Extracting compressed representation with self-attention has been widely used in other works (cited and presented by the authors) like COCOM with CTX tokens as well as ICAE with memory-tokens.
 - Span denoising tasks with encoder-decoder models has also been widely studied

Novelty lies in the combination of these two, but a reader would expect a more thorough analysis of why this method could be superior to previous work, what component is critical, what percentage of masking is required, are there other denoising tasks that could work better. 

While the authors have included baselines like CCProxy with the same training setup, the paper would greatly benefit from external baselines.

The paper would also benefit from RAG setup experiments with the presented method for bigger potential impact.

A few explanations in text do not seem to match figures/table caption: e.g. Figure 1. "init. using pooled input tokens" while in text the pooled representations are added to learnable vector. Table 6. DeCAL 2x (auto-encode) presented as "denoising replaced by auto-encoding" is presented in the text as "auto-encoding (mixed with text continuation)"

### Questions
Can you provide more details on the CCProxy setting ? ("start from our baseline model" + "decoder-only model" --> Which part of the model do you use?). The training data setting seems different also (128M vs. 26M C4 examples)

Why do you repeat the same learnable vector? Did you try having several learnable vectors? How did you initialize this vector?

Did you evaluate the representations obtained from pre-training (keeping the encoder frozen), while fine-tuning only the decoder on downstream tasks?

Did you try other span masking strategies / ratios?

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
This paper propose a new method to learn compressed representation of text. It considers a setting with an encoder-decoder model, with the goal of reducing the number of hidden representations from the encoder that are fed to the decoder. More precisely, the paper proposes a modification of the T5 encoder-decoder architecture. For a sequence of N tokens, the proposed method, called DeCAL, will compress the sequence to N/C continuous representations, where C is the compression factor. These compressed representations are obtained by adding N/C embeddings to the input of the encoder. These embeddings are obtained by applying mean-pooling to the token embeddings of the input sequence. A learnable embedding is also added to each mean pooled representation. Then, this augmented sequence is processed normally by the T5 encoder. The only modification are the position bias: for the additional tokens, the integer average of the position of the original tokens are used. Then, the output corresponding to the added tokens are used as input of the cross attention of the decoder, which is unchanged. The model is then pre-trained with the same denoising task as in T5, and fine-tuned on downstream tasks such as QA or summarization. The method is then evaluated on summarization, question answering and retrieval tasks. An ablation study is performed to study the impact of pre-training, of the task used to train the model (T5 denoising vs. reconstruction) and different ways to obtain the compressed representations, showing that the different components are important.

### Strengths
Overall, I believe that this paper is borderline regarding acceptance to ICLR.

On the positive side, I think that the paper is well written and easy to follow. The proposed method is simple and sound, and the practical implementation is solid. I enjoyed the ablation experiments, and believe that its conclusion could impact the future development of learning compressed text representation. More precisely, the paper show that:
- training an encoder from scratch leads to better results than fine-tuning a model pre-trained without compression ;
- initializing the compressed representations from pooled embeddings of the text tokens leads to significantly better results than learned parameters only, such as memory tokens that were previously considered (here, it would be interesting to compare to a version with different parameters for each position of the compressed representation) ;
- the experiments show that using this way to obtain compressed representations leads to significantly better results than doing a attention pooling at the end of the encoder, hence showing that using more "compute" to learn the compressed representation leads to better results. I am wondering how this results is linked to the fact that the model is pre-trained in this way, and if the result would still be true in the context of fine-tuning only.

### Weaknesses
But on the other hand, I also believe that the paper suffers from some limitations.

First, as currently proposed in the paper, the DeCAL method is strictly more computationally expensive than the T5 baseline, or other methods used as comparison in the paper. This limits the usability in practice of DeCAL, and I think that it would make the paper stronger to discuss (and potentially explore) how this could be useful in real world settings.

A second limitation is that, I believe that a different model was fine-tuned for each downstream task. I think it would have been more interesting to explore if a single model could learn to compress text representations for multiple downstream tasks. I think that this is important, as the information needed for different downstream tasks could be quite different. Thus, learning a single model that compresses well for many tasks is potentially more challenging than learning to compress individually for each task. I believe that most previous work learn a single model for multiple tasks.

Finally, I feel that the comparisons to previous work is a bit weak. There has been multiple papers exploring how to obtain compressed representations, and this paper only compared to a "Proxy" method, where the authors chose the key elements of previous work. To me, this comparison feels more like an ablation than a comparison to existing methods. In particular, the setting considered in the paper is quite different from what is considered by others (see previous point), potentially making the comparison unfair.

### Questions
For the CCProxy baseline, did you use a single learnable vector shared for all the positions?

Did you try the DeCAL method in a multitask setting?

Additional references:

- J. Mu, X. L. Li, N. Goodman. Learning to compress prompts with gist tokens, 2024
- J. Tang, Z. Zhang, et al. GMSA: Enhancing context compression via group merging and layer semantic alignment, 2025
- S. Eyuboglu, R. Ehrlich, et al. Cartridges: Lightweight and general-purpose long context representations via self-study, 2025
- M. Louis, H. Dejean, S. Clinchant. Pisco: Pretty simple compression for retrieval-augmented generation, 202

### Soundness
3

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
5

### Summary
Authors propose a compression method based on encode-decoder architecture and showed the evaluation of the method on several benchmark datasets. They have performed ablation on compression ratio and also compared with other methods.

### Strengths
- performed experiments on a several benchmark tasks
- shown variations on different compression ratios
- comapred with other compression methods
- paper is well written and easy to follow

### Weaknesses
- The comparson is limited to CCProxy and AttnPool 2x. The justification for CCProxy hyper-parameters are not good enough (variations in the models should be used, ablation needed). The existing methods like NUGGET show much higher performances. Also the experiments are shown on two datasets only. Ablaton needed.
- What if the T5 as a basemodel is changed? is this method generalizable to other architectures? Ablaton neeed. 
- for summarization/ tasks, more useful metrics should be used

### Questions
- What creates the unstability in training? 
- What variations are possible in token selection?

### Soundness
2

### Presentation
3

### Contribution
1
