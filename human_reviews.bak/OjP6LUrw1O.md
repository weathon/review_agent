# From GaLore to WeLore: How Low-Rank Weights Non-uniformly Emerge from Low-Rank Gradients

- Decision: Reject
- Scores: 5, 3, 6, 6

## Abstract
Modern Large Language Models (LLMs) are composed of matrices with billions of elements, making their storage and processing quite demanding in terms of computational resources and memory usage. Being significantly large, such matrices can often be expressed in low-rank format with potential to relax resource requirements. Unlike prior works which focus on developing novel matrix decomposition algorithms, in this work we first study the emergence of low-rank structures across matrices within different layers of LLMs and establish a consequential relationship between the gradient dynamics and emerging low-rank expressiveness of matrices. Our findings reveal that different layers exhibit varying levels of converged low-rank structure, necessitating a non-uniform rank reduction across them to minimize performance drop due to compression. In view of that, we present Weight Low-Rank Projection (WeLore) that unifies weight compression and memory-efficient fine-tuning as ONE, in a data-agnostic and one-shot way. WeLore capitalizes the heavy-tail distribution of singular values to identify a suitable rank reduction ratio for matrices within LLMs. Going beyond only as a compression technique, WeLore categorizes weight matrices into Low-rank Components (LRCs) and Non-Low-rank Components (N-LRCs) based on their ability to express themselves as low-rank. Our gradient perspective and extensive experiments illustrate that LRCs tend to have better finetuning capabilities and can closely mimic (sometimes outperform) the training loss trajectory and performance of full-finetuning with notable memory and compute footprint reduction. For example, finetuning a 50% compressed LLaMa-2 7B model using only a fraction of parameters in LRCs (WeLore) can outperform its full finetuning with ~3x better throughput and ~0.6x GPU requirement.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper establishes a connection between gradient dynamics during pretraining and weight spaces to understand the degree of low rankness of different layer weights. This can be useful for assigning ranks to each layer's weights in an adaptive and more informed manner. The authors unify the goal of memory-efficient finetuning and weight compression by showing that layers with consistent low-rank gradients also result in low-rank weights (termed LRCs in the paper). The experiments show that training only the LRCs can be competitive or even outperform full finetuning for the Llama family of models.

### Strengths
1. This paper establishes a link between the gradient dynamics during pretraining and the rank of the weight matrix for a pre-trained model. This relationship leads to adaptive low-rank weights across layers and memory-efficient finetuning.
2. Extensive experiments show that starting with a 50% compressed model, Welore achieves better performance than LoRA, GaLore and full finetuning.

### Weaknesses
1. The new observation in this paper i.e. link between low-rank structure within gradient during pretraining and the low-rankness of resulting weight matrices is interesting but it is difficult to understand its significance for adaptive low-rank estimation across layers. WeLore uses a normalized singular value thresholding technique to decide layer-wise ranks, but it's unclear why one would need access to pretraining information (gradient dynamics) for this. Singular Value Decomposition (SVD) on a weight matrix can directly reflect the singular value distribution, which can then be compared across layers. It makes sense to compress the layers with a heavy tail distribution more, and I am not sure one needs access to gradient dynamics to arrive at that conjecture and establish LRCs and N-LRCs. 

2.  In Section 4, the authors establish that finetuning LRCs offers memory efficiency because the gradients are inherently low rank for LRCs. However, it is not well concluded as to why freezing N-LRCs during backpropagation closely mimics the performance of full finetuning and even outperforms it. Deciding to train only LRC components offers memory-efficient finetuning but does not explain the performance being akin to full finetuning. The results in Figures 1 and 8 show full finetuning with a 50% compressed model, which in my understanding does not use WeLore style adaptive rank selection. If this is true, then it is expected that such a model has higher perplexity than the WeLore model even before finetuning, and further finetuning a better model (i.e. WeLore) would make it even better. It would be interesting to see results with actual full-finetuning and understand the performance difference between that and Welore.

3. The results presented in 5.2.2 (Table 4) are somewhat confusing. Llama-2 7B has a perplexity of about ~7 on the C4 dataset, but most numbers in this table show extreme degradation compared to that (for example, almost 100% increase in perplexity at 50% reduction with Welore, and worse for higher compression). Table 4 shows a perplexity comparison between LoRA and Welore-based continual finetuning, but Welore tends to require higher GPU memory which defeats the purpose of efficient finetuning. If I try to compare iso-memory, ideally I should be comparing LoRA 1st column (8.21 PPL) with WeLore 3rd column (17.87 PPL) at ~27,000 MB memory.

### Questions
1. Point 5 in Section 5.2.1 is not well explained. How does the non-uniform rank reduction ratio identified through WeLore translate to SVD on activation spaces? Does this originate from the connection between activation and weight spaces through gradient dynamics? Further reasoning for this transferability would be well appreciated.

2. In Section 5.1, the authors mentioned that they use Llama-130M to understand the relationship between gradient dynamics and its impact on the weight space of the pre-trained model. Does this property hold across a family of models, or does one always need access to pretraining information to understand which weights are LRCs or N-LRCs?

3. There are some phrases/terms in the paper which are ambiguous. For example, on page 4 the authors use "early-bird saturation property and can’t accumulate rich error signals". Are the authors linking the rank dynamics of layers with their importance during pretraining? 

4. Why are all the full finetuning/LoRA results on a 50% compressed model? It is important to understand how Welore performs in context to the standard LoRA setting, where Wo is the full rank frozen weight matrix.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors introduce a novel non-uniform matrix-decomposition-based model compression method and a continual memory-efficient fine-tuning method based on the compressed model. They propose that the non-uniform distribution of singular values in gradients leads to a corresponding non-uniform low-rank structure in the weights.  The method categorizes weight matrices into low-rank components and non-low-rank components, and sets the low-rank ones as trainable. The authors validate their theoretical findings through extensive empirical evaluations across multiple datasets and model architectures, showing that the proposed method outperforms the uniform and OWL-based decomposition methods approaches in terms of both computational efficiency and model performance.

### Strengths
- A novel and effective non-uniform rank determination method is introduced, leveraging straightforward linear search, followed by the proposal of an efficient low-rank decomposition technique.
- The informative weights are effectively identified based on the proposed method, as evidenced by the decomposition and fine-tuning experiments.
- The core of non-uniform decomposition is elucidated through the lens of gradient dynamics, thereby enhancing interpretability.

### Weaknesses
- This paper explores the emergence from gradient dynamics to rank determination, yet it lacks a clear linkage between gradients and ranks. If I understand it correctly, the rank determination function does not rely on any gradient information, as indicated by the equal 1.
- Employing the same SVD decomposition method, the main experiment, as depicted in Table 1, omits a SOTA decomposition method baseline, such as SVD-LLM. The current baseline of uniform and OWL-based methods alone is inadequate for a comprehensive effectiveness evaluation.
- The proposed fine-tuning method is not a general fine-tuning method, relying on the proposed decomposition for informative matrix identification. The correlation between the two proposed methods appears somewhat rigid.  The fine-tuning experiments can serve as a dual verification for the informative matrix identified by the previous method, but not a sole fine-tuning method.
- Certain figures lack clarity, for instance, the similar colors employed in Figure 7.

### Questions
1. What is the primary challenge addressed in this paper: rank determination or the relationship between gradient dynamics and rank determination?
2. This paper claims that “first study the emergence of low-rank structures across matrices within different layers”, yet, to my knowledge,  there has already been some research about non-uniform rank allocation research, such as :
    - [Feature-based Low-Rank Compression of Large Language Models via Bayesian Optimization]
    - [MoDeGPT: Modular Decomposition for Large Language Model Compression]
- Could the proposed fine-tuning method be applied to fine-tune uncompressed models or models compressed by other methods? If not, it could still serve as a post-calibration method, which remains valuable.
- It would be even better if there were more in-depth research and description on gradients and compression.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Weight Low-Rank Projection (WeLore) that captures the heavy tail distribution of singular values to identify the suitable rank reduction ratio for matrices in LLMs and only updates the parts of the model with low rank structures. To be specific, they categorize the matrices into Low-rank Components (LRCs) and Non Low-rank Components (N-LRCs) through the normalized singular value thresholding technique. Their method only backpropagates through LRCs and outperform its full-finetuning with less gpu consumptions.

### Strengths
1. The paper is generally well written and easy to read.
2. The idea of analyzing weight low-rank subspace through the lens of gradient behavior and adaptively applying low rank structures to the matrices is new.
3. They empirically show their method outperforms other baseline methods with various LLMs and tasks.

### Weaknesses
1. Can the authors clarify if WeLore can be applied during pretraining, or if it is limited to fine-tuning scenarios since this method requires the pretrained checkpoint to choose which matrices can be categorized as LRCs? If it is limited to fine-tuning, how does this compare to methods like Galore that can be used during pretraining?

### Questions
1. Please fix some typos. For example, in line 179, meerged -> merged. Also, for section 5.2.1, there are total 6 points but no numbering for 4th one.
2. Why does Welore consume less GPU memory than Galore in Table 4 for LLama2-13B 30%?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper starts by empirically studying how low-rank structures emerge "differently" across various layers in LLMs during training. Based on this analysis, the authors introduce WeLore, a unified and data-agnostic weight compression and memory-efficient fine-tuning method, by categorizing the weight matrices into low-rank components (LRCs) & non-low-rank components (N-LRCs). Finally, WeLore outperforms its full finetuning baseline with 3x better throughput and ~0.6x GPU requirement in their Llama-2 7B fine-tuning experiments.

### Strengths
I personally enjoyed the analysis in Section 2 most. There are many things to be studied in training dynamics and gradient properties of LLMs, which can be used for improving LLM training, and I found analyses in this paper fall under this category. Indeed, authors design the new algorithm (WeLore) based on their analysis, and validate its effectiveness on various experiments including compressing pre-trained LLMs and parameter-efficient fine-tuning.

### Weaknesses
The authors stated that "GaLore theoretically argues that the gradient matrix becomes low-rank during training but does not establish how the gradient behavior accumulates in the weight space". While I appreciate their empirical analyses, having some theoretical argument on their weight space analysis can be helpful. I believe that pre-training experiments with WeLore would make the paper even stronger, I am reluctant to lower my score, as I understand many researchers don't have enough computational resources.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
