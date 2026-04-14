# Theory, Analysis, and Best Practices for Sigmoid Self-Attention

- Decision: Accept (Poster)
- Scores: 5, 6, 6

## Abstract
Attention is a key part of the transformer architecture. It is a sequence-to-sequence mapping that transforms each sequence element into a weighted sum of values. The weights are typically obtained as the softmax of dot products between keys and queries. Recent work has explored alternatives to softmax attention in transformers, such as ReLU and sigmoid activations. In this work, we revisit sigmoid attention and conduct an in-depth theoretical and empirical analysis. Theoretically, we prove that transformers with sigmoid attention are universal function approximators and benefit from improved regularity compared to softmax attention. Through detailed empirical analysis, we identify stabilization of large initial attention norms during the early stages of training as a crucial factor for the successful training of models with sigmoid attention, outperforming prior attempts. We also introduce FLASHSIGMOID, a hardware-aware and memory-efficient implementation of sigmoid attention yielding a 17% inference kernel speed-up over FLASHATTENTION2 on H100 GPUs. Experiments across language, vision, and speech show that properly normalized sigmoid attention matches the strong performance of softmax attention on a wide range of domains and scales, which previous attempts at sigmoid attention were unable to fully achieve. Our work unifies prior art and establishes best practices for sigmoid attention as a drop-in softmax replacement in transformers.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper gives an in-depth analysis of sigmoid attention for use on transformer architectures. It shows theoretically that sigmoid attention based transformers have the universal approximation property and that sigmoid attention has a better local Lipshitz constant than softmax based attention, namely that the former has Lipshitz constant that scales quadratically and that the latter has Lipshitz constant that scales exponentially. The authors conduct a variety of ablations and experiments giving an in depth analysis of when and where sigmoid attention should be used. Furthermore, they develop a hardware aware based implementation on sigmoid attention termed FlashSigmoid that has a much faster inference than the well known FlashAttention2.

### Strengths
**Originality:** The paper gives a good in depth analysis of the use of the sigmoid function as an activation for attention and provides an original perspective of how it should be used. I applaud the authors for taking the time to give such a detailed analysis.

**Quality:** The quality of the paper is good clearly showing that the authors have taken the time to give the reader a clear understanding of sigmoid based attention mechanisms in transformer architectures. Furthermore, they undertake a variety of experiments giving the reader insights into how sigmoid attention performs on practical tasks.

**Clarity:** The paper is well written and easy to follow with lots of details provided in the appendix. 

**Significance:** I believe the real significance of this paper is in understanding how sigmoid attention compares to softmax attention and when and where it should be used, which I believe is something that is not found in current literature. Furthermore, the authors do develop a hardware aware implementation of sigmoid attention that does much better than the current state of the art for softmax attention.

### Weaknesses
**Novelty:** The main issue I have with the paper is its novelty. Although the authors give an in depth analysis of sigmoid attention I don't see it adding value to the community as their results often show that it only does comparable to softmax. The main novelty of the paper I feel is that they develop FlashSigmoid but I feel this is more of an engineering feat and is not enough for a paper to be accepted into ICLR. I applaud the authors for their in depth analysis and their various ablations but I am still questioning whether there is real quality in the paper. Please see my questions below. The proof of UAP is very nice but is essentially a slight change of the proof given by Yun et al., which the authors do say themselves. Furthermore, the authors provide worst case Jacobian bounds which I found very interesting. In section 3.2 they show that the Lipshitz constant of sigmoid attention grows quadratically w.r.t a ball of radius R and that softmax grows exponentially. This suggest that the gradients of softmax attention should explode causing issues with model convergence, yet I don't see this in their experiments. Softmax often performs on par with sigmoid. Thus I don't see how this theoretical analysis transfers into a useful statement for the reader to gain insight into transformers using sigmoid attention over softmax. I think it would be helpful if the authors could tell me in a few sentences what is novel about sigmoid attention apart from FlashSigmoid?

**Experiments:** I found some of the experimental results in the main paper a bit confusing as the authors tend to over do it with ablation after ablation. It would have been much better for the authors to put most of the ablations in the appendix and keep the main experimental results in the main paper. I felt this clouds the other experiments. For example, it might be fine to just put 6 of the graphs in the section 5.1 in the paper and then put the other 6 in the appendix. Similarly, in figure 7 and 8 I feel it would be enough to just put one of them, say layer scale in the main paper, and put qk_norm results in the appendix. This would free up space for you to really talk about in what way you see sigmoid attention yielding a better attention mechanism for the community to use.

Also in figure 2 I can't really see any difference between softmax and sigmoid. In figure 11 the authors compare sigmoid with layer scale and QK norm with other activations and I notice some of the other activations like GeLU and ReLU do on par with sigmoid on the vision tasks. Does this mean for vision tasks activations like ReLU and GeLU are just as good as softmax so therefore sigmoid is not offering any benefit? 

My overall feeling is that this paper would make a great journal paper as it gives a very in depth analysis of just one activation and its use for attention based mechanisms. However, I will be willing to change my mind if the authors could answer my questions below.

### Questions
1). Putting aside your FlashSigmoid hardware aware attention. Could you explain what is the real benefit of using sigmoid over softmax? Is there any real benefit in terms of training/performance? Could you also clearly explain the disadvantages of using sigmoid attention?

2). I am very interested in your regularity results in section 3.2. I thank you for providing an in depth proof in appendix D. I found this result rather paradoxical though. Your proof shows that the Lipshitz constant of sigmoid attention scales quadratically, see line 159, and it is known that the Lipshitz constant of softmax scales exponentially, see line 162. This implies that sigmoid attention has much better Jacobian regularity and softmax has very bad Jacobian regularity. Wouldn't we see this in training in that it should therefore be very difficult to train transformers with softmax activation as the backpropagated weights would explode as they grow exponentially? However, we don't see this in practice. Most transformers use softmax as an activation and have no real problem with training. For example, most ViTs use softmax and their models can converge. Yet looking at figure 2 for example, I don't see this. Could you comment on this? I wonder if other architectural mechanisms like skip connections is in some sense negating the effect of the exponential scaling of the Lipshitz constant of softmax attention and that is why we still see such transformers being able to train well. Are the other parts of the transformer architecture somehow mitigating the bad Lipshitz scaling that softmax attention has?

3). I noticed in various experiments, see figure 2, table 2, table 3, that sigmoid attention is comparable to softmax attention. However, since sigmoid attention has a Lipshitz constant that has much better regularity than that of softmax (quadratic over exponential) shouldn't we see sigmoid attention doing much better than softmax?

4). You mention in your abstract, see line 023 to 024, that previous attempts at sigmoid attention have not been able to achieve results on par with softmax. However, in the main paper I don't see you comparing to any of those previous attempts. Could you cite those previous attempts for me so I can go and check those papers? Did you compare your methods with those papers? I think this would help your case as it would show just how your methodology does in comparison to those previous attempts you mention in the abstract. The paper is very dense so if this is somewhere in the appendix and I have missed it then I do apologize.

5). In section 5.5 on Autoregressive large language modelling, you mention in line 508 that a slight disparity between sigmoid and softmax is observed at 1B scale with sequence length 4096 which you are able to address using a hybrid-norm by adding an additional normalization layer. I checked appendix G.3 but was not able to follow the explanation. Could you explain why you need extra normalization and in what way it helps the efficiency of sigmoid attention? As the Jacobian of sigmoid attention has quadratic regularity I would expect that much less normalization should be needed for a transformer architecture employing such an attention layer and rather you would need it on the transformer using the softmax based attention. Also, what is the parameter increase by adding an extra normalization layer and does this add any practical disadvantages to sigmoid attention for such tasks?

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
4

### Summary
This paper revisits sigmoid activations in attention mechanism and point out that the main problem with sigmoid attention is that of large initial attention norms. Then they propose solutions with in-depth theoretical and empirical analysis. Also, authors introduce FLASHSIGMOID, a hardware-aware and memory-efficient implementation yielding 17% inference kernel speed-up over FLASHATTENTION2. Furthermore, authors demonstrate that performance across language, vision, and speech are comparable with softmax attention.

### Strengths
**Originality:** This paper provides in-depth mathematical analysis of sigmoid attention specially in Universal Approximation Property and Regularity. Also, authors identify stabilization of large initial attention norms during  the early stages of training as a crucial factor for sigmoid attention.

**Quality:** This paper provides detailed mathematical proofs and very extensive experiments and ablation study on sigmoid attention including supervised image classification, self-supervised image representation learning as well as automatic speech recognition and auto-regresive language modeling. The performance is comparable with softmax attention. Furthermore, I appreciated the implementation of FLASHSIGMOID for efficiency and speedup

**Clarity:** The paper is written very well and easy to follow. 

**Significance:** Challenging softmax and understanding how activation works in attention mechanism is appreciated. Softmax attention is computationally cost and authors revisit sigmoid activation function to speedup inference time. I believe this paper will inspire the community to rethink attention mechanism.

### Weaknesses
1. In Sec 3.2, the authors state that the Lipschitz constant provides insight into the robustness of the network and the ease of optimizing it. They then present a theorem stating that, in $\mathbb{R}^2$, the local Lipschitz constant of SigmoidAttn is much lower than the worst local Lipschitz constant of SoftmaxAttn. This suggests that sigmoid attention should be easier to train than softmax. However, this is inconsistent with the experiments presented by the authors in Fig. 2, Fig. 3, and Fig. 4.

2. Table 2 shows that sigmoid attention causes unstable training, whereas softmax does not. I believe a theoretical analysis of training stability is necessary.

### Questions
1. It would be helpful if the authors could make Fig. 2 clearer.

2. It would be better if the authors conducted experiments on large-scale language models (around 7B parameters).

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper provides theoretical and empirical analysis for sigmoid attention. The theoretical contribution is the proof that transformers with sigmoid attention are universal function approximators. Authors also empirically linked the training instability of original sigmoid attention to large  attention norm, and proposed  a few options to improve it (switch to Alibi, or adding attention bias. They also did ablation study of the effect of layerScale and QK normalization on training stability. Since sigmoid attention is theoretically more expensive than regular soft-max, authors proposed new Flash-sigmoid  - efficient implementation for GPU. Combining multiple techniques they demonstrated that sigmoid attention can potentialy replace the  classical softmax attention on multiple tasks without loss in accuracy or speed.

### Strengths
The paper is very detailed analysis of sigmoid attention vs softmax attention. The sigmoid attention is not very original, but authors did a solid job by exploring its theoretcial foundations and execution solid ablation study to decide if sigmoid attention is viable replacement for  soft-max attention . 
 
The first theoretical contribution - the proof that sigmoid attention can be used as universal approximation to continuous function- is executed well, but has limited novelty, and not very interesting . For me the most interesting theoretical  contribution was the analysis of regularity for sigmoid attention. Btw, I think  the proof of Theorem 2 is missing the key assumption for reqularity is that  ||Wq, Wk"|| and ||Wv|| should be limited. So we have to  make special measure during training to control or normalize  them. 

The experimental part (Section 5) is very good. Most interesting parts 
1.  observation that training of sigmoid-attention based transformers is very sensitive to initial norm of |sigma (QK)"|v| . 
2. demonstarion that combination of layer-scale with  QK norm can help to  stabilize training in some cases
3. very solid ablation study (CV, ASR, LM) 
 
I also liked that authors spent time and effort to rewrite Flash-attention to support Flash-sigmoid and its description in Appendix F. This clearly helped to speed-up whole model (Step-time in Table 3) 

The paper is well writen and it is easy to read.

### Weaknesses
The main weakness of the paper that it doesn't answer main question  "Why should we switch from original softmax attention to sigmoid attention"?: 
- will Sigmoid-Attention  be more stable than original Softmax Attention? 
- will it help with long context? 
On the positive  side, thanks to flash-sigmoid we can see some speed-up for LLM inference (Table 3) 

More details: 
The paper started with explanation that original "softmax in SoftmaxAttn is not without limitations. For instance, the softmax function can sometimes lead to a concentration of attention on just a few features (Yang et al., 2018; Ganea et al., 2019), potentially neglecting other informative aspects of the input data." As far as I remember Yang's 2018 paper was about soft-max attention work as low-rank factorization, and Ganea's 2019 was about last soft-max layer.  How this is related to sigmoid-attention? 
The paper says that the second reason to explore sigmoid-attention was " applying SoftmaxAttn requires performing a row-wise reduction along the length of the input sequence, which in the case of efficient attention kernels (Dao et al., 2022; Dao, 2023), slows down computations." You will still need gather operation along previous sequence to sum  V vectors scaled with attention weights over previous keys, so it's not clear how using  sigmoid attention can do noticable difference in this case (see e.g.  "Online normalizer calculation for softmax" by M. Milakov, 2018 for detailed analysis)
 
The first theoretical contribution -- the proof that sigmoid attention can be used as universal approximation to continuous function--  has limited novelty. This is mostly  re-write of original Yun's 2020 proof.  The main  addition is  the proof that sigmloid-based transformers can do contextual mapping (Appendix C2)  . The observation in C1 that sigmoid can approxiamte Heaviside step function when lambda --> infty is  obvious.  

The analysis of computational complexity is based on computing flops. But the real complexity should take into consideration memory access (read/write ) instead. 

A few comments related to the experimental part: 
-  it looks like the proposed remedy ( bias b=-10  only partially helps to control the  increase in the amplitude of |act * V} (graph 6) 
- ASR: sigmoid (RoPE with bias) performs badly on long audio ( Table 2) 
- LLM: Sigmoid (ALibi) underperform vs Softmax (Alibi) . No results for Sigmoid (RoPE)

### Questions
Very solid experimental study, but I am still not convinced with paper conclusion: "Our findings establish sigmoid attention as a viable alternative to [soft-max]. 
1) is it worth to  switch from original softmax attention to sigmoid attention? 
2) will training Sigmoid-Attention be more stable than original Softmax Attention?

### Soundness
3

### Presentation
3

### Contribution
2
