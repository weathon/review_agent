# ThinK: Thinner Key Cache by Query-Driven Pruning

- Decision: Accept (Spotlight)
- Scores: 8, 5, 8, 8, 5

## Abstract
Large Language Models (LLMs) have revolutionized the field of natural language processing, achieving unprecedented performance across a variety of applications. 
However, their increased computational and memory demands present significant challenges, especially when handling long sequences.
This paper focuses on the long-context scenario, addressing the inefficiencies in KV cache memory consumption during inference. 
Unlike existing approaches that optimize the memory based on the sequence length, we identify substantial redundancy in the channel dimension of the KV cache, as indicated by an uneven magnitude distribution and a low-rank structure in the attention weights.
In response, we propose ThinK, a novel query-dependent KV cache pruning method designed to minimize attention weight loss while selectively pruning the least significant channels. Our approach not only maintains or enhances model accuracy but also achieves a reduction in KV cache memory costs by over 20\% compared with vanilla KV cache eviction and quantization methods. For instance, ThinK integrated with KIVI can achieve $2.8\times$ peak memory reduction while maintaining nearly the same quality, enabling a batch size increase from 4$\times$ (with KIVI alone) to 5$\times$ when using a single GPU. Extensive evaluations on the LLaMA and Mistral models across various long-sequence datasets verified the efficiency of ThinK.  Our code has been made available at https://github.com/SalesforceAIResearch/ThinK.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper propose a new way to prune KV cache that focuses on reducing least significant channels. The method is query-dependent and reduces KV cache size upon previous KV compression methods while maintaining almost the same quality with previous methods. It also discovers the fact that activated key cache is sparse for a given query.

### Strengths
1. Very extensive benchmark scenario. Evaluated with models from 7B to 70B on 10+ datasets. The accuracy drop is indeed negligible.
2. The method proposed is simple and works on top of previous methods. This could be a nice addition to memory-efficient transformer inference.
3. Writing is easy to follow and method description is clear.

### Weaknesses
1. Method is dependent on query but did not show online pruning delay costs. It would be nice to show a few numbers on long the pruning takes.
2. Did not discuss enough for value cache. This is understandable since value cache is indeed harder to compress according to my own experience. However, you should add more discussion on the implication with this. For example, you claimed that you have 2.8 peak memory saving, does this include saving with V cache? More discussion is needed on this thread to make readers better understand the use scenario of your method (this has nothing to do with the validity of your key cache pruning, it is just unclear to me how much actual benefit comes with your method and I am worried of readers may over-estimate your gain). 
3. KIVI is not your contribution. When you present the result in abstraction, please report clearly your improvement SEPARATELY. What is your gain above KIVI? Compressing K cache by 60% without accuracy drop is a very good contribution already!

### Questions
1. Could you address the questions in weakness?
2. I think you should be more modest about your improvement. Your method is solid and could be impactful (ex. you potentially could also help reduce computation needed in attention calculation by reducing K cache channel dimension). I suggest to make it clear about what part is your contribution and what part is the previous methods (ex. in abstract), and dig deeper into why your improvement in this paper could benefit inference system-level performance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes ThinK, a novel KV cache compression technique that uses query-driven channel pruning for keys and values. Specifically, channel importance is established through $|| Q.K^T ||$ magnitude, and channels with the highest scores are selected. Extensive experiments over the Llama and Mistral family of models demonstrate a 20% reduction in KV cache size when ThinK is used in conjunction with existing KV cache compression techniques.

### Strengths
1. The paper is well-written and concise, which makes it relatively easy to follow. 
2. The authors conduct an extensive set of experiments (along with ablation studies) to show performance improvements in the form of reduced KV cache footprint as well as the possibility of enabling larger batch sizes through their technique.

### Weaknesses
1. Authors overlook relevant baselines [1-3] which also reduce KV cache size by focusing on the hidden dimension (hence effectively pruning channels). Comparison with these techniques is important to put this work in context with other such KV cache compression techniques.

2. The performance benefits are not very significant since the authors show key cache compression by up to 40% and value cache up to 30%, leading to only ~20% improvement in KV cache memory footprint. 

[1] Saxena et al., "Eigen Attention: Attention in Low-Rank Space for KV Cache Compression.", ArXiv 2024.

[2] Yuan et al., "ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models", ArXiv 2024.

[3] Chang et al., "Palu: Compressing KV-Cache with Low-Rank Projection", ArXiv 2024.

### Questions
1. Which dataset was used to find channel importance as shown in Section 3.2? Is a pretaining dataset like WikiText a good fit for establishing channel importance for a range of downstream tasks, or task-specific data is necessary? It would be helpful to understand the generalizability of channel importance score S.

2. Is query-driven pruning a better metric than SVD (as used in Eigen Attention [1]) to find channel importance? Comparison with such works can further clarify the superiority and/or limitations of ThinK.

3. (Suggestion) A pruning ratio ($\lambda$) of 0.4 implies a 40% reduction in the key cache, which means a 20% reduction for the KV cache. This should be specified in Section 4.2.

4. Can the authors provide some discussion on why it's harder to compress the value cache as compared to the key cache (Table 1-7 vs Table 8 in Appendix)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a novel pruning method named THINK to reduce KV cache memory size for long-context inputs. The authors observed a gap in optimizing the channel dimension for KV cache parameters. They show that THINK can reduce memory size by 20% with minimal accuracy loss.

### Strengths
- Timely problem
- Novelty
- Solid experimental results

### Weaknesses
- Comparison with related work

### Questions
Thank you for submitting to ICLR 2025. The research problem is both interesting and timely. The motivating examples are helpful in identifying existing research gaps in channel pruning. The solution is well-reasoned, resulting in significant memory improvements.

What are the authors' reasons for choosing the three baselines?

It would also be helpful to see the results of THINK combined with existing methods on the sequence length dimension and the layer dimension.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents ThinK, a query-dependent K cache pruning method for long-context LLM inference. ThinK takes advantage of the channel-wise sparsity property of K cache in existing KV cache tensors, and uses a greedy algorithm to remove insignificant channels and to optimize for the loss in attention weights after pruning. ThinK can be used together with many existing token eviction techniques (H2O) and KV cache compression techniques (KIVI). Experiment results show that on long-context benchmark tasks, ThinK achieves a comparable level of accuracy when used jointly with H2O or SnapKV, as compared to using H2O or SnapKV alone, but lower memory consumption.

### Strengths
`+` Channel-wise sparsity for K cache isn't a new thing, but this is (probably) one of the first papers that present a solid method for channel-wise K cache pruning. 

`+` Compatibility with existing token eviction or cache compression methods. 

`+` Solid and extensive evaluation. I am convinced by tables 2 and 3 that ThinK is good at detecting and pruning insignificant channels in the K cache, since the accuracy drop is very low even for a pruning ratio of 0.4 or 0.5, so this is a reasonably designed method for K cache compression.

### Weaknesses
`-` Unfortunately, V cache typically demonstrate token-wise sparsity, so this method cannot be generalized to V cache. To some people, this might seem like a smart engineering hack for a specific type of tensors with known distribution properties. **(Clear: The reviewer is clear with the presentation of section #2 after the rebuttal period)**

`-` The presentation of evaluation results and the choice of evaluation metrics are very concerning to me. There is an overwhelmingly large focus on accuracy numbers (Tables 2, 3, 4, 5, and 6), but one of the most important benefits enabled by KV cache compression/pruning is increased throughput and decreased TTFT --- These two metrics do not even appear in the evaluation. Note that reduced memory usage (Figure 3a) is only an intermediate benefit you care about: Eventually, the hope is that reduced memory usage can enable larger batch size, and consequently larger throughput. Though Figure 3b has some numbers on batch size, it does not show how ThinK has the potential of boosting inference throughput. **(Clear: The authors presented updated numbers on latency / throughput during the rebuttal period)**

`-` Minor issues: Typos on line 188 ("pruning matrix"->"pruning metrics"?); weird sentence on lines 214-215 ("Based on the observations ..."). **(Clear: The authors fixed these minor issues during the rebuttal period)**

### Questions
Please refer to "weaknesses".

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents a KV cache optimization method with channel wise pruning for reducing the inference cost under long context scenario. The method is based on the previously proposed efficient context compression method ( H2O, KIVI, and SNAP). Specifically, the paper utilized a query driven method to involve the query for key channel selection.

### Strengths
The proposed method is simple and straight forward, with direct structured pruning on the key channel.
The evaluation shows the pruned results achieves comparable results with the non-pruned baselines

### Weaknesses
Lack of experiments compared to directly applying to vanilla models.
Why the proposed method is basing on the pruned method, instead of directly on the vanilla model? The observations come from the fact of that there are large outliers alongside the channel. This not an unique attribute on the compressed model (i.e. H2O). Yet, experimenting on the compressed model introduces confounding variables which affect the analysis. The current manuscript lacks experiments directly apply the method on the Vanilla model.

In the meantime, the current manuscript lacks the details about how the K cache selected K cache is removed. This is important in order to consider the real memory benefits. Is the K cache fully removed? Or full of the KV cache are always stored in memory? If the selected channel of the K cache is removed, when new queries come, how to upgrade to the new selected K cache size? If the whole K cache are always stored in memory, how is the memory efficiency comes from?

It is also worth to indicating whether the compression is done starting at the prefilling phase, or the prefill phase is concrete. The experiments of many the previous approach (i.e. H2O) did not compress during prefill phase. Detailing it will make the understanding of the experimental results more comprehensive.

Typos:
Sec 3.1 line 149 KV cach
Line 175 k is bold?

### Questions
See weakness section.

### Soundness
3

### Presentation
3

### Contribution
2
