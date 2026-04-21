# Your Mixture-of-Experts LLM Is Secretly an Embedding Model for Free

- Avg Score: 6.67
- Decision: Accept (Oral)
- Scores: 8, 6, 6

## Abstract
While large language models (LLMs) excel on generation tasks, their decoder-only architecture often limits their potential as embedding models if no further representation finetuning is applied. Does this contradict their claim of generalists? To answer the question, we take a closer look at Mixture-of-Experts (MoE) LLMs. Our study shows that the expert routers in MoE LLMs can serve as an off-the-shelf embedding model with promising performance on a diverse class of embedding-focused tasks, without requiring any finetuning. Moreover, our extensive analysis shows that the MoE routing weights (RW) is complementary to the hidden state (HS) of LLMs, a widely-used embedding. Compared to HS, we find that RW is more robust to the choice of prompts and focuses on high-level semantics. Motivated by the analysis, we propose MoEE combining RW and HS, which achieves better performance than using either separately. Our exploration of their combination and prompting strategy shed several novel insights, e.g., a weighted sum of RW and HS similarities outperforms the similarity on their concatenation. Our experiments are conducted on 6 embedding tasks with 20 datasets from the Massive Text Embedding Benchmark (MTEB). The results demonstrate the significant improvement brought by MoEE to LLM-based embedding without further finetuning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper explores the use of the routing weights of a mixture-of-expert LLM for embedding tasks. Concatenating those routing weights over layers gives them surprisingly good scores. The paper claims that the routing weights are more robust to varying prompts than the contextualized representations and that the routing weights complement the standard contextualized representations well for embedding-centric downstream tasks. The experimental part covers tasks from the MTEB benchmark, e.g., on textual similarity, classification, and clustering. The results shows that the combination of routing weights with hidden states are preferable over the current best-practice of just using the hidden states of language models for embedding-centric tasks.

### Strengths
The idea is original, and the experimental results confirm its effectiveness. The MoEE technique (essentially combining hidden states with routing weights) scores higher in 5 out of 6 tasks and gives equal performance on the 6th task. This hints at the approach of making use of MoE routing weights could be generally considered for embedding-centric tasks. The paper thoroughly tests the contribution of hidden states and routing weights to the overall performance (Sec 3.2). The paper further explores different variants of dealing with multiple sequences, i.e. concatenation or calculating and interpolating cosine similarity separately for hidden states and routing weights. Here the results show that the sum variant is in most cases preferable. Generally, a strength of the approach is that no further training is required, but that those embeddings just need to be extracted from a language model. The paper further presents an ablation study comparing different token/layer selection strategies for the hidden states and routing weights -- interestingly here, routing weights already perform better than the hidden states.

### Weaknesses
To some extent the paper relies PromptEOL, i.e. prompting the model to distill the meaning of a sentence into one word. While this is clearly stated in the paper already, it could potentially hint at further limitations; for example, when a large context would need to be compressed into a single word.

Although the baseline selection is solid, a few more commonly used models for embedding-centric tasks could be employed. SentenceBERT, ColBERT, & BertScore come to my mind, as those are commonly employed for text similarity and such tasks.

### Questions
Do you have any insights how the length of the sequence has an influence on embedding quality?

Given last-layer last-token HS performs best in the ablations, I assume that is what you use for the main table. Is this correct or did I miss anything?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper first shows that RW and HS of MoE LLM are complementary for embedding-focused tasks. And they proposed two method to combine these two representations, concat and sum respectively. They benchmark the performance of the method on 6 embedding tasks of MTEB and show that it can outperform both RW and HS, as well as other previous baselines. In addition, they demonstrate the robustness of the method to different prompts.

### Strengths
1. This paper explores the utility of the RW and HS components of MoE LLMs for embedding-focused tasks, an area that has not yet been explored by others. 
2. They have comprehensive experiment on 6 embedding tasks of MTEB with three different MoE LLMs and the method is stable across different prompts.
3. The logic is easy to follow and understand.

### Weaknesses
1. Although the authors have compared their method to previous approaches, the baselines used are somewhat outdated. For instance, the latest baseline of this paper is published in 2021. This a old topic or authors just missed some latest baselines ( I think this could limit the contribution of this paper to the field.)?
2. The method seems not sensitive to the scale of MoE LLM, could author provide some explanation?
3. Though authors demonstrate the stability of the method compared to HS, it is not comprehensive without stability of baselines in the main experiment.
4. It seems the performant variant of their method "sum". And "sum" depends on the hyper parameter alpha. Could authors provide more details of how to tune alpha? Is alpha consistent across all experiments?

### Questions
1. Could author provide more recent baselines, cause the best baseline in this paper from 2021?
2. Could author explain their method is not sensitive to the scale of models?
3. Could authors also report the stability of baselines from the main experiments?
4. Could authors provide more details about how to tune alpha and is it consistent across experimentation?

### Soundness
3

### Presentation
3

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
The paper investigates the potential of Mixture-of-Experts (MoE) large language models (LLMs) as effective, training-free embedding models by leveraging their routing weights (RW) in addition to the commonly used hidden state (HS) embeddings. The main contributions include:
- **Complementary Embeddings**: The authors discover that RW embeddings capture high-level semantic features and nuances that HS embeddings may miss, making RW a valuable, complementary embedding source. RW is also found to be more robust to prompt variations than HS.
- **MoEE**: Based on these findings, the authors introduce MoEE, a method combining RW and HS embeddings. They evaluate two strategies: concatenating RW and HS and a weighted sum of their similarities. The weighted sum approach performs better, balancing output-specific information from HS with the broader, input-sensitive features from RW.
- **Performance on Benchmark Tasks**: Experiments on the Massive Text Embedding Benchmark (MTEB) across tasks like classification, clustering, and semantic textual similarity demonstrate MoEE’s effectiveness, consistently outperforming RW and HS alone. MoEE shows particular advantages in tasks requiring deep semantic understanding, even surpassing some fine-tuned embedding models when prompt tuning (PromptEOL) is applied.

The findings suggest that combining RW and HS in MoE models offers a rich, stable embedding approach, expanding the utility of LLMs in embedding tasks without the need for further fine-tuning.

### Strengths
- The paper introduces an innovative use of routing weights (RW) as a standalone language model embedding, presenting a fresh perspective on embedding capabilities in Mixture-of-Experts LLMs.
- The authors conduct extensive experiments to underscore the complementary nature of hidden states (HS) and routing weights (RW), providing thorough insights into their distinct contributions.
- The empirical evaluation is well-rounded, demonstrating consistent performance improvements when HS and RW are combined, showcasing the effectiveness of the proposed approach across diverse embedding tasks.

### Weaknesses
- While MoEE introduces two methods for combining HS and RW embeddings (concatenation and weighted sum), the concatenation variant appears simplistic and less effective than the weighted sum in terms of similarity calculation. Future work could explore more sophisticated aggregation methods to fully leverage the complementary strengths of HS and RW.
- The claim that RW embeddings are more robust than HS, based solely on prompt variation tests, lacks comprehensive support. Other factors, such as model size (or maybe architectural variations), should be examined to substantiate this claim.
- The choice to evaluate on only a subset of the Massive Text Embedding Benchmark (MTEB) raises questions about generalizability; it would be helpful to understand the criteria behind this selection and whether other tasks or datasets might yield different insights.

### Questions
Following the weakness, I have the questions below
- **Aggregation Methods**: Given that the weighted sum variant outperforms concatenation for similarity calculations, have you considered more complex aggregation techniques? Could alternative methods further enhance the complementary benefits of HS and RW embeddings?
- **Robustness of RW Embeddings**: In claiming that RW embeddings are more robust than HS, have you explored robustness across other dimensions such as model size or architectural variations? What other factors do you think could influence this robustness?
- **MTEB Task Selection**: Could you provide insights into why only a subset of MTEB tasks was chosen for evaluation? Were there specific criteria for this selection, and do you anticipate different results if other tasks or datasets were included?

### Soundness
3

### Presentation
3

### Contribution
2
