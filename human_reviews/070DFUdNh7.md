# GraphGPT: Graph Learning with Generative Pre-trained Transformers

- Decision: Reject
- Scores: 5, 5, 3, 5

## Abstract
We introduce GraphGPT, a novel model for Graph learning by self-supervised Generative Pre-training Transformers. Our model transforms each graph or sampled subgraph into a sequence of tokens representing the node, edge and attributes reversibly using the Eulerian path first. Then we feed the tokens into a standard transformer decoder and pre-train it with the next-token-prediction (NTP) task. Lastly, we fine-tune the GraphGPT model with the supervised tasks. This intuitive, yet effective model achieves superior or close results to the state-of-the-art methods for the graph-, edge- and node-level tasks on the large scale molecular dataset PCQM4Mv2, the protein-protein association dataset ogbl-ppa and the ogbn-proteins dataset from the Open Graph Benchmark (OGB). Furthermore, the generative pre-training enables us to train GraphGPT up to 400M+ parameters with consistently increasing performance, which is beyond the capability of GNNs and previous graph transformers. The source code and pre-trained checkpoints will be released soon to pave the way for the graph foundation model research, and also to assist the scientific discovery in pharmaceutical, chemistry, material and bio-informatics domains, etc.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces GraphGPT, a new model for graph learning using Generative Pre-training Transformers. By converting graphs or their subgraphs to token sequences via an Eulerian path and pre-training with next-token-prediction, GraphGPT outperforms or matches state-of-the-art methods on various datasets, for tasks at graph, edge, and node levels, including PCQM4Mv2, ogbl-ppa, and ogbn-proteins from OGB. Its generative pre-training allows training with 400M+ parameters with increasing performance, surpassing the capability of GNNs and previous graph transformers.

### Strengths
- The paper presents a compelling and effective approach to converting graph learning problems into NLP problems, which is an innovative bridge between the domains of NLP and graph learning.
- The method showcased state-of-the-art performance, specifically on the ogbl-ppa link prediction dataset.

### Weaknesses
1. **Discussion on limitations**: The paper does not delve into potential limitations of the proposed method.
2. **Reference gap**: Despite the extensive literature on graph transformers, like the works of Rampášek et al. 2022 and Chen et al. 2022, the paper cites only a few, leading to a lack of comprehensive context.
3. **Computational concerns**: The proposed method, while achieving good performance on some datasets, appears to demand significant computational resources. There is a clear lack of discussion on the resources' requirements and the computation time compared to other graph models.
4. **Transferability issues**: GraphGPT has to be re-trained for each specific dataset, raising concerns about its adaptability across diverse graph modalities or sizes. This constraint, paired with the computational burden of pre-training a large GraphGPT, casts doubts on its practical applicability.
5. **Performance claims**: The paper might be overstating its model's performance. While it excels in the ogbl-ppa link prediction dataset, its performance appears mediocre in both graph and node-level prediction tasks when compared against similar scale GNN/GT baseline models.

_References:_

- Rampášek, Ladislav, et al. "Recipe for a general, powerful, scalable graph transformer." NeurIPs 2022.
- Chen, Dexiong, Leslie O’Bray, and Karsten Borgwardt. "Structure-aware transformer for graph representation learning." ICML 2022.

### Questions
**Q1**: What factors contribute to the model's superior performance in the ogbl-ppa link prediction dataset, yet not replicating similar success in node and graph-level prediction tasks? Some intuitive explanation would be helpful.

I will be happy to increase my rating if the authors can address all my concerns and questions

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents GraphGPT, an adaptation of the Transformer network tailored for graph data. GraphGPT is adept at handling a wide range of graph datasets and various graph-related tasks, including node, edge, and graph prediction. A notable innovation in this work is the use of (semi-)Eulerian paths to transform graphs into sequences of tokens. This transformation is designed to be lossless and reversible, ensuring the integrity of the original graph data. GraphGPT is pre-trained on the NTP task and is fully compatible with the Transformer's decoding architecture. This compatibility allows GraphGPT to fully leverage the benefits of generative pre-training. Empirical results showcased in the paper highlight GraphGPT's performance, achieving near or on par with state-of-the-art results in tasks at graph level, edge-level, and node level.

### Strengths
The paper is well-written and logically structured. The application of (semi-)Eulerian paths is a  novel idea, demonstrating considerable versatility in converting graph structures into decoding sequences. The diverse empirical results underscore the adaptability of GraphGPT, showcasing its ability to yield competitive performance across a range of traditional graph tasks.

### Weaknesses
The empirical results highlight a critical aspect of GraphGPT, especially in terms of its performance relative to the number of parameters. While the approach is versatile and straightforward, making it adaptable for various tasks, I find its substantial size problematic. The graph-level and node-level task performances, in my opinion, don't seem to justify the significantly larger scale of GraphGPT compared to its competitors.

I recognize the primary advantage of GraphGPT as its simplicity in application across different graph tasks. However, I question the rationale for preferring GraphGPT in scenarios other than edge-level tasks. While alternative methods may be more intricate and rely on "tricks," they often prove to be more efficient and effective.

Moreover, I am skeptical of the assertion that GraphGPT is immune to over-smoothing and over-squashing, based solely on the moderate performance improvements observed with increased parameter size. This claim demands further scrutiny, particularly when considering the relatively poor performance in node classification tasks.

I believe that a more comprehensive investigation is required, especially focusing on how node parameters are represented. It's essential to determine whether the modest improvements in performance are indeed due to the avoidance of over-smoothing and over-squashing or whether other factors are at play.

### Questions
see weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes GraphGPT, a novel model for graph representation learning. The key ideas are: 1) Transform graphs into sequences of tokens via Eulerian paths to preserve structure information. 2) Pretrain the transformer decoder with next token prediction on the graph sequences. 3) Fine-tune on downstream supervised graph tasks by formatting them to be compatible with the decoder. Experiments on graph, edge and node classification tasks demonstrate strong performance and consistently improving results when scaling up GraphGPT.

### Strengths
- The graph-to-sequence transformation using Eulerian paths is an elegant way to serialize graphs for the transformer. This avoids complex feature engineering.
- Leveraging generative pretraining enables scaling up to hundreds of millions of parameters, overcoming limitations of GNNs.
- GraphGPT consistently improves with more data and parameters, demonstrating generalization ability.

### Weaknesses
- Performance on some node and edge tasks is not state-of-the-art.
- Limited analysis of what properties are learned during pretraining and their utility.
- Does not experiment with very large models in the billions of parameters range.

### Questions
- For large graphs, how are the subgraph sampling parameters (depth, context size) chosen? Is there a principled way to set these?
- What is the computational complexity of the graph-to-sequence transformation? How does it scale?
- Is the decoder-only transformer architecture sufficient for graph tasks or would an encoder-decoder be beneficial?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes  a novel model for Graph learning by self-supervised Generative Pre-training Transformers. The proposed method includes: 1) transforming the (sub)graphs into a reversible sequence of tokens via the Eulerian path, 2) pre-training a transformer decoder using the NTP task, and 3) fine-tuning the transformer. The paper investigates various graph related tasks: graph-/edge-/node-level tasks

### Strengths
The paper proposed an interesting angle for graph pretraining, which takes advantage of the great performance of the breakthrough of LLM (transformer) into graph learning. The idea of Eulerian path is also interesting. The paper has investigated various graph tasks, including graph-/edge-/node-level tasks; and also consider small/large graph

### Weaknesses
-- Due to high variance of graph benchmakrs, to prove the effectiveness of the proposed methods (graph pretraining & finetuning), I would experct the authors provide more benchmarks results. 

-- There are various graph pretraining methods proposed (i.e. either combined with transformer or not, use contrastive learning or not, etc). Can the author compared with various other pretraining methods? 

-- The ablation study of pretraining is helpful. i.e. table 5. Just curious, are their other ablation study on finetuning?

### Questions
See above.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
