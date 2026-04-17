# Learning Communication between Language Models through Dense Vectors

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Communication between language models plays a crucial role in the inference process of large language models (LLMs), occurring both iteratively within a single model for multi-step reasoning (auto-regression) and interactively across multiple models for collaborative intelligence. While current systems primarily facilitate such communication through natural language, this paper proposes a novel paradigm of using continuous dense vector in continuous space. Our approach eliminates the unnecessary embedding and de-embedding steps when LLM interact with another, enabling more efficient information transfer, fully differentiable optimization path, and exploration of capabilities beyond human heuristics. We place such stripped LLMs as vertexes and optimizable seq2seq modules as edges to construct LMNet, a directed graph with similar structure as MLPs, and performs end-to-end gradient-descent for efficient optimization. As two exemplar applications, we show the proposed method can effectively improve LLM's general intelligence, and customizing LLM with limited data. We also provide detailed discussion and analysis about learning communication through dense vectors.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LMNet, a graph-style architecture where multiple pre-trained LLMs are stripped of their embedding/de-embedding layers to form vertex transformers, which communicate via small trainable edge seq2seq modules carrying dense vector messages. The authors instantiate LMNet with shared vertex weights and end-to-end autoregressive training, then evaluate two settings: (i) “general intelligence” improvements using a 1.1B-parameter LMNet built from Qwen2.5-0.5B, trained on public data, and (ii) data-limited customization where edges are trained and compared to PEFT baselines like LoRA on MMLU, GSM8K, and E2E. Results show sizable gains over Prompt/SFT and competitive performance versus similarly sized monolithic LLMs.

### Strengths
- Recasts inter-model communication as learned dense messaging rather than natural-language tokens, enabling end-to-end optimization across models and edges; the layer-wise fully connected topology + edge translators is interesting. The overall idea is conceptually novel. 
- Method is well specified: vertex/edge definitions, aggregation by sum, and a training recipe that first optimizes edges, then all parameters. 
- Evaluated on diverse benchmarks. 
- Provided case studies for de-embed intermediate states to probe what’s carried on the “wires”.

### Weaknesses
- One set of experiments I believe is missing is that the performance comparison between different width and depth of the LMNet, the results will be more convincing if there is plot showing Num of vertexes v.s. Performance, and showing that the performance positively scales with the vertex network size. 
- One stated motivation (replace inefficient NL messages in multi-agent systems) doesn’t really match the implemented setup (single final decoder; interior modules pass only the prompt sequence). What they’ve actually built/benchmarked is much closer to stacked, cross-connected transformer blocks that exchange dense features before any token is produced, not agents sending complete messages to one another.

### Questions
- Typo: line 172 missing a space between ‘single’ and ‘X’.
- In Line 217-219: the author mention an alternative where each vertex could auto regressively generate multiple token embedding sequence, by didn’t specified how. In the normal LLMs, such decoding is controlled by the EOS token so the LLM knows when to stop the autoregressive process, I am curious how to do that without the decoding of EOS in the intermediate layers? 
- A fair comparison shouldn’t only be about training compute; it should also hold test-time compute fixed. LMNet’s per-token inference effectively runs the vertex transformer N (i.e., number of vertexes in the net) times, so a base model given an equal test-time budget could use test-time scaling tricks to spend similar compute and might close some of the gap. The paper acknowledges LMNet increases inference latency per token roughly with layer depth L (sequential) even if same-layer vertices parallelize, but it doesn’t benchmark compute-matched inference baselines.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents LMNet, a new method for learning communication between LLMs besides exchanging natural language token. The key idea is to strip out the embedding and de-embedding layer and treat those as vertices, and connect them with trainable seq2seq edge modules, forming a fully connected directed graph that is differentiable e2e. 

The authors demonstrates two applications, showing increase in both general intelligence/reasoning task (MMLU, GSM8K etc.) and they also showed that LMNet can be used to train with customized, small scale data.

### Strengths
1. Reimagining LLM communication: the paper makes a very clean observation that current multi-LLM systems still talk in discrete natural language, even though the models internally think in dense vectors. That forces every intermediate module to do an unnecessary (and non-differentiable) de-embed → embed step, which is bad both for information efficiency and for gradient flow. The proposed fix is easy to understand. It also clearly distinguishes this work from latent CoT papers, which stay inside a single model’s latent space, whereas this paper explicitly targets inter-model communication.

2. Nice empirical signal: The cost is well-justified and the performance gain on benchmarks like MMLU and GSM8K. This justify for the better information flow that LMNet is designed for.

### Weaknesses
1. Limited ablation: The design of stacking a fully-connected seq2seq module is not studied in depth and seems a bit arbitrary. It remain unclear what the source of gain is. The authors should conduct ablation studies on different topology and putting different capacity of edge modules. Could you show that the design of the specific components of the architecture is actually useful? Like replacing seq2seq module with pure MLP/using a sparse topology.

2. Parameter size as confound and scalability issue: In the main experiment the author points out that the LMNet variant of Qwen-0.5B actually become 1.1B and so this intuitively feels like the performance gain is almost guaranteed. Although the authors compare to models with similar size, it isn't an apple-to-apple comparison with non-Qwen models as for many benchmarks the base Qwen-2.5-0.5B is already better, and versus Qwen2.5-1.5B it's still quite lagging on MMLU and GSM8K. Also, maybe I'm understanding this in a wrong way but does that mean for larger models (say 70B), the LMNet variant would become even bigger?

3. Communication not studied: The main experiment used the same model on the vertices but communication between LLMs are usually with different sharer and receiver. I'm wondering if this architecture generalizes to different models on vertices.

### Questions
See weaknesses.

### Soundness
3

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
2

### Summary
The paper proposes LMNet, a method for connecting multiple language models through continuous dense vectors instead of discrete tokens, treating LLMs as vertices in a differentiable graph with trainable edge modules for communication. The key innovation is removing embedding/de-embedding layers between consecutive LLMs, allowing hidden states to flow directly between models through trainable "edge" modules. The authors construct a directed graph architecture where vertices are stripped transformers (without embedding layers) and edges are small seq2seq modules (typically single attention blocks). To reduce parameters, they employ parameter sharing where all vertices use the same pre-trained LLM.

The paper demonstrates two applications: (1) improving general intelligence by training a 1.1B parameter LMNet based on Qwen2.5-0.5B vertices, achieving ~40% relative performance gains with <0.2% additional training cost as claimed in the paper, and (2) data-efficient domain adaptation, where LMNet outperforms fine-tuning and latent reasoning methods on MMLU, GSM8K, and E2E benchmarks.

### Strengths
- The research question in this paper is interesting. The framing of computational graphs is novel and enables end-to-end gradient-based optimization.
- The paper demonstrates improvments on general tasks, math reasoning, knoweldge benchmarks and domain adaptation.

### Weaknesses
- LMNet is compared against its base model, making it unsurprising to gain performance improvement. Would be better if a model with comparable size is compared. Llama3.1-1B is provide in the paper, but its performance is even worse than the base model which should be problematic.
- The 5-layer architecture also introduce signifcant lower inference compared to a single pass.
- Performance on some benchmarks such as ARC-C, GPQA shows minimal gains or even degradation compared to Qwen2.5-1.5B.
- Passing hidden states between models is not new considering latent reasoning. But it is a bit overstated by claiming as "beyond human constraints". It's important to provide concrete evidences that dense vector communication can beat prompt-based approaches.
- The performance on communications between two different models are unknown especially these with different hidden dimensions.
- What happens at larger models such as 7B models? Does parameter sharing still work?

### Questions
Please see weakness

### Soundness
2

### Presentation
2

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
This paper proposes a new paradigm for communication between language models by directly exchanging dense continuous vectors instead of natural-language tokens. The authors construct a directed graph, LMNet, where each vertex is a language model and each edge is a trainable seq2seq mapping that learns how to translate hidden representations between models. The entire structure is optimized end-to-end via gradient descent. The paper argues that this dense communication removes redundant embedding/de-embedding steps and enables differentiable multi-model cooperation. Two illustrative applications are presented: (1) enhancing general reasoning ability and (2) customizing models with limited data.

### Strengths
The idea of representing inter-model communication as differentiable dense vectors is conceptually novel and could inspire new research on multi-agent LLM systems.

The proposed LMNet graph abstraction provides a potentially unifying framework for studying information flow between models.

The paper touches on an interesting question of whether model interactions must occur through natural language at all, which is intellectually provocative.

### Weaknesses
The motivation for removing the token layer is unconvincing. Tokenization does not necessarily cause semantic loss, while replacing discrete tokens with dense vectors may introduce additional noise, instability, and loss of interpretability.

No clear empirical evidence demonstrates that dense communication improves performance, efficiency, or convergence compared with existing approaches (e.g., natural-language interaction, hidden-state distillation, or adapter-based transfer).

Experiments are limited to small-scale toy settings without strong baselines or ablation studies, making it difficult to assess generality or practical benefit.

The paper lacks theoretical analysis of communication capacity, robustness, or scalability.

Overall, the work feels more like a conceptual proposal than a rigorously validated method.

### Questions
Can the authors provide quantitative comparisons showing improvements in reasoning accuracy, efficiency, or resource usage versus token-based communication?

How stable and interpretable are the learned dense communication vectors across tasks or model sizes?

Have the authors analyzed whether the introduced dense mapping modules amplify noise or reduce robustness?

### Soundness
2

### Presentation
2

### Contribution
1
