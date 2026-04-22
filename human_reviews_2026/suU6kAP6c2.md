# Generalized Parallel Scaling with Interdependent Generations

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Parallel LLM inference scaling involves sampling a set of $N>1$ responses for a single input prompt. However, these $N$ parallel responses tend to be generated independently from each other, partitioning compute resources and leaving potentially useful information in one generation untapped by others. This is in contrast to response length scaling where past computation is used in all future steps. For higher quality responses and response sets, we propose Bridge to generate interdependent responses in parallel by rethinking batched LLM hidden states as holistic tensors rather than independent slices. With only a small amount (2.8\%-5.1\%) of new parameters, Bridge improves the relative mean accuracy gains from reinforcement learning with verifiable rewards by up to 39\% and boosts consistency of correct responses. Trained once, Bridge scales to any generation width, all with greater performance than independent generations, unlocking a more general mode of parallel scaling that effectively leverages information between sequences, compatible with any post-generation aggregation technique.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Their method allows the LLM to attend to parallel generations from the same prompt. They do this with a new transformer block that attends to the previous token representation from the other generations. They initialize the layers to have no contribution, and fine tune the new architecture with SFT followed by RL. Their results show that this works better than adding the equivalent number of parameters to the transformer.

### Strengths
Novelty:
Existing methods use multiple LLM generations, but theirs is the first I know of in which other generations influence the model during generation.

Clarity:
The paper was easy to understand, Figure 2 was clear and aided in understanding.

Results:
The results are strong. I found figure 5 especially convincing, since it shows that increasing the generation width helps up to a point.

Significance:
It's common to sample multiple responses from LLMs, so this method is widely applicable.

### Weaknesses
Error bars

None of the plots have error bars and there’s only one run of each method. This makes it hard to tell if the differences are real in some cases where the method results are close to the baseline.

Reproducibility

The code is not provided, and there is no reproducibility statement. This is especially important for this paper since the main contribution is the new transformer layer so open sourcing the implementation would be very useful.

Baselines

They don’t compare with any existing methods. Some existing methods take advantage of multiple generations to improve LLM accuracy, for example self consistency. The authors could compare against self consistency, or show the result of using self consistency on top of Bridge.

Minor clarity issues:

The motivation in the introduction is high level and abstract. It says “Independent generations for the same prompt leave potentially useful information derived from other responses unutilized, limiting the performance ceiling.” but it doesn’t explain what this potentially useful information is. Add concrete examples to help the reader understand the motivation.

The paper doesn’t explain what RL with verifiable rewards is, leading to confusion which don’t already understand it. For example, the paper should explain if Bridge can also work with other types of RL for language models like RLHF, or if it’s only applicable to RLVR.

### Questions
none

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Bridge, a module for LLMs that applies attention between (past) tokens across parallel samples. The proposed method enables interdependence between traditionally independent parallel responses. The block is trained using RLVR and optional SFT warm up. The paper presented experiments showing that Bridge achieves better accuracy on math benchmarks over the base model.

### Strengths
- The paper presents, to the best of my knowledge, a novel idea of enabling attention between tokens across samples in a batch, a dimension that typically maintains independence. 
- The evaluation applies Bridge and baselines to a variety of different math benchmarks

### Weaknesses
- The improvement in accuracy is rather small compared to baselines. Bridge does not perform much better than P-Match, which is the baseline given the equivalent amount of compute and in some cases also not much better than just RLVR only. As such, it seems most of the accuracy improvements are from additional compute/training over the original model rather than from the Bridge block design.
- An important motivation to enable parallelism is to enable lower latency while increasing compute. The evaluation does not present the inference latency of Bridge and baselines
- The evaluation lacks comparison to methods that enable parallelism during LLM decoding (such as those cited in the paper). Several of those methods (e.g. Multiverse, Pasta) already have mechanisms for encoding interdependency between different threads by enabling parallel threads to join together and respawn. The evaluation should evaluate against these methods to compare how Bridge's method of enabling interdependency performs.
- The notation of Eq 1 and Eq 2 are difficult to parse

### Questions
- Is there a latency cost due to the additional computation in Bridge?
- How does Bridge perform compared against using existing methods enabling parallel computation (with and without interdependence)? Even though they eventually join together into a single response, a simple string post-processing step can split the response back into a set of responses by splitting on the special tokens these methods use for spawning parallel threads. How does Bridge compare to doing this?

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes an architectural modification to the transformer architecture to allow parallel generations to communicate with each other during generation. It shows notable improvements in accuracy on math benchmarks.

### Strengths
The paper introduces a novel and clever technique that offers a nice performance improvement on math benchmarks at a modest cost. It offers an appropriate level of technical detail and analysis.

### Weaknesses
The paper only evaluates math benchmarks, so it's unclear how well the approach would work in other domains.

It's not clear that P-Match, rather than any of the alternatives described in the related works section, is the appropriate baseline.

Figures lack CIs.

### Questions
Can you address the concerns around generalization to non-math domains, and appropriate comparison approaches?

What is your hypothesis for why, as shown in Figure 7, norm ratios for Bridge blocks are so low?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors propose a new technique, aptly called Bridge, for post-training language models to perform well at parallel generation. By allowing different sequence positions across a batch to effectively communicate with each other, each independent generation suddenly becomes interdependent. 

The authors evaluate the technique on the canonical math datasets and produce reasonable baselines to compare Bridge against. They find that Bridge leads to improvements in both the pass@k and G-Pass@8_{tau}.

### Strengths
Strengths:
- The paper is well written. Coherent and understandable (albeit a few minor places I would have appreciated a little more context).
- This approach seems reasonable for introducing interdependence across generations.
- Reasoning in LLMs is an important area. AFAIK this is the first paper that takes a serious stab at introducing cross information sharing across parallel generations.

### Weaknesses
Weaknesses:
I left the paper feeling like something was missing. The proposed architecture seems reasonable, but I have no idea for what it’s actually doing for the model? 
- Does the model begin to sample in semantically different directions given context from other models? I.e., does it increase diversity? 
- If one trace is suddenly going in the right direction, do other traces update their reasoning accordingly?

Something as simple as BERTScore (similar to here [1,2]) or some notion of entropy across traces in the different settings could provide some insights into what’s going on.

A preliminary step in understanding what’s going on was done on line [457] but this seems both unsatisfactory and overly short. 

[1] https://arxiv.org/abs/2502.01697 
[2] https://openreview.net/forum?id=gvsdQ72Peg&noteId=gvsdQ72Peg

### Questions
Minor points:
- How did you get the numbers 30%, 50% and 23% on line [323]? I can’t recreate them from the table using Avg col?
- In line [365] I would have appreciated a definition of coverage.

### Soundness
4

### Presentation
4

### Contribution
3
