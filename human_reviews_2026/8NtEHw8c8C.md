# Probing Neural Topology of Large Language Models

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 8, 4

## Abstract
Probing large language models (LLMs) has yielded valuable insights into their internal mechanisms by linking neural activations to interpretable semantics. However, the complex mechanisms that link neuron’s functional co-activation with the emergent model capabilities remains largely unknown, hindering a deeper understanding and safer development of LLMs. In this work, we introduce graph probing,
a method for uncovering the functional connectivity of LLM neurons and relating it to language generation performance. By probing models across diverse LLM families and scales, we discover a universal predictability of next-token prediction performance using only neural topology, which persists even when retaining just 1% of neuron connections. Strikingly, probing on topology outperforms probing on activation by up to 130.4% and 67.7% on perplexity and space/time semantic regression respectively, suggesting that neural topology contains orders of richer information of LLM performance than neural activation, which can be easily extracted with simple linear or MLP probes. To explain the dependence between neural topology and language performance, we identify default networks and hub neurons in LLMs and provide causal evidence by interventional experiments on multiple benchmarks, showing that LLMs actually exploit these topological information. Further analyses suggest that neural topology can be effectively leveraged to improve the efficiency, reliability, and safety of LLMs through proof-of-concept applications in model pruning, hallucination detection, and LLM fingerprinting. Codes and data for the graph probing toolbox are available at https://anonymous.4open.science/r/llm-graph-probing-71BD/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new method for probing, inspired by methods in neuroscience. Using this new method, which they term *graph probing*, one computes the pairwise Pearson correlations between neurons across timesteps, and feeds this into a linear or MLP probe. Using this method, the authors are able to more successfully probe for quantities like model perplexity on a sequence, as compared to standard probes. Interpreting the values in the correlation matrix as a graph, they also find hub neurons (those with high degree), and show that ablating these disproportionately hurts model performance, indicating their functional importance. They finally use these probes for pruning, hallucination detection, and model fingerprinting.

### Strengths
This paper is well-written and polished: the text is clear, and the graphs are well-constructed. I also think that this particularly style of probing is novel: I haven't heard of others using it before.

### Weaknesses
**Unclear Purpose of Proposed Method**: The primary use of probing in interpretability is, as noted by this paper, "linking neural activations to interpretable semantics". But this paper never actually does this, which is the one thing that one would expect from a probing paper. In fact, since it's restricted to sequence-level probing, it can't do many probing tasks, which are often token-level. 

Instead, it probes for things like perplexity (why would you want a probe to predict that?) and uses probes to identify important neurons (we know that such neurons exist, and don't need probes to find them), identify hallucinations (this is useful, but there's a lot of past work to compare to), and model fingerprinting (same issue). Ultimately, an interpretability method should help us understand better how models work, but I don't think this paper does so; instead, it focuses on applications that appear useful on the surface, but are hamstrung by insufficient baselines / connection to prior work. In doing so, it ends up in an awkward position where it doesn't make a very novel or significant contribution to either model interpretability or capabilities. 

**Weak Evaluation / Insufficient Baselines**: This paper doesn't perform enough / the right baselines, making its results less sound:
- For the probing for perplexity experiments, the baseline probes only receive the last token representation. You should give them something that combines information from all of the positions; it doesn't make sense to baseline against a method that receives so much less information in its input.
- For the pruning experiments, why not compare to any other pruning methods? If the point of this method is to be useful, then it should really be more useful than alternatives.
- Ditto for the hallucination experiments.
- Ditto again for the model fingerprinting experiments. I'd also love more discussion about how / whether the Pythia experiments really reflect a real-world fingerprinting use-case. 

**Relation to prior work**: This paper actually cites a lot of prior work! But I think that it misses out on some crucial work that would provide context. For example: 
- this paper stresses the finding of so-called hub neurons, but it is well-known that neurons vary in importance with respect to individual tasks; see [Vig and Belinkov](https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html) who find that just a few neurons matter for a gender bias task. For papers looking at this from a more general POV, see [Timkey et al](https://aclanthology.org/2021.emnlp-main.372/), who investigate neurons whose activations are very high-magnitude, and [Dettmers et al](https://arxiv.org/abs/2208.07339), who find that not quantizing these neurons is important, as doing so hurts model performance. It follows fairly naturally that zero ablating these - a much more destructive intervention than quantization - will hurt model performance as well. The authors push this "hub neuron" / "default network" narrative very hard, but these findings are already known, and the neuro connection seems forced.
- Where is the related work on pruning / hallucinations / model fingerprinting? For the first two cases, there has been tons of interp work to compare to. 
- Re: probing, please cite work from before 2022! There is a lot of thoughtful probing literature that predates probing in mech interp; indeed, a lot of that probing literature is better than what has emerged recently. You can look at [BERTology](https://aclanthology.org/2020.tacl-1.54/) for classic probing papers, but see [Hewitt and Liang](https://aclanthology.org/D19-1275/) for some thoughts on how to do probing, and e.g. [Voita et al.](https://aclanthology.org/2020.emnlp-main.14/) and [Pimentel et al.](https://aclanthology.org/2020.acl-main.420/) for other probing methods. Connecting more to the probing literature might help shed some light on important questions like: why do we do probing in the first place? What are we looking to find, and what would a new method have to bring, in order to be useful?
- I think the authors are much too sanguine re: the results of NeuroAI, and this shows in their review of prior work. Maybe be a bit more cautious in describing its putative successes.

### Questions
- Why do you refer to "neural topology"? I expected something about e.g. the topology of neural activations in the model, but it's actually just correlations between neurons over a given input.

### Soundness
1

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces graph probing, a probing method that treats an LLM’s neurons as nodes in a functional connectivity graph derived from token-time activation. Instead of probing raw activations, the method flattens the adjacency matrix of functional connectivity graph and trains lightweight MLP (and GNN) probes to predict LLM performance metrics such as perplexity. By probing models across diverse LLM families, the paper argues that the technique can predict next-token prediction performance using only neural topology.  They report gains up to 130%.

Beyond correlation, the paper probes causal structure in neural topology. It identifies a stable “default network” of high-degree hub neurons that recur across inputs, then shows that ablating only the top ~1% of degree-ranked neurons causes drastic MMLU accuracy drops.

Finally, the authors showcase practical applications of neural topology, including topology-informed pruning that removes many low-degree neurons with modest accuracy loss for downstream tasks.

### Strengths
I find the paper well written and the empirical results are solid and sound. In terms of the main result, it shows neural topology probes reliably predict next-token performance and outperform activation-only baselines across multiple LLM families/sizes, with careful 8:2 train/test splits and standard regression metrics. The experimental setups are valid.

The paper also shows that the results are robust. In particular, performance persists under heavy graph sparsification and across model sizes and layers, suggesting the signal isn’t a fragile artifact of a single scale. 

Besides correlation, the paper offers a causal and interpretability perspective. The author(s) identify a stable “default network” of hub neurons and show that ablating the top 1% (by degree) yields large MMLU drops. This is stronger than random or activation-based selections, suggesting that models actually use this topology. These insights yield practical applications such as sparsity pruning and hallucination detection on TruthfulQA. 

The paper demonstrates that the technique is general. I appreciate the breath of the evaluations. It considers various model families and scales, including Qwen and Pythia series. In the appendix, it also describes experiments on VLMs by probing different sizes of LLaVA-v1.5. 

To my knowledge, the observations from the paper is novel and I believe it's a nice contribution.

### Weaknesses
While the paper identifies a default network, I find it insufficient in explaining exactly what the network does. In addition, I'd love to see a mechanistic interpretation of “why” topology predicts performance.

If I understand correctly, building the per-sequence correlation graphs requires collecting full hidden-state time series and computing pairwise correlations, then training a probe. Although sparsification helps at inference time, the initial topology extraction can be costly in $n$ and $t$. How large is the computational overhead as we scale $n$ and $t$? Alternatively, it would be great to consider more efficient or approximate version of the algorithm.

Related to that, another minor concern is that the experiments of this paper top out at 14B parameter models. I am a bit worried about the computational cost of the technique. It would be nice to expand how much time / memory it would require to train & apply the probes as we scale model sizes. 

For the Truthful QA experiment, the inputs are constructed by concatenating each question with its true or false answer. In practice, users interact with LLM in a free form way. Could the paper clarify if the their method can help hallucination detection in a realistic setting?

### Questions
I wonder how scalable the technique is. Specifically, the method requires you to construct a n by n adjacency graph. Would this still feasible for, say, trillion-parameter models?

For the pruning experiment, after removing low-degree neurons, did you try brief re-tuning (namely, a bit more fine-tuning) to recover accuracy?

In Appendix A.4, you used 10,000 random text sequences to construct neural connectivity graphs. I wonder if 10k is a universal constant that suffices for any models or scales, or if it should vary as well. In general, what is the sample complexity of training the graph probes? How many sequences are needed for the graph features to stabilize and the training loss of the probes to converge?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This study proposes a novel method called Graph Probing, which analyzes the functional connectivity topology among neurons in LLMs to uncover its deep relationship with language generation performance. By adapting concepts from cognitive neuroscience, particularly functional connectivity and functional brain networks, the work introduces a fresh and innovative perspective to LLM interpretability. Empirical results reveal that LLMs exhibit stable default functional networks and hub neurons, and that the topological organization of these networks significantly influences model performance. The approach also shows promising potential for practical applications, such as model pruning, hallucination detection, and model fingerprinting.

### Strengths
1.The integration of functional connectivity concepts from neuroscience into LLM analysis is conceptually innovative and may contribute to mechanistic interpretability.  
2.The authors have demonstrated that functional connectivity outperforms activation in predicting model performance, i.e., perplexity.
3.The authors have demonstrated the promising potential of “graph probing” for practical applications, such as model pruning, hallucination detection, and model fingerprinting.

### Weaknesses
1.The manuscript is kind of overloaded while the validation of each point seems insufficient. 
2.Some interpretations of the results might be overstated.
3.Experimental results on large scale models is missing.

### Questions
1.The manuscript is kind of overloaded. The authors tried to put too many points in one study, including graph probing, hub neurons, default network, functional specifications and three potential applications. All the points seem relate to “topology”. However, the limit of space makes the validation of each point to some extent insufficient. I would like to suggest the authors to focus on the main findings and elaborate experimental details, results and interpretations.
2.I am curious about the relationship between graph probing and hub neurons. In graph probing, it is practical to identify key connections that contribute significantly to the prediction of model performance (perplexity here). Are the neurons that constitute those important connections align with hub neurons? Or to what extent the nodes in important connections overlap with hub neurons?
3.All reported results are about small-scale LLMs. Given the focus on large models, results on at least 7B-class architectures are essential. Although the appendix mentions experiments on Qwen2.5-14B, no results, metrics, or analysis from this model are reported.   
4.About functional specification: Given extremely high correlation of neural topology (Figure 8b), I not very sure that this results show the functional specification or the default network. Actually, if you don’t put “law” in the middle, or apply a color bar with a wider range, probably the clustering effective would not be that clear.

Minor concerns:
1.The mathematical definition of Neural Topology in Section 2 is incomplete: key symbols such as n (neuron index?) and t (time step? input token?) are not defined, undermining clarity and reproducibility.  
2.A lot of the interpretation in the manuscript might be overstated. For example, but not being limited to:
“This remarkable stability confirms the existence of a default network in LLMs, where a fixed set of hub neurons play dominant roles regardless of the input, suggesting a fundamental organizing principle of their internal structure.” ---The DMN is a quite complex system in the brain. It’s functionality is far beyond “play dominant roles regardless of the input”.
“These findings strongly suggest that LLMs develop distinct and linearly separable topological patterns for different knowledge domains, allowing for easy extraction of the context or subject being processed.”---The experimental results do not support this claim. Separable topological patterns should be demonstrated before drawing such a strong conclusion.

### Soundness
2

### Presentation
2

### Contribution
3
