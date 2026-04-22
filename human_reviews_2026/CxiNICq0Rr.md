# Verifying Chain-of-Thought Reasoning via Its Computational Graph

- Avg Score: 6.50
- Decision: Accept (Oral)
- Scores: 6, 8, 4, 8

## Abstract
Current Chain-of-Thought (CoT) verification methods predict reasoning correctness based on outputs (black-box) or activations (gray-box), but offer limited insight into \textit{why} a computation fails. We introduce a white-box method: \textbf{Circuit-based Reasoning Verification (CRV)}. We hypothesize that attribution graphs of correct CoT steps, viewed as \textit{execution traces} of the model's latent reasoning circuits, possess distinct structural fingerprints from those of incorrect steps. By training a classifier on structural features of these graphs, we show that these traces contain a powerful signal of reasoning errors. Our white-box approach yields novel scientific insights unattainable by other methods. (1) We demonstrate that structural signatures of error are highly predictive, establishing the viability of verifying reasoning directly via its computational graph. (2) We find these signatures to be highly domain-specific, revealing that failures in different reasoning tasks manifest as distinct computational patterns. (3) We provide evidence that these signatures are not merely correlational; by using our analysis to guide targeted interventions on individual transcoder features, we successfully correct the model's faulty reasoning. Our work shows that, by scrutinizing a model's computational process, we can move from simple error detection to a deeper, causal understanding of LLM reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors leverage recent mechanistic interpretability techniques to develop a new white-box method for assessing the internal computations performed by a model while solving simple reasoning tasks. They show that they, using their circuit-based reasoning verification (CRV) outperform considered greybox and blackbox baselines on CoT-step-correctness prediction tasks on three small synthetic and math datasets.

### Strengths
Advancing the field of mechanistic interpretability towards the practically relevant task of model debugging is a timely and highly relevant contribution. This work makes an important first step towards a scalable "white-box" technique for systematically tracing back model failures (in this case reasoning errors) to models' internal states and (hypothesized) learned algorithms.

The work performs thorough evaluations with several decent baselines and including several interesting ablations to understand the strengths and weaknesses of the proposed method. E.g., OOD evaluation shows that the method does not transfer well to unseen reasoning tasks and the feature analysis shows that ablating the topological graph features hurts the performance least. While in particular the second finding is a little bit in tension with the storyline of the paper of exploiting structural properties of the models latent algorithms, it is good scientific practice. 

It is interesting that a mostly straightforward adaptation of transcoder based circuits in the proposed nice framing of graph classification outperforms the considered baselines on partial-CoT-correctness prediction.

The qualitative causal experiment of building CRV-informed interventions to correct LLM outputs shows that methods like this may unlock unexpected downstream usecases.

### Weaknesses
My major concerns with this paper are:
* In the main experiment in table 1 the probing baseline gets outperformed by most of the black-box methods. This suggests to me that the probing setup might be flawed: the datasets seem tiny, was there enough data to train the probes? Probing performance is sensitive to the layer and token position. While it is common to aggregate hidden states over multiple tokens, if token sequences get too long like in CoT this seems like a too simplistic way of probe construction. Maybe it would make sense to consider a simple 1-layer attention based aggregation of the hidden states instead of the mean or a sliding window or something like that. I would be surprised if probing really underperforms simple baselines like the black-box ones as suggested by table 1.
* The authors emphasize that the white-box nature of their approach is important to achieve this level of performance and being able to pick up on structural graph properties were the key factor for the success of the method. However, they also show that gradient boosting based classifier is mostly robust to ablation of topological feature and achieves almost the same level of performance considering only local and global graph features. This to me also suggests that a much simpler grey-box technique, e.g., not requiring a substitute, transcoder-based attribution should achieve similar performance on the considered tasks.
* The approach is termed white-box however its white-box aspects are under-explored. The qualitative CRV-inspired intervention goes into the right direction, however, an in-depth analysis of the white-box aspects of the work would significantly strengthen the contribution. Another thing to consider to add towards the whitebox/interpretability/understanding framing would be to leverage transcoder feature annotations based on max activating examples and so on to help the debugging process. 

Minor concerns: 
* Many things, such as explanation of the circuit computation and baseline methods, are deferred to the related work. It would be nice to convey the main ideas and intuitions in the main paper. If the paper gets accepted I'd like to request a short explanation of the attribution based circuit finding algorithm and some of the baselines considered in table 1.
* Some related work on reasoning model interpretability that is closely related is missing in the related work section. E.g., [1] provide some preliminary evidence that CoT reasoning indeed involves some learned algorithms. The same goes for [2], which reverse engineers a self-verification circuit in a toy-model.

[1] https://arxiv.org/abs/2510.07364v3
[2] https://arxiv.org/abs/2504.14379

### Questions
My overall assessment of the paper would improve if the authors convincingly address the main concerns raised in the weaknesses section.

I would also encourage the authors to provide stronger evidence supporting their claim that the proposed approach is genuinely white-box in nature.

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
The paper demonstrates a great use case of using interpretability techniques for a practical application of detecting invalid cases of chain of thought. In particular, the authors build out an attribution graphs for valid and invalid CoT steps. These graphs in turn can be turned into features for a classifier to classify valid and invalid CoT steps, using graphical features such as global graph statistics, node statistics, or topological features. These features are used to train a Gradient Boosting Classifier. 

These experiments were conducted on carefully crafted datasets, for a boolean reasoning task, arithmetics, and carefully annotated GSM8k. The authors compare to a wide range of baselines, each of which are categorized as "black-box" or "gray-box" methods, in which their approach (CSV) outperforms all baselines.

I appreciate that the authors check for cross-domain generalization of their CRV classifier. It turns out that CRV does not generalize well, suggesting that CRV (and the underlying attribution graph) likely uses specific signatures per domain. It would be interesting to see how to build a CRV model that generalizes well across domains.

The authors also demonstrate that CRV's stay effective regardless of the difficulty of the problem. The authors control for task difficulty with number of steps required to solve the problem, and CRV remains effective as task difficulty is scaled. 

A natural question that arises is what kind of features CRV relies on. By ablating away some features from the attribution graph, the authors identify that "local features" (i.e., node influence) matter the most, while "global features" such as topological features matter less.

The authors dig deeper to look at transcoder features (as opposed to features of the attribute graph) to identify specific concepts. As expected and as seen in numerous SAE work, intervening on these features have causal power to control the CoT reasoning traces. 

Overall a strong contribution that leverages attribution graphs for a tangible application.

### Strengths
Overall the paper is nicely scoped and demonstrates a clear win in leveraging attribution graphs to identify valid and invalid chain-of-thought steps. This provides a practical benefit of recent interpretability methods, which (as exciting as interpretability findings are) don't always happen. I hope to see more work that connects from interpretability insights to tangible actionable applications.

### Weaknesses
Overall the paper make claims/contributions that are well-scoped and nicely backed by their experiments. If I had to list out some weaknesses....

* Figure 3: How much of this visualization is an artifact of projecting high-dimensional features to 2d? The claims of "computational near misses" and "zone of computational integrity" seem a bit hand-wavy based on these visualizations.

* I'm not sure what other baselines would have been possible, but IIUC some of these baselines seem a bit unfair in that some of them don't have as much access to the LM's hidden states? For instance, PPL/MaxProb/PPL are overly simplified scalar summarizations of the model's forward pass. Is that correct?

* Any thoughts on how to scale the diagnoses of identifying relevant features (of the transcoder) and trying causal intervention experiments? 

* I appreciate that the authors tested for cross-domain generalization, although the results seem to indicate that they don't. Any thoughts on how to build a diagnostic classifier that does generalize across domains? 

* The fact that LR probe performed so poorly is a bit surprising. It's also very surprising to see that layer 0 worked best for the LR probe. Why do you think Layer 0 performed best? Unrelatedly, could the poor performance of LR probe be cause of averaging the hidden states across tokens? Lastly, do you think simply using mean differences of hidden states (for correct vs. incorrect cases) could work better?

### Questions
Most of my questions have been asked in the "weaknesses" section.

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
4

### Summary
The paper proposes Circuit-based Reasoning Verification (CRV), a method to detect errors in LLM reasoning. The core hypothesis is that reasoning failures might manifest as detectable "structural fingerprints" on a model's computational graph. To study this, the authors modify a transformer model by replacing each MLP with a transcoder (a sparse, interpretable autoencoder trained to imitate the MLP’s function). Using this replacement model, they construct attribution graphs that capture causal connections between tokens, features, and logits during each step of a Chain-of-Thought (CoT).

From these graphs, CRV extracts a set of structural and topological features (e.g., node influence, connectivity, entropy) and trains a diagnostic classifier to predict whether a reasoning step is correct or incorrect. Experiments on synthetic Boolean and arithmetic tasks, as well as GSM8K, suggest that CRV outperforms black-box (e.g., logit-based) and gray-box (e.g., hidden-state probe) baselines. However, the method does not generalise well across reasoning domains, which the authors interpret as evidence that error patterns are domain-specific.

The paper also uses CRV to analyse a single reasoning failure on a simple mathematical expression and show that causal interventions (modifying a single transcoder feature) can correct the models prediction.

### Strengths
- The paper's core idea of treating the model's computational graph as a debuggable execution trace is compelling. 
- The case study demonstrating the correction of a reasoning error by intervening on a single transcoder feature is interesting.

### Weaknesses
- Prior work found that models’ hidden representations often contain detectable traces of reasoning success or failure (e.g., [1, 2]), with simple probes achieving strong performance in similar settings. In contrast, the LR probe baseline reported here performs notably poorly, suggesting it may not have been well-tuned. This undermines confidence in the relative performance gains claimed for CRV.
- CRV combines two major components: (A) replacing dense MLPs with sparse, interpretable transcoders, and (B) constructing and analyzing an attribution graph. The paper attributes its success primarily to (B), yet it never evaluates a simpler baseline that trains a classifier directly on transcoder activation statistics (A) without graph construction. The ablations in Table 3 suggest that “Node Influence & Activation Stats” make up for most of the performance, implying that much of the signal may come from the sparse representation itself. 

[1] A. Zhang et al., ‘Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification’, in Second Conference on Language Modeling, 2025.

[2] O. Obeso, A. Arditi, J. Ferrando, J. Freeman, C. Holmes, and N. Nanda, ‘Real-Time Detection of Hallucinated Entities in Long-Form Generation’, arXiv [cs.CL]. 2025.

### Questions
NA

### Soundness
2

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
3

### Summary
The paper proposes CRV, circuit based reasoning verification, an approach for classifying whether reasoning traces are correct and where they might have possibly made an error. This is done through tracing an attribution graph of an extended retrained formulation of the original LLM. Experiments on boolean, arithmetic and GSM8K reasoning tasks show better performance compared to black- and grey-box methods.

### Strengths
Conceptual:
- The idea is very nice and approaches the reasoning veracity problem from a much more fine-grained angle than black- and grey-box  baselines. The overall architecture, i.e. transducers, attribution graph and classifiers on activation and other features are a very nice addition to the mechanistic interpretability community, as far as I know of the literature.

Experimental:
- I like the cross-domain experiment. Even though it shows limited domain generalization, it is a highly useful information and shows the authors adherence to an exhaustive evaluation, even when results are not favourable.
- The intervention is a nice idea. However, it is done on only one single data point (one more in the appendix?) and I wonder about whether you can really identify the reason for incorrect reasoning and do interventions more generally.

Writing:
- The writing is very well done and a pleasure to read.

### Weaknesses
Conceptual:
- The method, as acknowledged, is computation heavy and cannot be used without expensive retraining to identify reasoning weaknesses. I think it would be an interesting follow-up question whether such a white-box method could be directly grafted on an existing LLM.
- Some questions on the fairness and experimental setup and comparison to baselines persist, please see below.

### Questions
- It is ironic that you need Gradient Boosting, a non-NN learning method, to better understand NNs.
- Is the proposed method also applicable to other LLMs, e.g. Qwen? I do not expect you to perform this experiment, but would like to know your intuition behind that. 
- Transcoder:
  - When you replace the original LLM's MLPs by transcoders, how similar are their performance? For example, if the model is significantly altered through that, then whether you can detect faulty reasoning for the surrogate model might not have much to do with detecting it for the original model.
  - Related to the above question: Is the experimental evaluation fair w.r.t. other models? I guess you evaluate other models on the original Llama, while your method is on the modified one. If there is any shift in the MLP, possibly it might become easier overall to detect faulty reasoning.
  - Related to that topic: I have read the appendix on the transcoder training and I am aware that it is done correctly. I still would like to have your opinion on the remaining difference to the original model.
- Could you train a model on all three tasks simultaneously and what would its performance be? From what I understand you train a model for boolean, arithmetic and GSM8K separately.
- Please add an exact formulation for each benchmark metric and how you evaluate all these numbers exactly.

### Soundness
4

### Presentation
3

### Contribution
4
