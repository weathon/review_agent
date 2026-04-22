# An Information-Theoretic Parameter-Free Bayesian Framework for Probing Labeled Dependency Trees from Attention Score

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Figuring out how neural language models comprehend syntax acts as a key to revealing how they understand languages. 
We systematically analyzed methods for finding syntax structures in models, namely _probing_, and found limitations yet widely exist in previous probing practice. 
We proposed a method capable of estimating mutual information (MI) and extracting dependency trees from attention scores in a mathematical-rigorous way, requiring no additional network training effort.
Compared with previous approaches, it has a much simpler model, while being able to probe more complex dependency trees, also transparent for fine-grained explanation.
We tested our method on several open-source LLMs and demonstrated its effectiveness by systematically comparing it with a great many competitive baselines. Several informative conclusions can be drawn by further analysis of the results, shedding light on our method’s explanatory potential.
Our code is released at https://github.com/ChristLBUPT/IPBP.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper aims to reconstruct dependency trees from the attention scores of LLMs using the IPBP method. Precisely, IPBP consists on estimating the Mutual Information between dependencies in a labeled dataset and attention scores. With mutual information, tree reconstruction is performed and evaluated with UAS and LAS metrics. Overall the method proposes a parameter-free way to predict syntax from LLMs. Results show that IPBP is has superior performance to baselines and it works for 7 different LLMs.

### Strengths
* The paper spots a limitation of probing techniques and the given solution is simple and elegant.
* The method is well described and mathematically rigorous, both for MI estimation and tree reconstruction.
* The results gather several LLMs and well-thought baselines.

### Weaknesses
There are several relevant works which are not cited

* Labeled Dependency probing: 
     - https://arxiv.org/abs/2412.05571
* Probing attention values: Attention scores probing: 
     - https://www.pnas.org/doi/10.1073/pnas.1907367117
     - https://arxiv.org/abs/1906.02715
* Probing ambiguous sentences with causal analysis:
     - https://arxiv.org/abs/2211.09748
* Syntactic probes are biased towards linearity
     - https://arxiv.org/abs/2508.03211

The results presented fall short at the current version of the manuscript, only different LLMs and baselines are evaluated.
There are no insights about LLM depth and tree depth and how syntactic parsing happens inside the model.

Ablations are based on a single and arbitrary sentence “In order to protect the environment, ecofriendly industries were”, this analysis should be done per dependency type and using specific tokens for the logits and not the average. For example, long-range dependencies like subject-verb agreement in PP sentences are good candidates to assess syntactic capability.

Also, the LAS values are far from SOTA, which might be justified by the fact that it is parameter-free. I think the paper could benefit from a syntactic dataset and thus a more controlled setting.

### Questions
How are the multi-token words treated? Are attention values averaged?
How do the maximum/mean layer-wise MI values look like? Within and across dependency types?
What is the baseline MI when using a randomly initialised LLM?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles the problem of probing labelled dependency trees from attention score when a transformers-based LM processes a sentence. Different from several work using deep neural networks (which, according to the paper, is "explaining by unexplainability", the paper proposes to use mutual information between dependency labels and attention scores. Tree reconstruction is done by (i) using the MI to estimate probability that a pair of tokens are linked by a labelled dependency arc, and (ii) using Eisner algorithm. 

The paper demonstrates that proposed method can reconstruct (un)labelled dependency tries with higher UAS/LAS scores than the baselines. Interesting, the paper shows that (i) LM decoders capture left/right dependency adaptively, and (ii) different layers capture different types of dependency arcs (local vs global).

### Strengths
* The paper is very interesting, especially without any learning. The idea of using mutual information is novel, though using Eisner algorithm in this case is pretty straightforward. 

* The real strength of the method lies in its simplicity and explainability. 

* Findings in Section 4.5. are particularly interesting, show-casing the strength of the method over learning-based ones.

### Weaknesses
* The use of $MI(l,A)$ to compute $P(x_i,x_j;l)$ doesn't seem information-theoretically motivated. How close is it to estimate $P(x_i,x_j;l|A)$?

* It's unclear whether the tree reconstructed by the Eisner algorithm faithfully expresses the syntax structure from the LM. For instance, thanks to the constraints (and biases) of the algorithm, eventhough $P(x_i,x_j;l)$ > $P(x_u,x_v;l)$, the former pair may not form a dependency arc but the latter may. In this counterexample, it's hard to say that the reconstructed tree faithfully expresses the syntax structure of the LLMs.

* The head-selection experiment doesn't capture intrinsic head-selection performance because it is evaluated via the down-stream task (tree reconstruction) using Eisner algorithm. Is there a way to evaluate head-selection performance in an intrinsic manner? 

*  The tree-reconstruction evaluation comparison doesn't include results in the literature.

### Questions
* How do left/right-branching baselines perform in terms of UAS?

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
2

### Summary
This paper proposes a parameter-free probing method for detecting how well attention maps in decoder-only transformers encode dependency tree information. This approach relies on estimating the mutual information between the attention scores of a pair of tokens and the dependency label, and no added learning of probe weights. This approach is evaluated for a number of models and compared against various baselines on UAS and LAS. Results show that this approach is revealing more syntactic structure than previous methods.

### Strengths
Parameter-free probes generally seem like a good idea and this approach generates MI estimates that can be useful for more fine-grained analysis.  
The paper is thoughtful about considering a number of different baselines in their experiments.

### Weaknesses
There is a mismatch between the attention scores being used for this analysis (bidirectional) and the actual scores being used by the model during generation (causal). It seems a bit strange to make claims about the syntax structure encoded in the model that is never actually used for generation. This point makes the conclusions and implications of the findings a bit hard to reason about.

### Questions
-

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper takes a step to address how LLMs comprehend syntax by proposing a novel method to extract labeled dependency trees directly from attention scores without training any additional neural networks. They aim to solve two problems: Directed Tree Extraction and Mutual Information Estimation for LLMs in a parameter-free Bayesian framework. Experiments are conducted comparing against several baselines, including probing and V-information comparing on labeled attachment scores (LAS), and unlabeled attachment scores (UAS) , where they show superiority of their method.

### Strengths
I found the paper overall easy to read, and the proposed Bayesian approach is sound, and I think the baselines considered are sufficient.

### Weaknesses
I am not an expert in this specific area, so my comments focus on high-level suggestions that could help improve the paper overall.

1. How does this work compare with Wu et al. (2020)’s *Perturbed Masking*, which is mentioned as a parameter-free approach? It would be valuable to include a direct comparison on the same datasets and metrics to better position the proposed method.

2. Could the authors consider a simpler baseline by using averaged attention scores directly as edge weights for Eisner or Chu-Liu-Edmonds decoding? This would help clarify how much improvement comes from the proposed components.

3. For ablations, it would be helpful to vary the bandwidth parameter for KDE across a range (for example, 0.5x, 1x, 2x, and 5x of the current setting) and report UAS and LAS results to evaluate sensitivity.

4. Another useful ablation would be to compare logarithmic opinion pooling with simpler aggregation methods such as the arithmetic mean or the standard geometric mean to understand whether the more complex approach provides a meaningful benefit.

5. The motivation and implications section could be made stronger. At present, the introduction seems to overstate the practical and scientific significance given the modest LAS scores of around 30 to 49\%. It would help if the authors discussed more candidly when attention-based syntax extraction is sufficient, when hidden states are preferable, and what these results reveal about whether large language models actually encode syntactic structure.

### Questions
Refer to the Weakness

### Soundness
3

### Presentation
3

### Contribution
3
