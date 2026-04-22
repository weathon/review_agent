# Latent Debate: A Surrogate Framework for Interpreting LLM Thinking

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Understanding the internal thinking process of Large Language Models (LLMs) and the cause of hallucinations remains a key challenge. 
To this end, we introduce \emph{latent debate}, a novel framework for interpreting model predictions through the lens of implicit internal arguments. Unlike the current work of self-consistency and multi-agent debate, which relies on explicit debates among multiple answers or multiple models, latent debate captures the hidden supporting and attacking signals that arise within a single model during a single inference step. 
We first present a model- and task-agnostic conceptual framework, and then instantiate it symbolically to approximate the thinking process of LLMs on True/False prediction tasks.
Empirical studies demonstrate that latent debate is a faithful surrogate model that has highly consistent predictions with the original LLM.
Further analysis reveals strong correlations between hallucinations and debate patterns.
These findings position latent debate as a potential framework for understanding internal mechanisms of LLMs, especially for scenarios where internal (dis)agreements appear during the inference steps.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Latent Debate as an interpretability framework that treats layer-level and token-level hidden states as “arguments”, maps them via an Argument Interpreter to True/False polarity and strength, and uses QBAF “thinking module” to get a final decision.

### Strengths
1. The overall idea sounds interesting.
2. The framework is training-free and has no intensive computational burden.
3. The authors incorporated ablation studies and some hallucination analysis on their approach.

### Weaknesses
1. **Evaluation looks toyish; baselines are too weak; no comparison to close related methods**

   * Datasets are toy datasets (500 each) rather than large-scale, realistic NLP benchmarks; the setting feels **too toy** to support their claims.
   * Baselines are **self-constructed and too simple**. Although the authors discussed DoLa / Logit Lens / Internal Consistency approaches briefly in text, which is good, they did not compare their method to these baselines.

2. **Writing quality is poor; core ideas are hard to follow on first pass**
   * Some keywords like "debate" and "arguments" are quite vague to understand and do not have concrete definitions. The paper probably needs a problem statement part up front (“What exactly is the concrete question you want to solve? ”). I even find this important question unclear for me on first pass.
   * Key design (token-wise weights) is moved to Appendix, which is clearly not a good practice since reviewers and readers are not required to read beyond main text. 
   * Many details are missing. For instance, the crucial similarity component for token weight is not specified (which model? metric?). The appendix only states a generic “cross-encoder similarity model” without naming or detailing it. 
   * There are many typos, e.g., Sec. 3.2.1 uses “trasparent” (should be transparent).
   * The illustrations are not so good as well. For example, figure 2, presented as the core framework overview, is quite ambiguous for the readers.

3. **Poor reproducibility and missing details**

   * Many key details for reproducibility are unclear, including: no specifics on the similarity backbone or metric (cosine? model name? pooling?); MLP classifier used in Section 5.1 completely lacks details on its architecture and training details.  
   * To make matters worse, the authors did not provide their code. Combined with the missing details mentioned above, it is really difficult to reproduce the results, while ICLR has a high standard on reproducibility and encourages a reproducibility statement in submissions.

### Questions
1.  Argument Interpreter uses a Logit Lens-like approach. However, Logit Lens (and the actual logits computation in models like Llama) typically **applies LayerNorm** (or RMSNorm) after hidden states and before the `W_unemb` projection. This normalization step is critical for maintaining the correct scale and distribution of logits. Could you please explain why you chose to omit this important normalization step? Would this issue affect your framework's interpretability?

2.   All current experiments are about binary True/False tasks with clear answers. How would your framework apply to more complex tasks that lack clear binary outcomes, such as open-ended text generation or general QA?

3.  To better illustrate the working mechanism of the framework, I strongly recommend providing concrete case studies to illustrate how your framework works under concrete scenarios.
   
4.  Please improve the presentation quality as the current version is not so easy to follow and many necessary details for reproducibility are missing.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a surrogate framework for interpreting models through argumentation frameworks. The authors instantiate it using hidden states and the unembedding matrix. The results show that (1) using QBAF as the reasoning module explains model predictions better than simple feature-based methods, and (2) MLPs trained on features extracted from debate patterns predict hallucinations more accurately than using individual features.

Although I am not familiar with argumentation frameworks, I find the surrogate modeling approach conceptually interesting. The experiments demonstrate the potential of explaining LLM behaviors through the latent debate perspective. However, the writing could be clearer, and I have several questions about the methodology and experiments, which I detail below.

### Strengths
- The conceptual framework is intuitive and well-motivated by argumentation theory.
- While the logit lens technique used in instantiation is not novel, the experiments provide solid support for the argumentation-based approach.

### Weaknesses
- For the instantiation, it is unclear why all tokens are treated as thinking steps (Line 288). In LLMs, only the final token attends to all previous ones. If earlier token representations lack full contextual information, it is questionable how they can form meaningful arguments about sentence correctness.
- The writing lacks clarity. More descriptive figure captions would improve readability, rather than relying solely on definitions in the main text. The authors should also clearly define key terms (e.g., “hallucination”) in the context of the tasks.
- More experimental details are needed for reproducibility, such as the specific versions of the models used.
- The paper focuses mainly on knowledge-based toy tasks (e.g., fact judgment). It would be more valuable to examine whether reasoning questions can be explained under the same framework, which would make the work more convincing and significant.

### Questions
- Please specify the model versions and provide more experimental details for better reproducibility.
- Could you explain the rationale for defining each token as a thinking step? (see the first weakness)
- What proportion of the questions produce hallucinations (are hallucinations incorrect answers in this task)? Is there an obvious imbalance between hallucinated and non-hallucinated samples when training the MLP?
- Could you try to demonstrate the effectiveness of the proposed approach on reasoning tasks?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper hypothesizes that models, 'internally debate' the truth of things.
A framework is proposed where model activations are taken to act as arguments for or against the matter at hand or other arguments.

The framework is instantiated by using 'logit lens'-style probing in order to quantify to what degree each activation points into the 'true' or 'false' direction. Activations toward 'true' are considered arguments in favor and 'false' are against, with the degree of alignment acting as argument strength. From there it is hypothesized that 'arguments' on each token position argue for or against the 'argument' in the next token position, and that for the last token, the 'argument' argues for or against the 'argument' in the next layer. This structure is formalized as a graph, and a semantics is proposed that allows for propagating argument strengths through this graph to arrive at a final number that is hypothesized to predict the model's final prediction (between true and false).

Finally, features of the graph are used to predict hallucinations.

### Strengths
The paper is clearly written, results are presented clearly, and it has helpful diagrams.

The fact that the amount of disagreement between true/false directionality within a model is predictive of hallucination is a interesting finding.

### Weaknesses
The results require better baselines: 
(1) using only the rightmost hidden state from layer L-1, since that is the 'closest' to where the actual prediction happens. 
(2) compute the consistency score for all arguments, and present the max.
Without that, I don't think the benefit is properly established: to be beneficial, the outcome of the QBAF procedure should be more predictive than the variables that go into it. 

The same goes for the hallucination detection, where the results are not compared to any existing methods.

The language of debate/argument is not clearly established as metaphorical.

### Questions
No question.

### Soundness
1

### Presentation
3

### Contribution
1
