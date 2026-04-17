# Salience Aware Mark-Steered Prompting For LLMs

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
The efficacy of Large Language Models (LLMs) is heavily dependent on the quality of user-provided prompts. Consequently, many optimization methods focus on augmenting prompts with extensive details to provide comprehensive context. However, these methods often produce verbose and information-saturated prompts, which inadvertently causes LLMs to lose focus on the most critical instructions. This phenomenon, known as attention dilution, significantly constrains model performance on tasks requiring comprehension of long contexts. To address this issue, we propose Salience Aware Mark-Steered Prompting (MSP), a novel framework designed to mitigate attention dilution by explicitly steering the model's focus toward the most critical information within the prompt. MSP consists of two stages: first, Gradient-Guided Mask Search (GGMS) automatically identifies the most influential tokens. Second, Mark-Steered Decoding (MSD) persistently guides the model by amplifying the influence of these key tokens at every step of the generation process, improving the model's alignment with core user instructions. We evaluate the effectiveness of MSP on five widely used benchmarks with three representative LLMs of multiple scales. The experimental results show that MSP yields consistent performance gains over state-of-the-art baselines, and its strong performance across diverse tasks and models highlights its robustness and generalizability. Our implementation is provided in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce a 2-stage method to improve model steering. The method first automatically identifies the most influential tokens and then linearly amplifies their influence, using some computed overhead. Results for three LLMs on 4 datasets show some improvements, compared to prompting-based baselines as well as baselines requiring human prompt annotation.

### Strengths
- The studied problem is important and timely
- The introduced method seems intuitive
- The paper is well-written and clear throughout
- The performance improvements in Table 1 are substantial

### Weaknesses
There seems to be a lack of clarity in the evaluation/baselines. The authors propose a two-step procedure, with both steps being novel (to the best of my knowledge). The first step selects a span of the prompt to emphasize while the second step emphasizes it.

The two non-prompting baselines (PASTA and SPA) they compare against require selecting a span of the prompt to emphasize, i.e. they are designed only for the second step -- how did the authors do this selection (i.e. the first step)? The authors do provide some comparisons of different naive methods for the first step in Table 3 - it would be helpful to see these two methods in this table as well to see whether the second step is actually providing gains or whether the gains come from just omitting the first step. Additionally, it might be nice to see a less restrictive way of selecting the span, i.e. prompting the model to to select the span before emphasizing it (as is done in AutoPASTA https://arxiv.org/abs/2409.10790). It would additionally be nice to understand more details about how the baselines were run (e.g. how was head selection conducted for PASTA?)

While the authors do show some ablations for Step 1, it would be nice also to see ablations comparing against different methods for emphasizing the selected span in Step 2. Examples include contrastive decoding, prompt emphasis (e.g. adding asterisks or context-faithful prompting: https://arxiv.org/abs/2303.11315), or pasta upweighting.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the problem of identifying salient tokens in language model inputs, which are tokens that are most critical for generation quality and alignment with user intent. The authors propose an algorithm that uses gradient-based updates to find which input tokens, when masked, most significantly impact the model's output probability distribution. They then amplify these identified salient tokens by scaling their logits during generation.

Although interesting, the paper has several significant weaknesses. The gradient-based masking approach treats the fundamentally discrete mask as continuous, making it unclear whether high gradients reliably indicate importance. The connection to user intent is also poorly validated since the paper lacks qualitative analysis showing which tokens are actually identified as salient and how masking them affects generated outputs. Most importantly, the method is computationally expensive, requiring multiple forward and backward passes with three task-dependent hyperparameters (s, k, w) whose optimal values must be tuned per model and task. Hence, this raises questions about whether simpler alternatives like chain-of-thought prompting might achieve similar results more efficiently.

### Strengths
- The problem of salient token identification is an interesting and timely topic.
- The paper is well written and easy to follow.

### Weaknesses
- **Algorithm:** The proposed algorithm identifies salient tokens by computing gradients with respect to a mask. However, the mask is treated as continuous, even though it is fundamentally combinatorial (as seen in Eq. (2)). It is unclear how high or low gradients reliably indicate token importance, as a gradient could be high at m = 1 but result in a better loss at m = 0 in Eq. (3). Additionally, the authors refine the mask by substituting masked and unmasked tokens one at a time and evaluating Eq. (3). This choice is not justified, particularly compared to other well-established binary optimization frameworks, such as genetic algorithms.

- **Relationship with user intent:** The framework claims to identify tokens most relevant to user intent (page 5). However, it is difficult to see how the algorithm achieves this. A qualitative analysis is necessary: which tokens are identified as salient by the algorithm, and how does masking them affect the generated outputs? The current focus on quantitative results alone does not fully validate the framework. Moreover, masking tokens just based on the probability of generation of a sequence could have unintended consequences, such as slightly changing generated tokens while significantly impacting meaning or factual correctness. This raises concerns about the objective function in Eq. (3), which requires further justification (especially qualitative one).

- **Complexity:** The method relies on masking input tokens and performing multiple forward and backward passes across many mask combinations, particularly during s gradient-guided mask updates. After identifying k (a hyperparameter) salient tokens, another hyperparameter w scales their logits. The optimal values of s, k, and w are task- and model-dependent (page 9), which introduces a high computational overhead. It is unclear whether simpler approaches, such as chain-of-thought prompting, could achieve comparable results with similar or even lower computational cost.

- **Rigor:** In Eq. (3), m is defined as {0,1}^T, but since it is applied via a Hadamard product with x, I imagine it should be {0,1}^n. Additionally, the claim that “during autoregressive generation, the model tends to assign equal attention to all input tokens” is inaccurate. Transformers compute attention scores based on token embeddings, so the model inherently encodes information about token importance. Therefore, the claim in Section 3 is misleading.

- **Minor typo:** In Eq. (2), x should be bolded as it represents a vector of input tokens.

### Questions
Please refer to Weaknesses section.

### Soundness
2

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
4

### Summary
In this paper, the authors propose a method to reduce the attention dilution problem in LLMs. First, the method picks out the salient tokens in the input text by using gradient optimization to maximize the probability of change of output tokens. Then, during the decoding stage, the mask and unmasked input are fed to the model to get logit vectors. The final vector used is the weighted sum of two vectors. Experiments on five datasets and three LLMs demonstrate the effectiveness of the proposed method over baseline and two SOTA models.

### Strengths
1. Attention dilution is an interesting and important issue in LLMs and needs more research on it.
2. The method is intuitive to increase the attention of the model towards certain tokens in the source text.
3. Experiments demonstrate the significant improvement of the proposed method over the baseline.
4. The writing of the method section is relatively clear and easy to follow.

### Weaknesses
1. The research question is unclear. Although the abstract and introduction mention prompt optimization, the input in the experiments is still native input, where no prompt optimization methods are applied. This makes the core problem, attention dilution, suspicious and unclear. Are you focusing on the attention problem of the original prompt or the optimized prompt, or both? If you focus on the original prompt, does attention dilution exist (according to the abstract)? Since the instructions of the proposed datasets are short, is there any analysis to show that the model really suffers from this problem?

2. Lacks some baselines. Although SOTA and vanilla models are included, it is still unclear whether the mask generation algorithm is SOTA. For instance, can we directly ask GPT what the most salient tokens are in the sentences? Or do we have a metric to evaluate the quality of the mask? In addition, some token compression techniques might also serve as the baselines, such as LongLLMLingua. 

3. Lacks some analysis. First, it is not certain how the proposed method improves the performance. Does the proposed method let the model notice some important tokens so that the model can answer the question correctly? At least some case studies should be helpful. Second, an analysis of attention dilution (such as an attention map) might enhance the understanding of the core issue that the paper wants to solve. Third, since some of the tokens in the input are directly masked, thus, the influence of grammatical correctness may lead to a response change, although the token itself might not be semantically important. For instance, sometimes LLM cannot generate y because the sentence after the mask is a wrong sentence.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
