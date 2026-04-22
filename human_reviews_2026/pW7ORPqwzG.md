# Epidemiology of Large Language Models: A Benchmark for Observational Distribution Knowledge

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 2

## Abstract
Artificial intelligence (AI) systems hold great promise for advancing various scientific disciplines, and are increasingly used in real-world applications. Despite their remarkable progress, further capabilities are expected in order to achieve more general types of intelligence. A critical distinction in this context is between factual knowledge, which can be evaluated against true or false answers (e.g., "what is the capital of England?"), and probabilistic knowledge, which reflects probabilistic properties of the real world (e.g., "what is the sex of a computer science graduate in the US?").
Much of previous work on evaluating large language models (LLMs) focuses on factual knowledge,
while in this paper, our goal is to build a benchmark for understanding the capabilities of LLMs in terms of knowledge of probability distributions describing the real world.
Given that LLMs are trained on vast amounts of text, it may be plausible that they internalize aspects of these distributions. 
Indeed, this idea has gained traction, with LLMs being touted as powerful and universal approximators of real-world distributions. At the same time, classical results in statistics, known under the term curse of dimensionality, highlight fundamental challenges in learning distributions in high dimensions, challenging the notion of universal distributional learning. In this work, 
we develop the first benchmark to directly test this hypothesis, evaluating whether LLMs have access to empirical distributions describing real-world populations across domains such as economics, health, education, and social behavior. Our results demonstrate that LLMs perform poorly overall, and do not seem to internalize real-world statistics naturally.
This finding also has important implications that can be interpreted in the context of Pearl’s Causal Hierarchy (PCH). Our benchmark demonstrates that language models do not contain knowledge on observational distributions (Layer 1 of the PCH), and thus the Causal Hierarchy Theorem implies that interventional (Layer 2) and counterfactual (Layer 3) knowledge of these models is also limited.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a benchmark to evaluate LLMs' inherent cognitive abilities in describing real-world population distributions, and tests both open-source and closed-source models on this benchmark.

### Strengths
The research motivation behind this paper is interesting. Testing LLMs' abilities in describing real-world population distributions is a good evaluation angle.

### Weaknesses
**Presentation:**
The presentation could be significantly improved. Sec. 2.1 and 2.2 contain excessive background details that do not directly contribute to the paper’s core message. Meanwhile, the main experimental analysis takes only about one page, which is insufficient to convey meaningful insights. The authors should better balance specificity and abstraction—focusing on the most critical aspects in the main text and moving only secondary details to the appendix. 

**Soundness:**
The evaluation methodology appears questionable. For example, the paper estimates probabilities such as $P(\text{answer A})$ and $P(\text{answer B})$ from next-token likelihoods for prompts like:

> “Has a person aged 16 years ever used marijuana?
> A. Yes
> B. No”

However, this approach assumes the model will respond exactly with “A” or “B.” In practice, instruction-tuned LLMs often generate free-form text such as “According to various studies…” or “Most surveys indicate that…”. Such valid but differently structured responses are excluded from the current evaluation, which could bias the measurement and fail to capture the true underlying model belief.

**Task selection:**
Many tasks involve potentially sensitive or biased categories (e.g., employment status by sex or race). Since modern LLMs are intentionally trained to mitigate stereotypes and demographic biases, they may underperform in these tests—not because they fail to model real-world distributions, but because they are *debiased by design*. Consequently, these tasks may not accurately reflect the model’s understanding of human distributions. The benchmark would benefit from more neutral and interpretable task designs, or from prompt formulations that explicitly specify an objective statistical perspective.

### Questions
1. Have you examined the full next-token probability distribution for each prompt? It would be interesting to see the top 50 likely continuations to understand whether plausible alternative responses (e.g., “According to surveys…” or “Most people…”) were excluded by the current evaluation protocol. This analysis could help assess whether the current evaluation fairly captures the model’s belief distribution.

2. Many questions reflect dynamic real-world statistics that change over time. Have you considered adding temporal qualifiers (e.g., “as of 2020” or “in the past decade”) to make the evaluation more precise?

### Soundness
1

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
The paper builds a benchmark to test whether LLMs encode real population distributions in the US across 10 datasets and 169 tasks. It elicits model probabilities via next token choices for multiple choice questions and scores them with an L1 distance that is normalized by baselines and calibrated with a bootstrap of the ground truth. Results show low scores overall. Best models reach about 22 out of 100 on low dimensional tasks and 17 out of 100 on higher dimensional ones. The work frames the stakes using Pearl’s Causal Hierarchy and argues that weak Layer 1 knowledge implies limits for intervention and counterfactual claims. Code and datasets are organized for reproducibility.

### Strengths
1. Focuses on observational distributions, not factual QA. Ties the goal to Pearl’s hierarchy with a precise statement of implications. 

2. Ten public US datasets across health, education, labor, finance, crime, and attitudes. Tasks span 1d and higher conditional distributions.

3. Shows consistently poor performance across many open and closed models, and separates low vs high dimensional regimes.

### Weaknesses
1. Main analysis leans on multiple choice next token elicitation. Small wording or answer labeling changes can shift probabilities. The paper mentions a probability prompt variant only in the appendix.

2. High-dimensional results cover four datasets with a modest number of contexts, so evidence about the curse of dimensionality remains suggestive.

3. Some low-dimensional tasks may appear on public sites. The paper notes better scores where tables are likely online, but does not audit data leakage or pretraining overlap

4. The study isolates pure model priors, but real systems often retrieve. A RAG or table lookup baseline would contextualize how far pure priors lag.

5. Methodological novelty is limited; the main contribution is the benchmark

### Questions
1. How robust are scores to prompt templates, label order, and few-shot examples. It would be helpful to show a sensitivity table per task family. 

2. Do calibration or temperature scaling methods change the L1 distances meaningfully, especially for models that are miscalibrated on probabilities?

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
3

### Summary
The paper introduces a benchmark that probes whether large language models encode observational, population-level distributions rather than only "factual" recall (e.g., "Who is the President of France?"). The tasks span major public US datasets across health, education, labor, finance, and crime, posed as templated questions, eliciting conditional distributions from models. Scores are computed by a distance from ground truth using an L1 formulation with permutation-averaged answer ordering. The authors position the results using Pearl’s causal hierarchy to argue that weak "Layer-one knowledge" in LLMs undercuts claims about possible interventional or counterfactual competence within LLMs. The main empirical finding is that current models fare poorly overall in distributional knowledge, with modest advantages when the queried statistics plausibly exist verbatim on the web; the paper also explores likelihood prompting, closed-weight models, retrieval, and small finetuning, all with limited apparent upside.

### Strengths
The motivation is overall solid: observational knowledge is a prerequisite for meaningful causal claims, and the benchmark operationalizes that premise in a way that is modular and easy to extend. The scoring method emphasizes stability and avoids KL’s brittleness; the use of permutation averaging usefully mitigates option-ordering artifacts. The paper triangulates results across open and closed models, contrasts question-answer with likelihood prompting, and even tests retrieval and finetuning; together, these analyses lend credibility to the central negative result. The paper’s framing using the causal hierarchy is helpful (although not essential to the contribution) and well-explained for a general ICLR audience. The limitations/ethics/reproducibility statements are thorough.

### Weaknesses
- The conceptual contrast between “factual” and “probabilistic” knowledge is possibly overstated. Many distributional summaries are themselves "facts" about the world expressed in text, so the paper could temper claims that distributional knowledge is categorically distinct from factual knowledge and instead clarify that the difference lies in aggregation and calibration rather than a deeper ontological difference.

- The paper occasionally implies that models are claimed to “approximate real-world distributions” in general, yet the base model in LLMs is simply trained to perform next-token distributions over text. The Altman quote is suggestive, but does it set up the paper's contribution (to a degree) relative to a "strawman" argument whose prevalence in the scientific literature is not convincingly articulated? A more concrete discussion of how token likelihoods should, in the limit, generalize to world frequencies would be interesting, as would an outline of assumptions about how "facts" and "knowledge" are to be distinguished or represented in the pretraining corpus.

- The reported improvements on government statistics appear strongest where tabulations are likely present in training data; the paper notes this, but a more systematic audit of overlap or a held-out slice that cannot plausibly appear verbatim would strengthen the story about what the models do and do not internalize. If *all* distributional quantities were present in the training data, how would this affect the implications of these (or future) findings? This question is related to the overall wonder about how distributional quantities are themselves facts, or facts seen from a certain perspective, so the distinction between facts and knowledge outlined in the paper can at times be strained. 

- The paper claims, "all models perform poorly"; a more objective quantification of this would be useful. For example, if humans perform far better than LLMs in this task, that would be a more defensible and objective statement than the generic "poor" performance claim, which is not defined relative to a standard that would be expected from "good" vs. "bad" performance.

- The paper's central claims are vulnerable to the possibility that LLM probabilities are well-calibrated but simply shifted. Systematic bias (but correct relative likelihoods or likelihood rankings) would I believe be quantified as lack of knowledge in this framework. Relatedly, I believe there is at least some work already indicating that LLMs might struggle to generate complex distributional quantities, while performing much better at tasks involving ranked choice, which can then yield better implied probabilities [arxiv:2306.17563; relatedly: arXiv:2403.14859].

### Questions
What fraction of prompts in each dataset are likely to have near-verbatim tables on the open web, and can the authors report scores conditioned on that attribute (verbatim and non-verbatim) to try to separate memorization from generalization, if this has not been done already?

### Soundness
2

### Presentation
3

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
The authors study the ability of LLMs to answer population-level statistical questions (e.g., what is the probability that a US citizen is male?). They establish a methodology for questioning and eliciting probabilities from open and closed weight LLMs, and assemble a benchmark of questions grounded on datasets containing data on demographics, consumer behavior, education, etc. They evaluate the models using either yes/nor question answers --  from which they elicit probabilities either by the relative frequency of repeated questioning or by renormalization of the next token probability (this only for open weight models) -- or a likelihood strategy that directly queries for the probability of some event encoded as histograms. Results show that performance is overall poor, which closed-weight models outperforming closed-weight models and likelihood questioning outperforming yes-no questioning for most models.

### Strengths
To my knowledge, the task investigated by this work is novel. It is certainly a useful task and questioning wether LLMs can satisfactorily answering such type of questions is scientifically and practically useful. The benchmark and methodology are reasonable. The paper is  mostly clear and well written, and cites relevant related work.

### Weaknesses
While the task addressed is relevant, it is not clear that those type of questions are the most relevant for the end users (say, data analysts or domain experts). The text brings no justification for the particular choice of statistical questions analysed. For example, one might be more interested in deciding the mode of a distribution or statistics (median, interquartile range etc) instead of a marginal distribution of a categorical variable. The discussion about SCMs and causality are not very relevant for this actual work; on the other hand, many important empirical results are left for appendices, as the comparison of likelihood vs. yes-no questioning, performance of closed vs. open weight and fine-tuning. This makes the work poorly self-contained.

### Questions
I can understand that possessing some knowledge about basic statistical facts might be beneficial for a general purpose LLM. However, one would not expect an LLM to have implicit knowledge about every dataset, even if public. So in this sense the benchmark of datasets is rather arbitrary and the results not robust. Wouldn't be better to investigate the ability of LLMs to answer such questions when given information about the dataset (say, about the data at hand or even in the form of statistics)?

### Soundness
3

### Presentation
3

### Contribution
2
