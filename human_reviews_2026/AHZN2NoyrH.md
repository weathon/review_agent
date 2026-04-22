# Fact-Checking with Large Language Models via Probabilistic Certainty and Consistency

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 6

## Abstract
Large language models (LLMs) are increasingly used in applications requiring factual accuracy, yet their outputs often contain hallucinated responses. While fact-checking can mitigate these errors, existing methods typically retrieve external evidence indiscriminately, overlooking the model’s internal knowledge and potentially introducing irrelevant noise. Moreover, current systems lack targeted mechanisms to resolve specific uncertainties in the model’s reasoning. Inspired by how humans fact-check, we argue that LLMs should adaptively decide whether to rely on internal knowledge or initiate retrieval based on their confidence in a given claim. We introduce Probabilistic Certainty and Consistency (PCC), a framework that estimates factual confidence by jointly modeling an LLM’s probabilistic certainty and reasoning consistency. These confidence signals enable an adaptive verification strategy: the model answers directly when confident, triggers targeted retrieval when uncertain or inconsistent, and escalates to deep search when ambiguity is high. Our confidence-guided routing mechanism ensures that retrieval is invoked only when necessary, improving both efficiency and reliability. Extensive experiments across three challenging benchmarks show that PCC achieves better uncertainty quantification than verbalized confidence and consistently outperforms strong LLM-based fact-checking baselines. Furthermore, we demonstrate that PCC generalizes well across various LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper works towards LLM-based fact-checking, aiming to improve performance and make it more efficient. This is based on an uncertainty-guided mechanism to decide how to fact-check a statement: Either through a deep search, a targeted search, or by directly answering from the fact-checking LLM’s parametric knowledge.

### Strengths
1. The method is tested on a good amount of datasets (3), LLMs (5, including frontier LLMs), and against 3 baselines.
2. The paper promises to release code upon acceptance for reproducibility.

### Weaknesses
Weaknesses, in order of magnitude:
1. Some statements in the paper are false:
    1. "Across all datasets and models, PCC consistently yields the lowest ECE (Figure 3).", but in Figure 3, PCC is only the best in 4/15 cases.
    2. “harmonically combining them [internal certainty and reasoning consistency] yields consistently better calibration.”, actually reasoning consistency has strictly better ECE than the harmonic combination (PCC?) in 8/15 cases
    3. “Across Sci-Fact, FeLMWk, and HoVER, PCC achieves the highest AUC values”, again this is not true according to Figure 6.
2. Other claims are illogical
    1. “verbal confidence is systematically overconfident and poorly aligned with accuracy”, but only the ECE was measured, which can be high both due to under and overconfidence.
        
    2. “internal certainty and reasoning consistency each capture only partial signals”, how can this be followed from the previous ECE plots? 
    3. There is often the logical fallacy of “The overall performance is better, hence the decision strategy of when to apply which strategy must be better” (e.g., lines 413-415). As explained above and below, the results indicate the decision strategy is very noisy. The more likely hypothesis in my eyes is that the overall performance is better simply because we mix-in more inference-time compute (deep search) in some cases. In any case, there are two potential hypotheses for why the overall performance is better (if it is), and to conclude with one of them requires to run ablation experiments.
3. The results are quite weak: 
    1. The uncertainty to do the fact checking has an AUROC of at best 0.62 (0.5 is random) in Figure 6. Hence, it is likely that the method just randomly decides whether to rely on internal knowledge, do a “deep search” or a “targeted search”
    2. The overall performance (i.e., deciding which strategy to apply and then applying it) is often times only marginally better than baselines, and no error bars are reported. This is although error bars would be important especially for Fact Checking, since the benchmarks for this are relatively noisy.
4. The paper in its current state is very unclear: 
    1. In Section 2.3, PCC is benchmarked, and argued to be one of the best methods. But PCC was never introduced in Section 2.2 or 2.1. Is PCC Internal Certainty multiplied with Reasoning consistency? 
    2. I cannot understand Figure 5 at all, the legend (and caption) seem to be different from what is actually being shown.
    3. In Section 3, how is the deep search implemented, and how is self-reflection implemented? Is this just using the prompts in the appendix to prompt GPT 4o differently (including web search)?

### Weaknesses that did not influence my score and don’t need rebuttal but would be great to fix in the revised version

1. “derived from log-probability margins between verdict tokens”, actually it is derived from probabilities, not log probs, according to the first equation
2. Please number your equations

### Questions
In Section 2.1, tau(c) is a half deterministic: If the top two tokens are equal, then you give certainty 1, otherwise you base it off the probabilities of True and False tokens in the vocabulary. What happens if you remove the first bit and just always use the probabilities?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the unreliability of LLMs' verbal confidence in fact-checking by proposing Probabilistic Certainty and Consistency (PCC), which combines internal certainty from token probabilities with reasoning consistency from NLI-scored contradictions between adversarially-prompted rationales. Experiments across three benchmarks with multiple LLMs show PCC provides reliable confidence estimates.

### Strengths
- The motivation is clear and addresses a real problem. The paper identifies that verbal confidence is poorly calibrated across models and provides a reasonable human analogy for why combining decisiveness with reasoning stability might help.
- The proposed two-dimensional confidence framework offers conceptual value. Decomposing confidence into internal certainty and reasoning consistency is interpretable and could potentially reveal different types of model uncertainty.
- The paper is well-organized and easy to follow. The presentation is systematic, and the figures effectively illustrate the core ideas.

### Weaknesses
I do have some concerns with the paper that I believe should be addressed:

- The authors mention that thresholds α and β are chosen empirically based on the distributions, but they provide no concrete procedure for selecting these values, no sensitivity analysis showing how performance varies with different thresholds, and no solution to the acknowledged generalization problem, which makes me question how we would actually deploy this method in new domains.
- The paper uses only 187 samples from SciFact and 190 from HoVER, which raises concerns about the statistical significance of the reported improvements and the generalizability of the findings. I wonder why the authors didn't validate their approach on larger-scale QA benchmarks, such as TruthfulQA or CommonsenseQA, which would provide more robust evidence.
- The paper emphasizes in the Abstract that PCC improves efficiency. But I also noticed that the method requires generating 2*K rationales and performing K² NLI inferences per claim. At the same time, the authors report no actual runtime measurements or cost comparisons, making the efficiency claims hard to believe.
- The entire reasoning consistency framework hinges on DeBERTa-v3-base-mnli-fever-anli accurately detecting contradictions; however, this paper provides no validation that this NLI model performs reliably in fact-checking contexts, meaning that any errors in the NLI component directly propagate into confidence estimates.
- Key ablations are missing. I don't see a comparison with a baseline that uses token-level probabilities without the adversarial prompting framework, which makes it impossible to isolate whether the gains come from moving away from verbal confidence or specifically from the consistency modeling.

### Questions
Please see the weaknesses I've outlined.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces the Probabilistic Certainty and Consistency (PCC) framework, a novel and well-articulated approach for estimating LLM factual confidence by jointly modeling probabilistic certainty and reasoning consistency. The adaptive verification pipeline, which dynamically routes claims to different strategies based on PCC signals, is clear, rigorous, and impactful contribution. The key strength of the method is the empirical evidence that shows PCC's superior calibration (lower ECE) over existing baselines across multiple LLM families.

### Strengths
1. The notion of Probabilistic Certainty and Consistency (PCC) framework is conceptually and methodologically sound, and empirically shown to be superior to single-signal methods.

2. The adaptive verification pipeline, which dynamically routes claims to different strategies based on the two-dimensional PCC confidence profile, is logical for optimizing efficiency and resources.

### Weaknesses
**Novelty**: The key components of PCC are log-probability based certainty and NLI-based consistency, both of which are well-established uncertainty estimation methods. The novelty is mainly in how they are integrated and adapted; no fundamentally new theoretical concept or learning framework has been proposed. Therefore, the overall novelty appears to be quite light. 

**Fixed threshold for routing decision**: Although the authors articulated this issue as a limitation in the Appendix, in my opinion, this is closely related to the novelty aspect of the paper as mentioned above. Thus, I recommend that the authors conduct and include a sensitivity analysis to show how performance metrics are affected by variations in $\alpha$ and $\beta$, to inform future implementations about the robustness of these thresholds.

**More recent datasets**: It is not clear why the authors did not consider datasets such as FEVEROUS, EX-FEVER, AVeriTeC, which are more recent than HOVER.

### Questions
Is there any reason for not considering more recent datasets, such as FEVEROUS, EX-FEVER, and AVeriTeC?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper seeks to avoid systematic RAG usage by measuring a model's confidence in a claim in two ways. They check output token probability, as well as answer consistence when prompted for counterfactuals. They empirically demonstrate a well performing pipeline.

### Strengths
concrete approach with direct and obvious technical applications w.r.t. today's customer facing industrial models. The applied methods seem to be designed to be robust to a model's output variation, (use of NLI, and mapping of multiple tokens to TRUE/FALSE outputs). Reproducibility efforts are made, with prompts being provided, and good method descriptions. Extensive and relevant model and dataset use. Comparison against multiple baselines.

### Weaknesses
the specific problem driven pipeline makes the authors choose strong assumptions:

> Lexical uncertainty v.s. Semantic uncertainty is not directly adressed. As discussed by Kuhn et al 2023, the model could output multiple different (inconsistent) tokens that would nonetheless have the same meaning. Is there a reason why your 2 metrics would not fall prey to this?

> In this specific context of optimising RAG retrieval, I notice multiple runs are required, with extra sampling steps. It would be relevant to have an estimation of when it becomes interesting to pay this cost compared to other existing metrics. Is it systematically cheaper to sample counterfactuals from the model than it is to run retrieval?

> (Minor: maybe mention earlier that you use multiple baselines--which is a strength. early on it seems you use only verbalized confidence)

> line 172/173 makes the assumption a model "should" not easily be swayed. Why this seems somewhat instinctive form an anthropomorphic perspective, empirical results are quite strong, there is no explanation or theoretical modelisation of why this should be, and this is not tested empirically either. This seems perhaps to be more of a hypothesis.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
