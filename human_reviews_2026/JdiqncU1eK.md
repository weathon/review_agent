# COFT: Counterfactual–Conformal Decoding for Fair Chain‑of‑Thought Reasoning in Large Language Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Large language models (LLMs) can reveal and amplify societal biases during chain-of-thought (CoT) generation. We present COFT (Chain of Fair Thought), a training-free decoding method that provides instance-level fairness control with statistical guarantees for any frozen causal language model. 
COFT operates in three stages. First, it creates a masked counterfactual prompt by replacing sensitive spans with neutral tokens. Second, it compares the factual and masked logit distributions through lightweight logit fusion to attenuate attribute-driven biases. Third, it uses dual-branch split-conformal calibration to certify per-step candidate token sets at a user-chosen risk level.
We evaluate COFT across six models and multiple bias benchmarks. Our method reduces standard bias metrics by 30–55\% (median 38\%) while preserving task utility and language quality. Reasoning accuracies remain unchanged within run-to-run noise margins. The computational overhead is modest, equivalent to one additional forward pass (<=11%). 
COFT offers a clear, auditable path to safer CoT generation with significant bias reduction, negligible utility loss, and no requirement for retraining, auxiliary classifiers, or weight access.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces COFT, a training-free method for reducing bias in the chan-of-thought reasoning of LLMs. COFT uses parallel forward passes with an original factual prompt and a concurrent counterfactual prompt where sensitive attributes such as sex or race changed to just be [MASK]. Basically during inference the next token is sampled from the conformal set that is defined in terms of candidates being sufficiently likely (by some calibrated amount) to appear in both the true and [MASK] streams. COFT demonstrates improved bias mitigation compared to a range of baselines across multiple bias benchmarks.

### Strengths
The paper tackles an important problem of addressing bias in COT reasoning. Crucially COFT is training-free and just an inference-time intervention, which makes it practical and applicable to fixed LLMs. Since COFT uses conformal prediction it is able to obtain statistical guarantees, which is a big plus. I also think the use of conformal here was non-obvious and clever, specifically the dual-branch split-CP. The theoretical analysis provides justification for the method. The empirical evaluations and ablations are for the most part really thorough and demonstrative of the usefulness of COFT for simultaneous bias mitigation and utility preservation. I think this work provides an important contribution to figuring out how to use conformal during decode-time.

### Weaknesses
The masking mechanism can be a bit strong and ruin context too much, as acknowledged in the appendix.

I do think it is important to acknowledge that the guarantees from conformal just give us token-level or marginal guarantees here, no guarantee on the entire sequence as a whole and I think the sequence-level guarantee from the union bound is too loose to be very meaningful since for $1-T\alpha$ I think in practice $T$ will be too large for any reasonable $\alpha$ from conformal to make this bound meaningful. 

COFT still requires logit access, which may not be the case for many SOTA models. 

I think the literature review for other conformal methods applied to LLMs is a bit lacking and could be improved, for example adding eg. (1) Large language model validity via enhanced conformal prediction methods, (2) Conformal Arbitrage: Risk-Controlled Balancing of Competing Objectives in Language Models, (3) Conformal Tail Risk Control for Large Language Model Alignment, and some other references within those.

### Questions
I'm a bit confused by the overhead claims, why is it not a ~100% increase in computation cost to make two full forward passes? Maybe I am missing something crucial but would be happy to understand better. 

Would it be possible to report on how often the empty-set fallback is triggered?

### Soundness
3

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
The paper introduces COFT (Chain of Fair Thought), a novel, training-free method designed to control societal bias in Large Language Models (LLMs) at the decoding stage, particularly during chain-of-thought (CoT) reasoning. The method merges counterfactual logit fusion with a dual-branch conformal filtering process to produce a fair and reliable set of possible next tokens, based on the user’s chosen risk tolerance. Across six different models, this approach is empirically proven to significantly reduce standard bias metrics by 30-55% while preserving task utility and language quality, all with a predictable overhead comparable to one extra forward pass!!!

### Strengths
1. The method is training-free, and it is adaptable to all different types of llms.
2. COFT provides distribution-free, finite-sample guarantees through its use of conformal prediction (CP).
3. Convincing Experimental Results.
4. Strong theoretical jusftification 
5. Using a masked probe, geometric logit-blending, and a dual-branch conformal filter is novel.

### Weaknesses
1. Although the method is training-free, it still depends entirely on the predefined sensitive spans $S$. Because it focuses only on specific "lexical" spans, it may miss indirect or rephrased signals of bias. Mistakenly missing a sensitive span (a false negative) or masking a neutral one (a false positive) could shift what the model considers "fair." 
2. Masking a span with “[MASK]” is a questionable step. The authors themselves note that this can remove helpful context or confuse references. Even though the dual-branch conformal predictor tries to correct for this, if the meaning changes too much between the original and masked text, the certified set $C_t$ can become very small or even empty, triggering a fallback.
3. The evaluation primarily addresses single-attribute bias. Extending it to multiple attributes via joint or factorized masks is discussed but not explored. Although the factorized approach improves precision, it scales computationally with $K$ attributes, making it expensive for intersectional fairness.
4. The guarantees depend on the assumption of "exchangeability (A2)" between the calibration data and the test data. If the distribution of prompts encountered a drifts, the guarantees degrade. The results can be different. 
5. The presentation is not too great.

### Questions
1. The author mentioned that "[MASK]" is "tokenizer-stable". Can you experiment with alternative sentinels, such as a neutral placeholder phrase (e.g., "a person") or a syntactically valid but semantically empty token?
2. The algorithm falls back to arg max sampling if the certified set $C_t$ is empty. In your experiments, how often did this "empty set" event occur?
3. The fusion parameter $\lambda$ is chosen via a Pareto-knee rule on a validation set. How stable is this chosen $\lambda$ value across different tasks and domains? 
4. Can you compare three strategies on a dataset with intersectional bias: (1) COFT-Joint, (2) COFT-Factorized, (3) COFT-Single?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Chain of Fair Thought (COFT) for fair model response generation. The proposed approach COFT generates factual and maksed counterfactual logits trajectories. The two logits trajectories are fusioned together and followed by Dual-Branch conformal prediction for next token certification. The propsoed approach is evaluated against a suite of baselines on multiple standard fairness datasets. The experimental results show that the proposed approach achieved SOTA peformance in terms of fairness with small degradation on utility.

### Strengths
1. The proposed approach achieves pretty good performance on fairness without sacraficing utility. 
2. The proposed approach is evaluated on a comprehensive fairness benchmarks.

### Weaknesses
1. Due to that it needs to maintain two trajectories, the decoding efficiency is worse compared to other approaches. This is an issue especially for low resources computing environments.
2. The utility benchmarks are limited. It's not clear how this approach impacts the model performance on other types of tasks such as long-form response generation.

### Questions
How does the proposed approach performing on long-form generation tasks? Will the generation quality become worse as the generated sequence becomes longer and longer?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper considers the problem of mitigating bias in LLMs without retraining or performance degradation. They propose COFT, an inference-time approach which creates a counterfactual view of the prompt with sensitive attributes masked, and then at each step of decoding, mixes the per-token logits given the masked and unmasked state, and lastly determines the set of eligible next tokens to sample from according to a conformal threshold applied to both the mixed and counterfactual distribution.

### Strengths
- The problem setting and motivation are reasonable (improving LLM fairness while retaining utility).
- The core technical idea, to operationalize fairness as invariance to counterfactual masking of protected attributes, i.e., by replacing attributes in the prompt by a neutral mask, is interesting.
- The experiments are thorough and consider several open-source LLMs across several datasets, although the models are fairly out-of-date (Llama 2, Mistral, Qwen 2).

### Weaknesses
The major shortcoming of this paper is that the writing style is extremely terse and therefore hard to digest. For example, there are many, many semi-colons stitching together sentence fragments with dense technical buzzwords. For example, at line 122: "Representation-space debiasing (e.g., INLP; adversarial objectives) removes attribute subspaces or reduces recoverability in hidden states (Ravfogel et al., 2020; 2022; Elazar & Goldberg, 2018); such global projections are prompt-agnostic and may erase legitimate, context-linked semantics or miss non-linear effects (Liang et al., 2023) and typically still need weight updates." There were many such sentences like this that took me several reads to parse.

I strongly suspect that this is because almost all of the paper was written or rephrased by a frontier reasoning LLM (I say frontier because ultimately the text does make sense if one spends enough time parsing it, and reasoning because the writing's terseness seems like it could have emerged from reasoning RL with a length penalty). 

There are several features of the paper that seem to support this claim, including the terseness and ubiquitous use of semicolons, issues in spacing, the layout of Figure 1, and a few sentences with strange content such as "Main text focuses on representative hardware/settings to conserve space while keeping conclusions intact" (line 421). That being said, I was not able to find definitive proof of this claim.

### Questions
Though the paper appears to be technically sound with an interesting method and thorough experiments, the writing is very hard to parse due to its terseness. I would appreciate if the authors could commit to a significant rewrite to make the text more digestible, and would disclose any use of LLMs in drafting the text (as far as I can see, this was not disclosed in the submitted draft).

### Soundness
3

### Presentation
1

### Contribution
3
