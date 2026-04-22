# When Can you TRUST Large Language Models?

- Avg Score: 2.00
- Decision: Reject
- Scores: 0, 2, 4, 2, 2

## Abstract
Quantifying neural network model uncertainty is a difficult problem that has far-reaching implications on our ability to improve model reliability. Uncertainty quantification is especially difficult in the context of LLMs and autoregressive models, as standard methods for uncertainty measurement that apply to single outputs often fail to capture the semantic complexity of the entire autoregressive output. To remedy this gap, we introduce TRUST (Temperature-Related Unambiguity via Similarity Tracking) scores, a novel approach for quantifying LLM uncertainty which reasons about uncertainty $\textit{across the entire model output}$ rather than being limited to a small number of subsequent tokens. TRUST scores take advantage of the natural semantic branching of LLM outputs for nonzero temperatures, and calculate uncertainty based on semantic similarity of multiple output rollouts for an LLM model. We show that TRUST outperforms industry standard uncertainty methods within complex multi-token language tasks like predicting math problem difficulty, and also can be distilled into efficient forward-pass models for easy inference. Crucially, TRUST scores can be calculated with nothing more than standard LLM calls and require zero white-box access to model internals.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The authors propose TRUST, a method for measuring the variance of sequences output by an LLM. Focusing on benchmarks that attempt to measure variability in the ground-truth output distribution rather than just correctness, they show that it outperforms a handful of naive baselines. They also show that small, efficient BERT-style models can approximate the TRUST metric.

### Strengths
I appreciate that the authors focus on efficiency, which is often overlooked in LLM uncertainty quantification papers, and also emphasize the difference between hallucination detection and uncertainty quantification, which is rarely discussed.

### Weaknesses
The glaring issue with this paper is that it does not situate itself properly in the literature. The motivating observations in section 3.2 (namely, that LLM output variance must be computed at the semantic level, that LLMs can't estimate their uncertainty themselves, and that LLMs produce different outputs at nonzero temperatures) are not original to this paper, and indeed are precisely the facts that motivate "semantic uncertainty" (Kuhn et al. 2023, Farquar et al. 2024). That method bears remarkable similarities to this one, but despite having become a universal baseline in this subfield, it is mentioned here only once, and in the related work. Even the idea of training a more efficient model to approximate the metric has been explored already in the semantic uncertainty literature, most notably in Kossen et al. 2024. I think this paper needs to be substantially revised and put in dialogue with this (large) body of work before I can consider accepting it.

Separately, in no particular order:

- The evaluations in the paper are non-standard and, in my opinion, somewhat flawed. The synthetic dataset does not actually elicit meaningful variance in LLM outputs, since each of the possible outputs to each question appears to be completely semantically distinct from the others. Therefore, you're not actually measuring TRUST's robustness along "semantically-invariant dimensions such as synonyms or word choice."
- Fine-tuning a model on the evaluation dataset before evaluating it seems a bit confounding. Fine-tuning (especially on synthetic data) probably collapses the variance of model outputs on prompts like this, which is precisely what we're trying to predict here.
- The idea of using the BERT model to predict TRUST scores outright, rather than just relegating it to the role of predicting similarity scores, seems dubious to me, and the fact that it does well in these evaluations probably says more about the evaluations (which, again, are not standard in the literature) than it does about the BERT model. Some problems that are difficult for one language model are easy for another. If you're using the same BERT TRUST score predictor with no input from the language model being evaluated (this is my interpretation of how BERT was used in the paper), you'll never capture that.
- Section 3.4 is a little strange to me. Why does it matter that TRUST, under ideal conditions, is strictly greater than MSP? A trivial metric that always just returns 1 would also be strictly greater than MSP. The analysis also oversteps; for example, the claim "Evidently, TRUST scores applied only to next-token prediction are generally the square of the MSP score in addition to higher order terms, but are always lower bounded by the square MSP." is false, since in reality you don't have a perfect judge.

### Questions
Could you clarify how the BERT approximation works? Is my understanding in the previous section correct?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method for measuring the uncertainty of LLMs in long-range text generations. The method relies on generating multiple samples from the LLM and computing their pairwise semantic similarity using a judge model. The average semantic similarity is then used as the uncertainty score. The authors show that their proposed measure has high correlation with a mixing factor on a preference dataset they propose, and that it can be predictive of the difficulty level on MATH questions.

### Strengths
- The work is tackling an important problem of uncertainty estimation in LLMs.
- The proposed TRUST scores operate in a black-box mode.

### Weaknesses
- **Significant related works are omitted.** This work seems to fit into the broad area of semantic uncertainty estimation in long-form LLM generations, which is not new. Yet, most related works are not discussed by the paper. In particular, semantic entropy proposed by [1, 2] feels like a more comprehensive version of the approach proposed in this paper, combining semantic clustering with entropy. Compared to semantic entropy, I do not see any methodological or empirical novelty provided by this submission.
- **Empirical evaluation is limited.** The paper does not compare to important baselines such as semantic entropy. Further, the evaluations are highly limited, non-standard and overall questionable. In particular:
    - Although the TRUST scores are presented as an uncertainty metric, no calibration results are provided, showing whether the proposed TRUST scores can be reliably used to predict model’s accuracy.
    - The preference dataset used in Sec.  4.2 seems arbitrary and not particularly suited for evaluation of uncertainty metrics. For starters, it is unclear to me what would be the value of quantifying uncertainty on a preference dataset? (In this case there is no notion of “accuracy”, so the uncertainty score cannot really be used to improve the reliability of the model). That makes the evaluation of the any proposed uncertainty metric on this dataset very difficult. Secondly, why is correlation with the mixing factor a desirable property? What is the purpose of fine-tuning of the model on this dataset?
    - On the MATH dataset, the TRUST scores are not used to predict the accuracy of the model, but rather the difficulty level of the question. While the model accuracy can be related to task difficulty, the two are not equivalent. It seems to me that what the experiments in section 4.3 are doing is showing that more difficult tasks lead to more diverse generations (which makes intuitive sense, as in maths problems there is usually more diverse ways to reach a conclusion on a more complex task). However, the paper does not in any way address whether the TRUST scores can be used to predict the accuracy of the model.
- **Lacking implementation details and ablations.** In section 3.3 the authors propose to estimate the TRUST scores using a predictive model. However, no relevant details about the training of the said predictive models are provided, deeming the results not reproducible. Further, the authors do not ablate the effect of the number of generations N, or whether pairwise vs sequential comparisons are more efficient.
- **Theory does not seem relevant.** In section 3.4 the authors provide a theoretical results meant to validate their proposed TRUST score. However, the result is based on an extremely simplified setting (single-step generation, ideal judge, binary 0-1 label), making this theoretical aspect a dubious contribution. Furthermore, it is unclear why being lower bounded by squared MSP is a desirable property.


[1] https://arxiv.org/pdf/2302.09664  
[2] https://www.nature.com/articles/s41586-024-07421-0

### Questions
- What is meant by “first possible response” in line 315? What’s the rationale behind using the mixing factor in the preference dataset? How do the datasets obtained with the different mixing factors actually differ?
- What is the training procedure for the TRUST predictive model? What dataset was used? Was a linear model used?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TRUST (Temperature-Related Unambiguity via Similarity Tracking), a black-box method for estimating uncertainty in large language models. Instead of relying on token-level entropy or max-softmax probabilities (MSP), TRUST quantifies semantic uncertainty by generating multiple completions from a model at a fixed non-zero temperature and measuring their pairwise semantic similarity using a “judge” LLM. High similarity implies low uncertainty. In parallel, a lightweight BERT mode is trained to predict these TRUST scores efficiently. The paper shows a theoretical link between TRUST and squared MSP in the single-token limit and evaluates the method on a small synthetic dataset and the MATH benchmark, where TRUST (and BERT estimation version) outperform entropy, MSP, and ensemble-based baselines.

### Strengths
1. Using semantic similarity across temperature-sampled completions for uncertainty estimation is intuitive and seems to be new.

2. Works in a pure black-box setting, requiring no access to logits or gradients.

3. Simple relationship to MSP gives theoretical grounding.

4. Clear writing and easy to follow.

### Weaknesses
1. Only two datasets; no downstream tasks (e.g., calibration or hallucination detection). More experiments can be more convincing of the results and it should not be a lot of efforts.

2. Sensitivity to the choice of the “judge” LLM and maybe to prompts too, perhaps some variants and analysis is needed.

3. Requires multiple generations per input; distillation helps but need to be tested properly. Some follow up experiments are needed.

4. Not sure how Theorem 1 generalise to vary length output and beyond single token cases.

### Questions
Please see my weakness section for more information.

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
5

### Summary
This paper introduces a method for estimating LLM uncertainty by sampling multiple outputs at nonzero temperature and computing the pairwise semantic similarity between outputs using another LLM as a “judge.” The authors claim that TRUST offers a black-box, semantically aware measure of uncertainty and that it outperforms conventional baselines such as MSP and entropy. Experiments are conducted on a toy “preference” dataset and the MATH dataset, where TRUST is used to predict task difficulty.

### Strengths
- The paper is written in readable English and the experimental setup is described in reasonable detail.
- The idea of leveraging temperature-induced sampling variance is intuitive (though not new).

### Weaknesses
1. **No technical contributions.** TRUST is just a rephrasing of “semantic similarity among multiple LLM samples” , a concept already well explored in the literature. For example, Semantic Uncertainty [1]. The authors do not identify or address any new technical challenge. The theoretical section merely shows that TRUST $\approx$ (MSP)², which is trivial and contributes nothing conceptually. The baselines are also very weak. Only entropy, MSP, and ensembling are compared. The paper completely ignores more relevant uncertainty or calibration methods (semantic uncertainty, conformal prediction, calibration-based OOD detection, etc.). The claimed “superiority” is therefore meaningless.
2. **Unconvincing experiments.** The “simple uncertainty prediction” task is an small-scale, synthesized, unverified dataset of random preferences. It has nothing to do with real uncertainty estimation. The “difficulty prediction” experiment on MATH is also questionable: predicting problem difficulty is not the same as predicting model uncertainty. None of the evaluations measure whether TRUST correlates with correctness or epistemic uncertainty.
3. **No link between uncertainty and accuracy.** The authors explicitly state they “did not consider model accuracy” when evaluating uncertainty, which undermines the entire motivation. Measuring uncertainty without checking calibration or correlation with correctness is meaningless.
4. **Poor presentation and professionalism.** Figures are low-resolution raster images rather than vector graphics. Many equations occupy large space for trivial content, such as Eq. 1 and 2. The writing often reads like a blog post rather than a research paper. The overall presentation does not meet the standards expected of ICLR submissions.
5. **No insight or takeaways.** The paper neither advances understanding of uncertainty in LLMs nor introduces any practical or theoretical framework that others can build upon. It is an incremental, poorly justified application of existing ideas.

[1] Kuhn, Lorenz, Yarin Gal, and Sebastian Farquhar. "Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation." arXiv preprint arXiv:2302.09664 (2023).

### Questions
Please refer to the weakness.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes TRUST (Temperature-Related Unambiguity via Similarity Tracking), a black-box method for estimating uncertainty in LLMs. Instead of relying on logits or token-level entropy, TRUST samples multiple completions from the same prompt at nonzero temperature and uses a separate judge LLM to compute pairwise semantic similarity among outputs. The average similarity serves as an inverse measure of uncertainty: higher similarity means lower uncertainty. The authors show a theoretical link between TRUST and squared Maximum Softmax Probability (MSP) in the single-token limit and claim strong empirical results on both a synthetic preference dataset and the MATH benchmark for difficulty prediction. A distilled BERT model trained to approximate TRUST achieves similar performance with lower inference cost.

### Strengths
+ The proposed uncertainty score requires no access to logits or model internals, making it deployable with closed APIs.
+ Outperforms entropy, MSP, and even ensemble-based metrics in tasks such as MATH difficulty prediction.

### Weaknesses
- The theoretical depth is limited only to a one-token, idealized setting, while multi-token behavior is purely empirical. The roles of temperature, number of samples, and judge prompt format are heuristic without principled justification.
- Generating multiple completions and pairwise similarity judgments may be expensive in computational cost. The paper did not conduct a complexity analysis.
- The use of pairwise semantic distances and averaging is intuitive but lacks a clear probabilistic interpretation or calibration guarantee.
- Benchmarks are small (a toy preference set and MATH).

### Questions
Can the authors demonstrate why the output text of LLMs will have natural sampling variance/uncertainty given a nonzero temperature?

### Soundness
2

### Presentation
2

### Contribution
2
