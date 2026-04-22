# BADI: Black-box and Anytime-valid Dataset Identification for Large Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large language models (LLMs) are trained on massive, uncurated internet datasets that often include copyrighted material, making training data identification essential for intellectual property protection. *Dataset inference (DI)* addresses this challenge by extracting diverse training membership features for a suspect set, aggregating them, and applying statistical tests to assess if that suspect set contributed to the model’s training. However, current DI methods face two major limitations that hinder their practical deployment. First, they require gray-box access to token probabilities, while state-of-the-art LLM APIs usually return only generated tokens. We address this issue by approximating per-token probabilities from label-only outputs, making *black-box DI* feasible. Second, existing DIs rely on p-value for statistical tests that necessitate a fixed suspect set and a predetermined significance level. This either leads to high computational costs for large suspect sets, especially in the black-box setup, or yields inconclusive results for smaller sets, since adding new suspect data points post-hoc might be necessary to provide strong enough evidence, but it invalidates statistical guarantees based on p-values. To overcome this limitation, we introduce a black-box DI framework based on *e-values* and sequential testing. The e-values offer anytime-valid guarantees and support optional continuation, enabling safe accumulation of evidence, reducing inconclusive outcomes and compute costs. Through these two fundamental advances, our **B**lack-box and **A**nytime-valid **D**ataset **I**dentification (BADI) method enables practical data auditing for LLMs, supporting their trustworthy deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces BADI, a black box method for dataset identification. The proposed black-box DI framework used e-values for hypothesis testing.

### Strengths
- This paper is well-organized.
- As the authors stated, the testing framework is based on e-values. This could introduce more flexibility in the task.

### Weaknesses
- Only the Pythia model family is studied in this work. 
- The rationale of using Eq.(1) for token probabilities needs further justification.
- Few baselines are studied in the experiment section.

### Questions
- In the experiment stage, only the Pythia model family is studied. This is a model released at 04.2023, it is trained on **300B tokens** [1]. Typically, today, LLMs like Meta-Llama-3.1 [2], "We pre-train Llama 3 on a corpus of about **15T multilingual tokens**, compared to 1.8T tokens for Llama 2." The Pythia model only consumes **1/50 tokens** of Meta-Llama-3.1. And the training tokens are always increasing, for example, "Qwen3 has been pre-trained on **36 trillion tokens**"[3].

- With the increasing of training tokens, many questions occur. For example, there is a large chance that similar examples to the test one exist, and this will lead to false-positive results. It is a well-known problem of the traditional DI/MIA method [4]. When nearly all web data is used for training, the detection will become hard, and choosing/creating a held-out set is a problem.

- A summary of the above two points, the reviewer believes that we should conduct more analysis with recent models rather than only using the Pythia model. This could give us more information about the proposed method. Can it really deal with modern LLMs training on web-scale data?

- For Eq.(1), it assumes a token-to-token mapping relationship between the generated sentence and ground truth. Will a semantically similar generated sentence get a low score with the ground truth? It may have different word orders, but basically the same meaning. How do we deal with and consider this case?

- In Figure 3, it seems that the reviewer only provides PETAL and Baseline for comparison. How about the performance of other DI/MIA methods? Could the authors provide other metrics like AUC?

[1] https://huggingface.co/EleutherAI/pythia-12b

[2] https://arxiv.org/pdf/2407.21783

[3] https://arxiv.org/pdf/2505.09388

[4] Do Membership Inference Attacks Work on Large Language Models? https://arxiv.org/pdf/2402.07841

### Soundness
3

### Presentation
3

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
This paper proposes BADI, a new framework for black-box and anytime-valid dataset inference in LLMs. The goal is to determine whether a suspect dataset was used during an LLM’s pretraining, addressing concerns around copyright and data transparency. The method is empirically evaluated on multiple Pythia models (410M, 6.9B, 12B) and diverse Pile subsets. Experiments show that BADI identifies training datasets with higher efficiency (fewer samples required) and lower false-positive rates than baselines (PETAL and a RoBERTa-similarity baseline), offering a practical framework for real-world LLM auditing.

### Strengths
1. The paper situates itself well within the context of data auditing, copyright compliance, and LLM transparency.
2. The authors argued that they proposed the first black-box DI framework that avoids reliance on per-token logits.
3. The step-by-step framework (Figure 1) is clear and pedagogically strong.

### Weaknesses
- The paper claims generality to “API-based black-box models,” but does not empirically test BADI on true commercial APIs (GPT, Claude, Gemini). All experiments are performed on open Pythia models (Biderman et al., 2023). A key claim "BADI enables accurate dataset identification in black-box settings" is not fully validated since Pythia allows full access.
- While e-values provide anytime-validity, the choice of betting strategy (λt bounds, ONS parameters) may significantly affect power. No experiments systematically compare BADI’s sequential testing to classical p-value–based tests under equivalent conditions.
- Some formulae (e.g., kernel MMD payoff mapping) and hyperparameters for the online regressor are only briefly described, lacking mathematical detail or convergence justification. Morover, the description of “STRIP-K% PROB” lacks sensitivity analysis across K values.

### Questions
- Have the authors tested BADI on real black-box APIs (e.g., OpenAI, Anthropic, Google)? If not, how confident are they that the token similarity estimation remains meaningful when model outputs are diverse or truncated?
- How was the sigmoid mapping in Eq. (1) empirically chosen? Is there a theoretical justification for using 2·sim as the scaling factor, and how sensitive is performance to this parameter?
- The paper adopts an Online Newton Step (ONS) for adaptive staking. How critical is this choice? Would simpler strategies (fixed λ or proportional betting) degrade performance substantially?
- The paper reports ≈1 % false positive at 5 % significance. Was this averaged over datasets, or verified under a global Type-I control across all tests?

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
3

### Summary
This paper presents a method for identifying if certain training data has been used in the training of an LLM. They do this also with black box models behind an API by approximating per token probabilities. 

The estimation of token probabilities for black box models is done with a surrogate model and a computation of semantic similarity between the surrogate model's response and the true token predicted by the black box model. These are mapped to token probabilities.

### Strengths
The paper presents a novel approach for both estimating token probabilities in black box settings as well as predicting inference attacks

### Weaknesses
Existing LLMs, when they are properly aligned, they are not really trained to generate continuations but to answer with reasoning given a query. In such setting, any inference membership extraction method will be limited. The evaluation is done with the Pythia models which are trained on subsets of the Pile dataset. These models are not aligned by default so the applicability of the method will be limited about this. Please let me know if I misinterpreted this.

This is a small concern, but I would suggest to define e-values before describing why they are suited to this particular issue or at least refer to the section where this is mentioned.

### Questions
Have you tested the estimation of token probabilities with models that you have access to the actual token probabilities?

Would this approach work with models that are fully aligned with RL optimization or similar methods?

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
3

### Summary
Overall, this is a good practical work that addresses important issues in dataset inference. While the evaluation could benefit from additional robustness tests, I believe readers of ICLR would find this work useful and interesting. Hence, I believe it should be accepted. 


The authors outline current limitations of dataset inference (DI) methods: they require gray-box access to token probabilities, and they rely on p-value statistical tests which incur high computational costs and allow “gaming” significance levels. The authors propose to estimate token probabilities and use “e-values” with sequential testing to overcome these limitations, leading to BADI. This is more practical, as specifying testing dataset size becomes a flexible, sequential process, and one does not require gray-box access to LLMs.

The general method is to aggregate dataset inference features (largely results of membership inference techniques) for a given dataset, construct an e-value from the features. The features often leverage token-level probabilities – these are estimated using a sigmoid-based calibration method which avoids use of a surrogate model. The semantic similarity of a produced token with a reference token is used as a proxy for the probability of the original token.  Membership inference features are computed from the estimated token probabilities. The membership features are then used to continuously train a scoring model. The scores are passed to a betting model which should accumulate evidence in the case of true positives (true training samples). One may stop the process if sufficient evidence has been gathered to reject the null hypothesis (that the samples were not part of training).

Experiments with Pythia+Pile validate the method presented by the authors. They compare BADI with two baseline black-box methods leveraging RoBERTa scores and PETAL. The experiments would benefit from additional robustness tests (see weaknesses).


Minor comments:
Line 131 consider rephrasing “However, most existing MI methods for LLMs, even the stronger gray-box attacks do not perform better than random guessing.” Perhaps “However, for most existing MI methods for LLMs, even the stronger gray-box attacks do not perform better than random guessing.”
Line 152: grammar issue “Consider a randomly ordered points” 
Line 160 “BADI rely” -> “BADI relies”

### Strengths
The paper is strongly motivated in practical scenarios and presents a useful solution to a challenging problem: dataset inference in black-box scenarios with flexible resource expenditure

The methods of the paper are sound, and the claims are largely substantiated by the experiments with Pythia+Pile.

### Weaknesses
The description of estimating token probabilities is not very clear. How is this semantic similarity measured? The “ground truth sequence” appears to be the one you would like to estimate the probability for – it would be helpful to refer to it as that if so.

There is not a study of robustness to choice of held-out distribution data. What if your trusted hold-out data is polluted with training samples (some percentage) or its distribution does not match that of the training samples? How does this affect evidence growth rate?

I would like to see more comparisons with additional dataset inference (DI) methods, however I believe there could be difficulties in creating apples-to-apples comparisons between p-value methods and e-value methods.

### Questions
How sensitive is the method to choice of non-member data? 

Do the non-member data need to come from the same distribution as the member data? If so, how would one acquire this data in practical settings? Is it robust to pollution from the training set?

What if you do have access to the logits? How does this impact evidence gathering rate compared with probability estimation?

### Soundness
3

### Presentation
3

### Contribution
3
