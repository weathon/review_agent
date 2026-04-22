# PEML: Prototype-enhanced Meta-learning for Multi-lingual Text Classification

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Multi-lingual text classification is a challenging task in natural language processing, which not only faces language differences between multiple languages but also faces the challenge of scarce annotated data. This paper proposes a prototype-enhanced meta-learning (PEML) method to address the challenges in the multi-lingual text classification task. The PEML method consists of two steps: firstly, to enhance the model's ability to understand multi-lingual samples, we design a multi-lingual label-fusion technique to better map labels from different languages into a unified semantical space; secondly, in response to the problem that class prototypes for support sets are difficult to apply to query sets in meta-learning, we use a query-enhanced technique to associates the prototype vectors of the support set with samples in the query set. After training with our method, the classification model can quickly update the class prototypes to the data distribution of the query set, thereby expanding the model's multi-lingual classification ability from the support set to the unseen query set. Extensive experiments demonstrate that the proposed method significantly outperforms state-of-the-art methods in multi-lingual text classification tasks. The code and data of this paper will be released on GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes PEML, a prototype-enhanced meta-learning framework for few-shot multilingual text classification. It introduces a label-fusion module to align label semantics across languages and a query-enhancement mechanism that refines class prototypes using unlabeled query samples with language-balanced selection and optimal transport. Without relying on prompts or external knowledge, PEML effectively mitigates cross-lingual variance and support-set sparsity, and achieves new state-of-the-art performance on multilingual few-shot benchmarks.

### Strengths
Originality
- Proposes a prototype-enhanced meta-learning framework (PEML) specifically for multi-lingual few-shot text classification — a setting where most prior prototype-based meta-learning work assumes monolingual or does not explicitly address cross-lingual semantics.
- Introduces two conceptually new components targeted to the multilingual regime:
  - Label-fusion module to align label semantics across languages into a shared embedding space.
  - Query-enhancement via prototype update using optimal transport with language-balanced augmentation — a novel way to inject query information into prototype construction beyond support samples.
- The originality arises not from inventing a completely new paradigm but from composing known ideas (contrastive alignment, adapters, prototype update) into a multilingual few-shot meta-learning design.

Quality
- Method design is technically reasonable and addresses known failure modes of prototypical networks in multi-lingual and few-shot conditions (prototype instability, semantic misalignment).
- The ablation table isolates each component (LF, LA, CL, QE, MLA, PU) and shows consistent monotonic degradation when removed, supporting causal contribution claims.
- Comparisons include a broad and relevant set of baselines (prompting, PLMs, meta-learning, multilingual LLMs), and PEML consistently wins across 2/4/8-shot regimes.
- Hyperparameter sensitivity (e.g., R) and visualization further support claims.
- No obvious methodological flaw is apparent in the described pipeline; design choices are justified and empirically validated.

Clarity
- The paper is generally well-structured: clear abstract, motivation, modular breakdown, and conceptual figure.
- The narrative from challenge → design principle → module → loss → experiments is coherent and easy to follow.
- Mathematical notation is consistent and sufficiently detailed; the prototype update description is unusually precise for an applied meta-learning paper.
- Ablations and visualization are clearly presented and interpretability of results is high.

Significance
- Addresses a high-value, real-world setting: multilingual classification under low-resource supervision.
- Shows notable and consistent gains over competitive SOTA methods.
- The approach is generalizable: similar principles could transfer to other multilingual meta-learning problems (e.g., QA, IE), suggesting broader methodological impact.

### Weaknesses
Novelty of some components is arguable / not positioned clearly
- Both contrastive alignment of multilingual labels and query-informed prototype refinement have precedents (e.g., LAQDA for using query signals to adjust prototypes; multilingual contrastive alignment from MFLSCI / SKPT lines). The paper does not explicitly discuss why existing multilingual contrastive approaches (e.g., MFLSCI 2025; SKPT 2024) are insufficient, nor contrast its design against LAQDA’s query-aware prototype mechanism for the same failure mode.

Evaluation scope is not fully commensurate with the claims
- The main table evaluates only on three languages and a single dataset family; yet the paper claims “applicable to multilingual few-shot in general”. Results on macro-typologically distant languages (e.g., morphologically rich languages, right-to-left, low-script resources) are absent — which weakens the generalization claim. Add at least a second dataset with different linguistic classes.

Interpretation of gains is overly attributional without causal stress tests
- Improvements are attributed to prototype enhancement, yet there is no stress test where prototypes are intentionally perturbed, or tasks with increased intra-class language divergence, to show robustness specifically comes from PU+QE not from model capacity.

Missing computational and complexity discussion
- The method adds multiple components (adapters, contrastive stage, optimal transport) but does not quantify overhead vs. prior meta-learning baselines. For meta-learning methods used in ultra-low-resource contexts, efficiency cost is a first-order concern. Report training latency / memory and FLOPs comparisons against PBML/LAQDA and discuss trade-offs.

No comparison to LLM-based multilingual prompting with synthetic labels
- The paper claims SOTA over multilingual LLM baselines (e.g. Sailor2), but does not compare to LLM-based zero/few-shot prompting with translated synthetic exemplars, which is the most realistic practitioner baseline today. Add a baseline: GPT-4/5 / Sailor with 3-shot translated demonstrations or CoT verbalizers, to validate that PEML is competitive beyond PLM + meta-learning baselines.

### Questions
- Does PEML require label names in all languages explicitly at both train and test time? If not, how is the model used when label names exist only in one language (which is the norm in real deployments)? Do authors have a fallback variant / ablation?
- The claims are made toward “multilingual MLTC” in general. Can the authors either
  - justify that EN–ZH–VI is sufficiently representative of typological diversity, or
  - commit to experiments on at least one non-fusional / non-analytic language (e.g., Arabic / Hindi / Turkish) in the camera-ready?
- Could the authors add a baseline where GPT-style multilingual LLM is given a translated few-shot prompt for comparison? If not, can they provide a justification why this is outside scope for MLTC today, given that LLM few-shot prompting is by far the most common industrial baseline?

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
The paper proposes a meta-learning framework for multilingual text classification. The framework include a multilingual label-fusion part, which maps labels in different languages into a unified semantic space, and a query enhanced part to associate the class prototype vectors with query set samples. Experiments show advantages over several baselines.

### Strengths
The pipeline seems reasonable for multilingual tasks.

### Weaknesses
1. The paper fails to clarify the basic formulation, especially for the multilingual part. (sec 3.1 does not introduce any notation or setting related with multiple languages).

2. The paper does cite several papers on meta-learning, but does not include any comparison to multilingual methods, for example, [1] and [2].

3. The experiments are conducted on privately collected data, which makes reproductivity impossible, while there are multilingual tasks in many different aspects [3].

4. The organization may need to be improved. Currently many important aspect of the experiments, e.g. data, baselines, are left in the appendix.

[1] Investigating meta-learning algorithms for low-resource natural language understanding tasks.

[2] Meta-Learning for Effective Multi-task and Multilingual Modelling

[3] XTREME: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation.

[4] PAWS-X: A cross-lingual adversarial dataset for paraphrase identification.

### Questions
No further questions.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Prototype-Enhanced Meta-Learning (PEML), a novel method specifically designed to address the key challenges in Multi-lingual Text Classification (MLTC) under the few-shot scenario. This task is difficult due to language differences between multiple languages and the scarcity of annotated data.

The core motivation for PEML is overcoming the limitations of standard Prototypical Networks (PN) in MLTC, where intra-class language differences make class prototypes easily lack sufficient representativeness, leading to incorrect query sample classification. The proposed solution is a metric-based meta-learning approach that efficiently learns class prototype vectors by simultaneously updating category information and language information

### Strengths
The paper designs a multi-lingual label-fusion technique that uses a Language Adapter (LA) and a language-weighted Network to better map labels from different languages (demonstrated with Chinese, English, and Vietnamese) into a unified semantical space. This approach is further reinforced by a cross-lingual contrastive loss $l_{contra}$  which encourages semantic alignment of the same label across different languages. Extensive experiments confirm that PEML significantly outperforms state-of-the-art methods under multi-lingual few-shot scenarios.

### Weaknesses
1. All experiments utilize BERT-base-multi-lingual-uncased (mBERT) as the backbone encoder. While mBERT is a standard baseline, its cross-lingual alignment capabilities are known to vary.

2. The authors should test PEML using a different class of multi-lingual encoders, such as a code-switching architecture or a highly aligned multi-lingual model (if applicable and fair to baselines), to demonstrate that the performance gains are derived specifically from PEML's fusion and enhancement mechanisms, rather than relying on mBERT's intrinsic multi-lingual features being marginally optimized by the adapter structure.

### Questions
1. Since PEML and LAQDA share the goal of using query samples for prototype optimization, the authors should conduct an ablation study specifically quantifying the isolated performance gain derived from the language-balanced sampling within the PU module. This would confirm that the superior robustness is due to this language-aware strategy and not just the sheer inclusion of auxiliary query samples.

2. All reported few-shot experiments, including the main results (Table 1) and ablation studies (Table 2), are conducted strictly under the 2-way K-shot setting (e.g., 2-way 2-shot, 2-way 4-shot, 2-way 8-shot). Why was the evaluation limited to N=2? Since the goal of the Multi-lingual Label Adapter (MLA) is to enhance discriminative capacity, evaluating PEML under a higher N-way setting (e.g., 5-way K-shot) would provide a more rigorous test of the model’s ability to separate multiple complex, unseen categories simultaneously.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
An improved meta-learning framework for multilingual text categorization is proposed. The method is based on prototype networks, using a number of improvements to handle the multilingual aspect and adapt to the meta-learning framework. Experiments on (proprietary?) data show clear improvements over a number of alternatives, and further analysis is provided, including an ablation study of the many components in the proposed framework.

### Strengths
Efficient multilingual text classification is definitely a relevant task in a increasingly multilingual, connected world. The few-shot setting is also very sensible in a multilingual context. Experimental results support the claim of increased performance, with clear gains in extreme few-shot situations.

### Weaknesses
Minor point: The case for meta-learning of text classification has always seemed a bit artificial. Providing a true real-life example where it would be needed would greatly help.

The main weakness is clarity and presentation in both the methodology and experimental sections. With the notable exception of Sec. 3.4.2, there is little detail on the various parts of the framework. As an example, the Language Adapter (Sec. 3.3.1) is described as a down-projection, non-linear transformation and up-projection. Neither is described further -- what are the parameters, how are the projections chosen, optimized on what? The notation is sometimes confusing. For example L is the input sentence length in 3.3, but possibly the number of support examples in Fig. 1 (unless all examples have the same length, which would be odd). K is either the number of supports per language, or the self-attention key vector (similarly Q). The paper is definitely not self-contained re. experimental detail. There is no detail on how the nine alternatives (mBERT...Sailor2) are trained or used, even in the appendix. Finally, although the observed differences in performance seem highly likely to be significant, it is hard to overlook the lack of either statistical significance testing or assessment of the uncertainty in experimental results.

### Questions
(l.107) Is the claim of better generalization based solely on the limited experiments, or is there some theoretical support? Also l.289-290.

(3.4.1) Can you clarify where the multilingual aspect is taken into account in this process? This sounds like a standard self-attention setup.

(l.290) How many "remaining slots" can there be after sampling three times R/3 samples?

(l.422) Claim is questionable: if anything, the MLA seems to have the smallest impact and hardly support the claims here ("introduce semantic information", "more task-consistent representation space"...)

(l.455) Is the claim of stronger compactness supported in any way? Visually there is little impact.

Typos:

l.063: easily be applied -> easy to apply

l.104: there are few works -> there is little work

l.463: as -> are (shown)

### Soundness
2

### Presentation
1

### Contribution
3
