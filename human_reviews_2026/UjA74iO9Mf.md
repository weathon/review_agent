# GinSign: Grounding Natural Language into System Signatures for Temporal Logic Translation

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Natural language (NL) to temporal logic (TL) translation enables engineers to specify, verify, and enforce system behaviors without manually crafting formal specifications—an essential capability for building trustworthy autonomous systems. While existing NL–to-TL translation frameworks have demonstrated encouraging initial results, these systems either explicitly assume access to accurate atom grounding or suffer from low grounded translation accuracy. In this paper, we propose a framework for Grounding Natural Language Into System Signatures for Temporal Logic translation called GinSign. The framework introduces a grounding model that learns the abstract task of mapping NL spans onto a given system signature: given a lifted NL specification and a system signature $\mathcal{S}$, the classifier must assign each lifted atomic proposition to an element of the set of signature-defined atoms $\mathcal{P}$. We decompose the grounding task hierarchically—first predicting predicate labels, then selecting the appropriately typed constant arguments. Decomposing this task from a free-form generation problem into a structured classification problem permits the use of smaller masked language models and eliminates the reliance on expensive LLMs. Moreover, since the grounding is captured as an abstract task without hard-coding the state space, our approach can generalize to new (or modified) state spaces without retraining. Experiments across multiple domains show that frameworks which omit grounding tend to produce syntactically correct lifted LTL that is semantically nonequivalent to grounded target expressions, whereas our framework supports downstream model checking and achieves grounded logical-equivalence scores of 95.5%, a $1.4\times$ improvement over SOTA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an end-to-end grounded "natural language to temporal logic" generative model. The paper points out that existing NLP-to-TL transformation methods and models are not usable in practice due to the lacking connections between the generated atomic propositions are linked to the predicates & constants. 

For the grounding process, the paper proposes a hierarchical approach that first does predicate grounding and then argument grounding. Both processes are approached as natural language classification tasks based on BERT model, with an additional filtering step to simplify the problem. For grounding classification model, the paper proposes a non-standard formulation that aims to map a text fragment (given by the lifing model) to a candidate system signature symbol (predicted / typed constant). The classification model is obtained by fine-tuning pre-trained BERT.

The paper reports clear gains over GPT3-5/4.1 and Lang2LTL baselines with an ability to generalize over unseen predicates/constants.

### Strengths
The paper points out an important shortcoming of existing methods and proposes a sensible approach.

The experimental results are interesting, (partly) demonstrating the advantages of the method.

### Weaknesses
The results are VLTL-Bench only. 

Parts of the paper are difficult-to-read without domain expertise. Overall, I believe , the paper could have been written in a more accessible way considering that it is a submission to a primarily a machine learning conference.

While the work itself is valuable, parts of the work is clearly highly domain dependent. Personally, I find it hard to clearly estimate the degree of relevance to ICLR community, although it has interesting aspects in terms of the generalization abilities of the proposed BERT based formulation and the potentially increasing interest to formal languages in machine learning research.

### Questions
Section 5.1 claims that other datasets like Navigation / Cleanup World / GLTL are not applicable. There is also the DeepLTL dataset (Hahn et al). Even if it is not possible to evaluate in terms of grounding, is it not possible to evaluate in terms of overal LTL synthesis ability?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the problem of translating natural language (NL) specifications into temporal logic (TL) by grounding the resulting atomic propositions into system signatures. The proposed framework, GinSign, introduces a hierarchical translation pipeline that separates (1) the deduction of the logical structure (lifted temporal logic) from (2) the grounding of atomic propositions to a predefined system signature composed of types, predicates, and constants.

The key insight is that grounding NL specifications to semantically meaningful system entities produces more executable and interpretable formal specifications. The two-level grounding approach first predicts the predicate (predicate grounding) and then connects it to the appropriate arguments (argument grounding). A key claim is that by reframing grounding as a structured classification task, GinSign can employ smaller encoder-only models instead of expensive LLMs, thereby improving efficiency while maintaining accuracy. 

Experiments on the VLTL-Bench benchmark, which includes three domains (Search and Rescue, Traffic Light, and Warehouse), show that GinSign achieves near-perfect predicate grounding across all domains, outperforming GPT-3.5 Turbo, GPT-4.1 Mini, and prior systems such as Lang2LTL. For argument grounding and logical equivalence evaluation, GinSign also consistently outperforms GPT-based and NL2LTL baselines, achieving ≥ 90% grounded logical equivalence accuracy.

### Strengths
- The paper is clearly written, and the authors situate their contribution well within the growing literature on natural language to temporal logic translation.
- The two-level grounding process (first predicates, then arguments) is sound and mirrors how logical forms are constructed compositionally.
- The results show that GinSign substantially improves grounding accuracy and logical equivalence over strong LLM baselines, including GPT-4.1.
- Explicitly grounding TL specifications in system signatures could benefit downstream tasks such as automated verification, planning, or control synthesis.

### Weaknesses
- While the hierarchical design is reasonable, the technical novelty feels incremental. The improvement largely comes from providing the model with *more structured context* (the system signature and a lifted template) and recasting the task as classification, rather than introducing new learning or reasoning mechanisms.
- The approach assumes access to fully specified system signatures (types, predicates, and constants). Although the authors acknowledge this limitation, real-world domains often provide only partial or evolving ontologies, which limits the method’s general applicability.
- Because the evaluation requires access to both lifted and grounded atomic propositions, widely used datasets could not be used, which limits cross-domain validation.
- The framework relies heavily on prompt engineering and structured templates rather than principled modeling. The paper does not discuss *why* hierarchical separation or classification-based formulation helps beyond empirical evidence, or what insights generalize to other grounding or reasoning tasks.
- It remains unclear whether GinSign can operate effectively when only natural language input is available (without lifted sentences or system signatures).

### Questions
1. How would the approach perform if only NL input were available, without lifted sentences or explicit system signatures?
2. Could the model infer or learn parts of the system signature automatically, rather than relying on manual specification?
3. How sensitive are the results to prompt design or structured templates? Would smaller or less informative prompts substantially degrade performance?
4. The paper attributes most grounding errors to linguistic ambiguity. Could the authors elaborate on whether other factors, such as missing predicates in the system signature or type mismatches during argument binding, also contributed to these failures?

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
4

### Summary
This paper introduces GinSgin to address the “grounding problem” in Natural Language (NL) to Temporal Logic (TL) translation, where most existing pipelines produce "lifted" TL formulas and are not directly suitable for formal verification. The main contribution is a hierarchical grounding approach that runs after standard lifting and translation. It involves a lightweight BERT model that classifies an NL span into a system-defined predicate and then grounds the NL span into specific constants from a set of filtered “prefix”. Experiments on the VLTL-Bench demonstrate that this approach achieves high accuracy and outperforms zero-shot LLM-based baselines on the GLE metric.

### Strengths
- This paper clearly identifies and tackles a practical bottleneck of grounding in the NL-to-TL problem that most prior work overlooks.

- The proposed hierarchical decomposition is a straightforward and effective way to reduce the search space and leverage the type checking to improve overall performance.

### Weaknesses
- The entire framework depends on a known, fixed, finite, and static system signature $<T,P,C>$, which is a strong and often unrealistic assumption for open-world or evolving systems, weakening the contribution’s practical impact. The paper focuses only on grounding ambiguity, offloading logical and semantic ambiguity to upstream modules.
- The method’s effectiveness and scalability to large-scale real-world signatures are unproven. The OOD claims are not fully supported by a cross-domain generalization test, but rather by the intra-domain test (Table 5).
- The SOTA-beating claims (Table 3) are based on an unfair comparison. GinSign (fine-tuned) is compared against zero-shot, non-SOTA LLMs (GPT3.5/4.1-Mini) using a flat and plain prompt, rather than the hierarchical one that GinSign benefits from.

### Questions
- The comparison to LLMs in Table 3 appears unfair. Could the authors provide results for a stronger baseline that prompts an SOTA LLM (e.g., GPT-4o) using the same hierarchical decomposition that GinSign benefits from or other common prompting techniques (e.g., CoT, Self-reflection)?

- How do the authors expect the prefix-sharding and tournament mechanism to scale to significantly larger signatures, for example, a system with thousands of constants? What are the practical computational limits?

- The OOD experiment is intra-domain. Have the authors attempted a cross-domain generalization test (e.g., training on Traffic Light and testing on Warehouse) to validate the "transferable reasoning" claim?

- Could the authors add definitions for formal methods notation (like $\Sigma^\omega$) to improve accessibility for a broader audience?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses a critical bottleneck in natural language (NL) to temporal logic (TL) translation: the lack of semantic grounding of atomic propositions (APs) to system-specific definitions, which renders existing TL outputs syntactically valid but practically unusable for system verification. 

The proposed solution, GinSign, introduces a framework that treats grounding as a hierarchical classification problem against a formal system signature. The key innovation is a two-step process: first grounding a natural language span to a predicate from the signature, and then grounding its arguments from type-filtered constants. This is operationalized using a novel prefix enumeration technique with a masked language model (like BERT), transforming grounding into a scalable, domain-agnostic classification task. 
The authors evaluate GinSign on the VLTL-Bench dataset (Search and Rescue, Traffic Light, Warehouse domains), showing that it achieves 95.5% grounded logical equivalence (GLE) (a 1.4x improvement over state-of-the-art (SOTA) methods like Lang2LTL), 100% predicate grounding accuracy across all domains, and robust argument grounding ($\ge$ 90% $F_1$). Critically, GinSign’s outputs support downstream model checking, a capability missing in prior NL-to-TL frameworks.

### Strengths
- Originality: The formulation of grounding as a prefix-based hierarchical classification is distinct from all existing approaches (e.g., end-to-end LLM generation, embedding-similarity) and is a highly creative solution.

- Quality: The evaluation is thorough: it includes isolated grounding (to validate individual components), end-to-end translation (to assess full pipeline performance), and OOD ablation (to test generalization). The use of multiple baselines ensures that GinSign’s advantages are not overstated.

- Clarity: Complex components are explained with algorithms and examples, making the framework reproducible for other researchers.

- Significance: Unlike prior work that focuses on syntactic correctness, GinSign prioritizes semantic grounding—this shifts NL-to-TL from a "theoretical exercise" to a tool that can be integrated into formal verification workflows for autonomous systems.

### Weaknesses
- Limitations of Evaluation Domains: The empirical validation is confined to the VLTL-Bench. While it contains three distinct domains, the scale and complexity of the system signatures may not fully represent large-scale, real-world systems (e.g., full autonomous vehicle specifications). Performance and scalability on signatures with orders-of-magnitude more constants remain an open question.

- Limited TL Coverage: GinSign only supports propositional LTL. The paper mentions extending to metric LTL (with time bounds) or first-order LTL (with quantifiers) as future work, but it does not discuss the technical challenges of grounding for these variants. 
Constant Grounding Bottleneck in Warehouse: While GinSign’s Warehouse argument grounding (94.2% $F_1$) outperforms baselines, it is lower than in other domains (Traffic Light: 97.9%, Search and Rescue: 91.1%). The authors attribute this to lexically diverse constants but do not provide a detailed error analysis (e.g., which constants are most often misgrounded, why). Such analysis could guide targeted improvements.

- Dynamic Signature Handling: The framework assumes static system signatures at inference time. For systems where signatures evolve (e.g., adding new predicates/constants), GinSign would require retraining or reconfiguring the prefix—no strategy for incremental adaptation is proposed.

### Questions
1. (About Constant Grounding Error Analysis) For the Warehouse domain, could you provide specific examples of misgrounded constants and explain why the current framework fails in these cases?
2. (About Scalability to Large Signatures) For systems with thousands of constants (e.g., a warehouse with 1,000 unique items), the current shard-based classification may become inefficient. Have you explored retrieval-augmented methods? 
3. (About Dynamic Signature Adaptation) For systems where new predicates/constants are added post-deployment, how would you update GinSign without full retraining? Could the prefix-enumeration mechanism be combined with few-shot learning to adapt to new signature elements?

### Soundness
3

### Presentation
3

### Contribution
3
