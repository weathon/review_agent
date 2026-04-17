# Atom-anchored LLMs speak Chemistry: A Retrosynthesis Demonstration

- Decision: Reject
- Scores: 4, 6, 6

## Abstract
Applications of machine learning in chemistry are often limited by the scarcity and expense of labeled data, restricting traditional supervised methods. In this work, we introduce a framework for molecular reasoning using general-purpose Large Language Models (LLMs) that operates without requiring labeled training data. Our method anchors chain-of-thought reasoning to the molecular structure by using unique atomic identifiers. First, the LLM performs a zero-shot task to identify relevant fragments and their associated chemical labels or transformation classes. In an optional second step, this position-aware information is used in a few-shot task with provided class examples to predict the chemical transformation.
We apply our framework to single-step retrosynthesis, a task where LLMs have previously underperformed. Across academic benchmarks and expert-validated drug discovery molecules, our work enables LLMs to achieve high success rates in identifying chemically plausible reaction sites ($\geq$90%), named reaction classes ($\geq$40%), and final reactants ($\geq$74%). 
Ultimately, our work establishes a general blueprint for applying LLMs to challenges where molecular reasoning and molecular transformations are key, positioning atom-anchored LLMs as a powerful solution for data-scarce chemistry domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a novel, label-free framework for molecular reasoning using general-purpose LLMs that operates without requiring labeled training data. LLM's chain-of-thought is applied to the molecular structure by using unique atomic identifiers, which enables the one-shot identification of relevant chemical fragments and transformation classes for the retrosynthesis task. This approach is highly significant as it bypasses the need for large, labeled datasets. The algorithm achieves high success rates test on both academic benchmarks and expert-validated drug discovery molecules.

### Strengths
The paper is well written, and the figures are helpful.

### Weaknesses
See questions

### Questions
Can this idea be generalized to more complicated tasks, e.g., molecules?

My understanding is that the LLM is not trained on chemical data but only on language data. What is the intuition that LLM transfers its knowledge to such a different area, without finetuning?

What is the inference cost of this LLM approach compared to existing, traditional approaches?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an atom-anchored reasoning framework that enables general large language models to perform chemically interpretable retrosynthesis without any task-specific training data. By introducing atom-level mappings and a two-stage chain-of-thought process, the model achieves strong performance in identifying reaction centers, classifying reaction types, and predicting reactants, demonstrating its capability to reason over molecular structures in a transparent and generalizable way.

### Strengths
* The framework requires no labeled data, making it highly adaptable to data-scarce chemical domains.

* The atom-level anchoring enhances interpretability and aligns model reasoning with real chemical logic.

* The approach generalizes across benchmarks and real drug molecules, showing practical applicability.

### Weaknesses
* The model's performance relies heavily on well-crafted prompts and example selection.

* It struggles with edge cases involving unusual reaction mechanisms or rare functional groups.

### Questions
* Can the method generalize to multi-step synthesis and stereoselective reactions?

* Would fine-tuning on limited chemistry-specific reasoning traces further improve accuracy?

### Soundness
3

### Presentation
3

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
The paper proposes a method for leveraging LLMs for tasks in molecular analysis and specifically single-step retrosynthesis. The method relies on atom-mapping syntax for SMILES strings, and uses LLMs as evaluators at the level of atoms and bonds. The framework then uses these fine-grained evaluations to judge the potential for disconnection, and then to predict the reaction that could be used to materialize the disconnection.
Interesting idea overall.

### Strengths
The authors provide a new method to improve the analytic capabilities of LLMs, and show that this improves dramatically their applicability on complex chemical analysis tasks.
the evaluation is thorough and uses public benchmarks and datasets, evaluates different LLMs both closed, open and across multiple scales. the prompts are released as well which contributes to reproducibility.

### Weaknesses
Looking at the sizes of the datasets used for evaluation, it would be important to do a discussion on costs and latencies of the proposed methodology. For instance, how many LLM calls are required per problem? how does it scale? e.g. it seems like the method scales with the number of atoms in the molecule. Please discuss this further.

The authors describe the advantages of the method more as a way of collecting synthetic data given how scarce real data is in specific fields. It would be a very interesting contribution, and would really make a better case for this claim, if some generated dataset was released for use of the community. Or if you trained another model, or analyzed the data, etc. Something related to creating a dataset and demonstrate its usefulness. I believe a lot of data was already created during the creation of the results presented here, so shouldn't be very problematic.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
