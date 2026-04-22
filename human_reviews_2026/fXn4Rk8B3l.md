# TamperBench: Systematically Stress-Testing LLM Safety Under Fine-Tuning and Tampering

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
As increasingly capable open-weight large language models (LLMs) are deployed, improving their tamper resistance against unsafe modifications, whether accidental or intentional, becomes critical to minimize risks. However, there is no standard approach to evaluate tamper resistance. Varied data sets, metrics, and inconsistent threat settings make it difficult to compare safety, utility, and robustness across different models and defenses. To this end, we introduce TamperBench, the first unified framework to systematically evaluate the tamper resistance of LLMs. TamperBench (i) curates a repository of weight-space fine-tuning attacks and latent-space representation attacks; (ii) allows for testing state-of-the-art tamper-resistance defenses; and (iii) provides both safety and utility evaluations. TAMPERBENCH requires minimal additional code to specify any fine-tuning configuration, alignment-stage defense method, and metric suite while ensuring end-to-end reproducibility. In this work, we use TamperBench to evaluate 21 open-weight LLMs, including defense-augmented variants, across nine tampering threats using standardized safety and capability metrics with hyperparameter sweeps per model-attack pair.  Code is available at: https://anonymous.4open.science/r/TamperBench-71DD/README.md

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents TamperBench for evaluating the tamper-resistance of LLMs. The method uses a standardized set of attacks on model weights and representations, which measurs the impact on safety with the evaluator and on capability with the benchmark. For each attack, hyperparameter optimization is used to find configurations on the safety-utility pareto front, from which a final configuration is selected for evaluation. The method is applied to benchmark 19 models against nine attack types.

### Strengths
* Originality: This paper introduces a unified framework for systematic LLM safety evaluation, and a formal threat taxonomy for tampering.

* Clarity: The paper is well-structured and clearly defines its core concepts.
* Significance: It provides a standardized tool to establish a potentially foundational benchmark for the field.

### Weaknesses
* This paper's conclusions about the trade-off between safety and utility may not be generalizable to real-world scenarios where these concepts are defined more broadly. The approach's assessment of safety and utility hinges on two metrics: StrongREJECT and MMLU-Pro. Consequently, safety is reduced to a single harm score, which potentially ignores subtle biases, while utility is narrowly defined as multiple-choice knowledge, which ignores other critical capabilities like coding, reasoning, or creativity.
*  The methodology selects the attack configuration from the pareto front that maximizes StrongREJECT score. This could be a questionable design choice, as it inherently favors the most damaging attack, regardless of how catastrophically it might degrade model utility. This may not represent a realistic attacker's goal (who might prefer stealth) and biases the benchmark towards showcasing easily detectable, high-impact failures rather than more insidious, utility-preserving tampering.
* The paper's conclusion about tamper-resistance are fundamentally constrained by its predefined nine attacks. The framework can only measure resistance to known threats, which makes its broader claims about general tamper-resistance potentially overstated.

### Questions
* Regarding Weakness 1, why were StrongREJECT and MMLU-Pro considered sufficient proxies for safety and utility? have the authors considered the risk that tampering could specifically preserve these metrics while degrading other unmeasured, but equally important, qualities? for example, an attack could degrade a model's ability to write code or reason through a problem while leaving its MMLU-Pro score largely intact, which leads the benchmark to incorrectly assess the damage.
* Regarding Weakness 2, what is the justification for selecting the single point of maximum harm from the pareto front? why is this specific point considered more representative for the benchmark than any other? moreover, how does this selection criterion allow for a meaningful evaluation of a model's resistance to the "covert stealthy tampering" threat that the paper itself identifies as a key concern?
* Regarding Weakness 3, how can the authors be confident that performance against this specific nine attacks is a reliable indicator of a model's general resilience against novel, yet-to-be-discovered attack methods? 

the paper claims the framework is extensible. then what is the precise process for a third-party researcher to integrate a new attack into the benchmark and ensure that the results are comparable? is there a governance model?

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
3

### Summary
The paper presents TamperBench, a unified framework for evaluating the tamper-resistance of open-weights LLMs under several weight-space and latent-space attacks, and defenses. In contrast to other works where attacks, defenses and metrics are evaluated in isolation, TamperBench standardizes the threat model and pipeline. Several open-weight models are evaluated across 9 tampering strategies, and results report the Pareto configuration that maximizes the StrongREJECT safety metric while also showing the MMLU-Pro capabilities metric. To reduce the variance of non-embedding attacks, the authors run multiple trials. Overall, the results show that tampering generally increases harmfulness and often reduces capabilities, highlighting a safety vs capability tradeoff under tampering attacks.

### Strengths
- The benchmark consolidates attacks, defenses and evaluation in a single unified framework, with a clear standardization of threats settings.
- The evaluation is very broad and covers 19 models and 9 attack strategies
- Having multiple trials for non-embedding attacks is very helpful and greatly reduces the variance.

### Weaknesses
- For a “large-scale” benchmark, the included models appear relatively small. It would be helpful to have a discussion about scaling to bigger models.
- For some attacks, data are very small (LoRA fine-tuning with 64 examples). These settings are fine, but could amplify variance.
- Picking Pareto points that maximize StrongREJECT is a conservative metric, but it can lead to a misinterpretation of the results when the capabilities drop sharply.
- Having other safety and capabilities metrics can help make the results more clear.

### Questions
- I wonder if you tried other metrics other than StrongREJECT and MMLU-Pro before choosing those. In general, it would be interesting to know if the conclusion changes with different safety and capabilities metrics, especially with a discussion on picking the Pareto points. An option could be to report "constrained" Pareto points, with a constraint on the capabilities drop.
- It would be helpful to run some attacks under bigger datasets, like the LoRA fine-tuning attack.
- Do you expect larger models to exhibit different safety vs capabilities trade offs?

### Soundness
3

### Presentation
3

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
This paper introduces TamperBench, a unified, extensible benchmark for evaluating the tamper resistance of open-weight large language models (LLMs). The authors include several existing benchmarks in helpfulness and safety evaluation such as MMLU-Pro, or Strongrejct and test 19 models in their unified benchmarks.

### Strengths
1. The evaluation of LLMs safety and helpfulness is crucial. And a unified framework to evaluate them could largely save the efforts in reproducing different env settings.

2. This paper include white-box, black-box,  latent-space representation, and fine-tuning attacks, which covers a wide range.

3. The presentation is easy to read and follow.

### Weaknesses
1. Lack of novelty. This paper seems to only combine existing helpfulness and safety evaluation metric together, and there are not new metrics or benchmarks.

2. I think the main contribution comes from the re-organization of existing benchmarks. However, some examples in this paper such as MMLU, StrongREJECT have well-structured open-source code, making them easily to employ and test. I don't think re-organize them have saved a lot of time costs rather than directly use their official code. This seems trivial.

3. Lack of human evaluation. For a safety evaluation benchmark, you should at least personally check the cases in case of false positive rate.

4. Apart from this re-organization, is there any other contributions or any other problems you find existing benchmarks have and solve in this paper?

### Questions
Please see the weakness.

### Soundness
3

### Presentation
2

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
This paper identifies a key gap in large language model (LLM) safety research: the lack of standardized evaluation for tamper resistance. The authors argue that current methods for testing defenses against model modifications (like fine-tuning attacks) are ad-hoc, using varied datasets, metrics, and threat settings, which makes fair comparison impossible.

To solve this, they introduce TamperBench, a unified and extensible benchmark framework designed to systematically stress-test the safety of open-weight LLMs against tampering.

Overall, the idea behind this paper is good, and I think it could become a strong benchmark, but due to the number of weaknesses identified I cannot recommend it for acceptance at this time. If the authors address my concerns, I would be willing to raise my score.

### Strengths
- Even though there are some key missing citations (see below), this paper is fairly extensive with its citations.
- The paper develops a standardized evaluation framework for an important class of attacks (representation-space and weight-space attacks), which differ from the input-space attacks most often considered in LLM robustness evaluations.
- Hyperparameter sweeps for the fine-tuning attacks is a crucial consideration (although see concern below about needing different optimizers). Too many papers in this area only consider one or a small number of fine-tuning attacks.

### Weaknesses
Weaknesses:
- For the fine-tuning attack hyperparameters, it would have been good to consider different optimizers as well as different hyperparameters.
- I don't know if defending against weight-space and representation-space attacks should be grouped together as "tamper-resistance". Tamirisa et al. introduced the term tamper-resistance in the context of fine-tuning attacks, and it seems useful to maintain precision in terminology (even though some latent space attacks can be considered subsets of fine-tuning attacks).
- Figure 5 is indecipherable. StrongREJECT and MMLU-Pro are mixed together in the figure with no clear distinction, it's not clear which numbers are better, and the color coding on the right side of the plot have two scales each with no explanation!
- Why are actual methods that try to improve tamper-resistance and latent space robustness not benchmarked? E.g., there are no numbers for LAT or TAR.
- Section 2 is confusingly structured. It would be cleaner to structure it as different "threat models" that are considered in TamperBench, each one clearly stating the assumptions, knowledge/resources, and goals of the attacker and defender. Currently it uses nonstandard terms like "threat setting" and "tamper-resistance goals and defenses". Granted, non-adversarial/accidental tampering wouldn't fit into a threat model framing. Overall, though, section 2 could be presented more clearly.
    - For example, Section 2.3 is about the gap in the literature that TamperBench seeks to fill. This belongs in a related work section, not alongside the threat models! Readers won't expect to find it here, which makes following along harder.
- I don't really agree with the main taxonomy in Figure 3. Why is malicious tampering described as "disregards API protections". Isn't the most common setting here fine-tuning open-weight models? If so, it's kind of weird to describe this as disregarding API protections, since there is no API to speak of. I wouldn't want the community adopting this taxonomy as currently written. It might make more sense to have a 2x2 grid: intentional vs unintentional tampering and API vs open-weight tampering. It seems like those are the two relevant axes being described in this paper. Maybe a third axis is overt vs covert tampering.

Missing citations:
- "Self-Destructing Models: Increasing the Costs of Harmful Dual Uses of Foundation Models" from Peter Henderson et al. (a key early paper that I'm surprised is not cited)
- "Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks" and "Forgetting Outside the Box: Scrubbing Deep Networks of Information Accessible from Input-Output Observations" from Aditya Golatkar et al. (some of the first papers describing resistance to fine-tuning attacks as a key metric for unlearning)

Suggestions (not counting toward score):
- It would be interesting to discuss the continued relevance of API-based tampering attacks, since AI companies are really focusing on continual learning and long-term memory, which may end up being addressed with something like per-user PEFT.
- Please use `` and '' for quotations in LaTeX. E.g., "may emerge when ”safe”" on line 155 uses incorrect quotations.

### Questions
See weaknesses

### Soundness
2

### Presentation
1

### Contribution
2
