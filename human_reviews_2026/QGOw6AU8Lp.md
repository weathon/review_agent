# BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
LLM-as-a-Judge has been widely adopted across various research and practical applications, yet the robustness and reliability of its evaluation remain a critical issue. A core challenge it faces is bias, which has primarily been studied in terms of known biases and their impact on evaluation outcomes, while automated and systematic exploration of potential unknown biases is still lacking. Nevertheless, such exploration is crucial for enhancing the robustness and reliability of evaluations. To bridge this gap, we propose BiasScope, a LLM-driven framework for automatically and at scale discovering potential biases that may arise during model evaluation. BiasScope can uncover potential biases across different model families and scales, with its generality and effectiveness validated on the JudgeBench dataset. It overcomes the limitations of existing approaches, transforming bias discovery from a passive process relying on manual effort and predefined bias lists into an active and comprehensive automated exploration. Moreover, based on BiasScope, we propose JudgeBench-Pro, an extended version of JudgeBench and a more challenging benchmark for evaluating the robustness of LLM-as-a-judge. Strikingly, even powerful LLMs as evaluators show error rates above 50\% on JudgeBench-Pro, underscoring the urgent need to strengthen evaluation robustness and to mitigate potential biases further.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem of bias in the "LLM-as-a-Judge" evaluation paradigm, particularly in the context where existing evaluation methods are failing due to the safety guardrails of modern models. The authors propose BIASSCOPE, an automated, LLM-driven framework for systematically discovering unknown evaluation biases. The framework operates via an iterative loop: a "teacher" model uses known biases to perturb a dataset, and then analyzes the errors made by the "target" judge model to identify and propose new potential biases. These new biases, after being validated by confirming they increase the target model's error rate, are added to an expanding bias library for subsequent iterations. The authors also created a more challenging new benchmark, JudgeBench-Pro, by augmenting the existing JudgeBench. Experiments show that even the most powerful LLMs exhibit error rates that often exceed 50% on JudgeBench-Pro, revealing a significant vulnerability in the robustness of current judge LLMs.

### Strengths
**1. Novel Framework for Automated and Scalable Bias Discovery**

The paper's primary contribution is BiasScope, a novel framework for automatically discovering previously unknown biases in LLM-as-a-Judge. It moves beyond the limitations of prior work, which largely focused on verifying a small set of known biases. The proposed iterative, two-phase process where a teacher model identifies new biases by analyzing a target model's errors is a significant methodological innovation. This provides a scalable and systematic approach to exploring the vast and complex space of potential evaluation biases.

**2. Valuable Contribution of a New, Challenging Benchmark**

This work creates JudgeBench-Pro, a new and more challenging benchmark for evaluating the robustness of judge LLMs. By augmenting an existing dataset with the diverse biases uncovered by BiasScope, the authors have developed a valuable resource for the community. The striking experimental result is a high-impact finding that clearly demonstrates the severity of the bias problem and the utility of JudgeBench-Pro.

**3. Thorough Experimental Validation**

The paper is supported by a comprehensive and rigorous set of experiments. Beyond validating the core framework, the authors provide valuable ablation studies on design choices, such as the impact of the teacher model's capability and the effectiveness of the strategy. Crucially, the paper "closes the loop" by demonstrating that the biases discovered by BiasScope can be used to create augmented data for DPO training, which in turn successfully mitigates the model's biases. This end-to-end demonstration strongly validates the practical utility and real-world relevance of the proposed framework.

### Weaknesses
**1. Heavy Reliance on a teacher model**

The framework's validity heavily relies on the capabilities of the "teacher" model. Critical steps, such as identifying a new bias from an error explanation, defining it, and merging it with existing biases, are delegated to this single LLM. This process lacks scientific rigor and its objectivity is questionable, as it is unclear whether the framework is discovering fundamental biases of the target model or simply distilling the inherent quirks and latent biases from the teacher model itself.

**2. Instability and Path Dependency of the Iterative Discovery Process**

The iterative, self-expanding nature is susceptible to instability. The feedback loop could potentially amplify noise: an incorrectly identified "bias" in an early iteration might be added to the library, leading the framework to discover more "pseudo-biases" based on the initial error. Furthermore, the entire discovery process exhibits strong path dependency on the small initial set of seven biases, which questions the comprehensiveness and universality of its findings.

**3. Limited Scope of Discovered Biases (Absence of Social Dimension)**

Following the weakness 2, despite the general term "bias," the biases discovered are almost exclusively cognitive or stylistic in nature. The framework, in its current implementation, does not appear to effectively discover the more subtle and critical societal biases (e.g., related to gender or race) that are a central concern in AI fairness research.

**4. Failure to Analyze Biases in Critical Closed-Source Models**

The paper's bias discovery process was exclusively applied to open-source models. While closed-source models were evaluated on the final JudgeBench-Pro benchmark, they were not used as "target models" within the framework itself. This is a major limitation, as the most widely used and influential "LLM-as-a-Judge" systems are proprietary.

**5. Lack of Transparency in the Construction of JudgeBench-Pro**

While JudgeBench-Pro is a key contribution, the paper lacks transparency regarding the manual verification process. The authors state that samples were "manually verified" to ensure misjudgments stemmed from bias, but provide no details on the annotation protocol, the number of annotators, their expertise, or inter-annotator agreement. This opacity makes it difficult to independently assess the quality and validity of this new benchmark.

### Questions
**1.** The discovery process demonstrates a path dependency on the initial set of seven cognitive biases and subsequently identifies mostly cognitive or stylistic biases. Have the authors considered starting with a different seed set of biases, for instance, social biases related to gender or race?

**2.** The bias discovery framework was exclusively applied to open-source models. While acknowledging potential cost constraints, could the authors discuss the feasibility of analyze critical closed-source models? Do they hypothesize that these models would exhibit a different set of inherent biases compared to their open-source counterparts?

**3.** For the manual verification step in the construction of JudgeBench-Pro, could the authors provide more details on the annotation protocol? Specifically, information on the number of annotators, their expertise, and the IAA would be critical for assessing the benchmark's quality and validity.

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
4

### Summary
The authors propose BIASSCOPE, a LLM-driven framework for automatically and at scale discovering potential biases that may arise during model evaluation. The authors claim that BIASSCOPE can uncover potential biases across different model families and scales, with its generality and effectiveness validated on the JudgeBench dataset. Finally, they introduce JudgeBench-Pro, an extended version of JudgeBench and a more challenging benchmark for evaluating the robustness of LLM-as-a-judge.

### Strengths
* This work has a clear and effective definition of what it seeks to control, namely -- "Systematic, non-random tendencies exhibited by a Judge LLM during answer evaluation, which can lead its assessments to deviate from objective and equitable standards, thereby affecting the robustness and reliability of the evaluation". Their validation methodology (page 4, lines 184-198) operationalizes this: if injecting a bias into an incorrect response causes judges to choose it more often, that bias has impact.
* The authors have discovered some genuinely interesting biases which are underexplored in the current literature, including novelty bias, exact match bias, and authority bias. It makes intuitive sense that LLM judges would exhibit these biases.
* The choice of the JudgeBench dataset was sound for most of the experiments presented in the paper.

### Weaknesses
I could certainly be persuaded that this paper is ready for ICLR, but I think there are enough experimental gaps that I cannot fully endorse it as-is.

* All biased responses are generated by Qwen2.5-72B, which has its own errors, biases and preferences. For instance, Table 1 includes 4 Qwen models (Qwen2.5-1.5B, Qwen2.5-7B, Qwen2.5-14B, Qwen3-8B), so the authors may inject self-preference bias automatically alongside whatever biases they discover. At the very least, the authors should ablate this choice and ensure that the generative model never matches the downstream judge to avoid self-preference bias. Biased responses in JudgeBench-Pro are also longer which can introduce length bias, and their ablation study is on a different dataset than their main results. Why not do the length ablation on the JudgeBench dataset? 
* Why are the sets of models in Table 1 (main findings on judgebench) and Table 9 (judgebench-pro adversarial dataset) non-intersecting? Table 1 validates only using weak judges which have very high base error rates, close to random chance. Table 9 / Figure 3 evaluate stronger models, but only use the adversarial dataset which seems to be very hard. This latter doesn't tell us much about the expected case when the judge model is strong, which is what many practitioners will care about. This issue is quite central to understanding whether BiasScope is actually discovering real, meaningful biases that persist even when the judges are strong / frontier models.
* Table 5 shows that the authors' method injects correctness noise, causing chosen and rejected answers to become equal. This is a significant problem and it is not even necessary. Since the authors used JudgeBench, which includes GT answers and extraction mechanisms, it should be possible to simply extract anything that looks like a final answer from the response before transforming it, then transform it, then add back the final response. This would guarantee that the transforming LLM did not alter the final answer. In fact, one could even do it without exposing what the original question was and do post-processing filtration to ensure no new answers were accidentally injected during the transformation.
* The authors' proposed solution is not applicable to deviations from "equitable standards", as the authors indirectly claim in their introduction by first defining bias as a devation from "equitable standards" and then stating their method can automatically discover biases. While this claim appears valid for deviations from objective standards (which cause models to be less correct), in order to cover equitable standards, it would need to be shown that BiasScope can discover biases that emerge given equally correct answers with different meta-properties such as demographic, cultural or stylistic divergence as well. The bias library in Section H omits such biases. This weakness could be addressed simply by changing the introduction a bit.
* (nit) The caption on Table 1 is too shallow; it should explain briefly the original measurement scheme of JudgeBench, the adversarial delta strategy employed in BiasScope, where random chance is (50%).

### Questions
* Biased responses in JudgeBench-Pro are longer which can introduce length bias, and their ablation study is on a different dataset than their main results. Why not do the ablation on JudgeBench again?
* Why not use mechanical answer preservation via extraction?
* Why not use diverse transformation LLMs to avoid self-bias (e.g., use Llama for Qwen judges, Qwen for Llama judges)?
* Why not a 2-pass transformation with the second pass condensing extremely long answers to avoid the length confound?

### Soundness
2

### Presentation
4

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
The paper introduces BiasScope, an automated framework for discovering and validating hidden biases in LLM-as-a-Judge systems. It operates in two phases—bias discovery and bias validation—using a teacher model to identify and confirm potential biases. The authors also build JudgeBench-Pro, a new benchmark revealing that even advanced models (like GPT-4o) remain highly vulnerable to evaluation bias.

### Strengths
Novel framework for automated bias discovery in LLM evaluators.

Strong experimental evidence across multiple models and datasets.

The new JudgeBench-Pro benchmark contributes valuable resources for future research.

### Weaknesses
While the contribution is significant, several limitations remain. The iterative bias discovery process is computationally demanding, restricting scalability for large evaluations. The framework depends heavily on the quality of the teacher model — if the teacher itself is biased, those biases may cascade into the discovery process. 

Additionally, the interpretability of the “discovered” biases is often shallow; many are validated statistically but not semantically explained, which reduces their practical utility for bias mitigation or fairness auditing. Lastly, evaluation mainly relies on error rate, which oversimplifies bias characterization and overlooks directionality or intensity.

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents BIASSCOPE, a fully automated framework for discovering unknown biases in the LLM-as-a-Judge paradigm. Prior work has mostly focused on known biases (e.g., length, position, self-bias), whereas BiasScope aims to systematically discover and validate new ones. The framework operates iteratively in two phases: 1. Bias Discovery: A teacher model injects controlled perturbations (based on a seed bias library) into pairwise evaluation data, prompting the target model to reveal its bias tendencies. The teacher then identifies potential new biases from the model’s misjudgments and explanations. Bias Validation: Candidate biases are tested on a held-out dataset. Those that reliably increase error rates are confirmed and added to the bias library. Experiments span multiple LLM families (Qwen, LLaMA, Mistral, InternLM). BiasScope uncovers dozens of novel biases (e.g., novelty bias, exact-match bias), increases JudgeBench error rates across models by 5–11%, and scales across data sizes.

### Strengths
* Important research question: The work tackles a major open challenge in the “LLM-as-a-Judge” field: detecting unknown and latent biases, which directly affect fairness and reliability in automatic evaluation.

* Strong experimental coverage. Results are presented across seven target models (Table 1), multiple domains (math, reasoning, coding, knowledge), and ablations on teacher models (Table 2), validation timing (Table 3), and explanation depth (Table 4).

### Weaknesses
* Limited novelty: While the paper presents a well-engineered framework, I found that the most significant perturbation component has already been presented in CALM. The methodological novelty lies mainly in adding a search-based framework based on CALM. Consequently, the conceptual advancement, though practical and valuable, may be viewed as incremental rather than groundbreaking.

* Dependence on teacher quality. Table 2 shows strong teacher influence, yet the framework assumes the teacher itself is unbiased and robust. If the teacher introduces their own systematic bias, discovered biases may reflect that rather than the target model.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2
