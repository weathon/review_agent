# Uncovering Competency Gaps in Large Language Models and Their Benchmarks

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
The evaluation of large language models (LLMs) relies heavily on standardized benchmarks. These benchmarks provide useful aggregated metrics for a given capability, but those aggregated metrics can obscure (i) particular sub-areas where the LLMs are weak (''model gaps'') and (ii) imbalanced coverage in the benchmarks themselves (''benchmark gaps''). We propose a new method that uses sparse autoencoders (SAEs) to automatically uncover both types of gaps. By extracting SAE concept activations and computing saliency-weighted performance scores across benchmark data, the method grounds evaluation in the model's internal representations and enables comparison across benchmarks. As examples demonstrating our approach, we applied the method to two popular open-source models and ten benchmarks. We found that these models consistently underperformed on concepts that stand in contrast to sycophantic behaviors (e.g., politely refusing a request or asserting boundaries) and concepts connected to safety discussions. These model gaps align with observations previously surfaced in the literature; our automated, unsupervised method was able to recover them without manual supervision. We also observed benchmark gaps: many of the evaluated benchmarks over-represented concepts related to obedience, authority, or instruction-following, while missing core concepts that should fall within their intended scope. In sum, our method offers a representation-grounded approach to evaluation, enabling concept-level decomposition of benchmark scores. Rather than replacing conventional aggregated metrics, CG complements them by providing a concept-level decomposition that can reveal why a model scored as it did and how benchmarks could evolve to better reflect their intended scope. Code is available at [anonymized].

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel and unified methodology for (a) evaluating the capabilities of LLMs and (b) appraising the effectiveness of static Question-Answer benchmarks for testing for those capabilities. The authors use Sparse Auto-Encoders (SAEs), a technique from mechanistic interpretability, to do this. 

Using (I assume) pretrained SAEs with a pre-defined dictionary of tens of thousands of concepts, the normalized activation for each concept for each benchmark item is computed by summing the activation for that node of the hidden layer for each token, divided by the number of tokens in the benchmark prompt. For each benchmark, its coverage of a concept is computed as the normalized activation divided by the average concept activation across all concepts and items in the benchmark. It is not totally clear to me, but I assume that this is done relative to one LLM, rather than both or all possible LLMs, a point I come back to later. The coverage across all benchmarks is the average coverage across the set of benchmarks. A similar strategy is taken for models, except the per-benchmark model performance is modulated by some 'performance scoring policy', which I take to be the model's actual (or average?) score on a particular item of a benchmark.

Using these metrics, the authors evaluate two models (Llama-3.2-8B and Gemma2-2B) on 10 benchmarks. They find that these benchmarks show skewed coverage across the concept space, missing important concepts like meta-cognition and legal concepts. Both models also perform poorly on important safety-related concepts, such as politely rejecting inappropriate requests.

### Strengths
This paper is commendable, and constitutes a novel and innovative contribution to the growing literature on AI Evaluation. It is reasonably clearly written and well presented, apart from a few parts highlighted below. The web app is a cool feature that is certainly a great contribution of the work, if the empirical results stand up. The results that common benchmarks fail to measure capabilities/concepts that they purport to is well-made and empirically backed up, in contrast to much of the theoretical and/or purely verbal arguments made on this topic (Burden, 2024; Hernández-Orallo, 2017, 2020; Jo and Wilson, 2025; Raji et al., 2021). The results that models fail on many safety-related concepts is a striking result worth emphasising, although this may be a feature of model size and/or degree of safety post-training.



Burden, J. (2024). Evaluating ai evaluation: Perils and prospects. arXiv preprint arXiv:2407.09221.
Hernández-Orallo, J. (2020). AI evaluation: On broken yardsticks and measurement scales. In Workshop on evaluating evaluation of AI systems at AAAI. Menlo Park: Association for the Advancement of Artificial Intelligence.
Hernández-Orallo, J. (2017). Evaluation in artificial intelligence: from task-oriented to ability-oriented measurement. Artificial Intelligence Review, 48(3), 397-447.
Jo, N., & Wilson, A. (2025). What Does Your Benchmark Really Measure? A Framework for Robust Inference of AI Capabilities. arXiv preprint arXiv:2509.19590.
Raji, I. D., Bender, E. M., Paullada, A., Denton, E., & Hanna, A. (2021). AI and the everything in the whole wide world benchmark. arXiv preprint arXiv:2111.15366.

### Weaknesses
The paper currently has three weaknesses, in decreasing order of severity:
1. There are some reliability and validity checks to be done. First, it would be useful to verify that the same (or broadly the same) benchmark gaps are recovered with (a) different language models doing the prediction (or even just different seeds/temperatures) and (b) perturbed benchmark data (i.e., syntactic/superficial variants of the same prompts). This would verify that the benchmark gaps are reliable and not simply statistical artefacts of the current set up. Second, checking the predictive validity of the model gaps is imperative. Figure 5 is an anecdotal example, but it is necessary to find a set of model gaps on a subset of the benchmarks and then test the models on items from the held out set of benchmarks that correspond to those concepts. This would show that the model gaps are predictive of performance on new, unseen data. The authors could report in a table the success rates on held-out items from model-gap and non-model-gap concepts, so that the difference is clear.
2. The result about models performing poorly when, for instance, rejecting certain requests from users, is intriguing. The authors should consider running a parallel analysis with Llama-Guard-3-8B to see if that model gap disappears with post-training. If so, this would be great evidence in favour of the conclusion that Llama-3.1-8B genuinely has a safety gap.
3. There is some missing literature. Burden (2024), Burden et al. (2023), and Hernández-Orallo (2017, 2020) all discuss the problems of aggregating scores on benchmarks which is discussed in the first paragraph of the introduction. Similarly, much recent work has explored the redundancy of common NLP benchmarks and propose sophisticated strategies to determine and overcome that redundancy: Raji et al. (2021), Kipnis et al. (2024), Wang et al. (2025), Polo et al. (2024), Zhou et al. (2025). Indeed, the related work section should discuss Item Response Theory (IRT) as one alternative method for doing data-driven 'model gap'/capability discovery in large language models, which contrasts with the current approach of using predefined concept dictionaries (which is much more in-line with Burden et al. 2023).
4. There are a number of textual problems with the paper at the moment. The most important are handled by the first two of my questions below, namely, how benchmark gaps are computed and how the performance scoring policy is defined. I also think a similar box to the 'Model Gap' box in Section 3.2 is necessary for Benchmark Gap in Section 3.1, for clarity and consistency. There is a syntax error on lines 174-5, it should be: "that quantifies the degree to which a concept c was activated...". There is a missing ref to the appendix on line 332. I would recommend a read-through to check all syntax and references.


Burden, J. (2024). Evaluating ai evaluation: Perils and prospects. arXiv preprint arXiv:2407.09221.
Burden, J., Voudouris, K., Burnell, R., Rutar, D., Cheke, L., & Hernández-Orallo, J. (2023). Inferring capabilities from task performance with bayesian triangulation. arXiv preprint arXiv:2309.11975.
Hernández-Orallo, J. (2020). AI evaluation: On broken yardsticks and measurement scales. In Workshop on evaluating evaluation of AI systems at AAAI. Menlo Park: Association for the Advancement of Artificial Intelligence.
Hernández-Orallo, J. (2017). Evaluation in artificial intelligence: from task-oriented to ability-oriented measurement. Artificial Intelligence Review, 48(3), 397-447.
Kipnis, A., Voudouris, K., Buschoff, L. M. S., & Schulz, E. (2024). metabench--A Sparse Benchmark of Reasoning and Knowledge in Large Language Models. arXiv preprint arXiv:2407.12844.
Polo, F. M., Weber, L., Choshen, L., Sun, Y., Xu, G., & Yurochkin, M. (2024). tinyBenchmarks: evaluating LLMs with fewer examples. arXiv preprint arXiv:2402.14992.
Raji, I. D., Bender, E. M., Paullada, A., Denton, E., & Hanna, A. (2021). AI and the everything in the whole wide world benchmark. arXiv preprint arXiv:2111.15366.
Wang, Y., Ying, J., Cao, Y., Ma, Y., & Jiang, Y. (2025). EffiEval: Efficient and Generalizable Model Evaluation via Capability Coverage Maximization. arXiv preprint arXiv:2508.09662.
Zhou, L., Pacchiardi, L., Martínez-Plumed, F., Collins, K. M., Moros-Daval, Y., Zhang, S., ... & Hernández-Orallo, J. (2025). General scales unlock ai evaluation with explanatory and predictive power. arXiv preprint arXiv:2503.06378.

### Questions
* It is not clear to me how benchmark gaps are computed. Is it with respect to one or both of the language models? Should we interpret these coverage scores more as expectations over the space of possible language models?
* How is the performance scoring policy for the model gaps computed? I was assuming it was related to the score of the model on that benchmark item, but that would be a binary pass or fail, right? Or is it the log-probability of the correct answer token, the proportion of correct answers over some set of trials with high temperature, or something else entirely? It should be described somewhere.
* How does the benchmark gap calculation work if neither model is very good at, say, meta-cognition or legal reasoning? Surely this could be a reason why those concepts are misrepresented - i.e., simply because the models with respect to whom the benchmark gap scores are computed are incapable of performing well on those tasks. I'm sure I missed something there.
* What is the base rate activation of each of the concepts in each SAE. Is it possible that some of the low activating concepts simply do not ever see the high rates of activation of the others? How do you control for this base rate effect?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Competency Gaps (CG) — a novel evaluation framework leveraging Sparse Autoencoders (SAEs) to uncover two forms of evaluation shortcomings: Benchmark gaps (concepts underrepresented or missing in existing benchmarks) and Model gaps (concepts where models systematically underperform). By analyzing SAE concept activations across 10 benchmarks and 2 open-source LLMs (Gemma2-2B-Instruct and Llama3.1-8B-Instruct), the authors demonstrate that: Benchmarks overrepresent instruction-following and authority-related concepts. Models perform well on sycophantic or helpful concepts but poorly on boundary-maintaining, time reasoning, or rejection scenarios. The tool allows transparent exploration of benchmark and model concept coverage through an interactive web interface.

Key contributions:

- Methodological innovation: an SAE-based quantitative method for identifying benchmark and model coverage gaps.
- Empirical study: evaluation on 10 diverse benchmarks spanning reasoning, factuality, ethics, and math.

### Strengths
- This paper introduces a use of Sparse Autoencoders (SAEs) to analyze benchmark coverage and model competence at the concept level, moving beyond aggregate accuracy metrics. The framework provides a fresh interpretability-driven lens on how benchmark distributions shape perceived model performance—an idea with clear originality and conceptual significance.
- The use of sparse autoencoders to quantify benchmark and model “concept coverage” is well-formulated and technically sound, providing a coherent analytical framework.

### Weaknesses
- Experiments are restricted to two medium-sized instruction-tuned models (Gemma2-2B and Llama3.1-8B). Without results from smaller or larger models, the claimed “systematic competency gaps” may reflect architecture-specific or fine-tuning artifacts rather than generalizable phenomena.
- The proposed method assumes that SAE activations correspond to stable, human-interpretable concepts. However, the paper does not validate this assumption—for example, through multiple SAE runs, layer sensitivity, or concept consistency analysis. This dependency weakens the robustness of the theoretical foundation.
- While visualizations are compelling, there is no statistical significance testing or correlation analysis linking concept gaps to performance outcomes. The evidence remains descriptive rather than inferential, limiting confidence in the paper’s stronger claims.

### Questions
1. Can you demonstrate that modifying the composition of benchmark tests (e.g., removing overrepresented concepts or adding missing ones) actually changes model rankings or the measured capabilities?
2. The current experiments only include two medium-sized, instruction-tuned models. Have you tested smaller or larger models, as well as base models, to verify that the observed biases are generalizable?

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
4

### Summary
- This paper shows that current LLM benchmarks miss many concepts that models actually have, and thus fail to reveal their true weaknesses.
- Using SAEs, the authors map model behavior to fine-grained concepts to uncover gaps in both benchmarks and models.
- While benchmarks overtest abilities like instruction-following, models still struggle with polite refusal, boundary setting, and time reasoning, essentially the opposite of sycophancy, aligning well with the benchmark-gap findings.

### Strengths
**1. Introducing internal latent features for evaluation framework**

> This paper raises an interesting point that model evaluation does not always need to follow human-defined cognitive categories. Instead, it explores assessing capabilities through the model’s own latent features. Introducing SAEs in this context is a reasonable and promising direction, as it provides a way to diagnose models based on the representations they learn.

**2. Providing a meta-level perspective on benchmark coverage**

> It offers a useful way to see, from a broader and more meta perspective, how much of a model's overall capability space current benchmarks actually cover.

### Weaknesses
**1. Gap between stated motivation and actual research objective**

> Benchmarks are ultimately meant to communicate and compare core abilities at a broader level for a broad audience, not to enumerate every internal concept or serve primarily as a debugging tool. As a result, while this approach is useful for detailed model analysis, it does not resolve the aggregation problem highlighted in the motivation and is perceived as a diagnostic tool, rather than a practical framework for advancing benchmark design. **It is unclear how descriptive concept labels can inform high-level capability understanding or guide model improvement, particularly given the trade-offs across such highly granular features.**

**2. Lack of clarity on selecting latent features aligned with evaluation objectives**
> **The paper does not explain how to determine which latent features are meaningful and should be treated as evaluable capabilities.** Given that LLMs contain numerous internal dimensions that may not correspond to cognitive abilities, a principled criterion for selecting relevant features seems necessary. If such a mechanism is already discussed in the paper, clarification would be helpful.

**3. Missing discussion on SAE training data construction**

> Coverage of sparse concepts likely depends on constructing SAE training data that is as diverse as possible, ideally broad enough to approximate the full range of concepts encoded in the model itself, similar to the diversity of an LLM training corpus. However, the paper does not describe how the SAE training data was curated to ensure such conceptual diversity, which seems to be a crucial factor for achieving robust sparse-concept coverage.

**4. Need for insights beyond overall trend in benchmark coverage**

> The method shows the high-level trend of how much of a model's abilities current benchmarks cover (e.g., Figure 3, 4). This is useful, but for the tool to be more impactful, it would be helpful if it also suggested which kinds of abilities are missing and should be evaluated next to make benchmarks more balanced and fair. In other words, beyond showing the coverage trend, it would be valuable to connect these results to concrete capability areas that matter for evaluation.

**5. Model-dependent concept sets**

> Although each model can acquire different latent features, this approach does not provide a shared evaluation standard and therefore evaluates models relative to their own internal concept representations.

### Questions
- Conversely, are there capabilities that benchmarks such as BigGenBench (77 tasks) can evaluate but that do not emerge in the latent-based concept mapping? If so, could you share concrete examples?

- Could you elaborate on how such a large space of fine-grained conceptual units can be systematically organized and validated? Given that the absorption/common-pattern groupings are themselves highly granular, what principled structure do you envision for representing assessing these concepts in a coherent manner?

### Soundness
2

### Presentation
2

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
The paper proposes to assess benchmarks and models in the conceptual space extracted by sparse autoencoders on its own representations. Having SAEs, the move is natural. This procedure surfaces familiar difficulties (palindromes, temporal reasoning) without supervision. Not a discovery, but a confirmation that the method is sound and that makes the results convincing. 
The extreme skewedness of the concept distribution is an interesting insight.
As well as that the worst-performing concepts are often the opposites of these sycophantic behaviors, that was expected, but given the unsupervised nature of the method, it is remarkable.

### Strengths
It is a tool that could turn useful to people in the interpretability field.
As stated before, it confirms expected behaviors without any supervision, supporting the soundness of the method...

...but at the same time in this paper does not emerge anything particularly new.

The latter is not a Strength, but I think also not a Weakness. Honestly I find this structured format of reviews annoying, it is more difficult to write an organic judgement.

### Weaknesses
The main problem I have with this work is its writing. It is heavy. The overabundance of references sometimes interrupts the conceptual through-line. Papers should be written to be read, not merely to accompany code, and here the balance leans too far towards the latter.

Also the work remains confined to its enclave. It does not try to speak beyond those already initiated into SAEs, and the writing reflects this inward posture, should be more self-contained, and just more pleasant to read. A paper should be an invitation to thought. Here, I don't feel invited.

For reference, this is a well written paper in the interpretability area: https://transformer-circuits.pub/2025/attention-qk/index.html

### Questions
Now I guess I should write my conclusions in the Questions section, since there is no other section afterwards.

The contribution is real, if modest. A small community will find this tool useful, its functioning is sound and well justified. It is not well presented but indeed its ambition was not to reach a greater public.

The unfortunate fact is that I am part of the greater public, and I also have the opinion that any paper should be an invitation to thought for a large enough community.

Nevertheless, my recommendation is a poster. I would not feel good in seeing this paper rejected. I would put 7. There is no 7 so I put 6 and will change to 8 later. I will do it anyways, but would appreciate if the authors make the paper more accessible and pleasant to read, for example with a brief explanation of SAEs.

### Soundness
3

### Presentation
1

### Contribution
2
