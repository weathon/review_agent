# MSC-180: A Benchmark for Automated Formal Theorem Proving from Mathematical Subject Classification

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Automated Theorem Proving (ATP) represents a core research direction in artificial intelligence for achieving formal reasoning and verification, playing a significant role in advancing machine intelligence. However, current large language model (LLM)-based theorem provers suffer from limitations such as restricted domain coverage and weak generalization in mathematical reasoning. To address these issues, we propose MSC-180, a benchmark for evaluation based on the MSC2020 mathematical subject classification. It comprises 180 formal verification problems—3 advanced problems from each of 60 mathematical branches—spanning from undergraduate to graduate levels. Each problem has undergone multiple rounds of verification and refinement by domain experts to ensure formal accuracy.Evaluations of state-of-the-art LLM-based theorem provers under the pass@32 setting reveal that the best model achieves only an 18.89\% overall pass rate, with prominent issues including significant domain bias (maximum domain coverage 41.7\%) and a difficulty gap (significantly lower pass rates on graduate-level problems). To further quantify performance variability across mathematical domains, we introduce the coefficient of variation (CV) as an evaluation metric. The observed CV values are 4–6 times higher than the statistical high-variability threshold, indicating that the models still rely on pattern matching from training corpora rather than possessing transferable reasoning mechanisms and systematic generalization capabilities.MSC-180, together with its multi-dimensional evaluation framework, provides a discriminative and systematic benchmark for driving the development of next-generation AI systems with genuine mathematical reasoning abilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a benchmark named MSC-180 that contains 180 Lean4 formalized problems spanning 60 mathematical branches to evaluate LLMs’ performance in formal theorem proving across different subjects and different difficulty levels. The dataset is constructed via an extract, autoformalize, LLM verification, and human verification process. The paper performs comprehensive experiments. It shows that current models have relatively low overall accuracy, with the best pass@32 rate being 18.89%, and they exhibit significant domain bias, with a maximum domain coverage of 41.7%. Additionally, the paper introduces the Coefficient of Veriation@k (CV@k) metric to quantify cross-domain performance consistency. It observes high CV values, which suggests the current model may rely on pattern matching rather than systematic generalization.

### Strengths
The strengths of the paper are as follows:

1. **Interesting new metric:** This paper introduces the CV@k metric, which provides a normalized measure of performance consistency across various math domains, independent of absolute accuracy levels. This metric offers an additional point of view to traditional accuracy-based evaluation by quantifying the stability and balance of model performance across diverse areas. 
2. **Comprehensive experiment analysis:** The experiment section is relatively comprehensive and well-structured. It encompasses multiple analytical dimensions, including (1) cross-domain performance comparison across 60 mathematical branches, (2) difficulty-level analysis distinguishing undergraduate and graduate problems, and (3) in-depth investigation of the BFS-prover’s behavior. This analysis provides valuable insights into the strengths and limitations of different models and enhances the paper’s clarity.
3. **High-difficulty problems:** The benchmark focuses on problems of significant difficulty, with 80.6% sampled from graduate-level textbooks. The low performance of advanced expert provers (with best pass@32 being 18.89%) demonstrates that MSC-180 provides a relatively large headroom for measuring future improvements in expert prover systems.

### Weaknesses
Despite the strengths of the paper, there are also many significant weakness that would clearly harms the contribution of the paper.

1. **Limited contribution:** The most significant weakness of the paper is its lack of a clear innovation point compared to existing works. Firstly, there are many cross-domain benchmarks in the field, like ProofNet, PutnamBench, and FormalMATH, and the paper does not distinguish MSC-180 between them. Secondly, the necessity of using MSC2020 as the classification system is not elaborated correctly, as other works also provide a comprehensive classification of categories. The MSC2020 system is merely a different evaluation method. Finally, the significant domain-different capability of the current prover has been found in the Goedel-Prover paper.
2. **Data construction pipeline problems:** The core methodology for constructing the benchmark data is well-established since DeepSeek-Prover. This makes the paper's methodology doubtful in terms of its level of contribution and novelty. Besides, the implementation details of the construction are not properly disclosed. For example, which version of the DeepSeek model is applied, what is the prompt for extracting theorems, and what is the exact standard for human evaluation? There is no qualitative or quantitative disclosure of the process, which affects the truthworthness of the constructed pipeline.
3. **Small size of the benchmark:** The scale of the benchmark presented in the paper is too small. In fact, having only three records of data per domain makes it difficult to draw domain-related conclusions with statistical significance. This weakness directly affects the core argument of the paper, which is that the MSC180 dataset can help researchers to study domain-specific behavior of provers.

### Questions
1. What is the major difference between the paper's proposed methods and benchmarks compared to existing works?
2. What are the details of the construction of benchmark data, and how does your methodology differ from traditional pipeline?
3. What is the statistical significance to support the major dataset-level and field-specific conclusions obtained in the experiment?

### Soundness
3

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
Regarding the issue that existing Automated Theorem Proving are limited by finite domain coverage and weak generalization ability, this paper proposes a formal math problem evaluation set, MSC-180, based on the MSC2020 classification standard. It includes 180 problems distributed across 60 categories, with 3 problems in each category. The data sources are textbooks ranging from undergraduate to graduate level difficulty. Specifically, the construction approach involves autoformalization, followed by verification through Lean compilation and semantic alignment judged by LLM-as-a-judge, and finally mannual reconstruction by experts. Testing on various recent provers shows relatively low proof rates and limited domain coverage, demonstrating the challenge and broad domain coverage of this dataset.

### Strengths
1. The MSC-180 dataset proposed in the paper claims to cover 60 major categories of mathematics, which encompasses a broader range of topics compared to commonly used ATP datasets such as miniF2F. This broader coverage could have potential value for evaluating ATP performance across a wider variety of mathematical topics.

2. The paper introduces the use of the coefficient of variation as a metric to assess ATP performance across different mathematical topics. This provides a meaningful reference for evaluating the generalization ability of ATP systems over diverse mathematical subjects.

3. From the experimntal results, it seems that current provers do not perform well on this dataset. Given that performance on commonly used datasets like miniF2F is approaching saturation, this dataset may serve as a valuable challenge benchmark.

### Weaknesses
1. The dataset is far too small, which significantly undermines its practical value. With only three problems per mathematical category, evaluations of provers’ proving ability and generalization become highly susceptible to randomness; both the variance and mean of scores will be unstable and may lack statistical significance. Having only three problems also fails to adequately cover the breadth and depth of domain knowledge for each category, so inferring a prover’s overall capability in a field from performance on just three problems is insufficient. For reference, the [MSC taxonomy]([msc2020.org](https://msc2020.org/)) contains 63 two-digit classifications, 529 three-digit classifications, and 6,022 five-digit classifications; an ideal design would include at least 10–20 problems per category.

2. The proposed automated formalization pipeline lacks novelty and appears unnecessary. The pipeline—syntax checking via Lean 4 followed by semantic-alignment checking using LLM-as-a-judge—has been explored in multiple prior works [1,2,3]. As the authors note (Line 245), “We observed that over 90% of the auto-generated code exhibited semantic incompleteness, all propositions were manually reconstructed and rewritten.” Given this, why not adopt a manual annotation pipeline from the outset? The authors should justify the necessity of the automated formalization step. Additionally, the auto formalization pipeline inherently filters for problems that are easier to autoformalize, introducing a selection bias into the dataset.

3. The analysis section lacks insight. Much of the analysis merely restates table contents (e.g., Lines 398–415 essentially repeat Table 2). Although MSC-180 divides the dataset into different domains, the paper does not compare or analyze which models excel in which specific domains. The analysis focuses on aggregate counts of theorems proved across all categories rather than providing a fine-grained, interpretable discussion of per-topic strengths and weaknesses, which is a missed opportunity.

4. Concerns about dataset quality and contamination. The dataset files linked by the authors appear to contain multiple languages, suggesting heterogeneous sources; language mixing also limits accessibility for some researchers. The paper labels the problems as “high-difficulty” (Lines 019, 095), yet the stated sources are “classic graduate and undergraduate textbooks” (Lines 079, 201); describing them as “advanced” might be more accurate than “high-difficulty.” Using classic textbooks may also increase dataset contamination, since textbooks are common training data for LLMs. For example, the provided first sample, Gödel’s β-Function Lemma, is a well-known result with abundant online material, which risks leakage and inflating apparent model performance.

5. Numerous writing and presentation issues impede readability. Examples I noted include:

- Lines 022, 030, 199, 341: missing space after period at sentence end.
- Lines 046, 431: missing period.
- Lines 038–046: repeated full-form + abbreviation uses for “LLM” and “ATP” within the same paragraph are awkward.
- Line 053: when reporting the 90.4% accuracy, the dataset should be explicitly named (miniF2F); the current phrase “on a specific test set” is ambiguous.
- Line 054: “sutdies” lacks citation.
- Line 056: citations for Goedel-Prover v1 and v2 are confused — the negative correlation claim appears to belong to Goedel Prover v1 but cites v2.
- Figures 1 and 2: inconsistent capitalization and formatting.
- Figures 2 and 3: odd ordering — Figure 3 is first mentioned at Line 188, but Figure 2 is not mentioned until Line 205.
- Line 211: the model “deepseek” is referenced without specifying the exact model/version
- Line 214: “kimina tool” likely refers to the [Kimina-Autoformalizer-7B](https://huggingface.co/AI-MO/Kimina-Autoformalizer-7B) but is unclear.
- Lines 180, 214, 215, 252: inconsistent notation of “Lean4” vs “Lean 4” (spacing should be unified).
- Line 352: contains a typo.
- Line 484: claims “An anonymized version of the code is also available” but the provided link does not offer the code.

References
 [1] Ying, H., Wu, Z., Geng, Y., Wang, Ji., Lin, D., & Chen, K. (2024, November 13). *Lean Workbook: A large-scale Lean problem set formalized from natural language math problems*. The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track. https://openreview.net/forum?id=Vcw3vzjHDb#discussion
 [2] Peng, Z., Yao, Y., Ma, K., Guo, S., Li, Y., Zhang, Y., Zhang, C., Zhang, Y., Yu, Z., Li, L., Liu, M., Xia, Y., Shen, J., Wu, Y., Cao, Y., Zhang, Z., Huang, W., Liu, J., & Zhang, G. (2025). *CriticLean: Critic-Guided Reinforcement Learning for Mathematical Formalization*
 [3] Yu, Z., Peng, R., Ding, K., Li, Y., Peng, Z., Liu, M., Zhang, Y., Yuan, Z., Xin, H., Huang, W., Wen, Y., Zhang, G., & Liu, W. (n.d.). *FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models*.

### Questions
1. What is the precise definition of "high-difficulty" used for selecting problems? Figure 1(b) shows that 6,071 problems passed semantic verification, but only 180 problems underwent expert review. This appears to be an aggressive filtering step. Please provide more detailed selection criteria and the rationale for reducing to 180 items.

2. In the abstract (Line 027): "The observed CV values are 4–6 times higher than the statistical high-variability threshold," but the manuscript does not further explain this in the main text. How is the "statistical high-variability threshold" defined? Which specific "observed CV values" are being referred to? Please clarify the metric, its computation, and the threshold used.

3. The paper cites Goedel Prover v2 (Line 052) and notes that its weights are publicly available; Goedel Prover v2 is currently state-of-the-art on the conventional ATP benchmark miniF2F. Yet Table 1 and Table 2 do not include Goedel Prover v2 results, which seems like a notable omission. Could the authors report Goedel Prover v2’s performance on MSC-180? If experiments with Goedel Prover v2 were not performed, please justify why it was excluded.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces MSC-180, a benchmark designed to systematically evaluate the reasoning and generalization abilities of large language model (LLM)-based theorem provers. The dataset includes 180 formally verified problems (three per branch) spanning 60 mathematical domains, categorized by the MSC2020 taxonomy and covering undergraduate to graduate difficulty. All problems have undergone human expert verification. The authors evaluate multiple SOTA provers under a pass@32 setup and report an overall best score of 18.89%, highlighting both the narrow domain coverage (max 41.7%) and limited transfer of reasoning across topics. They further introduce a coefficient of variation (CV) metric to quantify domain-wise performance variability, finding it several times above statistical baselines, indicating strong pattern-matching bias. The benchmark, analysis, and code are all released for reproducibility and future study.

### Strengths
1. The dataset is curated and verified carefully by domain experts, ensuring high quality and formal soundness across 60 domains.

2. There is clear problem taxonomy, reproducible setup, and well-defined evaluation metrics (pass@k, Domain@k, CV) make the study credible.

3. Balanced interpretation: The discussion doesn't overclaim; it clearly states both the models’ current limitations and partial strengths (e.g., structured reasoning in applied math).

4. The paper is logically structured, precise, and accessible; figures and tables are clear. And they released their code to make themself even more convincing.

### Weaknesses
1. The work stops at benchmarking, without concrete methods proposed to improve reasoning, this is a significant limitation.

2. Limited diversity of evaluated models. While several provers are tested, it would be useful to include symbolic or neuro-symbolic baselines to contextualize LLM gaps.

3. It remains unclear how representative these 180 problems are of real downstream theorem-proving workloads in Lean, Coq, or Isabelle.

4. Some details can be further justified, such as: is there any statistical threshold for so-called "high variability", have you compared with datasets like MiniF2F, ProofNet, or LeanDojo would situate MSC-180 better in the existing ecosystem, etc.

### Questions
Addressing the weakness points would be enough to make me think twice on my decision!

### Soundness
3

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
The paper introduces MSC-180, a domain-balanced benchmark of 180 expert-verified Lean4 problems drawn from 60 MSC2020 mathematical areas to assess cross-domain generalization in automated theorem provers. Evaluations of leading LLM-based provers show low Pass@32 performance (best 18.89%), strong domain bias, and large variability across fields, indicating limited abstract reasoning and reliance on pattern matching. The authors also propose the Coefficient of Variation to measure stability across domains, arguing that MSC-180 provides a rigorous foundation for developing more genuinely generalizable formal reasoning systems.

### Strengths
- Valuable and well-curated dataset.
- The introduction of the coefficient of variation as a cross-domain stability metric is a thoughtful and useful addition.

### Weaknesses
- The writing needs quite some polishing (e.g., missing a stop in line 319, 323, 336, 426, 431, 435, 440, 444, 449).
- Lack of in-depth comparison with previous dataset like ProofNet, FormalMATH, and FATE [1].
- The data processing process is not well illustrated. For example, how is the semantic verification being carried using the DeepSeek API? What are the exact criteria for human verification and selection? 

[1] Shen, Ziju, et al. "REAL-Prover: Retrieval Augmented Lean Prover for Mathematical Reasoning." arXiv preprint arXiv:2505.20613 (2025).

### Questions
- What are the 60 domains included, and how are the problems distributed across them?
- In line 249, what exactly is meant by the “completeness of the formalized problem”?
- How is difficulty determined and assigned in MSC-180, and what is the exact distribution across difficulty levels?

### Soundness
3

### Presentation
1

### Contribution
2
