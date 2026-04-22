# Beyond Benchmarks: Toward Causally Faithful Evaluation of Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Current large language models (LLMs) evaluations overlook that measured LLM performance is produced on a full evaluation system, including many indispensable components, such as workloads, prompting methods, decoding parameters, and the supporting software–hardware stack.  Without an explicit, controlled specification of the evaluation system, attributing performance differences to the model itself is unreliable.  Our experiments reveal that uncontrolled testing may lead to accuracy variations of up to 70\%. To address this urgent issue, we introduce LLM evaluatology, a principled methodology that reduces the evaluation problem to accurately attributing  the outcomes to the effect of the evaluated LLM, which is a high-dimensional causal-attribution problem. Empirical results demonstrate that LLM evaluatology not only enhances interpretability and causal validity, but also yields evaluations that are more robust, reproducible, and trustworthy than prevailing benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a framework called LLM Evaluatology that aims to disentangle a model’s intrinsic capability from the confounding effects of evaluation configurations such as decoding parameters and prompt settings. To this end, the authors introduce the concept of a Minimal Evaluation System (MES) for in-distribution testing and an augmented variant (A-MES) designed to assess robustness under distributional shifts. They further perform an ANOVA analysis to quantify how individual components (e.g., temperature, top-p, max tokens) contribute to performance variance across large language models.

### Strengths
This paper starts from a very interesting motivation: benchmark scores are not inherent properties of a model but the outcome of a specified evaluation system. The authors aim to disentangle a model’s intrinsic ability from other influencing factors, such as hyperparameter settings and prompting strategies. This is an important and realistic observation. In related work, we indeed see that different hyperparameters can dramatically change model performance, and that various prompting methods (e.g., Chain-of-Thought) can affect different models in a different way. A systematic evaluation framework would therefore allow for fairer comparisons between models. By introducing the MES/A-MES framework, the authors make a useful attempt to formalize an evaluation system with systemetically varying configurations. This effort contributes to the ongoing discussions about reproducibility and standardization in LLM evaluation.

### Weaknesses
1. I think the experimental design has major issues in how component value ranges are selected. For instance, the paper uses a temperature range of {0.0,1.0,2.0}, which is overly discrete and unrealistic. Most models are tuned for values between 0 and 1. Similarly, the smallest max_tokens value is set to 10, which would truncate nearly all responses. These settings would almost never be used in real deployment. Such unrealistic experimental configurations seem to have caused major issues in the results. If the parameter discretization and value ranges were reasonable, I would expect to observe a roughly unimodal distribution of model accuracies in Figure 4(a). However, all models show a violin shape with large mass near 0. I think these near-zero accuracies likely arise from extreme settings (e.g., temperature = 2.0 causing random outputs, or max_tokens = 10 cutting off answers). Such mixtures of valid and invalid runs inflate variance and render the reported evaluation unreliable. 

2. The motivation and construction of A-MES are confusing. The authors state their motivation as “performance on a single workload can be misleading, as models may succeed on problems they have effectively memorized yet fail when the same reasoning must be applied under slightly altered conditions” (with an example shown in Line 171). However, I don’t think this example is appropriate. Adding an irrelevant distractor sentence is more like testing to robustness to adversarial perturbation rather than an analogical generation test for new use cases. This design choice also makes the subsequent definition of A-MES less convincing.
Also, the definition of out-of-distribution workloads in Line 250 does not align with the common understanding of OOD evaluation as well. It seems more related to removing petential data contamination during training rather than testing true distributional shift. Moreover, I believe it would make the A-MES results more credible to disclose more detailed documentation of the construction of the perturbed datasets.

3. This paper still has much room for improvement in presentation. I’ve listed several issues in the following Questions section. In addition, the term *LLM workload* appears to be borrowed from CPU benchmarking. Since the authors intentionally distinguish between task and workload, this concept should have been introduced earlier in the paper. As LLM workload is not a conventional term in the LLM community, a clearer definition would help readers better understand the authors’ intention.

### Questions
1. In figure 4(a), I couldn’t locate the reported accuracy line. I guess it should not appear in the label?

2. In Table 3 and Figure 5(a), it appears that models without reported AIME'24 scores are displayed as 0, which could be misleading to readers. Could the authors clarify how missing data is represented?

3. Regarding the "ground truth" in Figure 5(a), could the authors elaborate on why this serves as a meaningful baseline? By restrict the configuration space to the 5 most influential components, you retained the factors that dominate the variance of accuracy and fixed factors that contribute negligible variance, then the confidence interval computed on the unrestricted space will naturally cover this restricted mean. But I don’t think this indicates any robustness of the evaluation system, it merely reflects the reduced variance caused by restricting the configuration space. Also, considering the concerns mentioned earlier, I doubt if this can be called a ground truth value that reflects the intrinsic ability of an LLM.

4. I also find several parts of the paper unclear. For example, Table 1 and Table 2 are referred to as “evaluation settings” and “evaluation conditions,” and the row names in Table 2 appear to correspond to the column names in Table 1, while their listed values are very different. I carefully read corresponding sections but still could not determine under which configuration setting the experiments were actually conducted. I currently assume it corresponds to Table 2 as it was referred to in the evaluation section, but please correct me if this interpretation is mistaken.

5. If I understand correctly, your method essentially computes the mean accuracy over a predefined configuration space. However, in practical use, users typically operate under a fixed configuration. Moreover, we often observe that different models and tasks require different optimal hyperparameters, at least evident for temperature. I wonder whether a benchmark should evaluate models by averaging across all possible ones instead of under a representative configuration. Let’s assume one model achieves excellent performance within a narrow temperature range on a specific task, while another model performs slightly worse but maintains decent accuracy across a wider range. Which model should be considered better in that case? How can you justify that averaging performance over configurations is a fairer comparison than evaluating models under a representative or optimized fixed configuration?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper shows that LLM performance depends strongly on evaluation settings like prompt format and decoding, meaning current benchmarks often measure the evaluation setup rather than the model. It proposes a structured evaluation framework (MES / A-MES) to control and vary these components, revealing that model rankings can change significantly under different conditions. The goal is to make LLM evaluations more reliable and attributionally faithful.

### Strengths
- clear and compelling problem framing: the paper condignly argues that most current LLM evaluations conflate model ability with configuration choices
- conceptual contribution is novel and original: MES/A-MES provide a structured way to explicitly specify and vary evaluation conditions
- empirical evidence of variance strengthens the need for an evaluation framework
- ANOVA to quantify the contribution of prompt format, COT, decoding parameters is useful

### Weaknesses
- method is conceptual rather than algorithmic
- unclear if this generalizes beyond academic reasoning benchmarks
- complexity and practical burden on practitioners might be too high; unclear scalability
- limited human evaluations --> maybe human ablation could be added
- causal language might be overstated

### Questions
- Which parts of A-MES generation are automated and which require expert authoring? How would scalability be possible?
- DO you expect ANOVA factor importance to generalize across models and tasks, or is it benchmark-specific?
- How should practitioners choose a single evaluations core after MES/A-MES exploration? What is the summary statistic?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper points out that using existing benchmark to provide a single evaluation of LLM may be reliable, as the authors find out that the same LLM’s performance can vary significantly in a benchmark given different configuration. Therefore, the authors uses Minimal Evaluation System for LLM evaluation and specify 10 dimensions, covering the question posted to LLM, method of prompting as well as parameters like temperature. The authors further augment existing benchmarks by generating transformed questions or entirely new questions to test the robustness of LLM in novel settings. The authors also evaluate the importance of the 10 dimensions for each LLM.

### Strengths
Originality: this paper argues that the configuration of a evaluation settings is important and define 10 important dimensions for the evaluation settings, which is a unique contribution to make the evaluation more systematic.

Quality: this paper applies extensive experiments on recent LLMs, which shows rich insights of robustness of models.

Clarity: this paper presents the idea in a relatively clear way, especially the literature review part, which systematically categorizes and covers the major approaches of LLM evaluation. 

Significance: this paper tries to solve a very fundamental problem, which is building a systematic framework to evaluate the LLM, rather than testing LLM in a single setting, which could lead to evaluation bias.

### Weaknesses
1.	This paper doesn’t reveal enough details of implementation of the workload augmentation part. For instance, in “distractor insertion” in Transformed part, the authors didn’t mention how to insert distractor that is relevant to the question context. Note that we can always insert random sentences into questions, but if those sentences are totally out of the context of question, those sentences may mimic how LLM users’ additional comments distracts LLM in real life usage. More importantly, for Out-of-distribution workloads, the authors didn’t specify how to generate those new workloads. The authors didn’t list the algorithm but just mention that they rely on “textbook/academic statements”. However, generating new questions are inherently challenging and non-trivial, which is a critical gap that needs to fill if the authors would like to claim this contribution of A-MES.

2.	This paper’s attribution method themselves are value because it is essentially randomly controlled experiment, but I’m not sure whether the authors highlight them as “high dimensional causal attribution”. The word “causal” in title and abstract makes readers think that authors make some unique contributions in applying some novel causal method (say, treatment effect estimation) in observed data, which is really challenging. But in fact, the contribution to the causal method itself is not really there, as authors just apply ANOVA method on randomly controlled experiment data, which is correct but not inherently challenging in causal.

### Questions
1.	In line 242 to 249 about Analogical transformation (Transformed) workloads, do you have a systematic and scalable way to generate “distractors”? I would assume those distractors  should be still relevant to the questions’ context to actually serve as a distractor, is it true (just as your example in line 178 to 181 , you add that “symmetric” sentence, which still seems to be relevant to the topic of geometry)? If so, what are the ways to generate distractors that are relevant to this topic? Did you use a separate LLM to generate high quality distractors?  Also, how do you “swapping problem statements and conditions”? In some math problems, if you put the question as your condition, and your condition as question, I don’t think it is a valid math problem anymore. Or by “swapping”, do you mean just swap their positions in sentences?

2.	In line 250 to 255, first, the method (a) of recent-source adaptation (harvesting new questions such as new college entrance examination questions) itself is sound, but how does this method differ from those Dynamic or continuously refreshed benchmarks such as LiveCodeBench?  In method (b), how do you make sure the questions you generated from textbook/academic statements are valid questions themselves? Also, how do you generate the correct answers of those questions. If you generate those new (question, answer) pairs from LLM, then it doesn’t seem to make sense, because those questions are inherently new in nature and there is no guarantee the generating-LLM can generate the 100% correct answers.  

3.	In Table 2, why would the value of Question Paraphrase be binary (Yes and No)? I thought to paraphrase a question, there should be multiple ways to do it (rather than just yes or no).

4.	In line 324, is 500 random sampling enough? I think in Section 4.1, the total combination of all configuration is more than 7,000. Therefore, is 500 random sample enough to cover all configuration?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper argues that benchmark scores conflate model ability with many “indispensable” evaluation components (workloads, prompts, decoding, software/hardware). It proposes LLM evaluatology: define a Minimal Evaluation System (MES) and an Augmented MES (A-MES) that expands workloads (seen/analogical/novel) and then attribute performance via controlled sampling and ANOVA.

### Strengths
- The evaluation of LLM is of great importance in the research communities of LLM. Makes a compelling case that current single-config benchmarks are causally unfaithful and often misleading.
- Shows big accuracy variation across configs and non-transitive rank changes
- Quantifies main effects/interactions (Question Format, CoT, max tokens among top factors), lending interpretability to what actually matters.

### Weaknesses
- The proposed solutions seems trivial and handicrafted.
- The stopping rule (mean/CI thresholds) is reasonable, but guidance on sample sizes per workload/model and variance across random seeds would help practitioners replicate cost-accuracy trade-offs.
- Table 3 appears to list GPT-3.5 accuracy as 1.10, which must be a typo (accuracy >1). A thorough proofread for such inconsistencies would improve credibility.
- The main narrative leans heavily on AIME (plus MMLU/GPQA in appendix). Including code, tool-use, long-context, and multi-turn interactive settings in-depth would strengthen generality.

### Questions
See cons.

### Soundness
3

### Presentation
3

### Contribution
2
