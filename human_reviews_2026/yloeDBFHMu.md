# Deep Literature Survey Automation with an Iterative Workflow

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Automatic literature survey generation has attracted increasing attention, yet most existing systems follow a one-shot paradigm, where a large set of papers is retrieved at once and a static outline is generated before drafting. This design often leads to noisy retrieval, fragmented structures, and context overload, ultimately limiting survey quality. Inspired by the iterative reading process of human researchers, we propose IterSurvey, a framework based on recurrent outline generation, in which a planning agent incrementally retrieves, reads, and updates the outline to ensure both exploration and coherence. To provide faithful paper-level grounding, we design paper cards that distill each paper into its contributions, methods, and findings, and introduce a review-and-refine loop with visualization enhancement to improve textual flow and integrate multimodal elements such as figures and tables. Experiments on both established and emerging topics show that IterSurvey substantially outperforms state-of-the-art baselines in content coverage, structural coherence, and citation quality, while producing more accessible and better-organized surveys. To provide a more reliable assessment of such improvements, we further introduce Survey-Arena, a pairwise benchmark that complements absolute scoring and more clearly positions machine-generated surveys relative to human-written ones.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper develops a new framework: IterSurvey, which produces finer-grained outlines and supports multi-modal inputs and outputs for more comprehensive surveys, and constructs Survey-Arena, a pairwise evaluation benchmark that ranks generated surveys alongside human-written ones, enabling more reliable and interpretable assessment of survey quality.

### Strengths
1. An iterative mechanism is designed for survey generation and optimization.

2. Integration that includes multi-modal elements.

3. Introduce the Survey-Arena pairing evaluation benchmark.

### Weaknesses
1. The methodology is not innovative enough, and the iterative mechanism or  review-and-refine are not novel contribution. What is the difference between it and autosurvey [1].

2. Does the introduction of iteration mechanism and paper card have a big impact on the time consumption of the whole framework?

3. The cost of different methods is not compared.

[1] Yidong Wang, Qi Guo, Wenjin Yao, Hongbo Zhang, Xin Zhang, Zhen Wu, Meishan Zhang, Xinyu Dai, Qingsong Wen, Wei Ye, et al. Autosurvey: Large language models can automatically write surveys. Advances in neural information processing systems, 37:115119–115145, 2024.

### Questions
1. What is the difference of he iterative mechanism and review-and-refine between IterSurvey and Autosurvey [1].
2. What are the impact of iteration mechanism and paper card on the time consumption of the whole framework?
3. The cost of different methods should be compared.
4. Did you replicate the experiment independently, generating only one review per topic for comparison?

[1] Yidong Wang, Qi Guo, Wenjin Yao, Hongbo Zhang, Xin Zhang, Zhen Wu, Meishan Zhang, Xinyu Dai, Qingsong Wen, Wei Ye, et al. Autosurvey: Large language models can automatically write surveys. Advances in neural information processing systems, 37:115119–115145, 2024.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an auto-literature review generation framework *i.e.*, IterSurvey. The core idea of IterSurvey is to mimic the iterative research process of human researchers, through a cyclic outline generation module, incrementally conducting literature retrieval, reading, and updating to ensure the exploratory and coherent nature of the review. Additionally, to more reliably evaluate the quality of the generated reviews, the paper constructs a paired evaluation benchmark named Survey-Arena, which provides a more robust performance ranking by directly comparing machine-generated reviews with those written by humans.

### Strengths
The paper abandons the static "one-time" planning mode and instead adopts a dynamic, iterative workflow which may iteratively improve the quality of the survey. Besides, injecting multimodal items (table and figure) into a survey is interesting.

### Weaknesses
1. Lack of efficiency and cost analysis: The iterative framework proposed in the paper, especially the cyclic generation of outlines and multi-round "review and optimization," seems to require a large amount of computational resources and time. However, the paper does not provide a comparative analysis of the efficiency (such as the time required to generate a review) and cost (API usage) between IterSurvey and other baseline methods. 

2. The generation details of the "paper card" need to be supplemented: The process of constructing paper cards seems to need lots of time (using LLM to generate motivation, contribution, and so on). The authors need to add more details.

3. The authors need to verify the effectiveness of the iterative pipeline. For example, the performance improvement is seen with the increase in iterations.

### Questions
Please refer to the weakness.

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
This paper proposes IterSurvey to fix flaws of one-shot automated survey systems (noisy retrieval, fragmented structure). It uses recurrent outline generation, paper cards (distilling papers’ key info), and a review loop. Experiments show it outperforms baselines (e.g., AutoSurvey) in coverage, coherence and citations. The authors also introduce Survey-Arena for reliable pairwise assessment against human surveys, noting IterSurvey is an assistive tool with data from public arXiv papers.

### Strengths
- The integration of paper cards and a review-and-refine loop with visualization enhancements significantly improves textual flow, cross-sectional coherence, and multimodal integration, as evidenced by the experimental results.

- The proposed Survey-Arena, a pairwise evaluation benchmark that compares machine-generated surveys with human-written ones, offers a more reliable assessment and effectively addresses the limitations of absolute scoring methods.

### Weaknesses
My main concerns are the limitations of the paper's coverage and the objectivity of the evaluation. If the author can address my questions, I would be happy to increase the rating:
- The retrieval database includes only 680K computer science papers from arXiv, resulting in limited coverage of other disciplines and constraining the framework’s generalization ability to non-CS domains.

- The iterative workflow—involving recurrent outline generation and multiple review-and-refine loops—requires repeated retrieval and model invocations, potentially leading to higher computational costs compared to one-shot systems.

- The human evaluation involves only seven PhD-level experts and compares performance against two baselines (AutoSurvey, SurveyForge), with a small sample size that may restrict the representativeness and robustness of the findings.

- The framework lacks evaluation on established benchmarks such as SurveyBench (proposed in SurveyForge), which would enhance the comparability and credibility of its performance assessment.

### Questions
- In the recurrent outline generation module of IterSurvey, how is the similarity threshold τ—which determines whether to accept candidate outline updates—specifically calibrated, and what effects do different values of τ have on the stability of the outline and its exploratory capability?

- Regarding paper cards, which distill papers into their core contributions, methods, and findings, does the framework adopt a specific criterion for deciding which details to retain or discard during the distillation process, and how does it prevent the omission of critical information?

- The ablation analysis presented in Table 5 appears insufficient; it would be preferable to conduct a single-component ablation study to more clearly assess the contribution of each module.

- In the comparative evaluation with baselines (e.g., AutoSurvey, SurveyForge), why was the analysis limited to the “content quality” and “citation quality” dimensions, without including a comparison of computational efficiency (e.g., time consumption and resource utilization)?

- While IterSurvey performs effectively on emerging topics lacking existing surveys, does it incorporate adaptation strategies for domains with extremely sparse literature, such as newly developed interdisciplinary fields?

- Finally, to what extent does the underlying base model rigidly determine the overall performance of IterSurvey?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the limitations of existing one-shot approaches to automatic literature survey generation, which often result in fragmented structures and information overload. Inspired by the iterative process of human researchers, the authors propose IterSurvey, a novel framework that employs a recurrent outline generation mechanism. In this framework, a planning agent incrementally retrieves and reads papers while dynamically updating the survey's outline to ensure coherence and exploration. A key feature is the use of "paper cards" to succinctly capture each paper's core elements, alongside a review-and-refine loop that integrates textual and multimodal content to enhance readability. Experimental evaluations demonstrate that IterSurvey produces surveys that significantly outperform state-of-the-art baselines in terms of content coverage, structural quality, and organization. Furthermore, the authors introduce Survey-Arena, a new pairwise comparison benchmark, to provide a more reliable method for assessing the quality of generated surveys against human-written ones.

### Strengths
1. Innovative Iterative Paradigm:The core strength of the work is its departure from the conventional one-shot generation model. By thoughtfully mimicking the iterative reading and writing process of human experts, the proposed framework provides a more natural and effective solution to the complex task of survey generation.

2. Well-Designed Mechanisms for Fidelity and Flow:The introduction of "paper cards" ensures faithful, paper-level grounding of the survey content. Combined with the dedicated review-and-refine loop for visualization enhancement, the framework successfully addresses key challenges in maintaining textual flow and integrating multimodal information.

3. Comprehensive Evaluation and New Benchmark:The paper not only shows strong experimental results on established metrics but also introduces Survey-Arena, a valuable benchmark that moves beyond absolute scoring. This pairwise comparison tool offers the research community a more nuanced and reliable way to evaluate machine-generated surveys in relation to a human standard.

### Weaknesses
1. Ablation Studies: Ablation experiments are required to validate the effectiveness of the individual modules designed within the workflow.

2. Regarding Evaluation Metrics: In Table 1, several evaluation scores are very close to the baselines. The persuasiveness of the current results is insufficient; it should be tested whether the performance gap remains similarly close if the scale of the scoring metric is widened.

3. Human Alignment: For the LLM-as-a-judge evaluation method, its reliability must be demonstrated through comparison with human evaluation. This is particularly critical for highly creative tasks like survey writing, which cannot be assessed simply by checking items against a template. Evaluating more creative aspects is both more difficult and more valuable.

### Questions
please refer to the weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
