# VACT: A Video Automatic Causal Testing System and a Benchmark

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
With the rapid advancement of text-conditioned Video Generation Models (VGMs), the quality of generated videos has significantly improved, bringing these models closer to functioning as "world simulators" and making real-world-level video generation more accessible and cost-effective. However, the generated videos often contain factual inaccuracies and lack understanding of fundamental physical laws. While some previous studies have highlighted this issue in limited domains through manual analysis, a comprehensive solution has not yet been established, primarily due to the absence of a generalized, automated approach for modeling and assessing the causal reasoning of these models across diverse scenarios. To address this gap, we propose VACT: an **automated** framework for modeling, evaluating, and measuring the **causal understanding** of VGMs in real-world scenarios. By combining causal analysis techniques with a carefully designed large language model assistant, our system can assess the causal behavior of models in various contexts without human annotation, which offers strong generalization and scalability. Additionally, we introduce multi-level causal evaluation metrics to provide a detailed analysis of the causal performance of VGMs. As a demonstration, we use our framework to benchmark several prevailing VGMs, offering insight into their causal reasoning capabilities. Our work lays the foundation for systematically addressing the causal understanding deficiencies in VGMs and contributes to advancing their reliability and real-world applicability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces VACT(Video Automatic Causal Testing), a fully automated framework to evaluate and measure causal understanding in Text to Video Generation Models.
The work leverages a large language model pipeline to assess causal behavior under difference scenarios via intervention experiments.
Analysis of causal learning are defined under 3 levels; text consistency, generation consistency and rule consistency.
The paper benchmarks several state-of-the-art VGMs (e.g., CogVideo, Hailuo, Kling, Veo3-Fast, Pika) across 19 scenarios and 718 evaluation videos, identifying significant gaps between human-like causal understanding and current model capabilities.

### Strengths
Work is well-motivated and introduces the first automated causal benchmarking pipeline offering a scalable solution to evaluating VGMs.\
Multiple self-checking and self-correction in causal rule extraction using LLMs to mitigate errors in the generation process.\
Crowdsourced comparisons between LLM-generated and human-annotated causal systems show that LLM annotations can outperform humans on rationality and soundness.\
The causal hierarchy (text / generation / rule consistency) provides a clear and extensible structure for measuring causal learning at multiple levels.

### Weaknesses
While the work attempts to offer a comprehensive evaluation suite to benchmark VGMs under the 3 mentioned frameworks evaluation is done only using VLLM for answer retrieval without any kind of upper-bound it is difficult to draw any conclusion from the metrics provided by the benchmark. An alternative would be perhaps an average or consensus across several VLLMs
Furthermore, N/A ratio is quite significant. Are the scores adjusted in anyway to account for this?

Claims at line 395 "Thus, our benchmark specifically requires models to handle multiple variables at once, including those corresponding to less common scenarios. The results highlight the models’ difficulty in dealing with complex properties and rare situations. This suggests that current models are still constrained to common scenarios and lack the generalization capability needed to effectively combine independent variables in broader contexts—a necessary capacity for building world simulators." needs to be better supported either with ablations such as models performance vs number of variables or complexities.\
The benchmark’s “truth” and “observe” metrics are under explained, and it is not entirely clear how to interpret discrepancies between them in causal terms. If the score is separated by grouping samples with and without generation errors, how much samples exists within each group? Is there a high deviation in the samples sizes?\

Minor comment\
Authors have a significant bulk of details in the appendix, i think they would be better served in summarizing some these in more details in main body itself especially on the evaluation process in section 4 and 5 to provide a better picture on the numerical significance of the results.

### Questions
Have you compared LLM-based evaluations to human causal judgments on a subset of data?
How stable are causal scores across different evaluator models or prompt perturbations?

The reliance on a single evaluator model and absence of clear upper/lower bounds make the causal metrics difficult to interpret confidently.\
The conceptual novelty,motivation and potential value are strong, but the empirical validation and benchmarks are insufficient for me to confidently believe in the supported the claimed conclusions.\

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a pipeline to check the factual and physical accuracy of videos generated by AI video generators. Pipeline is claimed to automatically extract out important rules that must be followed by generated videos from prompts and VLMs are used to check if these rules are followed in the generated videos. Rule extraction process is claimed to make use of causal graphs and is totally self-verified. So, yes, the problem of factual and physical accuracy is completely solved almost, as per the paper.

### Strengths
- Problem targeted by the paper is important

### Weaknesses
- presentation is quite poor and the paper lacks clarity along with typos. Paper seems like a rush job.
- my main issue is that paper claims that everything---exhaustive rule extraction, causal graph generation, physical quantification/estimation, rules verification---just magically works---we just had to mention the tasks to LLMs and VLMs, which no prior work did, and this paper simply did it.
- In reality and my experience, all these steps are very imperfect and self-checks are not reliable, and requires human oversight, but the paper claims otherwise, which I am not able to buy.
- Exhaustive rule extraction itself is extremely challenging
- Quantities like speed, density, etc are relative, but somehow VLMs are able to accurately measure them absolutely
- Paper also takes anecdotal observation and extends to be statistical 
- "LLM-generated annotations surprisingly outperformed those from human" when compared with human annotations---just how? How are apples going to look more like oranges than other oranges?!
- Experimental results and analysis is lacking. I did not find good insights in them. Qualitative results are also missing.
- I am really sorry, but I am not able to digest, that everything just magically works, and quite a few things in this paper are not adding up for me.

### Questions
Please see weaknesses, and consider them as questions.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces an automated framework to test whether text-to-video generative models exhibit causal understanding of physical scenarios. It uses an LLM to propose scenario-specific causal systems, performs interventional prompt design over root variables. It evaluates generated videos with three metrics: text consistency, generation consistency, and rule consistency. On the constructed comprehensive benchmark dataset, current models often fail to respect causal rules despite good visual fidelity.

### Strengths
- Task novelty: An automatic evaluation framework is critical for VGM development. The proposed LLM-driven construction of scenario-specific causal graphs and DNF rules, with self-check and rule-check loops, reduces manual effort and scales across domains.
- Metric quality: Clear separation of text consistency, generation consistency, and rule consistency, plus “truth/observe” variants that expose degenerate behavior. 
- Significance: The benchmark spans 19 scenarios and shows consistent causal errors in current systems; results provide actionable targets for future VGM development. 
- Human validation: Crowd experiments indicate the LLM’s causal system proposals meet or exceed human annotations on requirement, rationality, and soundness.

### Weaknesses
- Measurement dependency on VLLMs: Answer retrieval uses a VLLM with limited manual auditing. If the VLLM misreads subtle events, metric validity is affected. Authors note “random manual checks,” but a systematic calibration study is absent.
- Binary discretization and visibility constraints: Mapping continuous physics to binary, video-visible variables may oversimplify rules and bias towards easily observable effects. 
- LLM both proposes and helps judge: The same family of models proposes rules and generates probes; risk of confirmation bias remains.

### Questions
- How sensitive are the metrics to the choice of VLLM and its prompt template?
- Can you provide per-scenario confusion analyses showing which outcomes most often collapse to “common” behaviors?
- Do results hold when variables are non-binary (e.g., 3-level speed)? If not, how would VACT generalize?

### Soundness
3

### Presentation
3

### Contribution
3
