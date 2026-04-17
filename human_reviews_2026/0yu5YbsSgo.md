# Toward Cybersecurity-Expert Small Language Models

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large language models (LLMs) are transforming everyday applications, yet deployment in cybersecurity lags due to a lack of high-quality, domain-specific models and training datasets. To address this gap, we present CyberPal 2.0, a family of cybersecurity-expert small language models (SLMs) ranging from 4B–20B parameters. To train CyberPal 2.0, we generate an enriched chain-of-thought cybersecurity instruction dataset built with our data enrichment and formatting pipeline, SecKnowledge 2.0, which integrates expert-in-the-loop steering of reasoning formats alongside LLM-driven multi-step grounding, yielding higher-fidelity, task-grounded reasoning traces for security tasks. Across diverse cybersecurity benchmarks, CyberPal 2.0 consistently outperforms its baselines and matches or surpasses various open and closed-source frontier models, while remaining a fraction of their size. On core threat-investigation tasks—such as correlating vulnerabilities and bug tickets with weaknesses—our best 20B-parameter model outperforms GPT-4o, o1, o3-mini, and Sec-Gemini v1, ranking first, while our smallest 4B-parameter model ranks second. On core cyber threat intelligence knowledge tasks, our models outperform almost all tested frontier models, ranking second only to Sec-Gemini v1. To foster reproducibility and practical adoption, we will release our models as open source.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents CyberPal 2.0, a family of cybersecurity-expert small language models across 4B to 20B parameters trained on SecKnowledge 2.0 dataset that created through expert-in-the-loop and multi-step grounding. The author claimed models in this work demonstrate frontier-level performance on cybersecurity benchmarks, with the 20B model outperforming GPT-4o, o1, o3-mini, and Sec-Gemini v1 on multiple tasks such as threat investigation tasks as well as smaller models with deployable on-premises for enterprise security operations.

### Strengths
1. Rigorous evaluation tasks: Authors of this paper use nine diverse, open cybersecurity benchmarks, following Sec-Gemini’s protocol to enable comparison with frontier models, showing comprehensive evaluation on CyberPal 2.0’s effectiveness task-based.

2. Dataset Pipeline: Actually I am more interested in the dataset pipeline used by CyberPal 2.0. SecKnowledge 2.0 integrates expert-in-the-loop schema refinement and multi-source grounding, addressing hallucination and factuality issues

3. Feasible and realistic motivation: This work focuses on small LLMs, which addresses real enterprise constraints with on-premises deployment and data residency with quantization results showing minimal degradation at 8-bit, making models deployment-ready.

### Weaknesses
1. Dataset transparency: One thing I am concerned about the dataset is even though the pipeline description is detailed (that is actually one strength as I noted), but dataset composition statistics (e.g., source mix, balance, potential overlaps with test sets) are not fully reported

2. Architectural novelty: One concern from the reviewer’s side is that work uses standard Qwen base models with conventional fine-tuning; the contribution is primarily data-centric rather than introducing novel model architectures or training algorithms.

3. Weakness on model evaluation: CyberPal is based on Qwen model, while in Tab. 1, authors only compared this work with Qwen 3 and GPT-oss. That might be intuitive as finetuning a LLM with specialized data will improve the performance of the model, from this point the actual effectiveness of this model family need further discussion and comparison. Actually it might be better to compare with models from other families (LLaMA, Mixtral etc.) even the task used for comparison is comprehensive.

### Questions
1. I am curious about what happens when using different LLMs (open-source alternatives like LLaMA, Gemma etc.) for the enrichment pipeline instead of only with GPT-oss or Qwen? How robust are the gains to this choice from the data pipeline of this work?

2. This paper focus on deployable of LLMs for cybersecurity. Have CyberPal 2.0 models been tested by security analysts in real world environments (not only the experiment environment)? What qualitative feedback exists on practical utility beyond benchmark scores, especially when deploying under limited budget situation?

3. I wonder have the authors considered data leakage prevented between SecKnowledge 2.0 and the evaluation benchmarks? Any data profiling (as the weakness point 1 mentioned) on this overlap?

4. How generalizable are these models to unseen real-world telemetry or even multilingual threat data?

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
This paper presents CyberPal 2.0, a family of cybersecurity-expert small language models trained on threat intelligence and security operations tasks. The authors introduce SecKnowledge 2.0, a chain-of-thought–enriched, expert-in-the-loop dataset that combines schema-driven reformatting, document grounding, and multi-step reasoning supervision to improve data fidelity and interpretability. Using this dataset, the models fine-tuned from Qwen and GPT-OSS bases achieve 7–14 % average gains over strong open-source baselines and often match or surpass frontier models such as GPT-4o, o1, and Sec-Gemini v1 on benchmarks like CTI-Bench and SecEval. Through ablations and LLM-as-a-Judge evaluations, the paper attributes these gains primarily to the enhanced reasoning quality of SecKnowledge 2.0.

### Strengths
- **Originality:** The paper introduces *CyberPal 2.0*, a comprehensive suite of small language models explicitly tailored for cybersecurity operations, combining domain-specific reasoning, expert-in-the-loop supervision, and multi-step grounding.  
- **Quality:** The methodology is systematic, featuring a well-designed data enrichment pipeline that integrates expert schema definitions, document retrieval, and reasoning-structure enforcement, validated through extensive ablations, multiple benchmarks, and LLM-as-a-Judge experiments.  
- **Empirical Rigor:** Results are comprehensive and demonstrate improvements over both open-source and frontier models across a wide range of benchmarks (e.g., CTI-Bench, SecEval, CyberMetric).  
- **Significance:** The work addresses an important and underexplored domain: cybersecurity, where privacy, compliance, and reliability constraints limit large-model deployment, offering a practically impactful alternative.

### Weaknesses
- **Limited ablation on dataset components:** The paper attributes its performance gains to the SecKnowledge 2.0 enrichment pipeline but does not disentangle the individual effects of expert-in-the-loop formatting, multi-source grounding, and LLM-judged reformatting. A more detailed ablation or sensitivity study is needed to identify which specific components most contribute to the observed improvements.  

- **Benchmark coverage limitations:** The evaluation focuses predominantly on cyber threat intelligence (CTI) tasks, leaving other key cybersecurity areas, such as malware analysis, intrusion detection, and digital forensics, underexplored. Broader coverage would strengthen claims of supporting the full spectrum of security operations and improve generalizability.  

- **Potential data contamination risks:** Since the instruction-tuning dataset is derived from the original SecKnowledge corpus, which overlaps conceptually with benchmarks like CTI-Bench RCM (e.g., CWE→CVE mappings), there is a risk that models have seen structurally similar data during training. The paper should clarify what de-duplication, filtering, or split-stratification strategies were applied to prevent data leakage and benchmark contamination.  

- **Lack of qualitative analysis of reasoning behavior:** The paper provides strong quantitative results but minimal qualitative evaluation of model reasoning patterns or failure cases. Presenting representative reasoning trajectories or side-by-side examples contrasting CyberPal 2.0 with baselines would substantiate claims of improved interpretability, factual grounding, and reasoning depth.

### Questions
- What were the total training hours required for each model size, and could the authors report them to help future researchers better estimate training efficiency and computational cost?
- Why was the evaluation conducted only on two CTIBench tasks (MCQ and RCM) rather than the full set of five?

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
5

### Summary
This paper introduces a family of small language models named CyberPal 2.0 designed specifically for cybersecurity tasks. The models are aimed at supporting full security operation workflows. The models are trained on an enhanced instruction dataset called SecKnowledge 2.0, which combines expert-crafted structures, multi-source reasoning, and document grounding. CyberPal 2.0 outperforms standard and even some frontier general-purpose models (like GPT-4o and o1) on cybersecurity benchmarks such as CTIBench and SecEval, while being much smaller and usable in private, on-premise settings.

### Strengths
1. CyberPal 2.0 consistently ranks at or near the top in cybersecurity-specific evaluations, sometimes outperforming leading closed-source models.

2. The authors detail the full pipeline, training recipe, and benchmarks, and plan to release datasets and model checkpoints for community use.

3. The use of multiple specialized benchmarks reflects a thorough and well-targeted evaluation of the model’s real-world applicability.

### Weaknesses
1. SecKnowledge 2.0 was mainly curated from structured and clean sources, but cybersecurity data in practice is often noisy, unstructured, and diverse. This creates a risk that the models may not generalize well to real-world environments.

2. Most of the evaluation is done on static, text-based benchmarks. However, real security workflows involve dynamic analysis, live investigations, and multi-modal data like logs and binaries, which the paper does not address.

3. Moreover, the model’s performance is evaluated on synthetic or curated benchmarks (e.g., CTIBench, SecEval, CyberMetric-2000), many of which are either derived from prior academic work or generated through LLM pipelines themselves. It would be better to consider real-world or end-to-end user studies involving SOC workflows or live incident data.

4. There is no technical discussion about how to prevent the models from being used for malicious purposes, even though the paper has claimed "responsible use".

### Questions
Please check the comments above.

### Soundness
3

### Presentation
3

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
This paper presents CyberPal 2.0, a suite of cybersecurity-oriented language models ranging from 4B to 20B parameters, along with a high-quality instruction-tuning dataset SecKnowledge 2.0. The key innovations lie in the expert-in-the-loop reasoning framework and the LLM-guided multi-step evidence retrieval enhancement, which jointly improve factual consistency and interpretability in reasoning chains. Across several authoritative cybersecurity benchmarks (e.g., CTI-RCM), the 20B model surpasses GPT-4o, o1, and Sec-Gemini v1, while the 4B model approaches o3-mini, demonstrating that high-quality domain-specific data can enable small models to rival state-of-the-art large models.

### Strengths
1. High-quality and reproducible domain data construction (SecKnowledge 2.0): The paper proposes a rigorous and reproducible methodology for building high-quality cybersecurity instruction-tuning datasets.

2. Fine-tuning from base models outperforms post-trained models: Fine-tuning directly from base models yields significantly higher performance (+15.16% vs. +5.62%), especially on reasoning-intensive tasks such as RCM (+22.7% vs. +1.85%). This finding provides valuable insights for the community’s training paradigm, challenging the conventional belief that “larger or more aligned models make better starting points.”

3. Strong alignment between performance and practicality: Experiments demonstrate that model capability and usability are well unified. The 20B model ranks first on CTI-RCM, outperforming closed-source models such as Sec-Gemini v1 and o1, while the 4B model performs comparably to o3-mini and far surpasses Mistral-Large. This highlights that domain-specific small models can achieve optimal cost-effectiveness in critical security tasks, meeting the pressing needs of enterprise on-premise deployment.

4. Comprehensive evaluation framework: The paper establishes an extensive evaluation system covering nine major benchmarks (including RCM, Adversarial CTI, and Relationship Prediction) and introduces a 10-category cybersecurity knowledge taxonomy (Appendix C) for fine-grained multi-label analysis, ensuring both broad coverage and deep alignment with real-world security operations.

### Weaknesses
1. Insufficient discussion on the generalization boundaries of SecKnowledge 2.0: All experiments are built upon and evaluated within the SecKnowledge data ecosystem, lacking zero-shot or few-shot transfer validation on entirely independent third-party datasets (e.g., real SOC logs or unseen APT reports). This limitation may lead to an overestimation of the model’s generalization ability.

2. Weak analysis of security risks and adversarial robustness: While the paper emphasizes “defensive use,” it lacks evaluation of the model’s behavior under adversarial conditions such as prompt injection, jailbreak, or hidden backdoor attacks (e.g., whether the model could be induced to generate exploit code).

3. Missing empirical data on inference latency and deployment performance: Although 8-bit and 4-bit quantization are mentioned, the paper omits critical deployment metrics such as inference speed (tokens/s), memory footprint, and batch throughput. For security operations, real-time responsiveness (e.g., <100 ms for SIEM integration) is as crucial as accuracy, and inferring deployability solely from model size is insufficiently rigorous.

### Questions
1. Section 3.3.2 states that the retrieval process retains up to four documents (K=2, R=2). However, tasks such as CTI-RCM often require integrating heterogeneous evidence from multiple sources (e.g., CVE descriptions, vendor patch bulletins, Bugzilla discussions, and CWE definitions). If critical evidence resides beyond the top-4 retrieved documents (e.g., due to limited query diversity or suboptimal ranking) or authoritative sources such as NVD are not indexed, can the model proactively detect evidence gaps and request supplementary retrieval?

2. Appendix B shows that fine-tuning from Qwen3-8B-base yields an average +9.54% gain over post-trained initialization (notably +20.85% on RCM). The authors attribute this to the base model’s greater receptiveness to high-quality supervision signals. However, since post-trained models are already aligned with general instruction distributions, could it be that their internal representations have solidified generic reasoning patterns, thereby hindering the learning of domain-specific causal chains in cybersecurity (e.g., vulnerability → exploitation precondition → adversary TTP)?

### Soundness
3

### Presentation
2

### Contribution
3
