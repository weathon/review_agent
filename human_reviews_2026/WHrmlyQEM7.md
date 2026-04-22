# Evaluating Text Creativity across Diverse Domains: a Dataset and Large Language Model Evaluator

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
Creativity evaluation remains a challenging frontier for large language models (LLMs). Current evaluations heavily rely on inefficient and costly human judgments, hindering progress in enhancing machine creativity. While automated methods exist, ranging from psychological testing to heuristic- or prompting-based approaches, they often lack generalizability or alignment with human judgment. To address these issues, in this paper, we propose a novel pairwise-comparison framework for assessing textual creativity, leveraging shared contextual instructions to improve evaluation consistency. We introduce CreataSet, a large-scale dataset with 100K+ human-level and 1M+ synthetic creative instruction-response pairs spanning diverse open-domain tasks. Through training on CreataSet, we develop an LLM-based evaluator named CrEval. CrEval demonstrates remarkable superiority over existing methods in alignment with human judgments. Experimental results underscore the indispensable significance of integrating both human-generated and synthetic data in training highly robust evaluators, and showcase the practical utility of CrEval in boosting the creativity of LLMs. We will release all data, code, and models publicly to support further research.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper tackles the challenge of automatically evaluating textual creativity in large language models. Current creativity assessment rely heavily on human judgement, which is subjective, inconsistent, and expensive.
The author propose a new evaluation framework which is a context-aware, pairwise comparison method that judges creativity between two responses to the same prompt. The author propose a new dataset containing over 1 million instruction-response pairs across 87 domains, mixing human and synthetic data. Over this, the author train a new LLM-based creativity evaluation on the dataset.

### Strengths
1. The overall idea is clear and well-motivated. Focusing on fine-grained text-level creativity rather than broad model-level creativity makes the work much more practically valuable.
2. The authors explicitly explore the challenge of human annotation inconsistency through empirical analysis, which shows solid motivation and thoughtful experimental design.
3. The two key assumptions used for weak supervision are both empirically validated, and the paper carefully considers position bias in pairwise evaluation that is easily to be ignored.
4. The experimental setup is comprehensive and convincing. The baselines cover a wide range of categories—traditional metrics, general LLMs, and fine-tuned evaluators. The study includes analyses of consistency, ablation on data composition, OOD generalization, and even the ability to enhance creativity through DPO training. Altogether, this gives the results strong credibility.

### Weaknesses
1. There’s no discussion of whether the model maintains interpretative correctness—in other words, does it still “understand” why a response is creative under its defined criteria?
2. The OOD evaluation is relatively small in scale, which limits how strongly the generalization claim can be made.
3. Some references, such as Zhao (2024), have already been officially published, but the citations still point to arXiv versions. The bibliography needs to be updated.

### Questions
1. The CreataSet-Base domain distribution seems quite unbalanced. Is creativity really isotropic across categories? Some subdomains (like business writing vs. poetry) might not be equally suited for creativity evaluation. A more fine-grained per-domain analysis would help clarify this.
2. Were the choices of Qwen2.5-14B-Instruct and MiniCPM-2B-SFT arbitrary? Why were these two particular models selected for response generation and augmentation?
3. The pairwise comparison framework works well for controlled benchmarking, but its practical usability might be limited. Do you think it could be extended to continuous or scale-based scoring while preserving consistency?
4. It would be interesting to test CrEval on more classical creativity benchmarks—like Alternative Uses or other divergent-thinking tasks—where the “good” and “bad” responses are clearer.
5. In real-world creative contexts, can CrEval replace or complement the Consensual Assessment Technique (CAT)?

### Soundness
3

### Presentation
3

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
This paper focuses on evaluating the creativity of LLM-generated text. Specifically, it identifies two key challenges in evaluating creativity: 1) ensuring consistency in creative annotation for human experts; and 2) training a large-scale model for evaluating creativity given the scarcity of evaluation corpora. To address these challenges, this paper proposes CreataSet and CrEval, a large-scale pairwise comparison framework for cross-domain text creativity evaluation and a logic learning model (LLM)-based evaluator, respectively. Testing on various baseline models validates the effectiveness of CreataSet in evaluating creativity.

### Strengths
1) Large-scale and multi-domain dataset. CreataSet includes 100K+ human-level and 1M+ synthetic creative instruction-response pairs across 87 domains, which is promising in providing a scalable fundation for studying creative generation and evaluation.
2) Improved human label protocol. The proposed context-aware pairwise comparison protocol improves inter-annotator consistency (evaluated by ICC).
3) Comprehensive experiments. Multiple metics are applied for providing an through evaluation, such as F1score, Kappa score and Agreement rate.

### Weaknesses
1. The rules for quantifying creativity are not differentiated across different domains.For example, creativity is manifested differently in poetry and scientific writing. Future work should further differentiate the measurement of creativity for each domain.
2. Insufficient example/failure case analysis. The paper presents overall statistics but does not systematically list typical examples of discrepancies between CrEval and human behavior.
3. Insufficient generalization analysis. The paper lacks an assessment of the enhancement effect of CreataSet on diverse baseline models.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces CreateSet, a dataset with evaluation pipeline including a performant evaluator—CrEval. The author enriched the responses to existing prompts by generating more responses (and formed CreateaSet-Ext). The work annotated over 3,000 samples, with more using weak labels. The author extensively studied if the dataset helped create a more power evaluator.

### Strengths
- A reasonably well-curated dataset with good mix of topics for coverage.
- Good amount of human labels, and put into good use in training evaluator models with good comparison with other metrics and very extensive model lineup. I appreciate the authors showing many proprietary results. 
- Evaluation of the trained evaluator is complete and convincing.

### Weaknesses
- The definition of creativity, which is subjective, should be detailed better in this work. This is a key bottleneck of this work's quality and rigor.
- CrEval are comparison pairs of creativity. However, there might be some value to an absolute scale of creativity, especially if we want to rank model responses quickly. Motivation here is less clear.
- This work suffers from a few overstatements:
    - The paper prides itself over context awareness (i.e., showing a prompt when evaluating responses for creativity), including using an entire Fig 1 to emphasize. But the authors fail to explicitly demonstrate if this is common.
    - "87 domains" is a stretch. "Domain" is defined too loosely across the paper.
- While I believe performance parity with o3 is not so important given the model scale difference, the authors could use more motivation on why a smaller model as an evaluator is important when a large model can do better. 

I found these weaknesses not fundamental and am happy to update my views during the coming discussion period.

### Questions
- Line 43, How is problem-solving a single domain?
- It is important to break down the language used in the dataset. Is this dataset 100% (Simplified) Chinese?
- Line 47 "most methods evaluate creativity at the model or subject level rather than at the level of individual responses" Please better support this claim.
- How are deepseek models "Proprietary LLMs"?
- How are annotators compensated?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles the problem of evaluating text creativity by building a large-scale dataset (CreataSet and its extended version, CreataSet-Ext) and training a pairwise creativity evaluator CrEval. Each instruction is paired with multiple responses generated by different models under both ordinary and creativity-oriented prompts, with human-annotated labels serving as ground truth. The paper finds that CrEval aligns well with human judgments, generalizes to unseen domains, and can further help enhance model creativity. The results show that creativity-oriented prompts and stronger models tend to produce more creative responses, and data composition meaningfully affects CrEval’s performance.

### Strengths
The experiments are fine. CrEval consistently outperforms strong baselines including large proprietary models across proposed metrics. The paper includes appropriate ablations, along with OOD tests on external datasets. The authors further show CrEval can be used to improve model creativity

### Weaknesses
-  The paper mentions constructing tuples (I, R1, ..., Rk) but does not specify the exact value of k used in experiments. How many responses are generated and used per instruction? Does this vary across data sources?
- In constructing CreataSet-Ext, they prompt two models to generate more responses for augmenting each instruction. But there is no testing of whether these k responses actually exhibit meaningful diversity. If these responses are similar to each other, it could limit the model’s ability to learn fine-grained distinctions.
- At the beginning of the paper, they define creativity as “ideas or artifacts that are new, surprising and valuable”, which I personally appreciate. While novelty and surprise are well-measured, the “valuable” aspect is not explicitly evaluated. Aside from gpt-4o-mini filtering, there is no systematic check for helpfulness. A response can be novel but incoherent or unhelpful, which weakens the notion of genuine creativity.
-  The two core assumptions are validated with only 50 samples each with three annotators. This seems insufficient given the scale of the dataset (1M+ samples).
- The paper treats creativity evaluation as a pairwise comparison task without deeply analyzing what constitutes creativity or providing any meaningful understanding. What specific features, semantic patterns, or structural elements does CrEval learn to recognize as creative?
- Personally the task are a bit strange and adhoc. I would not treat them as something that requires creativity 
-Data contamination could inflate the performance of models being evaluated, especially for proprietary models whose training data composition is not fully disclosed.
- I am not convinced LLM eval is a surrogate for human eval. Are your 18 humans experts ?? Without knowing much we cant conclude anything

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
2
