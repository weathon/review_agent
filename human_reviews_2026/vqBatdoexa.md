# Finding the Cracks: Improving LLMs Reasoning with Paraphrastic Probing and Consistency Verification

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Large language models (LLMs) have demonstrated impressive performance across a variety of reasoning tasks in domains such as mathematics, coding, and planning, particularly when guided by chain-of-thought prompting to elicit intermediate reasoning steps. However, their problem-solving ability often declines on more complex tasks due to hallucinations and the accumulation of errors within these intermediate steps. Recent work has introduced the notion of critical tokens—tokens in the reasoning process that exert significant influence on subsequent steps. Prior empirical studies suggest that replacing critical tokens can refine reasoning trajectories and lead to correct answers. Nonetheless, reliably identifying and exploiting critical tokens to enhance LLM reasoning remains challenging. To address this, we propose the Paraphrastic Probing and Consistency Verification (PPCV) framework, which leverages critical tokens to improve reasoning performance. PPCV operates in two stages. In the first stage, we roll out an initial reasoning path from the original question and then concatenate paraphrased versions of the question with this reasoning path. Feeding these inputs into the LLM yields token-level logits, from which we identify critical tokens based on mismatches between the predicted top-1 token and the expected token in the reasoning path. A criterion is employed to confirm the final critical token. In the second stage, we substitute critical tokens with candidate alternatives and roll out new reasoning paths for both the original and paraphrased questions. The final answer is determined by checking the consistency of outputs across these parallel reasoning processes. We evaluate PPCV on mainstream LLMs, including Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2 and Qwen3-32B, across multiple benchmarks covering mathematics and logical reasoning. Extensive experiments demonstrate that PPCV substantially enhances the reasoning performance of LLMs compared to baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new inference time optimization method: paraphrastic probing and consistency verification. The main idea is (1) identify the critical tokens (tokens that may trigger very different reasoning paths) by running inference on paraphrased inputs + original response; (2) generate new rollouts by replacing the critical tokens in the original response and finally do a majority vote on all responses.
The empirical results show that the proposed method outperforms previous inference-time scaling methods such as self-consistency, across reasoning-intensive tasks such as Math and ARC-challenge.

### Strengths
- The paper is well-written. The method is clearly explained and easy to follow.
- The experimental results and ablation supports most of the paper claims, and shows the importance of both the critical tokens and the paraphrased consistency.

### Weaknesses
- One important baseline is still missing: simply perform majority voting but on the paraphrased questions (be sure to keep the total number instances the same as the PPCV method) => this will show that whether the rollout based on critical tokens really helps. Although the authors provide the comparison with random tokens, it does not make sense, since replacing random tokens may artificially shift the model's distribution too much and hurt it's original performance.
- The computational analysis is too brief and lacks important details: I am suspicious about the correctness of the inference time figure: in most datasets, PPCV uses only about 2x the inference time than simple CoT, this does not seem to be possible. Becuase the computation overhead of PPCV comes from (1) N forward pass on q1...qN with the orignal question; (2) K * N generation sequences for each critical token and each question variant. Particularly, suppose the critical token is on average in the middle of the response, the stage (2) should add approximately 0.5 * K * N times of the original CoT generation time (regardless of the prefilling time).

### Questions
- The fonts in most figures are too small, e.g., Figure 3,4,5; would be good to enlarge

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
This paper aims to improve the reasoning performance of LLMs by proposing a Paraphrastic Probing and Consistency Verification (PPCV) framework. Specifically, PPCV consists of two stages: 1) roll out the initial reasoning path from the original question, then concatenate paraphrased questions with the reasoning path, which is fed into the LLM to identify the critical tokens; 2) leverage the extracted critical tokens to refine the initial reasoning path via a self-consistency mechanism. Experiments on multiple mathematics and logical reasoning benchmarks show that PPCV outperforms the vanilla CoT and some baseline decoding methods.

### Strengths
-	The paper is overall well-written and easy to follow.
-	The idea of locating the critical tokens makes sense, and does not rely on external models.
-	Experimental results show that PPCV outperforms the other counterparts by a clear margin.

### Weaknesses
-	My main concern is the efficiency of the proposed framework. If I understand correctly, PPCV requires multiple passes of LLMs, e.g., paraphrasing the question, obtaining the initial reasoning paths, obtaining the critical tokens by feeding into multiple paraphrased questions, and generating a group of new trajectories. The total framework is complex and would lead to much inference latency. The comparison of the inference time in Figure 7 is also confusing. For example, in the case of the SVAMP test set, PPCV only brings a slight inference latency against the vanilla CoT method, which is obviously counterintuitive. The authors should provide more explanation.
-	The technical contribution is limited. In my opinion, PPCV does not propose new technologies, but uses existing methods and ideas to solve a relatively new problem.
-	In the ablation study, the authors do not analyze the influence of paraphrased questions. For example, remove the paraphrase processes in PPCV and explore the performance changes.
-	Algorithm 1 is a little hard to paraphrase. It will be better to improve the presentation carefully.

### Questions
See the Weaknesses.

### Soundness
3

### Presentation
2

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
This paper introduced PPCV, a framework that leverages critical tokens to improve reasoning performance. First, the framework roll out an initial reasoning path from the original question and then concatenate paraphrased versions of the question with this reasoning path. Everything is passed to the LLM to get token-level logits to identify critical tokens based on mismatches between the predicted top-1 token and the expected token in the reasoning path. In the second stage, critical tokens are substituted with candidate alternatives and the framework rolls out new reasoning paths for both the original and paraphrased questions. The final answer is determined by checking the consistency of outputs across these parallel reasoning processes.

First, the authors conduct an empirical analysis of the effectiveness of critical tokens: identify the one critical token, replace it. Compare performance with self-consistency. Indeed, critical token replacements is more beneficial.

In terms of methodology, an LLM is first prompted to paraphrase a problem multiple times while keeping all numerical values etc the same. Then a reasoning path is computed with the original question. The reasoning path is concatenated to each paraphrased version. Then the top-1 log-probs are computed for each token. The first one that does not match the the top-1 on the original problem is considered critical.
To refine the original reasoning path with alternative tokens, an LLM is prompted from the beginning until the critical token. A consistency score is computed on all alternatives.

Regarding the experiments, Llama 8B, Mistral 7B, and Qwen 3 32b are used on multiple reasoning datasets. Baseline consists of CoT, ToT, guided decoding, predictive decoding and phi-decoding. Each dataset is paraphrased 3-5 times. PPVC seems to lead to the highest performance on 7B thinking models and Qwen3 32B. I would encourage the authors to compare higher reasoning models and use pass@k. The ablations are well done. Finally, I'm a bit disappointed by the computational cost analysis with Figure 7. There is no confidence interval, variation, and it is hard to weight the trade-off between performance and efficiency. Could you provide a Pareto plot with performance vs throughput and include confidence interval?

### Strengths
- Interesting idea.
- Good ablation analysis.

### Weaknesses
- No analysis with larger reasoning models.
- No use of pass@k.
- Computational cost analysis is insufficient.

### Questions
- Could you compare performance on larger reasoning models?
- Could you report pass@k for Table1 and Table2 (k>= 4)?
- In Table 3&ç, please add the original performance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes PPCV (Paraphrastic Probing and Consistency Verification), a two-stage framework for improving LLM reasoning by identifying and leveraging "critical tokens" - tokens that significantly influence subsequent reasoning steps. The method first uses paraphrased versions of questions to identify critical tokens where the model's predictions diverge, then substitutes these tokens with alternatives and uses consistency to select the final answer.

### Strengths
* Novel approaches. The method of identifying critical tokens is sound and interesting. The method does not require heavy token-level annotation to get token-level classification.
* Results are promising. Improvements are observed consistently across 5 different datasets and 2 models.

### Weaknesses
My main concern is that the methods do not seem generalizable.

* First, it is hard to control the quality of the paraphrased questions. Although authors design careful instructions to ensure numbers are not changed, we do not have an evaluation metric to control.

* Second, the methods can only be applied to greedy decoding; otherwise, the current method will tend to select non-top-1 tokens as the critical token.

* Third, the current methods work well for one critical token but is hard to generalize to multiple critical tokens.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
