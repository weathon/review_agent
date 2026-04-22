# Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Repetitive lexical patterns in LLM output, termed "slop," degrade writing quality through over-use and make AI-generated text immediately recognizable. We present Antislop, a comprehensive framework providing tools to both detect and eliminate these overused patterns. Our approach combines three innovations: (1) The Antislop Sampler, which uses backtracking to suppress unwanted strings at inference time without destroying vocabulary. (2) An automated pipeline that profiles model-specific slop against human baselines and generates training data. and, (3) Final Token Preference Optimization (FTPO), a novel fine-tuning method that operates in logit-space on individual tokens, surgically adjusting logits wherever a banned pattern has appeared in an inference trace. We demonstrate that some slop patterns appear over 1,000 times more frequently in LLM output than human text. The Antislop Sampler successfully suppresses 8,000+ patterns while maintaining quality, whereas token banning becomes unusable at just 2,000. Most importantly, FTPO achieves 90% slop reduction while maintaining or improving performance in cross-domain evals including GSM8K, MMLU, and creative writing tasks. In contrast, DPO suffers significant degradation in writing quality and lexical diversity despite achieving weaker suppression. We release all code and results datasets under MIT license.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents Antislop, a system for reducing over-represented lexical and stylistic patterns in LLM outputs. It has three components: (1) an inference-time Antislop Sampler that detects matched strings or regex patterns and backtracks to resample with a tunable penalty, (2) an automated “slop fingerprinting” pipeline that builds pattern lists by comparing model outputs to human baselines, and (3) a training objective FTPO that localizes updates to the token where a banned pattern would begin. Experiments on creative writing tasks and standard benchmarks report large reductions in targeted patterns with small quality changes. Throughput costs of the sampler are measured, and FTPO is proposed to retain suppression without runtime overhead.

### Strengths
1. Improving the humanness of LLM generation is an important problem with a lot of real-world applications. This paper provides a complete recipe from decoding to training, including a backtracking sampler, an automated fingerprinting step, and a localized training objective.
1. The decoding overhead is measured and a training procedure is designed to internalize the discouragement of undesired patterns.
1. Experiments span multiple model sizes and families and include both creative writing outcomes and collateral checks on standard tasks, with ablations and throughput analyses that support the claims

### Weaknesses
1. Training to discourage specific tokens or sequences has been studied somewhat extensively [1]. The contribution here can feel incremental without a stronger theoretical or empirical separation from this line of work.
1. Broader testing on recent creative-writing setups would strengthen the work. For instance, testing on poems, stories, and jokes could serve as a complementary setting [2]. 
1. Does suppressing the most prominent patterns surface other generation patterns? These patterns may not be human like either. I am curious about the results of the study in section 3 applied to the fine-tuned model. 

[1] Neural Text Generation with Unlikelihood Training. Welleck, Kulikov, Roller, Dinan, Cho, Weston. 2019.  ￼
[2] Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity. Zhang et al. 2025.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

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
This paper introduces ANTISLOP, a framework designed to identify and mitigate repetitive, low-diversity “slop” patterns in large language model outputs. The authors propose Final Token Preference Optimization (FTPO), a novel preference optimization method that regularizes model logits through a margin-based loss to discourage redundant phrasing. Combined with the Antislop Sampler, which automatically generates preference pairs from repetitive outputs, the framework enables self-supervised fine-tuning to improve linguistic diversity and stylistic quality. Extensive experiments across creative writing and reasoning datasets demonstrate that ANTISLOP effectively reduces repetition while maintaining fluency and coherence.

### Strengths
"slop" is a very interesting entry point. The experiments are comprehensive and well-executed, covering multiple datasets and comparing FTPO with existing training methods.

### Weaknesses
* The paper lacks qualitative examples illustrating how FTPO replaces repetitive phrases with new ones. Does the model exhibit “over-pruning” or semantic drift? Without such examples, the linguistic significance of the contribution remains unclear.

* The theoretical foundation of FTPO’s uniqueness and generalization is underdeveloped. At present, it still feels quite similar to DPO. You could first address this from a writing and framing perspective — for instance, by clarifying why existing preference optimization methods fail to handle “slop” effectively. Highlighting this gap might help crystallize the underlying theoretical novelty of FTPO.

### Questions
Why you choose mmlu and gsm8k? In my opinion, they are more related to scientific writing (maybe), rather than creative writing. But your dataset is based on creative writing.

### Soundness
4

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
This paper proposes a new method named Antislop to prevent the large language model (LLM) from generating repetitive or overused sequences while maintaining its generalization capability. More specifically, Antislop can build a preference dataset by identifying the overused patterns in the target LLMs and then regenerating the sequences from the starting point of the detected pattern with resampling. Based on the preference dataset, the finetuning method called final token preference optimization (FTPO) is applied to adapt the target LLM, such that it could generate repetitive sequences much less frequently. Experimental results show that the FTPO can achieve a higher suppression rate of overused patterns than DPO and logit-based token banning while maintaining its performance on the downstream tasks.

### Strengths
The strengths of the paper are listed as follows.    
1. It is great that the authors have a plan to release their dataset and codes for reproducibility.   
2. The paper is well motivated. Generating repetitive patterns is a big issue for LLMs and could affect the output quality significantly. Therefore, an effective method to address this issue is important.

### Weaknesses
The weaknesses of the paper are listed as follows.
1. The training-based method is model-specific. It requires building a different preference dataset for different models, which is costly and thus limits its application and impact.   
2. The experimental results are not convincing enough. First, the authors only compare their method with DPO and logit-based token banning. There are many existing works mentioned in the related work section. However, the authors did not provide any justification for not comparing with the remaining existing works on this topic. Second, the authors did not provide an ablation study of the regularization terms defined in Section 5.2.    
3. Some important technical details are not clear. I list some of them as follows.   
a. When quantifying the slop, it is unclear how to define the pattern $p$.   
b. When detecting the banned patterns, Antislop will lower the initiating token’s probability. However, it is unclear how to adjust the probabilities of the remaining tokens.    
c. In FTPO, it is unclear how to obtain the set of chosen tokens for each rejected token.    
d. Many hyperparameters in FTPO are logit-space operations. Will it make the choice of hyperparameters tricky since the search space is large?    
e. In equation of $L_{target}$, it is unclear how to decide the value of $y_{ref}$.     
4. The writing and organization of the paper can be improved for better readability. First, it would be better to place Figure 1 on page 5 since the first time it is referred to is on page 5. Second, it would be better if the authors could provide a figure to illustrate the difference between the proposed method and the other types of existing works to highlight the technical novelty of the proposed method. Third, before you describe the loss function of FTPO, it would be better to clearly define the form of the dataset. For example, does each data sample have labels or pseudo-labels? How to get these labels? Is the loss only calculated on the response part of each sample? Last, when introducing the benchmarks, it would be better to describe the evaluated dataset and the metrics separately instead of merging them into the same part.   
5. The FTPO can prevent the target LLMs from generating the overuse patterns detected by the preference dataset. However, it is unclear whether it will incur new overuse patterns in the finetuned LLMs.    
6. It would be better if the authors could provide the cost analysis of the FTPO, which is important to show the scalability of the proposed method, especially when the target LLM is large.

### Questions
Please refer to the weaknesses section for details.

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
The paper proposes an approach to discourage repetitive patterns generated from LLMs. It first gives an approach to identify repetitive pattern by sampling model outputs and comparing frequencies of ngrams with standard frequencies. Once identified they propose and compare a couple of ways to reduce their occurrence while maintaining generation quality. The first approach is to backtrack whenever a bad pattern is detected and then regenerate from that point by sampling again while discouraging the token that led to the bad pattern by renormalizing probabilities. To mitigate the decrease in throughput due to the backtracking, they propose to use finetuning approaches using synthetic data. They propose a new approach FTPO which precisely edits logits for preferred and rejected tokens, and is able to maintain the generation quality better than DPO finetuning while reducing the undesired n-gram frequency more substantially.

### Strengths
- The problem of reducing repetitive patterns in LLM generation is well motivated and an important one to solve in order to get natural texts from LLMs.

- The soft-banning approach proposed provides customizable adjustment strengths which are useful to add flexibility.

- FTPO approach is well motivated with well thought out components to encourage the logit distribution you want while constraining the logits so that they don’t change too much, with different components of the loss encouraging different goals.

- The results show that FTPO can provide better tradeoff between quality of output and reduction in slop than DPO and other approaches (Figure 3).

### Weaknesses
- The slop fingerprinting would be dependent on what queries are fed into the model to get the outputs, and so it might be difficult to get fingerprints (overrepresented n-grams) that are more universal (i.e. show up in wide-scale use in general domains).

- Not enough attention is paid to detecting and filtering semantic fingerprints that are beyond specific ngrams and of more semantic nature. For example patterns like ”it’s not just X, it’s Y.” are discussed in the paper very briefly in section 6.5 which proposes using regular expressions to detect them, but it might lead to reduction in output quality if we prevent entire clause structures like that from being generated (which is not studied in the work).

- The approach of soft banning would require tuning strength parameters to use for renormalizing probabilities and it is not clear to me that a single parameter setting will work well across multiple domains in a wide scale real-life deployment.

### Questions
- What was the value of n in your ngrams for the investigation in section 3 and how did you decide on a value for that?

- I would suggest adding equation numbers to equations (e.g. the loss equations in FTPO) for ease of referring to them

- In line 323, it says "we freeze all layers except the last 5 and lm head" This does not sound like standard practice.. do you do experiments with training all layers via LoRA and see how that works compared to only the last 5 layers? It is okay to maybe use a lower rank LoRA but use it on all layers if gpu memory constraints are there.

### Soundness
3

### Presentation
3

### Contribution
3
