# AttentionInfluence: Adopting Attention Head Influence for Weak-to-Strong Pretraining Data Selection

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Recently, there has been growing interest in collecting reasoning-intensive pretraining data to improve the reasoning ability of LLMs. Prior approaches typically rely on supervised classifiers to identify such data, requiring labeling by humans or LLMs, often introducing domain-specific biases. Since attention heads are crucial to in-context reasoning, we propose \textbf{AttentionInfluence}, a simple yet effective, \textbf{training-free} method \textbf{without supervision signal}. Our approach enables a \textbf{small pretrained language model} to act as a strong data selector through a simple attention head masking operation. Specifically, we identify retrieval heads and compute the loss difference incurred by masking them. We apply AttentionInfluence to a 1.3B-parameter dense model to conduct data selection on the SmolLM corpus of 241B tokens, and mix the corpus with the selected subset comprising 73B tokens to pretrain a 7B-parameter dense model using 1T training tokens and the Warmup-Stable-Decay (WSD) learning rate schedule. Experimental results demonstrate substantial improvements, ranging from \textbf{1.4pp} to \textbf{3.5pp}, across several knowledge-intensive and reasoning-heavy benchmarks (i.e., MMLU, MMLU-Pro, AGIEval-en, GSM8K, and HumanEval). This demonstrates an effective \textbf{Weak-to-Strong} scaling property, with small models improving the performance of larger models---offering a promising and scalable path for reasoning-centric data selection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors proposed to use a pretrained LLM to select high-quality and reasoning-intensive pretraining data. Specifically, they first find the retrieval heads of a small model, and then calculate the loss gap with or without these retrieval heads. A higher loss gap indicates a higher reasoning intensity of the data. Experiments have demonstrated the effectiveness.

### Strengths
1. The paper has a clear structure and is easy to understand.
2. The proposed method has good practical application scenarios.

### Weaknesses
1. The experimental design may not be reasonable enough. Compared to the baseline, the training data is mixed with additional screened 73B data. Should the baseline data also include randomly sampled 73B data?

2. Lack of further experimental analysis. In order to further validate the practical application value of the proposed method, the following analysis may be necessary:

2-1. Is the search head consistent across different corpus data? If not, is it necessary to conduct targeted searches for specific language materials?

2-2. Do the screening model and training model need to be from the same series? For example, can the data filtered by Llama be used to train Qwen?

2-3. In practical applications, CPT data filtering may be a more common scenario. In this scenario, how effective is the proposed method? For example, in CPT training that requires enhanced reasoning ability, the baseline model trained on 400B corpus, while the comparison method trained on high-quality 100B corpus filtered from 400B corpus. If the performance of the comparison method can actually reach or even exceed that of the baseline model, it can demonstrate greater practical value.

2-4. Performance and efficiency analysis of different screening models.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to improve data selection for pre-training by leveraging signals from attention heads. Specifically, the authors analyze how large language models allocate attention during reasoning and generation, and introduce a new metric, the AttentionInfluence Score, which quantifies the relative importance of tokens and data based on their attention contributions. The proposed approach first identifies attention heads that are critical for reasoning, masks them in a reference model, and then computes the attention differences between the base and reference models to measure the influence of the data. The authors use the 1.3B model for data selection on the SmolLM corpus, and then pretrain the 7B model on the combined corpus of SmolLM and the selected data instances, showing that the proposed data selection strategy outperforms relevant baselines.

### Strengths
* The design of the proposed metric (AttentionInfluence Score) is convincing for data selection. 
* The proposed data selection process outperforms relevant baselines.

### Weaknesses
* The samples used to identify the important attention heads are very important for the later data selection, as the data instances for pre-training are selected mostly from their signals. In Section 4.1, the authors mention that it is derived from 800 synthetic samples, and it is questionable whether the selected data instances are just very similar to those synthetic samples. Also, more details on constructing those samples and their quality should be provided. Lastly, it would be great if the authors could justify why only the top 5% of the attention heads are selected for data selection. 
* It is not intuitive that the authors select the pre-training data from the SmolLM corpus and then use the SmolLM corpus + the selected data instances from the same SmolLM corpus. In other words, the selected data instances for pre-training are just a subset of the SmolLM corpus (which is also used for pre-training), and it seems this setting just unsamples the existing data rather than demonstrating true data selection benefits.
* For pre-training research, it would be great to show the scaling law as a function of the number of parameters, in addition to the number of tokens provided. 
* It is unclear why the authors report the results without learning rate decay settings in the main tables. 
* Recent training strategies of modern foundation models include mid- and post-training. It is questionable whether the pre-trained model with the proposed data selection strategy can still be effective after mid- and post-training. In addition to this, it would also be interesting to see whether the proposed data selection strategy can be beneficial for the mid- and post-training stages, where the data selection process is typically more rigorous than the pre-training stage. 
* The term Llama2-like-1.3B model is unclear. Is this not the Llama2 model?
* In Line 204, two references are broken.

### Questions
Please see Weaknesses above.

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
This paper proposes AttentionInfluence, a training-free and unsupervised method for pretraining data selection. The key idea is that data activating more retrieval heads are high-quality and encode reasoning-related behaviors. Using a 1.3B model to select the top ~20% (73B tokens) from the SmolLM corpus (241B tokens) based on the AttentionInfluence score, and mixing them to train a 7B model with 1T total training tokens, the approach outperforms both unsupervised and supervised baselines on reasoning and knowledge benchmarks.

### Strengths
1. The paper introduces a new perspective by leveraging mechanistic interpretability (retrieval head behavior) for pretraining data selection.
2. It provides detailed ablations and qualitative analyses.
3. The method is effective as demonstrated by the pretraining experiments while being entirely training-free and unsupervised.

### Weaknesses
Since only one pretraining corpus (SmolLM) and one pretraining model (a 7B model) are used, the robustness and generalizability of the method may be limited. Considering the high cost of pretraining and the theoretical generality of the AttentionInfluence method, it should be possible to further verify its effectiveness through post-training experiments.

### Questions
1. The full SmolLM corpus contains 241B tokens, and the selected subset adds another 73B tokens, while the total training uses 1T tokens. How many of these 1T tokens come from the selected subset (for both AttentionInfluence and FineWeb-Edu Classifier), and how does this proportion differ from the baseline?
2. Is AttentionInfluence applicable to the mid- or post-training stage? Could you provide results using a smaller and different corpus and a different model at the mid- or post-training stage to verify the robustness of this method?

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
The paper proposes AttentionInfluence, a new method for efficient pre-training data selection by leveraging the retrieval heads. AttentionInfluence identifies the important attention heads in a small LLM for retrievals and selects pre-training data examples based on the loss difference over examples between keeping and masking out such attention heads. Experiments show that AttentionInfluence selects data that improves downstream performance on knowledge-intensive and reasoning-intensive tasks, and is more efficient than other data selection baselines, as a small LLM is employed as the data selector.

### Strengths
1. The paper proposes a new pre-training data selection method with a focus on the efficiency of data selection and weak-to-strong generalization. Such new perspectives on pre-training data selection contribute to the literature beyond language modeling.

2. The proposed method is well grounded in the interpretability literature, and experiments across multiple benchmarks provide empirical support.

3. The paper presents comprehensive analyses of different design choices associated with the proposed method.

### Weaknesses
1. There exists a mismatch between the functionality of retrieval heads (long-context retrieval and reasoning) and the downstream task of the paper (pre-training data selection), and this leads to my concern about whether the proposed method is appropriate and well-motivated. In the literature, the retrieval heads are shown to be important for long-context retrieval, understanding, and reasoning tasks (e.g., needle-in-the-haystack), but their influences on short-context tasks are much less strong. In the pre-training literature, retrieval heads are also discussed more in the context of long-context pre-training or context extension. However, this paper does not specifically target long-context pre-training, and all the downstream tasks being evaluated (e.g., those in Table 1 and Table 2) are short-context tasks. Therefore, in my opinion, there is a mismatch between the methodology and the downstream task in this paper. While the author might have been aware of the effect of context length, as Section 6 shows that AttentionInfluence selects longer data examples, the discussion is rather limited; this paper needs to be better motivated by including more discussions/experiments on the effects of context length.

2. The empirical result is relatively weak compared to the baselines. For example, in Table 1, AttentionInfluence-1.3B is worse on average compared with the FineWeb-Edu Classifier baseline, and < 1% better than the simple PPL filter baseline. In a sense, this is intuitive because of the mismatch in W1: most of the evaluation tasks are short-context, and data selected by leveraging retrieval heads might not show large enough effects for such tasks. I would expect AttentionInfluence to outperform other baselines more on long-context tasks. 

3. The analyses depend heavily on loosely defined metrics. Several analyses in Section 6 use the metrics of Education Score and Reasoning score to emphasize the strength of the proposed method. While I appreciate the in-depth analyses present, the two metrics are loosely defined: (1) They are not commonly used metrics in the literature, as I did not find references provided in this paper or relevant papers in the literature that use these metrics, especially the education score; (2) They are not well-defined in the LLM-as-a-judge prompt. As the prompt in Appendix J, there is no definition for the term "educational value" and the definition for the term "reasoning-intensive" is also slightly vague. Given the vague definitions, it is unclear if the LLM-as-a-judge scores accurately capture the desired features of selected data. For example, the education scores in Table 10 and Table 11 clearly saturate and cannot differentiate between different methods at all.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
2
