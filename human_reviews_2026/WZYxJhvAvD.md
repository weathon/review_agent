# Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
The releases of powerful open-weight large language models (LLMs) are often not accompanied by access to their full training data. Existing interpretability methods, particularly those based on activations, often require or assume distributionally similar data. This is a significant limitation when detecting and defending against novel potential threats like backdoors, which are by definition out-of-distribution.
In this work, we introduce a new method for understanding, monitoring and controlling fine-tuned LLMs that interprets weights, rather than activations, thereby sidestepping the need for data that is distributionally similar to the unknown training data. We demonstrate that the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors. By monitoring the cosine similarity of activations along these directions, we can detect salient behaviors introduced during fine-tuning with high precision.
For backdoored models that bypass safety mechanisms when a secret trigger is present, our method stops up to 100% of attacks with a false positive rate below 1%. For models that have undergone unlearning, we detect inference on erased topics with accuracy up to 95.42% and can even steer the model to recover "unlearned" information. Besides monitoring, our method also shows potential for pre-deployment model auditing: by analyzing commercial instruction-tuned models (OLMo, Llama, Qwen), we are able to uncover model-specific fine-tuning focus including mathematical problem solving, emoji usage, and Midjourney prompt generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new method to infer behaviors that LLMs learned during finetuning, relying on the weight difference between the base and finetuned model, and not only on the activations.

More specifically, for projection matrices in almost every layer, they decompose the difference into singular vectors and keep the top-$k$ singular vectors. Then, at inference, they compare the similarity of activations with the selected singular vectors. If the similarity exceeds a threshold, it is flagged as abnormal.

They show that their method can be used to monitor and detect inputs/outputs corresponding to backdoor triggers and unlearned content. It can also be used to steer the model to relearn previously unlearned content. Lastly, they show that they can infer which topics were present in the finetuning dataset.

### Strengths
- The method is well described and easy to understand. While, as acknowledged by the authors, using SVD on weight differences has already been applied in model merging, their method provides a novel application of SVD with interesting potential for model interpretability and monitoring.
- In their evaluation, the authors apply their method to several backdoor and unlearning techniques, as well as for open-ended auditing on multiple models.
- The authors provide a theoretical guarantee on the FPR of their method, assuming that the non-anomalous data come from the same distribution as the training set.

### Weaknesses
- The experimental results are disappointing.  
	- For backdoor monitoring, as shown in Table 7, the variance for FFT backdoors is very high, with success rates reaching almost 0 in some cases. This is very similar to what we observe in Table 12 for PPO with full weights instead of the difference.  
	- For open-ended auditing, the results are very noisy. With Olmo, some flagged keywords (midjourney or image prompt) are not actually present in the training dataset, whereas some keywords like math are strongly present in the training data but not as strongly in the flagged data. Hence, the correlation is very weak, which means the interpretation for closed models is not precise. Moreover, the authors use keyword search because clustering was too noisy, which suggests that the signal, if any, is very noisy. Additionally, unlike clustering, keyword search is more biased, harder to reproduce, and harder to generalize.  
- Unlike activation-based methods, WeightWatch requires access to the base model weights, which may not be available in some cases.  
- Section 4 presents a case study to show why activation-based methods are not as powerful, but the experimental details are not clearly presented, making it hard to understand exactly what results are presented in Figures 2 and 3 (which activations are used for PCA? for probing?...).

### Questions
- What is the cause of the high variance observed in Table 7?
- For the open-ended experiment (Table 6), could you provide a more systematic analysis of the data and make the results less noisy?
- When steering models, could you provide additional LLM benchmark results (such as HumanEval or GSM8K)? Also, could you add an experiment in which you steer a "normal" model (which may occur due to false positives)?
- The theoretical value from Remark 1 assumes the validation data come from the same distribution as the test data. What happens if you calibrate your method on a different dataset than the test set? Additionally, the observed FPR in your experiments is around 1\%. Why don't you increase the size of your calibration dataset to achieve a lower FPR?
- Could you describe more precisely what data are plotted in Figures 2 and 3?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a weights-based method for detecting behaviors introduced during LLM fine-tuning. They focus on open-weights models, for which multiple fine-tunes are available online, and for which one normally has access to the pre-trained checkpoint.  

The goal is to detect at inference time behaviors introduced during fine-tuning, without needing access to the model’s fine-tuning corpus. This is especially relevant in safety settings. For example, a malicious actor might have planted a backdoor into the model, causing it to act maliciously if a trigger phrase is present in the prompt. Data-dependent approaches, such as SAEs, are prone to miss such a hidden behavior in case no prompt eliciting the malicious behavior is known. 

Their method, called WEIGHTWATCH consists of computing the singular value decomposition of the weight differences between the fine-tuned and pre-trained models. The top $k$ left singular vectors for each layer are referred to as behavior vectors. 

At inference time, one can then compute the cosine similarity of output activations with the singular values identified. If such cosine similarities fall outside the usual range, the sample is labeled as anomalous for that model.  

Importantly, the “usual values” for cosine similarities between linear layer outputs and the behavior vectors need to be calibrated on a calibration set. However, this differs from data-dependent methods in that the calibration set need not contain examples of anomalous behaviors, or otherwise correspond to the fine-tuning corpus.  

Across nine controlled backdoor setups (LoRA, full-parameter, and poisoned-PPO), WEIGHTWATCH flags 56–100% of trigger activations “on first sight” while keeping false positives under ~1% on benign data; on the PPO trojan models it attains near-perfect detection where black-box BEAT and simple activation-norm baselines falter. For unlearning, it detects 36–95% of queries that touch erased content at ≤1.8% FPR, and its steering variant can partially undo unlearning (e.g., recovering 76.9% of WMDP-Bio and 38.6% of WMDP-Cyber on Zephyr-RMU) and even jailbreak a circuit-breaker model to 82.1% success. Finally, an “in-the-wild” audit of popular instruction-tuned models surfaces model-specific fine-tuning imprints (e.g., marketing strategies, Midjourney-style prompt generation, Chinese ideological content, and distinctive language distributions), illustrating utility for pre-deployment model forensics as well as runtime monitoring.

### Strengths
- Simplicity of implementation: the method does not involve significant custom or complex routines; rather, it relies on standard primitives such as SVD and cosine similarity. This lowers the barrier to adoption by the wider community, which is an important strength. 
- Strong backdoor detection results, especially for LoRA: results in table 2 indicate WEIGHTWATCH can detect nearly all anomalous behavior instances in backdoored models at low false positive rates. Results for full fine-tuning are also strong, but weaker than for LoRA. Given the popularity of LoRA for fine-tuning open models (due to compute constraints of non-frontier-lab players), this nonetheless remains an important area to do well in when it comes to backdoor detection. 
- Demonstrating unsupervised re-learning of harmful capabilities: the author’s results on re-activating unlearned dangerous capabilities highlight a cheap and simple attack vector on open-source models. It is a relevant research direction to the community to devise fine-tuning methodologies robust to such forms of re-learning. 
- Clarity and presentation: the ideas in the paper are cleary explained and easy to follow. The presentation is high-quality throughout.

### Weaknesses
- The presentation in section 6.3 could be clearer. For instance, it is not clear to me what is the criterion for bolding numbers in table 6, nor how exactly the presented numbers allow us to trace back the observed behaviors of OLMo back to ShareGPT. Regarding keyword search, it is not clear how the authors selected the keywords.  
- It would be helpful to include more detail in Appendix F clarifying these points. That will help alleviate any concerns readers might have regarding cherry-picking or foregone conclusions. 

Other than this point, I consider the paper to be a high-quality contribution to the LLM transparency and interpretability literature.

### Questions
- Would the authors be able to give a more complete overview of methodology in Section 6.3, especially pertaining to how keywords were selected, and what are the logical steps for concluding that e.g. Midjourney prompt writing or Chinese ideology feature in the fine-tuning data mix?

### Soundness
3

### Presentation
4

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
This paper performs data-free fine-tuning behavior detection based on weight differences. Specifically, it determines whether fine-tuning has introduced certain behaviors by measuring the similarity between the input activations and the principal directions of the weight difference. The authors demonstrate the effectiveness of the method in backdoor, unlearning, and open-ended auditing scenarios.

### Strengths
1. The behavior detection approach proposed in this paper has the practical advantage of not requiring access to the fine-tuning data.


2. During inference, the method only needs to compute the similarity between the input activation and the precomputed weight difference directions, resulting in low additional cost.


3. The experiments cover multiple tasks, including backdoor detection, unlearning detection, and open-ended auditing, demonstrating the effectiveness of the method.

### Weaknesses
1. The paper presents a meaningful application, but the novelty is relatively limited. The core insight that the correlation between input activations and the weight difference directions introduced by fine-tuning has already been discussed in prior work. The contribution here mainly lies in applying this insight to behavior detection.


2. The implementation details of the proposed detection method are not clearly described. It is unclear which layers are examined during detection, and whether the method reports detection when any layer exceeds the similarity threshold or only when all do. It is also not clarified whether there exists a general principle across different tasks for layer selection.Additionally, it is unclear how the similarity threshold is chosen. 

3. The proposed method requires a normal set to establish the expected activation range for benign behavior. However, in practice, it is difficult to guarantee that such a normal range is representative, since user inputs can be arbitrarily diverse. Although the authors use a calibration set and MMLU to demonstrate a low false positive rate, this evidence is not fully convincing. Additional evaluations on more diverse normal tasks could help mitigate this concern and strengthen the claim that the method maintains low false positives in realistic open-ended usage.


4. In the backdoor detection setting, it is unclear what the false positive rate is for harmful questions that do not contain the trigger.


5. In Section 5, I was not able to clearly identify which base models were used for the fine-tuning in the experimental setup.

### Questions
See Weaknesses

### Soundness
2

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
2

### Summary
This paper introduces WEIGHTWATCH, a weight-based interpretability and monitoring framework for fine-tuned large language models (LLMs). Unlike prior activation-based interpretability or anomaly-detection methods that rely on access to representative datasets, this approach analyzes the top singular vectors of the weight difference between a fine-tuned model and its base model. The authors argue that these vectors encode fine-tuning–induced behavioral shifts, allowing for data-free detection, monitoring, and steering of fine-tuned or malicious behaviors such as backdoors and unlearning artifacts.

Empirical evaluations across backdoored, unlearned, and commercial instruction-tuned models (e.g., Qwen, Llama, OLMo) demonstrate impressive detection precision (up to 100% backdoor detection, <1.2% FPR) and steering ability (e.g., recovering “unlearned” information from RMU models). The paper concludes that weight-space analysis can serve as a scalable, unsupervised auditing tool for open-weight models.

### Strengths
- Shifting interpretability from activation space to weight space is impactful. It tackles the practical limitation that fine-tuning datasets are often unavailable. This framing makes WEIGHTWATCH a scalable tool for open-weight model auditing.
- The experiments are broad and thorough, covering multiple applications: (1) detecting and mitigating backdoors, (2) verifying and even recovering unlearned knowledge, and (3) auditing open-weight models (OLMo, LLaMA, Qwen).

### Weaknesses
- A major limitation of this approach is its assumption of paired access to both the base and the fine-tuned model weights. While this condition holds for many open-weight ecosystems (e.g., Llama, Qwen), it is often violated in real-world scenarios where the base model is obfuscated. In such cases, WEIGHTWATCH cannot construct meaningful ΔW matrices, as the singular vectors derived from raw weights would lose semantic correspondence across unaligned hidden dimensions.
- While coverage across backdoor/unlearning models is broad, most reported metrics lack statistical confidence intervals or standard deviations. Additionally, the “in-the-wild” auditing remains largely qualitative. Quantitative correlations between discovered directions and fine-tuning datasets would significantly strengthen claims.

### Questions
- Can the proposed approach be applied to quantized or partially fine-tuned models (e.g., adapters or LoRAs)?

- Could combining WEIGHTWATCH with activation-based monitoring yield complementary benefits?

### Soundness
3

### Presentation
3

### Contribution
3
