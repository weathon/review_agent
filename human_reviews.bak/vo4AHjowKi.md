# Training-free LLM-generated Text Detection by Mining Token Probability Sequences

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Large language models (LLMs) have demonstrated remarkable capabilities in generating high-quality texts across diverse domains. However, the potential misuse of LLMs has raised significant concerns, underscoring the urgent need for reliable detection of LLM-generated texts. Conventional training-based detectors often struggle with generalization, particularly in cross-domain and cross-model scenarios. In contrast, training-free methods, which focus on inherent discrepancies through carefully designed statistical features, offer improved generalization and interpretability. Despite this, existing training-free detection methods typically rely on global text sequence statistics, neglecting the modeling of local discriminative features, thereby limiting their detection efficacy. In this work, we introduce a novel training-free detector, termed \textbf{Lastde}\footnote{The code and data are released at \url{https://github.com/TrustMedia-zju/Lastde_Detector}.} that synergizes local and global statistics for enhanced detection. For the first time, we introduce time series analysis to LLM-generated text detection, capturing the temporal dynamics of token probability sequences. By integrating these local statistics with global ones, our detector reveals significant disparities between human and LLM-generated texts. We also propose an efficient alternative, \textbf{Lastde++} to enable real-time detection. Extensive experiments on six datasets involving cross-domain, cross-model, and cross-lingual detection scenarios, under both white-box and black-box settings, demonstrated that our method consistently achieves state-of-the-art performance. Furthermore, our approach exhibits greater robustness against paraphrasing attacks compared to existing baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents an AIGC detection method named Lastde. It is training free and can used both local and global statistics for
AIGC detection. An more efficient version named Lastde++ is also proposed for real-time detection.

### Strengths
1. This paper presents an effective AIGC detection method, which can be very useful in the GenAI era.

2. The proposed method in sound and does not rely on training.

3. The experiments show promising results.

### Weaknesses
1. It would be better if the paper writing can be improved. For example, figure 2 is not so easy to understand.

2. My biggest concern is that strong baselines are not included in experiments, such as GPTZero.

3. Can the proposed method work well in different domains?

4. Is the proposed method robust to some adaptive attacks like asking LLMs to mimic the human writing?

### Questions
1. It would be better if the paper writing can be improved. For example, figure 2 is not so easy to understand.

2. My biggest concern is that strong baselines are not included in experiments, such as GPTZero.

3. Can the proposed method work well in different domains?

4. Is the proposed method robust to some adaptive attacks like asking LLMs to mimic the human writing?

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
4

### Summary
The paper introduces a training-free method called Lastde for detecting LLM-generated text by analyzing TPS through a blend of local and global statistical features. Lastde incorporates temporal dynamics in TPS with diversity entropy to capture subtle distinctions between human and AI-generated text. An enhanced version, Lastde++, offers faster, real-time detection and outperforms existing methods across various scenarios. Extensive experiments demonstrate that Lastde++ provides superior robustness against paraphrasing attacks and cross-lingual challenges, establishing it as a powerful benchmark in LLM-generated text detection​.

### Strengths
- The paper attempts to solve an important LLM-generated text detection problem without further training 

- The proposed method is straightforward and intuitive, and the experiments presented in the paper are comprehensive with solid results.

- The readers can easily understand the proposed method and follow the content of the paper.

### Weaknesses
- I recommend including more evaluations on separate sets of LLM-generated text and human-written text ( i.e., evaluating the method on sets containing only LLM-generated text and only human-written text). This would provide valuable insights into how the proposed method works and whether the detection method performs better on LLM-generated text, or human-written text, or both.

- Paraphrasing attacks pose a significant threat in the context of LLM-generated text detection. It is highly recommended to use a more powerful LLM paraphraser, which could better highlight the proposed method’s ability to protect against such attacks.

- How did you predetermine the detection threshold, especially when the method is training-free and there is no prior knowledge about the LLM-generated text for the detector? The paper does not discuss the details of how the threshold was chosen but only provides a specific value used in a certain scenario. The threshold is a key component that leads to good detection performance no matter how good the proposed scores are extracted.

- Generally speaking, when we attempt to detect whether a given text is generated by LLMs, we usually do not know which specific models malicious users have employed to produce the text. It would be interesting to see a further experiment conducted on a mixed set of LLM-generated text created from various sources LLMs. The evaluation of detection abilities in the above experimental settings more closely matches real-world scenarios.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a training-free detector for LLM-generated text by analyzing token probability sequences. The proposed Lastde method combines both global and local statistics, using diversity entropy to capture temporal dynamics, thus achieving improved cross-domain detection and robustness against paraphrasing attacks. Evaluations on 6 datasets show that Lastde outperforms existing detectors.

### Strengths
- The method is highly effective across cross-domain and cross-model scenarios without needing retraining.
- Lastde++ enables real-time detection with minimal computational cost, outperforming many established methods.
- Exhibits strong resistance to paraphrasing attacks, maintaining accuracy in varied textual manipulations.

### Weaknesses
- The detection accuracy depends significantly on the chosen proxy model, especially in black-box settings, affecting usability across model types.
- The choice of proxy model in black-box scenarios is still unclear, and the use of GPT-j need more justification.
- Performance drops with shorter texts, which limits applicability in scenarios like social media or short Q&A responses.

### Questions
How does Lastde handle highly stylized human-written text that might mimic AI-generated patterns?
Can Lastde be adapted or optimized for human-LLM co-authored text (e.g., LLM-revised text)?
Would a hybrid approach combining Lastde with a lightweight training-based model further enhance detection accuracy?

### Soundness
2

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
5

### Summary
This paper introduces a novel training-free detector, termed Lastde that synergizes local and global statistics for enhanced detection. They introduce time series analysis to LLM-generated text detection, capturing the temporal dynamics of token probability
sequences. By integrating these local statistics with global ones, their detector reveals significant disparities between human and LLM-generated texts. They also propose an efficient alternative, Lastde++ to enable real-time detection. Extensive experiments on six datasets involving cross-domain, cross-model, and cross-lingual detection scenarios, under both white-box and black-box settings, demonstrated
that their method consistently achieves state-of-the-art performance. Furthermore, their approach exhibits greater robustness against paraphrasing attacks compared to existing baseline methods.

### Strengths
originality: treating text detection as time series analysis is novel and the approach seems promising, though previous work SeqXGPT already did this. 

quality: I like the solid experiments from this paper since it compared with many previous baselines on a bunch of models and datasets. From all the experimental results, we can see the great potential of this method. 

clarity: This paper is well-written and easy to follow. The tables and figures and clear.

significance: This provides a new method for detection and potentially useful for detection.

### Weaknesses
AUROC is not a very good measure for real-world use because FPR in real life is very important for this task, especially for students. I would like to know the TPR rate at 1% FPR.

In my opinion, treating the detection as a time series analysis is not entirely new. For example, your cited another work SeqXGPT also treats the token logits as waves, though your approach is different. Also, there is no comparison with SeqXGPT.

There are many hyperparameters in this algorithm to tune, weakening its potential practical usage.

Many tables rely on results on very small models like GPT-2 Neo-2.7 OPT-2.7, which makes it less significant.

### Questions
1. I have one question. I understand that this is a training-free method. But it seems it includes various ingenious designs such as multiscale log-probability sequence, sliding-segmentation and diversity entropy and etc. And I would assume this can not be designed at the first glance. So thinking from the first principle, given the token logits, if the token logtis can be a good indicator through another function f(), then what's the best f()? perhaps we can directly a network to find the f()?

2. It is unclear to me whether the used models like llama are the base model or chat model because instruction-tuned models are more difficult to detect and base models are easy.

3. Do you need to know the prompt and question used for generating the text for detection? in reality, we do not know this.

### Soundness
3

### Presentation
4

### Contribution
3
