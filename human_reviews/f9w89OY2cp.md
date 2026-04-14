# Amulet: ReAlignment During Test Time for Personalized Preference Adaptation of LLMs

- Decision: Accept (Poster)
- Scores: 6, 6, 5, 6

## Abstract
How to align large language models (LLMs) with user preferences from a static general dataset has been frequently studied. However, user preferences are usually personalized, changing, and diverse. This leads to the problem that the actual user preferences often do not coincide with those trained by the model developers in the practical use of LLMs. Since we cannot collect enough data and retrain for every demand, researching efficient real-time preference adaptation methods based on the backbone LLMs during test time is important. To this end, we introduce **Amulet**, a novel, training-free framework that formulates the decoding process of every token as a separate online learning problem with the guidance of simple user-provided prompts, thus enabling real-time optimization to satisfy users' personalized preferences. To reduce the computational cost brought by this optimization process for each token, we additionally provide a closed-form solution for each iteration step of the optimization process, thereby reducing the computational time cost to a negligible level. The detailed experimental results demonstrate that Amulet can achieve significant performance improvements in rich settings with combinations of different LLMs, datasets, and user preferences, while maintaining acceptable computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new framework, Amulet, for single-user preference alignment at test time. Amulet treats each token generation step as an independent online learning problem and achieves real-time, personalized preference alignment without requiring model retraining or fine-tuning by introducing a closed-form solution. Experimental results demonstrate Amulet’s strong alignment performance and efficiency across various combinations of LLMs, datasets, and preference dimensions.

### Strengths
Amulet offers an innovative test-time preference alignment method, enabling real-time adaptation to user preferences during generation without fine-tuning and thereby reducing resource costs.

The closed-form solution design significantly enhances computational efficiency, where detailed theoretical analysis is provided.

Validation across different LLM and dataset combinations shows Amulet’s robustness and broad applicability in multiple scenarios.

### Weaknesses
The paper only compares with Pref and Linear Alignment (LA), lacking comparisons with other mainstream alignment methods. To improve credibility, it’s recommended to introduce more alignment methods as baselines on at least one foundation model.

Amulet assumes sufficient user history for preference alignment, but its performance with insufficient user history is not discussed, impacting the method’s practicality.

### Questions
How does Amulet perform if user history information is insufficient or lacking?

How does Amulet perform while compared with more SOTA baselines?

### Soundness
4

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
This paper investagates the issue that current LLM alignment methods focus on static datasets rather than personalized, changing user preferences. Besides, the authors recognize that retraining models for each new preference is costly and impractical. Therefore, the authors introduce a training-free framework that allows real-time optimization of LLM outputs to match user preferences during test time, namely Amulet, which formulates each token generation as a separate online learning problem guided by user prompts. Experiments demonstrate notable performance improvements across different LLMs, datasets and preference types.

### Strengths
The paper is well-organized. It is reasonable for me to perform test time alignment in LLMs for light-weight preference optimization.

The proposed online learning decoding process by formulating each token generation as a separate online learning problem seems novel. Besides, the authors provide a closed-form solution to reduce computational costs

Experiments with several datasets and backbone LLMs demonstrate the effectiveness of the proposed method.

### Weaknesses
Insufficient baselines. Only LA is used as the baseline model, likely due to the scarcity of tuning-free test time alignment approaches. However, other baselines from related topics, such as those introduced in the related work, could be adapted to verify the effectiveness.

One major merit of tuning-free test time alignment is light-weighted. To this end, the time and computational complexity could be analyzed. The computational cost could be reported.

Broken sentences, such as Line 209: 'Since we provide a general framework that is unrelated to the utility.'

### Questions
See the above Strengths and Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a framework called AMULET for personalization alignment at test time. It borrows ideas from Contrastive Decoding and FTRL. During inference, it obtains two token generation probability distributions based on a base prompt and a user-specific personalized prompt, treating each step of token generation as an independent online optimization problem. The authors model this optimization problem and provide a closed-form solution to balance personalization and performance.

### Strengths
Strengths:
1. The general idea is a hot topic in the field of LLMs (personalization alignment and test-time personalization alignment).
2. The idea of treating each step of token generation as an independent optimization problem to solve test-time alignment is an interesting exploration.
3. The paper is well-written, and the methods and formulas are easy to understand.

### Weaknesses
Weaknesses:
1. The experimental setup is not very clear. How did the authors tune the hyperparameters? See Questions below.
2. Based on my understanding, the method described in the paper ultimately results in a weighted of the base prompt generation probability, user-specific prompt generation probability. Therefore, the authors should provide a baseline that only adjusts the fine-tuned $\alpha$, which essentially becomes Contrastive Decoding.
3. The authors emphasize the importance of real-time preference alignment but still conduct tests on statistical datasets, which cannot fully validate their claim. I recommend the authors supplement experiments on streaming data (such as recommender systems).

### Questions
1. Although the reward model used by the author showed excellent performance on the benchmark, the results presented in the paper indicate that the llama series of models experienced the most significant performance improvement. I suppose this might be related to the choice of the reward model. So, are there results using 4o to evaluate Figure 2 and Figure 5?
2. Experimental setting: Is T fixed in the experiments, or is early stopping based on some metrics? What is the tuning range for other parameters, and what hyperparameters were ultimately selected? Additionally, if the authors directly tune hyperparameters on the test set, it could lead to data leakage.
3. How are the user-specific preference prompts for each dataset constructed?
4. Has the author tried other model sizes, either smaller or larger models?
5. The datasets selected by the author, aside from Personal Preference Eval, do not seem to have strong requirements for personalization. Could the author provide a more detailed explanation for the selection of the remaining three datasets, with a focus on their relevance to personalization?

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
3

### Summary
This work presents Amulet, a new framework to adapt language models to individual user preferences at test time. It treats the decoding of each token as a separate online learning problem, guided by both a base prompt and a preference prompt. It also provides a closed-form solution for each iteration step of the optimization process, making the method pretty computational efficient. Experimental results demonstrate the superiority of Amulet across various combinations of LLMs, datasets, and user preferences.

### Strengths
S1: Test-time realignment for personalized preferences is an interesting research topic in the field of LLMs.

S2: The proposed framework approaches the decoding of each token as an independent online learning problem and introduces a closed-form solution for optimization, which is novel to me.

S3: The paper is well-written, and the experimental results seems good.

### Weaknesses
W1: Insufficient baselines. Only Pref and LA are included for comparison. More alignment approaches are needed. Additionally, the Pref baseline appears trivial in its implementation. Given that the studied preference dimensions in this work are easy to define, it can be effective to use more sophisticated prompt engineering approaches that could serve as stronger baselines, e.g., emphasizing the output format.

W2: Lack of Evaluation on Implicit Preferences. While the paper demonstrates effectiveness on eight explicit preference dimensions (creative, verbose, concise, etc.), it is unknown whether the proposed method can solve the scenarios with implicit user preference.

W3: Dataset Relevance. Three out of four used datasets (HelpSteer, Truthful QA, and UltraChat) appear to have limited relevance to the task of aligning with user preferences.

### Questions
See the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
