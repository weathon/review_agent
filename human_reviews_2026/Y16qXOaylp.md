# Beyond In-Domain Detection: SpikeScore for Cross-Domain Hallucination Detection

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 8

## Abstract
Hallucination detection is critical for deploying large language models (LLMs) in real-world applications. Existing hallucination detection methods achieve strong performance when the training and test data come from the same domain, but they suffer from poor cross-domain generalization. In this paper, we study an important yet overlooked problem, termed generalizable hallucination detection (GHD), which aims to train hallucination detectors on data from a single domain while ensuring robust performance across diverse related domains. In studying GHD, we simulate multi-turn dialogues following LLMs' initial response and observe an interesting phenomenon: hallucination-initiated multi-turn dialogues universally exhibit larger uncertainty fluctuations than factual ones across different domains. Based on the phenomenon, we propose a new score SpikeScore, which quantifies abrupt fluctuations in multi-turn dialogues. Through both theoretical analysis and empirical validation, we demonstrate that SpikeScore achieves strong cross-domain separability between hallucinated and non-hallucinated responses. Experiments across multiple LLMs and benchmarks demonstrate that the SpikeScore-based detection method outperforms representative baselines in cross-domain generalization and surpasses advanced generalization-oriented methods, verifying the effectiveness of our method in cross-domain hallucination detection.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper tackles the cross-domain hallucination detection problem in large language models (LLMs). While existing detectors perform well on in-domain data, they struggle to generalize to unseen domains. The authors formalize this gap as a new task, Generalizable Hallucination Detection (GHD) — training a detector on a single domain while ensuring robust performance across diverse domains.
The paper further introduces SpikeScore, a simple, interpretable, and transferable curvature-based indicator that quantifies abrupt fluctuations in model uncertainty across multi-turn dialogues. By capturing the spike-like instability patterns that typically occur when a hallucinated response triggers self-contradictory corrections, SpikeScore serves as a domain-invariant signal for hallucination detection.

### Strengths
1) Simplicity, interpretability, and transferability. SpikeScore is a remarkably simple yet effective approach that requires only minimal dialogue simulation. It does not rely on domain-specific features and comes with a theoretical guarantee of cross-domain stability, making it both practical and generalizable.
2) Intuitive connection to model behavior. The use of “curvature of uncertainty fluctuations” provides an interpretable signal that directly reflects a model’s self-contradiction and confidence reversals, offering clear insight into the dynamics of hallucination emergence.

### Weaknesses
While the paper presents an interesting perspective and a well-executed framework, there are two major weaknesses that limit its overall strength.
1) First, the theoretical and experimental analysis lacks sufficient depth. The paper approaches hallucination detection from a multi-turn dialogue viewpoint, yet multi-turn consistency itself is already a predefined training task in most post-training and RL pipelines. Modern LLMs are typically reinforced to remain consistent across turns—for example, (a) maintaining the same correct stance rather than conforming to user bias, or (b) refining vague answers through discussion. Such strategies are widely studied in prior research and extensively practiced in industry, often involving over a hundred dialogue turns, far beyond the 20-turn setup used in this paper. As a result, SpikeScore may primarily capture the degree of multi-turn consistency training embedded in each model rather than providing an independent signal of hallucination robustness. This makes it difficult to disentangle whether the observed cross-domain separability arises from the proposed metric itself or from model-specific post-training characteristics.
2) Second, the proposed approach relies heavily on simulated dialogue continuations and direct model generations, which limits its practicality. The method assumes open-ended, free-form interactions, making it less applicable to structured or tool-augmented settings such as function calling, retrieval-augmented generation (RAG), or other pipeline-based workflows where dialogue continuity is not the central mechanism. Consequently, the real-world usability of SpikeScore beyond conversational LLM scenarios remains uncertain.

### Questions
1) Referring to the issue raised in the Weaknesses section, how is GHD related to the consistency-enhancement tasks commonly introduced during post-training or RL of LLMs?
2) How could experiments or theoretical analyses be designed to further verify and support this viewpoint in Q1?

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
This paper introduces SpikeScore, a simple yet effective metric to detect hallucinations in LLMs under cross-domain settings. The key observation is that hallucinated answers, when probed in multi-turn self-dialog environments, exhibit sharp uncertainty spikes, whereas truthful responses show smoother trajectories. The authors formalize this phenomenon, propose SpikeScore as the maximum second-order difference in a scoring trajectory, provide theoretical justification through probabilistic separability bounds, and perform extensive cross-domain evaluations across six datasets and four open-source LLMs. Results show consistent improvements over both training-free and training-based baselines, and over cross-domain oriented methods like PRISM and ICR Probe.

### Strengths
1. Tackles a realistic and impactful problem setting (generalizable hallucination detection).
2. Novel insight that hallucination triggers instability in multi-turn self-dialogue.
3. Simple but powerful metric (SpikeScore) with no need for additional finetuning.
4. Strong cross-domain results vs powerful baselines (PRISM, ICR probe).

### Weaknesses
1. Theory relies on assumptions that may not always hold in real LLM behavior.
2. No evaluation on more complex generative formats (e.g., long-form reasoning beyond QA).
3. Domain choice mainly QA/knowledge tasks; extension to code or vision-language tasks not discussed.

### Questions
1. How robust is SpikeScore under adversarial or “polite-aligned” prompting that suppresses contradictions?
2. Would combining SpikeScore with chain-of-thought introspection further improve results?
3. Can the approach be extended to detect hallucination in long-form generation or agentic workflows?
4. Any evidence whether multilingual or multimodal models show similar spike patterns?

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
3

### Summary
The paper proposes a method for hallucination detection based on SpikeScore. The score is based on the observation that when LLM hallucinates, contradictions are more often observed in a dialog. SpikeScore is designed to capture the local fluctuation of probability to generate a few consecutive answers in the dialog. 

Experiments are done on several public datasets, comparing the SpikeScore method with several training-free, training-based and cross-domain methods. It is shown that SpikeScore outperforms all of them.

### Strengths
1. The main idea is based on the observation that in case of hallucination, LLM will produce inconsistent answers in a dialog. The proposed idea is interesting and new. It extends the Consistency-based methods by generating multiple answers in a dialog sequence. This extension is sound.

2. The paper provides a strong motivation from concrete observations.

3. The paper contains theoretical analyses on the method.

4. The experiments show clearly the advantage of SpikeScore.

### Weaknesses
1. The experimental results may be presented more clearly. Tables 1 and 2 that describe the main results are quite confusing. The two parts of each table are not explained. It is also unclear how leave-one-out is done for training-based methods. It is said that "training-based methods train on each dataset (columns) while all methods are evaluated on the remaining five datasets". So what is the Mean AURA in SEP under TriviaQA/Llama 3.2-3B? Is this the mean AURA tested on 5 other datasets using the model trained on TriviaQA? In that case, the number is not directly comparable to other training-free methods in the same column, which should have been tested on TriviaQA.

2. To generate dialogs, a set of prompts are designed and organized in several types. It is unclear how they are selected to generate a dialog. Are they randomly chosen? Do they follow some fixed patterns? One would believe that the selection of prompts may influence the SpikeScore. Have you observed this phenomenon? How do you deal with such a variability due to the selection of prompts?

### Questions
See the questions in Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper finds that in multi-turn conversations across different domains, subsequent responses triggered by hallucinations typically exhibit greater uncertainty fluctuations compared to factual responses. Based on this observation, this paper proposes a new metric called SpikeScore which quantifies fluctuations through maximum second-order differences. Experiments on multiple datasets demonstrate the effectiveness of the proposed SpikeScore in distinguishing between hallucinated and non-hallucinated responses across domains, and theoretical analysis further provides theoretical guarantees for its generalization capability.

### Strengths
1. This paper observes that in multi-turn conversations triggered by hallucinations, LLMs frequently engage in self-correction. By employing the maximum second-order difference to measure local fluctuations, this paper provides an effective and novel perspective for cross-domain hallucination detection.
2. Extensive cross-domain experiments have demonstrated the superiority of the proposed SpikeScore. Theoretical analysis additionally provides validation for its effectiveness in distinguishing between hallucinated and factual responses.

### Weaknesses
The claim of cross-domain generalization is not sufficiently supported. Although there are variations in tasks, such as question answering (CommonsenseQA, TriviaQA), reading comprehension (Belebele, CoQA), and mathematical reasoning (Math, SVAMP), the linguistic styles across these datasets may exhibit similarities. More cross-domain scenarios  are expected for evaluation.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
