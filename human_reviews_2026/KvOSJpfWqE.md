# PRISON: Unmasking the Criminal Potential of Large Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
As large language models (LLMs) advance, concerns about their misconduct in complex social contexts intensify. Existing research has overlooked the systematic assessment of LLMs’ criminal potential in realistic interactions, where criminal potential is defined as the risk of producing harmful behaviors such as deception and blame-shifting under adversarial settings that could facilitate unlawful activities. Therefore, we propose a unified framework PRISON, to quantify LLMs' criminal potential across five traits: False Statements, Frame-Up, Psychological Manipulation, Emotional Disguise, and Moral Disengagement. Using structured crime scenarios grounded in reality, we evaluate both criminal potential and anti-crime ability of LLMs. Results show that state-of-the-art LLMs frequently exhibit emergent criminal tendencies, such as proposing misleading statements or evasion tactics, even without explicit instructions. Moreover, when placed in a detective role, models recognize deceptive behavior with only 44\% accuracy on average, revealing a striking mismatch between expressing and detecting criminal traits. These findings underscore the urgent need for adversarial robustness, behavioral alignment, and safety mechanisms before broader LLM deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces PRISON, a novel framework for assessing the capabilities of LLMs to a) develop criminal strategies and b) detect criminal acts. Building upon plots and situations from mainstream movies, the framework creates settings in which an LLM is queried to solve the situation with malicious intent, e.g., covering up an accident. The first aspect of the study explores whether LLMs can come up with valid, illegal/harmful strategies given the individual setting. The second aspect takes the opposite view and tests whether the same LLMs can detect illegal actions in the generated strategies. Evaluating multiple recent LLMs, the paper demonstrates that language models show high criminal capabilities while their detection capabilities for such actions are limited.

### Strengths
- The paper is very well-written and easy to follow. All figures and findings are clean and straightforward to understand. Overall, the paper formatting quality is above average.
- Investigating the criminal potential of LLMs is an interesting avenue, and leveraging the scripts of movies to create an evaluation framework is a smart idea. The findings that there exists a mismatch between criminal actions and criminal detection are intriguing. I also like that the paper not only distinguishes between criminal/non-criminal but also analyzes the different kinds of criminal traits.
- The experimental evaluation covers multiple LLMs (8 different models) and settings. Whereas some recent models, e.g., GPT-5, Qwen-3, DeepSeek-R1, are missing, the provided models offer a good mix of non-reasoning LLMs.

### Weaknesses
- The framework setting feels somewhat artificial. While I understand the intention behind the dataset, I am not fully convinced that the evaluations genuinely assess a model’s criminal capabilities. When looking at examples in the Appendix (e.g., Table 5), it often feels as if the model is writing a novel. On one hand, such narrative-style outputs could indeed be misused for criminal purposes. However, I am not sure whether these outputs are actually harmful, since it remains unclear to what extent the proposed strategies go beyond straightforward ideas or common scenes from movies. While, in another context, detailed instructions for building a bomb could clearly cause harm, suggesting to push a car into a lake (which requires no expert knowledge) seems more like repeating a movie or book scenario. I understand that we want LLMs to avoid producing such suggestions, but given that the model appears to be engaging in creative writing, this behavior might be acceptable in some cases.
- The crime detection capabilities of LLMs are not compared against a human baseline. Since LLMs have less information than the “God-setting,” a human baseline would help quantify the actual performance gap. Given that some contextual information is missing from the detection model’s input, it might be that certain actions cannot be reliably classified as criminal.
- As the framework is based on only ten movies, the diversity of scenarios may be limited.
- No large reasoning models, such as Qwen3 or DeepSeek-R1, are evaluated. It would be interesting to see whether stronger reasoning capabilities improve a model’s ability to either generate or detect criminal content.

Small remark:
- There is a missing space in L046.

### Questions
- How does a human baseline perform on the detection task compared to the LLMs? Is there sufficient information contained in the inputs actually to solve the task (could be answered by a user study)?

### Soundness
3

### Presentation
4

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
This paper introduces PRISON, a novel evaluation framework designed to assess the "criminal potential" of Large Language Models (LLMs) in complex, multi-turn social interactions. The authors define criminal potential as the risk of an LLM generating harmful behaviors like deception, manipulation, or blame-shifting in adversarial contexts that could facilitate unlawful activities. The paper's main contributions are (1) the PRISON framework itself, as a new benchmark for a critical and understudied safety dimension , (2) a quantification of LLMs' "criminal potential" , and (3) the identification of the significant gap between an LLM's ability to generate and its ability to detect such behaviors. These contributions are timely, novel, and significant for the AI safety community

### Strengths
1. This work moves beyond traditional, static safety evaluations (e.g., simple harmful Q&A, abstract moral dilemmas) to tackle the much more complex and realistic threat of LLMs participating in deceptive, multi-turn social interactions. The "criminal potential" concept is a valuable and well-defined framing of a risk that is highly relevant as LLMs are integrated into agentic systems. This paper addresses a clear and important gap in the current safety literature.

2. The PRISON framework is thoughtfully constructed. Grounding the five-trait taxonomy in established psychometric instruments from criminal psychology  provides a strong theoretical foundation that is often lacking in other safety benchmarks. Furthermore, the tri-perspective (Criminal, Detective, God) evaluation design is an intelligent and effective method for simultaneously measuring the expression of harmful traits and the detection of them within a unified system

### Weaknesses
1. The 44% "Objective Trait Detection Accuracy" (OTDA)  is a headline-grabbing result. However, its significance is difficult to interpret without more details on the "Detective" agent's task.

2. Regarding the "God" perspective validation: A Cohen's Kappa of 0.65 is "substantial" but not "near perfect." Could you provide a qualitative breakdown of the disagreements between your human annotators and the GPT-4o judge? Are there specific traits (e.g., "Psychological Manipulation" vs. "False Statements") that are more ambiguously defined or harder for the LLM to judge correctly?

3. The scenario generation from films is a clever way to source complex social dynamics. However, film narratives are inherently dramatic and conflict-driven. How do you account for the potential domain mismatch between these "dramatized" scenarios and more mundane, real-world criminal interactions? Is it possible this choice of data source biases the "Criminal Traits Activation Rate" (CTAR) upwards?

### Questions
See above

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
The paper introduces PRISON, a tri-perspective evaluation framework designed to assess both the criminal potential and detection capability of LLMs in adversarial social scenarios. It models how LLMs behave under roles such as criminals, detectors, and gods, simulating information flow and perspective differences to capture both understanding and detection of illegal behaviors.
The study finds interesting observations. For example, popular LLMs often exhibit criminal traits: generating deceptive or harmful advice, even without explicit malicious prompts. However, they perform poorly when detecting similar behavior.
The evaluation is extensive, which leverages context-rich, film-inspired scenarios to ensure realism while maintaining ethical control.

### Strengths
- The proposed framework is novel and studies an important aspect of LLM safety.
- The tri-perspective approach (criminal, detector, god) is innovative and captures the complexity of adversarial scenarios effectively.
- Comprehensively quantifies the criminal tendencies of various LLMs, providing valuable insights into their capabilities and limitations.

### Weaknesses
- The performance gap between criminal generation and detection may be the nature of the task itself, rather than a specific shortcoming of LLMs.
- The scenarios are primarily adapted from classic crime films, which may limit representativeness of real-world criminal contexts.
- Lack of technical discussion about why certain behaviors emerge in LLMs.

### Questions
Overall, this is a well-structured and insightful study that contributes meaningfully to our understanding of LLM safety in adversarial contexts. The PRISON framework is a valuable addition, offering a creative way to stress-test models' tendencies toward criminal expression and their ability to detect manipulative behavior. However, I have a few concerns:

---
(1) Nature of the Performance Gap:

The observed gap between ”criminal expression“ and “detection” might reflect the nature of the task itself rather than a true model deficiency. For humans as well, generating deception is often easier than detecting it, since detection requires background knowledge and reasoning about intent. It would strengthen the paper if the authors could further analyze whether this gap truly indicates a model limitation or simply the inherent difficulty of the task.

---
(2) Use of Film-Based Scenarios:

Lines 220–221 mention that the scenarios are adapted from films. However, film plots are not necessarily realistic representations of real-world criminal behavior. Why not use more authentic materials such as court transcripts, online forums, or real-world investigative documents to improve realism and ecological validity?

---
(3) Lack of Technical Analysis:

The performance gap essentially reflects two underlying technical issues:
- insufficient safety alignment, since the model still tends to follow harmful or deceptive instructions; and
- limited long-context understanding, as detecting criminal or deceptive behavior often requires reasoning over extended context.
It would be helpful if the authors could analyze these aspects more deeply to clarify the technical reasons behind the observed gap.

### Soundness
3

### Presentation
3

### Contribution
3
