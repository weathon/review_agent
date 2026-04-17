# Closing the Gap Between Text and Speech Understanding in LLMs

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Large Language Models (LLMs) can be adapted to extend their text capabilities to speech inputs. However, these speech-adapted LLMs consistently underperform their text-based counterparts—and even cascaded pipelines—on language understanding tasks. We term this shortfall the text–speech understanding gap: the performance drop observed when a speech-adapted LLM processes spoken inputs relative to when the original text-based LLM processes the equivalent text. Recent approaches to narrowing this gap either rely on large-scale speech synthesis of text corpora, which is costly and heavily dependent on synthetic data, or on large-scale proprietary speech datasets, which are not reproducible. As a result, there remains a need for more data-efficient alternatives for closing the text-speech understanding gap. In this work, we analyze the gap as driven by two factors: (i) forgetting of text capabilities during adaptation, and (ii) cross-modal misalignment between speech and text. Based on this analysis, we introduce SALAD—Sample-efficient Alignment with Learning through Active selection and cross-modal Distillation—which combines cross-modal distillation with targeted synthetic data to improve alignment while mitigating forgetting. Applied to 3B and 7B LLMs, SALAD achieves competitive performance with a strong open-weight model across broad-domain benchmarks in knowledge, language understanding, and reasoning, while training on over an order of magnitude less speech data from publicly available corpora.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the text-speech understanding gap—the performance drop when speech-adapted LLMs process spoken inputs versus text inputs to text-based LLMs. 
The authors decompose this gap into two factors: (i) catastrophic forgetting of text capabilities during speech adaptation, and (ii) cross-modal misalignment between speech and text representations.
Based on this analysis, they propose SALAD (Sample-efficient Alignment with Learning through Active selection and cross-modal Distillation), which combines cross-modal knowledge distillation with active data selection to improve alignment while mitigating forgetting. Applied to Qwen2.5 3B and 7B models, SALAD achieves competitive performance with recent speech-adapted LLMs while reportedly using much less training data.

### Strengths
1. Well-motivated problem decomposition: The two-factor analysis separating forgetting from cross-modal misalignment is conceptually clear and provides a useful framework. The mathematical formalization using KL divergence (Equations 2-3) enables quantitative measurement of both factors.
2. Rigorous empirical analysis (Section 3): The systematic study of how different training objectives (α parameter) affect forgetting and misalignment is valuable. The scaling law analysis with fitted curves (Table 2) and cross-validation provides insights into training dynamics.
3. Clear presentation and organization: The paper is well-written with logical flow from problem analysis to solution. Mathematical formulations are precise and figures effectively communicate key results.

### Weaknesses
1. Missing validation of core motivation: The paper motivates end-to-end approaches over cascaded systems by citing ability to capture "paralinguistic richness essential for natural spoken interaction" (Introduction). Yet no experiments evaluate paralinguistic understanding (emotion, prosody, speaker characteristics). The distillation objective enforcing identical text-speech distributions may actually suppress these cues, contradicting the motivation. This is very important because if content is the only thing that you want to model, then you can simply do cascaded system and in fact ASR+LLM backbone (Qwen2.5) is still significantly better as shown in Table 3.
2. Limited experimental scope: Only multiple-choice QA tasks—no open-ended generation. Would be curious to see if findings still hold in open-ended generation.
3. Entirely synthetic evaluation undermines validity: All benchmarks evaluate TTS-generated speech rather than natural speech. This raises questions about whether improvements generalize to real speech with acoustic variability, accents, and spontaneous characteristics. Table 8's limited test on VoiceBench shows performance degradation with different TTS speakers, but more analysis is needed.
4. Active selection assumes high-misalignment clusters represent domain gaps, but they could equally represent intrinsically difficult content or TTS artifacts. Some discussion around this would be helpful.

### Questions
Check weaknesses.

### Soundness
2

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
4

### Summary
This paper addresses the persistent performance gap between large language models (LLMs) adapted for speech input and their original text-based counterparts. While LLMs excel at text-based language understanding, adapting them to process speech directly results in a notable drop in performance - which is termed as "text - speech understanding gap".  The key contributions of this paper include:
1. it analyzes the text–speech understanding gap in a quantifiable way and diagnoses 2 main factors that may cause this gap: forgetting and cross-modal misalignment.
2. it proposes a 2-stage training strategy called SALAD that first does cross-modal distillation on natural speech data to improve alignment and mitigate forgetting, followed by an active synthetic speech sample selection to address domain coverage issue.

Empirical results show that SALAD achieves competitive performance compared to strong open-source speech-adapted LLMs, with much less speech data.

### Strengths
Originality:
This paper formalize the common phenomenon in speech-adapted LLMs where the LLMs exhibits significant gap between text and speech understanding. The paper also proposes a quantifiable metric to measure this phenomenon and provides statistical measure for both forgetting and cross-modal misalignment.

Quality:
1. The paper presents a thorough empirical evaluation, benchmarking SALAD against a wide range of open-source speech-adapted LLMs.
2. The analysis appears to be rigorous, accompanied with clear quantification of the factors contributing to the "text - speech understanding gap". 

Clarity:
The paper is well organized and easy to follow. 

Significance:
1. This work attempts to address a challenge when training a speech-LLM: it requires massive amount of speech data to achieve on-par speech understanding capabilities which is often not accessible. SALAD exhibits comparable performance while being less data-hungry.
2. The insights into the root cause of the text-speech understanding gap could be influential for future research on speech-LLMs.

### Weaknesses
1. The experiments focus on English speech and text, which couldn't address the generalizability to other languages or domains. Similarly, the synthetic data only contains 1 voice, which raises question about how the model can adapt to speaker variance.
2. Besides the quantitative results, it would be beneficial to provide some qualitative analysis to intuitively demonstrate how SALAD training minimize the text speech gap.

### Questions
1. In Table 3, How to interpret that SALAD-3B often achieves better/smaller text-speech gap than SALAD-7B? Doesn't larger model require more training steps?
2. Could there any negative impacts on model's capabilities in understanding audio/paralinguistic inputs when adopting SALAD training?
3. In Table 4, does the fact that SALAD-7B gets boosted on text task after SALAD training indicate that the training sets employed is giving SALAD models advantage in these tasks? (i.e. in-domain vs out-of-domain?)

I am giving a rating of 6 but am open to reconsider when authors answer these questions.

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
This paper identifies and analyzes the "text-speech understanding gap" in speech-adapted Large Language Models (LLMs), attributing it to two quantifiable factors: forgetting of pre-trained text capabilities and cross-modal misalignment. To bridge this gap, the authors propose SALAD, a two-stage method comprising: (1) Cross-modal distillation from the text-based LLM teacher to the speech-adapted student model, and (2) Active selection of a minimal amount of synthetic speech data to target residual domain misalignment. Experiments on 3B and 7B models show that SALAD achieves competitive performance with state-of-the-art models while using significantly less speech data.

### Strengths
- Formally defining and quantifying the text-speech gap via forgetting and misalignment.
- The paper is well-written and easy to follow.
- The paper shows that high performance can be achieved with significantly less data.

### Weaknesses
- The paper does not meet the standard for a thorough related work section, failing to properly situate itself within the current literature and justify its novelty.
- The paper fails to discuss and contrast its approach with highly relevant work, such as BLSP-KD and TASTE.
- The paper lacks of an ablation study on the hyperparameters, such as K and γ.

### Questions
- Given that  BLSP-KD already demonstrated the power of cross-modal distillation for this problem, what is the marginal contribution of the active learning component? 
- Please provide an ablation on the cluster count K and the exponent γ.
- The paper choses a "worst-case" encoder to make a strong claim. However, how does SALAD compare against state-of-the-art non-causal encoders with built-in alignment mechanisms (e.g., TASTE)?

### Soundness
3

### Presentation
3

### Contribution
3
