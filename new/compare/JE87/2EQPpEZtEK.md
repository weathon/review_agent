# Review

## Summary
This paper proposes DiSTAR, a zero-shot text-to-speech framework that operates entirely in a discrete residual vector quantization (RVQ) code space and tightly couples an AR language model with a masked diffusion model, without forced alignment or a duration predictor.

## Soundness
3

## Presentation
2

## Contribution
3

## Strengths
1. The paper is well-written and easy to follow.
2. The proposed DiSTAR framework is novel, which operates entirely in a discrete residual vector quantization (RVQ) code space and tightly couples an autoregressive (AR) language model with a masked diffusion model.
3. The authors provide extensive experiments and ablations to demonstrate that DiSTAR surpasses state-of-the-art zero-shot TTS systems in robustness, naturalness, and speaker/style consistency, while maintaining rich output diversity.

## Weaknesses
1. Some experimental details are missing. For example, what is the specific implementation of the  RVQ resynthesized baseline in Table 1? What is the training cost (e.g., training time, GPU hours) of different models in Table 1?  What is the detailed configuration of the baseline E2TTS, F5TTS?
2. Some experimental results are not consistent with the descriptions in the paper. For example, in line 431, the authors state that "As model capacity grows, DISTAR yields consistent improvements on objective metrics, closely matching the scaling behavior reported for discrete-token autoregressive systems and indicating a healthy scaling trajectory." However, in Table 1, DiSTAR-medium (0.3B) does not show consistent improvements compared to DiSTAR-base (0.15B) on some metrics, e.g., SIM and UTMOS on LibriSpeech test-clean, and SIM and CMOS on SeedTTS test-en.
3. Some experimental results are not convincing. For example, in Table 3, the authors compare different decoding strategies. However, the results of "Sample 1 1" and "Sample 0.95 0.8" are almost the same, which is not sufficient to demonstrate the effectiveness of different decoding strategies.
4. The authors state that "we implement classifier-free guidance (CFG) for the masked diffusion module by independently dropping (i) the AR LM conditioning output and (ii) the past-code window with probabilities 0.1 and 0.1, respectively". However, this is not consistent with the original classifier-free guidance, which drops the conditioning input with probability 0.1 and the predicted tokens with probability 0.1. It is not clear why the authors choose to drop the past-code window instead of the predicted tokens.

## Questions
1. What is the specific implementation of the  RVQ resynthesized baseline in Table 1?
2. What is the training cost (e.g., training time, GPU hours) of different models in Table 1?
3. What is the detailed configuration of the baseline E2TTS, F5TTS?
4. Why the authors choose to drop the past-code window instead of the predicted tokens in the classifier-free guidance?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4