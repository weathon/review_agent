# Robustness of Probabilistic Models to Low-Quality Data: A Multi-Perspective Analysis

- Decision: Accept (Poster)
- Scores: 6, 2, 2, 6

## Abstract
This paper investigates a critical challenge in modern machine learning: 
how different probabilistic models withstand low-quality training data. 
Through a systematic, comparative investigation, 
we reveal a stark spectrum of robustness. 
Empirically, we find that autoregressive language models 
exhibit remarkable resilience against both token-level noise 
and structural corruption (for GPT-2, test NLL increases modestly from 
2.87 to 3.59 despite 50\% corruption). 
By sharp contrast, class-conditional diffusion models degrade catastrophically 
under identical noise levels (image-label consistency plummets by 56.81\%), 
while image classifiers show a moderate vulnerability that diminishes with dataset scale.
To explain these discrepancies, 
we analyze the results through a multi-perspective lens 
integrating information theory, PAC learning, and gradient dynamics. 
This framework identifies what informational properties drive robustness, 
why they are required for generalization, 
and how the optimization process achieves this resilience.
These analyses suggest that robustness is heavily influenced by two key principles: 
the richness of conditioning information, which constrains the learning problem, 
and the absolute information content of the training data, which allows the signal from correct information to dominate statistical noise.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a systematic experimental analysis of the sensitivity of mainstream probabilistic models—specifically autoregressive, diffusion, and classifier models—to low-quality data. The authors identify two primary factors governing robustness: the "richness of conditioning information" and the "absolute information content" of the training data. The paper further provides theoretical grounding for these findings from the perspectives of information theory, PAC learning, and gradient dynamics.

### Strengths
1. The paper provides a systematic and logically self-consistent experimental investigation that effectively demonstrates the varying tolerances of autoregressive models, diffusion models, and image classifiers to low-quality data.
2. The two key factors proposed as governing robustness—the "richness of conditioning information" and the "absolute information content"—are insightful and provide a strong conceptual framework for understanding the observed phenomena.
3. The explanatory analysis accompanying the experimental results is thorough and convincing, particularly the multi-perspective theoretical support that unifies the empirical findings.
4. The paper's technical details are clearly presented, and the experimental setup appears to be highly reproducible.

### Weaknesses
1. The study's focus on training models "from scratch" (as noted in Sec 3.5) overlooks the dominant "pre-training and fine-tuning" paradigm used in modern practice. The robustness of a large model pre-trained on massive datasets may differ significantly when fine-tuned on low-quality data. This omission limits the direct applicability of the paper's findings to many real-world scenarios.
2. The analysis relies almost exclusively on unstructured, random noise (as acknowledged in Appendix B). This is a significant limitation, as real-world data noise is often structured or correlated (e.g., systematic mislabeling). It would substantially strengthen the claims to include even a simple experiment with structured noise. For instance, in the classification task, could the authors investigate a scenario where noise is not uniformly random but fixed to specific incorrect labels (e.g., all instances are mislabeled as class 'B')?

### Questions
1. Following on from weakness #2, the paper's conclusions on robustness, especially the explanation rooted in gradient averaging (Sec 4.3), are derived from experiments using unstructured random noise. How well would these principles generalize to real-world, structured noise patterns where the "noise" gradients are coherent, not random, and thus would not average out?
2. The experimental design in Sec. 3.1 attempts to isolate noise effects by fixing the “equivalent correct sample exposure,”but this approach changes both batch size and iteration count, introducing confounding factors. A control experiment with fixed batch size and varying noise ratio would better isolate the true impact of gradient averaging from training configuration effects.
3. Section 3.3 argues for the effect of context richness by comparing the WMT 2014 translation task (sparse context) with the CNN/DailyMail summarization task (rich context). As these are fundamentally different tasks, it is difficult to attribute the observed difference in robustness solely to the richness of the conditioning information. A more convincing demonstration would involve comparing robustness within the same task.
4. I am not an expert in this specific area. Based on the paper's clear writing and well-structured experiments, I am open to adjusting my score pending the feedback from reviewers with deeper expertise.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies how different models behave when trained with low-quality (corrupted) data and proposes a three-part explanation with information-theoretic, PAC, and gradient-based perspectives. Empirically, the authors report: 
1. Autoregressive LMs remain strong even with heavy token/target corruption
2. Class-conditional diffusion models collapse as label corruption grows (while FID stays roughly flat, the accuracy of conditional generation drops)
3. Image classifiers are moderately sensitive on small datasets but surprisingly robust at ImageNet scale. 

Then they offer interpretations of these empirical observations based on distinct theoretical frameworks.

### Strengths
The empirical observation that GPT-2 and ImageNet-scale classification can tolerate, or even thrive under heavy label/target corruption is interesting and perhaps worth documenting.

### Weaknesses
Beyond that, the theoretical analysis/explanation they provide seems weak and superficial. Each of the three perspectives allows for straightforward counter-arguments, and the paper stops short of unifying them into a single, predictive theory. The writing is polished and persuasive on the surface, but the theoretical substance is thin; as a result, the paper takes rather longer to read than it should.
Below, I provide my view on each perspective the paper provides.

1) Information-theoretic perspective

- Residual information cannot explain “no degradation” at 50% noise. The relative information loss calculation under uniform noise decays with the number of classes but cannot go below $p_e$, which is its asymptotic limit. Hence, this argument cannot by itself explain why the full-ImageNet classifier exhibits no degradation (and even slight improvement) with ~50% incorrect labels.

- The paper argues that the “absolute information content” drives robustness, but in their experiments (to my understanding) the total exposure to correct samples, i.e., absolute information, is held constant across noise conditions. If so, the framework again does not clarify why the full-ImageNet setting is uniquely robust while smaller-scale settings are not, given that the protocol equalizes clean-signal exposure.

2) PAC perspective

- They merely discuss a classical bound, not a targeted explanation. The appeal to $\Omega(\epsilon^{-1}\log\delta^{-1} + d\epsilon^{-1} \log \epsilon^{-1})$ lower bounds is standard and doesn’t address the core issue in these experiments. Even when the number of correct samples seen is held fixed, additional corrupted samples still inject competing supervision that makes the signal harder to extract. This is simply ignored and the message of Subsection 4.2 reads as if “making the number of clean samples $m$ very large” is the operative solution. However, it does not clarify why $m$ being large should make the training robust to noisy samples.

- The link from “conditioning richness” to effective VC dimension $d$ is hand-wavy. The diffusion example is particularly strained: conceptually, conditional diffusion model jointly learns (i) the data distribution and (ii) a class-conditional guidance. Even with label corruption, part (i) is achieved (as the authors report, the FID remains stable), and what fails is mainly the alignment between the conditional label and the type of the images generated. This is like an unconditional diffusion model equipped with inaccurately trained classifier guidance. Simply attributing this to the larger VC dimension compared to sparse conditions blurs the comprehensive picture. To properly support such a claim, one would need experiments that vary the number of classes in class-conditional generation or also consider text-to-image diffusion setting (richer conditioning) in parallel, and then compare those outcomes. Direct comparison of conditional diffusion model with image classifier does not seem appropriate to me.

3) Gradient-based perspective

- The authors previously mentioned that they use larger batch sizes to stabilize the optimization process under high-noise conditions. However, the reported standard deviations in Table 4 rather show the noisy model exhibiting lower loss variance than the clean baseline at comparable batch sizes. If the loss variance is intended as a proxy for gradient stability, this observation seems inconsistent with their comment on training instability in high-noise regimes. It is genuinely difficult to see what the authors intend to claim with this analysis.

- Line 457 mentions that incorrect gradients are diverse/orthogonal and thus cancel, which might be true considering the statistics of Table 4. However, the paper does not directly test this (e.g., cosine similarity distributions between per-example gradients from clean vs. corrupted samples, magnitude of gradient sum from clean vs. corrupted samples).

### Questions
See Weakness section.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors of this paper provide a review of generative models' evaluation in the context of corrupted (noisy) data, taking into account several levels of noise injection, and explore their corresponding robustness. They observe that autoregressive language models, are  more resilient to low quality data, compared to diffusion models.
In order to explain these discrepancies, they analyze the results through employing tools from information theory, PAC learning, and gradient dynamics. Through their analyses they conclude that robustness is affected by two factors, namely the richness of conditioning and the absolute information content of the training data, where the former is driven by the learning problem and the latter is related to the data per se

### Strengths
In this paper the authors provide a comprehensive review of two classes of generative models: autoregressive models for text generation and class-conditional diffusion models, in the presence of noisy/low quality data, for several signal to noise ratios.
In order to perform their analysis they employ metrics from a spectrum of perspectives, namely information theory, PAC learning, and gradient dynamics. They conclude that the robustness of the method is dependent on the task (richness of conditioning) and the training data (absolute information content)

### Weaknesses
The authors test two different types of generative models under different tasks, for different types of data (categorical and continuous) using different metrics.
They do not provide a single task that test these models against, using the same criteria, so in a sense, the comparison in not objective/very informative.

Also, the results that they present under no corrupted data do not match the ones in the corresponding literature 
For example the test accuracy of CIFAR-100 was found to be significantly higher than the one reported here, at [1] and [2], namely 84.3% and 75.22% respectively


[1] "Text-to-Image Diffusion Models are Zero-Shot Classifiers", Clark et al., NeurIPS 2023, and 
[2] "Better Diffusion Models Further Improve Adversarial Training", Wang et al., ICML 2023

### Questions
I would like to ask the authors if they could please :

-- compare coth generative models under same task and same metrics;
-- employ the SOTA of both generative models whilst analyzing their performance

### Soundness
1

### Presentation
1

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
[Please note that, unlike for the other papers in my batch, I am expert only for a subset of the material in this paper, so my review and score is overall pretty reserved/ weak confidence. I will of course participate in the discussion period, and am amenable to changing my score. I apologize if some of my questions seem basic; answering them will nevertheless help me engage more effectively in the discussion period. I will do my best to give a fair assessment, or recommend that the AC down-weight my review if by the end of the discussion period if I have not met my own (I think high) bar for substantive participation.]

The paper studies how different classes of probabilistic models (autogressrive LLMs, sequence-to-sequence transformers, class-conditional diffusion models, image classifiers) behave when trained with noisy (low quality) data. The work systematically injects controlled random corruption into inputs or labels and compares how performance changes across these architectures/ training dynamics. 

Here is my summary of the setup and contributions (please let me know if you think I missed something important):

**Core setup**
- Noise model: random uniform corruption of tokens or labels; corruption ratios from 10–100% (effective error rate $\tfrac{r}{1+r}$
- 4 types of models, as discussed above 
- 2 types of experiments: (a)  constant clean-signal exposure, scales total compute by $1+r$ to keep the number of correct samples constant; (b) fixed budget, noisy data displacing clean data
- Overall, finds that AR models are pretty resilient; seq-to-seq models with richer conditioning can be made more robust; diffusion models can degrade a ton; classifiers are more sensitive on smaller datasets; some nice results showing that gradient averaging can help

### Strengths
This paper has many strengths:

- I don't think I've ever seen a paper put this extent of models into one controlled framework. I think the design here is very careful, and highlights meaningful differences between these model types and training dynamics

- The theoretical framing is clean. It ties together a variety of ideas (information theory, PAC, gradient perpsective) to try to explain what's going on here, to give more explanation beyond the high-level takeaways (e.g., that the richness of the context matters).

- The experiments are well constructed to support the theoretical analysis; they seem very careful and contained. 

- Overall, the writing does not over-lcaim causal mechanisms and is very clear about limitations (e.g., scale). I appreciate this a lot, given the ambitious nature of this paper. I think this is a clear/refreshing strength of the work.

### Weaknesses
As stated in my summary, please note I am not an expert on all material in this paper. I've combined observations that I think might be weaknesses with related questions.

**Noise model**

All injected noise is random and unstructured (uniform token or label replacement). In practical settings, low-quality data are often structured (e.g., correlated, systematically mislabeled). Is it fair to say that the results therefore demonstrate robustness to _stochastic corruption_, not necessarily to _realistic dataset noise_?

**Experimental design**

In the constant/clean-signal experiments, increasing compute by $1+r$ changes the effective LR, regularization. I think the isolation of robustness, as a result, is not entirely clean. Can you please discuss this furhter? 

**Theory vs. experiments**

Gradient-averaging explanation is intuitive but the experiments don't go into this, unless I missed something. There's no ablation separating variance reduction from bias.

**A couple of instances of slight over-claims**

E.g., I think it is over-generalized to say things like decoder-only LMs are largely insensitive to low-quality data; the writing in general is very careful, but things like this should be hedged. For that experiment, the study uses one model and random noise, not realistic web contamination or fine-tuning noise.

**Novelty re: theory**

This isn't a big deal, but wanted to ask about it. My take take is that the theory isn't super novel; the principles elicited restate known relationships between sample complexity, conditioning, SNR. What's nice here is the unification, but I don't think it's fundamentally new.

### Questions
I've integrated by questions in the "weaknesses" section so that they are directly next to observations I've made. I'm hoping this is clearer than separating them out.

### Soundness
3

### Presentation
3

### Contribution
3
