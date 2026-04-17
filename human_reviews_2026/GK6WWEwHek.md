# Linguistic Properties and Model Scale in Brain Encoding: From Small to Compressed Language Models

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Recent studies have shown that full-precision Transformer-based large language models (LLMs) increasingly improve predictions of human brain activity as model parameters are scaled. However, the corresponding growth in size and computational cost limits their interpretability and practical deployment, particularly in applications such as brain-computer interfaces (BCIs), which demand low-latency and efficient models. To address this gap, two efficiency-oriented approaches that have emerged can be used: (i) adopting small language models (SLMs), which achieve competitive performance at substantially lower computational cost, and (ii) compressing LLMs through quantization, which reduce computational demands while retaining much of their original capacity.
However, it remains unclear whether such SLMs or compressed LLMs can effectively capture brain-relevant representations and achieve brain alignment comparable to that of full-precision LLMs. Specifically, our study is motivated by four key questions: (i) can compressed LLMs preserve brain alignment, which is critical for deployment; (ii) how do compressed LLMs compare to SLMs, informing trade-offs for practical applications; and (iii) if ultra-low-resource applications demand even smaller footprints, can compressed SLMs still maintain alignment with brain activity? (iv) Which aspect of linguistic competence (discourse, morphology, syntax, semantics, or reasoning) most strongly influences brain alignment as model size or quantization method varies? In this work, we systematically evaluate LLMs (7B), their quantized counterparts, SLMs (1B and 3B), and their quantized variants to assess how model scale and compression jointly affect brain alignment, using fMRI recordings collected during naturalistic story listening. Our findings indicate that 3B SLMs achieve brain prediction performance comparable to both full-precision and compressed LLMs across whole-brain and core language regions. In contrast, 1B SLMs show a significant drop in brain alignment, particularly in semantic-processing regions. Notably, while most quantization methods preserve alignment, GPTQ quantization leads to reduced brain alignment across both LLMs and SLMs. Finally, benchmarking with the FlashHolmes suite shows that quantization primarily degrades discourse, syntax, and morphology, while leaving overall brain alignment intact.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper evaluates the brain alignment (via Deniz et al. 2019 fMRI recordings) of large language models, small language models, and thereof quantized versions. 
Large (language) models have been linked to activity in the human brain, but these might be too big for BCI applications and the impact of compression remains unclear.
The main claims are 
that quantization (with AWQ and SmoothQuant) mostly preserves brain alignment, 
that there is comparable brain alignment of 3B and 7-8B models (but not 1-1.5B),
and that the reason for lower brain alignment of GPTQ-quantized models and smaller models is reduced linguistic competence.

### Strengths
1. As far as I am aware the first study to evaluate the effect of quantization on brain alignment
2. Various techniques for quantization considered (I'm not an expert in this though)
3. Methodology is clearly described, making the work reproducible

### Weaknesses
## 1. Core claims insufficiently tested
The core claim "Quantized variants of LLMs preserve most of their brain alignment" is never directly evaluated as far as I can tell. The reader is referred to Figure 2, but is there a significant change between the base model and its quantized counterpart? How large is that change? (GPTQ is noted to reduce alignment by 5-7%, what about AWQ and smooth?)
Similar question for the claim that "3B SLMs achieve brain alignment comparable to their larger 7B–8B LLMs" -- missing stats and exact quantification with confidence intervals. Otherwise it is difficult to generalize any of these findings. I would also like to see the statement "requiring substantially fewer resources" detailed in numbers.

## 2. Benefits of quantization not quantified
A major motivation for this work is that quantized and small language models are easier to run, because of model size, throughput, or energy requirements. I would have expected a plot or table that contrasts runtime costs vs brain alignment. As far as I understand this is the main motivation, but it never materializes in the paper.

## 3. Figures and claims are not directly linked
While the text interprets the figures, it would be much easier for a reader to digest the empirical support for the claims if figures were structured according to the claims. E.g., for the first claim why not have two bars, one being the average alignment of baseline LLMs, and the other the average alignment of quantized versions? The claim would then be that the bars are approximately equal, which makes it easier to parse the analysis. 

Code is not available, but authors promise its release upon publication of the paper.

Minor:
* L078: I don't understand this sentence "achieve competitive performance compared to their larger counterparts the shelf with fewer resources"
* L122: "we evaluate models with respect to two hypotheses: (1) Brain Alignment [...] (2) Linguistic Competence" -- those are not hypotheses but tests.
* Figure 3: very difficult to read, text is too small and graphic is pixelated
* it would be great to test these on more models and datasets, but that is of course always true...

### Questions
Please address major weaknesses 1-3.

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
This paper investigates whether small language models (SLMs) and quantized large language models (LLMs) can maintain brain alignment—i.e., the ability to predict human brain activity during naturalistic language processing—as effectively as full-scale LLMs. Using fMRI data from participants listening to stories, the authors evaluate models across scales (1B to 8B parameters) and quantization methods (AWQ, GPTQ, SmoothQuant). They also assess linguistic competence using the FlashHolmes benchmark, which spans morphology, syntax, semantics, discourse, and reasoning.

### Strengths
Originality: First to jointly study model scale, compression, and brain alignment in a controlled neuroimaging setting.

Quality: Strong empirical foundation with multiple model families, quantization methods, and brain regions.

Clarity: Clear distinction between brain alignment and task performance, with nuanced interpretation of their divergence.

Significance: Offers practical recommendations for neuroAI applications, especially in low-latency or low-resource environments like BCIs.

### Weaknesses
1.Model scale upper limit: The largest model evaluated is 8B; extrapolating findings to 13B+ models (DeepSeek-R1-Distill-Qwen-14B, Qwen3-32B, DeepSeek-R1-Distill-Qwen-32B, etc.) remains unclear. 

2.Compression scope: The motivation and experimental design of this article are not convincing. Only post-training quantization is studied; pruning, distillation, or structured compression are not explored.

3.Modality limitation: Only fMRI is used; MEG or ECoG could reveal temporal dynamics of alignment loss.

4.Decoder gap: The study focuses on encoding (brain prediction); decoding (stimulus reconstruction) is not addressed.

### Questions
1.Why limit the largest model to 8B? Do you expect 3B SLMs to still align with 13B+ models, or would the gap widen?

2.Any plans to test pruning or distillation? These methods may preserve structure better than quantization—how might they affect alignment?

3.Would MEG or ECoG reveal different sensitivities? fMRI is slow; could temporal resolution expose real-time alignment loss?

4.Why not evaluate decoding performance? For BCI use, reconstructing stimuli from brain data is crucial—do SLMs or quantized models degrade decoding fidelity?

5.Could you release the brain alignment scores per task/region? This would help correlate specific linguistic deficits with regional brain activity loss.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates how model scale and post-training quantization influence the alignment between language model representations and human brain activity. Using fMRI data collected during naturalistic story listening, and embeddings extracted from both large and small language models (LLaMA-3.2 and Qwen-2.5 families), the authors train voxel-wise encoding models to predict neural responses. They further evaluate each model’s linguistic abilities across five domains (discourse, syntax, semantics, morphology, reasoning) using the FlashHolmes benchmark, enabling an analysis of which linguistic properties best support brain alignment.

The results indicate that: (1) models around 3B parameters achieve comparable brain alignment to 7B–8B models, suggesting diminishing returns beyond moderate scale; (2) quantization methods differ substantially, AWQ and SmoothQuant largely preserve alignment, whereas GPTQ causes noticeable degradation; and (3) discourse-level abilities show the strongest correlation with brain predictivity. Overall, the study highlights the potential of efficient small or quantized models for NeuroAI applications, while emphasizing that not all compression methods are equally compatible with brain alignment.

### Strengths
Timely and relevant topic. The paper addresses an important and underexplored question: how scaling and compression strategies affect the brain-alignment properties of language models. This is of both scientific and practical interest for NeuroAI research.

Comprehensive evaluation design. By combining voxel-wise encoding models with a linguistic benchmark (FlashHolmes), the study bridges neural and computational levels of analysis and identifies which linguistic properties support brain alignment.

Systematic comparison of quantization methods. Evaluating AWQ, GPTQ, and SmoothQuant across multiple model sizes is methodologically rigorous and provides clear guidance for selecting compression techniques in brain-related applications.

### Weaknesses
**Lack of statistical validation.**
Figure 2 compares alignment across quantization methods for the same model (e.g., Qwen), but no statistical tests, confidence intervals, or measures of variability across runs or participants are reported. Consequently, it is unclear whether observed differences, such as the apparent improvement after AWQ compression, are meaningful or fall within noise. Without proper significance testing, it is difficult to interpret whether certain compression methods reliably affect alignment.

**Narrow model range and limited scaling claims.**
The study only examines models up to 8B parameters, whereas prior scaling-law studies extend to 72B [1]. The claim that alignment saturates around 3B is therefore tentative, as larger models might still yield improvements.

Reference
[1] Richard Antonello, Aditya Vaidya, and Alexander Huth. Scaling laws for language encoding models in fmri. Advances in Neural Information Processing Systems, 36, 2024.

**Placement and discussion of limitations.**
The discussion of limitations is relegated to the appendix, which weakens the main narrative. Given that all tested models are relatively small, the study might not capture true scaling behavior. The normalization by noise ceiling can make small-model results look stronger than they are, this point deserves more transparent discussion in the main text.

**Statistical reporting and subject variability.**
The paper lacks an analysis of subject variability, it is unclear whether the results in Figure 2 and related analyses hold consistently across participants. Reporting variance or confidence intervals across subjects would greatly strengthen the reliability of the findings.

**Limited interpretability discussion.**
The relationship between degraded linguistic properties (e.g., discourse, syntax) and specific brain regions is underdeveloped. A more detailed interpretation could link linguistic theory to neural substrates.

**Overly verbose writing.**
Some procedural details (e.g., preprocessing, HRF estimation) could be moved to the appendix, allowing the main text to focus on results and their interpretation.

### Questions
Clarity of figures.
Figure 2 should clearly label “1.5B.” Figure 3 is difficult to interpret due to unclear color scales and range indicators. Similar issues appear in the flatmap visualizations in the appendix (such as Figure 6-7, 11-20).

Layer selection.
Which model layers were used for brain alignment? Did you observe consistent optimal layers across models and quantization methods?

Compression degree.
What are the actual compression ratios achieved by each method for different model sizes? How do AWQ, GPTQ, and SmoothQuant compare in terms of model size reduction and their impact on brain alignment?

Statistical testing.
Did you conduct permutation or bootstrap tests to assess whether alignment differences between quantization methods are significant? Visual trends alone are insufficient to judge reliability.

Subject-level robustness.
Were results consistent across individual participants? Showing variance or confidence intervals across subjects would strengthen the argument.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates whether small or quantized language models can achieve brain alignment comparable to large full-precision LLMs. Using fMRI data from participants listening to naturalistic stories, the authors evaluate models such as LLaMA-3 and Qwen-2.5 and their quantized variants Brain alignment is measured via voxel-wise encoding correlations, and linguistic competence is assessed using the FlashHolmes benchmark covering morphology, syntax, semantics, discourse, and reasoning. The authors report that 3B models perform comparably to 7B–8B ones, while GPTQ quantization harms both linguistic competence and brain alignment.

### Strengths
1.	The study addresses an emerging question about model efficiency and neural alignment, offering a systematic comparison between model size, quantization, and brain predictivity. The experimental framework—combining fMRI encoding and linguistic probing—is technically sound and clearly presented.
2.	The paper is well written and carefully structured, with clear visualizations and comprehensive references to recent brain–language modeling literature.
3.	The attempt to connect linguistic task performance with neural alignment is conceptually interesting and could, in principle, guide future investigations of how language models capture brain-relevant information.

### Weaknesses
1.	The overall motivation and significance are limited. While scaling and efficiency are relevant engineering questions, it is unclear why quantized models are meaningful for cognitive neuroscience. Brain–language alignment research is primarily driven by scientific, not deployment, objectives, so the practical incentive for compressing models in this context is weak. As presented, the work is more an engineering benchmark than a scientific contribution.
2.	The dataset and experimental scope are narrow. The Moth Radio Hour fMRI corpus includes only nine subjects and a single passive listening task. Such limited data cannot support broad conclusions about how model scale or quantization affects neural language representations. Results on one dataset may not generalize to other linguistic or cognitive paradigms.
3.	The link between linguistic competence and brain alignment is tenuous. The five FlashHolmes task categories are only a partial proxy for linguistic processing in the brain, and correlations between these scores and voxel alignment are descriptive rather than causal. 
4.	The main conclusion—that 3B models can match or surpass 7B–8B models, challenging previous “scaling laws” in brain encoding—is not well supported. The tested model range is small (maximum 7B), and the data are noisy and task-limited. Prior literature consistently finds positive scaling effects; overturning that trend would require more comprehensive evidence and statistical testing.
5.	Minor issues: Figure 2’s legend mislabels Qwen where it should read LLaMA.

### Questions
1.	There is a growing body of work on visual brain alignment showing that factors such as model scale, training data diversity, and dataset domain strongly affect alignment quality. Why does this paper not analyze similar factors for language models, beyond simple parameter count and quantization?
2.	Current language modeling research has diversified far beyond transformers, including diffusion-based, state-space (e.g., Mamba), and other architectures. How do you expect these non-transformer paradigms to influence brain alignment? Do you believe the transformer mechanism itself is critical for the observed effects, or could similar patterns emerge with alternative architectures?

### Soundness
2

### Presentation
3

### Contribution
2
