# GeneBreaker: Jailbreak Attacks against DNA Language Models with Pathogenicity Guidance

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
DNA, encoding genetic instructions for almost all living organisms, fuels groundbreaking advances in genomics and synthetic biology. Recently, DNA Language Models have achieved success in designing synthetic functional DNA sequences, even whole genomes of novel bacteriophage, verified with wet lab experiments.  Such remarkable generative power also brings severe biosafety concerns about whether DNA language models can design human viruses. With the goal of exposing vulnerabilities and informing the development of robust safeguarding techniques, we perform a systematic biosafety evaluation of DNA language models through the lens of jailbreak attacks. Specifically, we introduce JailbreakDNABench, a benchmark centered on high-priority human viruses, together with an end-to-end jailbreak framework, GeneBreaker. GeneBreaker integrates three key components: (1) an LLM agent equipped with customized bioinformatics tools to design high-homology yet non-pathogenic jailbreak prompts, (2) beam search guided by PathoLM and log-probability heuristics to steer sequence generation toward pathogen-like outputs, and (3) a BLAST- and function-annotation–based evaluation pipeline to identify successful jailbreaks. On JailbreakDNABench, GeneBreaker successfully jailbreaks the latest Evo series models across 6 viral categories consistently (up to 60\% Attack Success Rate for Evo2-40B). Further case studies on SARS-CoV-2 spike protein and HIV-1 envelope protein demonstrate the sequence and structural fidelity of jailbreak output, while evolutionary modeling of SARS-CoV-2 underscores biosecurity risks. Our findings also reveal that scaling DNA language models amplifies dual-use risks, motivating enhanced safety alignment and tracing mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents the first systematic biosafety evaluation of DNA Language Models, specifically the state-of-the-art Evo series models, against jailbreak attacks. The authors introduced JailbreakDNABench, a comprehensive benchmark of six high-priority viral categories and evaluation pipeline for systematic biosecurity risk assessments. Also, they introduced GeneBreaker, a novel jailbreak framework that probes vulnerabilities of DNA language models. They find that Genebreaker can successfully jailbreak the  series models on all 6 viral categories, achieving an Attack Success Rate of up to 60.0% on the Evo2-40B.

### Strengths
- Overall clear and well-structured paper
- Addresses one of the most critical safety concerns surrounding generative AI applied to biology, providing the first systematic evidence of a jailbreak vulnerability in frontier DNA-LM.
- Novel and effective jailbreaking framework, GeneBreaker, that combines domain-specific knowledge and modern LLM jailbreak tactics at is able to successfully jailbreaks the latest Evo series models across 6 viral categories.

### Weaknesses
- The reliance on PathoLM and logP to guide chunk decoding may limit their search space. Did the authors did any study on using other methods of pathogen scoring to test the robustness of their methods.   
- The lack of wet lab experiments to verify their findings. It will further boost the impact of their work.

### Questions
- Have the authors explored any alternative scoring functions to guide the beam search?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a framework to evaluate whether genomic foundation models can be jailbroken to produce sequences closely resembling regulated human pathogens. The contributions include
(i) JailbreakDNABench, a curated benchmark covering six viral categories.
(ii) GeneBreaker, which combines an LLM agent for high-homology, non-pathogenic prompt design, beam search guided by PathoLM + log-prob heuristics, and an evaluation pipeline using BLAST and VADR function checks.
On JailbreakDNABench, GeneBreaker successfully jailbreaks the latest Evo series models across 6 viral categories consistently.

### Strengths
(i) Novelty: The paper provides the first framework to evaluate whether genomic foundation models can be jailbroken to produce sequences closely resembling regulated human pathogens.

(ii) Technical completeness: LLM-assisted prompt construction + guided beam search + BLAST/VADR evaluation.

(iii) Robust experimental results: On JailbreakDNABench, GeneBreaker successfully jailbreaks the latest Evo series models across 6 viral categories consistently.

### Weaknesses
(i) Dataset transparency: The paper does not provide a full list of sequences/accession IDs or the dataset statistics (e.g., dataset sample size)

(ii) There are two biological statements, it is a little hard for people without much biological background to understand.
 
(iii) Wrong statements: In Table 3, GenomeOcean just uses the Mistral architecture, not an MOE model.

### Questions
(i) Dataset size: I do not see the dataset sample size, but I am concerned about whether the sample size is enough in each task as a benchmark.

(ii) The tasks are about the virus. Are there other aspects we can think about to evaluate whether genomic foundation models can be jailbroken?

(iii) What kind of insights can we get to avoid the jailbreak problem when developing the genomic foundation models?

### Soundness
3

### Presentation
2

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
This paper investigates a serious and emerging problem: whether DNA language models (DNA-LMs), large generative models trained on genomic data, can be jailbroken to produce potentially dangerous biological sequences, such as viral genes. The authors build a new benchmark called JailbreakDNABench, covering six classes of human viruses, and propose a red-teaming system named GeneBreaker to test how easily these models can be exploited.

GeneBreaker uses three main components:
(1) a large language model agent that automatically finds non-pathogenic DNA sequences similar to real viruses to craft jailbreak prompts;
(2) a beam search guided by another model (PathoLM) and a probability-based score to push DNA-LMs toward pathogen-like outputs; and
(3) a BLAST- and annotation-based pipeline to check if the generated sequences resemble known human viruses.

The authors test several state-of-the-art DNA models (Evo2-40B, GenomeOcean, GENERator) and find that the larger the model, the higher the jailbreak success rate, reaching up to 60% similarity for some virus categories. Case studies on SARS-CoV-2 and HIV-1 show that the generated DNA can recreate proteins with high structural similarity to real viral proteins when analyzed using AlphaFold3.

### Strengths
While many works study jailbreaks in text LLMs, almost none examine biological foundation models. The proposed JailbreakDNABench gives the community a starting point to measure and compare biosafety risks. The GeneBreaker framework combines a prompt-designing LLM, a guided beam search, and a BLAST-based evaluation pipeline in a way that feels methodical and reproducible. The idea of using PathoLM as a guidance signal is clever and biologically grounded. The authors test multiple large DNA models across six virus categories and analyze how model size, prompt similarity, and search parameters affect attack success. These broad results make the findings more convincing.

### Weaknesses
1. The paper stops at exposing risks but provides no concrete defense or biosafety governance mechanism beyond brief veto filtering in the Appendix.

2. The benchmark seems built entirely by the authors. How can we be sure JailbreakDNABench isn’t biased toward viruses that are easier to hit, making GeneBreaker look stronger?

3. The experiments use only five trials per model. With so few runs, can we confidently say Evo2-40B is truly more vulnerable than the smaller ones?

4. While plausible, this conclusion isn’t proven. Larger models correlate with higher similarity scores, but correlation ≠ causation. No controlled safety-aligned baseline is tested to confirm that scaling causes more risk.

### Questions
1. Why is 90 % BLAST identity chosen as the success threshold? Did you test whether this threshold actually correlates with any functional or structural similarity beyond sequence overlap?

2. Can GeneBreaker jailbreak models that are explicitly safety-aligned or fine-tuned with filtering mechanisms, or does it only work on open, research-grade DNA models?

### Soundness
3

### Presentation
3

### Contribution
2
