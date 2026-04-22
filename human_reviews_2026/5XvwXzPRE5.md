# BasePrompt: Self-Prompting Genome Language Models for RNA Fitness Prediction

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Genome Language Models (GLMs) pre-trained on trillions of nucleotides already exhibit strong zero-shot RNA fitness predictors, yet they cannot be steered toward a specific assay the way a language model is steered by a prompt.
We close this gap by letting GLMs prompt themselves.
Our method, BasePrompt, asks GLMs to propose short nucleic-acid prefixes and postfixes that maximally activate the fitness signal for a given sequence.
To overcome the causal, forward-only nature of most GLMs, we exploit reverse-complement symmetry and generate upstream as well as downstream prompts without ever updating weights or using labeled variants.
For zero-shot RNA fitness prediction on RNAGym, BasePrompt achieves a 6.0\% relative improvement over the SOTA Evo2 7B model and 6.6\%–16.4\% over other GLMs, as measured by Spearman correlation.
Auxiliary DNA tasks show the same prompting method compresses native-context information into shorter, model-aligned tokens, boosting pathogenicity classification and next-k-base prediction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents BasePrompt, an inference-time approach that asks a GLM to generate short 5′/3′ DNA prompts and concatenates them to nucleotide sequences to improve zero-shot fitness or variant effect prediction. The authors report consistent improvements on RNAGym and other benchmarks.

### Strengths
- Adapting prompting to genomic sequences and leveraging reverse-complement symmetry is an interesting and effective idea.
- Consistent improvements are reported on several benchmarks.
- Figures are easy to follow.

### Weaknesses
Major
- Reported gains (e.g., Spearman from 0.271 to 0.289 in Figure 5) lack confidence intervals or statistical tests. Without per-assay distributions and significance testing, one cannot tell whether improvements are robust.
- The abstract and introduction claim the method "asks GLMs to propose... prefixes and postfixes that maximally activate the fitness signal for a given sequence." This description strongly implies a task-aware optimization process. However, the actual method described in this paper is standard, task-agnostic autoregressive generation.

Minor
- Figure 9: Complement Srtand -> Complement Strand
- Figure 10-12: Stand -> Strand.

### Questions
- Can you please provide the absolute (non-normalized) Spearman r and AP values for the experiment in Figure 6?
- Does using random sequences (like RandSeq or genomic sampling) as prompts improve performance on the RNAGym benchmark?

### Soundness
2

### Presentation
2

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
This paper proposes BasePrompt, a self-prompting method for Genome Language Models (GLMs) that generates contextual sequence prefixes and suffixes to improve zero-shot RNA fitness prediction. Unlike text prompts, these are biologically meaningful subsequences constructed using reverse-complement symmetry, enabling bidirectional context without fine-tuning. Evaluations on RNAGym (70 RNA DMS assays), ClinVar, and Next-base Prediction tasks show consistent gains—up to 16% improvement in Spearman correlation over strong GLM baselines (Evo1–Evo2, GENERator). BasePrompt introduces no extra training and minimal inference overhead, establishing a new paradigm for in-context learning in genomics.

### Strengths
* This work presents a neat and simple trick for DNA language models, extending prompting to non-text biological sequences through a self-generated, symmetry-aware context mechanism.
* It achieves state-of-the-art zero-shot performance across multiple benchmarks, outperforming larger GLMs without any retraining.
* The method is architecturally general, working consistently across diverse GLMs and demonstrating robustness and scalability.

### Weaknesses
I did not observe any obvious weaknesses in this approach. However, it’s worth noting that the method remains fundamentally prompting-based in nature.

### Questions
1. When performing the inverse transcription process, how is RNA splicing handled? Certain RNA types such as mRNA contain introns in their original DNA sequences. Have the authors conducted any experiments or analyses related to this?
2. Regarding the prompting mechanism, what is the exact value of N used for both ends? How was the optimal prompt length determined? Was it tuned empirically, or did the model simply continue until the end of the sequence token?
3. In Table 9, performance drops are observed for Evo-1 on tRNA and Evo-2 on Ribozyme. Have the authors conducted any deeper analysis to understand the cause of these declines?
4. Could the authors provide an assay-level performance breakdown of their methods in the RNAGym benchmark similar to Supplementary Figure 2?
5. Could the authors also present a species-level performance breakdown, for example perf. changes on viral and eukaryotic RNAs, to better interpret the generalization behavior?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
BasePrompt is a self-prompting framework for genome language models (GLMs) that generates bidirectional nucleotide prompts to enhance zero-shot RNA fitness prediction.
The method addresses two key challenges: the lack of effective task-specific steering mechanisms for genomic LMs, and the inherent unidirectionality of causal language models, which limits contextual information extraction.
To overcome these, BasePrompt leverages the reverse-complement symmetry of DNA to generate both upstream (5′) and downstream (3′) prompts autoregressively, enabling the GLM to simulate bidirectional context without any model fine-tuning or supervision.
It is evaluated on RNAGym, a large benchmark comprising 70 deep mutational scanning assays, and compared against several genome-scale GLMs, showing consistent zero-shot improvements across multiple metrics; additional DNA tasks such as ClinVar variant classification and next-base prediction support the method’s generality.
BasePrompt is efficient, model-agnostic, and improves predictive accuracy without additional training, but its evaluation omits comparisons to mRNA-specific models and established biological UTRs, limiting the strength and generality of its conclusions.
In summary, BasePrompt is a novel and computationally efficient method that introduces self-prompting to genome language models, yielding consistent but modest zero-shot gains in RNA-related prediction tasks.

### Strengths
## Innovative Self-Prompting Concept

The idea of letting a genome language model prompt itself is novel and interesting. It creatively adapts the prompt engineering paradigm from NLP to the genomics domain, where “prompting” is not straightforward. The use of reverse-complement sequences to create both 5’ and 3’ prompts is a clever solution to exploit bidirectional context despite using a unidirectional (causal) model.

## Robustness and Efficiency

The method is shown to work across multiple model sizes and variants, indicating generality. Moreover, it’s computationally efficient – prompts are generated only for a fixed set of reference sequences (e.g. the 70 assay reference RNAs in RNAGym) rather than for every single variant. This means the overhead is low relative to the huge number of predictions made, making the approach practical for large-scale inference.

### Weaknesses
## Omission of relevant mRNA-specific baselines

A major concern is that the comparisons focus only on genome-scale or noncoding RNA language models that were never trained on mRNA data, whereas the task of RNA fitness prediction is about mRNA. In effect, the authors compare BasePrompt-augmented models against baselines that are out-of-domain for mRNA (e.g. generic genomic models or models trained on lncRNAs), which is akin to comparing against a misaligned or even random baseline for those specific tasks. Crucially, the evaluation excludes leading mRNA-focused models such as mRNA-FM, CaLM, and Uni-RNA, which are large pretrained models explicitly trained on coding RNA/transcript data. These specialized models likely represent the strongest baselines for mRNA-related fitness predictions, and ignoring them undermines the rigor and fairness of the evaluation. As a result, the reported improvements might be overstated – we cannot tell if BasePrompt would still outperform or even match these state-of-the-art mRNA-trained models, since they were not included. This gap in baseline selection affects the soundness of the experimental claims: without directly comparing to the most relevant prior models, the paper’s claim of superior performance is not fully convincing or generalizable to practical settings where one would naturally use an mRNA-trained model.

## Lack of real-world UTR baseline comparison

Another weakness is in how the paper evaluates the quality of the prompt sequences (especially for tasks involving untranslated regions, e.g. 3’ or 5’ UTRs). The authors generate synthetic upstream/downstream prompts, but they do not compare these prompts against any well-known, biologically-established UTR sequences. In synthetic biology and mRNA engineering, certain UTR sequences are commonly used because they are known to strongly influence expression. For example, the β-globin 3′ UTR is a widely used “classic” UTR element (often appended to mRNAs to enhance stability/translation). Such sequences provide a real-world baseline for performance. By failing to compare BasePrompt’s generated UTR prompts to any standard UTR (like β-globin or other commonly used regulatory UTRs), the authors miss an important check: Are the prompts that BasePrompt finds actually more informative or effective than simply using a known strong UTR? Without this comparison, it is hard to gauge the biological plausibility and practical advantage of the prompts. This omission limits the generality of the conclusions – for instance, if a simple known UTR could achieve similar or better fitness prediction signal when appended, then BasePrompt’s advantage might not be as significant in real laboratory or clinical contexts. In summary, not benchmarking against real-world sequences means we cannot yet conclude that BasePrompt’s prompt-generation is yielding truly superior or meaningful sequences compared to what practitioners already use.

### Questions
See Weaknesses

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
The paper introduces BasePrompt, a method that enhances genome language models (GLMs) for zero-shot RNA fitness prediction by letting them generate their own prompts. Unlike traditional NLP prompting, where users manually provide input cues, BasePrompt allows GLMs to create short nucleic acid prefixes and suffixes to boost prediction accuracy, without needing labeled data or model fine-tuning.

To overcome the unidirectional nature of autoregressive GLMs (which only look forward), the authors exploit the reverse-complement symmetry of DNA, enabling bidirectional prompt generation (both upstream and downstream of a sequence). These prompts are then concatenated to RNA sequence variants before prediction, improving performance across tasks.

### Strengths
1. Conceptual Originality

The paper proposes the idea of self-prompting genome language models (GLMs) that utilize reverse-complement symmetry for bidirectional inference. This represents a creative adaptation of prompt-based reasoning from NLP to genomics. While not groundbreaking, it introduces a fresh perspective on how pretrained sequence models can internally generate task-relevant prompts. However, since reverse-complement techniques are well-known in genomic modeling, the originality is moderate rather than strong.

2. Methodological Quality

The experimental results show consistent but modest improvements across several Evo-series models, with reported gains of 6–16% in Spearman correlation. The methodology is clearly described at a conceptual level, and the experiments are competently executed on the RNAGym benchmark. Nonetheless, the paper lacks statistical validation, ablation studies, and robustness analyses, limiting confidence in its empirical rigor. As a result, the technical quality is adequate but below top-tier standards.

3. Presentation and Clarity

The paper is well-structured and easy to follow, with clear motivation, figures, and narrative flow. The visual explanations of BasePrompt’s bidirectional prompting mechanism are particularly helpful. However, some implementation details—such as prompt generation procedures, sampling settings, and parameter sensitivity—are under-specified. This reduces transparency and reproducibility. Overall, the presentation is clear but not deeply explanatory.

4. Scientific and Practical Significance

BasePrompt touches on a promising intersection between language modeling and biological inference, suggesting that large genome models may self-optimize through internal prompting. Despite this potential, the impact is limited by the narrow experimental scope (focused mainly on RNA fitness prediction) and the strong dependence on Evo-series GLMs. The contribution is thus incremental rather than transformative for the field of computational genomics.

### Weaknesses
1. Insufficient Theoretical Foundation and Interpretability

The paper lacks a deep theoretical explanation for why the proposed method is effective. The performance improvements of BasePrompt are demonstrated empirically, but without strong biological or information-theoretic justification. Moreover, the biological meaning of the generated “self-prompts” is not explored — it remains unclear whether they correspond to functional motifs or biologically relevant patterns.

2. Limited Experimental Scope and Questionable Generalizability

Experiments are mainly conducted on a single benchmark, RNAGym, without evaluation on other RNA-related tasks such as secondary structure prediction, RNA–protein binding, or splicing. This narrow scope limits the evidence for BasePrompt’s robustness and general applicability across broader biological contexts.

3. Strong Dependence on Specific Models and Lack of Statistical Validation

The method heavily depends on the Evo-series genome language models (GLMs), with limited testing on other architectures. Additionally, the paper does not include statistical significance testing or detailed error analysis, making it difficult to assess the reliability and generality of the reported performance gains.

4. Limited Novelty and Biological Validation

The main innovation—using reverse-complement symmetry for bidirectional prompting—builds on ideas already explored in genomic modeling. While the approach provides an inference-time improvement, it lacks deeper biological validation or theoretical advancement. As a result, both its novelty and biological significance are somewhat limited.

### Questions
see weakness.

### Soundness
2

### Presentation
3

### Contribution
2
