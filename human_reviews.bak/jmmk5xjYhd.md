# MalTrans: Unsupervised Binary Code Translation with Application to Malware Detection

- Decision: Reject
- Scores: 5, 6, 5, 5

## Abstract
Applying deep learning to malware detection has drawn great attention due to its notable performance. With the increasing prevalence of cyberattacks targeting IoT devices, there is a parallel rise in the development of malware across various Instruction Set Architectures (ISAs). It is thus important to extend malware detection capacity to multiple ISAs. However, training a deep learning-based malware detection model usually requires a large number of labeled malware samples.
The process of collecting and labeling sufficient malware samples to build datasets for each ISA is labor-intensive and time-consuming.
To reduce the burden of data collection, we propose to leverage the ideas and techniques in Neural Machine Translation (NMT) for malware detection. Specifically, when dealing with malware in a certain ISA, we translate it to an ISA with sufficient malware samples (such as X86-64). This allows us to apply a model trained on one ISA to analyze malware from another ISA. Our approach reduces the data collection effort by enabling malware detection across multiple ISAs using a model trained on a single ISA. We have implemented and evaluated the model on six ISAs, including X86-64, i386, ARM64, ARM32, MIPS32, and s390x. The results demonstrate its high translation capability, thereby enabling superior malware detection across ISAs.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes MalTrans, a malware detection model that leverages translations to parse rarely-available programs in specific ISAs.
In particular, MalTrans is firstly trained with self-supervised learning to translate from one ISA (like x86) to another one (ARM32). This is done by compiling several open-source programs to different targets (x86, ARM, etc), and train a transformer model to translate them correctly. Once this is done, programs are then ported to an ISA (like x86) which is very common, and on which it is easier to build a malware detector. The paper presents different ablation studies on how to setup the components of MalTrans, while achieving better results in terms of translation and detection rate.

### Strengths
**Translating between ISAs is good advancement.** Since there are a lot of devices also with "esoteric" ISAs, this tool can really enable a good first detection on malware targeting IoT.

**Interesting embedding analysis.** It is interesting to see that instructions cluster together depending on their functionality.

**Good experimental results and ablation study.** It is very important that models are followed by extensive ablation study, to show how their performance changes and why, and this paper provides rightfully.

### Weaknesses
**Motivation is weak.** While the authors state that IoT malware are spreading more than anything, I would like to ask them detailed reports on this statement. We know that there is an on-going issue with Windows malware (https://www.virustotal.com/gui/stats by looking at the file types) and Android malware. But IoT is a blurred terminology, and I fail to understand this alarming threat.
Hence, the motivation that we need to detect malware for IoT does not hold that much.

**Missing details on how BLEU is computed.** I imagine that the BLUE is computed on the sequence to match, but it is not really clear how it was used. Hence, I believe that the authors computed the BLEU between the predicted sequence and the ground-truth created from the available source code. Is this correct? Also, it was computed on what? Basic-block level, single-instruction level? The authors must specify these details.

**i386 is x86 (32bit).** I found confusing to read the translation from i386 to x86, and I also found very confusing the fact that the BLEU score from i386 to x86 is **only** 0.4 (which means that the sequences match only by half). The 32bit version of x86 is i386. Were the authors thinking about translating to x64 (64 bit) from i386? Also, in the paper there is written x86-64 or x86. The authors should specify whether they target the 32- or 64bit ISA.
Also, while it is possible to translate from 32bit to 64bit, it does not make sense to do the opposite.

**Missing examples.** Can the authors submit examples of their translations? This might help understand better the quality of the translation.

**Maybe submit to a security venue?** While the proposal is interesting, the content might better fit another venue closer to the security community. This is a minor comment, not considered during the review process.

### Questions
1. How the authors computed the BLEU score for their experiments?
2. Which specific ISAs are the authors targeting with their experiments?
3. Is it true that IoT malware are so numerous to outmatch all other domains (like Windows and Android)?

### Soundness
3

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
This paper proposes an unsupervised binary code translation framework called MALTRANS,
designed to facilitate malware detection across multiple Instruction Set Architectures (ISAs).
It addresses the challenge of labeled data availability by leveraging neural machine translation (NMT).
The study focuses on overcoming the limitations posed by insufficient malware datasets across various ISAs.

The proposed system, MALTRANS, translates a given source ISA to a target ISA,
such as x86-64. It features a shared encoder with separate decoders for each ISA.
The paper reports that MALTRANS achieves strong performance across five different ISAs: x86-64, i386, ARM32, ARM64, and s390x.

This paper presents an interesting approach to cross-ISA malware detection,
with clear performance improvements over prior work and a broader scope by covering multiple architectures,
including some less commonly explored. The unsupervised learning framework,
which reduces dependency on labeled malware datasets, is a practical strength.
The thorough evaluation further demonstrates the system’s robustness.

The reviewer finds the paper sound and notes that it offers clear advantages over prior work.
However, several aspects warrant further consideration before acceptance. Detailed feedback is provided below.

Additionally, the discussion of real-world challenges, such as scalability and the handling of packed or obfuscated malware, could be expanded to provide a more comprehensive perspective. Strengthening the differentiation from previous approaches would also help better position the paper within the existing research landscape.

### Strengths
1. Practical Relevance:

The reviewer considers the paper's application across multiple ISAs, including i386, ARM32, ARM64, s390x, and X86-64,
to be a significant strength. This breadth ensures the framework’s practical relevance,
enabling deployment across diverse platforms and making an important contribution to cross-architecture malware detection.

2. Unsupervised Learning to reduce labelled data collection burden:

The reviewer believes that the use of unsupervised learning to eliminate
the need for labeled malware datasets is a meaningful contribution in the cybersecurity domain.

3. Improved Performance:

The paper reports strong empirical results, including higher BLEU and AUC scores compared to
prior baselines such as UnsuperBinTrans, to reflect the system's reliability.
The reviewer finds the improvements promising and the results suggest that the proposed system preserves code semantics across ISAs effectively,
leading to improved malware detection capabilities.

### Weaknesses
1. Prior work:

- The paper lacks coverage of several related prior works. While citing research from 1990, 2000, and 2021 is commendable,
there are more recent publications in this area. A few examples are provided in the related references below,
covering both static and dynamic translation.

- The reviewer emphasizes that ML-based code translation is an emerging but not entirely
new area of research. The comparison between the current
work and previous studies (e.g., works cited as [7], [9], [11], [12]) is
essential to position this research within the broader cybersecurity field.
It is crucial not to ignore or overlook prior work published in top security conferences.


2. UnsuperBinTrans:

The paper mentions that ``UnsuperBinTrans" is the first and only existing work. This is actually related to the prior comments.
However, without comparing or contrasting this system with other relevant studies, it is difficult to justify such a statement.

3. BPE merge times:

- How practical are the two principles utilized for BPE merge times?
- Would vocabulary size discrepancies not persist in practice? Or is the reviewer missing something?
- The paper recommends that each ISA’s vocabulary size be <12K. Is there any supporting evidence for this recommendation?

4. Model details, Training, and Testing

- The reviewer suggests including more details about the model architecture and hyperparameters used.
- What are the final evaluation losses for training across different ISAs?
- How many parameters does the model have, given that training takes approximately one day?
- Why was 4-gram precision selected? Why not explore higher precision levels?


5. Translation quality

- The translation quality seems suboptimal, with the highest BLEU score being 0.40.
- The reviewer feels the paper relies too heavily on BLEU as the sole metric.
- A comparison with the translation quality reported in other works, such as [11] and [12], would provide valuable insights.

6. Malware detection

- The model description lacks sufficient detail. The reviewer recommends including more information on the model, data preprocessing, hyperparameters, and other technical aspects.
- An error range for the reported results would enhance the clarity and reliability of the findings.

7. Discussion

- The reviewer suggests to discuss the scalability issue of the proposed system and how the system can be
 adapted to handle the packed or obfuscated malware.



Related references:

A. Static and Dynamic Translation

[1] Di Federico, Alessandro, and Giovanni Agosta. "A jump-target identification method for multi-architecture static binary translation." Proceedings of the International Conference on Compilers, Architectures and Synthesis for Embedded Systems. 2016.

[2] Gouicem, Redha, et al. "Risotto: a dynamic binary translator for weak memory model architectures." Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1. 2022.

[3] You, Yi-Ping, Tsung-Chun Lin, and Wuu Yang. "Translating AArch64 floating-point instruction set to the x86-64 platform." Workshop Proceedings of the 48th International Conference on Parallel Processing. 2019.

[4] Chen, Jiunn-Yeu, et al. "On static binary translation of arm/thumb mixed isa binaries." ACM Transactions on Embedded Computing Systems (TECS) 16.3 (2017): 1-25.

[5] Jiang, Jinhu, et al. "A System-Level Dynamic Binary Translator Using Automatically-Learned Translation Rules." 2024 IEEE/ACM International Symposium on Code Generation and Optimization (CGO). IEEE, 2024.

[6] Xie, Benyi, et al. "An Instruction Inflation Analyzing Framework for Dynamic Binary Translators." ACM Transactions on Architecture and Code Optimization 21.2 (2024): 1-25.


B. ML and Code Translation

[7] Wang, Junzhe, et al. "Can a Deep Learning Model for One Architecture Be Used for Others? Retargeted-Architecture Binary Code Analysis." USENIX Security 2023.

[8] Ma, Xiaoyue, Lannan Luo, and Qiang Zeng. "From One Thousand Pages of Specification to Unveiling Hidden Bugs: Large Language Model Assisted Fuzzing of Matter IoT Devices." USENIX Security 2024.

[9] Wang, Junzhe, Qiang Zeng, and Lannan Luo. "Learning Cross-Architecture Instruction Embeddings for Binary Code Analysis in Low-Resource Architectures." NAACL 2024.

[10] Xie, Danning, et al. "ReSym: Harnessing LLMs to Recover Variable and Data Structure Symbols from Stripped Binaries." CCS 2024.

[11] Zuo, Fei, et al. "Neural Machine Translation Inspired Binary Code Similarity Comparison beyond Function Pairs." NDSS 2019.

[12] Roziere, Baptiste, et al. "Unsupervised translation of programming languages." Advances in neural information processing systems 33 (2020): 20601-20611.

C. Malware detection

[13] Anderson, Hyrum S., and Phil Roth. "EMBER: an open dataset for training static pe malware machine learning models." arXiv preprint arXiv:1804.04637 (2018).

### Questions
1. How does this work compare and contrast with the related works mentioned ([7], [9], [11], [12])?
2. How can the paper justify the statement that UnsuperBinTrans is the first and only existing work without comparing or contrasting with other relevant works mentioned in the relevant references?
3. How practical are the two principles utilized for BPE merge times?
4. In reality, would a vocabulary size discrepancy exist unless the reviewer is missing something?
5. Is there supporting evidence to back the recommendation that "The vocabulary size of each ISA is recommended to be <12K"?
6. What are the final evaluation losses for the training of each ISA?
7. How many parameters does the model have if the training takes around one day?
8. Why did the paper utilize 4-gram precision? Why not higher than 4-gram precision?
9. How does the translation quality compare with other works such as [11] and [12]?
10. Should the reported results include an error range?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces an unsupervised binary code translation model (MalTrans) designed for malware detection across multiple ISAs. The main challenge tackled in this work is the unavailability of large data samples from ISAs such as i386, ARM64, ARM32, and s390x. This approach also operates without needing labeled malware samples for code translation. Experimental results show that after code translations, for an LSTM malware detection model, this archives AUC scores of 0.999, 0.986, 0.913, and 0.938 for i386, ARM32, ARM64, and s390x, respectively. This outperforms the two baselines (UnsuperBinTrans, IR-based Model).

### Strengths
1. Novelty: 

a. Uses a transformer-based model compared to the recent work using RNNs (UnsuperBinTrans)
b. Uses 3 different instruction normalization techniques (R1, R2, R3)
c. Experiments were conducted using multiple ISAs compared to previous work

2. The proposed approach achieves relatively better  BLEU scores and retains high AUC values in malware detection across different ISAs.

3. The main outcome of this work is the utilization of knowledge transfer across ISAs for malware detection: Doing code translation from ISA-2 to ISA-2. Have a trained model on ISA-2 with more data. Test for ISA-2 using the translated code for malware detection. 

4. The paper is well written with minimal mistakes. The structure is good.

### Weaknesses
1. Malware detection is a downstream task that the paper uses to show the evaluations. The paper does not do any innovation concerning malware detection. Similar to malware detection, any other task can be used here. The main innovation here seems to be a better translation mechanism of code compared to UnsuperBinTrans.

2. Among the contributions claimed in the paper, similar contributions were made in UnsuperBinTrans. The novelty is the model architecture, normalization scheme, and availability for more ISAs.

3. Assessment for s390x does not seem to be fair since there are only 2 malware samples (line 410).

4. It seems that the testing data sets are not consistent between Table 3 (a) and (b) for ARM65 and s390x. So the accuracies obtained cannot be directly compared.

### Questions
1. In Table 3 (b), it seems there results are for LSTM models with non-translated data. For example, according to the results, it is possible to get up to 0.999 with as little as 265 data samples for i386. 0.988 for ARM32 with 196 data samples. Does this contradict the initial claim of the paper? Do we actually need to go through a complex code translation process when we can get good results with 265 data points?

2. It would be interesting to see BLEU scores for MALTRANS models with smaller datasets. What would be the impact on the accuracy if few-shot learning is done? What should be the ideal ratio of data for 2 ISAs? Also, can we get better BELU scores if we have more data samples? Are these BELU scores for 1-gram? 

3. It would be better if it is possible to provide numbers for UnsuperBinTrans for other ISAs although it only focused on ARM32→X86. It should be trivial and a good comparison to do. Any possibility?

4. Why the accuracy results are better for ARM32→X86 although the BELU score is higher for ARM64→X86? Any explanation for that? 

5. It would be ideal to compare the performance of malware detection using classical machine learning techniques as a comparison with LSTM. Can that be done?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work proposes an architecture that utilizes an unsupervised binary code translation model called
MALTRANS to translate binaries from four different Instruction Set Architectures (ISAs) into a target ISA
(X86-64). A deep learning model trained exclusively on the target ISA is then employed for malware
detection.

### Strengths
**Highly relevant topic.** With the growing number of devices targeted by malicious software (malware),
especially within IoT devices, which come in a wide variety of architectures, the need for cross-ISA malware
detection systems is increasingly critical. This study addresses this need by proposing a detection system
capable of handling multiple ISAs.

**Novelty.** The subject is relatively novel, with the current state of the art lacking extensive work on cross-ISA
translation. Additionally, the proposed work could significantly contribute to the development of efficient
cross-platform malware detection systems.

**Good number of analyzed ISAs.** This study covers a good number of ISAs relevant to todayʼs market,
particularly focusing on Linux environments and IoT devices.

### Weaknesses
**Small malware dataset.** The malware detection dataset used by the authors is quite limited. Specifically,
for the X86-64 architecture analysis, only 1,238 malware samples were used, divided into an 80/20 traintest
split, which is insufficient given the prevalence of this architecture. Additionally, while the focus is on
the Linux and IoT landscape, the ARM architecture datasets were also notably small, despite the broader
sample availability in the literature. Given the importance of the topic, it would be recommended to conduct
experiments on a more extensive dataset to ensure more reliable results.

**Unclear description.** MALTRANS is introduced as an ISA-to-ISA translation tool and presented as such in
Figure 1, but later, it is also described as capable of detection. Clarification is needed to delineate its
functions. Does MALTRANS perform solely translation or detection as well? Better differentiation between
the two modules would enhance clarity.
Figure 2 shows the MALTRANS architecture; however, the flow is not entirely clear. For example, the text
mentions inputs like B'_src, but in the figure, B_src is shown instead of its noisy version. During stages (3
and 4), outputs B_tgt^ and B_src^ are produced, but the workflow suggests they are not subsequently
used, which conflicts with the description stating these outputs are fed into the model to predict the original
block.
In the paragraph "Comparison with Optimal Model," the dataset usage is unclear. Training and testing on as
few as 58 or even 2 samples divided 80/20 is suboptimal. Providing a clear dataset usage methodology
before discussing the results of the optimal model would help.


**Experiments performed on a single deep learning model.** While the chosen model is acceptable, there
are several state-of-the-art malware detection models for X86-64 that could be considered. The use of
pre-trained models in the detection module could yield a fairer comparison.
The acronyms for BLEU, AUC, and F1-score are not expanded. Although they are widely recognized, it
would be helpful to describe each acronym and provide brief descriptions.

**Typographical error (minor issue).** In section 4.2, the title "MODE ARCHITECTURE" likely should be "MODEL
ARCHITECTURE."

### Questions
Please comment on the main weaknesses:

- Experiments performed using a single deep learning model;
- Absence of the MIPS ISA;
- Limited malware dataset;
- Unclear methodological description.

### Soundness
2

### Presentation
2

### Contribution
3
