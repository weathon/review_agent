# The KoLMogorov Test: Compression by Code Generation

- Decision: Accept (Poster)
- Scores: 6, 5, 6, 10, 8, 5, 8

## Abstract
Compression is at the heart of intelligence. A theoretically optimal way to compress any sequence of data is to find the shortest program that outputs that sequence and then halts. However, such Kolmogorov compression is uncomputable, and code generating LLMs struggle to approximate this theoretical ideal, as it requires reasoning, planning and search capabilities beyond those of current models. In this work, we introduce the *KoLMogorov-Test* (KT), a compression-as-intelligence intelligence test for code generation LLMs. In KT a model is presented with a sequence of data at inference time, and asked to generate the shortest program that produces the sequence. We identify several benefits of KT for both evaluation and training: an essentially infinite number of problem instances of varying difficulty is readily available, strong baselines already exist, the evaluation metric (compression) cannot be gamed, and pretraining data contamination is highly unlikely. To evaluate current models, we use audio, text, and DNA data, as well as sequences produced by random synthetic programs. Current flagship models perform poorly - both GPT4-o and  Llama-3.1-405B struggle on our natural and synthetic sequences. On our synthetic distribution, we are able to train code generation models with lower compression rates than previous approaches. Moreover, we show that gains on synthetic data generalize poorly to real data, suggesting that new innovations are necessary for additional gains on KT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel aspect to evaluate the LLM: by testing its ability to compress discrete sequence data. The proposed benchmark features several advantages such as unlimited data availability and easy to control difficulty in synthetic data generation. The proposed benchmark can be viewed as a sub-task of code-generation. This paper further trained a few models to outperform current models in this task to explore LLM's edge in data compression.

### Strengths
- The studied problem is interesting and challenging for LLMs, and not seriously studied before.

- The proposed benchmark is described in detail, making it easy to understand and reproduce.

- The efforts to train a few LLMs proved that this problem is addressable, which offers more context and helps future researches.

### Weaknesses
- Better to provide a few examples on where the LLM's ability starts to drop: for tasks than that threshold, LLMs are able to produce perfect compression programs (and separately for each studied and fine tuned LLMs); for tasks harder than that, LLMs fails even with fine-tuning.

- Better study on the edge of LLM: if more training data are provided, if higher rank LoRA applied, or if better curriculum learning applied, will it be stronger, or stops at current level?

- I believe the LLM could perform better if more domain-specific presets of data are assembled, such as DNS sub-sequences with known patterns and known functionalities, yet this seems to be beyond the scope of this paper.

### Questions
See sections above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work presents a Kolmogorov test for code generation models.

Experiments are performed on synthetic data, natural text from Wikipedia, audio sequences from LibriSpeech, and human DNA sequences from the GRCh38 assembly.

Standard compression is compared with LLMs and code generation. Specifically, Gzip and LMiC are used as baselines, and compared with baselines that prompt open-weights (Llama-3.1) and closed-weights (GPT-4o) LLMs to generate the shortest Python program that produces the input, and also open-weight models trained on synthetic text and code.

### Strengths
1. The result of the open-weight models trained on synthetic text and code is impressive (84.5% accuracy),
and their compression rate outperforms existing baselines.

2. Compression by code generation may serve as a useful additional scalable benchmark for LLMs.

3. The paper is well-written, and the exposition is clear.

### Weaknesses
1. My main concern is that the results do not generalize at all from synthetic to real datasets (0 accuracy).

2. In addition to synthetic data, the data used in this work includes natural text, audio, and human DNA sequences from the GRCh38 assembly represented in FASTA format. In protein structure prediction (PSP), sequences of proteins are commonly represented in FASTA format, and their 3D structure predicted. It would be interesting to see the compression results for protein sequences and their structures.
It would be interesting to extend the real-world datasets to other data types.

3. The synthetic DSL includes 3 initiators x 7 modifies x 2 filters x 3 merger functions. It would be interesting to see this action space extended and understand how sensitive the results are as a function of the action space.

### Questions
How sensitive are the results as a function of the action space?

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
5

### Summary
The papers proposes a method to evaluate code based language models by using them a compression mechanism to approximately compute the Kolmorogov Complexity (which is uncomputable). The aim of the evaluation procedure is to generate the shortest program which produces a certain data point and then stops. The paper proposes simple experiments to show how to approximately compute the Kolmogorov complexity of datums from LLMs. Experiments in the paper use the state of art closed and open source language models, and show that even the SOTA models struggle on the Kolmogorov test proposed in the paper.

### Strengths
- The proposed test is an interesting information theoretic test which can be used to compute the information in data samples relative to a model. If samples have smaller programs, then we know that the LLM has understood the structure of that problem/sequence.
- Table 1 is an interesting figure as it shows that even SoTA models currently struggle on at least some benchmarks like Kolmogorov-Test proposed in the paper.
- The experiments in the paper are intuitive, seem to be easy to follow and replicate.
- Test proposed in the paper is different from the usual benchmarks and can be added to any existing code evaluation pipeline by simply computing the size of the program which generates the desired output from the test cases.

### Weaknesses
- The connection between Kolmogorov Complexity and Intelligence is not clear in the submission. If we take the popular code based benchmarks, and compute the size of the programs lets say at pass @ 10, then can we say that models which produce shorter programs are more intelligent?
- When using LLMs on the application side, users are often interested in easy to use/understand programs, will producing shorter programs be useful in this case?
- For the synthetic programs that are constructed as part of the dataset, is there a short proof for why they are the shortest length? Are you using some of the properties of the operations to assure that?
    - Here is an example where the training set may not contain the shortest program, for instance consider the following program:
        - seq_1 = range(0,10,2) seq_2 = range(1,11,2) seq_3 = interleave(seq_1,seq_2) output=seq_3 In this case the shorted program is range(0,11,1) and not the program in the training set. 
- Generating the shortest program for any given sequence is NP-Complete so what does line 246 correspond to? Are we trusting the LLM to generate the shortest programs? The baseline seems to be unreliable.
- In terms of the paper writing, it would certainly help if you define the Kolmogorov test or present it as an algorithm. While it’s understandable for people with background, it could be hard to understand for general audience.
- For the Lams in Table 1 especially the Llama models, why was few shot prompt / chain of thoughts reasoning not considered as good baselines? For the synthetic tasks, I believe using CoT with few-shot examples could certainly improve performance. It also seems that the SeqCoder-1.5B model is overfitting to the synthetic task instead of actually being intelligent. Would the claim here be that the 1.5B model is more intelligent than GPT-4o since it has a higher accuracy?

### Questions
- Are the Llama models instruction tuned or pre-trained?
- What’s the compression rate for the training data? How many bits can it be compressed to?
- The proposed method seems to be very much model dependent, even on the size of the model? Do you add the size of the model while using your test, so that you can normalize your method across models? Otherwise larger models can very well compress all short programs effectively, thus it would then be a test of how large the LLM is.
- In table 1 what could be the reason for the Llama models to have a low accuracy for 8-bit prediction, while having almost double accuracy for 16-bit case?
- A lot of the analysis seems to be heavily focused on Audio datasets/tasks, is there any particular reason for that? Instead of using existing code based benchmarks?
- Section 1 of this paper by Vitanyi (https://arxiv.org/pdf/0809.2754) has some interesting results on the Kolmogorov Complexity of different objects like, very simple objects, complex objects, random objects etc. I think some inspiration could be used from there to construct interesting examples for evaluating the LLMs, for instances showing if the existing SOTA LLMs can encode simple objects, and then progressively build on complex objects.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The authors introduce the KOLMOGOROV-TEST (KT), a novel "compression-as-intelligence" test for code generation in large language models (LLMs). Using KT we can prompt or train LLMs to generate the shortest possible programs that reproduce given data sequences, emulating an ideal form of data compression known as Kolmogorov compression, which is theoretically optimal but incomputable priori. The contribution has several advantages: it provides a myriad supply of problem instances of varying difficulty, uses a robust metric (compression rate), and minimizes risks of pretraining data contamination. In evaluating current models like GPT-4o Omni and LLAMA-3.1-405B across multi-modal data types (audio, text, DNA, and synthetic sequences), the authors find that these models perform poorly on both natural and synthetic data. Also authors showed that while training on synthetic data shows improved compression rates, the gains do not transfer well to real-world data for KT.

### Strengths
**Originality**:  
The authors present an innovative approach by introducing a "compression-as-intelligence" test (KOLMOGOROV-TEST, or KT) that challenges LLMs to generate the shortest programs across multiple data modalities, a novel and ambitious application of LLMs in program synthesis. The use of a multi-modal framework (including audio, text, DNA, and synthetic data) is particularly original, expanding the scope of LLMs in code generation.

**Quality**:  
The authors collected program-sequence pairs for supervised training and evaluation, developing a compositional domain-specific language (DSL) and an automatic data generation framework. This framework enables consistent and reliable data generation while avoiding potential biases through uniform priors, which strengthens the robustness of the study. Additionally, the authors investigated the impact of execution feedback on performance, further enhancing the study's depth.

**Clarity**:  
The paper includes a detailed analysis section, providing insights into model performance and limitations. The authors clearly document failure cases, particularly highlighting instances where models fail by attempting to reverse-engineer the data generation process rather than compress it effectively. This transparency improves the readability and interpretability of the results.

**Significance**:  
KT sets a new benchmark for evaluating LLMs on program synthesis via compression tasks, offering a reliable metric (compression rate) that is both challenging and robust. The framework holds potential for broader applications and improvements in LLM-based code generation by identifying key areas for enhancement, such as in handling complex, real-world data compression.

### Weaknesses
**Limited Focus on Code-Specific LLMs**:  
Although the paper introduces a compression benchmark specifically for code-generating language models (CODE LMs), the experiments primarily use general-purpose LLMs. Testing specialized CODE LMs could offer a more accurate assessment of model performance for program synthesis tasks, as these models are often fine-tuned with code-specific vocabularies and architectures. Incorporating CODE LMs in future work would enhance the relevance and impact of the findings.

**Understudied Tokenization Impact**:  
The impact of tokenization on model performance is underexplored. Prior work (e.g., Delétang et al.) suggests that larger vocabularies may offer more expressive flexibility but can also complicate next-token prediction. Evaluating different tokenization strategies would shed light on their influence on compression rates and model accuracy, potentially guiding optimizations in vocabulary selection to balance expressiveness and predictive ease.

**Suggestions to test Length Generalization**:  
Scaling to longer sequences remains an open question. Applying techniques like Rotary Position Embeddings (ROPE) for scaling might provide insights into the model's capacity for length generalization. Additional experiments with longer sequences could clarify how well the model generalizes across variable input sizes, which is critical for practical applications of program synthesis.

**Limited Prompt Optimization**:  
The paper could benefit from more work on prompt optimization, as prompt quality can significantly impact model output. Experimenting with prompt tuning strategies or context-enhancing techniques could improve model performance and compression efficiency. This would help achieve the paper’s goal of generating shorter, more precise programs, as optimized prompts can guide the model more effectively toward compact solutions.

### Questions
1. **Live Benchmark Availability**:  
   Are there any plans to release KT as a live benchmark in addition to the dataset? A live benchmark could provide ongoing insights into model performance and facilitate community-driven improvements over time.

2. **Use of Code-Specific LLMs**:  
   Why weren’t code-specific LLMs, such as CodeLLAMA, DeepSeek Coder, or StarCoder, tested in this work? These models have vocabularies optimized for code, which may affect compression rates and generalization performance. Testing code-specific LLMs could provide better priors for transfer learning and potentially enhance performance on program generation tasks.

3. **Training Details**:  
   How many epochs or steps were the models trained on for this task? Understanding the training duration would clarify whether models had sufficient exposure to the data to learn effective compression strategies.

4. **Sequence Length and Token Count**:  
   The input sequence lengths vary from 16 to 1024 bytes, but could you clarify the number of tokens these represent? This information would help assess how sequence length impacts model performance and how the model tokenizes different sequence lengths.

5. **Data Type Configuration for LLAMA-3.1-405B**:  
   Which data type was used for LLAMA-3.1-405B (e.g., bf16, fp16, fp8)? Were any specific data types recommended by the model’s authors tested to ensure compatibility with the architecture? Correct data type configurations can be critical for performance, and confirming this would add clarity to the experimental setup.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents two contributions to the field of language model (LM) evaluation and training:

* KoLMogorov Test: A proposed test for language models grounded in compression principles with several benefits. Specifically, there is a preponderance of real-world examples (e.g. audio, text, DNA, or any other modality data that can be represented as bit sequence ) that can used to curate an evaluation set of desired degrees of hardness that cannot be otherwise used for fine-tuning LMs - preventing benchmark hacking. Also, this method allows for computationally cheap and reliable automated evaluation that can measure both the generation accuracy and quality, which is traditionally a bottleneck in language model evaluation setups.
* Synthetic instruction set for fine-tuning LMs for the compression task: This work also turns this test into an LM post-training task and provides a framework for generating a synthetic instruction set to enhance LMs' compression capabilities.

Additionally, this work provides interesting insights on LM fine-tuning:
1. Specifically, small LMs extensively fine-tuned on a synthetic instruction set outperform state-of-the-art prompted models and classical baselines (gzip algorithm).
2. that such performance gains by small fine-tuned models on synthetic data do not translate to the performance gains on real-world data

### Strengths
The authors have considered datasets of different modalities and degrees of hardness and analyzed state-of-the-art models from both open and closed-source spaces, making a convincing case for KT as a test for code-generation models. This test has the additional benefits of the corresponding evaluation sets not being directly usable for fine-tuning and the preponderance of hard examples in natural data sources.

The various ablations presented in this work comprehensively capture the limited performance of these SOTA LMs on the task of compression, which will be helpful to the broader community.

The framework for synthetic data generation proposed in this work could be used to augment instruction sets towards large-scale post-trainings of LMs aimed at generalized gains.

### Weaknesses
Two points that are fundamental to this work -
1. the efficacy of code-generation language models at compression tasks and 
2. some of the observations around fine-tuning LM on compression tasks 

have already been discussed in prior work, as the authors of this work aptly reference. This limits the novelty of the present work.

### Questions
Since compression is posited as a test of [coding] LM intelligence, it will be interesting to examine the other dimensions of an LM's "intelligence" that correlate with its compression efficacy. A potential experimental set-up to this end could involve fixing a base model and measuring the performance of its variants which have been fine-tuned on disparate tasks such as math reasoning, code, etc., on the KoLMogorov Test. If the KoLMogorov Test exhibits a good correlation with the other intelligence dimensions, its benefits may allow it to subsume other benchmarks. KT may also lend itself as a reward function during LM preference optimizations.

The observations in this work when fine-tuning small LMs on synthetic instruction sets for compression tasks - appear to echo other works from the math-reasoning domain, where extensive fine-tuning allows a relatively smaller model to outperform a larger base model (on tasks such as GSM8k), but without any generalized improvements. If the authors see any fine-grained parallels here - this discussion would be a useful contribution to the community.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 6

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper talks about compression of data using LLMs via code generation. The smallest possible way to compress a sequence of data called the Kolmogorov compression is uncomputable. Therefore, the paper tests the capability of code generating LLMs to generate a program that can produce the sequence, thus compressing it. They argue that achieving this requires the models to reason, plan, and search for complex patterns within the sequence to enhance compression performance. The paper evaluates state-of-the-art models, such as Llama-3.1-405b and GPT-4-o, in a zero-shot manner on real-world compression data sources like audio, text, and DNA, demonstrating that these models perform poorly compared to the deterministic gzip baseline. Additionally, the authors assess the models on a synthetic benchmark of (sequence, program) pairs created using a custom domain-specific language (DSL). On this synthetic distribution, they trained a code generation model that outperformed gzip with a uniform prior over the custom DSL. However, they also showed that these models don’t generalize to real data. The paper highlights significant differences between the distributions of real and synthetic sequences, with real sequences exhibiting more repetitions. Through ablation studies, the authors further demonstrate that the trained models do not generalize well to longer sequences.

### Strengths
The main strengths of the paper include :

- The paper studies the problem of compression as a code generation problem with current code language models. This is quite novel in the context of language models as previous works have primarily used language models for compression through arithmetic coding. This new framing of the task surely requires the model to understand patterns in the sequence and reason about how to segment the sequence into different subsequences.

- The paper proposes a dynamic benchmark comprising of (sequence, program) pairs which test model’s capability at compressing sequences which are generated using a custom domain specific language (DSL) comprising of different functions that can be realized as python functions. The paper shows that codeLMs trained on this synthetic benchmark can outperform deterministic compression methods like gzip when efficient compression priors are applied to generated functions. Notably, zero-shot prompted state-of-the-art models like GPT-4-o and Llama-3.1-504b fail to reproduce input sequence for more than 66% and 45% of the time, respectively. The dynamic nature of the benchmark, due to its dependency on the customizable DSL, prevents existing LLMs from gaming or memorizing it, as it can generate different examples of varying complexity that the model has not previously encountered during training. 

- The paper’s evaluation of code generation based compression on various real world data sources like audio, text and DNA is also noteworthy. In the literature of information theory and communication, these domains have been thoroughly studied for different compression based algorithms. Although none of the models discussed in the paper outperformed the gzip baseline for compression, this finding highlights an interesting phenomenon that is widely discussed in the literature: the numerical reasoning and counting limitations of these LLMs.

### Weaknesses
The major weaknesses of the paper include :

- Although the benchmark is effectively designed around compression as a code generation problem, the prompts used in the zero shot evaluation of the models are quite open-ended compared to the structured and closed nature of the DSL design that has been used in the synthetic experiments. This leads to an unfair comparison with the synthetic model.  With better prompt-designs, with better in-context examples, inclusion of DSL definitions in the prompt, ReACT prompts with verification in the loop, these SOTA models can achieve good performance. This is particularly relevant given that feedback and execution improved the accuracy of the trained model on audio-based data, and the majority of errors in SOTA models were execution-based.

- While the assessment using real-world data is sensible, expecting the smaller 1.5B/8B model trained on synthetic data to generalize effectively to real-world data is quite ambitious. The distribution of real data significantly differs from synthetic data; the former has more repetitions, while the latter exhibits more complexity in terms of interleaving sequences. The synthetic data used by the authors consists of smaller sequences and smaller length programs whereas real-world data with more repetitions would result in longer generated programs - something the model wasn't trained on, leading to a significant drop in accuracy. This decrease in accuracy for the trained synthetic model could also be attributed to model capacity, as similar models like LLaMa-3.1-8B also show lower accuracy on real-world data.

- Beyond the evaluation and training of models, it is highly unlikely that LLMs can be traditionally and reliably used for compression instead of deterministic methods like gzip. LLMs are inherently probabilistic models and cannot generate lossless compressions without hallucinations or mistakes in a zero-shot setup. The motivation behind designing such a complex benchmark around the Kolmogorov test is unclear. If the focus is on synthetically generated data/benchmark, small 1.5B models trained on 10k-1M perform well on them with some supervised training. However, if the focus is on real data, realizing real-world data compression as a code generation problem that outperforms gzip is very challenging without better base models, improved prompt design, or better synthetic data design.

### Questions
For the experiments, the following should have been additionally explored by the authors :

- The authors didn’t report the effect of execution and feedback on the SOTA LLMs. From Table 3, the authors have just presented the ablation on the trained synthetic model.
- The authors didn’t use models like openai-o1 which can reason before emitting any code and have shown to have good performance on code based tasks. It would be interesting how these inference based reasoning models perform on this task ? 
- The authors haven’t reported the accuracy and precision of the Seqcoder 8B 1M model on synthetic data. Does starting from a better base model help in accuracy ?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 7

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a benchmark designed to evaluate LLM's capabilities in compression as an indicator of intelligence. The Benchmark requires models to generate the shortest possible python program that reproduces an input data sequence, aligning with the concept of Kolmogorov complexity. The authors assess current language models' performance in KT using data from text, audio, and DNA, as well as synthetic data with specific patterns. Results show that models like GPT4-O and LLAMA-3.1-405B struggle with both natural and synthetic data. The authors also develop a Domain Specific Language (DSL) to create program-sequence pairs for supervised training, leading to trained models that achieve lower compression rates than traditional methods on synthetic data but perform poorly on real data. Authors have conducted further experiments and comparisons with different baselines including GZIP, "Language Model is Compression", etc. The paper suggests that further innovations are necessary for models to generalize well on real-world data within the KT framework.

### Strengths
1. Extensive amounts of experiments conducted and insights gathered from it are the major strengths of this paper. Experiments related to trained SeqCoder are  particularly helpful in highlighting the generalization issues and how much it could be overcome with synthetic data.
2. Clear articulation of the DSL Language and of the techniques used in generating the data. The DSL Language and its complexity seems sufficient for benchmarking SOTA LLMs.
3. Easy to read, with key pieces of information and additional analysis generally emphasized appropriately.
4. Setting aside the question of whether LLMs will be widely used to generate code to compress data, this benchmark highlights current limitations of SOTA LLMs, even with basic patterns such as repetition.

### Weaknesses
1. Though the breadth of the experiments is extensive, the Paper could have included experiments and analysis of popular prompting techniques that have been shown to increase the reasoning and coding capabilities of LLMs, such as Chain-of-Thought, Tree-of-Thoughts etc. Given the popularity of these techniques and how prevalent CoT is in all major reasoning benchmarks, including at least one prompting technique would have been useful in the analysis.

2. As the paper's main idea is influenced by the Hutter Prize [1], it would be useful to use any of the compressors in the leaderboard [2] as an additional baseline. Specifically, compressors that use LSTM (such as tensorflow-compress) would make make of an interesting baseline for the benchmark.

[1] - http://prize.hutter1.net

[2] - http://mattmahoney.net/dc/text.html

### Questions
See weaknesses for questions

Also, does any of the recent works on lifting the length constraint of LLMs help in Length generalization for SeqCoder? e.g. "Efficient Streaming Language Models with Attention Sinks".

### Soundness
4

### Presentation
3

### Contribution
2
