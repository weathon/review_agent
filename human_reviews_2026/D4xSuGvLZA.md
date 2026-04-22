# BriLLM: Brain-inspired Large Language Model

- Avg Score: 2.00
- Decision: Reject
- Scores: 0, 2, 2, 4

## Abstract
We introduce BriLLM, the first brain-inspired large language model that establishes a genuinely biology- and neuroscience-grounded machine learning paradigm. Unlike previous approaches that primarily mimic local neural features, BriLLM implements Signal Fully-connected flowing (SiFu) learning—the first framework to authentically replicate the brain's macroscopic information processing principles at scale. Our approach is uniquely validated by two core neurocognitive facts: (1) _static semantic mapping_ to dedicated cortical regions, and (2) _dynamic signal propagation_ through electrophysiological activity. This foundation enables transformative capabilities: inherent multi-modal compatibility, full node-level interpretability, context-length independent scaling, and global-scale simulation of brain-like language processing. Our 1–2B parameter models demonstrate stable learning dynamics while replicating GPT-1-level generative performance. Scalability analysis confirms feasibility of 100–200B parameter variants. BriLLM represents a paradigm shift from representation learning toward biologically-validated AGI foundations, offering a principled solution to current AI's fundamental limitations.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper introduces BriLLM, a brain-inspired neural architecture for language modeling. The architecture is a fully-connected graph where vector messages are passed in a bidirectional manner. The authors show that the loss decreases over training on a language modeling task.

### Strengths
The topic of submission (brain-inspired language modeling) is highly aligned to the conference.

### Weaknesses
This manuscript is not ready for publication at ICLR. Unfortunately, the neuroscientific motivation, grounding in past literature, theory, and experiments are preliminary. In particular, BriLLM is not the first brain-inspired architecture out there-- the submission mentions spiking neural networks but otherwise ignores a vibrant line of recent work including, e.g., TopoLM [1], Topoformer [2], Mixture of Cognitive Reasoners [3] etc. 

Otherwise, the paper makes quite a few meandering and unsupported claims. For instance:

- "Cognition emerges from electrophysiological signal flow (e.g., EEG patterns)"
- "The brain's direct semantic mapping to dedicated components represents a fundamentally simpler mechanism than representation learning's indirect vector encoding, aligning with evolutionary efficiency."

Experimentally, it is insufficient to show the loss curve and amount of sparsification-- a thorough comparison to other architectures, incl. GPT, TopoLM, etc, on at least the text perplexity is needed. Furthermore, the current method produces incoherent continuations, see Appendix A table, which makes the conclusion that "BriLLM is a transformative framework for genuine AGI development" (line 480) highly implausible. 

I recommend rejection as the manuscript does not demonstrate proper literature review and experimental practice. 

[1] TopoLM (Rathi et al., ICLR 2025)

[2] TopoFormer (Binhuraib et al., ICLR Representational Alignment Workshop 2024)

[3] Mixture of Cognitive Reasoners (AlKhamissi et al., 2025)

### Questions
N/A

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces BriLLM, a brain-inspired large language model based on Signal Fully-connected flowing (SiFu) learning, the first framework claimed to authentically replicate the brain’s macroscopic information processing principles.  It leverages two neurocognitive facts: static semantic mapping to dedicated cortical regions and dynamic signal propagation via electrophysiological activity.  BriLLM achieves multi-modal compatibility, full interpretability, context-length independent scaling, and simulates brain-like language processing.  With 1–2B parameter models matching GPT-1 performance and scalability to 100–200B parameters, it proposes a shift from representation learning toward biologically-validated AGI foundations.

### Strengths
- **Originality**: Attempts a biologically-grounded paradigm shift, though derivative of SNNs and neurocognitive models.
- **Quality**: Some conceptual alignment with neuroscience principles, but experimental validation is absent.
- **Clarity**: Limited by poor figure annotation and vague methodology.
- **Significance**: Targets AGI limitations, but lacks practical impact without evidence.

### Weaknesses
- **Methodological Flaws**: The SiFu graph (Definition 1) and signal tensor (Definition 2) lack validation against EEG data or cortical activation patterns.  The competitive activation formula ignores biological noise and synaptic delays.
- **Experimental Gaps**: No performance metrics (e.g., perplexity, BLEU) compare BriLLM to GPT-1 or modern LLMs.  Scalability claims to 100–200B parameters are theoretical without training data or hardware details.
- **Oversight**: Ignores computational cost trade-offs and energy efficiency compared to Transformers.  Potential biases in semantic mapping (e.g., cultural variability) are unaddressed.
- **Validation**: Claims of multi-modal compatibility and interpretability lack demonstration with real datasets or tasks.

### Questions
1.  Can the authors provide EEG or fMRI data (for example, brain encoding perfermance, or brain-like perfermance) comparisons to validate SiFu’s signal propagation against biological patterns?
2.  What are the specific perplexity, BLEU, or other metrics for BriLLM’s 1–2B models versus GPT-1, and why were modern LLMs excluded?
3.  How were the 100–200B parameter scalability estimates derived, and what training infrastructure supports this claim?
4.  Can the authors quantify the impact of biological noise or synaptic delays on SiFu’s competitive activation mechanism?
5.  Why were energy consumption and hardware requirements not compared to Transformer-based models?

### Soundness
4

### Presentation
1

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
This paper introduces BriLLM, a new brain-inspired LLM based on a paradigm called SiFu (Signal Fully-connected flowing) learning. Instead of using traditional representation learning like Transformers, SiFu creates a large graph where each node is a specific word or concept. The model processes language by dynamic signal propagation, where a signal flows through this graph, allowing it to handle sequences with linear $O(L)$ complexity, unlike the quadratic $O(L^2)$ complexity of Transformers. The authors trained 1-2B parameter models and claim they demonstrate stable learning and "GPT-1-level" generative abilities.

### Strengths
**Ambitious conceptual goal:** The paper attempts to address fundamental, recognized limitations of the Transformer paradigm, such as quadratic complexity in sequence length and the "black box" nature of representations. The goal of creating a new ML paradigm grounded in neuroscientific principles is ambitious and valuable.

### Weaknesses
This paper suffers from several fundamental weaknesses, ranging from incorrect characterizations of its own model to severe overstatements of its experimental results.
1. **Mischaracterization of non-representation learning:** The model's signal tensor $r \in \mathbb{R}^{d_{node}}$ is, by definition, a learned, dense representation of the state. This signal is updated at each step, precisely like the hidden state of an RNN. The model is a graph-based representation learning model.
2. **Superficial neuroscientific grounding:** The neurocognitive facts used for justification are superficial analogies. Static semantic mapping is simply a 1-to-1 mapping between a node and a token in a vocabulary. Dynamic signal propagation is an RNN-like state update. This makes it difficult to separate the genuine technical contribution from the complex and ultimately misleading terminology.
3. **Very weak experimental validation:** The paper provides no meaningful quantitative evaluation. The authors' excuse that the model "precludes direct comparisons to GPT-1's benchmarking"  is unfounded. BriLLM is a generative language model. It can and should be evaluated using standard metrics, such as perplexity, on a held-out test set.
4. **Overstated performance claims:** The claim to replicate "GPT-1-level generative performance"  is demonstrably false based on the paper's own provided samples. For example, in Table 6, the completion for "The English biologist Thomas Henry Huxley" is "coined World C that ADE XaZul 30 Ars lead singular shipb more smaller im". This output is incoherent gibberish. GPT-1 (2018) was capable of producing coherent, multi-sentence paragraphs. This discrepancy severely undermines the authors' credibility and the validity of their entire experimental section.
5. **Architectural impracticality:** The model's $O(n^2)$ parameter scaling with vocabulary size $n$ is one of its weaknesses. While the authors use sparsity to reduce the parameter count, they acknowledge that a standard 40k-token vocabulary would still require a 100-200B parameter model. This is a massive, sparse, and inefficient architecture compared to modern Transformers, which achieve parameter efficiency through weight sharing.
6. **Lack of clarity:** The description of the model's operation, particularly for prediction and training, is difficult to follow. Section 2.1 defines prediction as finding the node with the maximum signal energy via a complex formula, but Section 3.2 defines it as finding the node with the maximum L2 norm after signal aggregation.

### Questions
Please see the weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This study proposes BriLLM, a novel brain-inspired large language model centered on a new learning paradigm called Signal Fully-connected flowing (SiFu). Unlike mainstream deep learning frameworks (such as Transformers) that rely on vector-based representations and transformations, SiFu draws from fundamental principles of cognitive neuroscience to emulate macro-scale brain information processing. Specifically, it incorporates two biologically grounded mechanisms: (1) static semantic mapping, where semantic units are consistently represented in dedicated cortical regions, and (2) dynamic signal propagation, through which cognition emerges via electrophysiological activity spreading across neural pathways. The authors formalize this mechanism as a fully connected graph, where nodes correspond to semantic units (e.g., words) and edges represent learnable signal transmission pathways. Within this framework, BriLLM performs generative language modeling and demonstrates several promising properties, including full node-level interpretability, inherent multimodal compatibility, and context-length-independent scalability. Overall, this is a highly original work and proposes a principled alternative pathway toward artificial general intelligence.

### Strengths
The study introduces a genuinely novel, neuroscience-inspired learning paradigm with compelling potential advantages.

### Weaknesses
1. Lacks direct performance comparisons with well-established LLMs (e.g., GPT-style models) on standard benchmarks, making it difficult to assess practical competitiveness.
2. The model processes sequences strictly autoregressively in a fully recursive manner, precluding parallelization during training or inference, which may lead to slow computation for long sequences.
3. The paper risks overstating its contributions. For instance, Table 1 characterizes traditional deep learning models as “Task-specific” while labeling BriLLM a “Generalist AGI system”. In reality, BriLLM is currently a prototype for language modeling and falls far short of AGI capabilities. Conversely, models like GPT-3 and beyond are widely recognized as foundational steps toward general-purpose intelligent systems.

### Questions
What advantages does the model have in terms of training and inference speed compared to GPT?

### Soundness
2

### Presentation
3

### Contribution
2
