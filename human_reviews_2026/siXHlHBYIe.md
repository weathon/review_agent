# Web-CogReasoner: Towards Multimodal Knowledge-Induced Cognitive Reasoning for Web Agents

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
Multimodal large-scale models have significantly advanced the development of web agents, enabling them to perceive and interact with the digital environment in a manner analogous to human cognition. In this paper, we argue that web agents must first acquire sufficient knowledge to engage in cognitive reasoning effectively. Therefore, we decompose a web agent's capabilities into two essential stages: knowledge content learning and cognitive processes. To formalize this, we propose Web-CogKnowledge Framework, which categorizes knowledge into Factual, Conceptual, and Procedural domains. In this framework, knowledge content learning corresponds to the agent's processes of Memorizing and Understanding, which rely on the former two types of knowledge, respectively, representing the "what" of learning. Conversely, cognitive processes correspond to Exploring, grounded in Procedural knowledge, defining the "how" of reasoning and action. To facilitate knowledge acquisition, we construct the Web-CogDataset, a structured resource curated from 14 real-world websites, designed to instill the core knowledge necessary for a web agent systematically. This dataset serves as the agent's conceptual grounding—the "nouns" upon which comprehension is built—as well as the basis for learning how to reason and act. Building on this foundation, we operationalize these processes through a novel knowledge-driven Chain-of-Thought (CoT) reasoning framework, developing and training our proposed multimodal web agent, the Web-CogReasoner. Extensive experimentation reveals its significant superiority over existing models, particularly in its capacity for generalization to unseen tasks where its structured knowledge proves decisive. To facilitate rigorous and systematic evaluation, we introduce the Web-CogBench, a comprehensive evaluation suite designed to assess and compare agent performance across the delineated knowledge domains and cognitive capabilities. Our code and data are open sourced at https://github.com/Gnonymous/Web-CogReasoner.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents proposes a novel cognitive learning framework for web agents, inspired by Bloom’s taxonomy of human learning. The authors argue that effective web agents must acquire three types of knowledge—Factual, Conceptual, and Procedural—to perform reasoning and exploration tasks on the web. They introduce three key components: Web-CogKnowledge Framework, Web-CogDataset, and Web-CogBench, enabling a systematic curriculum that teaches agents to memorize, understand, and explore. The resulting model, Web-CogReasoner, integrates knowledge-driven chain-of-thought (CoT) reasoning, achieving 84.4% accuracy on Web-CogBench, outperforming Gemini 2.5 Pro, Claude Sonnet 4, and prior open-source agents on reasoning-intensive tasks

### Strengths
This work presents a conceptually rich and methodologically rigorous framework that advances the frontier of web-agent cognition. By grounding web reasoning in educational psychology (Bloom’s taxonomy), the authors provide a human-analogous paradigm for hierarchical learning—progressing from factual perception to procedural reasoning. The proposed dataset and benchmark cover 14 real-world websites with fine-grained multimodal tasks, yielding both scalability and interpretability. The experiments are comprehensive and ablation-supported, demonstrating clear contributions from each knowledge layer. The structured chain-of-thought (CoT) design substantially improves cross-task generalization and reasoning interpretability, and the authors provide transparent reporting and reproducibility details. Overall, the work represents a significant step toward cognitively inspired, knowledge-grounded web agents.

### Weaknesses
Despite its novelty, the study has several limitations. The model diversity is narrow—experiments primarily rely on Qwen 2.5 VL-7B, without testing multiple scales or architectures (e.g., Llama, Claude Opus, GPT-5), which constrains external validity. The framework’s dependence on supervised imitation learning raises concerns about long-horizon exploration, and its performance gap in cross-web generalization (≈10%) suggests limited adaptability beyond curated domains. Additionally, the theoretical connection between Bloom’s taxonomy and CoT reasoning is conceptually interesting but lacks formal grounding or cognitive validation. The framework also does not address multimodal robustness under dynamic webpages or visual perturbations, and scalability to open-world, continuously evolving web environments remains uncertain.

### Questions
1. How would Web-CogReasoner perform if trained with reinforcement learning or self-play instead of pure imitation, to strengthen procedural adaptability?

2. Could the authors include more diverse model families and scales (e.g., 7B–70B, open- and closed-source) to validate generality across architectures?

3. Have the authors explored integrating multimodal cognition—such as combining visual-textual reasoning with real-time interaction on dynamic or partially observable webpages?

4. To what extent is the Bloom’s-inspired hierarchy empirically necessary—would a non-hierarchical or single-stage variant achieve comparable performance?

5. Can the framework be extended toward lifelong or self-directed learning, enabling agents to autonomously acquire new procedural knowledge from unstructured online experiences?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Web-CogReasoner, a knowledge-induced cognitive reasoning framework for web agents. The key idea is to structure agent reasoning with a knowledge-driven chain-of-thought (K-CoT) aligned to three layers—factual, conceptual, and procedural—and to train a Qwen2.5-VL-7B–based agent via staged supervision on a new dataset (Web-CogDataset) collected from real websites. A companion benchmark (Web-CogBench) evaluates three capability dimensions—memory, understanding, and exploration—that map to the knowledge layers. Experiments report strong performance on Web-CogBench and competitive results on established web-agent benchmarks (e.g., VisualWebBench, WebVoyager, Mind2Web / Online Mind2Web), with ablations showing complementary gains from all three knowledge layers. These design choices are well-motivated by trends in web-agent evaluation and environment standardization.

### Strengths
**Clear, systematic framework**: The use of Bloom’s taxonomy to structure the agent’s learning and reasoning is conceptually coherent. While knowledge-layered modeling is not entirely novel, applying it concretely to web agents with CoT alignment is a valuable contribution.

**Well-structured data and evaluation system**: The pairing of Web-CogDataset and Web-CogBench, mapped by knowledge layers and ability dimensions, allows end-to-end evaluation from knowledge acquisition to interactive behavior.

**Reproducibility awareness and engineering detail**: The paper reports training setups, hardware, sequence lengths, and strategy choices, and includes responsible research statements. This transparency helps reproducibility and fosters trust.

### Weaknesses
**Website sampling bias**: The chosen 14 sites may disproportionately lean toward e-commerce or financial domains. This could bias the model’s strengths toward those styles; broader site diversity (e.g., social media, education, news, forums) would strengthen generalization.

**Lack of strong baseline comparisons with GPT-level models**: Without direct comparisons to large generalist models (e.g. GPT-4, GPT-5), it is hard to assess how far this approach is from “state-of-the-art general reasoning agents.”

### Questions
**Reliance on pseudo-labels and their reliability**: Some annotations are generated by Qwen-VL-72B; the paper should report the error rates, whether they performed human double-blind verification, or whether cross-model annotation consistency was checked (e.g., compare labels from Claude / Gemini).

**Judge bias / evaluator correlation**: The use of an LVM Judge for scoring may be correlated with the model training biases or data domain. Do authors provide inter-judge agreement or bootstrap confidence intervals, especially on the Understanding / Exploring tasks?

**Limited demonstration of active exploration / online feedback**: While procedural knowledge and exploration are core aims, the paper heavily leans toward structured CoT and imitation learning. More experiments incorporating online decision-making (e.g. RL fine-tuning, exploration during inference) would strengthen claims about interactive capability.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a Web-CogKnowledge framework inspired by Bloom's taxonomy, where web navigation knowledge is categorized into three levels and a two-phase training methodology is followed in order to enhance the cognitive capabilities of web agents. Based on this framework, the authors construct Web-CogDataset and use this dataset to create Web-CogBench. Furthermore, the authors train a Web-CogReasoner model and conduct evaluations against several popular models on multiple benchmarks.

Overall, this is an interesting paper attempting to frame web agent learning processes from a cognitive perspective following Bloom's taxonomy. However, all core ingredients are well-explored by related work under different names, resulting in insufficient technical novelty.

### Strengths
- Strong motivation and interesting idea to frame the problem using Bloom's taxonomy, attempting to model the learning process as three progressive stages (Memorizing, Understanding, Exploring) that map to different knowledge types.

- The dataset is well-structured following the proposed framework, covering 12 fine-grained tasks across factual, conceptual, and procedural knowledge, which could be a valuable contribution to the community for training models and conducting evaluations.

### Weaknesses
### Major

- The three proposed stages are essentially different names for well-established concepts like visual grounding, VQA, and action prediction that are widely used in current web agents, just reframed under cognitive terminology.

- To strength the argument of the proposed framework, the ablation study should demonstrate performance with and without each individual stage, rather than progressively adding stages one by one, to fully investigate the independent impact of each component in the proposed framework.

- Evaluations should be more comprehensive, missing some key benchmarks: static grounding tasks such as ScreenSpot series and dynamic environments such as WebArena and VisualWebArena that better reflect real-world web interaction scenarios.

- Web-CogBench remains static in nature and doesn't capture the dynamic, interactive aspects of real web environments where agents must adapt to changing page states and handle unexpected scenarios

### Minor

Line #343: 'via upervised fine-tuning' should be 'via supervised fine-tuning'

### Questions
- Regarding the POMDP formulation – how is the Markovian property maintained when previous action history is included in the 'trajectory history' section? Doesn't this violate the memoryless assumption?
- In Web-CogDataset, how do you define the single-step and multiple-step web tasks? What's the source?
- Can more details about the LVM Judge be disclosed?
- How is accuracy measured for open-ended responses in next page prediction?

### Soundness
3

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
The paper proposes Web-CogReasoner, a web agent trained with a two stage, knowledge first paradigm inspired by Bloom’s taxonomy: first learn Factual, Conceptual, and Procedural web knowledge, then exercise cognitive processes for reasoning and action. The authors build Web-CogDataset from 14 real sites with 12 fine-grained tasks, and introduce Web-CogBench to evaluate “Memorizing, Understanding, Exploring” capabilities aligned to those knowledge types. A  Web-CogReasoner is further trained with Qwen2.5-VL-7B as base model and  achieves strong performance on benchmarks including Web-CogBench and VisualWebBench.

### Strengths
The evaluation of Web-CogBench is aligned to the training data construction framework.  Web-CogBench measures “Memorizing, Understanding, Exploring,” directly mirroring the training objectives and enabling targeted diagnosis of capabilities.

The combination of screenshots with the accessibility tree leverages both visual cues and structured semantics of web pages.

### Weaknesses
Authors have placed much emphasis on the three aspects of “factual, conceptual, and procedural learning,” claiming that “higher-order reasoning is built on solid perceptual and conceptual foundations.” It would be great to see supporting experiments. 

It would be helpful to provide more details around the selection of “memorizing, understanding, and exploring” abilities as the tasks for Web-CogBench. Some questions that may be good to answer are: how well do these aspects cover the actual tasks performed by web agents. Are these necessary conditions for a strong web agent.

It would be good to know the reason behind choosing webvoyager and online-mind2web as the only two end to end web agent benchmarks, as they both rely on live websites and environments, which can bring noise to the experiment. In particular, it would be good to know the reason behind choosing online multimodal mind2web over multimodal mind2web. And it would be great to provide more details around experiment setup, e.g. is human or LLM judge used as evaluation. If Web-CogReasoner is set to take the role of a deep research style agent, then live environment with benchmarks such as GAIA could be a good choice.

### Questions
Listed above.

### Soundness
2

### Presentation
2

### Contribution
3
