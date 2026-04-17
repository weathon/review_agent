# LinguaMate: Language‑Guided Metamaterial Discovery via Symbolic-Driven Latent Optimization

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Metamaterials are microstructured materials whose tailored geometries unlock unusual mechanical responses. Metamaterial discovery aims at identifying novel microstructures towards specific applications, such as transportation, robotics, etc. Traditional knowledge-driven metamaterial discovery methods are computationally expensive. While recent data-driven generative models accelerate design, they demand explicit numerical targets and struggle to understand the language descriptions of a concept or idea that is critical for the early design stage. Conversely, large language models readily understand such language intents but lack geometric awareness and physical constraints. To bridge this gap between language understanding and geometric awareness, we propose L**inguaMate**, an inference-time multi-agent optimization framework that empowers language-guided metamaterial discovery via symbolic-driven latent optimization. By jointly aligning language, geometry, and property spaces, LinguaMate discovers physically valid microstructures that extend beyond the boundaries of existing literature and training data. Extensive experiments demonstrate that LinguaMate (1) improves structural validity by up to 34% in symmetry and nearly 98% in periodicity compared to the strongest generative baselines; (2) achieves about 6–7% higher prompt-guidance scores while maintaining superior diversity relative to advanced reasoning LLMs; (3) qualitative analyses confirm the effectiveness of symbolic logic operators in enabling programmable semantic alignment; and (4) real-world case studies further validate its practical capability in metamaterial discovery. We publish our code in https://anonymous.4open.science/r/LinguaMate-CC6F.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces LinguaMate, a multi-agent framework that bridges the gap between natural language prompts and geometric design for metamaterial discovery. It combines an Agent Designer for language space, an Agent Generator for geometric space, and an Agent Supervisor for mechanical property space. LinguaMate outperforms existing generative model baselines and LLM baselines.

### Strengths
1. This paper effectively addresses the challenging and significant problem of language-guided metamaterial discovery.

2. The work is clearly written, presenting a methodology that is both intuitive and well-motivated.

### Weaknesses
The primary weakness of this paper lies in its choice of baselines for performance comparison. To demonstrate the superiority of LinguaMate, the authors compare it against four material generative models and six standalone LLMs. The comparisons do not include any existing agent-based frameworks that also combine LLMs and generative models.
This paper is not the first to propose an agent-based approach for metamaterial discovery. Its contribution is positioned as a stronger agent framework compared to existing methods like MetaScientist. Therefore, a systematic and direct performance comparison against these established agent-based baselines is essential.
The finding that LinguaMate, a hybrid system, outperforms standalone generative models (which lack language understanding) or isolated LLMs (which lack geometric awareness) is hardly surprising. This comparison does not sufficiently validate the novelty of the proposed method. A rigorous evaluation would require:

1. Direct comparison against state-of-the-art agent baselines (e.g., MetaScientist).

2. An ablation study showing how LinguaMate's novel components improve upon those existing baselines.

### Questions
Please refer to the Weaknesses section above.

### Soundness
3

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
3

### Summary
This paper introduces LinguaMate, a language-guided metamaterial discovery framework that bridges the gap between natural language reasoning and geometry-aware generative modeling. The method employs a multi-agent system comprising three specialized agents: Designer (LLM-based language interpreter), Generator (geometry synthesizer with disentangled latent spaces), and Supervisor (property predictor and evaluator). The key innovation lies in symbolic-driven latent optimization—a set of four interpretable latent-space operators (Union, Mix, Intersection, Negation)—that enable programmatic semantic composition and cross-modal alignment between language, geometry, and physical properties.

### Strengths
- The introduction of programmable operators in Gaussian latent space provides interpretable and compositional control over design semantics, representing a meaningful contribution to controllable generation.
- The paper benchmarks against multiple strong baselines including VAEs, diffusion models, and advanced LLMs (GPT-4o, Gemini 2.0, DeepSeek-Reasoning), demonstrating consistent quantitative and qualitative improvements in validity, diversity, and language alignment.
- The inclusion of case studies involving finite element simulation and 3D printing adds valuable real-world credibility to the proposed system.

### Weaknesses
- The paper lacks formal justification for latent logic operators (Union, Mix, Intersection, Negation) semantic preservation properties and guarantees on maintaining manifold validity.
- The ablation study focuses primarily on loss terms but does not isolate the individual contributions of each symbolic operator or the human-in-the-loop component. It remains unclear which operator is most critical for performance, or how much improvement comes from human intervention versus automated agent collaboration.

### Questions
- The anonymous repository link appears to be inaccessible. Please verify the link is correctly configured for anonymous access.
- How does the framework handle cases where LLM-generated scaffolds contain geometric inaccuracies or are physically invalid? What mechanisms ensure that the Generator can still produce valid structures when starting from an imperfect scaffold?

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
3

### Summary
This work introduces a system for the generation of novel metamaterials, microstructures that exhibit particular material properties. These materials are represented by a periodic lattice with nodes and edges forming a repeating structure. The system developed by the authors consists of 3 parts: an "agent designer", which is an LLM that retrieves existing materials from available literature that matches the user's natural-language specified goals, an "agent generator", a VAE that defines a searchable latent space over materials and an "agent supervisor consisting of a ML-based property predictor and LLM for judging the properties of generated materials.  The generator uses an explicitly disentangled decoder structure to facilitate easier generation and control during latent space search along with a set of symbolic operations that allow for guided exploration in the space. By iteratively utilizing these 3 agents the systems generates new candidate Metamaterials. In their experiments, the authors show that their approach generates new valid Metamaterials at a higher rate than prior work.

### Strengths
**Motivation**

- The work is well motivated and tackles an under explored area of generative ML for metamaterials design
- The goals of the system and potential use cases are clear. The authors bring together a number of important recent developments in machine learning for this problem including LLM-based prompt understanding and VAEs for periodic structure prediction.

**Novelty**
- I am not an expert in this particular subfield, but I am not aware of prior work that allows for natural language guided design of Metamaterials. 
- As far as I know the authors approach to exploring the space of materials is also novel.

**Evaluation**
- The authors compare against a good range of reasonable, recent baselines for this area.
- The authors experimentally validated a generation from the system

Overall, this paper is interesting in its application and direction.

### Weaknesses
**Confusing writing**
- It was unclear to me until I looked at the supplement, that the output of the designer was a material specification. 
- For example it was very unclear in section 3.1.2 how the symbolic operations to apply are chosen and where the initial material comes from.
- I still don't fully understand the process of generating a material. It would be very helpful to have a walkthrough or pseudo code for the entire process of generating a new material in the main text. 

**Missing details**
- Table 1 is not very helpful as many of the symbols are not defined in the main text.
- It's not specified how the likelihoods for the generator are defined as far as I can tell. Details of the architecture, such as number of parameters is also missing.
- Unclear how baselines were adapted for this task. For example, as far as I know CDVAE does not generate edge structures and it's unclear how that was handled in this case

**Methodology**
- The system is very complex, without substantial justification for the complexity. There isn't really a formal justification that the LLM components are improving the performance.
-  Authors claim that the decoder is explicitly disentangled, but the diagram of the implementation suggests there are shared layers in the decoder. I don't understand how to reconcile these two things.
  - Given this, it's unclear how meaningful the symbolic operations actually are.
- The method relies on ML learned evaluation for optimizing for target properties, which could potentially bias the results.

**Evaluation**
- The main goal of this paper is property guided generation, however this isn't properly evaluated. The metrics reported are only validity, diversity and a questionable LLM evaluated metric for language guidance. It seems necessary to compare the properties of materials generated by LinguaMate to baselines by FE simulation or experimentally.
-  The authors claim to have experimentally evaluated a material generated by the model, but give no details about this material.

### Questions
- Can you discuss a real-world use case where the natural language goal specification would be important? Wouldn't the applications mentioned in the introduction ("biomedical devices, transportation systems, robotics") generally benefit from engineers specifying exact design requirements?
- Why does the designer not output lattice parameters? 
- Do the distributions in the generator optimization come from the encoder?

### Soundness
2

### Presentation
2

### Contribution
2
