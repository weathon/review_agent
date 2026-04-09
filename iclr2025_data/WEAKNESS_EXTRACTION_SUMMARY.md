# Weakness Extraction Summary: LLM Fine-Tuning & Selective Methods

## Overview
This document summarizes weakness patterns, limitations, and evaluation criticisms from six key papers related to fine-tuning, adaptation, and generalization of language models. These papers address complementary aspects of the selective fine-tuning problem.

---

## Paper 1: OS-ATLAS - Foundation Action Model for GUI Agents
**ID:** n9PDaFNi8t.txt
**Focus:** GUI grounding, multi-platform agent training, OOD generalization

### Explicit Weaknesses & Limitations
1. **Data heterogeneity undermines generalization**
   - Same action labeled differently across platforms (e.g., "tap" vs "click")
   - This inconsistency creates confusion during model training and decreases performance

2. **Limited data availability for fine-tuning**
   - Only 7k web samples and 0 desktop samples available for fine-tuning
   - Very limited data available for desktop and web platforms significantly degrades performance

3. **OOD generalization challenges**
   - Open-source VLMs lag significantly compared to closed-source models (GPT-4o, Gemini)
   - Poor performance in zero-shot OOD scenarios

4. **Data collection challenges**
   - Desktop/Mobile data collection is significantly more complex than web screenshots
   - Requires substantial engineering efforts for simulation environments and program design to mimic interactions
   - Heterogeneous data quality and compatibility issues across different platforms

### Methodological Concerns
- Requires distinct infrastructures for each platform to ensure data quality/compatibility
- Initial training uses only limited 3-agent datasets for OOD evaluation purposes
- Heavy reliance on engineering infrastructure for data synthesis

### Evaluation/Generalization Criticisms
- Limited benchmarks: Only evaluates on 6 benchmarks across 3 platforms
- OOD generalization remains a major bottleneck
- Success heavily dependent on pre-training corpus scale and quality

**Relevance to SFT selective fine-tuning paper:**
- Demonstrates that data inconsistency and heterogeneity are critical barriers to selective learning
- Shows importance of platform-specific expertise for proper adaptation
- Highlights that selective methods must handle domain-specific idiosyncrasies

---

## Paper 2: Domain Shift Tuning (DST) over Knowledge Gap
**ID:** ijwYWoChN9.txt
**Focus:** Parameter-efficient fine-tuning, domain adaptation, knowledge subnetworks

### Explicit Weaknesses & Limitations
1. **Catastrophic forgetting with full fine-tuning**
   - Size discrepancy between pre-training (16G-25TB) and target corpora (much smaller)
   - All-weights fine-tuning risks catastrophic forgetting and poor generalization
   - Especially problematic in low-resource settings where fine-tuning on limited data risks overfitting

2. **Limited interpretability**
   - Latent nature of knowledge in DST poses challenges for interpretability
   - Future work needed to enhance interpretability of latent variables

3. **Reduced effectiveness in domain-similar settings**
   - When target domain closely resembles source domain, benefits are less pronounced
   - Knowledge distributions may not differ significantly, reducing Knowledge Steering Layer impact
   - If target data already well-represented in source data, dynamic adjustment provides minimal benefit

4. **Computational complexity for knowledge determination**
   - Methods like variational Bayes and Dirichlet processes for optimal K number are computationally intensive
   - Current study defers automatic K determination to future work

### Methodological Concerns
- Relies on assumptions about subnetwork structure without full validation
- Knowledge alignment approach depends on proper identification of knowledge-equivalent subnetworks
- Error analysis shows categorical judgment difficulty when ground truth is personalized

### Evaluation/Generalization Criticisms
- Limited demonstration on truly low-resource scenarios
- Limitations in handling ethical expressions in given datasets
- Scope limited to specific NLP tasks (document understanding, summarization)

**Relevance to SFT selective fine-tuning paper:**
- Shows catastrophic forgetting as critical weakness of full fine-tuning
- Demonstrates importance of parameter-efficient approaches
- Highlights that selective methods need theoretical grounding in knowledge structures
- Illustrates the low-resource fine-tuning bottleneck

---

## Paper 3: UniCon - Unidirectional Control for Diffusion Models
**ID:** uJqKf24HGN.txt
**Focus:** Parameter-efficient adapter training, gradient computation efficiency

### Explicit Weaknesses & Limitations
1. **Computational overhead in adapter training**
   - Existing ControlNet approaches double computational overhead
   - Training large adapters for large-scale models poses significant engineering challenges
   - Gradient demands accumulate as model parameters grow (18GB ControlNet + 16GB DiT gradients)

2. **Architecture limitations with transformers**
   - Existing control adapter designs assume U-Net encoder-decoder architecture
   - Inadequate for transformer-based diffusion models due to inability to separate encoder/decoder
   - Full-parameter fine-tuning risks losing previously learned capabilities

3. **Limited generative capabilities with fixed parameters**
   - Modifying only intermediate features with fixed parameters restricts generative abilities
   - Alternative methods (LoRA) constrained by scale; full-parameter tuning risks capability loss

4. **Speed improvement limitations**
   - UniCon reduces parameter count but doesn't significantly improve sampling speed
   - Only achieves reduced computational overhead during training, not inference

### Methodological Concerns
- Relies on specific architectural assumptions about model structure
- Trade-offs between efficiency gains and capability maintenance not fully explored

### Evaluation/Generalization Criticisms
- Limited evaluation scope on downstream applications
- Space limitations prevent comprehensive ablation studies in main text
- Evaluation primarily on image generation tasks with constrained metrics

**Relevance to SFT selective fine-tuning paper:**
- Demonstrates gradient computation as critical bottleneck in selective training
- Shows importance of selective parameter updates vs. full fine-tuning
- Illustrates efficiency gains possible through unidirectional information flow
- Highlights challenge of maintaining capabilities while reducing parameters

---

## Paper 4: MetaUrban - Urban Micromobility Simulation Platform
**ID:** kFsWpSxkFz.txt
**Focus:** Embodied AI, generalization across environments, compositional learning

### Explicit Weaknesses & Limitations
1. **Limited real-world distribution coverage**
   - Object category distribution extracted from real-world data but lacks location/layout distribution
   - Difficulty in accurately reconstructing 3D scene distributions
   - Closed-set definitions in image datasets limit object coverage to 90 objects

2. **Generalization gaps in complex scenarios**
   - Models show superior performance in simple scenarios but limitations in complex social dynamics
   - Parallel movement coordination poses significant challenges
   - Different sensor modalities show dramatic performance variations (LiDAR >> Depth >> RGB)

3. **Environmental complexity challenges**
   - Long-horizon tasks in large-scale scenes bring unique challenges to mobile machines
   - Multifarious terrains cause failures: getting stuck, barely moving, or toppling
   - Diversity, particularity, and concentration of obstacles in urban spaces present unique challenges
   - Dense pedestrians and social navigation significantly reduce success rates (50% in some scenarios)

4. **Sensor dependency**
   - RGB sensors perform worst due to lack of 3D information and sensitivity to environmental variations
   - Depth sensors show moderate performance with challenges in extracting spatial features

### Methodological Concerns
- Requires carefully tuned multi-agent coordination algorithms (ORCA + Push & Rotate)
- Evaluation heavily dependent on simulation fidelity
- Cross-machine evaluation requires accounting for mechanical structure variations

### Evaluation/Generalization Criticisms
- Limited to simulation environment (sim-to-real gap not addressed)
- Success heavily influenced by specific environmental factors
- Behavioral evaluation shows dramatic degradation with increasing pedestrian density
- No clear solutions for improving generalization in high-complexity scenarios

**Relevance to SFT selective fine-tuning paper:**
- Shows that selective capabilities (e.g., social navigation) are harder to learn than basic skills
- Demonstrates complexity of distributed learning across heterogeneous agents
- Illustrates importance of compositional training for generalization
- Shows selective fine-tuning alone insufficient without proper data diversity

---

## Paper 5: On the Generalization of Preference Learning with DPO
**ID:** bGkPZtisSm.txt
**Focus:** Preference learning, fine-tuning for alignment, generalization theory

### Explicit Weaknesses & Limitations
1. **Theoretical gaps in preference learning**
   - Thorough understanding of generalization guarantees for preference learning remains lacking
   - Existing generalization theories not directly applicable to language model outputs
   - Theoretical analysis at early stages, largely underdeveloped

2. **Complexity of language modeling**
   - Output space of sentences is considerably more complex than scalar/categorical labels
   - Existing theories typically consider overparameterized models with near-optimal loss
   - Real-world practice involves finite gradient steps, not asymptotic optimization

3. **Limited applicability beyond DPO**
   - Framework primarily focuses on DPO; may not fully capture nuances of other preference methods
   - Multi-token response analysis becomes significantly more complex
   - Providing strong guarantees for multi-token responses is highly non-trivial

4. **Preference distribution assumptions**
   - Theory depends on specific structure of preference distribution and reward concepts
   - Conditions for generalization depend on number of preference concepts and response similarity
   - May not hold in highly diverse or novel preference domains

### Methodological Concerns
- Assumes finite-step training regime matching practice but losing connection to classical theory
- Reward margin analysis assumes tractable preference structures
- Single-token response focus simplifies but limits practical applicability

### Evaluation/Generalization Criticisms
- Empirical validation limited to contemporary LLMs (Llama-2, Llama-3.1)
- Limited diversity of preference datasets in experiments
- Theory assumes preference concepts remain stable across training

**Relevance to SFT selective fine-tuning paper:**
- Demonstrates theoretical gap in understanding selective preference optimization
- Shows that fine-tuning for alignment has specific generalization challenges
- Illustrates importance of finite-step training considerations (relevant to SFT)
- Highlights complexity of preference-based selective learning

---

## Paper 6: DEPT - Decoupled Embeddings for Pre-Training Language Models
**ID:** vf5aUZT0Fz.txt
**Focus:** Heterogeneous data training, vocabulary-agnostic learning, plasticity and generalization

### Explicit Weaknesses & Limitations
1. **Challenges with heterogeneous data sources**
   - Negative interference where diverse sources compete for capacity
   - "Curse of multilinguality": adding languages yields diminishing returns, especially for low-resource languages
   - Vocabulary dilution problem: high-resource languages better represented, low-resource languages underrepresented
   - Sub-optimal cross-lingual/domain performance despite expensive methods

2. **Vocabulary and tokenization challenges**
   - Major challenge: vocabulary dilution when representing multiple data sources with single tokenizer
   - Low-resource languages suffer from poor fertility and underrepresentation
   - Temperature-tuning of language sampling requires expensive model selection
   - Intensive language-specific heuristics needed (as seen in LLaMA)

3. **Limitations of SPEC variant**
   - Models require final global embedding for practical use despite decoupling approach
   - Local vocabularies limit generalization beyond broadest dataset in pre-training distribution
   - Inference on unseen data mixtures requires either broadest embedding or closest-to-target embedding
   - Predefined clusters approach requires advance knowledge of data sources

4. **Capacity and scalability trade-offs**
   - Addressing vocabulary dilution in highly multilingual models is extremely challenging
   - Providing sufficient tokens for all languages results in impractically large models
   - Custom vocabularies require increased embedding matrices (multiple copies)

### Methodological Concerns
- Requires careful multi-phase adaptive pre-training pipeline
- Depends on careful curation per data source
- Cannot strongly control data sampling rates without external coordination

### Evaluation/Generalization Criticisms
- Evaluation limited to specific language families and datasets
- OOD generalization tested only on related language/domain tasks
- Downstream evaluation limited to standard NLU tasks (MNLI, RACE, STSB)
- Scaling evaluated only up to billion-scale; claims about larger models speculative

**Relevance to SFT selective fine-tuning paper:**
- Demonstrates that heterogeneous fine-tuning requires careful vocabulary/tokenization handling
- Shows negative interference as key limitation of multi-domain fine-tuning
- Illustrates importance of selective parameter updates for efficiency
- Highlights generalization challenges when combining multiple data sources
- Shows that selective approaches must handle vocabulary and embedding layer carefully

---

## Cross-Paper Weakness Themes

### 1. **Generalization vs. Specialization Trade-off**
All papers struggle with the tension between:
- Maintaining pre-training knowledge (avoiding catastrophic forgetting)
- Adapting specifically to target domain/tasks
- Achieving good OOD performance while keeping specialized knowledge

**Critical for SFT:** Selective fine-tuning must balance these through careful parameter selection.

### 2. **Data Heterogeneity as Fundamental Challenge**
- Inconsistent labeling/formatting across domains (OS-Atlas)
- Competing capacity across diverse sources (DEPT, MetaUrban)
- Different semantic structures requiring different handling (DST)

**Critical for SFT:** Cannot use uniform fine-tuning strategy across heterogeneous data.

### 3. **Computational Efficiency vs. Capability**
- Full fine-tuning expensive but maintains full capacity (UniCon, DST)
- Parameter-efficient methods constrain capability (UniCon, DEPT)
- Trade-offs between gradient computation, memory, and speed

**Critical for SFT:** Selective methods must justify their parameter selection through capability analysis.

### 4. **Limited Theoretical Understanding**
- DPO paper: preference learning theory immature
- DST paper: knowledge subnetwork structure not fully understood
- MetaUrban: sim-to-real gap and generalization mechanisms unclear

**Critical for SFT:** Need rigorous framework for understanding which parameters affect which capabilities.

### 5. **Evaluation Scope Limitations**
- OS-Atlas: Limited benchmark coverage
- DEPT: Primarily downstream NLU tasks
- MetaUrban: Simulation-only evaluation
- DPO: Limited to alignment datasets

**Critical for SFT:** Selective fine-tuning evaluation should cover broader capability spectrum.

### 6. **Low-Resource Setting Challenges**
- DEPT: Low-resource languages suffer from vocabulary underrepresentation
- DST: Limited target corpus risks overfitting
- OS-Atlas: Desktop platform has only 7k samples
- MetaUrban: Sparse environmental coverage

**Critical for SFT:** Selective methods essential for low-resource scenarios but need special handling.

---

## Recommendations for SFT Selective Fine-Tuning Paper

### Address Known Weaknesses:
1. **Theory/Methodology**
   - Provide rigorous framework for understanding why selected parameters matter
   - Explain mechanism for avoiding catastrophic forgetting through selective approach

2. **Evaluation Design**
   - Comprehensive benchmarks covering multiple domains (not just narrow evaluation)
   - OOD generalization testing to show selective parameters maintain broader capabilities
   - Ablation studies showing which parameter types matter most for different capabilities

3. **Data Heterogeneity**
   - Show how selective fine-tuning handles inconsistent labeling across domains
   - Demonstrate handling of competing capacity when mixing domains
   - Address vocabulary/representation consistency

4. **Computational Justification**
   - Show clear efficiency gains (memory, speed, convergence)
   - Compare against full fine-tuning and other parameter-efficient methods
   - Justify parameter selection through capability analysis

5. **Generalization Analysis**
   - Demonstrate both in-domain and OOD generalization
   - Show maintained performance on tasks not in fine-tuning distribution
   - Address sim-to-real or distribution shift scenarios

6. **Low-Resource Scenarios**
   - Particularly important given finite-step training focus
   - Show stability with small target datasets
   - Address catastrophic forgetting risk

### Anticipated Reviewer Criticisms:
1. "Limited theoretical justification for parameter selection"
2. "Evaluation on too narrow benchmark set"
3. "Doesn't address catastrophic forgetting adequately"
4. "Unclear generalization beyond fine-tuning domains"
5. "Computational savings not clearly demonstrated vs. alternatives"
6. "Doesn't handle data heterogeneity/inconsistency"

---

## File Paths Referenced
- `/home/wg25r/review_agent/iclr2025_data/papers/n9PDaFNi8t.txt` - OS-Atlas
- `/home/wg25r/review_agent/iclr2025_data/papers/ijwYWoChN9.txt` - DST
- `/home/wg25r/review_agent/iclr2025_data/papers/uJqKf24HGN.txt` - UniCon
- `/home/wg25r/review_agent/iclr2025_data/papers/kFsWpSxkFz.txt` - MetaUrban
- `/home/wg25r/review_agent/iclr2025_data/papers/bGkPZtisSm.txt` - DPO Preference Learning
- `/home/wg25r/review_agent/iclr2025_data/papers/vf5aUZT0Fz.txt` - DEPT
