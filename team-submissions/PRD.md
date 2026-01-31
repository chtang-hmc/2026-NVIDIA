# Product Requirements Document (PRD)

**Project Name:** LABS Solver

**Team Name:** Squaxions

**GitHub Repository:** https://github.com/chtang-hmc/2026-NVIDIA

---

> **Note to Students:** > The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist. 
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.
> * **Goal:** The objective is to convince the reader that you have a solid plan, not just to fill in boxes.

---

## 1. Team Roles & Responsibilities [You can DM the judges this information instead of including it in the repository]

| Role | Name | GitHub Handle | Discord Handle
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | Brayden Mendoza | brayjmendoza | BrayJ |
| **GPU Acceleration PIC** (Builder) | Chengyi Tang | chtang-hmc | chengyitang |
| **Quality Assurance PIC** (Verifier) | Sofiia Zaozerska, Jiani Fu | szaozerska, fjn004 | sofigoldfox, jiafu1234 |
| **Technical Marketing PIC** (Storyteller) | Zaara Bhatia | Zaara230761 | zaarabhatia_31348 |

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm
* **Algorithm:** [Identify the specific algorithm or ansatz]
    * *Example:* "Quantum Approximate Optimization Algorithm (QAOA) with a hardware-efficient ansatz."
    * *Example:* "Variational Quantum Eigensolver (VQE) using a custom warm-start initialization."

* **Motivation:** [Why this algorithm? Connect it to the problem structure or learning goals.]
    * *Example (Metric-driven):* "We chose QAOA because we believe the layer depth corresponds well to the correlation length of the LABS sequences."
    *  Example (Skills-driven):* "We selected VQE to maximize skill transfer. Our senior members want to test a novel 'warm-start' adaptation, while the standard implementation provides an accessible ramp-up for our members new to quantum variational methods."
   

### Literature Review
* **Reference:** [Title, Author, Link]
* **Relevance:** [How does this paper support your plan?]
    * *Example:* "Reference: 'QAOA for MaxCut.' Relevance: Although LABS is different from MaxCut, this paper demonstrates how parameter concentration can speed up optimization, which we hope to replicate."

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)
* **Strategy:** [How will you use the GPU for the quantum part?]
    * *Example:* "After testing with a single L4, we will target the `nvidia-mgpu` backend to distribute the circuit simulation across multiple L4s for large $N$."
 

### Classical Acceleration (MTS)
* **Strategy:** [The classical search has many opportuntities for GPU acceleration. What will you chose to do?]
    * *Example:* "The standard MTS evaluates neighbors one by one. We will use `cupy` to rewrite the energy function to evaluate a batch of 1,000 neighbor flips simultaneously on the GPU."

### Hardware Targets
* **Dev Environment:** We will use Qbraid CPUs for initial testing and code verification, and Brev T4 for initial GPU testing. Our budget will be 10 hours of testing at $0.50/hr. Total budget will be $5.
* **Production Environment:** We will use the Brev A100-80GB for final benchmarks. Our budget will be 7 hours of benchmark at $1.50/hr. Total budget will be $10.50.

Total amount will be $15.50. We will leave $4.50 buffer in case of extra benchmarking needed or for idle runs.

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC

### Unit Testing Strategy
* **Framework:** pytest
* **AI Hallucination Guardrails:** [How do you know the AI code is right?]
    * *Example:* "We will require AI-generated kernels to pass a 'property test' (Hypothesis library) ensuring outputs are always within theoretical energy bounds before they are integrated."

### Core Correctness Checks
* **Check 1 (Symmetry):** [Describe a specific physics check]
    * *Example:* "LABS sequence $S$ and its negation $-S$ must have identical energies. We will assert `energy(S) == energy(-S)`."
* **Check 2 (Ground Truth):**
    * *Example:* "For $N=3$, the known optimal energy is 1.0. Our test suite will assert that our GPU kernel returns exactly 1.0 for the sequence `[1, 1, -1]`."

---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow
* **Plan:** [How will you orchestrate your tools?]
    * *Example:* "We are using Cursor as the IDE. We have created a `skills.md` file containing the CUDA-Q documentation so the agent doesn't hallucinate API calls. The QA Lead runs the tests, and if they fail, pastes the error log back into the Agent to refactor."

### Success Metrics
* **Metric 1 (Approximation):** [e.g., Target Ratio > 0.9 for N=30]
* **Metric 2 (Speedup):** [e.g., 10x speedup over the CPU-only Tutorial baseline]
* **Metric 3 (Scale):** [e.g., Successfully run a simulation for N=40]

### Visualization Plan
* **Plot 1:** [e.g., "Time-to-Solution vs. Problem Size (N)" comparing CPU vs. GPU]
* **Plot 2:** [e.g., "Convergence Rate" (Energy vs. Iteration count) for the Quantum Seed vs. Random Seed]

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC 

* **Plan:** [How will you avoid burning all your credits?]
    * *Example:* "We will develop entirely on Qbraid (CPU) until the unit tests pass. We will then spin up a cheap L4 instance on Brev for porting. We will only spin up the expensive A100 instance for the final 2 hours of benchmarking."
    * *Example:* "The GPU Acceleration PIC is responsible for manually shutting down the Brev instance whenever the team takes a meal break."

## 7. Scheduling

- classical algs schedule (Chengyi)
    -
    * write tests (15:00)
    * run tests on qbraid (16:00)
    * write gpu code (17:00)
    * run tests on t4 (17:30)
    * create benchmarking code for classical (22:00)
    * test small benchmark on t4 (23:00)

- quantum algs schedule (everyone else)
    -
    * finish notebook (15:00)
    * decide which algorithm to use (18:00)
    * write tests (19:00)
    * write documentation for algorithm (23:00)
    * finish code for quantum algorithm (23:00)
    * iterate if needed

    * run tests on qbraid (0:00)
    * write gpu code (1:30)
    * synthesize classical and quantum codebase (2:15)
        * quantum produces population that feeds to classical
    * run tests on t4 (2:45)
    * create benchmarking code for quantum (22:00 from Chengyi)
    * test small benchmark on t4 (3:30)
    * run entire benchmark on a100 (8:45)

- presentation schedule (Sofi)
    - 
    * need to write documentation throughout
    * record AI usage and write vibe log
        * record good/bad ai usage
    
    * presentation plan
    * fill in details
    * practice presentation

2:30 Finish all coding. Start final benchmarking. Begin working on presentation material.
8:00 Finish benchmarking, start final work on presentations.
9:00 Finish writing presentation. Start practicing for presentation. Final checks for code styling and provide documentation for any unclear parts.
9:50 Submit final submission.