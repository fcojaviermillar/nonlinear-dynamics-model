# Nonlinear Dynamics of Organizational Change: Agent-Based Model (ABM)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research_Complete-orange)

## 📄 Overview

This repository contains the source code, datasets, and high-resolution figures for the research paper: **"Nonlinear Dynamics of Organizational Change: A Phase-Dependent Agent-Based Model"**.

The study challenges traditional linear management theories by modeling organizations as complex adaptive systems. Using an **Agent-Based Model (ABM)** following the ODD protocol, we simulate how individual resistance thresholds ($\theta$) and organizational pressure ($\beta$) interact within a social network to produce emergent phenomena like **hysteresis** and **critical phase transitions**.

---

## 🧬 About the Lead Author

**Francisco Millar**
*Social Communicator & Psychology Senior Student (USEK, Chile)*
📍 *Temuco, La Araucanía.*

My core passion and research focus lie in the **Conceptualization and Formalization of Complex Systems**.

I view social sciences not as a final destination, but as a domain of abstract dynamics waiting to be translated into rigorous mathematical models. My goal is to apply **Agent-Based Modeling (ABM)** and non-linear dynamics to give structure to complex abstract concepts, regardless of the specific field.

I work from a **Twice-Exceptional (2e)** perspective:

* **Neurodivergent Approach (ASD/ADHD):** I leverage a cognitive profile characterized by high **Perceptual Reasoning** (WAIS-IV >99th percentile) to visualize system architectures and non-linear patterns that are often invisible in traditional qualitative analysis.
* **Interdisciplinary Synthesis:** Bridging the gap between abstract theory (Psychology/Communication) and quantitative simulation (Graph Theory, Code).

---

## 🧪 Key Findings

The simulation reveals three distinct operational regimes:

1.  **Homeostatic Regime:** High inertia. Increasing pressure yields negligible adoption (Energy waste).
2.  **Critical Transition Window:** The system becomes highly sensitive to initial conditions. Small inputs cause massive cascades (Tipping Points).
3.  **Saturation Regime:** Adoption is locked-in.

> **Key Insight:** Interventions must be phase-dependent. Applying high-pressure strategies in a homeostatic phase triggers "organizational memory" (hysteresis), making future change harder.

---

## 🛠️ Repository Structure

* `model.py`: **The Core Script.** Contains the `OrganizationalChangeABM` class, the logic for Equations 1 & 2 (Discrepancy & Sigmoid transition), and the plotting functions.
* `simulation_results_LHS.csv`: **Raw Data.** Results from the Global Sensitivity Analysis using Latin Hypercube Sampling ($n=128$).
* `sensitivity_chart.jpeg`: **Figure 1.** Visualization of parameter influence ($\beta$ and $\lambda$ dominance).
* `fig_paisaje_fase.jpg`: **Figure 2.** Heatmap of the Phase Landscape showing the Critical Window.
* `fig_efficacy_relative.jpg`: **Figure 3.** Bar chart comparing intervention strategies (Hubs vs. Baseline).

---

## 🚀 How to Run the Code

To replicate the figures and analysis:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/fcojaviermillar/organizational-change-abm.git](https://github.com/fcojaviermillar/organizational-change-abm.git)
    cd organizational-change-abm
    ```

2.  **Install dependencies:**
    You need Python 3.x and the following scientific libraries:
    ```bash
    pip install numpy pandas networkx matplotlib seaborn scipy
    ```

3.  **Execute the script:**
    ```bash
    python model.py
    ```
    *This will run the simulation loop, perform the OAT/LHS analysis, and generate the 3 high-resolution images in your folder.*

---

## 📚 Citation

If you use this model or code in your research, please cite it as follows:

```bibtex
@software{millar2025model,
  author       = {Millar, Francisco and Naranjo, David},
  title        = {{Nonlinear Dynamics of Organizational Change: Agent-Based Model source code}},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/fcojaviermillar/organizational-change-abm](https://github.com/fcojaviermillar/organizational-change-abm)}},
  version      = {1.0.0}
}
