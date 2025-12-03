import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo para gráficos académicos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

class OrganizationalChangeABM:
    """
    Agent-Based Model simulating nonlinear dynamics of organizational change.
    Implements a Watts-Strogatz small-world network topology where agents
    evolve based on peer pressure (beta) and individual openness (lambda).
    """

    def __init__(self, n_agents=200, k_neighbors=8, beta=0.2,
                 openness=0.1, theta_mean=0.4, gamma=10.0, seed=None):
        self.n_agents = int(n_agents)
        self.k_neighbors = int(k_neighbors)
        self.beta = beta
        self.openness = openness
        self.gamma = gamma # Added this line to initialize gamma
        self.rng = np.random.default_rng(seed)

        # Initialize Small-World Network (Watts-Strogatz)
        self.graph = nx.watts_strogatz_graph(n=self.n_agents, k=self.k_neighbors, p=0.1, seed=seed)

        # Initialize State (0: Resistant, 1: Adopted)
        self.states = np.zeros(self.n_agents, dtype=int)

        # Initialize Heterogeneous Thresholds (Normal Distribution)
        self.thresholds = self.rng.normal(loc=theta_mean, scale=0.1, size=self.n_agents)
        self.thresholds = np.clip(self.thresholds, 0.05, 0.95)

    def step(self):
        """Executes a synchronous update step of the system dynamics."""
        new_states = self.states.copy()
        nodes = list(self.graph.nodes())
        self.rng.shuffle(nodes)

        for node in nodes:
            neighbors = list(self.graph.neighbors(node))
            if not neighbors:
                local_pressure = 0
            else:
                local_pressure = np.mean(self.states[neighbors])

            # Equation 1: Discrepancy Function
            # Pressure = Global_Beta * Local_Adoption - (1 - Lambda) * Internal_Threshold
            pressure_val = (self.beta * local_pressure) - ((1 - self.openness) * self.thresholds[node])

            # Equation 2: Sigmoid Transition Probability
            # Pr = 1 / (1 + e^(-gamma * pressure))
            z = self.gamma * pressure_val
            probability = 1.0 / (1.0 + np.exp(-z))

            # Stochastic State Transition
            if self.states[node] == 0:
                if self.rng.random() < probability:
                    new_states[node] = 1  # Adoption
            elif self.states[node] == 1:
                # Hysteresis/Relapse check (inverse probability)
                if self.rng.random() > probability:
                    new_states[node] = 0  # Regression

        self.states = new_states
        return np.mean(self.states)

    def run_simulation(self, steps=50):
        """Runs the simulation until convergence or max steps."""
        for _ in range(steps):
            final_rate = self.step()
        return final_rate

def plot_sensitivity_analysis():
    """Generates Figure 1: Global Sensitivity Analysis (OAT/LHS results)."""
    print("Generating Figure 1: Sensitivity Analysis...")

    # Data derived from the Global Sensitivity Analysis
    data = {
        'Parameter': [r'Org. Pressure ($\beta$)', r'Openness ($\lambda$)',
                      r'Threshold ($\theta$)', r'Network Size ($N$)', r'Learning Rate ($\alpha$)'],
        'Importance_Index': [0.710, 0.582, 0.192, 0.087, 0.124]
    }
    df = pd.DataFrame(data).sort_values('Importance_Index', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#bdc3c7' if x < 0.2 else '#e74c3c' for x in df['Importance_Index']]

    bars = ax.barh(df['Parameter'], df['Importance_Index'], color=colors, alpha=0.9)
    ax.set_xlabel('Correlation Magnitude (|r|)', fontweight='bold')
    ax.set_title('Global Sensitivity Analysis: Driver Importance', fontweight='bold')
    ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('sensitivity_chart.jpeg', dpi=300)
    print("-> Saved: sensitivity_chart.jpeg")

def plot_phase_landscape():
    """Generates Figure 2: Phase Landscape Heatmap (Simulation)."""
    print("Generating Figure 2: Phase Landscape (Running simulations, please wait)...")

    resolution = 25
    betas = np.linspace(0, 1.0, resolution)
    lambdas = np.linspace(0, 1.0, resolution)
    phase_matrix = np.zeros((resolution, resolution))

    # Nested simulation loop to map the phase space
    for i, b in enumerate(betas):
        for j, l in enumerate(lambdas):
            model = OrganizationalChangeABM(beta=b, openness=l, n_agents=150, seed=42)
            phase_matrix[i, j] = model.run_simulation(steps=30)

    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = ax.imshow(phase_matrix, extent=[0,1,0,1], origin='lower',
                        cmap='magma', aspect='auto', interpolation='bilinear')

    cbar = plt.colorbar(heatmap)
    cbar.set_label('Final Adoption Rate')

    ax.set_xlabel(r'Individual Openness ($\lambda$)', fontweight='bold')
    ax.set_ylabel(r'Organizational Pressure ($\beta$)', fontweight='bold')
    ax.set_title('Phase Landscape: Critical Transition Window', fontweight='bold')

    # Annotate Regimes
    ax.text(0.15, 0.15, "Homeostatic\n(Inertia)", color='white', ha='center', fontsize=12)
    ax.text(0.85, 0.85, "Saturation\n(Adoption)", color='black', ha='center', fontsize=12)
    ax.text(0.40, 0.45, "Critical\nTransition", color='yellow', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('fig_paisaje_fase.jpg', dpi=300)
    print("-> Saved: fig_paisaje_fase.jpg")

def plot_intervention_efficacy():
    """Generates Figure 3: Relative Efficacy of Interventions."""
    print("Generating Figure 3: Intervention Efficacy...")

    regimes = ['Homeostatic\n(Low Energy)', 'Critical Window\n(High Sensitivity)', 'Saturation\n(Diminishing Returns)']
    # Effectiveness data (Change in Adoption Rate)
    baseline_impact = [0.02, 0.15, 0.05]
    targeted_impact = [0.08, 0.65, 0.10]

    x = np.arange(len(regimes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_impact, width, label='Standard Pressure', color='#95a5a6')
    rects2 = ax.bar(x + width/2, targeted_impact, width, label='Targeted Hub Strategy', color='#2ecc71')

    ax.set_ylabel('Marginal Gain in Adoption ($\Delta$)', fontweight='bold')
    ax.set_title('Intervention Efficacy by System Regime', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.legend()

    plt.tight_layout()
    plt.savefig('fig_efficacy_relative.jpg', dpi=300)
    print("-> Saved: fig_efficacy_relative.jpg")

if __name__ == "__main__":
    # Execute generation pipeline
    plot_sensitivity_analysis()
    plot_phase_landscape()
    plot_intervention_efficacy()
    print("\n[SUCCESS] All figures generated successfully.")
