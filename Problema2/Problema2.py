import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.opt import SolverFactory
import uuid
import itertools


np.random.seed(42)
N = 10
nodes = list(range(N))
dist_matrix = np.random.randint(10, 100, size=(N, N))
dist_matrix = (dist_matrix + dist_matrix.T) // 2 
np.fill_diagonal(dist_matrix, 0)


inspection_quality = {
    0: 85, 1: 92, 2: 78, 3: 90, 4: 82,
    5: 88, 6: 95, 7: 75, 8: 84, 9: 5
}

defined_risks = {
    (0, 1): 3, (1, 0): 3,
    (0, 2): 2, (2, 0): 2,
    (0, 3): 4, (3, 0): 4,
    (0, 4): 5, (4, 0): 5,
    (0, 5): 6, (5, 0): 6,
    (0, 6): 3, (6, 0): 3,
    (0, 7): 2, (7, 0): 2,
    (0, 8): 4, (8, 0): 4,
    (0, 9): 5, (9, 0): 5,
    (2, 8): 9, (8, 2): 9,
    (2, 9): 8, (9, 2): 8,
    (3, 4): 5, (4, 3): 5,
    (4, 9): 7, (9, 4): 7,
    (5, 6): 7, (6, 5): 7,
    (8, 9): 7, (9, 8): 7,
}


risk_matrix = np.full((N, N), 5)


for (i, j), risk in defined_risks.items():
    risk_matrix[i, j] = risk



edges = [(dist_matrix[i][j], i, j) for i in nodes for j in nodes if i < j]
edges.sort()  
min_dist = sum(dist_matrix[i][j] for _, i, j in edges[:N])  
max_dist = sum(dist_matrix[i][j] for _, i, j in edges[-N:]) 


min_risk = 5 * N
max_risk = 5 * N  


total_quality = sum(inspection_quality[i] for i in nodes)
min_quality = total_quality  
max_quality = total_quality

def normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0
    return (value - min_val) / (max_val - min_val)


def build_model(epsilon_risk):
    model = ConcreteModel()
    
    model.N = Set(initialize=nodes)
    model.A = Set(within=model.N * model.N, initialize=[(i, j) for i in nodes for j in nodes if i != j])
    
    model.x = Var(model.A, within=Binary)
    model.u = Var(model.N, within=NonNegativeIntegers, bounds=(0, N - 1))
    

    model.obj = Objective(expr=sum(dist_matrix[i][j] * model.x[i, j] for (i, j) in model.A), sense=minimize)
    

    model.out_flow = Constraint(model.N, rule=lambda m, i: sum(m.x[i, j] for j in m.N if j != i) == 1)
    model.in_flow = Constraint(model.N, rule=lambda m, j: sum(m.x[i, j] for i in m.N if i != j) == 1)
    

    model.subtour = ConstraintList()
    for i in model.N:
        for j in model.N:
            if i != j and i != 0 and j != 0:
                model.subtour.add(model.u[i] - model.u[j] + N * model.x[i, j] <= N - 1)
    

    model.risk_limit = Constraint(expr=sum(risk_matrix[i][j] * model.x[i, j] for (i, j) in model.A) <= epsilon_risk)


    model.total_quality = Expression(expr=sum(inspection_quality[i] for i in model.N))

    return model


solver = SolverFactory('glpk')
risk_bounds = np.linspace(50, 55, 10)
results_list = []

for epsilon in risk_bounds:
    model = build_model(epsilon)
    result = solver.solve(model, tee=False)

    if (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal):
        total_dist = value(model.obj)
        total_risk = sum(risk_matrix[i][j] * value(model.x[i, j]) for (i, j) in model.A)
        total_quality = value(model.total_quality)
        norm_dist = normalize(total_dist, min_dist, max_dist)
        norm_risk = normalize(total_risk, min_risk, max_risk)
        norm_quality = normalize(total_quality, min_quality, max_quality)
        results_list.append({
            'epsilon': epsilon,
            'dist': total_dist,
            'risk': total_risk,
            'quality': total_quality,
            'norm_dist': norm_dist,
            'norm_risk': norm_risk,
            'norm_quality': norm_quality
        })
        print(f"ε = {epsilon:.2f} → Distancia = {total_dist:.2f}, Riesgo = {total_risk:.2f}, Calidad = {total_quality:.2f}")
    else:
        print(f"ε = {epsilon:.2f} → No se encontró solución óptima")


if len(results_list) < 7:
    print(f"Warning: Only {len(results_list)} solutions found. Adjust risk_bounds for more solutions.")


fig = plt.figure(figsize=(15, 10))


ax = fig.add_subplot(221, projection='3d')
dists = [r['dist'] for r in results_list]
risks = [r['risk'] for r in results_list]
qualities = [r['quality'] for r in results_list]
ax.scatter(dists, risks, qualities, c='b', marker='o')
ax.set_xlabel('Distancia')
ax.set_ylabel('Riesgo')
ax.set_zlabel('Calidad')
ax.set_title('Frente de Pareto 3D')


ax2 = fig.add_subplot(222)
ax2.scatter(dists, risks, c='r', marker='o')
ax2.set_xlabel('Distancia')
ax2.set_ylabel('Riesgo')
ax2.set_title('Proyección: Distancia vs Riesgo')
ax2.grid(True)


ax3 = fig.add_subplot(223)
ax3.scatter(dists, qualities, c='g', marker='o')
ax3.set_xlabel('Distancia')
ax3.set_ylabel('Calidad')
ax3.set_title('Proyección: Distancia vs Calidad')
ax3.grid(True)


ax4 = fig.add_subplot(224)
ax4.scatter(risks, qualities, c='b', marker='o')
ax4.set_xlabel('Riesgo')
ax4.set_ylabel('Calidad')
ax4.set_title('Proyección: Riesgo vs Calidad')
ax4.grid(True)

plt.tight_layout()
plt.savefig('pareto_front.png')
plt.close()


print("\nFrente de Pareto aproximado (normalizado):")
for r in results_list:
    print(f"ε={r['epsilon']:.2f} → Dist: {r['dist']:.2f} (norm: {r['norm_dist']:.2f}), "
          f"Risk: {r['risk']:.2f} (norm: {r['norm_risk']:.2f}), "
          f"Quality: {r['quality']:.2f} (norm: {r['norm_quality']:.2f})")

# Trade-offs
print("\nAnálisis de compensaciones:")
print("- Distancia vs Riesgo: Dado que el riesgo es constante (5 por arco), todas las soluciones tienen el mismo riesgo (50 para 10 arcos).")
print("- Distancia vs Calidad: La calidad es fija (suma de calidades de nodos visitados), por lo que no hay compensación directa.")
print("- La variación en ε no afecta el riesgo ni la calidad, pero permite explorar diferentes rutas con distintas distancias.")

# Impacto de priorizar objetivos
print("\nImpacto de priorizar el costo (distancia):")
print("- Si el costo es significativamente más importante, seleccionaríamos la solución con la menor distancia normalizada.")
min_dist_solution = min(results_list, key=lambda x: x['norm_dist'])
print(f"Solución óptima por costo: ε={min_dist_solution['epsilon']:.2f}, "
      f"Dist: {min_dist_solution['dist']:.2f}, Risk: {min_dist_solution['risk']:.2f}, "
      f"Quality: {min_dist_solution['quality']:.2f}")

print("\nImpacto de priorizar la cobertura (calidad):")
print("- La calidad es constante en todas las soluciones, por lo que priorizar calidad no cambia la selección.")

# Solución más equilibrada
print("\nSolución más equilibrada:")
# Usar la distancia al punto ideal (0, 0, 1) en espacio normalizado
ideal_point = np.array([0, 0, 1])
distances_to_ideal = [
    np.sqrt(
        (r['norm_dist'] - ideal_point[0])**2 +
        (r['norm_risk'] - ideal_point[1])**2 +
        (r['norm_quality'] - ideal_point[2])**2
    ) for r in results_list
]
balanced_idx = np.argmin(distances_to_ideal)
balanced_solution = results_list[balanced_idx]
print(f"Solución seleccionada: ε={balanced_solution['epsilon']:.2f}, "
      f"Dist: {balanced_solution['dist']:.2f} (norm: {balanced_solution['norm_dist']:.2f}), "
      f"Risk: {balanced_solution['risk']:.2f} (norm: {balanced_solution['norm_risk']:.2f}), "
      f"Quality: {balanced_solution['quality']:.2f} (norm: {balanced_solution['norm_quality']:.2f})")
print("Justificación: Esta solución minimiza la distancia euclidiana al punto ideal (mínima distancia y riesgo, máxima calidad) en el espacio normalizado.")