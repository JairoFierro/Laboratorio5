import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.opt import SolverFactory

import pandas as pd
import seaborn as sns


# Inicialización lambas
lambdas = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6),
           (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]


f1_vec = []
f2_vec = []
solver = SolverFactory('glpk')

def hacer_model(lambda1, lambda2):

    # Crear modelo Pyomo
    Model = ConcreteModel()

    # Sets y parámetros del modelo
    numNodes = 5
    Model.N = RangeSet(1, numNodes)

    # Definición de las distancias (hops) entre nodos
    Model.h = Param(Model.N, Model.N, mutable=True)
    for i in Model.N:
        for j in Model.N:
            Model.h[i, j] = 999
    Model.h[1, 2] = 1
    Model.h[1, 3] = 1
    Model.h[2, 5] = 1
    Model.h[3, 4] = 1
    Model.h[4, 5] = 1


    # Conjuntos del problema
    # ['Alimentos', 'Medicinas', 'Equipos', 'Agua', 'Mantas']
    Model.R = Set(initialize=[1, 2, 3, 4, 5])
    Model.J = Set(initialize=[1, 2, 3, 4])
    Model.V = Set(initialize=[1, 2])
    Model.Z = Set(initialize=['A', 'B', 'C', 'D'])


    # Setear datos
    Model.recursos = { 1:[50, 5, 3,12], 2:[100, 2, 1, 15], 3:[120, 0.3, 0.5, 40],
                    4:[60, 6, 4, 15], 5:[40, 3, 2, 20]}

    Model.aviones = {1:[40, 35, 15, 0.020], 2:[50, 40, 20, 0.025],
                    3:[60, 45, 25, 0.030],4:[45, 38, 18, 0.022]}
    Model.caracteristicas = {"A":[800, 50, 1.2], "B":[1200, 70, 1.5],
                            "C":[1500, 100, 1.8], "D":[900, 80, 1.4]}

    Model.necesidades= {"A":[8, 6, 2, 0.6, 3], 
                        "B":[12, 9, 3, 0.9, 5],
                        "C":[16, 12, 4, 1.2, 7], 
                        "D":[10, 8, 2, 0.6, 4]}



    # Variables de decisión
    Model.x = Var(Model.R, Model.J, Model.V, Model.Z, domain=NonNegativeReals)
    Model.y = Var(Model.J, Model.V, Model.Z, domain=Binary)
    Model.u = Var(Model.J, domain=Binary)
    Model.b_incomp = Var(Model.J, Model.V, Model.Z, domain=Binary)

    # Parámetros derivados
    def impacto_rule(model, r): return model.recursos[r][0]
    def peso_rule(model, r): return model.recursos[r][1]
    def volumen_rule(model, r): return model.recursos[r][2]
    def mult_rule(model, z): return model.caracteristicas[z][2]
    def distancia_rule(model, z): return model.caracteristicas[z][0]
    def costo_fijo_rule(model, j): return model.aviones[j][2]
    def costo_var_rule(model, j): return model.aviones[j][3]

    # Obtener parametros del los datos
    Model.impacto = Param(Model.R, initialize=impacto_rule)
    Model.peso = Param(Model.R, initialize=peso_rule)
    Model.volumen = Param(Model.R, initialize=volumen_rule)
    Model.mult = Param(Model.Z, initialize=mult_rule)
    Model.distancia = Param(Model.Z, initialize=distancia_rule)
    Model.costo_fijo = Param(Model.J, initialize=costo_fijo_rule)
    Model.costo_variable = Param(Model.J, initialize=costo_var_rule)



    # Función objetivo: Maximizar impacto social
    Model.Z1 = sum(Model.impacto[r] * Model.x[r, j, v, z] * Model.mult[z] for r in Model.R for j in Model.J for v in Model.V for z in Model.Z)
    # Función objetivo: Costo total del transporte
    Model.Z2 = sum(Model.costo_fijo[j] * Model.u[j] for j in Model.J) + sum(Model.costo_variable[j] * Model.distancia[z] * Model.y[j, v, z] for j in Model.J for v in Model.V for z in Model.Z)

    # Normalizar y combinar las 2 funciones
    z1_max = 20000
    z2_max = 1500

    # Hacer normalización
    Model.obj = Objective(expr=lambda1 * (Model.Z1 / z1_max) - lambda2 * (Model.Z2 / z2_max), sense=maximize)


    # Restricciones
    # Restricción de pesos por viaje
    def peso_max_rule(model, j, v):
        return sum(model.recursos[r][1] * model.x[r, j, v, z] for r in model.R for z in model.Z) <= model.aviones[j][0]
    Model.peso_max = Constraint(Model.J, Model.V, rule=peso_max_rule)

    # Restricción de volumen por viaje
    def volumen_max_rule(model, j, v):
        return sum(model.recursos[r][2] * model.x[r, j, v, z] for r in model.R for z in model.Z) <= model.aviones[j][1]
    Model.volumen_max = Constraint(Model.J, Model.V, rule=volumen_max_rule)


    # Restricción de medicinas en el avion 1 (no se pueden transportar en el avión 1)
    def medicinas_no_avion1(model, v, z):
        return model.x[2, 1, v, z] == 0  # recurso 2 = Medicinas
    Model.restric_medicinas = Constraint(Model.V, Model.Z, rule=medicinas_no_avion1)

    # Incompatibilidad entre equipos médicos y agua potable (por viaje)
    M = 1000 

    def incompat_rule1(model, j, v, z):
        return model.x[3, j, v, z] <= M * model.b_incomp[j, v, z]

    def incompat_rule2(model, j, v, z):
        return model.x[4, j, v, z] <= M * (1 - model.b_incomp[j, v, z])

    Model.incompat1 = Constraint(Model.J, Model.V, Model.Z, rule=incompat_rule1)
    Model.incompat2 = Constraint(Model.J, Model.V, Model.Z, rule=incompat_rule2)


    Model.equipos = Var(Model.J, Model.V, Model.Z, domain=NonNegativeIntegers)

    # Restricción de un solo destino por viaje
    def una_zona_por_viaje(model, j, v):
        return sum(model.y[j, v, z] for z in model.Z) <= 1
    Model.una_zona = Constraint(Model.J, Model.V, rule=una_zona_por_viaje)

    # Restricción de disponibilidad de recursos por zona
    def disponibilidad_max(model, r):
        return sum(model.x[r, j, v, z] for j in model.J for v in model.V for z in model.Z) <= model.recursos[r][3]
    Model.disponibilidad_total = Constraint(Model.R, rule=disponibilidad_max)

    # Restricción para activar un vuelo
    def activar_u(model, j):
        return sum(model.y[j, v, z] for v in model.V for z in model.Z) <= 2 * model.u[j]
    Model.activacion_avion = Constraint(Model.J, rule=activar_u)

    # Si se transporta algo en el vuelo (j,v) a cualquier zona, entonces y[j,v,z] = 1
    def activar_vuelo(model, r, j, v, z):
        return model.x[r,j,v,z] <= M * model.y[j,v,z]
    Model.activa_y = Constraint(Model.R, Model.J, Model.V, Model.Z, rule=activar_vuelo)

    # Si algún y[j,v,z] está activo para un j, entonces u[j] = 1
    def activar_avion(model, j):
        return sum(model.y[j,v,z] for v in model.V for z in model.Z) <= M * model.u[j]
    Model.activa_u = Constraint(Model.J, rule=activar_avion)

    return Model


f1_vec = []
f2_vec = []
data = []
# Iterar lambdas y resolver el modelo para cada combinación
for l1, l2 in lambdas:
    model = hacer_model(l1, l2)
    results = solver.solve(model, tee=True)

    # Revisa que se haya encontrado una solución óptima
    if (results.solver.status == SolverStatus.ok) and \
       (results.solver.termination_condition == TerminationCondition.optimal):
        z1_val = value(model.Z1)
        z2_val = value(model.Z2)
        f1_vec.append(z1_val)
        f2_vec.append(z2_val)
        print(f"λ1 = {l1}, λ2 = {l2} → Z1 = {z1_val:.2f}, Z2 = {z2_val:.2f}")
        
        for r in model.R:
            for j in model.J:
                for v in model.V:
                    for z in model.Z:
                        cantidad = value(model.x[r, j, v, z])
                        if cantidad > 0:
                            data.append({
                                'Recurso': r,
                                'Avión': j,
                                'Viaje': v,
                                'Zona': z,
                                'Cantidad': cantidad
                            })

    else:
        print(f"λ1 = {l1}, λ2 = {l2}: No solución óptima encontrada")

# Gráfica del frente óptimo de pareto
plt.plot(f2_vec, f1_vec, marker='o')
plt.title('Frente de Pareto (Método de Sumas Ponderadas)')
plt.xlabel('Costo total (Z2)')
plt.ylabel('Impacto social (Z1)')
plt.grid(True)
plt.show()


# Convertir a DataFrame para graficar
df = pd.DataFrame(data)

# Crear gráfica tipo heatmap o diagrama de barras por zona, avión y viaje
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Zona', y='Cantidad', hue='Recurso')
plt.title(f'Distribución de recursos por zona (λ1={l1}, λ2={l2})')
plt.ylabel('Toneladas')
plt.xlabel('Zona')
plt.legend(title='Recurso')
plt.tight_layout()
plt.show()