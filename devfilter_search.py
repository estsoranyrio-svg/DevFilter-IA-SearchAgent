"""
Proyecto Integrador - Fase 2: Implementación Algorítmica y Análisis de Complejidad en Búsqueda
Caso C: DevFilter IA (Preselección Automatizada de Talento Tecnológico)
Universidad Militar Nueva Granada - FAEDIS
Estudiante: Sorany del Río Loaiza (Cód. 200569)
Carácter: Individual
"""

import time
from collections import deque
import heapq

# =====================================================================
# 1. CLASE DEL PROBLEMA (Formalización P = {S, s0, ACTIONS, RESULT, GOAL, c})
# =====================================================================
class DevFilterProblem:
    def __init__(self, initial_state=None, goal_thresholds=None):
        # s0 = (fase, techs, exp_val, repo_score)
        self.initial = initial_state or ('CV_RECIBIDO', 0, 0, 0)
        self.goal_thresholds = goal_thresholds or {'techs': 2, 'exp': 1, 'repo': 70}
        
        # Costos por acción c(s, a, s') en ms de CPU
        self.action_costs = {
            'VALIDAR_TECHS': 10,
            'DESCARTAR_CV': 5,
            'EVALUAR_EXP': 15,
            'DESCARTAR_EXP': 5,
            'AUDITAR_GITHUB': 30,
            'AUDITAR_INSUFICIENTE': 5
        }

    def actions(self, state):
        fase, techs, exp, repo = state
        if fase == 'CV_RECIBIDO':
            return ['VALIDAR_TECHS', 'DESCARTAR_CV']
        elif fase == 'TECHS_OK':
            return ['EVALUAR_EXP', 'DESCARTAR_EXP']
        elif fase == 'EXP_OK':
            return ['AUDITAR_GITHUB', 'AUDITAR_INSUFICIENTE']
        return []

    def result(self, state, action):
        fase, techs, exp, repo = state
        if action == 'VALIDAR_TECHS':
            return ('TECHS_OK', 3, exp, repo)
        elif action == 'DESCARTAR_CV':
            return ('DESCARTADO', techs, exp, repo)
        elif action == 'EVALUAR_EXP':
            return ('EXP_OK', techs, 1, repo)
        elif action == 'DESCARTAR_EXP':
            return ('DESCARTADO', techs, exp, repo)
        elif action == 'AUDITAR_GITHUB':
            return ('PRESELECCIONADO', techs, exp, 85)
        elif action == 'AUDITAR_INSUFICIENTE':
            return ('DESCARTADO', techs, exp, 40)
        return state

    def is_goal(self, state):
        fase, techs, exp, repo = state
        return (fase == 'PRESELECCIONADO' and 
                techs >= self.goal_thresholds['techs'] and 
                exp >= self.goal_thresholds['exp'] and 
                repo >= self.goal_thresholds['repo'])

    def step_cost(self, state, action, next_state):
        return self.action_costs.get(action, 1)

    # Heurística Admisible: h(n) <= h*(n)
    def h(self, node):
        fase, techs, exp, repo = node.state
        if self.is_goal(node.state):
            return 0
        h_val = 0
        if fase == 'CV_RECIBIDO':
            h_val += 10
        if fase in ['CV_RECIBIDO', 'TECHS_OK']:
            h_val += 15
        if fase in ['CV_RECIBIDO', 'TECHS_OK', 'EXP_OK']:
            h_val += 30
        return h_val

# =====================================================================
# 2. ESTRUCTURA DE NODO
# =====================================================================
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0 if parent is None else parent.depth + 1

    def expand(self, problem):
        return [
            Node(
                state=problem.result(self.state, act),
                parent=self,
                action=act,
                path_cost=self.path_cost + problem.step_cost(self.state, act, problem.result(self.state, act))
            )
            for act in problem.actions(self.state)
        ]

    def __lt__(self, other):
        return self.path_cost < other.path_cost

# =====================================================================
# 3. ALGORITMOS DE BÚSQUEDA
# =====================================================================
def breadth_first_search(problem):
    start_time = time.perf_counter()
    node = Node(problem.initial)
    if problem.is_goal(node.state):
        return node, 0, (time.perf_counter() - start_time) * 1000
    frontier = deque([node])
    explored = set()
    nodes_expanded = 0

    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        nodes_expanded += 1
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.is_goal(child.state):
                    elapsed = (time.perf_counter() - start_time) * 1000
                    return child, nodes_expanded, elapsed
                frontier.append(child)
    return None, nodes_expanded, (time.perf_counter() - start_time) * 1000

def depth_first_search(problem):
    start_time = time.perf_counter()
    frontier = [Node(problem.initial)]
    explored = set()
    nodes_expanded = 0

    while frontier:
        node = frontier.pop()
        nodes_expanded += 1
        if problem.is_goal(node.state):
            elapsed = (time.perf_counter() - start_time) * 1000
            return node, nodes_expanded, elapsed
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
    return None, nodes_expanded, (time.perf_counter() - start_time) * 1000

def astar_search(problem):
    start_time = time.perf_counter()
    start_node = Node(problem.initial)
    count = 0
    frontier = [(start_node.path_cost + problem.h(start_node), count, start_node)]
    explored = {}
    nodes_expanded = 0

    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node.state in explored and explored[node.state] <= node.path_cost:
            continue
        explored[node.state] = node.path_cost

        if problem.is_goal(node.state):
            elapsed = (time.perf_counter() - start_time) * 1000
            return node, nodes_expanded, elapsed

        nodes_expanded += 1
        for child in node.expand(problem):
            f_cost = child.path_cost + problem.h(child)
            if child.state not in explored or child.path_cost < explored[child.state]:
                count += 1
                heapq.heappush(frontier, (f_cost, count, child))
                
    return None, nodes_expanded, (time.perf_counter() - start_time) * 1000

# =====================================================================
# 4. EJECUCIÓN Y REGISTRO EXPERIMENTAL (BENCHMARK)
# =====================================================================
if __name__ == "__main__":
    problem = DevFilterProblem()
    print("=" * 65)
    print("BENCHMARK DE RENDIMIENTO (Caso C: DevFilter IA)")
    print("Estudiante: Sorany del Río Loaiza - Cód. 200569")
    print("=" * 65)
    
    sol_bfs, exp_bfs, t_bfs = breadth_first_search(problem)
    sol_dfs, exp_dfs, t_dfs = depth_first_search(problem)
    sol_ast, exp_ast, t_ast = astar_search(problem)

    print(f"BFS -> Nodos: {exp_bfs} | Tiempo: {t_bfs:.4f} ms | Costo g(n): {sol_bfs.path_cost if sol_bfs else 'N/A'}")
    print(f"DFS -> Nodos: {exp_dfs} | Tiempo: {t_dfs:.4f} ms | Costo g(n): {sol_dfs.path_cost if sol_dfs else 'N/A'}")
    print(f"A*  -> Nodos: {exp_ast} | Tiempo: {t_ast:.4f} ms | Costo g(n): {sol_ast.path_cost if sol_ast else 'N/A'}")
    print("=" * 65)
