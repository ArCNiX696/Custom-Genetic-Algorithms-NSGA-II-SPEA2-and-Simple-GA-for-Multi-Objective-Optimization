import numpy as np
import matplotlib.pyplot as plt
import time 
import csv
import os

class Individual:
    def __init__(self, x1, x2):
        self.x1 = max(0, x1)
        self.x2 = max(0, x2)
        self.f1 = self.objective1()
        self.f2 = self.objective2()
        self.constraints = self.evaluate_constraints()
        self.penalty = self.calculate_penalty()

    def objective1(self):
        return -self.x1 + 3 * self.x2

    def objective2(self):
        return 3 * self.x1 + self.x2

    def evaluate_constraints(self):
        g1 = self.x1 + 2 * self.x2 - 2
        g2 = 2 * self.x1 + self.x2 - 2
        return [g1, g2]

    def calculate_penalty(self):
        penalty = 0
        for g in self.constraints:
            if g > 0:
                penalty += g
        return penalty

    def dominates(self, other):
        if self.penalty == 0 and other.penalty == 0:
            return (self.f1 >= other.f1 and self.f2 >= other.f2 and 
                    (self.f1 > other.f1 or self.f2 > other.f2))
        else:
            return self.penalty < other.penalty

class Population:
    def __init__(self, size, bounds):
        self.size = size
        self.bounds = bounds
        self.individuals = [self.create_individual() for _ in range(size)]
        self.pareto_front = []
        self.best_chromosomes = []

    def create_individual(self):
        x1 = np.random.uniform(max(self.bounds[0][0], 0), self.bounds[0][1])
        x2 = np.random.uniform(max(self.bounds[1][0], 0), self.bounds[1][1])
        return Individual(x1, x2)

    def evolve(self, generations):
        history = []

        for generation in range(generations):
            self.selection()
            self.crossover()
            self.mutation()
            self.repair()
            self.update_pareto_front()
            self.best_chromosome(generation)
            history.append(self.pareto_front[:])
        
        return history

    def selection(self):
        new_individuals = []
        for _ in range(self.size):
            ind1, ind2 = np.random.choice(self.individuals, 2)
            if ind1.dominates(ind2):
                new_individuals.append(ind1)
            elif ind2.dominates(ind1):
                new_individuals.append(ind2)
            else:
                new_individuals.append(np.random.choice([ind1, ind2]))
            
        self.individuals = new_individuals

    def crossover(self):
        new_individuals = []
        for _ in range(self.size // 2):
            parent1, parent2 = np.random.choice(self.individuals, 2)
            alpha = np.random.rand()
            child1_x1 = alpha * parent1.x1 + (1 - alpha) * parent2.x1
            child1_x2 = alpha * parent1.x2 + (1 - alpha) * parent2.x2
            child2_x1 = alpha * parent2.x1 + (1 - alpha) * parent1.x1
            child2_x2 = alpha * parent2.x2 + (1 - alpha) * parent1.x2
            new_individuals.append(Individual(child1_x1, child1_x2))
            new_individuals.append(Individual(child2_x1, child2_x2))
        self.individuals.extend(new_individuals)

    def mutation(self):
        for ind in self.individuals:
            if np.random.rand() < 0.15:  
                delta = np.random.uniform(-0.1, 0.1)
                if np.random.rand() < 0.5:
                    ind.x1 = max(0, ind.x1 + delta)
                else:
                    ind.x2 = max(0, ind.x2 + delta)
                
                ind.f1 = ind.objective1()
                ind.f2 = ind.objective2()
                ind.constraints = ind.evaluate_constraints()
                ind.penalty = ind.calculate_penalty()
                # print(f"\n{'-'*100}\n")
                # print(f'ind.x1 = {ind.x1}\n')
                # print(f'ind.x2 = {ind.x2}\n')
                # print(f'ind.f1 = {ind.f1}\n')
                # print(f'ind.f2 = {ind.f2}\n')
                # print(f'ind.constraints = {ind.constraints}\n')
                # print(f'ind.penalty = {ind.penalty}\n')
                
    def repair(self):
        for ind in self.individuals:
            if ind.constraints[0] > 0:
                ind.x2 = max(0, (2 - ind.x1) / 2)
            if ind.constraints[1] > 0:
                ind.x2 = max(0, (2 - 2 * ind.x1) / 1)
            ind.x1 = max(0, ind.x1)
            ind.x2 = max(0, ind.x2)
            ind.f1 = ind.objective1()
            ind.f2 = ind.objective2()
            ind.constraints = ind.evaluate_constraints()
            ind.penalty = ind.calculate_penalty()

    def update_pareto_front(self):
        self.pareto_front = []
        for ind in self.individuals:
            if not any(other.dominates(ind) for other in self.individuals):
                self.pareto_front.append(ind)

    def get_pareto_front(self):
        return self.pareto_front

    def best_chromosome(self, generation):
        best = max(self.individuals, key=lambda ind: (ind.f1 + ind.f2) / (1 + ind.penalty))
        self.best_chromosomes.append(best)
        print(f"Generation {generation + 1}: Best chromosome x1 = {best.x1:.4f}, x2 = {best.x2:.4f}, f1 = {best.f1:.4f}, f2 = {best.f2:.4f}, penalty = {best.penalty:.4f}")
        
class GENOCOP:
    def __init__(self, pop_size, bounds, generations):
        self.pop_size = pop_size
        self.bounds = bounds
        self.generations = generations
        self.population = Population(pop_size, bounds)

    def run(self):
        history = self.population.evolve(self.generations)
        pareto_front = self.population.get_pareto_front()
        return history, pareto_front
    
    def find_best_solution(self,pareto_front):
        best_solution = max(pareto_front, key=lambda ind: (ind.f1 + ind.f2) / (1 + ind.penalty))
        return best_solution

    def plot_pareto_front(self, pareto_front):
        import matplotlib.pyplot as plt
        f1 = [ind.f1 for ind in pareto_front]
        f2 = [ind.f2 for ind in pareto_front]
        
        best_solution = self.find_best_solution(pareto_front)
        plt.figure(figsize=(8,8))
        plt.scatter(f1, f2, label='Pareto Front')
        plt.scatter(best_solution.f1, best_solution.f2, color='red', s=100, label='Best Solution')
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.title('Pareto Front')
        plt.legend()
        plt.tight_layout(rect=[0.0001,0.002,1,0.91])
        plt.figtext(0.1,0.1,f'best_solution: \nf1: {(best_solution.f1)},\nf2: {(best_solution.f2)}')
        plt.show()

    def plot_performance(self, runs, times,path,save=False):
        runs = range(1, runs + 1)

        plt.figure(figsize=(18, 8))
        plt.plot(runs, times, label='Time per run', color='red', marker='o')
        plt.xlabel('Runs')
        plt.ylabel('Time')
        plt.title('Time performance')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        if save == True:
            plt.savefig(path)
        
        plt.show()

    def plot_feasible_area(self):
        x1_values = np.linspace(0, 2, 400)
        x2_values = np.linspace(0, 2, 400)
        X1, X2 = np.meshgrid(x1_values, x2_values)

        G1 = X1 + 2 * X2 - 2
        G2 = 2 * X1 + X2 - 2

        feasible_area = np.logical_and(G1 <= 0, G2 <= 0)

        plt.figure(figsize=(10, 8))
        plt.contourf(X1, X2, feasible_area, levels=[0, 1], colors=['lightgrey'], alpha=0.5)

        cont1 = plt.contour(X1, X2, G1, levels=[0], colors='red', linestyles='dashed')
        cont2 = plt.contour(X1, X2, G2, levels=[0], colors='blue', linestyles='dashed')
        h1, _ = cont1.legend_elements()
        h2, _ = cont2.legend_elements()
        plt.legend([h1[0], h2[0]], ['g1(x1, x2) = x1 + 2 * x2 - 2', 'g2(x1, x2) = 2 * x1 + x2 - 2'])

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Feasible Area')
        plt.grid(True)
        plt.show()

    def plot_convergence(self, history):
        generations = len(history)
        num_non_dominated = [len(front) for front in history]
        spreads = [np.std([ind.f1 + ind.f2 for ind in front]) for front in history]

        iterations = np.arange(generations)

        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(iterations, num_non_dominated, marker='o', color='blue')
        plt.xlabel('Generation')
        plt.ylabel('Number of Non-Dominated Solutions')
        plt.title('Evolution of Number of Non-Dominated Solutions')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(iterations, spreads, marker='x', color='red')
        plt.xlabel('Generation')
        plt.ylabel('Diversity (Spread)')
        plt.title('Evolution of Diversity (Spread)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    generations=200

    genocop = GENOCOP(pop_size=500, bounds=[(0, 1), (0, 1)], generations=generations)
    # genocop.plot_feasible_area()
    one = True
    if one == True:
        history, pareto_front = genocop.run()
        genocop.plot_pareto_front(pareto_front)

    def multiple_runs(runs):
        file_path =os.path.join('./Statistics and results',f'GENOCOP I Performance in {runs} runs.csv')
        graph_path =os.path.join('./Statistics and results',f'GENOCOP I Performance in {runs} runs.png')
        best_solutions = []
        times = []

        for _ in range(runs):
            start_time = time.time()
            _, pareto_front = genocop.run()
            best_solution = genocop.find_best_solution(pareto_front)
            best_solutions.append((round(best_solution.f1,4), round(best_solution.f2,4)))
            finished_time = time.time() - start_time
            times.append(round(finished_time,4))

        mean_time = round(np.mean(times),4)
        std_dev_time =round(np.std(times),4)

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Run'] + [f'Run {i+1}' for i in range(runs)])
            writer.writerow(['Best Solutions'] + best_solutions)
            writer.writerow(['Runtime (s)'] + times)
            writer.writerow(['Mean Runtime (s)', mean_time])
            writer.writerow(['Standard Deviation of Runtime (s)', std_dev_time])
    
        genocop.plot_performance(runs, times,graph_path,save=True)

    multiple = False
    if multiple == True:
        multiple_runs(10)

    
    
