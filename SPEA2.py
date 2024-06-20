import numpy as np
import matplotlib.pyplot as plt
import time 
import csv
import os

class Individual:
    def __init__(self, x):
        self.x = x
        self.f1 = self.objective1()
        self.f2 = self.objective2()
        self.strength = 0
        self.raw_fitness = 0
        self.density = 0
        self.fitness = 0

    def objective1(self):
        return np.sin(self.x) + 1

    def objective2(self):
        return np.cos(self.x) + 1

    def dominates(self, other):
        return (self.f1 <= other.f1 and self.f2 <= other.f2) and (self.f1 < other.f1 or self.f2 < other.f2)

class SPEA2:
    def __init__(self, pop_size, bounds, generations):
        self.pop_size = pop_size
        self.bounds = bounds
        self.generations = generations

    def initialize_population(self):
        self.population = [Individual(np.random.uniform(self.bounds[0], self.bounds[1])) for _ in range(self.pop_size)]
        self.archive = []

    def calculate_strength(self):
        for ind in self.population:
            ind.strength = sum(1 for other in self.population if ind.dominates(other))

    def calculate_raw_fitness(self):
        for ind in self.population:
            ind.raw_fitness = sum(other.strength for other in self.population if other.dominates(ind))

    def calculate_density(self):
        for ind in self.population:
            distances = np.array([np.linalg.norm([ind.f1 - other.f1, ind.f2 - other.f2]) for other in self.population])
            distances = np.sort(distances)
            k = int(np.sqrt(len(self.population)))  # Typically the k-th nearest neighbor
            ind.density = 1 / (distances[k] + 2)

    def calculate_fitness(self):
        for ind in self.population:
            ind.fitness = ind.raw_fitness + ind.density

    def environmental_selection(self):
        combined = self.population + self.archive

        # Recalcular métricas de aptitud para la población combinada
        for ind in combined:
            ind.strength = 0
            ind.raw_fitness = 0
            ind.density = 0
            ind.fitness = 0

        self.calculate_strength()
        self.calculate_raw_fitness()
        self.calculate_density()
        self.calculate_fitness()

        # Filtrar individuos con fitness menor a 1 y ordenar
        combined = [ind for ind in combined if ind.fitness < 1]
        combined.sort(key=lambda ind: ind.fitness)

        # Asegurar que el archivo no exceda el tamaño de la población
        self.archive = combined[:self.pop_size]

        if len(self.archive) > self.pop_size:
            self.archive = self.truncate_archive(self.archive)

    def truncate_archive(self, archive):
        while len(archive) > self.pop_size:
            densities = [ind.density for ind in archive]
            max_density_index = densities.index(max(densities))
            archive.pop(max_density_index)
        return archive

    def tournament_selection(self):
        ind1, ind2 = np.random.choice(self.archive, 2)
        if ind1.fitness < ind2.fitness:
            return ind1
        return ind2

    def crossover(self, parent1, parent2):
        alpha = np.random.rand()
        child1_x = alpha * parent1.x + (1 - alpha) * parent2.x
        child2_x = alpha * parent2.x + (1 - alpha) * parent1.x
        return Individual(child1_x), Individual(child2_x)

    def mutation(self, ind):
        if np.random.rand() < 0.9:
            ind.x += np.random.normal(0, 0.1)
            ind.x = np.clip(ind.x, self.bounds[0], self.bounds[1])
            ind.f1 = ind.objective1()
            ind.f2 = ind.objective2()

    def evolve(self):
        for gen in range(self.generations):
            self.calculate_strength()
            self.calculate_raw_fitness()
            self.calculate_density()
            self.calculate_fitness()
            self.environmental_selection()

            new_population = []
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child1, child2 = self.crossover(parent1, parent2)
                self.mutation(child1)
                self.mutation(child2)
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)

            self.population = new_population

    def get_pareto_front(self):
        pareto_front = [ind for ind in self.archive if ind.raw_fitness == 0]
        return pareto_front
    
    def find_best_solution(self, pareto_front):
        best_solution = min(pareto_front, key=lambda ind: ind.f1 + ind.f2)
        return best_solution


    def plot_pareto_front(self, pareto_front, best_solution):
        f1 = [ind.f1 for ind in pareto_front]
        f2 = [ind.f2 for ind in pareto_front]

        plt.figure(figsize=(8, 8))
        plt.scatter(f1, f2, label='Pareto Front')
        plt.scatter(best_solution.f1, best_solution.f2, color='red', s=100, label='Best Solution')
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.title('Pareto Front')
        plt.legend()
        plt.tight_layout(rect=[0, 0.01, 1, 0.91])
        plt.figtext(0.12, 0.1, f'Best Solution: \nf1: {best_solution.f1},\nf2: {best_solution.f2}')
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

if __name__ == "__main__":
    pop_size = 100
    generations = 100 
    bounds = (0, 2 * np.pi)

    spea2 = SPEA2(pop_size=pop_size, bounds=bounds, generations=generations)
    one = False
    if one == True:
        spea2.initialize_population()
        spea2.evolve()
        pareto_front = spea2.get_pareto_front()
        best_solution = spea2.find_best_solution(pareto_front)
        spea2.plot_pareto_front(pareto_front, best_solution)

    def multiple_runs(runs):
        file_path =os.path.join('./Statistics and results',f'SPEA2 Performance in {runs} runs.csv')
        graph_path =os.path.join('./Statistics and results',f'SPEA2 Performance in {runs} runs.png')
        best_solutions = []
        times = []

        for _ in range(runs):
            start_time = time.time()
            spea2.initialize_population()
            spea2.evolve()
            pareto_front = spea2.get_pareto_front()
            best_solution = spea2.find_best_solution(pareto_front)
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
    
        spea2.plot_performance(runs, times,graph_path,save=True)

    multiple = False
    if multiple == True:
        multiple_runs(10)
