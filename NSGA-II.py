import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import time

class Individual:
    def __init__(self,x):
        self.x = x
        self.f1 = self.objective1()
        self.f2 = self.objective2()
        self.rank = None
        self.crowding_distance = 0
        self.best = None

    def objective1(self):
        return np.sin(self.x) + 1

    def objective2(self):
        return np.cos(self.x) + 1

class NSGA:
    def __init__(self,size,bounds,generations):
        self.size = size #Population number
        self.bounds = bounds
        self.population = [Individual(np.random.uniform(bounds[0], bounds[1])) for _ in range(size)]
        self.generations = generations

    def dominates(self,ind1, ind2):#If ind1 dominates ind2
        return ((ind1.f1 <= ind2.f1 and ind1.f2 <= ind2.f2) 
                and (ind1.f1 < ind2.f1 or ind1.f2 < ind2.f2))
    
    def fast_non_dominated_sort(self):
        fronts = [[]]
        for p in self.population:
            p.dominated_solutions = []
            p.domination_count = 0
            for q in self.population:
                if self.dominates(p,q):
                    p.dominated_solutions.append(q)
                elif self.dominates(q,p):
                    p.domination_count +=1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -=1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            
            i +=1
            fronts.append(next_front)
        return fronts[:-1]
        
    def calculate_crowding_distance(self,front):
        if len(front) == 0:
            return
        for ind in front:
            ind.crowding_distance = 0
        for m in range(2):
            if m == 0:
                front.sort(key=lambda ind: ind.f1)
                min_f = front[0].f1
                max_f = front[-1].f1
            else:
                front.sort(key=lambda ind: ind.f2)
                min_f = front[0].f1
                max_f = front[-1].f2

            if max_f == min_f:
                continue  # Avoid division by zero

            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            for i in range(1, len(front) -1):
                if m == 0:
                    front[i].crowding_distance += (front[i + 1].f1 -front[i - 1].f1) / (max_f - min_f)
                
                else:
                    front[i].crowding_distance += (front[i + 1].f2 -front[i - 1].f2) / (max_f - min_f)

    def binary_tournament(self,population):
        ind1 , ind2 =np.random.choice(population,2)
        if ind1.rank < ind2.rank:
            return ind1
        elif ind1.rank > ind2.rank:
            return ind2
        elif ind1.crowding_distance < ind2.crowding_distance:
            return ind1
        else:
            return ind2
        
    def arithmetic_crossover(self,parent1,parent2):
        alpha = np.random.rand()
        child1_x = alpha * parent1.x + (1 - alpha) * parent2.x
        child2_x = alpha * parent2.x + (1 - alpha) * parent1.x
        return Individual(child1_x), Individual(child2_x)
    
    def gaussian_mutation(self,ind):
        if np.random.rand() < 0.15:
            ind.x += np.random.normal(0,0.1)
            ind.x = np.clip(ind.x , self.bounds[0], self.bounds[1])
            ind.f1 = ind.objective1()
            ind.f2 = ind.objective2()
    
    def best_solution(self,pareto_front):
        best_solution = min(pareto_front, key=lambda ind: ind.f1 + ind.f2)
        return best_solution
    
    def evolve(self,verbose=False):
        history = []

        for generation in range(self.generations):
            fronts = self.fast_non_dominated_sort()
            for front in fronts:
                self.calculate_crowding_distance(front)
            new_population = []

            while len(new_population) < self.size:
                parent1 = self.binary_tournament(self.population)
                parent2 = self.binary_tournament(self.population)
                child1,child2 = self.arithmetic_crossover(parent1,parent2)
                self.gaussian_mutation(child1)
                self.gaussian_mutation(child2)
                new_population.append(child1)

                if len(new_population) < self.size:
                    new_population.append(child2)

            self.population = new_population
            history.append(fronts[0])
           
        self.best_sol = self.best_solution(self.population)

        if verbose == True:
                print(f"Generation {generation}: Best chromosome x = {self.best.x:.4f}, f1 = {self.best.f1:.4f}, f2 = {self.best.f2:.4f}")

        return self.population,history,fronts[0],self.best_sol

    def plot_pareto_front(self,pareto_front,best_solution):
        f1 = [ind.f1 for ind in pareto_front]
        f2 = [ind.f2 for ind in pareto_front]

        plt.figure(figsize=(8,8))
        plt.scatter(f1,f2,label='Pareto Front')
        plt.scatter(best_solution.f1, best_solution.f2, color='red', s=100, label='Best Solution')
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.title('Pareto Front')
        plt.legend()
        plt.tight_layout(rect=[0,0.01,1,0.91])
        plt.figtext(0.12,0.1,f'best_solution: \nf1: {(best_solution.f1)},\nf2: {(best_solution.f2)}')
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
    generations = 1000  
    bounds = (0, 2 * np.pi)

    nsga =NSGA(size=pop_size,bounds=bounds,generations=generations)
    one = True
    if one == True:
        population, history, fronts, best_solution = nsga.evolve()
        nsga.plot_pareto_front(fronts,best_solution)
        
    def multiple_runs(runs):
        file_path =os.path.join('./Statistics and results',f'NSGA-II Performance in {runs} runs.csv')
        graph_path =os.path.join('./Statistics and results',f'NSGA-II Performance in {runs} runs.png')
        best_solutions = []
        times = []

        for _ in range(runs):
            start_time = time.time()
            _,_,_,best_solution = nsga.evolve()
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
    
        nsga.plot_performance(runs, times,graph_path,save=True)

    multiple = False
    if multiple == True:
        multiple_runs(10)
    