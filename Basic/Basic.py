import numpy as np
import random


class Individual:
    def __init__(self, values, eval_f):
        self.values = values
        self.eval_f = eval_f

    def mutation(self, sigma):
        mutant_vals = np.random.normal(self.values, sigma)
        return Individual(mutant_vals, self.eval_f)

    def crossing(self, other):
        if len(other.values) != len(self.values):
            raise ValueError("Individuals should live in the same space (different length vectors).")
        cutting_point = random.randrange(1, len(self.values))
        child_vals = np.concatenate((self.values[:cutting_point], other.values[cutting_point:]))
        return Individual(child_vals, self.eval_f)

    def eval(self):
        return self.eval_f(self.values)


class Population:
    def __init__(self, eval_f, size, min_value, max_value):
        if len(min_value) != len(max_value):
            raise ValueError("differen length of bounds")
        self.size = size
        self.individuals = list(map(lambda values: Individual(values, eval_f),
                                list(np.random.uniform(min_value, max_value, size=(size, len(min_value))))))

    def eval(self):
        return map(lambda individual: individual.eval(), self.individuals)

    def mutation(self, sigma):
        self.individuals.append(random.choice(self.individuals[:self.size]).mutation(sigma))

    def crossing(self):
        self.individuals.append(random.choice(self.individuals[:self.size]).crossing(random.choice(self.individuals[:self.size])))

    def selection(self, elite_count, tournament_size):
        if elite_count > self.size:
            raise ValueError("Too big elite count")
        self.individuals.sort(key=lambda individual: individual.eval())

        not_selected_yet = self.individuals[elite_count:]
        # elitism
        self.individuals = self.individuals[:elite_count]

        while len(self.individuals) < self.size:
            # tournament
            tournament_winner_id = min(np.random.randint(0, len(not_selected_yet), size=tournament_size))
            self.individuals.append(not_selected_yet[tournament_winner_id])
            del not_selected_yet[tournament_winner_id]

    def iteration(self, n_mutations, sigma, n_crossings, elite_count, tournament_size):
        if tournament_size + elite_count > self.size + n_mutations + n_crossings:
            raise ValueError("Too big tournament size")

        for _ in range(n_mutations):
            self.mutation(sigma)

        for _ in range(n_crossings):
            self.crossing()

        self.selection(elite_count, tournament_size)

    def select_best(self):
        return min(self.individuals, key=lambda individual: individual.eval())
