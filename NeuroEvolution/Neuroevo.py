from my_net import *
import copy
import random


class Individual:
    def __init__(self, net, x, y):
        self.net = net
        self.x = x
        self.y = y

        # evaluating here, because we don't change individual, only generate new individuals
        # and evaluating is a very costly operation, so we do it only once for each individual
        self.eval_value = self.net.mse(x, y)

    def eval(self):
        return self.eval_value

    def mutation(self, sigma):
        mutant_net = copy.deepcopy(self.net)
        selected_layer_i = random.randrange(len(self.net.layers))
        mutant_net.layers[selected_layer_i].weights = np.random.normal(mutant_net.layers[selected_layer_i].weights, sigma)
        mutant_net.layers[selected_layer_i].biases = np.random.normal(mutant_net.layers[selected_layer_i].biases, sigma)
        return Individual(mutant_net, self.x, self.y)

    def crossing(self, other):
        child_net = copy.deepcopy(self.net)
        selected_layer_i = random.randrange(len(self.net.layers))
        child_net.layers[selected_layer_i].weights = other.net.layers[selected_layer_i].weights
        child_net.layers[selected_layer_i].biases = other.net.layers[selected_layer_i].biases
        return Individual(child_net, self.x, self.y)


class Population:
    def __init__(self, size, input_size, layers_sizes, activations, x, y, initializer="xavier"):
        self.individuals = []
        self.size = size
        for _ in range(size):
            net = Net(input_size)
            for layer_size, activation in zip(layers_sizes, activations):
                layer = DenseLayer(layer_size, activation)
                net.add(layer)
            net.kernel_init(initializer)
            self.individuals.append(Individual(net, x, y))

    def mutations(self, sigma, p):
        for individual in self.individuals[:self.size]:
            if random.uniform(0, 1) < p:
                self.individuals.append(individual.mutation(sigma))

    def crossings(self, p):
        for individual in self.individuals[:self.size]:
            if random.uniform(0, 1) < p:
                other = random.choice(self.individuals[:self.size])
                self.individuals.append(individual.crossing(other))

    def selection(self, elite_count, tournament_size):
        if elite_count > self.size:
            raise ValueError("Too big elite count")
        self.individuals.sort(key=lambda individual: individual.eval())

        not_selected_yet = self.individuals[elite_count:]
        # elitism
        self.individuals = self.individuals[:elite_count]

        if tournament_size > len(not_selected_yet):
            raise ValueError("Too big tournament size")

        while len(self.individuals) < self.size:
            # tournament
            tournament_winner_id = min(np.random.randint(0, len(not_selected_yet), size=tournament_size))
            self.individuals.append(not_selected_yet[tournament_winner_id])
            del not_selected_yet[tournament_winner_id]

    def iteration(self, p_mutation, sigma, p_crossing, elite_count, tournament_size):
        self.mutations(sigma, p_mutation)
        self.crossings(p_crossing)
        self.selection(elite_count, tournament_size)

    def best(self):
        return min(self.individuals, key=lambda individual: individual.eval())
