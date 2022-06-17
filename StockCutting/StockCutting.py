import numpy as np
import math
import copy
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm


class RectangleType:
    def __init__(self, width, height, value):
        self.width = width
        self.height = height
        self.value = value

    def from_left_down(self, x, y):
        return Rectangle(x, y, x + self.width, y + self.height, self.value)

    def __str__(self):
        return "RectangleType(width = " + str(self.width) + ", height = " + str(self.height) + ", value = " + str(self.value) + ")"


class RectanglesFactory:
    def __init__(self, path, delimiter=','):
        arr = np.genfromtxt(path, delimiter=delimiter)
        self.types = []
        for line in arr:
            self.types.append(RectangleType(line[0], line[1], line[2]))
            self.types.append(RectangleType(line[1], line[0], line[2]))

    def from_left_down(self, x, y):
        return random.choice(self.types).from_left_down(x, y)


class Rectangle:
    def __init__(self, x_min, y_min, x_max, y_max, value):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.value = value

    def __str__(self):
        return "Rectangle(x_min = " + str(self.x_min) + ", y_min = " + str(self.y_min) + ", x_max = " + str(self.x_max) \
               + ", y_max = " + str(self.y_max) + ", val = " + str(self.value) + ")"

    def not_overlaps(self, other):
        return other.y_max <= self.y_min or other.y_min >= self.y_max or \
               other.x_max <= self.x_min or other.x_min >= self.x_max

    def in_circle(self, radius):
        return (self.y_max ** 2 + self.x_max ** 2) <= radius ** 2 and \
               (self.y_min ** 2 + self.x_max ** 2) <= radius ** 2 and \
               (self.y_max ** 2 + self.x_min ** 2) <= radius ** 2 and \
               (self.y_min ** 2 + self.x_min ** 2) <= radius ** 2

    def under_line(self, a, b):  # is under the line or on the line <=> at least one of bottom edges is under the line
        return self.y_min < a * self.x_max + b or self.y_min < a * self.x_min + b

    def over_line(self, a, b):  # is over the line or on the line <=> at least one of top edges is over the line
        return self.y_max > a * self.x_max + b or self.y_max > a * self.x_min + b

    def move_left_to(self, x):
        width = self.x_max - self.x_min
        self.x_min = x
        self.x_max = x + width

    def move_down_to(self, y):
        height = self.y_max - self.y_min
        self.y_min = y
        self.y_max = y + height

    def left_in_circle(self, radius):
        return -math.sqrt(radius ** 2 - max(abs(self.y_min), abs(self.y_max)) ** 2)

    def down_in_circle(self, radius):
        return -math.sqrt(radius ** 2 - max(abs(self.x_min), abs(self.x_max)) ** 2)

    def width(self):
        return self.x_max - self.x_min

    def height(self):
        return self.y_max - self.y_min

    def density(self):
        return self.value / (self.height() * self.width())


class Individual:
    def __init__(self, radius, factory):
        self.radius = radius
        self.rectangles = []
        self.factory = factory

    def evaluate(self):
        eval_sum = 0
        for rectangle in self.rectangles:
            eval_sum += rectangle.value
        return eval_sum

    def add_new(self, new_rectangle):
        self.rectangles = list(filter(
            lambda rectangle: rectangle.not_overlaps(new_rectangle),
            self.rectangles
        ))
        if new_rectangle.in_circle(self.radius):
            self.rectangles.append(new_rectangle)
            return True
        else:
            return False

    def try_add_new(self, new_rectangle):
        fits = new_rectangle.in_circle(self.radius)
        for rectangle_here in self.rectangles:
            fits = fits and rectangle_here.not_overlaps(new_rectangle)
        if fits:
            self.rectangles.append(new_rectangle)
            return True
        else:
            return False

    def cross_by_line(self, other, a, b):
        self_copy = copy.deepcopy(self)
        other_copy = copy.deepcopy(other)
        self_copy.rectangles = list(filter(
            lambda rectangle: rectangle.over_line(a, b),
            self_copy.rectangles
        ))
        other_copy.rectangles = list(filter(
            lambda rectangle: rectangle.under_line(a, b),
            other_copy.rectangles
        ))
        for rectangle_there in other_copy.rectangles:
            self_copy.try_add_new(rectangle_there)
        self_copy.correct()
        return self_copy

    def sweep_left(self):
        self.rectangles.sort(key=lambda rect: rect.x_min)
        for i, rectangle in enumerate(self.rectangles):
            left = rectangle.left_in_circle(self.radius)
            for j in range(i):
                if not (rectangle.y_min >= self.rectangles[j].y_max or rectangle.y_max <= self.rectangles[j].y_min):
                    # rectangle j stops our given rectangle from falling further left
                    left = max(left, self.rectangles[j].x_max)
            rectangle.move_left_to(left)

    def sweep_down(self):
        self.rectangles.sort(key=lambda rect: rect.y_min)
        for i, rectangle in enumerate(self.rectangles):
            down = rectangle.down_in_circle(self.radius)
            for j in range(i):
                if not (rectangle.x_min >= self.rectangles[j].x_max or rectangle.x_max <= self.rectangles[j].x_min):
                    # rectangle j stops our given rectangle from falling further down
                    down = max(down, self.rectangles[j].y_max)
            rectangle.move_down_to(down)

    def grow_right(self):
        for rectangle in self.rectangles:
            for i in range(len(self.factory.types)):
                type = random.choice(self.factory.types)
                new = type.from_left_down(rectangle.x_max, rectangle.y_min)
                if self.try_add_new(new):
                    break

    def grow_up(self):
        for rectangle in self.rectangles:
            for i in range(len(self.factory.types)):
                type = random.choice(self.factory.types)
                new = type.from_left_down(rectangle.x_min, rectangle.y_max)
                if self.try_add_new(new):
                    break

    def correct(self):
        self.sweep_left()
        self.grow_right()
        self.sweep_down()
        self.grow_up()

    def plot(self):
        fig, ax = plt.subplots()
        plt.xlim(-self.radius, self.radius)
        plt.ylim(-self.radius, self.radius)
        circle = patches.Circle((0, 0), self.radius, edgecolor='black', facecolor='white')
        ax.add_patch(circle)
        for rect in self.rectangles:
            rect_image = patches.Rectangle((rect.x_min, rect.y_min), rect.width(), rect.height(),
                                           facecolor=cm.hot(1 - math.sqrt(rect.density())), edgecolor="black")
            ax.add_patch(rect_image)
        plt.show()

    def random_cross(self, other):
        b = self.radius * random.random() ** 2
        a = random.random()
        if random.getrandbits(1):
            a = -a
        if random.getrandbits(1):
            a = 1/a
        return self.cross_by_line(other, a, b)

    def random_mutation(self):
        child = copy.deepcopy(self)
        while True:  # emulating do while loop
            p = random.random() * 2 * math.pi
            r2 = self.radius * math.sqrt(random.random())
            x = math.cos(p) * r2
            y = math.sin(p) * r2
            random_rectangle = self.factory.from_left_down(x, y)
            if child.add_new(random_rectangle):
                break  # random rectangle added, so mutation completed

        child.correct()
        return child


class Evolution:
    def __init__(self, population_size, radius, path, delimiter=',', starting_rectangles=500):
        self.radius = radius
        self.factory = RectanglesFactory(path, delimiter)

        # initializing population
        self.population = []
        for i in range(population_size):
            individual = Individual(self.radius, self.factory)
            for j in range(starting_rectangles):
                p = random.random() * 2 * math.pi
                r2 = radius * math.sqrt(random.random())
                x = math.cos(p) * r2
                y = math.sin(p) * r2
                random_rectangle = self.factory.from_left_down(x, y)
                individual.try_add_new(random_rectangle)
            individual.correct()
            self.population.append(individual)

    def selection(self, population_limit, elite_count):
        if elite_count > population_limit:
            raise ValueError("elite count should be lesser than population limit")
        if len(self.population) < population_limit:
            raise ValueError("population limit is bigger than current population")
        self.population.sort(key=lambda individual: individual.evaluate(), reverse=True)

        # adding elite
        left = self.population[elite_count:]
        self.population = self.population[:elite_count]

        # roulette selecting other
        probs = list(map(lambda indiv: indiv.evaluate(), left))
        chosen = random.choices(left, weights=probs, k=(population_limit - elite_count))
        self.population.extend(chosen)

    def iter(self, n_mutations, n_crossings, elite_count):
        n = len(self.population)

        for i in range(n_crossings):
            father = random.choice(self.population)
            mother = random.choice(self.population)
            child = father.random_cross(mother)
            self.population.append(child)

        for i in range(n_mutations):
            mutation_candidate = random.choice(self.population)
            mutant = mutation_candidate.random_mutation()
            self.population.append(mutant)

        self.selection(n, elite_count)

    def best(self):
        return max(self.population, key=lambda individual: individual.evaluate())

