import math
import random
import matplotlib.pyplot as plt
import dataset as t


# -------- COST FUNCTION --------
# function we are attempting to optimize (maximize profit)
def maximize(items):
    t = profit(items)
    return t + check_weight(items, t)


# The function we are trying to maximize
def profit(items):
    total = 0
    for i in range(len(items)):
        total += items[i] * value[i]  # x * value
    return total


# If total weight exceeds max_weight;
# returning the negative of the 1st function value as penalty points from the function,
# reset the result value so that it does not take the existing value
def check_weight(items, element):
    total = 0
    for i in range(len(items)):
        total += items[i] * weight[i]

    if total <= max_weight:
        if total <= element:
            return element - total
        else:
            return 0
    else:
        return -element


# -------- PARTICLE --------
class Particle:
    def __init__(self, x):
        self.position = []  # particle position
        self.velocity = []  # particle velocity
        self.personal_best = []  # best position individual
        self.err_personal_best = -1  # best error individual
        self.err_personal = -1  # error individual

        # create n(=num_dimensions) random particles
        for i in range(num_dimensions):
            self.velocity.append(random.uniform(0, 1))  # random.uniform(-1, 1)
            self.position.append(x[i])

    # evaluate current fitness
    def evaluate(self, cost_func):
        self.err_personal = cost_func(self.position)  # cost_func --> maximize

        # check to see if the current position is a personal best
        if self.err_personal > self.err_personal_best or self.err_personal_best == -1:
            self.personal_best = self.position
            self.err_personal_best = self.err_personal

    def update_velocity(self, global_best):
        w = 0.9  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1  # cognitive constant (particle)
        c2 = 1.9  # social constant (swarm)

        for i in range(num_dimensions):  # process each particle
            r1 = random.random()
            r2 = random.random()
            # compute new velocity of current particle
            cognitive = c1 * r1 * (self.personal_best[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_position(self, bounds):
        for i in range(num_dimensions):  # process each particle
            # If velocity is too high, the particle can fly over the best value,
            # if is too low, the particle can’t make enough exploration. Velocity falls into the local optimum.
            max_bounds = (bounds[i][1] - bounds[i][0])
            # limit the range of current velocity
            if self.velocity[i] < -max_bounds:
                self.velocity[i] = -max_bounds
            elif self.velocity[i] > max_bounds:
                self.velocity[i] = max_bounds

            # compute new position using new velocity
            self.position[i] += self.velocity[i]

            # check if current position it's out of bound
            if self.position[i] > bounds[i][1]:
                # If the position is above the upper limit value, pull to the upper limit value
                self.position[i] = bounds[i][1]
            elif self.position[i] < bounds[i][0]:
                # If the position is below the lower limit value, pull down to the lower limit value
                self.position[i] = bounds[i][0]
            else:
                self.position[i] = round(self.position[i])


# -------- PSO --------
class PSO:
    fitness, iterations, step_weight, step_profit, global_best, err_global_best = [], [], [], [], [], -1

    def __init__(self, cost_func, x, bounds, num_particles, max_iter, verbose=False):
        global num_dimensions

        num_dimensions = len(x)  # numbers of items
        self.err_global_best = -1  # best error for group
        self.global_best = []  # best position for group

        # establish the swarm
        swarm = []
        for i in range(num_particles):
            swarm.append(Particle(x))

        # begin optimization loop
        iteration = 0
        while iteration < max_iter:
            #  evaluate fitness of particles in swarm
            for j in range(num_particles):
                swarm[j].evaluate(cost_func)

                # determine if current particle is the best (globally)
                if swarm[j].err_personal > self.err_global_best or self.err_global_best == -1:
                    self.global_best = list(swarm[j].position)
                    self.err_global_best = float(swarm[j].err_personal)

            # update velocities and position
            for j in range(num_particles):
                swarm[j].update_velocity(self.global_best)
                swarm[j].update_position(bounds)

            total_profit = 0
            total_weight = 0
            for i in range(num_particles):
                total_profit += self.global_best[i] * value[i]
                total_weight += self.global_best[i] * weight[i]

            self.step_profit.append(total_profit)
            self.step_weight.append(total_weight)
            self.fitness.append(self.err_global_best)
            self.iterations.append(iteration)

            if verbose:
                print(f'iter: {iteration:>4d}, global best solution: {self.global_best}')
            iteration += 1

    # -------- result --------
    def Result(self):
        # print final results
        print('\nFINAL RESULTS:')
        print(f'global best -> {self.global_best}')
        print(f'err global best -> {self.err_global_best}')
        total_profit = 0
        total_weight = 0
        for i in range(len(self.global_best)):
            # print("object ", i, ': ', self.global_best[i], sep='')
            # print(object[i], ': ', self.global_best[i], sep='')
            total_profit += self.global_best[i] * value[i]
            total_weight += self.global_best[i] * weight[i]
        print('-' * 20, '\n# obj: ', obj_num, '\nmax weight: ', max_weight)
        print('\nProfit: ', total_profit, ',\nWeight (Kg): ', total_weight, '\n', '-' * 20, sep='')

    # Plotting the results with plot [If we do not want to save the result image to the computer, the parameter named 'file_name' must remain empty!] ...
    def plot_result(self, file_name=''):
        plt.plot(self.step_weight, self.step_profit)
        plt.xlabel('Weight (kg)')
        plt.ylabel('Profit')
        plt.title('Profit vs Weight')
        plt.grid(True)

        if not (
                file_name == ''):  # If the variable named 'file_name' is not empty, save the file with that name in png format.
            file_name = file_name + ".png"
            plt.savefig(file_name)

        # plt.show()
        plt.close()

    def plot_evolution(self, file_name=''):
        plt.plot(self.iterations, self.fitness)
        plt.xlabel('iterations')
        plt.ylabel('fitness')
        plt.title('Evolution of solutions')
        plt.grid(True)

        if not (file_name == ''):  # If the variable named 'file_name' is not empty, save the file with that name in png format.
            file_name = file_name + ".png"
            plt.savefig(file_name)

        # plt.show()
        plt.close()


# -------- MAIN --------

num = input("insert number from 1 to 10: ")
# print('profit, weight')
dataset = t.data(True, int(num) - 1)
value = []
weight = []
for i in dataset:
    if i == 0:
        max_weight = dataset[i][1]
        obj_num = dataset[i][0]
    else:
        # print(dataset[i][0], "\t\t", dataset[i][1])
        value.append(dataset[i][0])
        weight.append(dataset[i][1])

# -------- EXECUTE --------
# print('\n[lower limit - upper limit]', sep='')
initial = []
bounds = []
length = len(value)

repetition = input("Knapsack with repetition of items (y/n)?: ").lower()
if repetition == "y":
    repetition = True
elif repetition == "n":
    repetition = False
if not isinstance(repetition, bool):
    print("error bool")

# initial position of particles  if rand(0,1)<0.5 --> x=0  else x=1
# for i in range(len(weight)):
#     initial.append(round(random.uniform(0, 1)))  # [x1, x2, ...]

if repetition:
    for i in range(len(weight)):
        initial.append(0)
        bounds.append((initial[i], math.floor(max_weight / weight[i])))  # [(x1_min,x1_max),(x2_min,x2_max)...]
        print('object', i, ': ', bounds[i][0], '-', bounds[i][1], sep='')
else:
    no_rep = [1 for i in range(length)]
    for i in range(len(weight)):
        initial.append(0)
        bounds.append((initial[i], no_rep[i]))
        # print('object', i, ': ', bounds[i][0], '-', bounds[i][1], sep='')

print('There are a total of ', length, ' variable...\n', sep='')

max_iter = int(input("Insert max of iteration: "))
# num_particles = int(input("Insert number of particles: "))

pso = PSO(maximize, initial, bounds, length, max_iter, verbose=False)
pso.Result()
pso.plot_result(file_name="pso_result")
pso.plot_evolution(file_name="evolution")
