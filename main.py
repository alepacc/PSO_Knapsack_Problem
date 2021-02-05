import math
import random
import matplotlib.pyplot as plt
import dataset as d


def sigmoid(n):
    return 1 / (1 + math.exp(-n))


def save_img_plot(file_name):
    if not (
            file_name == ''):  # If the variable named 'file_name' is not empty, save the file with that name in png format.
        file_name = file_name + ".png"
        plt.savefig(file_name)
    # plt.show()
    plt.close()


# -------- COST FUNCTION --------
# function to optimize (maximize profit)
def maximize(items):
    t = profit(items)
    return t + check_weight(items, t)


# The function trying to maximize
def profit(items):
    total = 0
    for _ in range(len(items)):
        total += items[_] * value[_]  # x * value
    return total


# If total weight exceeds max_weight
def check_weight(items, element):
    total = 0
    for _ in range(len(items)):
        total += items[_] * weight[_]

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
        """
        :param x: vector of initial position of particle
        """
        self.position = []  # particle position
        self.velocity = []  # particle velocity
        self.personal_best = []  # best position individual
        self.err_personal_best = -1  # best error individual
        self.err_personal = -1  # error individual

        # create n(=num_dimensions) random particles
        for _ in range(num_dimensions):
            self.velocity.append(random.uniform(0, 1))  # random.uniform(-1, 1)
            self.position.append(x[_])

    # evaluate current fitness
    def evaluate(self, cost_func):
        self.err_personal = cost_func(self.position)  # cost_func --> maximize

        # check to see if the current position is a personal best
        if self.err_personal > self.err_personal_best or self.err_personal_best == -1:
            self.personal_best = self.position
            self.err_personal_best = self.err_personal

    def update_velocity(self, global_best):
        w = 0.9  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1.2  # cognitive constant (particle)
        c2 = 1.9  # social constant (swarm)

        for i in range(num_dimensions):  # process each particle
            r1 = random.random()
            r2 = random.random()
            # compute new velocity of current particle
            cognitive = c1 * r1 * (self.personal_best[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_position(self, bounds, type_pso):
        for i in range(num_dimensions):  # process each particle
            if type_pso == "BPSO":
                # _________ BPSO _______#
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

                if self.position[i] < sigmoid(self.velocity[i]):
                    self.position[i] = 1
                else:
                    self.position[i] = 0
            elif type_pso == "PSO":
                # ### ----- PSO ------
                self.position[i] += self.velocity[i]
                # check if current position it's out of bound
                if self.position[i] > bounds[i][1]:
                    # If the position is above the upper limit value, pull to the upper limit value
                    self.position[i] = bounds[i][1]
                elif self.position[i] < bounds[i][0]:
                    # If the position is below the lower limit value, pull down to the lower limit value
                    self.position[i] = bounds[i][0]
                else:
                    self.position[i] = round(
                        self.position[i])  # position of particles: if rand(0,1)<0.5 --> x=0  else x=1


# -------- PSO --------
class PSO:
    fitness, iterations, step_weight, step_profit, global_best, err_global_best = [], [], [], [], [], -1

    def __init__(self, cost_func, x, bounds, num_particles, max_iter, type_pso, verbose=False):
        """
        :param cost_func: cost function for PSO
        :param x: vector of initial position of particle
        :param bounds: bounds for position of particle [lower limit - upper limit]
        :param num_particles: numbers of particle
        :param max_iter:  maximum number of iterations for the execution of PSO
        :param verbose: (default=False) print global best solution for each iteration
        :type verbose: bool
        :param type_pso: "PSO" or "BPSO"
        :type type_pso: str
        """

        global num_dimensions

        num_dimensions = len(x)  # numbers of items
        self.err_global_best = -1  # best error for group
        self.global_best = []  # best position for group

        # establish the swarm
        swarm = []
        for i in range(num_particles):
            swarm.append(Particle(x))  # x = array position of particle

        # begin optimization loop
        iteration = 0
        cnt_global_best = 0  # TODO: 10 iter
        while iteration < max_iter:  # and cnt_global_best != 10:
            global_best_prev = self.global_best  # save previus global best

            #  evaluate fitness of particles in swarm
            for j in range(num_particles):
                swarm[j].evaluate(cost_func)

                # determine if current particle is the best (globally)
                if swarm[j].err_personal > self.err_global_best or self.err_global_best == -1:
                    self.global_best = list(swarm[j].position)
                    self.err_global_best = float(swarm[j].err_personal)

            # if global best doesn't change after 10 times, the iteration terminate
            if global_best_prev == self.global_best:
                cnt_global_best += 1

            # update velocities and position
            for j in range(num_particles):
                swarm[j].update_velocity(self.global_best)
                swarm[j].update_position(bounds, type_pso)

            total_profit = 0
            total_weight = 0
            for i in range(len(x)):
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
    def result(self):
        # print final results
        total_profit = 0
        total_weight = 0
        for i in range(len(self.global_best)):
            # print("object ", i, ': ', self.global_best[i], ,'')
            # print(object[i], ': ', self.global_best[i], sep='')
            total_profit += self.global_best[i] * value[i]
            total_weight += self.global_best[i] * weight[i]
        print(f'global best -> {self.global_best}')
        # print(f'err global best -> {self.err_global_best}')
        print('Profit: ', total_profit, '\nWeight: ', total_weight, '\n', '-' * 20, sep='')

    # Plotting the results with plot [If we do not want to save the result image to the computer, the parameter named 'file_name' must remain empty!] ...
    def plot_result(self, file_name=''):
        x1, x2, y1, y2 = [], [], [], []
        for i in range(len(self.step_weight)):
            if i < len(self.step_weight)/2:
                x1.append(self.step_weight[i])
                y1.append(self.step_profit[i])
            else:
                x2.append(self.step_weight[i])
                y2.append(self.step_profit[i])
        plt.plot(x1, y1, label="PSO")
        plt.plot(x2, y2, label="BPSO")
        plt.xlabel('Weight')
        plt.ylabel('Profit')
        plt.title('PSO vs BPSO')
        plt.legend()
        plt.grid(True)
        save_img_plot(file_name)

    def plot_evolution(self, file_name=''):
        x1, x2, y1, y2 = [], [], [], []
        for i in range(len(self.iterations)):
            if i < len(self.iterations) / 2:
                x1.append(self.iterations[i])
                y1.append(self.fitness[i])
            else:
                x2.append(self.iterations[i])
                y2.append(self.fitness[i])
        plt.plot(x1, y1, label="PSO")
        plt.plot(x2, y2, label="BPSO")
        plt.xlabel('iterations')
        plt.ylabel('fitness')
        plt.title('Evolution of solutions')
        plt.legend()
        plt.grid(True)
        save_img_plot(file_name)


# -------- MAIN --------
def main():
    global value, weight, max_weight, obj_num, optimum
    num = input("insert number from 1 to 10: ")
    # print('profit, weight')
    dataset = d.data(True, int(num) - 1)
    value = []
    weight = []

    for i in dataset:
        if i == 0:
            max_weight = dataset[i][1]
            obj_num = dataset[i][0]
        elif i != len(dataset) - 1:
            # print(dataset[i][0], "\t\t", dataset[i][1])
            value.append(dataset[i][0])
            weight.append(dataset[i][1])
        else:
            optimum = dataset[i][1]

    # -------- EXECUTE --------
    initial = []
    bounds = []
    length = len(value)

    choice = input("Knapsack 0-1 (y/n)?: ").lower()
    if choice == "y":
        choice = True
    elif choice == "n":  # Knapsack with items repetitions
        choice = False
    if not isinstance(choice, bool):
        print("error bool")

    if not choice:
        print('\n[lower limit - upper limit]')
        for i in range(len(weight)):
            initial.append(0)
            bounds.append((initial[i], math.floor(max_weight / weight[i])))  # [(x1_min,x1_max),(x2_min,x2_max)...]
            print('object ', i, ': [', bounds[i][0], '-', bounds[i][1], ']', sep='')
    else:  # Knapsack 0-1
        upper_bound = [1 for _ in range(length)]  # upper bound
        for i in range(len(weight)):
            # # initial position of particles  if rand(0,1)<0.5 --> x=0  else x=1
            # initial.append(round(random.uniform(0, 1)))  # [x1, x2, ...]
            initial.append(0)
            bounds.append((initial[i], upper_bound[i]))
            # print('object', i, ': ', bounds[i][0], '-', bounds[i][1], sep='')

    print('There are a total of', length, 'variable...\n')

    max_iter = int(input("Insert max of iteration: "))
    num_particles = int(input("Insert number of particles: "))

    print('-' * 20, '\n# obj: ', obj_num, '\nmax weight: ', max_weight, '\noptimum profit to achieve: ', optimum)
    print('-' * 20, '\nFINAL RESULTS:')

    pso = PSO(maximize, initial, bounds, num_particles, max_iter, "PSO", verbose=False)
    print('_' * 10, 'PSO', '_' * 10, sep='')
    pso.result()
    # pso.plot_result(file_name="pso_result")
    # pso.plot_evolution(file_name="pso_evolution")

    pso = PSO(maximize, initial, bounds, num_particles, max_iter, "BPSO", False)
    print('_' * 10, 'BPSO', '_' * 10, sep='')
    pso.result()

    pso.plot_result(file_name="result")
    pso.plot_evolution(file_name="evolution")


if __name__ == '__main__':
    main()
# TODO: large_scale
# TODO: fix plotting
