import math
import random
import matplotlib.pyplot as plt
import dataset as d


def sigmoid(n):
    return 1.0 / (1.0 + math.exp(-n))


def save_img_plot(file_name):
    if not (file_name == ''):
        file_name = file_name + ".png"
        plt.savefig(file_name)
    # plt.show()
    plt.close()


# -------- COST FUNCTION --------
# function to optimize (maximize profit)
def maximize(items):       # items = particle position
    total_profit = profit(items)
    return check_weight(items, total_profit)


# The function trying to maximize
def profit(items):
    total_profit = 0
    for _ in range(len(items)):
        total_profit += items[_] * value[_]  # x * value
    return total_profit


# If total weight exceeds max_weight
def check_weight(items, total_profit):
    total_weight = 0
    for _ in range(len(items)):
        total_weight += items[_] * weight[_]

    if total_weight <= max_weight:
        return total_profit
    else:
        return 0


# -------- PARTICLE --------
class Particle:
    def __init__(self, x, decrease=False):
        """
        :param x: vector of initial position of particle
        :param decrease: bool value for choice of linear decreasing w
        """
        self.position = []  # particle position
        self.velocity = []  # particle velocity
        self.personal_best = []  # best position individual
        self.fit_personal_best = -1  # best fitness value
        self.fit_current = -1  # fitness value of the current solution

        self.decrease = decrease

        # create n(=num_items) random particles
        for _ in range(num_items):
            self.velocity.append(random.uniform(0, 1))
            self.position.append(x[_])

    # evaluate current fitness
    def evaluate(self, cost_func):
        self.fit_current = cost_func(self.position)  # cost_func --> maximize

        # check to see if the current position is a personal best
        if self.fit_current > self.fit_personal_best or self.fit_personal_best == -1:
            self.personal_best = self.position
            self.fit_personal_best = self.fit_current

    def update_velocity(self, global_best, decr, type_pso):
        global w, c1, c2
        # parameter inertia weight (how much weigh of the previous velocity)
        if self.decrease:  # LDIW
            w = round(0.9 - decr, 1)
        else:  # constant w
            w = 0.9
        c1 = 1.49  # cognitive constant (particle)
        c2 = 1.49  # social constant (swarm)

        for i in range(num_items):  # process each particle
            r1 = random.random()
            r2 = random.random()
            # compute new velocity of current particle
            cognitive = c1 * r1 * (self.personal_best[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            if type_pso == "BPSO":
                self.velocity[i] = self.velocity[i] + cognitive + social
            else:
                self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_position(self, bounds, type_pso):
        for i in range(num_items):
            if type_pso == "BPSO":
                # _________ BPSO _______ #
                # If velocity is too high, the particle can fly over the best value,
                # if is too low, the particle can’t make enough exploration. Velocity falls into the local optimum.
                vel_max = 4  # velocity [-4, 4]
                # limit the range of current velocity
                if self.velocity[i] < -vel_max:
                    self.velocity[i] = -vel_max
                elif self.velocity[i] > vel_max:
                    self.velocity[i] = vel_max

                if random.uniform(0, 1) < sigmoid(self.velocity[i]):
                    self.position[i] = 1
                else:
                    self.position[i] = 0

            elif type_pso == "PSO":
                # ----- PSO ------ #
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
    fitness, iterations, step_weight, step_profit, global_best, fit_global_best = [], [], [], [], [], -1

    def __init__(self, cost_func, x, bounds, num_particles, max_iter, type_pso, decrease=False, verbose=False):
        """
        :param cost_func: cost function for PSO
        :param x: vector of initial position of particle len(x)=number of items 
        :param bounds: bounds for position of particle [lower limit - upper limit]
        :param num_particles: numbers of particle
        :param max_iter:  maximum number of iterations for the execution of PSO
        :param type_pso: "PSO" or "BPSO"
        :type type_pso: str
        :param decrease: (default=False) parameter for linear decrease of inertia w
        :type decrease: bool
        :param verbose: (default=False) print global best solution for each iteration
        :type verbose: bool
        """

        global num_items

        num_items = len(x)  # number of objects
        self.fit_global_best = -1  # best fitness value of swarm
        self.global_best = []  # best position of swarm

        # establish the swarm
        swarm = []
        for i in range(num_particles):
            swarm.append(Particle(x, decrease))  # x = array position of particle

        # begin optimization loop
        iteration = 0
        while iteration < max_iter:
            #  evaluate fitness of particles in swarm
            for j in range(num_particles):
                swarm[j].evaluate(cost_func)

                # determine if current particle is the best (globally)
                if swarm[j].fit_current > self.fit_global_best or self.fit_global_best == -1:
                    self.global_best = list(swarm[j].position)
                    self.fit_global_best = float(swarm[j].fit_current)

            if decrease:
                decr = round(0.9 / max_iter * iteration, 1)  # linear decreasing inertia w (LDIW) from 0.9 -> 0.0
            else:
                decr = 0  # zero for not decreasing w

            # update velocities and position
            for j in range(num_particles):
                swarm[j].update_velocity(self.global_best, decr, type_pso)
                swarm[j].update_position(bounds, type_pso)

            total_profit = 0
            total_weight = 0
            for i in range(len(x)):
                total_profit += self.global_best[i] * value[i]
                total_weight += self.global_best[i] * weight[i]

            # parameters for plotting
            self.step_profit.append(total_profit)
            self.step_weight.append(total_weight)
            self.fitness.append(self.fit_global_best)
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
            total_profit += self.global_best[i] * value[i]
            total_weight += self.global_best[i] * weight[i]

        try:  # round float value
            total_profit = round(total_profit, 4)
            total_weight = round(total_weight, 4)
        except ValueError:
            pass


        print(f'global best -> {self.global_best}')
        print('Profit: ', total_profit, '\nWeight: ', total_weight, '\n', '-' * 20, sep='')

    # Plotting the results with plot [If we do not want to save the result image to the computer, the parameter named 'file_name' must remain empty!] ...
    def plot_result(self, file_name=''):
        x1, x2, x3, y1, y2, y3 = [], [], [], [], [], []
        for i in range(len(self.iterations)):
            if i < len(self.iterations) / 3:  # BPSO
                x1.append(self.step_weight[i])
                y1.append(self.step_profit[i])
            elif i < 2 * len(self.iterations) / 3:  # LDIW-PSO
                x2.append(self.step_weight[i])
                y2.append(self.step_profit[i])
            else:  # PSO
                x3.append(self.step_weight[i])
                y3.append(self.step_profit[i])
        plt.xlabel('Weight')
        plt.ylabel('Profit')
        plt.suptitle('PSO vs BPSO vs LDIW-PSO ', fontsize=18)
        plt.title('c1 = ' + str(c1) + ', c2 = ' + str(c2) + ', w = ' + str(w))
        plt.plot(x1, y1, linestyle='--', label="BPSO")
        plt.plot(x2, y2, linestyle='dotted', label="LDIW-PSO")
        plt.plot(x3, y3, linestyle='dashdot', label="PSO")
        plt.legend()
        plt.grid(True)
        save_img_plot(file_name)

    def plot_evolution(self, file_name=''):
        x1, x2, x3, y1, y2, y3 = [], [], [], [], [], []
        for i in range(len(self.iterations)):
            if i < len(self.iterations) / 3:  # BPSO
                x1.append(self.iterations[i])
                y1.append(self.fitness[i])
            elif i < 2*len(self.iterations)/3:  # LDIW-PSO
                x2.append(self.iterations[i])
                y2.append(self.fitness[i])
            else:
                x3.append(self.iterations[i])
                y3.append(self.fitness[i])

        plt.xscale('symlog')
        plt.xlabel('iterations')
        plt.ylabel('fitness')
        plt.suptitle('Evolution of solutions', fontsize=18)
        plt.title('c1 = ' + str(c1) + ', c2 = ' + str(c2) + ', w = ' + str(w))
        plt.plot(x1, y1, linestyle='--', label="BPSO")
        plt.plot(x2, y2, linestyle='dotted', label="LDIW-PSO")
        plt.plot(x3, y3, linestyle='dashdot', label="PSO")

        plt.legend()
        plt.grid(True)
        save_img_plot(file_name)


# -------- MAIN --------
def main():
    global value, weight, max_weight, num_obj, optimum

    # input for choose type of dataset and number of instance
    try:
        choice_dim = input("Choose type of Dataset. Low dimensional or Big dim? (l/b): ").lower()
        if choice_dim == "l":  # Knapsack dataset low dimensional
            choice_dim = True
            num = input("Choose one instance for testing. Insert number from 1 to 10: ")
        elif choice_dim == "b":  # Knapsack dataset large dimensional
            choice_dim = False
            num = input("Choose one instance for testing. Insert number from 1 to 21: ")
        if not isinstance(choice_dim, bool):
            print("error bool")
    except ValueError:
        print("Input error!")

    # choose type of dataset (low dimension OR large)
    dataset = d.data(int(num) - 1, low=choice_dim)
    value = []
    weight = []

    for i in dataset:
        if i == 0:
            max_weight = dataset[i][1]
            num_obj = dataset[i][0]
        elif i != len(dataset) - 1:
            value.append(dataset[i][0])
            weight.append(dataset[i][1])
        else:
            optimum = dataset[i][1]

    # -------- EXECUTE --------
    initial = []
    bounds = []

    # input for choose type of knapsack
    try:
        choice = input("Knapsack 0-1? (y/n): ").lower()
        if choice == "y":
            choice = True
        elif choice == "n":  # Knapsack with items repetitions
            choice = False
        if not isinstance(choice, bool):
            print("error bool")
    except ValueError:
        print("Input error!")

    # generate bounds and vector initial
    if not choice:
        print('\n[lower limit - upper limit]')
        for i in range(num_obj):  # number of objects
            initial.append(0)  # initial position of particle
            bounds.append((initial[i], math.floor(max_weight / weight[i])))
            print('object ', i, ': [', bounds[i][0], '-', bounds[i][1], ']', sep='')
    else:  # Knapsack 0-1
        upper_bound = [1 for _ in range(num_obj)]  # upper bound
        for i in range(num_obj):
            initial.append(0)
            bounds.append((initial[i], upper_bound[i]))

    print('There are a total of', num_obj, 'objects...\n')

    # input max iteration and number of particles
    try:
        max_iter = int(input("Insert max of iteration: "))
        num_particles = int(input("Insert number of particles: "))
    except ValueError:
        print("Input error!")


    # ----- RESULT ------
    print('-' * 20, '\n# obj: ', num_obj, '\nmax weight: ', max_weight, '\noptimum profit to achieve: ', optimum)
    print('-' * 20, '\nFINAL RESULTS:')

    pso = PSO(maximize, initial, bounds, num_particles, max_iter, "BPSO")
    print('_' * 10, 'BPSO', '_' * 10, sep='')
    pso.result()

    pso = PSO(maximize, initial, bounds, num_particles, max_iter, "PSO", decrease=True, verbose=False)
    print('_' * 10, 'LDIW-PSO', '_' * 10, sep='')
    pso.result()

    pso = PSO(maximize, initial, bounds, num_particles, max_iter, "PSO")
    print('_' * 10, 'PSO', '_' * 10, sep='')
    pso.result()

    pso.plot_result(file_name="result")
    pso.plot_evolution(file_name="evolution")


if __name__ == '__main__':
    main()
