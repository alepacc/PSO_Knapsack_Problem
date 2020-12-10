import math
import random
import matplotlib.pyplot as plt


# --- COST FUNCTION
# function we are attempting to optimize (minimize)
def maximize(x):
    t = profit(x)
    return t + kilogram(x, t)


# The function we are trying to maximize ...
def profit(x):
    total = 0
    for i in range(len(x)):
        total += x[i] * value[i]  # x * value
    return total

    # If kilogram exceeds max_weight;
    # returning the negative of the 1st function value as penalty points from the function,
    # reset_elements the result value so that it does not take the existing value ...


def kilogram(x, reset_element):
    total = 0
    for i in range(len(x)):
        total += x[i] * weight[i]

    if total <= max_weight:
        if total <= reset_element:
            return reset_element - total
        else:
            return 0
    else:
        return -reset_element


# PARTICLE
class Particle:
    def __init__(self, x):
        self.position = []  # particle position
        self.velocity = []  # particle velocity
        self.personal_best = []  # best position individual
        self.err_personal_best = -1  # best error individual
        self.err_personal = -1  # error individual

        for i in range(num_dimensions):  # (0, num_dimensions):
            self.velocity.append(random.uniform(-1, 1))
            self.position.append(x[i])

    # evaluate current fitness
    def evaluate(self, cost_func):
        self.err_personal = cost_func(self.position)

        # check to see if the current position is an individual best
        if self.err_personal > self.err_personal_best or self.err_personal_best == -1:
            self.personal_best = self.position
            self.err_personal_best = self.err_personal

    def update_velocity(self, global_best):
        w = 0.9  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1.9  # cognitive constant
        c2 = 1.9  # social constant

        for i in range(num_dimensions):
            r1 = random.random()
            r2 = random.random()

            cognitive = c1 * r1 * (self.personal_best[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_position(self, bounds):
        for i in range(num_dimensions):
            max_bounds = (bounds[i][1] - bounds[i][0])

            if self.velocity[i] < -max_bounds:
                self.velocity[i] = -max_bounds
            elif self.velocity[i] > max_bounds:
                self.velocity[i] = max_bounds

            self.position[i] = self.position[i] + self.velocity[i]

            if self.position[i] > bounds[i][0]:
                # If the position is above the upper limit value, pull to the upper limit value
                self.position[i] = bounds[i][1]
            elif self.position[i] < bounds[i][0]:
                # If the position is below the lower limit value, pull down to the lower limit value
                self.position[i] = bounds[i][0]
            else:
                self.position[i] = round(self.position[i])


class PSO:
    step_kg, step_profit, global_best, err_global_best = [], [], [], -1

    def __init__(self, cost_func, x, bounds, num_particles, max_iter, verbose=False):
        global num_dimensions

        num_dimensions = len(x)
        self.err_global_best = -1  # best error for group
        self.global_best = []  # best position for group

        # establish the swarm
        swarm = []
        for i in range(num_particles):
            swarm.append(Particle(x))

        # begin optimization loop
        iteration = 0
        while iteration < max_iter:
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, num_particles):
                swarm[j].evaluate(cost_func)

                # determine if current particle is the best (globally)
                if swarm[j].err_personal > self.err_global_best or self.err_global_best == -1:
                    self.global_best = list(swarm[j].position)
                    self.err_global_best = float(swarm[j].err_personal)

            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(self.global_best)
                swarm[j].update_position(bounds)

            total_profit = 0
            total_kg = 0

            for i in range(num_particles):
                total_profit += self.global_best[i] * value[i]
                total_kg += self.global_best[i] * weight[i]

            self.step_profit.append(total_profit)
            self.step_kg.append(total_kg)

            if verbose:
                print(f'iter: {iteration:>4d}, best solution: {self.err_global_best:10.6f}')
            iteration += 1

            if verbose:
                print(self.global_best)

        # print final results
        print('\nFINAL:')
        print(f'global best -> {self.global_best}')
        print(f'err global best -> {self.err_global_best}')

    def Result(self):
        print('\n\nRESULTS:\n\n')
        total_profit = 0
        total_kg = 0
        for i in range(len(self.global_best)):
            print(objects[i], ': ', self.global_best[i], sep='')
            total_profit += self.global_best[i] * value[i]
            total_kg += self.global_best[i] * weight[i]
        print('#' * 50, '\nProfit: ', total_profit, ',\nKg: ', total_kg, sep='')

    # Plotting the results with plot [If we do not want to save the result image to the computer, the parameter named 'file_name' must remain empty!] ...
    def plot_result(self, file_name=''):
        plt.plot(self.step_kg, self.step_profit)
        plt.xlabel('Kilogram (kg)')
        plt.ylabel('Obtained PROFIT')
        plt.title('Profit vs. Kilogram Chart')
        plt.grid(True)

        if not (
                file_name == ''):  # If the variable named 'file_name' is not empty, save the file with that name in png format.
            file_name = file_name + ".png"
            plt.savefig(file_name)

        plt.show()
        plt.close()


# Global var
objects = ['Television', 'Camera', 'Projector', 'Walkman', 'Radio', 'Mobile Phone', 'Laptop Computer']
value = [40, 85, 135, 10, 25, 2, 94]
weight = [11, 3, 9, 0.5, 7, 0.5, 4]
max_weight = 25

# --- EXECUTE
print('[item_name: lower limit - upper limit] \n', sep='')
initial = []
bounds = []
for i in range(len(objects)):
    initial.append(0)
    bounds.append(
        (initial[i], math.floor(max_weight / weight[i])))
    print(objects[i], ': ', bounds[i][0], '-', bounds[i][1], sep='')
print('\nThere are a total of ', len(objects), ' variable...\n\n', sep='')

#   (cost_func, x, bounds, num_particles, max_iter):
pso = PSO(maximize, initial, bounds, len(objects), max_iter=20, verbose=True)
pso.Result()
pso.plot_result(file_name="pso_result")

# initial = [5, 5]  # initial starting location [x1,x2...]
# bounds = [(-10, 10), (-10, 10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
# PSO(cost, initial, bounds, num_particles=15, max_iter=30)
