import gymnasium as gym
import numpy as np
import cv2
import random

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

# Initialize the Donkey Kong environment
env = gym.make('ALE/DonkeyKong-v5', render_mode="rgb_array")
obs, _ = env.reset()

# Define parameters for the genetic algorithm
population_size = 50
generations = 50
mutation_rate = 0.1
crossover_rate = 0.5
gene_length = 500  # Number of actions in a sequence
actions = env.action_space.n

# Function to extract the player's position from the game screen
def get_player_position(obs):
    # Convert the observation to a format suitable for OpenCV (RGB to BGR)
    image = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    # Define the color range for the player's detection (red color) (In BGR format)
    lower_color = np.array([62, 62, 190])
    upper_color = np.array([82, 82, 210])

    # Create a mask for the player's color
    mask = cv2.inRange(image, lower_color, upper_color)

    # Apply morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Visualize the mask for debugging
    cv2.imshow("Mask", mask)
    cv2.waitKey(1)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assuming the player is the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y)  # Return the (x, y) coordinate of the player's position
    return (0, 0)  # Return (0, 0) if the player is not found

# Function to determine the player's floor based on the y-coordinate
def get_floor(y):
    # Approximate y-coordinates for floors in Donkey Kong (example values, adjust as necessary)
    floor_heights = [190, 150, 110, 70, 30]
    for i, height in enumerate(floor_heights):
        if y >= height:
            return i + 1
    return len(floor_heights) + 1

# Define the fitness function
def fitness_function(genes):
    obs, info = env.reset(seed=seed)
    total_reward = 0
    max_height_achieved = float('inf')
    max_horizontal_distance = 0
    previous_position = get_player_position(obs)
    
    for action in genes:
        obs, reward, done, truncated, info = env.step(action)
        current_position = get_player_position(obs)
        
        # Calculate vertical progress
        max_height_achieved = min(max_height_achieved, current_position[1])
        
        # Calculate horizontal progress based on the current floor
        current_floor = get_floor(current_position[1])
        if current_floor % 2 == 1:  # Odd floors: reward moving right
            if current_position[0] > previous_position[0]:
                max_horizontal_distance += current_position[0] - previous_position[0]
        else:  # Even floors: reward moving left
            if current_position[0] < previous_position[0]:
                max_horizontal_distance += previous_position[0] - current_position[0]
        
        previous_position = current_position
        
        # Check for loss of life and reset if necessary
        if 'lives' in info and info['lives'] < 2:
            env.reset(seed=seed)
            break
        if done or truncated:
            break
    
    # Fitness combines vertical and horizontal progress with higher weight on vertical progress
    fitness_score = -max_height_achieved
    return fitness_score

# Initialize the population with random actions
population = [np.random.randint(0, actions, gene_length) for _ in range(population_size)]

# Genetic Algorithm
for generation in range(generations):
    print(f'Generation: {generation}')
    
    # Evaluate fitness of each individual in a single thread
    fitness_scores = np.array([fitness_function(individual) for individual in population])

    # Filter out individuals with a fitness of 0 and replace them
    for i in range(population_size):
        while fitness_scores[i] == 0:
            population[i] = np.random.randint(0, actions, gene_length)
            fitness_scores[i] = fitness_function(population[i])

    max_fitness = np.max(fitness_scores)
    print(f'Highest fitness: {max_fitness}')

    # Select the best individuals to be parents
    sorted_indices = np.argsort(fitness_scores)[-population_size // 2:]
    parents = [population[i] for i in sorted_indices]

    # Generate new population through crossover and mutation
    new_population = []
    for _ in range(population_size):
        if np.random.rand() < crossover_rate:
            # Crossover
            parent1, parent2 = random.sample(parents, 2)
            crossover_point = np.random.randint(0, gene_length)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        else:
            # Select a random parent
            child = random.choice(parents)

        # Mutation
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(0, gene_length)
            child[mutation_point] = np.random.randint(0, actions)

        new_population.append(child)

    population = new_population

# Select the best individual from the final population
best_individual = population[np.argmax([fitness_function(ind) for ind in population])]
print('Best individual actions:', best_individual)

# Test the best individual
obs, info = env.reset(seed=seed)
total_reward = 0
for action in best_individual:
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if 'lives' in info and info['lives'] < 3:
        obs, info = env.reset(seed=seed)
    if done or truncated:
        break
print('Total reward:', total_reward)
env.close()
