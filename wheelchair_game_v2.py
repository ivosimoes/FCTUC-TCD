import pygame
import random
import math
import sounddevice as sd
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
import joblib
import librosa
from sklearn.preprocessing import MinMaxScaler

# init pygame
pygame.init()

# define constants
SCREEN_WIDTH, SCREEN_HEIGHT = 855, 675
CELL_SIZE = 15  # cell size
CELL_EXPANSION = 3  # number of cells within each cell
FPS = 10
MAZE_ROWS = SCREEN_HEIGHT // CELL_SIZE // CELL_EXPANSION  # number of rows
MAZE_COLS = SCREEN_WIDTH // CELL_SIZE // CELL_EXPANSION # number of cols
INITIAL_SCORE = 0  # counter of crashes

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


# game screen set up
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Wheelchair Labyrinth")

last_classification_time = time.time()


class Maze:
    ''' Maze class. Sets up and draws the maze.'''
    
    def __init__(self):
        # base grid
        self.base_grid = [[1 for _ in range(MAZE_COLS)] for _ in range(MAZE_ROWS)]
        # expanded grid
        self.expanded_grid = [[1 for _ in range(MAZE_COLS * CELL_EXPANSION)] for _ in range(MAZE_ROWS * CELL_EXPANSION)]
        
        self.generate_maze()
        self.expand_maze()
        
        # place the target in a randow white cell
        self.target = self.get_random_white_cell()

    def generate_maze(self):
        # simple DFS algorithm to build the maze
        
        stack = [(1, 1)]
        self.base_grid[1][1] = 0

        while stack:
            row, col = stack[-1]
            neighbors = []

            if row > 1 and self.base_grid[row - 2][col] == 1:
                neighbors.append((row - 2, col))
            if row < MAZE_ROWS - 2 and self.base_grid[row + 2][col] == 1:
                neighbors.append((row + 2, col))
            if col > 1 and self.base_grid[row][col - 2] == 1:
                neighbors.append((row, col - 2))
            if col < MAZE_COLS - 2 and self.base_grid[row][col + 2] == 1:
                neighbors.append((row, col + 2))

            if neighbors:
                next_row, next_col = random.choice(neighbors)
                self.base_grid[next_row][next_col] = 0
                self.base_grid[(next_row + row) // 2][(next_col + col) // 2] = 0
                stack.append((next_row, next_col))
            else:
                stack.pop()

    def expand_maze(self):
        # expand rows so each place is a CELL_EXPANSION by CELL_EXPANSION mini-grid of places
        for row in range(MAZE_ROWS):
            for col in range(MAZE_COLS):
                if self.base_grid[row][col] == 0:
                    # expand cells
                    for i in range(CELL_EXPANSION):
                        for j in range(CELL_EXPANSION):
                            self.expanded_grid[row * CELL_EXPANSION + i][col * CELL_EXPANSION + j] = 0

    def get_random_white_cell(self):
        # find random white cell
        while True:
            row = random.randint(0, MAZE_ROWS-1)
            col = random.randint(0, MAZE_COLS-1)
            if self.expanded_grid[int((row+0.5)*CELL_EXPANSION)][int((col+0.5)*CELL_EXPANSION)] == 0:
                return (int((col+0.5)*CELL_EXPANSION), int((row+0.5)*CELL_EXPANSION))

    def draw(self):
        # draw the maze
        for row in range(MAZE_ROWS * CELL_EXPANSION):
            for col in range(MAZE_COLS * CELL_EXPANSION):
                color = WHITE if self.expanded_grid[row][col] == 0 else BLACK
                pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # draw the target
        target_x, target_y = self.target
        pygame.draw.rect(screen, GREEN, (target_x * CELL_SIZE, target_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))


class Wheelchair:
    '''Wheelchair class. Controls the wheelchair and game dynamics.'''
    
    def __init__(self, maze):
        self.x, self.y = 1.5*CELL_SIZE*CELL_EXPANSION, 1.5*CELL_SIZE*CELL_EXPANSION
        self.direction = 0  # angle in degrees (0 is right)
        self.speed = 0
        self.maze = maze  
        self.score = INITIAL_SCORE  # initial score
        self.win = False  # win flag

    def can_move(self, dx, dy):
        # compute new position
        new_x = (self.x + dx) // CELL_SIZE
        new_y = (self.y + dy) // CELL_SIZE

        # check if new position is white and whithin the board
        if 0 <= new_x < len(self.maze.expanded_grid[0]) and 0 <= new_y < len(self.maze.expanded_grid):
            return self.maze.expanded_grid[int(new_y)][int(new_x)] == 0
        return False

    def move(self):
        if self.win:
            return  # after win, disallow movement

        # compute dx and dy based on the direction (angle in degrees)
        rad = math.radians(self.direction)
        dx = self.speed * math.cos(rad)
        dy = -self.speed * math.sin(rad)

        # check if can move before changing position
        if self.can_move(dx, dy):
            self.x += dx
            self.y += dy
        else:
            # count collision
            self.score += 1
            self.speed = 0

    def rotate(self, direction):
        if self.win:
            return  # stop rotation after winning
        if direction == "LEFT":
            self.direction = (self.direction + 45) % 360
        elif direction == "RIGHT":
            self.direction = (self.direction - 45) % 360

    def stop(self):
        self.speed = 0

    def execute_command(self, command):
        if command == "FORWARD":
            self.speed = CELL_SIZE // 2
        elif command == "BACKWARD":
            self.speed = -CELL_SIZE // 2
        elif command == "LEFT":
            self.rotate("LEFT")
        elif command == "RIGHT":
            self.rotate("RIGHT")
        elif command == "STOP":
            self.stop()

    def check_win(self):
        # check if target is reached
        target_x, target_y = self.maze.target
        return (self.x // CELL_SIZE, self.y // CELL_SIZE) == (target_x, target_y)

    def draw(self):
        # draw wheelchair as a triangle
        rad = math.radians(self.direction)
        front_x = self.x + CELL_SIZE * math.cos(rad)
        front_y = self.y - CELL_SIZE * math.sin(rad)
        left_x = self.x + CELL_SIZE * math.cos(rad + math.radians(135))
        left_y = self.y - CELL_SIZE * math.sin(rad + math.radians(135))
        right_x = self.x + CELL_SIZE * math.cos(rad - math.radians(135))
        right_y = self.y - CELL_SIZE * math.sin(rad - math.radians(135))

        pygame.draw.polygon(screen, BLUE, [(front_x, front_y), (left_x, left_y), (right_x, right_y)]) 


def sound_capture_thread(wheelchair):
    '''thread for capturing the microphone sound and converting it to a command to the wheelchair.'''
    
    RATE = 16000  # sampling rate of 16kHz
    RECORD_SECONDS = 1  # capture 1 second of sound
    OVERLAP_SECONDS = .95  # overlap between segments
    NO_OVERLAP_SECONDS = RECORD_SECONDS - OVERLAP_SECONDS # new data on each segment

    AMP_THRESHOLD = 0.2  # minimum amplitude to consider sound
    GAP_TIME_THRESHOLD = 0.5  # minimum time between classifications

    buffer_size = int(RATE * RECORD_SECONDS)
    buffer = np.zeros(buffer_size, dtype=np.float32)

    def callback(indata, frames, timestamp, status):
        global last_classification_time
        nonlocal buffer
        if status:
            print(status)
        
        # add the new data to the buffer, rolling the current data to the left
        buffer = np.roll(buffer, -len(indata))
        buffer[-len(indata):] = indata[:, 0]

        center_buffer = buffer[len(buffer)//2 - len(buffer)//10 : len(buffer)//2 + len(buffer)//10]
        if np.max(np.abs(center_buffer)) < AMP_THRESHOLD or (time.time() - last_classification_time) < GAP_TIME_THRESHOLD:
            return
        
        last_classification_time = time.time()
        
        # convert the captured sound to a command
        command = process_sound(buffer)

        if command:
            wheelchair.execute_command(command)

    # open audio stream from microphone
    with sd.InputStream(device='Microfone (Realtek(R) Audio), MME', callback=callback, channels=1, samplerate=RATE, blocksize=int(RATE * NO_OVERLAP_SECONDS)):
        while True:
            time.sleep(NO_OVERLAP_SECONDS)  # Aguardar NO_OVERLAP_SECONDS


def main():
    '''Main function of the game.'''
    global start_recording_flag, start_recording_time
    
    clock = pygame.time.Clock()
    maze = Maze()
    wheelchair = Wheelchair(maze)
    running = True
    start_ticks = pygame.time.get_ticks()  # Tempo inicial

    # start sound capture thread
    sound_thread = threading.Thread(target=sound_capture_thread, args=(wheelchair,))
    sound_thread.daemon = True
    sound_thread.start()

    while running:
        screen.fill(WHITE)
        maze.draw()
        wheelchair.draw()

        # render points on screen
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Colisions: {wheelchair.score}", True, RED)
        screen.blit(score_text, (10, 10))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if wheelchair.win:
                    continue  # stop wheelchair control after winning
                if event.key == pygame.K_w:  # Forward
                    wheelchair.execute_command("FORWARD")
                elif event.key == pygame.K_s:  # Backward
                    wheelchair.execute_command("BACKWARD")
                elif event.key == pygame.K_a:  # Rotate left (45 degrees)
                    wheelchair.execute_command("LEFT")
                elif event.key == pygame.K_d:  # Rotate right (45 degrees)
                    wheelchair.execute_command("RIGHT")
                elif event.key == pygame.K_SPACE:  # Stop
                    wheelchair.execute_command("STOP")
                


        # check victory
        if wheelchair.check_win():
            wheelchair.win = True  # register victory
            font = pygame.font.SysFont(None, 44)
            win_text = font.render("Congratulations! You reached the target location!", True, GREEN)
            screen.blit(win_text, (SCREEN_WIDTH // 15, SCREEN_HEIGHT // 2))
            wheelchair.stop()  # stop the wheelchair

        wheelchair.move()

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

commands = {0: "", 1: "FORWARD", 2: "BACKWARD", 3: "LEFT", 4: "RIGHT", 5: "STOP"}

# load classifier from file
classifier = joblib.load('model.pkl')

def classify(features):
    ''' receives a feature vector and returns a prediction from the classifier '''
    prediction = classifier.predict(features)
    probs = classifier.predict_proba(features)
    return prediction, probs

def extract_features(sound):
    ''' extract the features of the new sound segment '''

    X_mfcc = librosa.feature.mfcc(y=sound, sr=16000, n_mfcc=10, n_fft=1024)
    X_dmfcc = librosa.feature.delta(data=X_mfcc)
    X_ddmfcc = librosa.feature.delta(data=X_mfcc, order=2)
    X_zcr = librosa.feature.zero_crossing_rate(y=sound)
    
    X_mfcc_scaled = np.concat((np.mean(X_mfcc.T, axis=0), np.std(X_mfcc.T, axis=0)))
    X_dmfcc_scaled = np.concat((np.mean(X_dmfcc.T, axis=0), np.std(X_dmfcc.T, axis=0)))
    X_ddmfcc_scaled = np.concat((np.mean(X_ddmfcc.T, axis=0), np.std(X_ddmfcc.T, axis=0)))
    X_zcr_scaled = np.array([np.mean(X_zcr), np.std(X_zcr)])
    
    X_all = np.concat((X_mfcc_scaled, X_dmfcc_scaled, X_ddmfcc_scaled, X_zcr_scaled))
    return X_all

def process_sound(sound):
    ''' receives a 1-second sound segment from the microphone and returns a command '''
    command = ""
    
    # Uncomment the follwing lines:
    
    # Load the scaler fitted with train and validation data and PCA ranking for best feature amount
    scaler = joblib.load('scaler.pkl')
    pca_ranks = joblib.load('feature_ranks.pkl')
    
    #time = [i*(1/16000) for i in range(0, 16000)]  # Instants in seconds
    #plt.plot(time, sound)
    #plt.xlabel("Time (s)")
    #plt.ylabel("Amplitude")
    #plt.title("Audio plot")
    #plt.savefig("Example.png")
    #plt.close()
    
    feature_vector = extract_features(sound)
    feature_vector = feature_vector[pca_ranks]  # Extract the best features according to PCA and LBFGS
    feature_vector = scaler.transform(feature_vector.reshape(1, -1))  # Model is trained according to minmax scaling
    command, probs = classify(feature_vector)
    print(f"Command: {command.item()}\nProbs: {probs}\n")
    
    # Silence and unknown are equivalent to no command
    command = "" if command == '_unknown_' or command == '_silence_' else command.item().upper()
    
    return command

if __name__ == "__main__":
    main()