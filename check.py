#import modules
import pygame
from pygame.locals import *
import random
import numpy as np
import pickle
import time

n = 3
total_wins = 0
total_loses = 0
total_ties = 0

pygame.init()

screen_height = n*100
screen_width = n*100
line_width = 6
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Tic Tac Toe')

#define colours
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

#define font
font = pygame.font.SysFont(None, 40)

#define variables
clicked = False
player = 1
pos = (0,0)
markers = []
game_over = False
winner = 0

#setup a rectangle for "Play Again" Option
again_rect = Rect(screen_width // 2 - 80, screen_height // 2, 160, 50)

#create empty 3 x 3 list to represent the grid
for x in range (n):
    row = [0] * n
    markers.append(row)



def draw_board():
    bg = (255, 255, 210)
    grid = (50, 50, 50)
    screen.fill(bg)
    for x in range(1,n):
        pygame.draw.line(screen, grid, (0, 100 * x), (screen_width,100 * x), line_width)
        pygame.draw.line(screen, grid, (100 * x, 0), (100 * x, screen_height), line_width)

def draw_markers():
    x_pos = 0
    for x in markers:
        y_pos = 0
        for y in x:
            if y == 1:
                pygame.draw.line(screen, red, (x_pos * 100 + 15, y_pos * 100 + 15), (x_pos * 100 + 85, y_pos * 100 + 85), line_width)
                pygame.draw.line(screen, red, (x_pos * 100 + 85, y_pos * 100 + 15), (x_pos * 100 + 15, y_pos * 100 + 85), line_width)
            if y == -1:
                pygame.draw.circle(screen, green, (x_pos * 100 + 50, y_pos * 100 + 50), 38, line_width)
            y_pos += 1
        x_pos += 1    


def check_game_over():
    global game_over
    global winner
    global total_wins
    global total_loses
    global total_ties

    x_pos = 0
    for x in markers:
        #check columns
        if sum(x) == n:
            winner = 1
            game_over = True
        if sum(x) == -n:
            winner = 2
            game_over = True
        #check rows
        row_sum = sum([markers[i][x_pos] for i in range(n)])
        if row_sum == n:
            winner = 1
            game_over = True
        if row_sum == -n:
            winner = 2
            game_over = True
        x_pos += 1

    #check cross
    cross1_sum = sum([markers[i][i] for i in range(n)])
    cross2_sum = sum([markers[i][n-1-i] for i in range(n)])

    if cross1_sum == n or cross2_sum == n:
        winner = 1
        game_over = True
    if cross1_sum == -n or cross2_sum == -n:
        winner = 2
        game_over = True

    #check for tie
    if game_over == False:
        tie = True
        for row in markers:
            for i in row:
                if i == 0:
                    tie = False
        #if it is a tie, then call game over and set winner to 0 (no one)
        if tie == True:
            game_over = True
            winner = 0

    if game_over:
        total_wins += (winner == 1)
        total_loses += (winner == 2)
        total_ties += (winner == 0)


def draw_game_over(winner):

    if winner != 0:
        end_text = "Player " + str(winner) + " wins!"
    elif winner == 0:
        end_text = "You have tied!"

    end_img = font.render(end_text, True, blue)
    pygame.draw.rect(screen, green, (screen_width // 2 - 100, screen_height // 2 - 60, 200, 50))
    screen.blit(end_img, (screen_width // 2 - 100, screen_height // 2 - 50))

    again_text = 'Play Again?'
    again_img = font.render(again_text, True, blue)
    pygame.draw.rect(screen, green, again_rect)
    screen.blit(again_img, (screen_width // 2 - 80, screen_height // 2 + 10))


with open(f'policy_pi_{n}x{n}.pkl', 'rb') as f:
    policy = pickle.load(f)


#main loop
run = True
game = 0
while run:

    if game >= 100:
        break

    #draw board and markers first
    draw_board()
    draw_markers()

    #handle events
    for event in pygame.event.get():
        #handle game exit
        if event.type == pygame.QUIT:
            run = False
        #run new game
        if game_over == False:
            if player == 1:
                
                b_state = ''

                for i in range(n):
                    for j in range(n):
                        if markers[j][i] != -1:
                            b_state += str(markers[j][i])
                        else:
                            b_state += '2'

                state = int(b_state, 3)
                action = policy[state]

                y = int(action/n)
                x = int(action%n)

                markers[x][y] = player
                player *= -1
                check_game_over()

                
            elif player == -1:                
                # use the probability transition function for the opponent
                empty_cells = [(i, j) for i in range(n) for j in range(n) if markers[i][j]==0]
                cell_i = np.random.randint(0, len(empty_cells))
                cell = empty_cells[cell_i]
                markers[cell[0]][cell[1]] = player
                player *= -1
                check_game_over()

    #check if game has been won
    if game_over == True:
        draw_game_over(winner)
        #check for mouseclick to see if we clicked on Play Again
        # if event.type == pygame.MOUSEBUTTONDOWN and clicked == False:
        #     clicked = True
        # if event.type == pygame.MOUSEBUTTONUP and clicked == True:
        #     clicked = False
        #     pos = pygame.mouse.get_pos()
            # if again_rect.collidepoint(pos):
                #reset variables
        game_over = False
        player = 1
        pos = (0,0)
        markers = []
        winner = 0
        #create empty 3 x 3 list to represent the grid
        for x in range (n):
            row = [0] * n
            markers.append(row)
        
        game += 1
        

    #update display
    pygame.display.update()

pygame.quit()


print('Total wins', total_wins)
print('Total loses', total_loses)
print('Total ties', total_ties)
