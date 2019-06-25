import pygame
import numpy as np
import math
import os

def zig_zag(side, remove=0.0, filename=None):


    # The image is tacitly based on a 17 by 17 grid
    square = 2 * (side // 34)
    origin = (side - (17 * square)) // 2

    # Represent the image as an array
    # For this pattern, there is a dot every four squares on the x axis
    #   and only one dot size per row. The pattern has four rows. Large (3)
    #   dots fall on row one, medium (2) dots on rows two and four, and small (1)
    #   dots on row three 
    sketch = np.zeros((17, 17), dtype=np.float32)
    for i in range(17):
        for j in range(17):
            if j % 4 == 0 and i % 4 == 2:
                    sketch[i][j] = 1
            if j % 4 == 1 and (i % 8 == 1 or i % 8 == 7):
                    sketch[i][j] = 2
            if j % 4 == 2 and i % 4 == 0:
                    sketch[i][j] = 3
            if j % 4 ==  3 and (i % 8 == 3 or i % 8 == 5):
                    sketch[i][j] = 2

    # Define colors
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Set up the canvas
    canvas = pygame.display.set_mode((side, side))

    # Prepare to draw circles
    circles = []
    for i in range(17):
        for j in range(17):
            center_x = origin + int(square * (0.5 + j))
            center_y = origin + int(square * (0.5 + i))
            if sketch[i][j] == 3: 
                circles.append([[center_x, center_y], int(square/2)])
            if sketch[i][j] == 2:
                circles.append([[center_x, center_y], int(square/4)])
            if sketch[i][j] == 1:
                circles.append([[center_x, center_y], int(square/8)])

    # Remove randomly selected circles
    to_remain = len(circles) - math.ceil(len(circles) * remove)
    while len(circles) > to_remain:
        del circles[np.random.randint(0, len(circles) - 1)]

    # Display the image until the user does something
    #   or the image is saved
    done = False
    clock = pygame.time.Clock()

    while done == False:
        clock.tick(10)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        canvas.fill(white)

        # Draw circles
        for i in circles:
            pygame.draw.circle(canvas, black, i[0], i[1])

        if filename != None:
            pygame.image.save(canvas, filename)
            done = True
        else:
            pygame.display.flip()
    pygame.quit()


def diagonal(side, remove=0.0, filename=None):


    # The image is tacitly based on a 17 by 17 grid
    square = 2 * (side // 34)
    origin = (side - (17 * square)) // 2

    # Represent the image as an array
    # For this pattern, there is a dot every four squares on the x axis
    #   and only one dot size per row. The pattern has four rows. Large (3)
    #   dots fall on row one, medium (2) dots on rows two and four, and small (1)
    #   dots on row three 
    sketch = np.zeros((17, 17), dtype=np.float32)
    for i in range(17):
        for j in range(17):
            if j % 4 == 0 and i % 4 == 2:
                    sketch[i][j] = 3
            if (j % 4 == 1 and i % 4 == 3) or (j % 4 == 3 and i % 4 == 1):
                sketch[i][j] = 2
            if j % 4 == 2 and i % 4 == 0:
                    sketch[i][j] = 1

    # Define colors
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Set up the canvas
    canvas = pygame.display.set_mode((side, side))

    # Prepare to draw circles
    circles = []
    for i in range(17):
        for j in range(17):
            center_x = origin + int(square * (0.5 + j))
            center_y = origin + int(square * (0.5 + i))
            if sketch[i][j] == 3: 
                circles.append([[center_x, center_y], int(3*square/4)])
            if sketch[i][j] == 2:
                circles.append([[center_x, center_y], int(3*square/8)])
            if sketch[i][j] == 1:
                circles.append([[center_x, center_y], int(3*square/16)])

    # Remove randomly selected circles
    to_remain = len(circles) - math.ceil(len(circles) * remove)
    while len(circles) > to_remain:
        del circles[np.random.randint(0, len(circles) - 1)]

    # Display the image until the user does something
    #   or the image is saved
    done = False
    clock = pygame.time.Clock()

    while done == False:
        clock.tick(10)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        canvas.fill(white)

        # Draw circles
        for i in circles:
            pygame.draw.circle(canvas, black, i[0], i[1])

        if filename != None:
            pygame.image.save(canvas, filename)
            done = True
        else:
            pygame.display.flip()
    pygame.quit()


def gen_images(side, framework, number, remove_range, filename):


    remove_values = (np.random.rand(10) * remove_range[1]) + remove_range[0]
    for i in range(number):
        fname = filename + '_' + str(i+1) + '.png'
        if framework == 'diagonal':
            diagonal(side, remove_values[i], fname)
        if framework == 'zig_zag': 
            zig_zag(side, remove_values[i], fname)

gen_images(500, 'zig_zag', 10, (.1, .7), 'zero')
pygame.quit()