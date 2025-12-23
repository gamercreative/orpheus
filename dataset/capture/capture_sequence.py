import pygame
import numpy as np
import time
import json

# ---------------- CONFIG ----------------
SCREEN_SIZE = (800, 600)
OUTPUT_FILE = "dataset/strokes.json"
FPS = 120  # capture rate
# ----------------------------------------

pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Draw your letter or shape")
clock = pygame.time.Clock()

all_strokes = []  # store sequences

def record_stroke():
    stroke = []  # (x, y, pen_down, timestamp)
    drawing = True
    pen_down = False

    while drawing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pen_down = True
            elif event.type == pygame.MOUSEBUTTONUP:
                pen_down = False
                drawing = False  # stop after pen lifted
            elif event.type == pygame.MOUSEMOTION:
                if pen_down:
                    x, y = event.pos
                    t = time.time()
                    stroke.append([x, y, 1, t])

        clock.tick(FPS)
        screen.fill((255, 255, 255))
        pygame.display.flip()

    return stroke

running = True
while running:
    screen.fill((255, 255, 255))
    pygame.display.flip()

    print("Draw one letter or shape with your tablet, then lift pen.")
    stroke = record_stroke()
    if stroke is None:
        break

    print(f"Captured {len(stroke)} points.")
    all_strokes.append(stroke)

    cont = input("Record another? (y/n): ")
    if cont.lower() != 'y':
        running = False

with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_strokes, f)

pygame.quit()
print(f"Saved {len(all_strokes)} sequences to {OUTPUT_FILE}")
