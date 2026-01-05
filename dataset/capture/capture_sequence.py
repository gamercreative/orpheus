import pygame
import time
import json

SCREEN_SIZE = (800, 600)
OUTPUT_FILE = "dataset/strokes2.json"
FPS = 120

pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Draw your letter or shape")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 32)

all_sequences = []
current_preview = []
button_rect = pygame.Rect(650, 520, 130, 50)

def draw_button():
    pygame.draw.rect(screen, (200, 200, 200), button_rect, border_radius=8)
    text = font.render("Next", True, (0, 0, 0))
    screen.blit(text, (button_rect.x + 35, button_rect.y + 15))

def capture_sequence():
    global current_preview
    print("Press ENTER to begin drawing...")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                waiting = False
        clock.tick(FPS)

    print("Recording... Press ENTER again to stop.")
    sequence = []
    preview_points = []
    pen_down = False
    recording = True

    while recording:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                recording = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pen_down = True
            if event.type == pygame.MOUSEBUTTONUP:
                pen_down = False

            if event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                t = time.time()
                pen_state = 1 if pen_down else 0
                sequence.append([x, y, pen_state, t])

                if pen_state == 1:
                    preview_points.append((x, y))

        screen.fill((255, 255, 255))
        if len(preview_points) > 1:
            pygame.draw.lines(screen, (0, 0, 0), False, preview_points, 2)
        pygame.display.flip()

        clock.tick(FPS)

    # trim trailing pen-up noise
    while sequence and sequence[-1][2] == 0:
        sequence.pop()

    current_preview = preview_points.copy()
    print(f"Captured {len(sequence)} valid points.")
    return sequence

running = True
capture_ready = True
last_sequence = None

instance = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN and capture_ready == False:
            if button_rect.collidepoint(event.pos):
                instance += 1
                print(instance)
                if last_sequence:
                    all_sequences.append(last_sequence)
                last_sequence = None
                current_preview = []
                capture_ready = True

    if capture_ready:
        last_sequence = capture_sequence()
        if last_sequence is None:
            running = False
            break
        capture_ready = False
        capture_ready = False

    # draw stored preview + button
    screen.fill((255, 255, 255))
    if len(current_preview) > 1:
        pygame.draw.lines(screen, (0, 0, 0), False, current_preview, 2)
    draw_button()
    pygame.display.flip()

    clock.tick(FPS)

# save
with open(OUTPUT_FILE, "w") as f:
    json.dump(all_sequences, f, indent=2)

pygame.quit()
print(f"Saved {len(all_sequences)} sequences to {OUTPUT_FILE}")
