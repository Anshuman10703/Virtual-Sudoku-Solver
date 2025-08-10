# main.py
#
# To run this code, you will need to install the following libraries:
# pip install opencv-python mediapipe numpy
#
import cv2
import mediapipe as mp
import numpy as np
import time
import math

# --- Hand Gesture Recognition ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# --- Sudoku Puzzles and Drawing ---
# Pre-defined Sudoku puzzles (now with 10 puzzles)
puzzles = [
    [ # Puzzle 1 (Original)
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ],
    [ # Puzzle 2 (Original)
        [0, 0, 0, 2, 6, 0, 7, 0, 1],
        [6, 8, 0, 0, 7, 0, 0, 9, 0],
        [1, 9, 0, 0, 0, 4, 5, 0, 0],
        [8, 2, 0, 1, 0, 0, 0, 4, 0],
        [0, 0, 4, 6, 0, 2, 9, 0, 0],
        [0, 5, 0, 0, 0, 3, 0, 2, 8],
        [0, 0, 9, 3, 0, 0, 0, 7, 4],
        [0, 4, 0, 0, 5, 0, 0, 3, 6],
        [7, 0, 3, 0, 1, 8, 0, 0, 0]
    ],
    [ # Puzzle 3 (Original)
        [0, 2, 0, 6, 0, 8, 0, 0, 0],
        [5, 8, 0, 0, 0, 9, 7, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0],
        [3, 7, 0, 0, 0, 0, 5, 0, 0],
        [6, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 8, 0, 0, 0, 0, 1, 3],
        [0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 9, 8, 0, 0, 0, 3, 6],
        [0, 0, 0, 3, 0, 6, 0, 9, 0]
    ],
    [ # Puzzle 4 (Original)
        [0, 0, 0, 6, 0, 0, 4, 0, 0],
        [7, 0, 0, 0, 0, 3, 6, 0, 0],
        [0, 0, 0, 0, 9, 1, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 1, 8, 0, 0, 0, 3],
        [0, 0, 0, 3, 0, 6, 0, 4, 5],
        [0, 4, 0, 2, 0, 0, 0, 6, 0],
        [9, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 1, 0, 0]
    ],
    [ # Puzzle 5 (Original)
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0]
    ],
    [ # Puzzle 6 (New)
        [1, 0, 0, 4, 8, 9, 0, 0, 6],
        [7, 3, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 1, 2, 9, 5],
        [0, 0, 7, 1, 2, 0, 6, 0, 0],
        [5, 0, 0, 7, 0, 3, 0, 0, 8],
        [0, 0, 6, 0, 9, 5, 7, 0, 0],
        [9, 1, 4, 6, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 3, 7],
        [8, 0, 0, 5, 1, 2, 0, 0, 4]
    ],
    [ # Puzzle 7 (New)
        [0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 6, 0, 0, 0, 0, 3],
        [0, 7, 4, 0, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 0, 2],
        [0, 8, 0, 0, 4, 0, 0, 1, 0],
        [6, 0, 0, 5, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 7, 8, 0],
        [5, 0, 0, 0, 0, 9, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 0]
    ],
    [ # Puzzle 8 (New)
        [2, 0, 0, 3, 0, 0, 0, 0, 0],
        [8, 0, 4, 0, 6, 2, 0, 0, 3],
        [0, 1, 3, 8, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0, 3, 9, 0],
        [5, 0, 7, 0, 0, 0, 6, 2, 1],
        [0, 3, 2, 0, 0, 6, 0, 0, 0],
        [0, 2, 0, 0, 0, 9, 1, 4, 0],
        [6, 0, 1, 2, 5, 0, 8, 0, 9],
        [0, 0, 0, 0, 0, 1, 0, 0, 2]
    ],
    [ # Puzzle 9 (New)
        [0, 0, 0, 8, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 3],
        [5, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 7, 0, 8, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 2, 0, 0, 3, 0, 0, 0, 0],
        [6, 0, 0, 0, 0, 0, 0, 7, 5],
        [0, 0, 3, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 6, 0, 0]
    ],
    [ # Puzzle 10 (New)
        [0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 6, 0, 0, 0, 0, 3],
        [0, 7, 4, 0, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 0, 2],
        [0, 8, 0, 0, 4, 0, 0, 1, 0],
        [6, 0, 0, 5, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 7, 8, 0],
        [5, 0, 0, 0, 0, 9, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 0]
    ]
]

def is_valid(board, num, pos):
    """Checks if placing a number in a cell is valid according to Sudoku rules."""
    row, col = pos

    # Check row
    for i in range(9):
        if i != col and board[row][i] == num:
            return False

    # Check column
    for i in range(9):
        if i != row and board[i][col] == num:
            return False

    # Check 3x3 box
    box_x = col // 3
    box_y = row // 3
    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if (i, j) != pos and board[i][j] == num:
                return False
    
    return True


def draw_sudoku_grid(image, board, initial_board, validity_board, selected_cell, pointed_cell):
    """Draws the Sudoku grid and numbers on the image."""
    h, w, _ = image.shape
    grid_size = min(h, w) - 100
    start_x = (w - grid_size) // 2
    start_y = (h - grid_size) // 2
    cell_size = grid_size // 9

    # Draw grid lines
    for i in range(10):
        thick = 4 if i % 3 == 0 else 1
        cv2.line(image, (start_x + i * cell_size, start_y), (start_x + i * cell_size, start_y + grid_size), (255, 255, 255), thick)
        cv2.line(image, (start_x, start_y + i * cell_size), (start_x + grid_size, start_y + i * cell_size), (255, 255, 255), thick)

    # Highlight pointed and selected cells
    if pointed_cell:
        r, c = pointed_cell
        cv2.rectangle(image, (start_x + c * cell_size, start_y + r * cell_size), (start_x + (c + 1) * cell_size, start_y + (r + 1) * cell_size), (255, 255, 0), 3)
    
    if selected_cell:
        r, c = selected_cell
        cv2.rectangle(image, (start_x + c * cell_size, start_y + r * cell_size), (start_x + (c + 1) * cell_size, start_y + (r + 1) * cell_size), (0, 255, 0), 4)

    # Draw numbers
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                text = str(board[i][j])
                font_scale = 1.5
                thickness = 3
                
                # Set color based on validity
                if initial_board[i][j] != 0:
                    color = (255, 255, 255) # Initial numbers are white
                else:
                    if validity_board[i][j]:
                        color = (0, 255, 0) # Correct user numbers are green
                    else:
                        color = (0, 0, 255) # Incorrect user numbers are red

                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = start_x + j * cell_size + (cell_size - text_size[0]) // 2
                text_y = start_y + i * cell_size + (cell_size + text_size[1]) // 2
                cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    return start_x, start_y, cell_size, grid_size

def draw_ui_elements(image, highlighted_button=None):
    """Draws UI elements like the number pad and other buttons."""
    ui_buttons = {}
    
    # --- Number Pad ---
    button_size = 60
    margin = 20
    pad_start_x = image.shape[1] - (button_size * 3) - (margin * 4)
    pad_start_y = (image.shape[0] - (button_size * 4) - (margin * 3)) // 2

    for i in range(1, 10):
        row, col = (i - 1) // 3, (i - 1) % 3
        x, y = pad_start_x + col * (button_size + margin), pad_start_y + row * (button_size + margin)
        button_color = (0, 200, 0) if highlighted_button == i else (200, 200, 200)
        cv2.rectangle(image, (x, y), (x + button_size, y + button_size), button_color, -1)
        cv2.putText(image, str(i), (x + 15, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        ui_buttons[i] = (x, y, button_size, button_size)
        
    row, col = 3, 1
    x, y = pad_start_x + col * (button_size + margin), pad_start_y + row * (button_size + margin)
    button_color = (0, 200, 0) if highlighted_button == 0 else (150, 150, 200)
    cv2.rectangle(image, (x, y), (x + button_size, y + button_size), button_color, -1)
    cv2.putText(image, "0", (x + 15, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    ui_buttons[0] = (x, y, button_size, button_size)
    
    # --- Next Puzzle Button ---
    btn_x, btn_y, btn_w, btn_h = 20, 120, 250, 50
    button_color = (0, 200, 0) if highlighted_button == 'next' else (0, 150, 0)
    cv2.rectangle(image, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), button_color, -1)
    cv2.putText(image, "Next Puzzle", (btn_x + 25, btn_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ui_buttons['next'] = (btn_x, btn_y, btn_w, btn_h)

    return ui_buttons

def get_pointed_item(finger_coords, item_rects):
    """Checks if the finger is pointing at any of the provided item rectangles."""
    if finger_coords is None:
        return None
    
    x, y = finger_coords
    for key, rect in item_rects.items():
        rx, ry, rw, rh = rect
        if rx < x < rx + rw and ry < y < ry + rh:
            return key
    return None

def draw_dwell_indicator(image, coords, progress):
    """Draws a circular progress indicator for dwelling."""
    if coords:
        center = (int(coords[0]), int(coords[1]))
        radius = 20
        end_angle = int(progress * 360)
        cv2.ellipse(image, center, (radius, radius), 270, 0, end_angle, (255, 255, 255), 5)


# --- Main Application ---
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280) # Set width
    cap.set(4, 720)  # Set height

    puzzle_index = 0
    initial_board = [row[:] for row in puzzles[puzzle_index]]
    board = [row[:] for row in puzzles[puzzle_index]]
    validity_board = [[True for _ in range(9)] for _ in range(9)]
    
    selected_cell = None
    
    hover_item = None
    hover_start_time = 0
    DWELL_TIME = 0.7 

    highlighted_button = None
    highlight_time = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        finger_tip_coords = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                h, w, _ = image.shape
                finger_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
                finger_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                finger_tip_coords = (finger_x, finger_y)

        grid_start_x, grid_start_y, cell_size, grid_size = draw_sudoku_grid(image, board, initial_board, validity_board, selected_cell, None)
        
        if highlighted_button is not None and time.time() - highlight_time > 0.2:
            highlighted_button = None
        
        ui_buttons = draw_ui_elements(image, highlighted_button)

        current_time = time.time()
        
        pointed_item = None
        item_rects = {}
        
        if selected_cell is None:
            # Add grid cells and UI buttons to the list of hoverable items
            for r in range(9):
                for c in range(9):
                    if initial_board[r][c] == 0:
                        item_rects[(r, c)] = (grid_start_x + c * cell_size, grid_start_y + r * cell_size, cell_size, cell_size)
            item_rects.update(ui_buttons) # Add UI buttons
            pointed_item = get_pointed_item(finger_tip_coords, item_rects)
            if pointed_item and isinstance(pointed_item, tuple):
                 draw_sudoku_grid(image, board, initial_board, validity_board, None, pointed_item)
        else:
            # Only number pad is hoverable when a cell is selected
            item_rects = ui_buttons
            pointed_item = get_pointed_item(finger_tip_coords, item_rects)
            draw_sudoku_grid(image, board, initial_board, validity_board, selected_cell, None)

        if pointed_item is not None:
            if pointed_item != hover_item:
                hover_item = pointed_item
                hover_start_time = current_time
            
            dwell_progress = (current_time - hover_start_time) / DWELL_TIME
            draw_dwell_indicator(image, finger_tip_coords, dwell_progress)

            if dwell_progress >= 1.0:
                if isinstance(hover_item, tuple): # It's a grid cell
                    selected_cell = hover_item
                elif hover_item == 'next':
                    puzzle_index = (puzzle_index + 1) % len(puzzles)
                    initial_board = [row[:] for row in puzzles[puzzle_index]]
                    board = [row[:] for row in puzzles[puzzle_index]]
                    validity_board = [[True for _ in range(9)] for _ in range(9)]
                    selected_cell = None
                    highlighted_button = 'next'
                    highlight_time = time.time()
                elif selected_cell is not None: # It's a number from the pad
                    row, col = selected_cell
                    num = hover_item
                    board[row][col] = num
                    if num != 0:
                        validity_board[row][col] = is_valid(board, num, (row, col))
                    selected_cell = None
                    highlighted_button = hover_item
                    highlight_time = time.time()
                
                hover_item = None
        else:
            hover_item = None

        if selected_cell is None:
            cv2.putText(image, "Point and hold on an empty cell or button", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(image, "Point and hold on a number to input", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow('Virtual Sudoku Solver', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
