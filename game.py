import pygame
import numpy as np
import torch
import os
import random
from collections import deque
import heapq  

pygame.init()
fps = 30

ACTIONS_4 = [(-1,0),(1,0),(0,1),(0,-1)]  # UP, DOWN, RIGHT, LEFT
CNN_CKPT_PATH = r"cnn_next_move_4dir_120k.pt" # add path
cnn_device = "cuda" if torch.cuda.is_available() else "cpu"

cnn_model = NextMoveCNN(in_ch=4, base=32, num_actions=4).to(cnn_device)
ckpt = torch.load(CNN_CKPT_PATH, map_location=cnn_device)
cnn_model.load_state_dict(ckpt["model"])
cnn_model.eval()

def _is_free_cell(map_np, r, c):
    return 0 <= r < map_np.shape[0] and 0 <= c < map_np.shape[1] and map_np[r, c] == 0

@torch.no_grad()
def cnn_next_step(map_np, occ_torch, start, goal, player_pos, prev_pos=None, recent=None):
    H, W = map_np.shape
    occ = occ_torch  

    s = torch.zeros((H, W), dtype=torch.float32)
    g = torch.zeros((H, W), dtype=torch.float32)
    c = torch.zeros((H, W), dtype=torch.float32)

    s[int(start[0]), int(start[1])] = 1.0
    g[int(goal[0]),  int(goal[1])]  = 1.0
    c[int(player_pos[0]), int(player_pos[1])] = 1.0

    X = torch.stack([occ.cpu(), s, g, c], dim=0).unsqueeze(0).to(cnn_device)  # [1,4,H,W]
    logits = cnn_model(X)[0]  # [4]
    probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
    order = np.argsort(-probs)  # best first

    r, col = int(player_pos[0]), int(player_pos[1])

    for a in order:
        dr, dc = ACTIONS_4[int(a)]
        nr, nc = r + dr, col + dc
        if not _is_free_cell(map_np, nr, nc):
            continue
        if prev_pos is not None and (nr, nc) == (int(prev_pos[0]), int(prev_pos[1])):
            continue
        if recent is not None and (nr, nc) in recent:
            continue
        return (nr, nc)

    return None

def find_nearest_free_cell(goal, map_np):
    h, w = map_np.shape
    visited = np.zeros_like(map_np, dtype=bool)
    q = deque()
    q.append((goal[0], goal[1]))
    visited[goal[0], goal[1]] = True

    while q:
        r, c = q.popleft()
        if map_np[r, c] == 0:
            return [r, c]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))
    return goal

test_path = r"map_dataset/test"  # add path
pt_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith(".pt")]
if not pt_files:
    raise FileNotFoundError("No .pt files in the selected folder")

def load_random_map():
    global map_np, start, goal, path, h, w
    sample_path = random.choice(pt_files)
    print("Chosen file:", sample_path)

    sample = MapSample.load(sample_path)
    map_np, start, goal, path = sample.numpy()

    if map_np[goal[0], goal[1]] != 0:
        goal[:] = find_nearest_free_cell(goal, map_np)

    h, w = map_np.shape

load_random_map()

occ_torch = (torch.tensor(map_np, dtype=torch.float32) > 0).float()

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width, screen_height = screen.get_size()

def recompute_geometry():
    global cell_size, map_width_px, map_height_px, offset_x, offset_y
    cell_size = min(screen_width // w, screen_height // h)
    map_width_px = w * cell_size
    map_height_px = h * cell_size
    offset_x = (screen_width - map_width_px) // 2
    offset_y = (screen_height - map_height_px) // 2

recompute_geometry()

pygame.display.set_caption("Path Planning Game")
player_pos = list(start)


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)        
DARK_BLUE = (0, 0, 139)    
RED = (255, 0, 0)


BTN_BG = (40, 40, 40)
BTN_BG_HOVER = (70, 70, 70)
BTN_TEXT = (255, 0, 0)


PLAYER_SKINS = [
    ("PURPLE",     (255, 0, 255)),
    ("RED",        (255, 0, 0)),
    ("DARK GREEN", (0, 100, 0)),
    ("ORANGE",     (255, 165, 0)),
    ("CYAN",       (0, 255, 255)),
    ("PINK",       (255, 105, 180)),
]
selected_skin_idx = 0
player_color = PLAYER_SKINS[0][1]

has_started = False

clock = pygame.time.Clock()
running = True

move_delay = 3
frame_count = 0

show_game_over = False
game_over_timer = 0
game_over_duration = 60

steps = 0
start_time = None
elapsed_time = 0
level_finished = False



INF = float("inf")

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(r, c, h, w, map_np):
    out = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and map_np[nr, nc] == 0:
            out.append((nr, nc))
    return out

class DStarLite:
    def __init__(self):
        self.s_start = None
        self.s_goal = None
        self.s_last = None
        self.k_m = 0.0
        self.g = {}
        self.rhs = {}
        self.U = []
        self.open_key = {}

    def _get_g(self, s):
        return self.g.get(s, INF)

    def _get_rhs(self, s):
        return self.rhs.get(s, INF)

    def _set_rhs(self, s, v):
        self.rhs[s] = v

    def calculate_key(self, s):
        v = min(self._get_g(s), self._get_rhs(s))
        return (v + heuristic(self.s_start, s) + self.k_m, v)

    def _push(self, s):
        k = self.calculate_key(s)
        self.open_key[s] = k
        heapq.heappush(self.U, (k[0], k[1], s))

    def _top_key(self):
        while self.U:
            k1, k2, s = self.U[0]
            cur = self.open_key.get(s, None)
            if cur is None or (k1, k2) != cur:
                heapq.heappop(self.U)
                continue
            return (k1, k2)
        return (INF, INF)

    def _pop(self):
        while self.U:
            k1, k2, s = heapq.heappop(self.U)
            cur = self.open_key.get(s, None)
            if cur is None or (k1, k2) != cur:
                continue
            del self.open_key[s]
            return (k1, k2, s)
        return (INF, INF, None)

    def update_vertex(self, u, map_np):
        h, w = map_np.shape
        if u != self.s_goal:
            best = INF
            for s2 in get_neighbors(u[0], u[1], h, w, map_np):
                best = min(best, 1 + self._get_g(s2))
            self._set_rhs(u, best)

        if u in self.open_key:
            del self.open_key[u]

        if self._get_g(u) != self._get_rhs(u):
            self._push(u)

    def compute_shortest_path(self, map_np):
        while self._top_key() < self.calculate_key(self.s_start) or self._get_rhs(self.s_start) != self._get_g(self.s_start):
            k_old_1, k_old_2, u = self._pop()
            if u is None:
                break

            k_new = self.calculate_key(u)

            if (k_old_1, k_old_2) < k_new:
                self._push(u)
                continue

            if self._get_g(u) > self._get_rhs(u):
                self.g[u] = self._get_rhs(u)
                for p in get_neighbors(u[0], u[1], map_np.shape[0], map_np.shape[1], map_np):
                    self.update_vertex(p, map_np)
            else:
                self.g[u] = INF
                self.update_vertex(u, map_np)
                for p in get_neighbors(u[0], u[1], map_np.shape[0], map_np.shape[1], map_np):
                    self.update_vertex(p, map_np)

    def initialize(self, map_np, start, goal):
        self.s_start = (int(start[0]), int(start[1]))
        self.s_goal  = (int(goal[0]),  int(goal[1]))
        self.s_last  = self.s_start
        self.k_m = 0.0
        self.g = {}
        self.rhs = {}
        self.U = []
        self.open_key = {}

        self.rhs[self.s_goal] = 0.0
        self.g[self.s_goal] = INF
        self._push(self.s_goal)
        self.compute_shortest_path(map_np)

    def update_start(self, map_np, new_start):
        new_s = (int(new_start[0]), int(new_start[1]))
        self.k_m += heuristic(self.s_last, new_s)
        self.s_last = new_s
        self.s_start = new_s
        self.compute_shortest_path(map_np)

    def next_step(self, map_np):
        s = self.s_start
        if s is None or self.s_goal is None:
            return None
        if s == self.s_goal:
            return s
        if self._get_g(s) == INF and self._get_rhs(s) == INF:
            return None

        best = None
        best_val = INF
        for s2 in get_neighbors(s[0], s[1], map_np.shape[0], map_np.shape[1], map_np):
            val = 1 + self._get_g(s2)
            if val < best_val:
                best_val = val
                best = s2
        return best

dstar = DStarLite()



prev_pos = None
recent_pos = deque(maxlen=25)

npc_policy = "cnn"  
no_progress = 0

NO_PROGRESS_LIMIT = 20 
MAX_TOTAL_STEPS_NPC = 6000 

def _dist_to_goal(pos, goal_rc):
    return abs(int(pos[0]) - int(goal_rc[0])) + abs(int(pos[1]) - int(goal_rc[1]))

def switch_to_dstar_from_current():
    global npc_policy, no_progress
    npc_policy = "dstar"
    no_progress = 0
    dstar.initialize(map_np, player_pos, goal)



def draw_button(surface, rect, text, mouse_pos, mouse_clicked, font):
    hovered = rect.collidepoint(mouse_pos)
    bg = BTN_BG_HOVER if hovered else BTN_BG
    pygame.draw.rect(surface, bg, rect, border_radius=14)
    pygame.draw.rect(surface, RED, rect, width=2, border_radius=14)

    label = font.render(text, True, BTN_TEXT)
    surface.blit(label, (rect.centerx - label.get_width() // 2,
                         rect.centery - label.get_height() // 2))
    return hovered and mouse_clicked



game_mode = "menu"  # menu / player / npc / character

def reset_same_map():
    global player_pos, show_game_over, game_over_timer
    global steps, start_time, elapsed_time, level_finished
    global has_started, prev_pos, recent_pos, npc_policy, no_progress

    player_pos = list(start)
    prev_pos = None
    recent_pos.clear()

    npc_policy = "cnn"
    no_progress = 0

    has_started = False
    show_game_over = False
    game_over_timer = 0
    steps = 0
    start_time = None
    elapsed_time = 0
    level_finished = False

def reset_new_map():
    global occ_torch
    load_random_map()
    occ_torch = (torch.tensor(map_np, dtype=torch.float32) > 0).float()
    recompute_geometry()
    reset_same_map()

def start_player_mode():
    global game_mode
    game_mode = "player"
    reset_same_map()

def start_npc_mode():
    global game_mode, prev_pos, npc_policy, no_progress
    game_mode = "npc"
    reset_same_map()
    prev_pos = None
    npc_policy = "cnn"
    no_progress = 0

def back_to_menu():
    global game_mode
    game_mode = "menu"
    reset_same_map()

def switch_mode_same_map():
    global game_mode
    if game_mode == "player":
        start_npc_mode()
    elif game_mode == "npc":
        start_player_mode()



def open_character_menu():
    global game_mode
    game_mode = "character"

def close_character_menu():
    global game_mode
    game_mode = "menu"

def apply_selected_skin():
    global player_color
    player_color = PLAYER_SKINS[selected_skin_idx][1]



title_font = pygame.font.SysFont(None, 72)
menu_font = pygame.font.SysFont(None, 38)
btn_font = pygame.font.SysFont(None, 34)

btn_w, btn_h = 320, 70
play_btn = pygame.Rect(0, 0, btn_w, btn_h)
sim_btn  = pygame.Rect(0, 0, btn_w, btn_h)
char_btn = pygame.Rect(0, 0, btn_w, btn_h)
exit_btn = pygame.Rect(0, 0, btn_w, btn_h)

play_btn.center = (screen_width // 2, screen_height // 2 - 40)
sim_btn.center  = (screen_width // 2, screen_height // 2 + 50)
char_btn.center = (screen_width // 2, screen_height // 2 + 140)
exit_btn.center = (screen_width // 2, screen_height // 2 + 230)



row_y = screen_height // 2 + 130
gap = 18
small_w, small_h = 220, 60

menu_btn = pygame.Rect(0, 0, small_w, small_h)
restart_btn = pygame.Rect(0, 0, small_w, small_h)
newmap_btn = pygame.Rect(0, 0, small_w, small_h)
switch_btn = pygame.Rect(0, 0, small_w, small_h)

total_w = small_w * 4 + gap * 3
start_x = (screen_width - total_w) // 2
menu_btn.topleft = (start_x + (small_w + gap) * 0, row_y)
restart_btn.topleft = (start_x + (small_w + gap) * 1, row_y)
newmap_btn.topleft = (start_x + (small_w + gap) * 2, row_y)
switch_btn.topleft = (start_x + (small_w + gap) * 3, row_y)

# Main loop

npc_steps_total = 0

while running:
    mouse_pos = pygame.mouse.get_pos()
    mouse_clicked = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_clicked = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if game_mode in ("player", "npc"):
                if event.key == pygame.K_r:
                    reset_same_map()

                if event.key == pygame.K_n:
                    reset_new_map()

    # Menu
    
    if game_mode == "menu":
        screen.fill(BLACK)

        title = title_font.render("PATH PLANNING GAME", True, RED)
        screen.blit(title, (screen_width // 2 - title.get_width() // 2,
                            screen_height // 3 - title.get_height() // 2))

        skin_name, _ = PLAYER_SKINS[selected_skin_idx]
        picked = menu_font.render(f"Selected Character: {skin_name}", True, RED)
        screen.blit(picked, (screen_width // 2 - picked.get_width() // 2,
                             screen_height // 3 + 40))

        if draw_button(screen, play_btn, "PLAY", mouse_pos, mouse_clicked, btn_font):
            start_player_mode()

        if draw_button(screen, sim_btn, "SIMULATION", mouse_pos, mouse_clicked, btn_font):
            start_npc_mode()
            npc_steps_total = 0

        if draw_button(screen, char_btn, "CHARACTER", mouse_pos, mouse_clicked, btn_font):
            open_character_menu()

        if draw_button(screen, exit_btn, "EXIT", mouse_pos, mouse_clicked, btn_font):
            running = False

        pygame.display.flip()
        clock.tick(fps)
        continue

    # Character
    if game_mode == "character":
        screen.fill(BLACK)

        title = title_font.render("CHOOSE CHARACTER", True, RED)
        screen.blit(title, (screen_width // 2 - title.get_width() // 2,
                            screen_height // 5 - title.get_height() // 2))

        cols_opt = 3
        box_w, box_h = 260, 80
        gap_x, gap_y = 22, 18

        total_grid_w = cols_opt * box_w + (cols_opt - 1) * gap_x
        start_x_grid = (screen_width - total_grid_w) // 2
        start_y_grid = screen_height // 5 + 80

        for i, (name, color) in enumerate(PLAYER_SKINS):
            rr = i // cols_opt
            cc = i % cols_opt
            rect = pygame.Rect(
                start_x_grid + cc * (box_w + gap_x),
                start_y_grid + rr * (box_h + gap_y),
                box_w, box_h
            )

            pygame.draw.rect(screen, BTN_BG, rect, border_radius=14)
            pygame.draw.rect(
                screen,
                RED if i == selected_skin_idx else (120, 120, 120),
                rect, width=2, border_radius=14
            )

            preview = pygame.Rect(rect.x + 18, rect.y + 18, 44, 44)
            pygame.draw.rect(screen, color, preview)
            pygame.draw.rect(screen, (0, 0, 0), preview, 2)

            label = btn_font.render(name, True, BTN_TEXT)
            screen.blit(label, (preview.right + 16, rect.centery - label.get_height() // 2))

            if rect.collidepoint(mouse_pos) and mouse_clicked:
                selected_skin_idx = i
                apply_selected_skin()

        back_rect = pygame.Rect(0, 0, 260, 70)
        back_rect.center = (screen_width // 2, screen_height - 110)

        if draw_button(screen, back_rect, "BACK", mouse_pos, mouse_clicked, btn_font):
            close_character_menu()

        pygame.display.flip()
        clock.tick(fps)
        continue


    
    if not level_finished:
        frame_count += 1
        if frame_count >= move_delay:
            moved = False
            new_pos = player_pos.copy()

            if game_mode == "player":
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    new_pos[0] -= 1
                if keys[pygame.K_DOWN]:
                    new_pos[0] += 1
                if keys[pygame.K_LEFT]:
                    new_pos[1] -= 1
                if keys[pygame.K_RIGHT]:
                    new_pos[1] += 1

                if 0 <= new_pos[0] < h and 0 <= new_pos[1] < w:
                    if map_np[new_pos[0], new_pos[1]] == 0:
                        moved = (new_pos != player_pos)
                        player_pos = new_pos
                        show_game_over = False
                    else:
                        show_game_over = True
                        game_over_timer = game_over_duration
                        player_pos = list(start)
                        steps = 0
                        start_time = None
                        elapsed_time = 0
                        has_started = False

            elif game_mode == "npc":
                npc_steps_total += 1

                if npc_steps_total > MAX_TOTAL_STEPS_NPC and npc_policy != "dstar":
                    switch_to_dstar_from_current()

                cur = (int(player_pos[0]), int(player_pos[1]))
                goal_rc = (int(goal[0]), int(goal[1]))
                dist_before = _dist_to_goal(cur, goal_rc)

                if npc_policy == "cnn":
                    nxt = cnn_next_step(
                        map_np, occ_torch, start, goal, player_pos,
                        prev_pos=prev_pos, recent=recent_pos
                    )

                    if nxt is None:
                        no_progress += 1
                    else:
                        nxt_pos = (int(nxt[0]), int(nxt[1]))
                        dist_after = _dist_to_goal(nxt_pos, goal_rc)

                        if dist_after >= dist_before:
                            no_progress += 1
                        else:
                            no_progress = max(0, no_progress - 1)

                        new_pos = [nxt_pos[0], nxt_pos[1]]
                        moved = (new_pos != player_pos)
                        prev_pos = player_pos.copy()
                        player_pos = new_pos

                        recent_pos.append((int(player_pos[0]), int(player_pos[1])))

                    if no_progress >= NO_PROGRESS_LIMIT:
                        switch_to_dstar_from_current()

                else:
                    if dstar.s_start is None:
                        dstar.initialize(map_np, player_pos, goal)

                    nxt = dstar.next_step(map_np)
                    if nxt is not None:
                        new_pos = [nxt[0], nxt[1]]
                        moved = (new_pos != player_pos)
                        prev_pos = player_pos.copy()
                        player_pos = new_pos
                        recent_pos.append((int(player_pos[0]), int(player_pos[1])))

                        dstar.update_start(map_np, (player_pos[0], player_pos[1]))

            if moved:
                if not has_started:
                    has_started = True
                if start_time is None:
                    start_time = pygame.time.get_ticks()
                steps += 1

            frame_count = 0

        elapsed_time = (pygame.time.get_ticks() - start_time) / 1000 if start_time is not None else 0

        if player_pos == list(goal):
            level_finished = True

    # Map
    screen.fill(BLACK)
    for i in range(h):
        for j in range(w):
            color = WHITE if map_np[i, j] == 0 else BLACK
            pygame.draw.rect(
                screen, color,
                (offset_x + j * cell_size, offset_y + i * cell_size, cell_size, cell_size)
            )

    pygame.draw.rect(screen, GREEN,
                     (offset_x + start[1] * cell_size, offset_y + start[0] * cell_size, cell_size, cell_size))
    pygame.draw.rect(screen, DARK_BLUE,
                     (offset_x + goal[1] * cell_size, offset_y + goal[0] * cell_size, cell_size, cell_size))

    pygame.draw.rect(screen, player_color,
                     (offset_x + player_pos[1] * cell_size, offset_y + player_pos[0] * cell_size, cell_size, cell_size))

    if not has_started:
        start_font = pygame.font.SysFont(None, 26)
        start_text = start_font.render("START", True, RED)
        text_x = offset_x + player_pos[1] * cell_size + cell_size // 2 - start_text.get_width() // 2
        text_y = offset_y + player_pos[0] * cell_size - start_text.get_height() - 6
        screen.blit(start_text, (text_x, text_y))

    if show_game_over and game_over_timer > 0 and game_mode == "player":
        go_font = pygame.font.SysFont(None, 48)
        go_text = go_font.render("GAME OVER!", True, RED)
        screen.blit(go_text, (screen_width // 2 - go_text.get_width() // 2,
                              screen_height // 2 - go_text.get_height() // 2))
        game_over_timer -= 1

    info_font = pygame.font.SysFont(None, 30)
    mode_label = "PLAY" if game_mode == "player" else "SIMULATION"
    if game_mode == "npc":
        stats = info_font.render(
            f"Mode: {mode_label} | NPC: {npc_policy.upper()} | Steps: {steps}  Time: {elapsed_time:.2f}s",
            True, RED
        )
    else:
        stats = info_font.render(f"Mode: {mode_label} | Steps: {steps}  Time: {elapsed_time:.2f}s", True, RED)
    screen.blit(stats, (10, 10))

    hint_font = pygame.font.SysFont(None, 26)
    hint = hint_font.render("R - restart | N - new map | ESC - exit", True, RED)
    screen.blit(hint, (10, screen_height - 30))

    if level_finished:
        if game_mode == "player":
            win_font = pygame.font.SysFont(None, 54)
            win_text = win_font.render("You Win!", True, RED)
            screen.blit(win_text, (screen_width // 2 - win_text.get_width() // 2,
                                   screen_height // 2 - win_text.get_height() // 2))

        if draw_button(screen, menu_btn, "MENU", mouse_pos, mouse_clicked, btn_font):
            back_to_menu()

        if draw_button(screen, restart_btn, "RESTART", mouse_pos, mouse_clicked, btn_font):
            reset_same_map()
            npc_steps_total = 0

        if draw_button(screen, newmap_btn, "NEW MAP", mouse_pos, mouse_clicked, btn_font):
            reset_new_map()
            npc_steps_total = 0

        switch_label = "SWITCH TO SIM" if game_mode == "player" else "SWITCH TO PLAY"
        if draw_button(screen, switch_btn, switch_label, mouse_pos, mouse_clicked, btn_font):
            switch_mode_same_map()
            npc_steps_total = 0

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
