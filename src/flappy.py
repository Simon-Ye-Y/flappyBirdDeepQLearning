import asyncio
import csv
import sys
import time
from pathlib import Path

import pygame
import torch
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

import numpy as np

from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window
from FlapPyBird.src.DQN import Agent


class Flappy:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()
        self.num_episodes = 50
        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )
        self.agent = Agent(
            gamma=0.99,
            epsilson=0.5,
            lr=0.001,
            input_dims=[3],
            batch_size=32,
            n_actions=2,
            max_mem_size=100000,
            eps_end=0.05,
            eps_dec=1e-4
        )
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_metrics.csv"
        if not self.log_file.exists():
            with self.log_file.open("w", newline="") as log_file:
                writer = csv.writer(log_file)
                writer.writerow(["episode", "total_reward", "average_loss", "epsilon"])
        self.print_interval = 10

    async def start(self):
        recent_rewards = []
        for episode in range(1, self.num_episodes + 1):
            self._reset_game()
            total_reward, average_loss = await self.train()
            self._log_episode(episode, total_reward, average_loss)
            recent_rewards.append(total_reward)
            if episode % self.print_interval == 0:
                window = recent_rewards[-self.print_interval:]
                avg_reward = sum(window) / len(window)
                print(
                    f"Episode {episode}: average reward {avg_reward:.2f}, "
                    f"epsilon {self.agent.epsilon:.3f}"
                )

    async def train(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)
        done = False
        observation = np.array(self.closest_entity(), dtype=np.float32)
        total_reward = 0.0
        total_loss = 0.0
        loss_updates = 0
        flap_cooldown = 15  # 设置一个冷却时间，例如10帧
        flap_counter = 0  # 初始化冷却计数器
        while not done:
            for event in pygame.event.get():
                self.check_quit_event(event)

            action = 0
            pipe_distance = max(abs(float(observation[2])), 1.0)
            if flap_counter == 0:
                action = self.select_action(observation)
                if action == 1:  # 如果选择跳跃
                    self.player.flap()
                    flap_counter = flap_cooldown
            else:
                flap_counter -= 1

            crossed = any(self.player.crossed(pipe) for pipe in self.pipes.upper)
            if crossed:
                self.score.add()

            next_state = np.array(self.closest_entity(), dtype=np.float32)
            done = self.player.collided(self.pipes, self.floor)

            step_reward = 100.0 if crossed else 0.0
            if self.pipes.upper and self.pipes.lower:
                upper_pipe = self.pipes.upper[0]
                lower_pipe = self.pipes.lower[0]
                if self.player.y > upper_pipe.rect.bottom or self.player.y < lower_pipe.rect.top:
                    step_reward -= 50.0 / pipe_distance
                else:
                    step_reward += 50.0 / pipe_distance
            if done:
                step_reward -= 100.0

            total_reward += step_reward
            self.agent.store_transition(observation, action, step_reward, next_state, done)
            loss = self.agent.learn()
            if loss is not None:
                total_loss += loss
                loss_updates += 1

            observation = next_state

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()
        average_loss = total_loss / loss_updates if loss_updates else 0.0
        return total_reward, average_loss

    async def splash(self):
        """Shows welcome splash screen animation of flappy bird"""

        self.player.set_mode(PlayerMode.SHM)

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    return

            self.background.tick()
            self.floor.tick()
            self.player.tick()
            self.welcome_message.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (
                event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (
                event.key == K_SPACE or event.key == K_UP
        )
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    async def play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        while True:
            if self.player.collided(self.pipes, self.floor):
                return

            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        """crashes the player down and shows gameover image"""

        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    if self.player.y + self.player.h >= self.floor.y - 1:
                        return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            self.config.tick()
            pygame.display.update()
            await asyncio.sleep(0)

    def closest_entity(self):
        # Assuming screen width as screen_width
        screen_height = self.config.window.height
        player = self.player

        # Set default values
        default_top = screen_height
        default_bot = 0

        # Get the position of the nearest upper and lower pipes
        nearest_pipe_x = 0
        nearest_entity_y_upper_bottom = default_bot
        nearest_entity_y_lower_top = default_top

        if len(self.pipes.upper) > 0 and len(self.pipes.lower) > 0:
            nearest_upper_pipe = self.pipes.upper[0]
            nearest_pipe_x = nearest_upper_pipe.x
            nearest_entity_y_upper_bottom = nearest_upper_pipe.rect.bottom
            nearest_lower_pipe = self.pipes.lower[0]
            nearest_entity_y_lower_top = nearest_lower_pipe.rect.top

        # Calculate the distance to the floor
        # distance_to_floor = self.config.window.height - player.rect.y - player.rect.height / 2

        state = [
            player.rect.y - nearest_entity_y_upper_bottom,
            # Vertical distance from the player to the nearest upper pipe
            player.rect.y - nearest_entity_y_lower_top,  # Vertical distance from the player to the nearest lower pipe
            nearest_pipe_x - player.rect.x,  # Horizontal distance from the player to the nearest pipe
            # distance_to_floor,  # Distance from the player to the floor

            # player.vel_y  # Player's vertical velocity
        ]
        return state

    def select_action(self, observation):
        state_tensor = torch.tensor(observation, dtype=torch.float32).to(self.agent.Q_eval.device)
        return self.agent.choose_action(state_tensor)

    def _reset_game(self):
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.player = Player(self.config)
        self.welcome_message = WelcomeMessage(self.config)
        self.game_over_message = GameOver(self.config)
        self.pipes = Pipes(self.config)
        self.score = Score(self.config)

    def _log_episode(self, episode, total_reward, average_loss):
        with self.log_file.open("a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([episode, total_reward, average_loss, self.agent.epsilon])
