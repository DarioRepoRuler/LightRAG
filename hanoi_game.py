import pygame
import sys
import time
import re
import json
import requests
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
import os

class TowerOfHanoi:
    def __init__(self, num_disks=3):
        self.num_disks = num_disks
        # Initialize pegs (S: source, A: auxiliary, T: target)
        self.pegs = {
            'S': list(range(num_disks, 0, -1)),  # Source peg with disks [3, 2, 1] for num_disks=3
            'A': [],                             # Auxiliary peg (empty)
            'T': []                              # Target peg (empty)
        }
        self.moves = []
        self.move_count = 0
        
    def get_state_description(self):
        """Generate a text description of the current state for the LLM."""
        # return f"""In tower of Hanoi puzzle with {self.num_disks} disks.
        #     The Tower of Hanoi rules dictate that you must move one disk at a time, only the top disk, between three pegs while never placing a larger disk on a smaller one.
        #     Disk numbers represent disk sizes, where a larger number = a larger disk.
        #     Current state (from bottom to top):
        #     Source peg (S): {self.pegs['S']}
        #     Auxiliary peg (A): {self.pegs['A']}
        #     Target peg (T): {self.pegs['T']}
        #     What is the next move to get all disks to the target peg T?
            
        #     Your answer must be formatted using the tag system: MDxYZ where:
        #     - M means Move
        #     - D followed by the disk number (e.g., Dx for disk x)
        #     - Source peg letter (S, A, or T)
        #     - Target peg letter (S, A, or T)
        #     For example, moving disk x from Source to Target would be MDxST.
        # If no move is needed because all disks are already on the target peg, use NM for No Move. Answer with the tag system only."""
        return f"""
            In tower of Hanoi puzzle with 3 disks.\
        The Tower of Hanoi rules dictate that you must move one disk at a time, only the top disk, between three pegs while never placing a larger disk on a smaller one. \
        Disk numbers represent disk sizes, where a larger number = a larger disk.\
        Current state (from bottom to top):\
        Source peg (S): {self.pegs['S']}
        Auxiliary peg (A): {self.pegs['A']}
        Target peg (T): {self.pegs['T']}
        What is the next move to get all disks to the target peg T?\
        Your answer must be formatted using the tag system: MDxYZ where:\
        - M means Move\
        - D followed by the disk number (e.g., Dx for disk x)\
        - Source peg letter (S, A, or T)\
        - Target peg letter (S, A, or T)\
        For example, moving disk x from Source to Target would be MDxST.\
        If no move is needed because all disks are already on the target peg, use NM for No Move.\
        Answer with the tag system only.\
        """

    def is_valid_move(self, disk, source, target):
        """Check if a move is valid."""
        # Check if the disk exists on the source peg
        if not self.pegs[source] or self.pegs[source][-1] != disk:
            return False
        
        # Check if the target peg is empty or the top disk is larger than the moving disk
        if not self.pegs[target] or self.pegs[target][-1] > disk:
            return True
        
        return False
    
    def make_move(self, disk, source, target):
        """Make a move if it's valid."""
        if self.is_valid_move(disk, source, target):
            # Remove the disk from the source
            self.pegs[source].pop()
            # Add the disk to the target
            self.pegs[target].append(disk)
            # Record the move
            self.moves.append((disk, source, target))
            self.move_count += 1
            return True
        return False
    
    def is_solved(self):
        """Check if the puzzle is solved (all disks are on target peg)."""
        return len(self.pegs['T']) == self.num_disks
    
    def parse_llm_response(self, response):
        """Parse the LLM response to extract the move tag."""
        # Look for {MDxYZ} pattern
        tag_match = re.search(r'\{(MD\d+[SAT][SAT])\}', response)
        if tag_match:
            return tag_match.group(1)
        
        # Also look for the tag without braces
        tag_match = re.search(r'MD\d+[SAT][SAT]', response)
        if tag_match:
            return tag_match.group(0)
        
        return None
    
    def execute_move_from_tag(self, move_tag):
        """Execute a move based on the tag from LLM."""
        if move_tag == "NM":
            print("No move needed - puzzle is solved.")
            return True
        
        # Parse the move tag format MDxYZ
        match = re.match(r'MD(\d+)([SAT])([SAT])', move_tag)
        if not match:
            print(f"Invalid move tag format: {move_tag}")
            return False
        
        disk = int(match.group(1))
        source = match.group(2)
        target = match.group(3)
        
        print(f"Executing move: Disk {disk} from {source} to {target}")
        return self.make_move(disk, source, target)

 
class TowerOfHanoiVisualizer:
    def __init__(self, hanoi, use_matplotlib=True, agent_type = 'LightRAG'):
        self.hanoi = hanoi
        self.use_matplotlib = use_matplotlib
        
        if use_matplotlib:
            self.setup_matplotlib()
        else:
            self.setup_pygame()
        if agent_type == 'Ollama':
            self.agent = OllamaAgent()
        elif agent_type == 'LightRAG':
            self.agent = LightRAGAgent()
        else:
            raise ValueError("Invalid agent type. Choose 'Ollama' or 'LightRAG'.")
    
    def setup_matplotlib(self):
        """Setup matplotlib visualization."""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ani = None
        
        # Define colors for disks
        self.colors = plt.cm.viridis(np.linspace(0, 1, self.hanoi.num_disks))
        
        # Set peg positions
        self.peg_positions = {'S': 1, 'A': 2, 'T': 3}
        
    def setup_pygame(self):
        """Setup pygame visualization."""
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tower of Hanoi with LLM Agent")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BROWN = (139, 69, 19)
        self.disk_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
        ]
        
        # Peg positions
        self.peg_width = 20
        self.peg_height = 300
        self.base_height = 20
        self.peg_positions = {
            'S': self.width // 4,
            'A': self.width // 2,
            'T': 3 * self.width // 4
        }
        
        # Disk dimensions
        self.max_disk_width = 200
        self.disk_height = 40
        self.disk_width_step = self.max_disk_width // (self.hanoi.num_disks + 1)
        
        # Font for text
        self.font = pygame.font.SysFont('Arial', 24)
        
        self.clock = pygame.time.Clock()
    
    def update_matplotlib(self, frame):
        """Update matplotlib visualization for animation."""
        self.ax.clear()
        
        # Draw pegs
        peg_height = 1.0
        peg_width = 0.05
        
        for peg, pos in self.peg_positions.items():
            # Draw peg
            self.ax.add_patch(plt.Rectangle((pos - peg_width/2, 0), peg_width, peg_height, color='brown'))
            
            # Draw disks on this peg
            disks = self.hanoi.pegs[peg]
            for i, disk in enumerate(disks):
                disk_width = 0.1 + (disk * 0.2)  # Scale disk width based on size
                disk_height = 0.1
                bottom = i * disk_height
                self.ax.add_patch(plt.Rectangle(
                    (pos - disk_width/2, bottom),
                    disk_width,
                    disk_height,
                    color=self.colors[disk-1]
                ))
        
        # Set plot limits and labels
        self.ax.set_xlim(0, 4)
        self.ax.set_ylim(0, 1.2)
        self.ax.set_xticks([1, 2, 3])
        self.ax.set_xticklabels(['Source', 'Auxiliary', 'Target'])
        self.ax.set_title(f'Tower of Hanoi - Move {self.hanoi.move_count}')
        
        if self.hanoi.moves:
            last_move = self.hanoi.moves[-1]
            self.ax.text(0.5, 1.1, f'Last move: Disk {last_move[0]} from {last_move[1]} to {last_move[2]}',
                      ha='center', transform=self.ax.transAxes)
        
        if self.hanoi.is_solved():
            self.ax.text(0.5, 1.05, 'Puzzle Solved!', ha='center', fontsize=16,
                      color='green', transform=self.ax.transAxes)
    
    def run_pygame_visualization(self):
        """Run the pygame visualization loop."""
        running = True
        solved = False
        # agent = OllamaAgent()
        agent = self.agent
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            self.screen.fill(self.WHITE)
            
            # Draw base
            pygame.draw.rect(self.screen, self.BROWN,
                            (self.width//8, self.height - 100,
                             3*self.width//4, self.base_height))
            
            # Draw pegs and labels
            for peg, x_pos in self.peg_positions.items():
                # Draw peg
                pygame.draw.rect(self.screen, self.BROWN,
                                (x_pos - self.peg_width//2,
                                 self.height - 100 - self.peg_height,
                                 self.peg_width, self.peg_height))
                
                # Draw peg label
                label = self.font.render(f"{peg} Peg", True, self.BLACK)
                self.screen.blit(label, (x_pos - 30, self.height - 70))
                
                # Draw disks on this peg
                disks = self.hanoi.pegs[peg]
                for i, disk in enumerate(disks):
                    disk_width = self.disk_width_step * disk
                    y_pos = self.height - 100 - (i + 1) * self.disk_height
                    pygame.draw.rect(self.screen, self.disk_colors[disk-1],
                                    (x_pos - disk_width//2, y_pos,
                                     disk_width, self.disk_height-5))
                    
                    # Draw disk number
                    disk_label = self.font.render(str(disk), True, self.WHITE)
                    self.screen.blit(disk_label, (x_pos - 5, y_pos + 5))
            
            # Draw move counter
            move_text = self.font.render(f"Moves: {self.hanoi.move_count}", True, self.BLACK)
            self.screen.blit(move_text, (50, 50))
            
            # Show last move if available
            if self.hanoi.moves:
                last_move = self.hanoi.moves[-1]
                last_move_text = self.font.render(
                    f"Last move: Disk {last_move[0]} from {last_move[1]} to {last_move[2]}",
                    True, self.BLACK)
                self.screen.blit(last_move_text, (50, 90))
            
            # Check if puzzle is solved
            if self.hanoi.is_solved() and not solved:
                solved = True
                solved_text = self.font.render("Puzzle Solved!", True, (0, 128, 0))
                self.screen.blit(solved_text, (self.width//2 - 80, 50))
            
            pygame.display.flip()
            
            # Get next move from LLM if not solved
            if not solved:
                state_description = self.hanoi.get_state_description()
                llm_response = agent.get_next_move(state_description)
                
                if llm_response:
                    move_tag = self.hanoi.parse_llm_response(llm_response)
                    if move_tag:
                        self.hanoi.execute_move_from_tag(move_tag)
                    else:
                        print("Couldn't extract move tag from LLM response")
                        print("Response:", llm_response)
                
                # Slow down animation
                time.sleep(1)
            
            self.clock.tick(30)
        
        pygame.quit()
    
    def run_matplotlib_visualization(self):
        """Run the matplotlib visualization."""
        agent = self.agent
        
        def animate(i):
            if not self.hanoi.is_solved():
                state_description = self.hanoi.get_state_description()
                llm_response = agent.get_next_move(state_description)
                
                if llm_response:
                    move_tag = self.hanoi.parse_llm_response(llm_response)
                    if move_tag:
                        self.hanoi.execute_move_from_tag(move_tag)
                    else:
                        print("Couldn't extract move tag from LLM response")
                        print("Response:", llm_response)
            
            self.update_matplotlib(i)
            
            # Stop animation if solved
            if self.hanoi.is_solved():
                self.ani.event_source.stop()
        
        self.update_matplotlib(0)  # Initial state
        self.ani = FuncAnimation(self.fig, animate, frames=20, interval=1500, repeat=False)
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """Run the visualization."""
        if self.use_matplotlib:
            self.run_matplotlib_visualization()
        else:
            self.run_pygame_visualization()
 
# Simulate with fake LLM responses for testing
class MockOllamaAgent:
    def __init__(self):
        # Predefined optimal moves for 3-disk Tower of Hanoi
        self.optimal_moves = [
            "MD1ST", "MD2SA", "MD1TS", "MD3ST", "MD1AT", "MD2AT", "MD1ST"
        ]
        self.move_index = 0
    
    def get_next_move(self, state_description):
        """Return a predefined move."""
        if self.move_index < len(self.optimal_moves):
            move = self.optimal_moves[self.move_index]
            self.move_index += 1
            return f"I'll move disk {move[2]} from {move[3]} to {move[4]}. {{MD{move[2]}{move[3]}{move[4]}}}"
        return "NM"  # No Move
    
class OllamaAgent:
    def __init__(self, model="qwen2mm"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
        
    def get_next_move(self, state_description):
        """Query the Ollama LLM for the next move."""
        try:
            payload = {
                "model": self.model,
                "prompt": state_description,
                "stream": False
            }
            print(f"Querying Ollama API with payload: {payload}")
            print(f"Prompt: {state_description}")
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"Response from Ollama API: {result}")
                return result.get("response", "")
            else:
                print(f"Error querying Ollama API: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception when calling Ollama API: {e}")
            return None

class LightRAGAgent:
    def __init__(self, model="qwen2mm", context="./tower_of_hanoi_dataset.txt"):
        self.model = model
        self.api_url = "http://localhost:11434"
        WORKING_DIR = "./tower_of_hanoi"
        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)
        self.rag = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=ollama_model_complete,
                llm_model_name="qwen2mm",
                llm_model_max_async=4,
                llm_model_max_token_size=32768,
                llm_model_kwargs={"host": self.api_url, "options": {"num_ctx": 32768}},
                embedding_func=EmbeddingFunc(
                    embedding_dim=768,
                    max_token_size=8192,
                    func=lambda texts: ollama_embedding(
                        texts, embed_model="nomic-embed-text", host=self.api_url
                    ),
                ),
        )
        with open("./tower_of_hanoi_dataset.txt", "r", encoding="utf-8") as f:
            self.rag.insert(f.read())

    def get_next_move(self, state_description, mode="hybrid"):
        """Query the Ollama LLM for the next move."""
        try:
            response = self.rag.query(state_description,param=QueryParam(mode=mode))
            move_tag = self.extract_move_tag(response)
        except Exception as e:
            print(f"Exception when calling LightRAG: {e}")
            return None
        return move_tag
    
    def extract_move_tag(self,response):
        print("Full Response:")
        print(response)

        print("\nExtracted Move:")
        move_code = None
        # Try first to find a move code that appears after "optimal" or "recommended"
        final_move_pattern = r"(?:optimal|recommended).*?[{]?([MN][DM][0-9]+[AST]{2}|NM)[}]?"
        match = re.search(final_move_pattern, response, re.IGNORECASE | re.DOTALL)

        # If not found with the above pattern, try looking for markdown headings
        if not match:
            heading_move_pattern = r"#{2,4}\s*[{]?([MN][DM][0-9]+[AST]{2}|NM)[}]?"
            match = re.search(heading_move_pattern, response)

        # If still not found, get the last occurrence of any move code
        if not match:
            all_matches = re.findall(r"[{]?([MN][DM][0-9]+[AST]{2}|NM)[}]?", response)
            if all_matches:
                move_code = all_matches[-1]
                print(move_code)
            else:
                print("No valid move code found in the response")

        return move_code
    
def main():
    # Create Tower of Hanoi game
    hanoi = TowerOfHanoi(num_disks=3)
    
    # Create visualizer
    visualizer = TowerOfHanoiVisualizer(hanoi, use_matplotlib=True)  # True for matplotlib, False for pygame
    
    input("Press Enter to start the game...")
    
    # Run visualization
    visualizer.run()
 
if __name__ == "__main__":
    main()