import pygame
import random
import numpy as np
import pymunk
import pymunk.pygame_util
import torch
import torch.nn.functional as F
from pathlib import Path

class Game:
    def __init__(self, headless=False, use_vqvae=False, vqvae_checkpoint=None, 
                 play_mode="optimal", terrain_mode="default", collecting_data=False,
                 random_jump_prob=0.15, miss_prob=0.10):
        # Initialize Pygame
        pygame.init()
        
        # Display settings
        self.BLOCK_SIZE = 60  # Pixels per grid block
        self.SCREEN_WIDTH = 14 * self.BLOCK_SIZE
        self.SCREEN_HEIGHT = 7 * self.BLOCK_SIZE
        self.FPS = 60
        self.headless = headless
        self.use_vqvae = use_vqvae
        
        # Data collection and behavior modes
        self.play_mode = play_mode  # "optimal", "random", "noisy", "mixed"
        self.terrain_mode = terrain_mode  # "default", "obstacle_rich", "balanced"
        self.collecting_data = collecting_data  # Suppress UI text during collection
        self.random_jump_prob = random_jump_prob
        self.miss_prob = miss_prob
        
        # VQ-VAE setup
        self.vqvae_model = None
        self.vqvae_device = None
        if use_vqvae:
            self._load_vqvae(vqvae_checkpoint)
        
        # Create window
        if headless:
            # Create a hidden surface for headless mode
            self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        else:
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption('GD Clone' + (' [VQ-VAE]' if use_vqvae else ''))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.SKY_BLUE = (135, 206, 235)
        self.CUBE_RED = (255, 107, 107)
        self.CUBE_BORDER = (0, 0, 0)
        self.BLOCK_GREEN = (76, 175, 80)
        self.BLOCK_BORDER = (45, 80, 22)
        self.SPIKE_RED = (255, 0, 0)
        self.SPIKE_BORDER = (139, 0, 0)
        self.TEXT_COLOR = (0, 0, 0)
        self.DEAD_COLOR = (255, 0, 0)
        
        # Font
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Physics space
        self.space = pymunk.Space() 
        self.space.gravity = (0, -2400)  # Downward gravity in pixels/s^2
        self.space.iterations = 10
        
        # Collision types
        self.COLLISION_TYPE_PLAYER = 1
        self.COLLISION_TYPE_BLOCK = 2
        self.COLLISION_TYPE_SPIKE = 3
        
        # Setup collision handlers
        self.setup_collision_handlers()
        
        # Grid constants
        self.GRID_BLOCK_SIZE = 1.0
        
        # Cube properties
        self.cube_size = self.BLOCK_SIZE
        self.cube_body = None
        self.cube_shape = None
        
        # Physics (world moves, not cube)
        self.speed = 6.0  # pixels per frame
        self.jump_impulse = 700
        
        # Calculate auto-play lookahead
        self.lookahead_blocks = 4
        
        # Game state
        self.is_dead = False
        self.score = 0
        self.auto_play = True
        self.running = True
        self.rotation_angle = 0
        self.is_jumping = False
        self.is_grounded = False
        self.last_ground_contact = False
        self.world_x = 0
        
        # Action tracking for dataset collection
        self.action_taken = 0  # 0 = no-op, 1 = jump
        self.episode_id = 0  # Increments on each reset
        
        # Level objects (visual tracking)
        self.objects = []
        self.last_column = 0
        
        # Physics bodies tracking
        self.physics_bodies = []
        
        # Initialize game
        self.reset()
        
    def _load_vqvae(self, checkpoint_path=None):
        """Load VQ-VAE model from checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = Path(__file__).parent / "checkpoints" / "vqvae_best.pt"
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"VQ-VAE checkpoint not found at {checkpoint_path}\n"
                f"Train a model first using: py -m vqvae.train"
            )
        
        print(f"Loading VQ-VAE from {checkpoint_path}...")
        
        # Import VQ-VAE model
        from vqvae.model import VQVAE
        
        # Determine device
        self.vqvae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.vqvae_device, weights_only=False)
        config = ckpt.get("config", {})
        
        # Create model
        self.vqvae_model = VQVAE(
            latent_dim=config.get("latent_dim", 16),
            num_embeddings=config.get("num_embeddings", 1024),
            commitment_cost=config.get("commitment_cost", 0.25),
            ema_decay=config.get("ema_decay", 0.99),
        ).to(self.vqvae_device)
        
        self.vqvae_model.load_state_dict(ckpt["model_state_dict"])
        self.vqvae_model.eval()
        
        epoch = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", "?")
        print(f"Loaded VQ-VAE from epoch {epoch}, val_loss={val_loss}")
        print(f"Device: {self.vqvae_device}")
    
    def _apply_vqvae(self):
        """
        Read pixels from self.screen, pass through VQ-VAE, write back to self.screen.
        """
        # Step 1: Read pixels from screen as numpy array
        # pygame.surfarray.array3d returns (W, H, 3) = (840, 420, 3)
        pixels = pygame.surfarray.array3d(self.screen)  # (840, 420, 3) uint8

        # Step 2: Convert to PyTorch tensor [1, 3, 420, 840]
        # pixels is (W, H, 3), we need (B, C, H, W)
        tensor = torch.from_numpy(pixels).permute(2, 1, 0).unsqueeze(0).float()  # [1, 3, 420, 840]

        # Step 3: Resize to VQ-VAE input resolution [1, 3, 256, 512]
        tensor = F.interpolate(tensor, size=(256, 512), mode='bilinear', align_corners=False)

        # Step 4: Normalize to [-1, 1]
        tensor = tensor / 127.5 - 1.0
        tensor = tensor.to(self.vqvae_device)

        # Step 5: Pass through VQ-VAE (encode -> quantize -> decode)
        with torch.no_grad():
            reconstructed = self.vqvae_model(tensor)[0]  # [1, 3, 256, 512]

        # Step 6: Resize back to original resolution [1, 3, 420, 840]
        reconstructed = F.interpolate(reconstructed, size=(420, 840), mode='bicubic', align_corners=False)
        reconstructed = reconstructed.clamp(-1, 1)

        # Step 7: Denormalize to [0, 255] uint8
        reconstructed = ((reconstructed + 1.0) * 127.5).clamp(0, 255).byte()

        # Step 8: Convert back to numpy (W, H, 3) for pygame
        result = reconstructed[0].cpu().permute(2, 1, 0).numpy()  # (840, 420, 3) uint8

        # Step 9: Write pixels directly back to self.screen
        pygame.surfarray.blit_array(self.screen, result)
        
    def setup_collision_handlers(self):
        """Setup collision detection handlers"""
        # Player hitting spike - use begin for instant death detection
        self.space.on_collision(
            collision_type_a=self.COLLISION_TYPE_PLAYER,
            collision_type_b=self.COLLISION_TYPE_SPIKE,
            begin=self.on_player_spike_collision
        )
        
        # Player hitting block - use pre_solve for continuous contact detection
        self.space.on_collision(
            collision_type_a=self.COLLISION_TYPE_PLAYER,
            collision_type_b=self.COLLISION_TYPE_BLOCK,
            pre_solve=self.on_player_block_collision
        )
        
    def on_player_spike_collision(self, arbiter, space, data):
        """Called when player touches spike"""
        if not self.is_dead:
            self.is_dead = True
        return False  # Don't resolve physics (no bounce), just detect death
    
    def on_player_block_collision(self, arbiter, space, data):
        """Called when player collides with block (pre_solve: every frame during contact)"""
        if self.is_dead:
            return True
            
        if len(arbiter.contact_point_set.points) == 0:
            return True
            
        # Get collision normal (points in direction player should move to separate)
        normal = arbiter.contact_point_set.normal
        
        # Get player bounds
        half_cube = self.cube_size / 2
        player_bottom = self.cube_body.position.y - half_cube
        player_top = self.cube_body.position.y + half_cube
        
        # Get the block we're colliding with
        block_body = None
        for shape in arbiter.shapes:
            if shape.collision_type == self.COLLISION_TYPE_BLOCK:
                block_body = shape.body
                break
        
        if not block_body:
            return True
        
        # Get block bounds
        half_block = self.BLOCK_SIZE / 2
        block_bottom = block_body.position.y - half_block
        block_top = block_body.position.y + half_block
        
        # Position-based detection (more reliable than normal-only with rounded corners)
        
        # Check vertical overlap to distinguish wall hits from ground/seam contact
        overlap_top = min(player_top, block_top)
        overlap_bottom = max(player_bottom, block_bottom)
        vertical_overlap = max(0, overlap_top - overlap_bottom)
        
        # If player bottom is very close to block top, it's ground (even if hitting seam)
        # But only if we're not moving upward quickly (otherwise we'd stick to walls while jumping)
        if abs(player_bottom - block_top) < 8 and self.cube_body.velocity.y <= 100:
            self.last_ground_contact = True
            self.is_grounded = True
            self.is_jumping = False
            return True
        
        # Check for wall collision: large vertical overlap means player is stuck in wall
        if vertical_overlap > self.cube_size * 0.6:
            # Most of the player is overlapping the block vertically — real wall
            self.is_dead = True
            return False
        
        # For everything else (ceiling, small overlaps), allow physics
        return True
        
    def reset(self):
        """Reset game state"""
        # Remove old player body from physics space (if it exists)
        if self.cube_body is not None:
            self.space.remove(self.cube_body, self.cube_shape)
            self.cube_body = None
            self.cube_shape = None
        
        # Clear physics space
        for body in self.physics_bodies:
            try:
                self.space.remove(body[0], body[1])
            except:
                pass
        self.physics_bodies = []
        
        # Reset game state
        self.is_dead = False
        self.score = 0
        self.rotation_angle = 0
        self.is_jumping = False
        self.is_grounded = False
        self.last_ground_contact = False
        self.world_x = 0
        
        # Increment episode ID on each reset
        self.episode_id += 1
        
        # For mixed mode, randomly select behavior for this episode
        if self.play_mode == "mixed":
            self._current_play_mode = random.choice(["optimal", "random", "noisy"])
        else:
            self._current_play_mode = self.play_mode
        
        # Clear level
        self.objects = []
        self.last_column = 0
        
        # Create player cube
        self.create_player()
        
        # Generate initial level
        self.generate_initial_level()
        
    def create_player(self):
        """Create player physics body"""
        # Create dynamic body for player
        mass = 1
        # Infinite moment prevents the physics engine from rotating the body.
        # Visual rotation is handled separately in update(). Without this,
        # the body rotates when it catches on seams between adjacent ground
        # blocks, causing corners to protrude and clip into neighbors.
        moment = float('inf')
        self.cube_body = pymunk.Body(mass, moment)
        # Start on ground level, not in the air
        self.cube_body.position = (2 * self.BLOCK_SIZE + self.cube_size/2, 
                                   1 * self.BLOCK_SIZE + self.cube_size/2 + 2)  # Slightly above ground
        
        # Create box shape with rounded corners (radius) to glide over seams
        # between adjacent ground blocks instead of catching on them.
        # Inner polygon is shrunk so that inner + 2*radius = original size.
        # Small radius (0.5) is enough to prevent catching without sliding off edges.
        corner_radius = 0.5
        inner_size = self.cube_size - 2 * corner_radius
        self.cube_shape = pymunk.Poly.create_box(self.cube_body, 
                                                  (inner_size, inner_size),
                                                  radius=corner_radius)
        self.cube_shape.friction = 0.0
        self.cube_shape.elasticity = 0.0
        self.cube_shape.collision_type = self.COLLISION_TYPE_PLAYER
        
        self.space.add(self.cube_body, self.cube_shape)
        
    def create_block(self, grid_x, grid_y):
        """Create a static block in physics space"""
        world_x = grid_x * self.BLOCK_SIZE
        world_y = grid_y * self.BLOCK_SIZE
        
        # Create static body
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (world_x + self.BLOCK_SIZE/2, world_y + self.BLOCK_SIZE/2)
        
        # Create box shape
        shape = pymunk.Poly.create_box(body, (self.BLOCK_SIZE, self.BLOCK_SIZE))
        shape.friction = 0.0
        shape.elasticity = 0.0
        shape.collision_type = self.COLLISION_TYPE_BLOCK
        
        self.space.add(body, shape)
        self.physics_bodies.append((body, shape))
        
    def create_spike(self, grid_x, grid_y):
        """Create a spike (triangle) in physics space"""
        world_x = grid_x * self.BLOCK_SIZE
        world_y = grid_y * self.BLOCK_SIZE
        
        # Create static body
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (world_x + self.BLOCK_SIZE/2, world_y + self.BLOCK_SIZE/2)
        
        # Create triangle shape (vertices relative to body position)
        half = self.BLOCK_SIZE / 2
        vertices = [
            (-half, -half),   # bottom-left
            (half, -half),    # bottom-right
            (0, half)         # top
        ]
        shape = pymunk.Poly(body, vertices)
        shape.friction = 0.0
        shape.elasticity = 0.0
        shape.collision_type = self.COLLISION_TYPE_SPIKE
        
        self.space.add(body, shape)
        self.physics_bodies.append((body, shape))
        
    def generate_initial_level(self):
        """Generate starting safe zone"""
        for x in range(20):
            self.objects.append({'type': 'block', 'x': x, 'y': 0})
            self.create_block(x, 0)
        self.last_column = 20
        
        for _ in range(5):
            self.generate_next_chunk()
            
    def generate_next_chunk(self):
        """Generate next chunk using patterns"""
        patterns = ['flat', 'gap', 'spike', 'platform', 'stair_up', 'stair_down']
        
        # Select weights based on terrain_mode
        if self.terrain_mode == "obstacle_rich":
            weights = [0.15, 0.2, 0.3, 0.15, 0.15, 0.05]
        elif self.terrain_mode == "balanced":
            weights = [0.25, 0.18, 0.25, 0.12, 0.12, 0.08]
        else:  # "default"
            weights = [0.4, 0.15, 0.2, 0.1, 0.1, 0.05]
        
        pattern = random.choices(patterns, weights=weights)[0]
        
        # Get current ground height
        last_height = 0
        if self.objects:
            last_objs = [obj for obj in self.objects if obj['x'] == self.last_column - 1 and obj['type'] == 'block']
            if last_objs:
                last_height = max(obj['y'] for obj in last_objs)
        
        if pattern == 'flat':
            length = random.randint(3, 6)
            for i in range(length):
                for y in range(last_height + 1):
                    self.objects.append({'type': 'block', 'x': self.last_column + i, 'y': y})
                    self.create_block(self.last_column + i, y)
            self.last_column += length
            
        elif pattern == 'gap':
            gap_size = random.randint(2, 3)
            self.last_column += gap_size
            for i in range(3):
                for y in range(last_height + 1):
                    self.objects.append({'type': 'block', 'x': self.last_column + i, 'y': y})
                    self.create_block(self.last_column + i, y)
            self.last_column += 3
            
        elif pattern == 'spike':
            length = random.randint(4, 6)
            for i in range(length):
                for y in range(last_height + 1):
                    self.objects.append({'type': 'block', 'x': self.last_column + i, 'y': y})
                    self.create_block(self.last_column + i, y)
            spike_x = self.last_column + length // 2
            self.objects.append({'type': 'spike', 'x': spike_x, 'y': last_height + 1})
            self.create_spike(spike_x, last_height + 1)
            self.last_column += length
            
        elif pattern == 'platform':
            length = random.randint(4, 6)
            for i in range(length):
                for y in range(last_height + 1):
                    self.objects.append({'type': 'block', 'x': self.last_column + i, 'y': y})
                    self.create_block(self.last_column + i, y)
            
            plat_height = last_height + 3
            if plat_height <= 5:
                for i in range(1, length - 1):
                    self.objects.append({'type': 'block', 'x': self.last_column + i, 'y': plat_height})
                    self.create_block(self.last_column + i, plat_height)
            
            self.last_column += length
            
        elif pattern == 'stair_up':
            if last_height < 4:
                for i in range(3):
                    for y in range(last_height + 1):
                        self.objects.append({'type': 'block', 'x': self.last_column + i, 'y': y})
                        self.create_block(self.last_column + i, y)
                self.last_column += 3
                
                for y in range(last_height + 2):
                    self.objects.append({'type': 'block', 'x': self.last_column, 'y': y})
                    self.create_block(self.last_column, y)
                self.last_column += 1
                
                for i in range(3):
                    for y in range(last_height + 2):
                        self.objects.append({'type': 'block', 'x': self.last_column + i, 'y': y})
                        self.create_block(self.last_column + i, y)
                self.last_column += 3
            else:
                for i in range(4):
                    for y in range(last_height + 1):
                        self.objects.append({'type': 'block', 'x': self.last_column + i, 'y': y})
                        self.create_block(self.last_column + i, y)
                self.last_column += 4
                
        elif pattern == 'stair_down':
            if last_height > 0:
                for i in range(2):
                    for y in range(last_height + 1):
                        self.objects.append({'type': 'block', 'x': self.last_column + i, 'y': y})
                        self.create_block(self.last_column + i, y)
                self.last_column += 2
                
                for i in range(3):
                    for y in range(last_height):
                        self.objects.append({'type': 'block', 'x': self.last_column + i, 'y': y})
                        self.create_block(self.last_column + i, y)
                self.last_column += 3
            else:
                for i in range(4):
                    self.objects.append({'type': 'block', 'x': self.last_column + i, 'y': 0})
                    self.create_block(self.last_column + i, 0)
                self.last_column += 4
    
    def simulate_jump_trajectory(self, start_x, start_y, start_vy, duration_frames=60):
        """Simulate jump trajectory and return list of (x, y) positions"""
        dt = 1.0 / self.FPS
        vx = self.speed * self.FPS
        vy = start_vy
        gravity = self.space.gravity[1]
        
        trajectory = []
        x, y = start_x, start_y
        
        for _ in range(duration_frames):
            x += vx * dt
            y += vy * dt
            vy += gravity * dt
            trajectory.append((x, y))
            
            # Stop if we've landed back at ground level or below
            if y <= start_y:
                break
        
        return trajectory
    
    def check_collision_at_point(self, x, y):
        """Check if player at position (x, y) would collide with spike or wall"""
        half_cube = self.cube_size / 2
        player_left = x - half_cube
        player_right = x + half_cube
        player_bottom = y - half_cube
        player_top = y + half_cube
        
        for obj in self.objects:
            obj_world_x = obj['x'] * self.BLOCK_SIZE
            obj_world_y = obj['y'] * self.BLOCK_SIZE
            
            obj_left = obj_world_x
            obj_right = obj_world_x + self.BLOCK_SIZE
            obj_bottom = obj_world_y
            obj_top = obj_world_y + self.BLOCK_SIZE
            
            # Check for overlap
            if (player_right > obj_left and player_left < obj_right and
                player_top > obj_bottom and player_bottom < obj_top):
                
                if obj['type'] == 'spike':
                    return 'spike'
                elif obj['type'] == 'block':
                    # Only count as wall collision if significantly overlapping (not just touching ground)
                    vertical_overlap = min(player_top, obj_top) - max(player_bottom, obj_bottom)
                    if vertical_overlap > self.cube_size * 0.3 and player_bottom > obj_bottom + 5:
                        return 'wall'
        
        return None
    
    def find_next_obstacle(self, player_x, player_y, ground_level):
        """Find the nearest obstacle that requires jumping"""
        check_distance = self.lookahead_blocks * self.BLOCK_SIZE
        obstacle_x = None
        obstacle_type = None
        obstacle_data = {}
        
        player_grid_x = int(player_x / self.BLOCK_SIZE)
        
        for dx in range(10, int(check_distance), 5):
            check_x = player_x + dx
            check_grid_x = int(check_x / self.BLOCK_SIZE)
            
            # Check for spike at ground level
            for obj in self.objects:
                if obj['x'] == check_grid_x and obj['type'] == 'spike':
                    if ground_level <= obj['y'] <= ground_level + 2:
                        if obstacle_x is None or obj['x'] * self.BLOCK_SIZE < obstacle_x:
                            obstacle_x = obj['x'] * self.BLOCK_SIZE
                            obstacle_type = 'spike'
            
            # Check for gap (no ground)
            has_ground = False
            for obj in self.objects:
                if obj['x'] == check_grid_x and obj['type'] == 'block' and obj['y'] <= ground_level:
                    has_ground = True
                    break
            
            if not has_ground:
                if obstacle_x is None:
                    # Found start of gap - find where gap actually starts (last ground block)
                    gap_start_grid = check_grid_x
                    # Look back to find the last ground block
                    for back_x in range(check_grid_x - 1, player_grid_x - 1, -1):
                        has_ground_here = False
                        for obj in self.objects:
                            if obj['x'] == back_x and obj['type'] == 'block' and obj['y'] <= ground_level:
                                has_ground_here = True
                                break
                        if has_ground_here:
                            gap_start_grid = back_x + 1
                            break
                    
                    # Find gap end
                    gap_end_grid = gap_start_grid
                    for forward_x in range(gap_start_grid, int((player_x + check_distance) / self.BLOCK_SIZE)):
                        has_ground_here = False
                        for obj in self.objects:
                            if obj['x'] == forward_x and obj['type'] == 'block' and obj['y'] <= ground_level:
                                has_ground_here = True
                                break
                        if has_ground_here:
                            gap_end_grid = forward_x
                            break
                        gap_end_grid = forward_x
                    
                    # Jump point should be at the edge before the gap
                    obstacle_x = (gap_start_grid - 0.5) * self.BLOCK_SIZE
                    obstacle_type = 'gap'
                    obstacle_data['gap_width'] = gap_end_grid - gap_start_grid + 1
                break
            
            # Check for wall
            for obj in self.objects:
                if obj['x'] == check_grid_x and obj['type'] == 'block':
                    if obj['y'] == ground_level + 1 or obj['y'] == ground_level + 2:
                        if obstacle_x is None or obj['x'] * self.BLOCK_SIZE < obstacle_x:
                            obstacle_x = obj['x'] * self.BLOCK_SIZE
                            obstacle_type = 'wall'
        
        return obstacle_x, obstacle_type, obstacle_data
    
    def auto_play_logic(self):
        """AI auto-play with trajectory simulation"""
        if self.auto_play and not self.is_dead and not self.is_jumping and self.is_grounded:
            # Use current play mode (respects mixed mode episode selection)
            current_mode = getattr(self, '_current_play_mode', self.play_mode)
            
            # Random mode: jump randomly on flat ground
            if current_mode == "random":
                if random.random() < self.random_jump_prob:
                    self.jump()
                    return
            
            # Get player current state
            player_x = self.cube_body.position.x
            player_y = self.cube_body.position.y
            ground_level = int(player_y / self.BLOCK_SIZE) - 1
            
            # Find the next obstacle
            obstacle_x, obstacle_type, obstacle_data = self.find_next_obstacle(player_x, player_y, ground_level)
            
            if obstacle_x is None:
                return  # No obstacle ahead, don't jump
            
            # Calculate distance to obstacle
            distance_to_obstacle = obstacle_x - player_x
            
            # Noisy mode: add jitter and deliberate misses
            if current_mode == "noisy":
                # Deliberately skip jump with miss_prob
                if random.random() < self.miss_prob:
                    return
                # Add timing jitter
                distance_to_obstacle += random.uniform(-0.5, 0.5) * self.BLOCK_SIZE
            
            # For gaps, we want to jump very close to the edge
            if obstacle_type == 'gap':
                # Jump when we're very close to the edge (within 0.6 blocks)
                if distance_to_obstacle < self.BLOCK_SIZE * 0.6:
                    self.jump()
                return
            
            # For spikes and walls, use trajectory simulation
            # Simulate jump from current position
            initial_vy = self.jump_impulse / self.cube_body.mass
            trajectory = self.simulate_jump_trajectory(player_x, player_y, initial_vy)
            
            # Check if we'll clear the obstacle with this jump
            will_clear = True
            for pos_x, pos_y in trajectory:
                if pos_x >= obstacle_x - self.cube_size/2 and pos_x <= obstacle_x + self.BLOCK_SIZE + self.cube_size/2:
                    # We're in the obstacle zone, check height
                    collision = self.check_collision_at_point(pos_x, pos_y)
                    if collision in ['spike', 'wall']:
                        will_clear = False
                        break
            
            # Decision logic for spikes/walls: jump at the optimal moment
            # We want to jump late enough to not waste height, but early enough to clear
            if will_clear and distance_to_obstacle < self.BLOCK_SIZE * 2.5:
                self.jump()
            elif not will_clear and distance_to_obstacle < self.BLOCK_SIZE * 1.5:
                # If we can't clear from current position and we're getting close, jump anyway
                # This handles cases where we need to jump earlier
                self.jump()
                
    def jump(self):
        """Make the cube jump"""
        if self.is_grounded:
            # Apply jump impulse
            self.cube_body.apply_impulse_at_local_point((0, self.jump_impulse))
            self.is_jumping = True
            self.is_grounded = False
            self.last_ground_contact = False
            # Set action_taken flag
            self.action_taken = 1
            # Debug
            # print(f"Jump! vy={self.cube_body.velocity.y:.1f}")
        else:
            # Debug - why can't we jump?
            pass
            # print(f"Can't jump: grounded={self.is_grounded}, jumping={self.is_jumping}, vy={self.cube_body.velocity.y:.1f}")
                
    def update(self):
        """Update game logic"""
        # Reset action_taken at the start of each frame
        self.action_taken = 0
        
        if not self.is_dead:
            # Auto-play logic
            self.auto_play_logic()
            
            # Keep cube moving right at constant speed
            self.cube_body.velocity = (self.speed * self.FPS, self.cube_body.velocity.y)
            
            # Track world position
            self.world_x = self.cube_body.position.x - 2 * self.BLOCK_SIZE
            self.score = int(self.world_x / 10)
            
            # Reset ground contact flag before physics step
            self.last_ground_contact = False
            
            # Step physics (collision callbacks will set last_ground_contact if touching ground)
            dt = 1.0 / self.FPS
            self.space.step(dt)
            
            # Prevent any physics rotation — visual rotation is handled separately
            self.cube_body.angle = 0
            self.cube_body.angular_velocity = 0
            
            # Update grounded state based on collision detection
            if self.last_ground_contact:
                # Collision callback confirmed ground contact this frame
                self.is_grounded = True
                self.is_jumping = False
            elif self.is_jumping or self.cube_body.velocity.y < -50:
                # Actively jumping (upward phase) or clearly falling
                self.is_grounded = False
            # Otherwise keep previous grounded state.  This gives a natural
            # 1-2 frame tolerance when sliding across adjacent ground tiles
            # whose contact callbacks may not overlap perfectly.
            
            # Rotation logic
            if not self.is_grounded:
                self.rotation_angle += 6
                if self.rotation_angle >= 360:
                    self.rotation_angle -= 360
            else:
                target_angle = round(self.rotation_angle / 90) * 90
                if abs(self.rotation_angle - target_angle) > 1:
                    self.rotation_angle += (target_angle - self.rotation_angle) * 0.3
                else:
                    self.rotation_angle = target_angle
            
            # Death by falling
            if self.cube_body.position.y < -self.BLOCK_SIZE:
                self.is_dead = True
            
            # Generate new chunks ahead of player
            rightmost_visible_x = (self.cube_body.position.x + self.SCREEN_WIDTH) / self.BLOCK_SIZE
            while self.last_column < rightmost_visible_x:
                self.generate_next_chunk()
            
            # Remove old objects (behind the camera view)
            to_remove = []
            for i, (body, shape) in enumerate(self.physics_bodies):
                if body.position.x < self.cube_body.position.x - 5 * self.BLOCK_SIZE:
                    to_remove.append(i)
            
            for i in reversed(to_remove):
                body, shape = self.physics_bodies[i]
                self.space.remove(body, shape)
                del self.physics_bodies[i]
            
            # Clean up objects list
            min_visible_x = (self.cube_body.position.x - 5 * self.BLOCK_SIZE) / self.BLOCK_SIZE
            self.objects = [obj for obj in self.objects if obj['x'] > min_visible_x]
                          
    def draw(self):
        """Render everything"""
        self.screen.fill(self.SKY_BLUE)
        
        # Camera follows cube in both X and Y - cube position in world space
        camera_x = self.cube_body.position.x - 2 * self.BLOCK_SIZE
        camera_y = self.cube_body.position.y - self.SCREEN_HEIGHT / 2
        
        # Draw objects (blocks and spikes from visual tracking)
        for obj in self.objects:
            world_x = obj['x'] * self.BLOCK_SIZE
            world_y = obj['y'] * self.BLOCK_SIZE
            screen_x = world_x - camera_x
            screen_y = world_y - camera_y
            
            # Only draw if visible
            if -self.BLOCK_SIZE <= screen_x <= self.SCREEN_WIDTH + self.BLOCK_SIZE:
                # Convert to pygame coordinates (flip Y)
                pygame_y = self.SCREEN_HEIGHT - screen_y - self.BLOCK_SIZE
                
                if obj['type'] == 'block':
                    rect = pygame.Rect(screen_x, pygame_y, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(self.screen, self.BLOCK_GREEN, rect)
                    pygame.draw.rect(self.screen, self.BLOCK_BORDER, rect, 2)
                    
                elif obj['type'] == 'spike':
                    points = [
                        (screen_x, pygame_y + self.BLOCK_SIZE),
                        (screen_x + self.BLOCK_SIZE, pygame_y + self.BLOCK_SIZE),
                        (screen_x + self.BLOCK_SIZE / 2, pygame_y)
                    ]
                    pygame.draw.polygon(self.screen, self.SPIKE_RED, points)
                    pygame.draw.polygon(self.screen, self.SPIKE_BORDER, points, 2)
        
        # Draw cube at fixed screen position (centered vertically)
        cube_screen_x = 2 * self.BLOCK_SIZE
        cube_screen_y = self.SCREEN_HEIGHT / 2 - self.cube_size/2
        
        # Create rotated cube surface
        cube_surf = pygame.Surface((self.cube_size, self.cube_size), pygame.SRCALPHA)
        pygame.draw.rect(cube_surf, self.CUBE_RED, (0, 0, self.cube_size, self.cube_size))
        pygame.draw.rect(cube_surf, self.CUBE_BORDER, (0, 0, self.cube_size, self.cube_size), 3)
        
        # Rotate and draw
        rotated_cube = pygame.transform.rotate(cube_surf, self.rotation_angle)
        cube_rect = rotated_cube.get_rect(center=(cube_screen_x + self.cube_size/2, 
                                                   cube_screen_y + self.cube_size/2))
        self.screen.blit(rotated_cube, cube_rect)
        
        # Draw UI
        if not self.collecting_data:
            if self.is_dead:
                text = self.font.render(f'DEAD - Score: {self.score} | Press R to Restart', 
                                       True, self.DEAD_COLOR)
            else:
                text = self.font.render(f'Score: {self.score}', True, self.TEXT_COLOR)
            
            self.screen.blit(text, (10, 10))
            
            # Draw controls
            mode_text = "AUTO" if self.auto_play else "MANUAL"
            controls = self.small_font.render(f'SPACE: Jump | A: {mode_text} | R: Reset | ESC: Quit', 
                                             True, self.TEXT_COLOR)
            self.screen.blit(controls, (10, self.SCREEN_HEIGHT - 30))
            
            # Debug info
            debug_text = self.small_font.render(
                f'Ground: {self.is_grounded} | Jump: {self.is_jumping} | VelY: {self.cube_body.velocity.y:.0f}', 
                True, self.TEXT_COLOR)
            self.screen.blit(debug_text, (10, 45))
        
        if not self.headless:
            # Apply VQ-VAE reconstruction: overwrite self.screen pixels in-place
            if self.use_vqvae and self.vqvae_model is not None:
                self._apply_vqvae()
            
            pygame.display.flip()
        
    def handle_events(self):
        """Handle input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not self.is_dead:
                    self.jump()
                    
                elif event.key == pygame.K_r:
                    self.reset()
                    
                elif event.key == pygame.K_a:
                    self.auto_play = not self.auto_play
                    print(f"Auto-play: {self.auto_play}")
                    
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                    
    def get_frame(self):
        """Get current frame as numpy array for dataset collection"""
        return pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])
    
    def get_state(self):
        """Return simplified grid state for AI training"""
        grid_width = 20
        grid_height = 10
        state = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
        
        # Player position in grid (fixed at x=2 on screen)
        player_grid_x = 2
        player_grid_y = int(self.cube_body.position.y / self.BLOCK_SIZE)
        
        # Objects relative to player position
        start_x = int((self.cube_body.position.x - 2 * self.BLOCK_SIZE) / self.BLOCK_SIZE)
        for obj in self.objects:
            rel_x = obj['x'] - start_x
            if 0 <= rel_x < grid_width and 0 <= obj['y'] < grid_height:
                val = 1 if obj['type'] == 'block' else 2
                state[obj['y']][rel_x] = val
        
        if 0 <= player_grid_y < grid_height and 0 <= player_grid_x < grid_width:
            state[player_grid_y][player_grid_x] = 3
            
        return state
        
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.FPS)
        
        pygame.quit()


if __name__ == '__main__':
    import sys
    
    # Check for --vqvae flag
    use_vqvae = '--vqvae' in sys.argv
    vqvae_checkpoint = None
    
    # Check for custom checkpoint path
    if '--checkpoint' in sys.argv:
        idx = sys.argv.index('--checkpoint')
        if idx + 1 < len(sys.argv):
            vqvae_checkpoint = sys.argv[idx + 1]
    
    game = Game(use_vqvae=use_vqvae, vqvae_checkpoint=vqvae_checkpoint)
    game.run()
