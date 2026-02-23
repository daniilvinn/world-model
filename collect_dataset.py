"""
Example script for collecting frame pairs (x, x+1) for world model training
"""
import pygame
import numpy as np
from game import Game
import os
from datetime import datetime
import traceback
import time

def collect_dataset(num_frames=1000, save_dir='dataset', play_mode='optimal', 
                   terrain_mode='default', random_jump_prob=0.15, miss_prob=0.10):
    """
    Collect frame pairs for world model training
    
    Args:
        num_frames: Number of frame pairs to collect
        save_dir: Directory to save the dataset
        play_mode: Behavior mode ('optimal', 'random', 'noisy', 'mixed')
        terrain_mode: Terrain generation mode ('default', 'obstacle_rich', 'balanced')
        random_jump_prob: Probability of random jumps (for random mode)
        miss_prob: Probability of missing jumps (for noisy mode)
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(save_dir, f'session_{timestamp}')
    os.makedirs(session_dir, exist_ok=True)
    
    # Initialize game in headless mode with collection flags
    game = Game(headless=True, play_mode=play_mode, terrain_mode=terrain_mode,
                collecting_data=True, random_jump_prob=random_jump_prob, 
                miss_prob=miss_prob)
    game.auto_play = True  # Enable auto-play
    
    frames_collected = 0
    prev_frame = None
    prev_state = None
    prev_action = None
    prev_episode_id = None
    death_frame_counter = 0  # Track frames collected during death
    
    print(f"Collecting {num_frames} frame pairs (headless mode, max speed)...")
    print(f"Play mode: {play_mode}, Terrain mode: {terrain_mode}")
    print(f"Saving to: {session_dir}")
    
    start_time = time.time()
    
    try:
        while game.running and frames_collected < num_frames:
            # Handle events (minimal in headless mode)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.running = False
            
            # Get state before update
            state_before = game.get_state()
            
            # Update game with fixed timestep (16.6ms physics)
            # but don't wait - collect as fast as possible
            game.update()
            game.draw()  # Still draw to surface for get_frame()
            
            # Get current frame and action
            current_frame = game.get_frame()
            action = game.action_taken  # 0 or 1
            episode_id = game.episode_id
            
            # Handle death frames: collect up to 5 frames after death
            if game.is_dead:
                death_frame_counter += 1
                
                # Save death frames (up to 5)
                if prev_frame is not None and death_frame_counter <= 5:
                    pair_path = os.path.join(session_dir, f'pair_{frames_collected:06d}.npz')
                    np.savez_compressed(
                        pair_path,
                        frame_t0=prev_frame,
                        frame_t1=current_frame,
                        action=prev_action,
                        episode_id=prev_episode_id,
                        state_t0=prev_state,
                        state_t1=state_before,
                        score=game.score,
                        is_jumping=game.is_jumping,
                        is_grounded=game.is_grounded,
                        is_dead=True  # Tag death frames
                    )
                    frames_collected += 1
                    
                    if frames_collected % 100 == 0:
                        elapsed = time.time() - start_time
                        fps = frames_collected / elapsed if elapsed > 0 else 0
                        print(f"Collected {frames_collected}/{num_frames} pairs (Score: {game.score}, {fps:.1f} pairs/sec)")
                
                # Reset after collecting enough death frames
                if death_frame_counter >= 5:
                    print(f"Game over at score {game.score}, collected {death_frame_counter} death frames, auto-restarting...")
                    game.reset()
                    prev_frame = None
                    prev_state = None
                    prev_action = None
                    prev_episode_id = None
                    death_frame_counter = 0
                else:
                    # Continue collecting death frames
                    prev_frame = current_frame.copy()
                    prev_state = state_before
                    prev_action = action
                    prev_episode_id = episode_id
            else:
                # Normal frame collection (not dead)
                death_frame_counter = 0
                
                # Save frame pair (t-1, t)
                if prev_frame is not None:
                    pair_path = os.path.join(session_dir, f'pair_{frames_collected:06d}.npz')
                    np.savez_compressed(
                        pair_path,
                        frame_t0=prev_frame,
                        frame_t1=current_frame,
                        action=prev_action,
                        episode_id=prev_episode_id,
                        state_t0=prev_state,
                        state_t1=state_before,
                        score=game.score,
                        is_jumping=game.is_jumping,
                        is_grounded=game.is_grounded,
                        is_dead=False
                    )
                    
                    frames_collected += 1
                    
                    if frames_collected % 100 == 0:
                        elapsed = time.time() - start_time
                        fps = frames_collected / elapsed if elapsed > 0 else 0
                        print(f"Collected {frames_collected}/{num_frames} pairs (Score: {game.score}, {fps:.1f} pairs/sec)")
                
                prev_frame = current_frame.copy()
                prev_state = state_before
                prev_action = action
                prev_episode_id = episode_id
            
            # No clock.tick() - run as fast as possible
            # Physics still uses fixed 16.6ms timestep in game.update()
    
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        traceback.print_exc()
        print("Attempting to restart collection...")
        # Auto-restart will be handled by outer loop in main
        raise
    finally:
        elapsed = time.time() - start_time
        print(f"\nDataset collection session ended!")
        print(f"Total pairs collected: {frames_collected}")
        print(f"Time elapsed: {elapsed:.2f}s")
        if frames_collected > 0:
            print(f"Average speed: {frames_collected / elapsed:.1f} pairs/sec")
        print(f"Saved to: {session_dir}")
        
        pygame.quit()

def load_frame_pair(filepath):
    """
    Load a frame pair from disk
    
    Returns:
        dict with keys: frame_t0, frame_t1, action, episode_id, state_t0, state_t1, score, is_jumping, is_grounded
    """
    data = np.load(filepath)
    return {
        'frame_t0': data['frame_t0'],
        'frame_t1': data['frame_t1'],
        'action': data['action'],
        'episode_id': data['episode_id'],
        'state_t0': data['state_t0'],
        'state_t1': data['state_t1'],
        'score': data['score'],
        'is_jumping': data['is_jumping'],
        'is_grounded': data['is_grounded']
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Collection Script')
    parser.add_argument('--num_frames', type=int, default=10000,
                        help='Number of frame pairs to collect')
    parser.add_argument('--play_mode', type=str, default='mixed',
                        choices=['optimal', 'random', 'noisy', 'mixed'],
                        help='Behavior mode for auto-play')
    parser.add_argument('--terrain_mode', type=str, default='balanced',
                        choices=['default', 'obstacle_rich', 'balanced'],
                        help='Terrain generation mode')
    parser.add_argument('--random_jump_prob', type=float, default=0.15,
                        help='Probability of random jumps in random mode')
    parser.add_argument('--miss_prob', type=float, default=0.10,
                        help='Probability of missing jumps in noisy mode')
    parser.add_argument('--phased', action='store_true',
                        help='Run phased collection (40%% optimal/default, 30%% random/balanced, 30%% noisy/obstacle_rich)')
    
    args = parser.parse_args()
    
    if args.phased:
        # Phased collection for diverse dataset
        total_frames = args.num_frames
        phase_configs = [
            {
                'name': 'Phase A (Optimal/Default)',
                'frames': int(total_frames * 0.4),
                'play_mode': 'optimal',
                'terrain_mode': 'default'
            },
            {
                'name': 'Phase B (Random/Balanced)',
                'frames': int(total_frames * 0.3),
                'play_mode': 'random',
                'terrain_mode': 'balanced'
            },
            {
                'name': 'Phase C (Noisy/Obstacle-Rich)',
                'frames': int(total_frames * 0.3),
                'play_mode': 'noisy',
                'terrain_mode': 'obstacle_rich'
            }
        ]
        
        print("=" * 60)
        print("Phased Dataset Collection")
        print("=" * 60)
        print(f"Total target: {total_frames} frame pairs")
        for config in phase_configs:
            print(f"  {config['name']}: {config['frames']} frames")
        print("=" * 60)
        print()
        
        max_retries = 10
        for phase_config in phase_configs:
            print(f"\nStarting {phase_config['name']}...")
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    collect_dataset(
                        num_frames=phase_config['frames'],
                        play_mode=phase_config['play_mode'],
                        terrain_mode=phase_config['terrain_mode'],
                        random_jump_prob=args.random_jump_prob,
                        miss_prob=args.miss_prob
                    )
                    print(f"\n{phase_config['name']} completed successfully!")
                    break
                except KeyboardInterrupt:
                    print("\nStopped by user")
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"\nAttempt {retry_count}/{max_retries} failed: {e}")
                    if retry_count < max_retries:
                        print(f"Restarting in 2 seconds...")
                        time.sleep(2)
                    else:
                        print(f"Max retries ({max_retries}) reached for this phase.")
                        break
        
        print("\n" + "=" * 60)
        print("Phased collection complete!")
        print("=" * 60)
    else:
        # Single collection session
        max_retries = 10
        retry_count = 0
        
        print("=" * 60)
        print("Dataset Collection Script")
        print("=" * 60)
        print(f"Target: {args.num_frames} frame pairs")
        print(f"Play mode: {args.play_mode}")
        print(f"Terrain mode: {args.terrain_mode}")
        print("Features:")
        print("  - Auto-restart on failure")
        print("  - Headless mode (no preview)")
        print("  - Fixed 16.6ms physics timestep")
        print("  - Maximum collection speed (no FPS limit)")
        print("  - Death frame collection (3-5 frames post-death)")
        print("=" * 60)
        print()
        
        while retry_count < max_retries:
            try:
                collect_dataset(
                    num_frames=args.num_frames,
                    play_mode=args.play_mode,
                    terrain_mode=args.terrain_mode,
                    random_jump_prob=args.random_jump_prob,
                    miss_prob=args.miss_prob
                )
                # If we get here, collection completed successfully
                print("\n" + "=" * 60)
                print("Collection completed successfully!")
                print("=" * 60)
                break
            except KeyboardInterrupt:
                print("\nStopped by user")
                break
            except Exception as e:
                retry_count += 1
                print(f"\nAttempt {retry_count}/{max_retries} failed: {e}")
                if retry_count < max_retries:
                    print(f"Restarting in 2 seconds...")
                    time.sleep(2)
                else:
                    print(f"Max retries ({max_retries}) reached. Exiting.")
                    break
    
    # Example: Load and inspect a frame pair
    # data = load_frame_pair('dataset/session_XXXXXXXX_XXXXXX/pair_000000.npz')
    # print(f"Frame shape: {data['frame_t0'].shape}")
    # print(f"State shape: {data['state_t0'].shape}")
