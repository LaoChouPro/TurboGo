import torch
import numpy as np
import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.board import GoBoard, BLACK, WHITE, SIZE
from src_large.model import TinyGoNet

def load_model(model_path, device='cpu'):
    model = TinyGoNet() # Uses large defaults (10 blocks, 128 channels)
    # Use map_location to handle loading on different devices
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
        
    model.to(device)
    model.eval()
    return model

def predict_move(model, board, color, device='cpu'):
    # Prepare input
    # Channel 0: Current Player
    # Channel 1: Opponent
    # Channel 2: Ones
    
    features = np.zeros((3, SIZE, SIZE), dtype=np.float32)
    my_stones = (board.board == color)
    opp_stones = (board.board == (3 - color))
    
    features[0] = my_stones.astype(np.float32)
    features[1] = opp_stones.astype(np.float32)
    features[2] = 1.0
    
    input_tensor = torch.from_numpy(features).unsqueeze(0).to(device) # (1, 3, 19, 19)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        
    # Get Top 5 moves
    top_probs, top_indices = torch.topk(probs, 5)
    
    moves = []
    for i in range(5):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        r, c = idx // SIZE, idx % SIZE
        moves.append(((r, c), prob))
        
    return moves

def evaluate_on_file(model_path, data_file):
    # Check device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    print(f"Loading large model from {model_path} on {device}...")
    try:
        model = load_model(model_path, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Reading game from {data_file}...")
    with open(data_file, 'r') as f:
        try:
            moves = json.load(f)
        except:
            print("Invalid JSON format.")
            return
        
    board = GoBoard()
    
    print(f"Simulating game with {len(moves)} moves...")
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for i, move in enumerate(moves):
        # move format: {"B": [x, y]}
        if not isinstance(move, dict) or len(move) == 0:
            continue
            
        color_str = list(move.keys())[0]
        color = BLACK if color_str == 'B' else WHITE
        coords = move[color_str]
        
        if not isinstance(coords, list) or len(coords) != 2:
            continue
            
        gt_r, gt_c = coords[0] - 1, coords[1] - 1
        
        # Validate
        if not (0 <= gt_r < SIZE and 0 <= gt_c < SIZE):
            continue
        
        # Check turn
        if board.next_player != color:
            board.next_player = color
            
        # Predict
        top_moves = predict_move(model, board, color, device)
        
        # Check if GT is in top moves
        top1_move = top_moves[0][0]
        
        is_top1 = (top1_move == (gt_r, gt_c))
        is_top5 = any(m[0] == (gt_r, gt_c) for m in top_moves)
        
        if is_top1:
            correct_top1 += 1
        if is_top5:
            correct_top5 += 1
            
        total += 1
        
        # Play the move
        board.play(gt_r, gt_c, color)
        
        if i % 20 == 0:
            # Print move info
            top1_str = f"({top1_move[0]},{top1_move[1]})"
            gt_str = f"({gt_r},{gt_c})"
            print(f"Move {i+1}: GT={gt_str}, Pred={top1_str} (P={top_moves[0][1]:.2f}) {'✓' if is_top1 else '✗'}")
            
    if total > 0:
        print(f"Game Finished. Accuracy Top-1: {correct_top1/total*100:.2f}%, Top-5: {correct_top5/total*100:.2f}%")
    else:
        print("No valid moves found.")

if __name__ == "__main__":
    # Use best_model_large.pth if exists
    model_path = "best_model_large.pth"
    data_file = "example_data.data"
    
    if os.path.exists(model_path) and os.path.exists(data_file):
        evaluate_on_file(model_path, data_file)
    else:
        print(f"Please train the model first (expected {model_path}).")
