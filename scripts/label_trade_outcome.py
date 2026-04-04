import json
from pathlib import Path
import readline  # for better input handling

def label_artifacts_interactively():
    artifacts_dir = Path("discovery_outputs")
    
    # Find all recommendation files
    for artifact in artifacts_dir.rglob("*.json"):  # adjust extension as needed
        with open(artifact) as f:
            data = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"Trade: {data.get('ticker')} on {data.get('timestamp')}")
        print(f"Recommendation: {data.get('portfolio', {}).get('action')}")
        print(f"Confidence: {data.get('analysis', {}).get('confidence')}")
        print(f"Signal Scores: T={data.get('analysis',{}).get('technical_score')}, "
              f"S={data.get('analysis',{}).get('sentiment_score')}, "
              f"F={data.get('analysis',{}).get('fundamental_score')}")
        
        print("\nWhat happened?")
        print("1 - Profitable trade")
        print("2 - Lost money")
        print("3 - Trade not executed (HITL rejected)")
        print("4 - Still open")
        
        choice = input("Your choice (1-4): ")
        
        if choice == "1":
            pnl = float(input("PnL %: "))
            data['actual_outcome'] = 'PROFIT'
            data['actual_pnl'] = pnl
        elif choice == "2":
            pnl = float(input("PnL % (negative): "))
            data['actual_outcome'] = 'LOSS'
            data['actual_pnl'] = pnl
        elif choice == "3":
            data['actual_outcome'] = 'REJECTED'
            data['actual_pnl'] = 0
        elif choice == "4":
            data['actual_outcome'] = 'OPEN'
        
        # Save back with outcomes
        with open(artifact, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Labeled {artifact.name}")

if __name__ == "__main__":
    label_artifacts_interactively()