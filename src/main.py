import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("Starting AdvisorAI pipeline...")
    
    # Step 1: Fetch data
    print("Step 1: Fetching data...")
    try:
        from src.data_fetcher.main import main as fetch_data_main
        fetch_data_main()
        print("Data fetching completed.")
    except Exception as e:
        print(f"Data fetching failed: {e}")
        print("Continuing with existing data...")
    
    # Step 2: Process data
    print("Step 2: Processing data...")
    try:
        from src.data_engineering.main import main as process_data_main
        process_data_main()
        print("Data processing completed.")
    except Exception as e:
        print(f"Data processing failed: {e}")
    
    # Future: Call models, prediction, etc. when implemented
    print("Pipeline complete.")

if __name__ == "__main__":
    main()