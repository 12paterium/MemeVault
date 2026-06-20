import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meme_vault import Metadata, load_metadata

def test():
    entries = load_metadata()
    print(f"Loaded {len(entries)} entries")
    for e in entries:
        print(f"  [{e.id}] {e.text}")

if __name__ == "__main__":
    test()
