"""Bootstrap AST weights and verify SHA-256."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Preproccesing.download_checkpoints import download_ast_weights

if __name__ == "__main__":
    download_ast_weights()
