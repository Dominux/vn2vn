from pathlib import Path
from uuid import UUID

from vn_extractor import VNExtractor


base_path = Path("/tmp/vn2vn")
id = UUID("6d0d5ffe-cccc-4268-8858-6518c0fa85ca")

def main():
    vn_extractor = VNExtractor(base_path)
    vn_extractor.extract_audio(id)


if __name__ == '__main__':
    main()
