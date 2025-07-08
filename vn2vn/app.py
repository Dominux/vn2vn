from pathlib import Path
from uuid import UUID

from vn2vn.vn_extractor import VNExtractor
from vn2vn.speech2text import gen_speech2text


base_path = Path("/tmp/vn2vn")
id = UUID("6d0d5ffe-cccc-4268-8858-6518c0fa85ca")


def main():
    vn_extractor = VNExtractor(base_path)
    vn_extractor.extract_audio(id)

    gen_speech2text()


if __name__ == '__main__':
    main()
