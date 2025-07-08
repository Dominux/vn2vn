from pathlib import Path
import shutil
import os
from uuid import UUID

from filetypes import FileType


class VNExtractor:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

        # self._remove_base_path()
        self._create_base_path()

    def extract_audio(self, id: UUID):
        input_full_path = self._build_full_path(id, FileType.InputVN)
        output_full_path = self._build_full_path(id, FileType.InputAudio)

        cmd = f'ffmpeg -y -i {input_full_path} -map 0:a -acodec libmp3lame {output_full_path}'
        os.system(cmd)

    def _build_full_path(self, id: UUID, filetype: FileType):
        return self._base_path / str(id) / filetype.value

    def _create_base_path(self):
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _remove_base_path(self):
        shutil.rmtree(self._base_path)
