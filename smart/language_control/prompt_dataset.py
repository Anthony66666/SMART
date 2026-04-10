import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from torch.utils.data import Dataset


@dataclass
class PromptScenarioExample:
    prompt: str
    scenario_id: str
    scene_path: Optional[str]
    raw: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "scenario_id": self.scenario_id,
            "scene_path": self.scene_path,
            "raw": self.raw,
        }


class PromptScenarioDataset(Dataset):
    """Loader for ProSim-Instruct style prompt-scenario annotations.

    The public ProSim annotation schema may evolve, so this loader accepts common
    prompt field names and keeps the raw record for downstream task-specific
    parsing. If load_scene=True, the corresponding WOMD/SMART pickle is attached
    under the ``scene`` key.
    """

    PROMPT_FIELDS = (
        "prompt",
        "instruction",
        "text",
        "text_instruction",
        "caption",
    )
    PROMPT_LIST_FIELDS = (
        "prompts",
        "instructions",
        "text_prompts",
        "text_instructions",
        "agent_prompts",
    )
    SCENARIO_FIELDS = ("sid", "scenario_id", "scene_id", "id")
    SCENE_PATH_FIELDS = ("scene_path", "womd_path", "smart_path", "data_path")

    def __init__(
        self,
        annotation_paths: Sequence[str],
        womd_root: Optional[str] = None,
        load_scene: bool = False,
    ) -> None:
        super().__init__()
        self.annotation_paths = [Path(path) for path in annotation_paths]
        self.womd_root = Path(womd_root) if womd_root is not None else None
        self.load_scene = load_scene
        self.examples = self._load_examples(self.annotation_paths)
        if not self.examples:
            raise ValueError("No prompt-scenario examples were found.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        item = example.to_dict()
        if self.load_scene:
            if example.scene_path is None:
                raise FileNotFoundError(f"No scene path is available for scenario {example.scenario_id}.")
            with open(example.scene_path, "rb") as handle:
                item["scene"] = pickle.load(handle)
        return item

    def _load_examples(self, paths: Iterable[Path]) -> List[PromptScenarioExample]:
        examples: List[PromptScenarioExample] = []
        for path in paths:
            if path.is_dir():
                files = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl"))
            else:
                files = [path]
            for file_path in files:
                for record in self._read_records(file_path):
                    scenario_id = self._extract_first(record, self.SCENARIO_FIELDS) or file_path.stem
                    scene_path = self._resolve_scene_path(record, scenario_id)
                    for prompt in self._extract_prompts(record):
                        examples.append(
                            PromptScenarioExample(
                                prompt=prompt,
                                scenario_id=str(scenario_id),
                                scene_path=scene_path,
                                raw=record,
                            )
                        )
        return examples

    def _read_records(self, path: Path) -> Iterable[Dict[str, Any]]:
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            return

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for record in payload:
                yield record
        elif isinstance(payload, dict):
            records = payload.get("data") or payload.get("annotations") or payload.get("examples")
            if isinstance(records, list):
                for record in records:
                    yield record
            else:
                yield payload
        else:
            raise ValueError(f"Unsupported annotation payload in {path}.")

    def _extract_prompts(self, record: Dict[str, Any]) -> List[str]:
        prompts: List[str] = []
        for field in self.PROMPT_FIELDS:
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                prompts.append(value.strip())
        for field in self.PROMPT_LIST_FIELDS:
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                prompts.append(value.strip())
            elif isinstance(value, list):
                prompts.extend(self._flatten_prompt_list(value))
            elif isinstance(value, dict):
                prompts.extend(self._flatten_prompt_list(value.values()))
        return list(dict.fromkeys(prompts))

    def _flatten_prompt_list(self, values: Iterable[Any]) -> List[str]:
        prompts: List[str] = []
        for value in values:
            if isinstance(value, str) and value.strip():
                prompts.append(value.strip())
            elif isinstance(value, dict):
                prompts.extend(self._flatten_prompt_list(value.values()))
            elif isinstance(value, list):
                prompts.extend(self._flatten_prompt_list(value))
        return prompts

    def _extract_first(self, record: Dict[str, Any], fields: Sequence[str]) -> Optional[Any]:
        for field in fields:
            value = record.get(field)
            if value is not None:
                return value
        return None

    def _resolve_scene_path(self, record: Dict[str, Any], scenario_id: Any) -> Optional[str]:
        explicit = self._extract_first(record, self.SCENE_PATH_FIELDS)
        if explicit is not None:
            path = Path(str(explicit))
            if not path.is_absolute() and self.womd_root is not None:
                path = self.womd_root / path
            return str(path)
        if self.womd_root is None:
            return None
        for suffix in (".pkl", ".pickle"):
            candidate = self.womd_root / f"{scenario_id}{suffix}"
            if candidate.exists():
                return str(candidate)
        return str(self.womd_root / f"{scenario_id}.pkl")
