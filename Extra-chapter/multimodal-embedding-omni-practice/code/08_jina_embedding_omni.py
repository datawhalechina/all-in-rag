from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
TOPIC_DIR = SCRIPT_DIR.parent
REPO_ROOT = TOPIC_DIR.parent.parent
DATA_DIR = TOPIC_DIR / "data"

MODEL_PATH = REPO_ROOT / "models/jina-embeddings-v5-omni-nano"
MODEL_REPO = "jinaai/jina-embeddings-v5-omni-nano"
SOURCES = {
    "img1": "beach1.jpg",
    "img2": "beach2.jpg",
    "audio": "example-audio-clip.wav",
    "video": "example-video-clip.mp4",
    "pdf": "paper_2506.18902_excerpt_2pages.pdf",
}

model_source = str(MODEL_PATH) if MODEL_PATH.exists() else MODEL_REPO
print(f"model_source={model_source}")

paths = {k: DATA_DIR / v for k, v in SOURCES.items()}
for p in paths.values():
    if not p.exists():
        raise FileNotFoundError(f"Missing local asset: {p}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
raw_model = AutoModel.from_pretrained(
    model_source,
    trust_remote_code=True,
    default_task="retrieval",
    modality="vision",
).eval().to(device)
processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)
st_model = SentenceTransformer(model_source, trust_remote_code=True, model_kwargs={"default_task": "retrieval"})

with torch.no_grad():
    docs = raw_model.embed(
        **processor(
            text=[
                "Document: A beautiful sunset over the beach",
                "Document: Un beau coucher de soleil sur la plage",
                "Document: 海滩上美丽的日落",
                "Document: 浜辺に沈む美しい夕日",
            ],
            padding=True,
            return_tensors="pt",
        ).to(device)
    ).float().cpu().numpy()
    img1 = raw_model.embed(**processor(images=[str(paths["img1"])], text="<image>", return_tensors="pt").to(device)).float().cpu().numpy()[0]
    img2 = raw_model.embed(**processor(images=[str(paths["img2"])], text="<image>", return_tensors="pt").to(device)).float().cpu().numpy()[0]

audio = st_model.encode(str(paths["audio"]), convert_to_numpy=True)
video = st_model.encode(str(paths["video"]), convert_to_numpy=True)
pdf = st_model.encode(str(paths["pdf"]), convert_to_numpy=True)

corpus = np.vstack([docs[0], docs[1], docs[2], docs[3], img1, img2, audio, video, pdf])
queries = [
    ("R1", "sunset on the beach"),
    ("R2", "waves and sunset on coast"),
    ("R3", "beach scene with warm orange sky"),
]
all_rounds: dict[str, np.ndarray] = {}

for name, q in queries:
    with torch.no_grad():
        qv = raw_model.embed(**processor(text=f"Query: {q}", return_tensors="pt").to(device)).float().cpu().numpy()[0]
        fusion = raw_model.embed(**processor(images=[str(paths["img1"])], text=q, return_tensors="pt").to(device)).float().cpu().numpy()[0]
    vectors = np.vstack([qv, corpus, fusion])
    all_rounds[name] = vectors
    n = np.linalg.norm(vectors, axis=1, keepdims=True)
    sim = (vectors / n) @ (vectors / n).T
    print(f"\n{name} | shape={vectors.shape} | q_first8={np.array2string(vectors[0,:8], precision=4)}")
    print(np.array2string(sim, precision=4, suppress_small=True))

base = all_rounds["R1"]
for name in ["R2", "R3"]:
    other = all_rounds[name]
    aligned = np.array(
        [
            float((base[i] @ other[i]) / (np.linalg.norm(base[i]) * np.linalg.norm(other[i])))
            for i in range(base.shape[0])
        ]
    )
    scores = np.array(
        [
            float((base[0] @ other[i]) / (np.linalg.norm(base[0]) * np.linalg.norm(other[i])))
            for i in range(1, 1 + corpus.shape[0])
        ]
    )
    print(f"\nR1 -> {name}")
    print(f"aligned={np.array2string(aligned, precision=4)}")
    print(f"best_corpus_idx={1 + int(np.argmax(scores))}, score={scores.max():.4f}")
