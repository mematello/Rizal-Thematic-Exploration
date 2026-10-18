# Known Code & Documentation Mismatches

This document inventories known cosmetic, label, or configuration mismatches in the active code files (`frontend/` and `backend/`). These items are intentionally **not modified** during documentation sync tasks to preserve the strict zero-diff requirement on application code (`git diff main..docs/sync -- backend/ frontend/` must remain strictly empty).

---

## 1. Frontend: Footer Tech Stack Badge (`Next.js 15`)

- **File**: `frontend/components/Footer.tsx`
- **Line**: 40
- **Current Code**:
  ```tsx
  <span className="px-3 py-1.5 rounded-md bg-[#1E293B] text-[#D4AF37] text-xs font-semibold tracking-wide">
    Next.js 15
  </span>
  ```
- **Actual State**: `frontend/package.json` specifies `"next": "^16.1.6"`. Production builds run on Next.js 16.1.6.
- **Proposed Fix**: Update label text from `Next.js 15` to `Next.js 16` in an upcoming frontend commit.

---

## 2. Frontend: Footer Backend Provider Label (`ZeroGPU`)

- **File**: `frontend/components/Footer.tsx`
- **Line**: 115
- **Current Code**:
  ```tsx
  <span className="font-semibold">Hugging Face API (ZeroGPU)</span>
  ```
- **Actual State**: The Hugging Face Space runs on the `zero-a10g` hardware tier, but the core FastAPI search engine (`engine.py`) executes on **CPU PyTorch** using precomputed static embedding caches. Only the dummy Gradio interface in `space.py` requests ZeroGPU.
- **Proposed Fix**: Update label to `Hugging Face Spaces (Gradio / zero-a10g)` or `Hugging Face Spaces API` to avoid implying that search inference requires active GPU slice allocation.

---

## 3. Backend: Development Server Launch Commands (`backend/README.md`)

- **File**: `backend/README.md`
- **Lines**: 31–32
- **Current Code**:
  ```bash
  export PYTHONHASHSEED=0   # PowerShell: $env:PYTHONHASHSEED="0"
  poetry run uvicorn app.main:app --reload
  ```
- **Actual State**:
  1. `PYTHONHASHSEED=0` is obsolete and no longer required after Step B determinism fixes (`ORDER BY Sentence.id` and in-memory `sorted()` tie-breakers).
  2. `--reload` introduces file-watching subprocess wrappers that cause timing jitter and non-deterministic overhead during performance profiling and benchmarks.
- **Proposed Fix**: Update to match root `README.md` and `HANDOFF.md`:
  ```bash
  poetry run uvicorn app.main:app --host 127.0.0.1 --port 8000
  ```
